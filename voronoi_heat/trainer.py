"""
Training and inference orchestrators for the Voronoi heat system.

This module holds the high-level loops (train, infer, baseline) and config.
All math, geometry, and IO details live in the dedicated submodules.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .utils import resolve_device, device_to_str, set_seed
from .schedule import TempSchedule, Logger, save_checkpoint, load_checkpoint
from .mesh_io import (
    load_mesh_any,
    preprocess_mesh,
    read_seeds,
    write_segments_obj,
    write_voronoi_vtp,
    write_viz_npz,
)
from .losses import (
    smoothness_loss_optimized,
    area_balance_loss_optimized,
    polyline_direction_loss,
    eq_margin_losses,
    gradient_only_edge_loss,
    gradient_only_edge_loss_x,
    seam_resolve_loss_on_cut,
    interface_penalty_grad_U,
)
from .seam_ops import (
    certify_segments_equal_distance,
    build_cut_mesh_from_segments,
    lift_sources_to_cut,
)
from .stage3_utils import (
    compute_stage3_face_pairs,
    prepare_stage3_polyline_cache,
)
from .voronoi_heat_torch import (
    VoronoiHeatModel,
    TrainableSources,
    heat_only_face_loss,
    heat_only_face_loss_vec,
    face_gradients,
    infer_labels_and_segments,
    heat_method_distances,
    extract_segments_in_face_hard,
    _stitch_polylines,
)
from .fast_stage2 import heat_only_face_loss_fast
from .snapshots import save_stage_snapshot, write_seam_transition_report


@dataclass
class TrainConfig:
    mesh: str
    seeds_file: str
    out_dir: str = "runs/voronoi_heat"
    device: str = "auto"
    weight_decay: float = 0.0
    cg_tol: float = 1.0e-6
    cg_iters: int = 200
    t: Optional[float] = None
    seed: int = 42
    grad_clip: float = 1.0
    log_every: int = 10
    ckpt_every: int = 100
    resume: Optional[str] = None
    warm_steps: int = 0
    warm_lr_start: float = 5e-3
    warm_lr_end: float = 5e-3
    warm_w_smooth: float = 0.0
    warm_w_area: float = 0.0
    warm_area_beta: float = 10.0
    stage2_steps: int = 0
    stage3_steps: int = 0
    stage2_lr_start: float = 5e-3
    stage2_lr_end: float = 5e-3
    stage3_lr_start: float = 1e-3
    stage3_lr_end: float = 1e-3
    stage2_w_jump: float = 1.0
    stage2_w_normal: float = 0.5
    stage2_w_tan: float = 0.25
    stage2_w_smooth: float = 0.0
    stage2_align_weight: float = 0.3
    stage3_w_jump: float = 0.0
    stage3_w_normal: float = 0.0
    stage3_w_tan: float = 0.1
    stage3_w_smooth: float = 1.0e-4
    stage3_align_weight: float = 0.3
    stage3_polyline_weight: float = 1.0
    stage3_polyline_npz: Optional[str] = None
    # Optional Stage-3 mode: skip CG solves and only match sources to Stage-2 snapshot
    stage3_no_solve: bool = False
    stage3_src_match_weight: float = 0.0
    topK_pairs: int = 3
    # On-seam equality and off-seam margin controls
    stage2_w_eq: float = 0.05
    stage3_w_eq: float = 0.02
    stage2_w_margin: float = 0.0
    stage3_w_margin: float = 0.10
    seam_eq_tau: float = 0.30
    seam_margin_delta: float = 0.05
    temp_mode: str = "linear"
    temp_steps: int = 1000
    start_beta: float = 6.0
    end_beta: float = 24.0
    start_kappa: float = 6.0
    end_kappa: float = 40.0
    start_srcT: float = 1.5
    end_srcT: float = 1.0
    auto_infer: bool = False
    infer_subdir: str = "final"
    auto_viz_npz: bool = True
    viz_npz_name: str = "viz_data.npz"
    grad_align_beta_edge: float = 12.0
    use_soft_flip: bool = True
    softflip_beta: float = 10.0
    softflip_margin: float = 0.1
    softflip_eta: float = 10.0
    softflip_alpha: float = 0.75
    softflip_kappa: Optional[float] = None
    warm_log_name: str = "warmup_log.csv"
    main_log_name: str = "train_log.csv"
    export_checkpoint_visuals: bool = False
    save_final_bundle: bool = False
    # Gradient-only controls
    gradonly_enable: bool = False
    gradonly_lam_tau: float = 1.0
    gradonly_lam_u: float = 0.0
    gradonly_cut_resolve_enable: bool = True
    gradonly_cut_every: int = 1
    gradonly_lam_eq: float = 0.0
    seam_cert_tau: float = 1.0e-3
    gradonly_interface_lambda: float = 0.0


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    resolved_device = resolve_device(cfg.device)
    cfg.device = device_to_str(resolved_device)
    if resolved_device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    with open(Path(cfg.out_dir) / "train_config.json", "w", encoding="utf8") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    V_raw, F_raw = load_mesh_any(cfg.mesh)
    V_np, F_np = preprocess_mesh(V_raw, F_raw, verbose=True)
    seeds = read_seeds(cfg.seeds_file)
    if seeds and max(seeds) >= V_np.shape[0]:
        raise ValueError(f"Seed index out of range for mesh (nV={V_np.shape[0]}).")

    model = VoronoiHeatModel(V_np, F_np, t=cfg.t, device=resolved_device)
    sources = TrainableSources(nV=model.V.shape[0], seeds=seeds, sharpness=10.0, device=resolved_device)
    sources.to(resolved_device)

    seed_face_mask = None
    if seeds:
        seed_tensor = torch.tensor(seeds, device=model.F.device, dtype=torch.long)
        if seed_tensor.numel() > 0:
            mask = torch.zeros(model.F.shape[0], dtype=torch.bool, device=model.F.device)
            for s in seed_tensor:
                mask |= (model.F == s).any(dim=1)
            seed_face_mask = mask

    opt = torch.optim.Adam(sources.parameters(), lr=1.0, weight_decay=cfg.weight_decay)

    stage3_poly_cache: Optional[Dict[str, Tensor]] = None
    stage3_source_target: Optional[Tensor] = None
    stage3_score_target: Optional[Tensor] = None
    stage3_segments_faces_np: Optional[np.ndarray] = None
    stage3_segments_bary_np: Optional[np.ndarray] = None
    if cfg.stage3_polyline_npz:
        seg_path = Path(cfg.stage3_polyline_npz)
        if not seg_path.is_absolute():
            seg_path = Path(cfg.out_dir) / seg_path
        if not seg_path.exists():
            raise FileNotFoundError(f"Stage-3 polyline file '{seg_path}' not found.")
        with np.load(seg_path, allow_pickle=False) as data:
            faces = data.get("hinge_segments_faces")
            bary = data.get("hinge_segments_bary")
        if faces is None or bary is None:
            raise KeyError(f"File '{seg_path}' does not contain hinge segment data.")
        stage3_segments_faces_np = faces.astype(np.int64, copy=False)
        stage3_segments_bary_np = bary.astype(np.float32, copy=False)

    mesh_area_total = model.A_f.sum()

    snapshot_dir = Path(cfg.out_dir) / "snapshots"
    warm_seam_faces_np: Optional[np.ndarray] = None
    stage2_seam_faces_np: Optional[np.ndarray] = None
    seam_report_written = False

    warm_steps = max(0, cfg.warm_steps)
    stage2_steps = max(0, cfg.stage2_steps)
    stage3_steps = max(0, cfg.stage3_steps)
    total_steps = warm_steps + stage2_steps + stage3_steps
    if total_steps <= 0:
        raise ValueError("At least one stage must have a positive number of steps.")

    stage2_start = warm_steps
    stage3_start = warm_steps + stage2_steps

    def _set_optimizer_lr(value: float) -> None:
        for group in opt.param_groups:
            group["lr"] = float(value)

    def _phase_for_step(step_idx: int) -> str:
        if step_idx < warm_steps:
            return "warm"
        if step_idx < stage3_start:
            return "stage2"
        return "stage3"

    def _phase_settings(phase: str) -> Dict[str, float]:
        if phase == "warm":
            return {
                "w_jump": 0.0,
                "w_normal": 0.0,
                "w_tan": 0.0,
                "align_weight": 0.0,
                "poly_weight": 0.0,
                "smooth_weight": cfg.warm_w_smooth,
                "area_weight": cfg.warm_w_area,
                "area_beta": cfg.warm_area_beta,
                "lr_start": cfg.warm_lr_start,
                "lr_end": cfg.warm_lr_end,
                "steps": warm_steps,
                "start": 0,
            }
        if phase == "stage2":
            return {
                "w_jump": cfg.stage2_w_jump,
                "w_normal": cfg.stage2_w_normal,
                "w_tan": cfg.stage2_w_tan,
                "align_weight": cfg.stage2_align_weight,
                "poly_weight": 0.0,
                "smooth_weight": cfg.stage2_w_smooth,
                "area_weight": 0.0,
                "area_beta": cfg.warm_area_beta,
                "eq_weight": cfg.stage2_w_eq,
                "margin_weight": cfg.stage2_w_margin,
                "lr_start": cfg.stage2_lr_start,
                "lr_end": cfg.stage2_lr_end,
                "steps": stage2_steps,
                "start": stage2_start,
            }
        return {
            "w_jump": cfg.stage3_w_jump,
            "w_normal": cfg.stage3_w_normal,
            "w_tan": cfg.stage3_w_tan,
            "align_weight": cfg.stage3_align_weight,
            "poly_weight": cfg.stage3_polyline_weight,
            "smooth_weight": cfg.stage3_w_smooth,
            "area_weight": 0.0,
            "area_beta": cfg.warm_area_beta,
            "eq_weight": cfg.stage3_w_eq,
            "margin_weight": cfg.stage3_w_margin,
            "lr_start": cfg.stage3_lr_start,
            "lr_end": cfg.stage3_lr_end,
            "steps": stage3_steps,
            "start": stage3_start,
        }

    first_phase = _phase_for_step(0)
    first_cfg = _phase_settings(first_phase)
    first_len = max(1, first_cfg["steps"])
    initial_lr = first_cfg["lr_end"] if first_len <= 1 else first_cfg["lr_start"]
    if initial_lr is not None:
        _set_optimizer_lr(initial_lr)

    temp_main_steps = cfg.temp_steps if cfg.temp_steps > 0 else stage2_steps + stage3_steps
    if temp_main_steps <= 0:
        temp_main_steps = max(1, stage2_steps + stage3_steps)
    temps_main = TempSchedule(
        start_beta=cfg.start_beta,
        end_beta=cfg.end_beta,
        start_kappa=cfg.start_kappa,
        end_kappa=cfg.end_kappa,
        start_srcT=cfg.start_srcT,
        end_srcT=cfg.end_srcT,
        total_steps=max(1, temp_main_steps),
        mode=cfg.temp_mode,
    )

    def _schedule_values(step_idx: int) -> Tuple[float, float, float]:
        if step_idx < warm_steps:
            return cfg.start_beta, cfg.start_kappa, cfg.start_srcT
        main_idx = step_idx - warm_steps
        return temps_main.at(main_idx)

    warm_logger = Logger(cfg.out_dir, filename=cfg.warm_log_name, enable_tb=False) if warm_steps > 0 else None
    main_logger = Logger(cfg.out_dir, filename=cfg.main_log_name, enable_tb=True)
    best_loss = float("inf")
    start_step = 0

    if cfg.resume:
        ckpt = load_checkpoint(cfg.resume)
        sources.load_state_dict(ckpt["sources"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", best_loss)
        print(f"Resumed from {cfg.resume} @ step {start_step} (best_loss={best_loss:.6f})")
        if start_step >= total_steps:
            start_step = total_steps - 1

    prev_phase: Optional[str] = None
    for step in range(max(0, start_step), total_steps):
        beta_pairs, kappa, src_temp = _schedule_values(step)
        phase = _phase_for_step(step)
        phase_cfg = _phase_settings(phase)

        if warm_steps > 0 and prev_phase == "warm" and phase == "stage2" and warm_seam_faces_np is None:
            warm_seam_faces_np = save_stage_snapshot(snapshot_dir, "warm", model, sources.logits)
            print(f"Warm snapshot saved with {len(warm_seam_faces_np)} seam faces.", flush=True)

        if stage2_steps > 0 and prev_phase == "stage2" and phase != "stage2" and stage2_seam_faces_np is None:
            stage2_seam_faces_np = save_stage_snapshot(snapshot_dir, "stage2", model, sources.logits)
            print(f"Stage2 snapshot saved with {len(stage2_seam_faces_np)} seam faces.", flush=True)
            if warm_seam_faces_np is not None and not seam_report_written:
                write_seam_transition_report(snapshot_dir, warm_seam_faces_np, stage2_seam_faces_np)
                seam_report_written = True

        phase_len = max(1, phase_cfg["steps"])
        phase_start_idx = phase_cfg["start"]
        phase_pos = max(0, step - phase_start_idx)
        if phase_len <= 1:
            target_lr = phase_cfg["lr_end"]
        else:
            ratio = min(max(phase_pos / float(phase_len - 1), 0.0), 1.0)
            target_lr = phase_cfg["lr_start"] + ratio * (phase_cfg["lr_end"] - phase_cfg["lr_start"])
        if target_lr is not None and not math.isclose(opt.param_groups[0]["lr"], target_lr, rel_tol=1e-9, abs_tol=1e-12):
            _set_optimizer_lr(target_lr)

        opt.zero_grad(set_to_none=True)

        zero = torch.zeros((), device=model.V.device, dtype=model.V.dtype)
        seam_loss = zero
        align_loss = zero
        smooth_loss = zero
        area_loss = zero
        poly_loss = zero
        eq_loss = zero
        margin_loss = zero
        terms = {"jump": zero, "normal": zero, "tan": zero}
        area_weight = float(phase_cfg.get("area_weight", 0.0))
        smooth_weight = float(phase_cfg.get("smooth_weight", 0.0))
        logits_t: Optional[Tensor] = None
        if area_weight > 0.0 or smooth_weight > 0.0:
            logits_t = sources.logits.t()

        # Capture Stage-3 source snapshot the first time we enter Stage-3
        if prev_phase != phase and phase == "stage3" and cfg.stage3_no_solve and stage3_source_target is None:
            stage3_source_target = sources(model.M_diag, temp=1.0).detach()
            stage3_score_target = -torch.log(stage3_source_target.clamp_min(1.0e-9))

        if phase != "warm" and not (phase == "stage3" and cfg.stage3_no_solve):
            B_heat = sources(model.M_diag, temp=src_temp)
            # Warm-start CG from previous U when available
            U, X, S = model(B_heat, cg_tol=cfg.cg_tol, cg_iters=cfg.cg_iters, x0=locals().get("_U_prev", None))
            _U_prev = U.detach()
            S_face = torch.stack([S[model.F[:, 0]], S[model.F[:, 1]], S[model.F[:, 2]]], dim=1).mean(dim=1)
            grad_face = face_gradients(S, model.F, model.gI, model.gJ, model.gK)

            if cfg.gradonly_enable:
                # Use unit-direction based gradient-only residual (no labels, no S-gating)
                seam_loss = gradient_only_edge_loss_x(
                    model,
                    X,
                    lam_tau=cfg.gradonly_lam_tau,
                    lam_u=cfg.gradonly_lam_u,
                    beta_edge=cfg.grad_align_beta_edge,
                )
                terms = {"jump": zero, "normal": zero, "tan": zero}
                # Explicit cut-and-resolve loop on certified segments (gradient-only)
                seam_cut_loss = zero
                cert_count = 0
                # Only run cut-and-resolve during Stage-3; keep Stage-2 pure
                if (
                    phase == "stage3"
                    and cfg.gradonly_cut_resolve_enable
                    and (cfg.gradonly_cut_every > 0)
                    and ((step % cfg.gradonly_cut_every) == 0)
                    and (seeds is not None)
                    and (len(seeds) >= 2)
                ):
                    with torch.no_grad():
                        # Prefer gradient-chosen pairs from current field; avoid label overrides
                        # Fall back to baseline certification if none are found
                        try:
                            (
                                _,
                                _,
                                _,
                                _,
                                seg_faces_np,
                                seg_bary_np,
                                seg_pairs_np,
                            ) = infer_labels_and_segments(
                                model,
                                seeds=None,
                                B_heat=B_heat.detach(),
                                cg_tol=cfg.cg_tol,
                                cg_iters=cfg.cg_iters,
                                no_label_override=True,
                                return_pairs=True,
                            )
                        except TypeError:
                            # Backward compatibility if return_pairs not supported
                            (
                                _,
                                _,
                                _,
                                _,
                                seg_faces_np,
                                seg_bary_np,
                            ) = infer_labels_and_segments(
                                model,
                                seeds=None,
                                B_heat=B_heat.detach(),
                                cg_tol=cfg.cg_tol,
                                cg_iters=cfg.cg_iters,
                            )
                            seg_pairs_np = None

                    if seg_pairs_np is not None and getattr(seg_pairs_np, "size", 0) > 0:
                        cert_faces_np, cert_bary_np = seg_faces_np, seg_bary_np
                        cert_pairs = [tuple(map(int, p)) for p in seg_pairs_np.tolist()]
                    else:
                        cert_faces_np, cert_bary_np, cert_pairs = certify_segments_equal_distance(
                            model,
                            seeds or [],
                            seg_faces_np,
                            seg_bary_np,
                            cg_tol=cfg.cg_tol,
                            cg_iters=cfg.cg_iters,
                            tau=cfg.seam_cert_tau,
                        )

                    if cert_faces_np.size > 0:
                        cert_count = int(cert_faces_np.size)
                        Vc_np, Fc_np, seam_infos = build_cut_mesh_from_segments(
                            V_np,
                            F_np,
                            cert_faces_np,
                            cert_bary_np,
                            cert_pairs,
                        )
                        model_cut = VoronoiHeatModel(Vc_np, Fc_np, t=cfg.t, device=resolved_device)
                        B_cut = lift_sources_to_cut(B_heat, V_np, F_np, Vc_np, seam_infos)
                        Uc, Xc, Sc = model_cut(B_cut, cg_tol=cfg.cg_tol, cg_iters=cfg.cg_iters)

                        if cfg.gradonly_interface_lambda > 0.0:
                            gradU = interface_penalty_grad_U(
                                model_cut,
                                Uc,
                                seam_infos,
                                lam_tau=cfg.gradonly_lam_tau,
                            )
                            Uc.backward(cfg.gradonly_interface_lambda * gradU, retain_graph=True)

                        seam_cut_loss = seam_resolve_loss_on_cut(
                            model_cut,
                            Sc,
                            seam_infos,
                            lam_tau=cfg.gradonly_lam_tau,
                            lam_eq=cfg.gradonly_lam_eq,
                            lam_u=cfg.gradonly_lam_u,
                        )
                        seam_loss = seam_loss + seam_cut_loss
            else:
                if any(abs(phase_cfg[key]) > 0.0 for key in ("w_normal", "w_tan")):
                    # Fully vectorized Stage-2 using unit-direction residuals and τ = n×∇Δ
                    seam_loss, terms = heat_only_face_loss_fast(
                        V=model.V,
                        F=model.F.long(),
                        S_vert=S,
                        n_f=model.n_hat,
                        gI=model.gI,
                        gJ=model.gJ,
                        gK=model.gK,
                        K=cfg.topK_pairs,
                        beta_pairs=beta_pairs,
                        w_normal=phase_cfg["w_normal"],
                        w_tan=phase_cfg["w_tan"],
                    )

            if phase_cfg["align_weight"] > 0.0:
                align_loss = contour_alignment_codex(
                    model.V,
                    model.F,
                    S,
                    beta_edge=cfg.grad_align_beta_edge,
                    use_soft_flip=cfg.use_soft_flip,
                    softflip_beta=cfg.softflip_beta,
                )

            if phase_cfg["poly_weight"] > 0.0 and phase == "stage3":
                if stage3_poly_cache is None:
                    with torch.no_grad():
                        if stage3_segments_faces_np is None or stage3_segments_bary_np is None:
                            (_, _, _, _, seg_faces_np, seg_bary_np) = infer_labels_and_segments(
                                model,
                                seeds=None,
                                B_heat=B_heat.detach(),
                                cg_tol=cfg.cg_tol,
                                cg_iters=cfg.cg_iters,
                            )
                            stage3_segments_faces_np = seg_faces_np
                            stage3_segments_bary_np = seg_bary_np
                        face_pair, face_conf = compute_stage3_face_pairs(
                            model,
                            X,
                            beta_edge=cfg.grad_align_beta_edge,
                        )
                        stage3_poly_cache = prepare_stage3_polyline_cache(
                            model,
                            seg_faces=stage3_segments_faces_np,
                            seg_bary=stage3_segments_bary_np,
                            face_pair=face_pair,
                            face_conf=face_conf,
                            seed_face_mask=seed_face_mask,
                        )
                        if stage3_poly_cache is None:
                            stage3_poly_cache = {
                                "faces": torch.zeros(0, dtype=torch.long, device=model.V.device),
                                "tangent": torch.zeros((0, 3), dtype=model.V.dtype, device=model.V.device),
                                "weight": torch.zeros(0, dtype=model.V.dtype, device=model.V.device),
                                "pairs": torch.zeros((0, 2), dtype=torch.long, device=model.V.device),
                            }
                        print(
                            f"Stage-3 polyline cache prepared with {stage3_poly_cache['faces'].numel()} faces.",
                            flush=True,
                        )
                if stage3_poly_cache is not None:
                    poly_loss = polyline_direction_loss(grad_face, stage3_poly_cache, model.n_hat)

            if (not cfg.gradonly_enable) and (phase_cfg.get("eq_weight", 0.0) > 0.0 or phase_cfg.get("margin_weight", 0.0) > 0.0):
                eq_loss, margin_loss = eq_margin_losses(
                    S,
                    S_face,
                    model.F,
                    beta_pairs=beta_pairs,
                    kappa=kappa,
                    K=cfg.topK_pairs,
                    tau_eq=cfg.seam_eq_tau,
                    delta=cfg.seam_margin_delta,
                )

        # Stage-3 no-solve: score-proxy matching + alignment on proxy (no CG)
        src_match_loss = zero
        if phase == "stage3" and cfg.stage3_no_solve:
            B_now = sources(model.M_diag, temp=src_temp)
            S_proxy_now = -torch.log(B_now.clamp_min(1.0e-9))
            if cfg.stage3_src_match_weight > 0.0:
                if stage3_score_target is None:
                    stage3_score_target = S_proxy_now.detach()
                delta = S_proxy_now - stage3_score_target
                src_match_loss = delta.pow(2).sum(dim=1).mean()
            if phase_cfg["align_weight"] > 0.0:
                align_loss = contour_alignment_codex(
                    model.V,
                    model.F,
                    S_proxy_now,
                    beta_edge=cfg.grad_align_beta_edge,
                    use_soft_flip=cfg.use_soft_flip,
                    softflip_beta=cfg.softflip_beta,
                )

        if smooth_weight > 0.0 and logits_t is not None:
            smooth_loss = smoothness_loss_optimized(logits_t, getattr(model, "edge_index", None))
        if area_weight > 0.0 and logits_t is not None:
            area_loss, _ = area_balance_loss_optimized(
                model.V,
                model.F,
                logits_t,
                phase_cfg.get("area_beta", cfg.warm_area_beta),
                mesh_area_total,
            )

        # Guard loss components against NaN/Inf to avoid corrupting parameters
        def _finite_or_zero(x: Tensor) -> Tensor:
            return torch.nan_to_num(x, nan=0.0, posinf=1.0e6, neginf=1.0e6)

        seam_loss = _finite_or_zero(seam_loss)
        align_loss = _finite_or_zero(align_loss)
        eq_loss = _finite_or_zero(eq_loss)
        margin_loss = _finite_or_zero(margin_loss)
        poly_loss = _finite_or_zero(poly_loss)
        smooth_loss = _finite_or_zero(smooth_loss)
        area_loss = _finite_or_zero(area_loss)

        if phase == "warm":
            total_loss = zero
            if area_weight > 0.0:
                total_loss = total_loss + area_weight * area_loss
            if smooth_weight > 0.0:
                total_loss = total_loss + smooth_weight * smooth_loss
        elif phase == "stage3" and cfg.stage3_no_solve:
            total_loss = cfg.stage3_src_match_weight * src_match_loss
            if phase_cfg["align_weight"] > 0.0:
                total_loss = total_loss + phase_cfg["align_weight"] * align_loss
            if smooth_weight > 0.0:
                total_loss = total_loss + smooth_weight * smooth_loss
            if area_weight > 0.0:
                total_loss = total_loss + area_weight * area_loss
        else:
            total_loss = seam_loss
            if phase_cfg["poly_weight"] > 0.0:
                total_loss = total_loss + phase_cfg["poly_weight"] * poly_loss
            if phase_cfg["align_weight"] > 0.0:
                total_loss = total_loss + phase_cfg["align_weight"] * align_loss
            if phase_cfg.get("eq_weight", 0.0) > 0.0:
                total_loss = total_loss + phase_cfg["eq_weight"] * eq_loss
            if phase_cfg.get("margin_weight", 0.0) > 0.0:
                total_loss = total_loss + phase_cfg["margin_weight"] * margin_loss
            if smooth_weight > 0.0:
                total_loss = total_loss + smooth_weight * smooth_loss
            if area_weight > 0.0:
                total_loss = total_loss + area_weight * area_loss

        total_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(sources.parameters(), cfg.grad_clip).item())
        opt.step()

        if (step % cfg.log_every) == 0 or step == total_steps - 1:
            scalars = {
                "loss": float(total_loss.item()),
                "seam": float(seam_loss.detach().cpu().item()),
                "seam_cut": float(seam_cut_loss.detach().cpu().item()) if 'seam_cut_loss' in locals() else 0.0,
                "cert_count": float(cert_count) if 'cert_count' in locals() else 0.0,
                "jump": float(terms["jump"]),
                "normal": float(terms["normal"]),
                "tan": float(terms["tan"]),
                "eq": float(eq_loss.detach().cpu().item()),
                "margin": float(margin_loss.detach().cpu().item()),
                "poly": float(poly_loss.detach().cpu().item()),
                "smooth": float(smooth_loss.detach().cpu().item()),
                "area": float(area_loss.detach().cpu().item()),
                "srcdiff": float(src_match_loss.detach().cpu().item()) if (phase == "stage3" and cfg.stage3_no_solve) else 0.0,
                "lr": float(opt.param_groups[0]["lr"]),
                "beta": beta_pairs,
                "kappa": kappa,
                "srcT": src_temp,
                "grad_norm": grad_norm,
                "align": float(align_loss.detach().cpu().item()),
            }
            target_logger = warm_logger if (phase == "warm" and warm_logger is not None) else main_logger
            target_logger.log(step, scalars)
            stage_tag = "W" if phase == "warm" else ("S2" if phase == "stage2" else "S3")
            print(
                f"[{stage_tag}{step:05d}] loss={scalars['loss']:.6f} seam={scalars['seam']:.6f} seam_cut={scalars['seam_cut']:.6f} "
                f"| jump={scalars['jump']:.4f} norm={scalars['normal']:.4f} tan={scalars['tan']:.4f} eq={scalars['eq']:.4f} margin={scalars['margin']:.4f} poly={scalars['poly']:.4f} srcdiff={scalars['srcdiff']:.4f} smooth={scalars['smooth']:.4f} area={scalars['area']:.4f} align={scalars['align']:.4f} "
                f"| lr={scalars['lr']:.3e} beta={beta_pairs:.2f} kappa={kappa:.2f} srcT={src_temp:.2f}"
            )

        prev_phase = phase

        if (step % cfg.ckpt_every) == 0 or step == total_steps - 1:
            state = {
                "step": step,
                "best_loss": best_loss,
                "sources": sources.state_dict(),
                "optimizer": opt.state_dict(),
                "config": asdict(cfg),
            }
            save_checkpoint(Path(cfg.out_dir) / f"ckpt_step_{step:06d}.pt", state)
            current_loss = float(total_loss.item())
            best_loss = min(best_loss, current_loss)

    # Save a training bundle and a viewer-friendly NPZ by default
    final_sources = sources(model.M_diag, temp=1.0).detach().cpu().numpy()
    if cfg.save_final_bundle:
        bundle = {
            "V": model.V.detach().cpu().numpy(),
            "F": model.F.detach().cpu().numpy(),
            "final_sources": final_sources,
        }
        bundle_path = Path(cfg.out_dir) / "final_bundle.npz"
        np.savez(bundle_path, **bundle)
        print(f"Saved geometry + final sources bundle to '{bundle_path}'.")

    # Also emit a viz_tool-compatible NPZ with 'vertices' and 'faces'
    # Uses argmax over final_sources as labels
    if getattr(cfg, "auto_viz_npz", True):
        labels_np = np.argmax(final_sources, axis=1).astype(np.int32)
        viz_npz_path = Path(cfg.out_dir) / getattr(cfg, "viz_npz_name", "viz_data.npz")
        write_viz_npz(
            viz_npz_path,
            model.V.detach().cpu().numpy(),
            model.F.detach().cpu().numpy(),
            labels_np,
            segments=[],
            scores=None,
            heat=None,
            segment_faces=None,
            segment_bary=None,
            sources=final_sources,
        )
        print(f"Saved viewer NPZ to '{viz_npz_path}'. Open with: python3 -m viz_tool {viz_npz_path}")


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def save_inference_outputs(
    model: VoronoiHeatModel,
    out_dir: str,
    labels: np.ndarray,
    segments: List[np.ndarray],
    scores: np.ndarray,
    heat: np.ndarray,
    segment_faces: np.ndarray,
    segment_bary: np.ndarray,
    viz_npz: Optional[str] = None,
    sources: Optional[np.ndarray] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(Path(out_dir) / "labels.npy", labels.astype(np.int32))
    with open(Path(out_dir) / "labels.txt", "w", encoding="utf8") as fh:
        for lbl in labels.tolist():
            fh.write(f"{int(lbl)}\n")
    write_segments_obj(Path(out_dir) / "segments.obj", segments)
    if viz_npz is not None:
        write_viz_npz(
            viz_npz,
            model.V.detach().cpu().numpy(),
            model.F.detach().cpu().numpy(),
            labels,
            segments,
            scores,
            heat,
            segment_faces,
            segment_bary,
            sources,
        )
        print(f"Saved viz NPZ to {viz_npz}")
    write_voronoi_vtp(
        Path(out_dir) / "voronoi.vtp",
        model.V.detach().cpu().numpy(),
        model.F.detach().cpu().numpy(),
        labels,
        segments,
        scores,
        heat,
        segment_faces,
        segment_bary,
        sources,
    )
    print(f"Inference complete. Saved labels and segments to '{out_dir}'.")


def infer_and_export(
    mesh_path: str,
    seeds_file: str,
    out_dir: str,
    *,
    t: Optional[float],
    device: str,
    cg_tol: float,
    cg_iters: int,
    viz_npz: Optional[str] = None,
) -> None:
    resolved_device = resolve_device(device)
    if resolved_device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    V_raw, F_raw = load_mesh_any(mesh_path)
    V_np, F_np = preprocess_mesh(V_raw, F_raw, verbose=False)
    seeds = read_seeds(seeds_file)
    if not seeds:
        raise ValueError("infer requires at least one seed")

    model = VoronoiHeatModel(V_np, F_np, t=t, device=resolved_device)
    nV = model.V.shape[0]
    C = len(seeds)
    B_sources = torch.zeros(nV, C, dtype=model.V.dtype, device=model.V.device)
    for c, s in enumerate(seeds):
        B_sources[s, c] = model.M_diag[s]

    labels, segments, scores, heat, segment_faces, segment_bary = infer_labels_and_segments(
        model,
        seeds,
        B_heat=B_sources,
        cg_tol=cg_tol,
        cg_iters=cg_iters,
    )

    save_inference_outputs(
        model,
        out_dir,
        labels,
        segments,
        scores,
        heat,
        segment_faces,
        segment_bary,
        viz_npz=viz_npz,
        sources=B_sources.detach().cpu().numpy(),
    )


def heat_distance_baseline(
    mesh_path: str,
    seeds_file: str,
    out_dir: str,
    *,
    t: Optional[float],
    device: str,
    cg_tol: float,
    cg_iters: int,
    viz_npz: Optional[str] = None,
) -> None:
    resolved_device = resolve_device(device)
    if resolved_device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    V_raw, F_raw = load_mesh_any(mesh_path)
    V_np, F_np = preprocess_mesh(V_raw, F_raw, verbose=False)
    seeds = read_seeds(seeds_file)
    if seeds and max(seeds) >= V_np.shape[0]:
        raise ValueError("Seed index out of range for mesh.")

    model = VoronoiHeatModel(V_np, F_np, t=t, device=resolved_device)
    labels_np, distances_np = heat_method_distances(model, seeds, cg_tol=cg_tol, cg_iters=cg_iters)

    distances_np = distances_np.astype(np.float32)
    labels_np = labels_np.astype(np.int32)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    np.save(out_dir_path / "heat_distances.npy", distances_np)

    S_tensor = torch.from_numpy(distances_np).to(model.V.device, dtype=model.V.dtype)
    labels_tensor = torch.from_numpy(labels_np).to(model.V.device, dtype=torch.long)
    s_per_seed = S_tensor.t().contiguous()

    raw_segments: List[Tuple[np.ndarray, np.ndarray]] = []
    faces_idx: List[int] = []
    bary_data: List[np.ndarray] = []
    for f in range(model.F.shape[0]):
        out = extract_segments_in_face_hard(
            model.V,
            model.F[f],
            S_tensor,
            labels_tensor,
            face_index=f,
            faces_all=model.F,
            s_per_seed=s_per_seed,
        )
        if out is not None:
            p0, p1, b0, b1 = out
            raw_segments.append((p0.detach().cpu().numpy(), p1.detach().cpu().numpy()))
            faces_idx.append(int(f))
            bary_data.append(torch.stack([b0, b1], dim=0).cpu().numpy())

    polylines = _stitch_polylines(raw_segments)
    if not polylines and raw_segments:
        polylines = [np.stack([seg[0], seg[1]], axis=0) for seg in raw_segments]

    if faces_idx:
        seg_faces_np = np.asarray(faces_idx, dtype=np.int64)
        seg_bary_np = np.asarray(bary_data, dtype=np.float32)
    else:
        seg_faces_np = np.zeros((0,), dtype=np.int64)
        seg_bary_np = np.zeros((0, 2, 3), dtype=np.float32)

    save_inference_outputs(
        model,
        out_dir,
        labels_np,
        polylines,
        distances_np,
        distances_np,
        seg_faces_np,
        seg_bary_np,
        viz_npz=viz_npz,
    )


__all__ = [
    "TrainConfig",
    "train",
    "infer_and_export",
    "heat_distance_baseline",
]
