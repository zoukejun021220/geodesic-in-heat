"""CLI entry point for the heat-based multi-seed Voronoi pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .pipeline import (
    TrainConfig,
    heat_distance_baseline,
    infer_and_export,
    train,
)


DEFAULT_TRAIN_CONFIG = "train_config.json"
DEFAULT_INFER_CONFIG = "infer_config.json"


def _load_config_mapping(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file '{path}' not found.")
    suffix = cfg_path.suffix.lower()
    text = cfg_path.read_text(encoding="utf8")
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to read YAML config files.") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{path}' must contain a JSON/YAML object.")
    return data  # type: ignore[return-value]


def _snapshot_defaults(parser: argparse.ArgumentParser, attr_name: str) -> None:
    defaults: Dict[str, Any] = {}
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == "help" or dest is argparse.SUPPRESS:
            continue
        defaults[dest] = action.default
    parser.set_defaults(**{attr_name: defaults})


def _merge_config(namespace: argparse.Namespace, defaults: Dict[str, Any], config: Dict[str, Any], valid_keys: Sequence[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if config:
        for key in valid_keys:
            if key in config:
                merged[key] = config[key]
    for key in valid_keys:
        if not hasattr(namespace, key):
            continue
        value = getattr(namespace, key)
        default_value = defaults.get(key)
        if key in merged:
            if value != default_value:
                merged[key] = value
        else:
            if value != default_value and value is not None:
                merged[key] = value
    return merged


def _add_train_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = sub.add_parser("train", help="Run optimisation for the Voronoi heat model.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config overriding defaults.")
    parser.add_argument("--mesh", type=str, default=None, help="Input mesh (.obj/.vtp/.vtk).")
    parser.add_argument("--seeds", type=str, default=None, dest="seeds_file", help="Text file with one seed index per line.")
    parser.add_argument("--out", type=str, default="runs/voronoi_heat", help="Output directory for checkpoints and logs.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for solves (auto, cpu, cuda, cuda:<idx>, or mps).",
    )
    parser.add_argument("--wd", type=float, default=0.0, dest="weight_decay", help="Weight decay for Adam.")
    parser.add_argument("--cg-tol", type=float, default=1e-6, dest="cg_tol", help="Tolerance for CG solves.")
    parser.add_argument("--cg-iters", type=int, default=200, dest="cg_iters", help="Maximum CG iterations.")
    parser.add_argument("--t", type=float, default=None, help="Optional diffusion time override.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--grad-clip", type=float, default=1.0, dest="grad_clip", help="Gradient clipping threshold.")
    parser.add_argument("--log-every", type=int, default=10, dest="log_every", help="Logging interval.")
    parser.add_argument("--ckpt-every", type=int, default=100, dest="ckpt_every", help="Checkpoint interval (steps).")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--warm-steps", type=int, default=0, dest="warm_steps", help="Number of warm-up steps.")
    parser.add_argument("--warm-lr-start", type=float, default=5e-3, dest="warm_lr_start", help="Starting learning rate during warm-up stage.")
    parser.add_argument("--warm-lr-end", type=float, default=5e-3, dest="warm_lr_end", help="Final learning rate during warm-up stage.")
    parser.add_argument("--warm-w-smooth", type=float, default=0.0, dest="warm_w_smooth", help="Smoothness weight during warm-up stage.")
    parser.add_argument("--warm-w-area", type=float, default=0.0, dest="warm_w_area", help="Area balance weight during warm-up stage.")
    parser.add_argument("--warm-area-beta", type=float, default=10.0, dest="warm_area_beta", help="Softmax sharpness for warm-up area balance loss.")
    parser.add_argument("--stage2-steps", type=int, default=0, dest="stage2_steps", help="Number of optimisation steps for stage 2 (seam discovery).")
    parser.add_argument("--stage2-lr-start", type=float, default=5e-3, dest="stage2_lr_start", help="Starting learning rate during stage 2.")
    parser.add_argument("--stage2-lr-end", type=float, default=5e-3, dest="stage2_lr_end", help="Final learning rate during stage 2.")
    parser.add_argument("--stage2-w-jump", type=float, default=1.0, dest="stage2_w_jump", help="Jump loss weight during stage 2.")
    parser.add_argument("--stage2-w-mirror", type=float, dest="stage2_w_jump", help=argparse.SUPPRESS)
    parser.add_argument("--stage2-w-normal", type=float, default=0.5, dest="stage2_w_normal", help="Normal loss weight during stage 2.")
    parser.add_argument("--stage2-w-tan", type=float, default=0.0, dest="stage2_w_tan", help="Tangent loss weight during stage 2.")
    parser.add_argument("--stage2-w-smooth", type=float, default=0.0, dest="stage2_w_smooth", help="Smoothness weight during stage 2.")
    parser.add_argument("--stage3-steps", type=int, default=0, dest="stage3_steps", help="Number of optimisation steps for stage 3 (straightening).")
    parser.add_argument("--stage3-lr-start", type=float, default=1e-3, dest="stage3_lr_start", help="Starting learning rate during stage 3.")
    parser.add_argument("--stage3-lr-end", type=float, default=1e-3, dest="stage3_lr_end", help="Final learning rate during stage 3.")
    parser.add_argument("--stage3-w-jump", type=float, default=0.0, dest="stage3_w_jump", help="Jump loss weight during stage 3.")
    parser.add_argument("--stage3-w-mirror", type=float, dest="stage3_w_jump", help=argparse.SUPPRESS)
    parser.add_argument("--stage3-w-normal", type=float, default=0.0, dest="stage3_w_normal", help="Normal loss weight during stage 3.")
    parser.add_argument("--stage3-w-tan", type=float, default=0.0, dest="stage3_w_tan", help="Tangent loss weight during stage 3.")
    parser.add_argument("--stage3-w-smooth", type=float, default=0.0, dest="stage3_w_smooth", help="Smoothness weight during stage 3.")
    parser.add_argument("--stage3-align-weight", type=float, default=0.0, dest="stage3_align_weight", help="Alignment penalty weight during stage 3.")
    parser.add_argument("--use-soft-flip", action="store_true", dest="use_soft_flip", default=True, help="Enable dominant-channel soft flip seam weighting (on by default).")
    parser.add_argument("--no-soft-flip", action="store_false", dest="use_soft_flip", help="Disable dominant-channel soft flip seam weighting.")
    parser.add_argument("--softflip-beta", type=float, default=10.0, help="Temperature for soft flip probabilities.")
    parser.add_argument("--softflip-margin", type=float, default=0.1, help="Reliability margin threshold for soft flip gating.")
    parser.add_argument("--softflip-eta", type=float, default=10.0, help="Logistic sharpness for the soft flip reliability gate.")
    parser.add_argument("--softflip-alpha", type=float, default=0.75, help="Blend factor between soft flip and legacy pair priors (1=new,0=legacy).")
    parser.add_argument("--softflip-kappa", type=float, default=None, help="Optional override for the hinge sharpness when soft flip is enabled.")
    parser.add_argument("--topK-pairs", type=int, default=3, dest="topK_pairs", help="Number of unordered pairs per face.")
    parser.add_argument("--temp-mode", type=str, choices=["linear", "cosine"], default="linear", help="Temperature schedule mode.")
    parser.add_argument("--temp-steps", type=int, default=1000, dest="temp_steps", help="Steps over which to run the temperature schedule.")
    parser.add_argument("--start-beta", type=float, default=6.0, dest="start_beta", help="Initial beta for pair softmax.")
    parser.add_argument("--end-beta", type=float, default=24.0, dest="end_beta", help="Final beta for pair softmax.")
    parser.add_argument("--start-kappa", type=float, default=6.0, dest="start_kappa", help="Initial hinge sharpness.")
    parser.add_argument("--end-kappa", type=float, default=40.0, dest="end_kappa", help="Final hinge sharpness.")
    parser.add_argument("--start-srcT", type=float, default=1.5, dest="start_srcT", help="Initial source temperature.")
    parser.add_argument("--end-srcT", type=float, default=1.0, dest="end_srcT", help="Final source temperature.")
    parser.add_argument("--no-auto-infer", action="store_false", dest="auto_infer", help="Disable automatic inference after training.")
    parser.add_argument("--infer-subdir", type=str, default="final", dest="infer_subdir", help="Subdirectory for automatic inference outputs.")
    parser.add_argument("--no-auto-viz-npz", action="store_false", dest="auto_viz_npz", help="Disable NPZ export during auto inference.")
    parser.add_argument("--viz-npz-name", type=str, default="viz_data.npz", dest="viz_npz_name", help="Filename for viz NPZ outputs.")
    parser.add_argument("--grad-align-beta", type=float, default=12.0, dest="grad_align_beta_edge", help="Sharpness parameter for gradient alignment.")
    parser.set_defaults(auto_infer=False, auto_viz_npz=True)
    _snapshot_defaults(parser, "_train_defaults")
    return parser


def _add_infer_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = sub.add_parser("infer", help="Run inference/export using a trained configuration.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config overriding defaults.")
    parser.add_argument("--mesh", type=str, default=None, help="Input mesh (.obj/.vtp/.vtk).")
    parser.add_argument("--seeds", type=str, default=None, dest="seeds_file", help="Seeds file with vertex indices.")
    parser.add_argument("--out", type=str, default=None, help="Output directory for inference results.")
    parser.add_argument("--t", type=float, default=None, help="Optional diffusion time override.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for solves (auto, cpu, cuda, cuda:<idx>, or mps).",
    )
    parser.add_argument("--cg-tol", type=float, default=1e-6, dest="cg_tol", help="Tolerance for CG solves.")
    parser.add_argument("--cg-iters", type=int, default=200, dest="cg_iters", help="Maximum CG iterations.")
    parser.add_argument("--viz-npz", type=str, default=None, dest="viz_npz", help="Optional path to export NPZ for visualisation.")
    _snapshot_defaults(parser, "_infer_defaults")
    return parser


def _add_baseline_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = sub.add_parser("baseline", help="Run heat-method distance baseline.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config overriding defaults.")
    parser.add_argument("--mesh", type=str, default=None, help="Input mesh (.obj/.vtp/.vtk).")
    parser.add_argument("--seeds", type=str, default=None, dest="seeds_file", help="Seeds file with vertex indices.")
    parser.add_argument("--out", type=str, default=None, help="Output directory for baseline results.")
    parser.add_argument("--t", type=float, default=None, help="Optional diffusion time override.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for solves (auto, cpu, cuda, cuda:<idx>, or mps).",
    )
    parser.add_argument("--cg-tol", type=float, default=1e-6, dest="cg_tol", help="Tolerance for CG solves.")
    parser.add_argument("--cg-iters", type=int, default=200, dest="cg_iters", help="Maximum CG iterations.")
    parser.add_argument("--viz-npz", type=str, default=None, dest="viz_npz", help="Optional path to export NPZ for visualisation.")
    _snapshot_defaults(parser, "_baseline_defaults")
    return parser


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-seed geodesic Voronoi via heat-only training.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_train_parser(sub)
    _add_infer_parser(sub)
    _add_baseline_parser(sub)
    return parser


def _run_train(args: argparse.Namespace) -> None:
    defaults = getattr(args, "_train_defaults", {}) or {}
    if hasattr(args, "_train_defaults"):
        delattr(args, "_train_defaults")
    config_data: Dict[str, Any] = {}
    config_arg = getattr(args, "config", None)
    if config_arg:
        config_data = _load_config_mapping(config_arg)
    elif Path(DEFAULT_TRAIN_CONFIG).exists():
        config_data = _load_config_mapping(DEFAULT_TRAIN_CONFIG)
    if hasattr(args, "config"):
        delattr(args, "config")

    valid_keys = [field.name for field in dataclass_fields(TrainConfig)]
    merged = _merge_config(args, defaults, config_data, valid_keys)
    if "stage2_w_jump" not in merged and "stage2_w_mirror" in merged:
        merged["stage2_w_jump"] = merged.pop("stage2_w_mirror")
    if "stage3_w_jump" not in merged and "stage3_w_mirror" in merged:
        merged["stage3_w_jump"] = merged.pop("stage3_w_mirror")

    missing = [key for key in ("mesh", "seeds_file") if not merged.get(key)]
    if missing:
        raise ValueError(
            "Missing required training option(s): " + ", ".join(missing) + "."
        )

    cfg = TrainConfig(**merged)
    train(cfg)


def _run_infer(args: argparse.Namespace) -> None:
    defaults = getattr(args, "_infer_defaults", {}) or {}
    if hasattr(args, "_infer_defaults"):
        delattr(args, "_infer_defaults")
    config_data: Dict[str, Any] = {}
    config_arg = getattr(args, "config", None)
    if config_arg:
        config_data = _load_config_mapping(config_arg)
    elif Path(DEFAULT_INFER_CONFIG).exists():
        config_data = _load_config_mapping(DEFAULT_INFER_CONFIG)
    if hasattr(args, "config"):
        delattr(args, "config")

    valid_keys = [
        "mesh",
        "seeds_file",
        "out",
        "t",
        "device",
        "cg_tol",
        "cg_iters",
        "viz_npz",
    ]
    merged = _merge_config(args, defaults, config_data, valid_keys)

    missing = [key for key in ("mesh", "seeds_file", "out") if not merged.get(key)]
    if missing:
        raise ValueError(
            "Missing required inference option(s): " + ", ".join(missing) + "."
        )

    device = merged.get("device", defaults.get("device", "auto"))
    cg_tol = merged.get("cg_tol", defaults.get("cg_tol", 1.0e-6))
    cg_iters = merged.get("cg_iters", defaults.get("cg_iters", 300))
    viz_npz = merged.get("viz_npz", defaults.get("viz_npz"))

    infer_and_export(
        mesh_path=merged["mesh"],
        seeds_file=merged["seeds_file"],
        out_dir=merged["out"],
        t=merged.get("t"),
        device=device,
        cg_tol=cg_tol,
        cg_iters=cg_iters,
        viz_npz=viz_npz,
    )


def _run_baseline(args: argparse.Namespace) -> None:
    defaults = getattr(args, "_baseline_defaults", {}) or {}
    if hasattr(args, "_baseline_defaults"):
        delattr(args, "_baseline_defaults")
    config_data: Dict[str, Any] = {}
    config_arg = getattr(args, "config", None)
    if config_arg:
        config_data = _load_config_mapping(config_arg)
    elif Path(DEFAULT_INFER_CONFIG).exists():
        config_data = _load_config_mapping(DEFAULT_INFER_CONFIG)
    if hasattr(args, "config"):
        delattr(args, "config")

    valid_keys = [
        "mesh",
        "seeds_file",
        "out",
        "t",
        "device",
        "cg_tol",
        "cg_iters",
        "viz_npz",
    ]
    merged = _merge_config(args, defaults, config_data, valid_keys)

    missing = [key for key in ("mesh", "seeds_file", "out") if not merged.get(key)]
    if missing:
        raise ValueError(
            "Missing required baseline option(s): " + ", ".join(missing) + "."
        )

    device = merged.get("device", defaults.get("device", "auto"))
    cg_tol = merged.get("cg_tol", defaults.get("cg_tol", 1.0e-6))
    cg_iters = merged.get("cg_iters", defaults.get("cg_iters", 300))
    viz_npz = merged.get("viz_npz", defaults.get("viz_npz"))

    heat_distance_baseline(
        mesh_path=merged["mesh"],
        seeds_file=merged["seeds_file"],
        out_dir=merged["out"],
        t=merged.get("t"),
        device=device,
        cg_tol=cg_tol,
        cg_iters=cg_iters,
        viz_npz=viz_npz,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.cmd == "train":
        _run_train(args)
    elif args.cmd == "infer":
        _run_infer(args)
    elif args.cmd == "baseline":
        _run_baseline(args)
    else:
        raise ValueError(f"Unknown command {args.cmd!r}")


if __name__ == "__main__":
    main()
