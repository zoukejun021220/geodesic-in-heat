"""
Loss functions and seam residual utilities for the Voronoi heat pipeline.

This module collects loss terms to keep the training pipeline concise:
- Smoothness and area balance regularizers
- Gradient-only seam objectives on the original mesh
- Polyline direction consistency loss (Stage-3)
- Cut-mesh seam residuals and an interface penalty gradient (prox coupling)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .grad_alignment import edge_pair_mirror_scores, edge_pair_mirror_scores_from_X
from .voronoi_heat_torch import (
    face_gradients,
    topk_pairs_per_face,
    argmin_labels,
    VoronoiHeatModel,
)


def _safe_normalize(x: Tensor, dim: int = -1, eps: float = 1.0e-9) -> Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


# -----------------------------------------------------------------------------
# Regularizers
# -----------------------------------------------------------------------------
def smoothness_loss_optimized(f_values: Tensor, vertex_edges: Tensor | None) -> Tensor:
    """Vectorised smoothness loss on per-vertex channels.

    Args:
        f_values: (N, C) scalar field logits per vertex (channels in dim=1).
        vertex_edges: (E, 2) undirected edges.
    """

    if f_values.numel() == 0 or vertex_edges is None or vertex_edges.numel() == 0:
        return f_values.new_tensor(0.0, device=f_values.device)

    edges = vertex_edges
    if edges.device != f_values.device:
        edges = edges.to(device=f_values.device)

    v1 = f_values[edges[:, 0]]  # (E, C)
    v2 = f_values[edges[:, 1]]  # (E, C)
    diff = v1 - v2
    return diff.pow(2).sum()


def area_balance_loss_optimized(
    points: Tensor,
    triangles: Tensor,
    f_values: Tensor,
    beta: float,
    mesh_area: Tensor | float,
) -> Tuple[Tensor, Tensor]:
    """Area balance regulariser using barycentric samples and softmax weighting.

    Returns (loss, fractions_per_channel).
    """

    if triangles.numel() == 0 or f_values.numel() == 0:
        zero = f_values.new_tensor(0.0, device=f_values.device)
        return zero, zero.new_zeros((f_values.shape[1],), dtype=f_values.dtype)

    device = points.device
    dtype = points.dtype
    num_channels = f_values.shape[1]

    bary_points = torch.tensor(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=dtype,
        device=device,
    )
    num_samples = bary_points.shape[0]
    bary_weights = torch.full((num_samples,), 1.0 / float(num_samples), dtype=dtype, device=device)

    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    p0 = points[v0]
    p1 = points[v1]
    p2 = points[v2]

    e1 = p1 - p0
    e2 = p2 - p0
    normals = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(normals, dim=1)

    f0 = f_values[v0].unsqueeze(1)
    f1 = f_values[v1].unsqueeze(1)
    f2 = f_values[v2].unsqueeze(1)

    b0 = bary_points[:, 0].view(1, num_samples, 1)
    b1 = bary_points[:, 1].view(1, num_samples, 1)
    b2 = bary_points[:, 2].view(1, num_samples, 1)

    f_interp = f0 * b0 + f1 * b1 + f2 * b2  # (T,S,C)
    s = torch.softmax(float(beta) * f_interp, dim=2)
    s_weighted = s * bary_weights.view(1, num_samples, 1)
    s_mean = s_weighted.sum(dim=1)  # (T,C)

    weighted_areas = s_mean * areas.unsqueeze(1)
    channel_areas = weighted_areas.sum(dim=0)  # (C,)

    mesh_area_tensor = torch.as_tensor(mesh_area, dtype=dtype, device=device)
    fractions = channel_areas / mesh_area_tensor.clamp_min(1.0e-12)

    target = fractions.new_full((num_channels,), 1.0 / float(num_channels))
    loss = torch.sum(torch.abs(fractions - target))
    return loss, fractions


# -----------------------------------------------------------------------------
# Stage-3 polyline direction loss
# -----------------------------------------------------------------------------
def polyline_direction_loss(grad_face: Tensor, cache: Dict[str, Tensor], face_normals: Tensor) -> Tensor:
    faces = cache["faces"]
    if faces.numel() == 0:
        return grad_face.new_tensor(0.0)

    tangent = cache["tangent"]
    weight = cache["weight"].clamp_min(1.0e-9)
    pairs = cache["pairs"]

    grads_face = grad_face[faces]  # (N, C, 3)
    C = grads_face.shape[1]
    pair_mask = (pairs >= 0) & (pairs < C)
    valid_pairs = pair_mask.all(dim=1)
    if not valid_pairs.any():
        return grad_face.new_tensor(0.0)

    faces = faces[valid_pairs]
    tangent = tangent[valid_pairs]
    weight = weight[valid_pairs]
    pairs = pairs[valid_pairs]
    grads_face = grads_face[valid_pairs]

    gather_idx = pairs.unsqueeze(-1).expand(-1, -1, 3)
    grads = torch.gather(grads_face, 1, gather_idx)

    face_n = face_normals[faces]
    seam_normal = _safe_normalize(torch.cross(face_n, tangent, dim=1), dim=1)
    grad_diff = grads[:, 0, :] - grads[:, 1, :]
    grad_diff_norm = grad_diff.norm(dim=1, keepdim=True).clamp_min(1.0e-9)
    cos = ((grad_diff / grad_diff_norm) * seam_normal).sum(dim=1).abs().clamp_max(1.0)
    deviation = 1.0 - cos
    loss = (weight * deviation).sum()
    return loss / weight.sum().clamp_min(1.0e-9)


# -----------------------------------------------------------------------------
# On-seam equality and off-seam margin (kept for ablations, can be disabled)
# -----------------------------------------------------------------------------
def eq_margin_losses(
    S: Tensor,
    S_face: Tensor,
    F: Tensor,
    *,
    beta_pairs: float,
    kappa: float,
    K: int,
    tau_eq: float,
    delta: float,
) -> Tuple[Tensor, Tensor]:
    """Compute on-seam equality and off-seam margin losses per face."""
    device = S.device
    dtype = S.dtype
    nF, C = S_face.shape
    if C < 2 or nF == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, zero

    _, pairs, pair_mask = topk_pairs_per_face(S_face, K=K)
    if pairs.numel() == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, zero

    pair_i = pairs[:, :, 0]
    pair_j = pairs[:, :, 1]

    Si = torch.gather(S_face, 1, pair_i)
    Sj = torch.gather(S_face, 1, pair_j)
    logits_pairs = -float(beta_pairs) * (Si + Sj)
    if pair_mask.numel() > 0:
        logits_pairs = logits_pairs.masked_fill(~pair_mask, -1.0e9)
    w_pairs = torch.softmax(logits_pairs, dim=1)
    if pair_mask.numel() > 0:
        w_pairs = w_pairs * pair_mask.to(w_pairs.dtype)

    S_vert = S[F]  # (nF,3,C)
    gather_i = pair_i.unsqueeze(1).expand(-1, 3, -1)
    gather_j = pair_j.unsqueeze(1).expand(-1, 3, -1)
    Svi = torch.gather(S_vert, 2, gather_i)
    Svj = torch.gather(S_vert, 2, gather_j)
    Delta = Svi - Svj  # (nF,3,P)
    edge_idx = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
    Da = Delta[:, edge_idx[:, 0], :]
    Db = Delta[:, edge_idx[:, 1], :]
    prod = Da * Db
    p_edge = torch.sigmoid(-float(kappa) * prod)  # (nF,3,P)
    p_face = p_edge.mean(dim=1)  # (nF,P)

    # On-seam equality loss
    eq_w = (w_pairs * p_face).clamp_min(1.0e-9)
    eq_num = (eq_w * (Si - Sj).abs()).sum()
    eq_den = eq_w.sum().clamp_min(1.0e-9)
    eq_loss = eq_num / eq_den

    # Off-seam margin
    seam_strength = (w_pairs * p_face).sum(dim=1)  # (nF,)
    margin_weight = (1.0 - seam_strength).clamp_min(0.0)
    diff_sorted, _ = torch.sort(S_face, dim=1)
    # ensure C>=2
    if S_face.shape[1] >= 2:
        gap = diff_sorted[:, 1] - diff_sorted[:, 0]
        margin_loss = torch.relu(float(delta) - gap) * margin_weight
        margin_loss = margin_loss.mean()
    else:
        margin_loss = torch.zeros((), device=device, dtype=dtype)

    return eq_loss, margin_loss


# -----------------------------------------------------------------------------
# Gradient-only seam residual on original mesh (label-change edges)
# -----------------------------------------------------------------------------
def gradient_only_edge_loss(
    model: VoronoiHeatModel,
    S: Tensor,
    *,
    lam_tau: float = 1.0,
    lam_u: float = 0.0,
    beta_edge: float = 12.0,
) -> Tensor:
    """Gradient-only residual across label-change edges using left/right faces."""

    device = model.V.device
    dtype = model.V.dtype
    edge_idx, edge_tris, pair_ids, scores, conf = edge_pair_mirror_scores(
        model.V, model.F, S, beta_edge=beta_edge
    )
    if edge_tris.numel() == 0 or edge_idx.numel() == 0:
        return torch.zeros((), device=device, dtype=dtype)

    va = edge_idx[:, 0]
    vb = edge_idx[:, 1]
    tL = edge_tris[:, 0]
    tR = edge_tris[:, 1]

    with torch.no_grad():
        labels = argmin_labels(S)
    li = labels[va]
    lj = labels[vb]
    diff = li != lj
    if not diff.any():
        return torch.zeros((), device=device, dtype=dtype)

    va = va[diff]
    vb = vb[diff]
    tL = tL[diff]
    tR = tR[diff]
    li = li[diff]
    lj = lj[diff]
    conf = conf[diff].clamp_min(1.0e-6)

    grad_face = face_gradients(S, model.F, model.gI, model.gJ, model.gK)
    grad_i_L = grad_face[tL, li, :]
    grad_j_R = grad_face[tR, lj, :]

    tau = model.V[vb] - model.V[va]
    tau = _safe_normalize(tau, dim=1)
    nL = model.n_hat[tL]
    n_e = _safe_normalize(torch.cross(nL, tau, dim=1), dim=1)

    nGi = (grad_i_L * n_e).sum(dim=1)
    nGj = (grad_j_R * n_e).sum(dim=1)
    tGi = (grad_i_L * tau).sum(dim=1)
    tGj = (grad_j_R * tau).sum(dim=1)

    r = (nGi + nGj) ** 2 + float(lam_tau) * (tGi - tGj) ** 2
    if lam_u > 0.0:
        Xi = _safe_normalize(grad_i_L, dim=1)
        Xj = _safe_normalize(grad_j_R, dim=1)
        r = r + float(lam_u) * (((Xi + Xj) * n_e).sum(dim=1) ** 2 + ((Xi - Xj) * tau).sum(dim=1) ** 2)
    # Guard against NaNs/Infs from degenerate edges or transports
    r = torch.nan_to_num(r, nan=0.0, posinf=1.0e6, neginf=1.0e6)

    e_len = (model.V[vb] - model.V[va]).norm(dim=1)
    len_gate = (e_len / (e_len.mean().detach() + 1.0e-12)).clamp(0.5, 2.0)
    w = conf * len_gate
    return (w * r).sum() / w.sum().clamp_min(1.0e-9)


def gradient_only_edge_loss_x(
    model: VoronoiHeatModel,
    X_face: Tensor,
    *,
    lam_tau: float = 1.0,
    lam_u: float = 0.0,
    beta_edge: float = 12.0,
) -> Tensor:
    """Gradient-only residual using unit directions (no labels, no scores).

    Uses hinge-transported unit fields to compute per-edge residual and aggregates
    with confidence from the X-based mirror scorer. Avoids label-change gating.
    """
    device = model.V.device
    dtype = model.V.dtype
    edge_idx, edge_tris, pair_ids, scores, conf = edge_pair_mirror_scores_from_X(
        model.V, model.F, X_face, beta_edge=beta_edge
    )
    if edge_tris.numel() == 0 or pair_ids.numel() == 0:
        return torch.zeros((), device=device, dtype=dtype)

    # Build edge tangent and in-plane normal from geometry
    va = edge_idx[:, 0]
    vb = edge_idx[:, 1]
    tL = edge_tris[:, 0]
    tR = edge_tris[:, 1]
    tau = _safe_normalize(model.V[vb] - model.V[va], dim=1)
    nL = model.n_hat[tL]
    n_e = _safe_normalize(torch.cross(nL, tau, dim=1), dim=1)

    # Gather unit directions for selected pairs per edge
    C = X_face.shape[1]
    P = pair_ids.shape[0]
    if C == 0 or P == 0:
        return torch.zeros((), device=device, dtype=dtype)

    # We will pick the best pair per edge as indicated by highest score
    scores_conf = scores * conf.clamp_min(1.0e-4).unsqueeze(1)
    best_idx = torch.argmax(scores_conf, dim=1)
    pair_best = pair_ids[best_idx]
    i = pair_best[:, 0]
    j = pair_best[:, 1]

    Xi_L = X_face[tL, i, :]
    Xj_R = X_face[tR, j, :]

    nXi = (Xi_L * n_e).sum(dim=1)
    nXj = (Xj_R * n_e).sum(dim=1)
    tXi = (Xi_L * tau).sum(dim=1)
    tXj = (Xj_R * tau).sum(dim=1)

    r = (nXi + nXj) ** 2 + float(lam_tau) * (tXi - tXj) ** 2
    if lam_u > 0.0:
        # Xi_L and Xj_R are already unit fields
        r = r + float(lam_u) * (
            ((Xi_L + Xj_R) * n_e).sum(dim=1) ** 2 + ((Xi_L - Xj_R) * tau).sum(dim=1) ** 2
        )
    r = torch.nan_to_num(r, nan=0.0, posinf=1.0e6, neginf=1.0e6)

    e_len = (model.V[vb] - model.V[va]).norm(dim=1)
    len_gate = (e_len / (e_len.mean().detach() + 1.0e-12)).clamp(0.5, 2.0)
    w = conf.clamp_min(1.0e-6) * len_gate
    return (w * r).sum() / w.sum().clamp_min(1.0e-9)


# -----------------------------------------------------------------------------
# Cut-mesh seam residuals and coupling
# -----------------------------------------------------------------------------
def seam_resolve_loss_on_cut(
    model_cut: VoronoiHeatModel,
    S_cut: Tensor,
    seam_infos: List[Tuple[int, int, int, int, int, Tuple[int, int], Tuple[int, int, int], np.ndarray, np.ndarray]],
    *,
    lam_tau: float,
    lam_eq: float,
    lam_u: float = 0.0,
) -> Tensor:
    """Compute left/right residual on cut mesh seam edges after re-solve."""
    device = model_cut.V.device
    dtype = model_cut.V.dtype
    if not seam_infos:
        return torch.zeros((), device=device, dtype=dtype)

    grad_face = face_gradients(S_cut, model_cut.F, model_cut.gI, model_cut.gJ, model_cut.gK)
    losses: List[Tensor] = []
    for pL, qL, faceL, pR, qR, faceR, pair, tri, b0, b1 in seam_infos:
        i, j = pair
        v_u = model_cut.V[torch.tensor(qL, device=device, dtype=torch.long)] - model_cut.V[torch.tensor(pL, device=device, dtype=torch.long)]
        tau = _safe_normalize(v_u.unsqueeze(0), dim=1).squeeze(0)
        nL = model_cut.n_hat[torch.tensor(faceL, device=device, dtype=torch.long)]
        n_e = _safe_normalize(torch.cross(nL, tau, dim=0).unsqueeze(0), dim=1).squeeze(0)

        g_i_L = grad_face[faceL, i]
        g_j_R = grad_face[faceR, j]
        nGi = (g_i_L * n_e).sum()
        nGj = (g_j_R * n_e).sum()
        tGi = (g_i_L * tau).sum()
        tGj = (g_j_R * tau).sum()

        r = (nGi + nGj) ** 2 + float(lam_tau) * (tGi - tGj) ** 2
        # Optional unit-direction symmetry terms (parity with edge loss)
        if lam_u > 0.0:
            Xi = _safe_normalize(g_i_L.unsqueeze(0), dim=1).squeeze(0)
            Xj = _safe_normalize(g_j_R.unsqueeze(0), dim=1).squeeze(0)
            r = r + float(lam_u) * (
                ((Xi + Xj) * n_e).sum() ** 2 + ((Xi - Xj) * tau).sum() ** 2
            )
        if lam_eq > 0.0:
            Si_p = S_cut[pL, i]
            Sj_p = S_cut[pL, j]
            Si_q = S_cut[qL, i]
            Sj_q = S_cut[qL, j]
            r = r + float(lam_eq) * (0.5 * ((Si_p - Sj_p) ** 2 + (Si_q - Sj_q) ** 2))
        # Guard each term to avoid NaNs/inf accumulation
        r = torch.nan_to_num(r, nan=0.0, posinf=1.0e6, neginf=1.0e6)
        losses.append(r)
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device, dtype=dtype)


def interface_penalty_grad_U(
    model_cut: VoronoiHeatModel,
    U_cut: Tensor,
    seam_infos: List[Tuple[int, int, int, int, int, Tuple[int, int], Tuple[int, int, int], np.ndarray, np.ndarray]],
    *,
    lam_tau: float,
) -> Tensor:
    """Gradient of seam interface penalty wrt U on the cut mesh (prox coupling)."""
    device = model_cut.V.device
    dtype = model_cut.V.dtype
    nV = model_cut.V.shape[0]
    C = U_cut.shape[1]
    gradU = torch.zeros((nV, C), device=device, dtype=dtype)

    F = model_cut.F
    gI = model_cut.gI
    gJ = model_cut.gJ
    gK = model_cut.gK

    for pL, qL, faceL, pR, qR, faceR, pair, tri, b0, b1 in seam_infos:
        i, j = pair
        if i < 0 or i >= C or j < 0 or j >= C:
            continue
        v_u = model_cut.V[torch.tensor(qL, device=device, dtype=torch.long)] - model_cut.V[torch.tensor(pL, device=device, dtype=torch.long)]
        tau = _safe_normalize(v_u.unsqueeze(0), dim=1).squeeze(0)
        nL = model_cut.n_hat[torch.tensor(faceL, device=device, dtype=torch.long)]
        n_e = _safe_normalize(torch.cross(nL, tau, dim=0).unsqueeze(0), dim=1).squeeze(0)

        fL = F[torch.tensor(faceL, device=device, dtype=torch.long)]
        fR = F[torch.tensor(faceR, device=device, dtype=torch.long)]

        uLi = torch.stack([U_cut[fL[0], i], U_cut[fL[1], i], U_cut[fL[2], i]])
        uRj = torch.stack([U_cut[fR[0], j], U_cut[fR[1], j], U_cut[fR[2], j]])
        gL = uLi[0] * gI[faceL] + uLi[1] * gJ[faceL] + uLi[2] * gK[faceL]
        gR = uRj[0] * gI[faceR] + uRj[1] * gJ[faceR] + uRj[2] * gK[faceR]

        r_n = (gL * n_e).sum() + (gR * n_e).sum()
        r_t = (gL * tau).sum() - (gR * tau).sum()
        w = (v_u.norm() + 1.0e-9)

        n_dot_L = torch.stack([
            (n_e * gI[faceL]).sum(),
            (n_e * gJ[faceL]).sum(),
            (n_e * gK[faceL]).sum(),
        ])
        t_dot_L = torch.stack([
            (tau * gI[faceL]).sum(),
            (tau * gJ[faceL]).sum(),
            (tau * gK[faceL]).sum(),
        ])
        n_dot_R = torch.stack([
            (n_e * gI[faceR]).sum(),
            (n_e * gJ[faceR]).sum(),
            (n_e * gK[faceR]).sum(),
        ])
        t_dot_R = torch.stack([
            (tau * gI[faceR]).sum(),
            (tau * gJ[faceR]).sum(),
            (tau * gK[faceR]).sum(),
        ])

        coeff_n = 2.0 * r_n * w
        coeff_t = 2.0 * r_t * float(lam_tau) * w

        idxL = fL.long()
        idxR = fR.long()
        gradU[idxL[0], i] += coeff_n * n_dot_L[0] + coeff_t * t_dot_L[0]
        gradU[idxL[1], i] += coeff_n * n_dot_L[1] + coeff_t * t_dot_L[1]
        gradU[idxL[2], i] += coeff_n * n_dot_L[2] + coeff_t * t_dot_L[2]

        gradU[idxR[0], j] += coeff_n * n_dot_R[0] - coeff_t * t_dot_R[0]
        gradU[idxR[1], j] += coeff_n * n_dot_R[1] - coeff_t * t_dot_R[1]
        gradU[idxR[2], j] += coeff_n * n_dot_R[2] - coeff_t * t_dot_R[2]

    return gradU


__all__ = [
    "smoothness_loss_optimized",
    "area_balance_loss_optimized",
    "polyline_direction_loss",
    "eq_margin_losses",
    "gradient_only_edge_loss",
    "seam_resolve_loss_on_cut",
    "interface_penalty_grad_U",
]
