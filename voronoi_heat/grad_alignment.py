"""
Edge-based gradient alignment loss to encourage smooth Voronoi seams.

Moved into the voronoi_heat package so pipeline imports remain self-contained.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .hinge_transport import transport_right_to_left

Tensor = torch.Tensor

ALIGN_STATS_KEYS: Tuple[str, ...] = (
    "align_pair_prior",
    "align_edge_conf",
    "align_cos",
    "align_mirror",
)


def empty_align_stats() -> Dict[str, float]:
    """Return zeroed statistics for alignment diagnostics."""
    return {key: 0.0 for key in ALIGN_STATS_KEYS}


def _safe_normalize(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _robust_quantile(values: Tensor, q: float, default: float = 1.0) -> Tensor:
    """Return quantile with kthvalue fallback for limited backends."""

    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.new_tensor(default)
    try:
        return torch.quantile(flat, q)
    except RuntimeError:
        k = int(q * max(flat.numel() - 1, 0))
        kth = flat.kthvalue(k + 1).values
        return kth


def _grad3d_intrinsic_pairs(
    e0: Tensor,
    e1: Tensor,
    h: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    """Compute intrinsic gradients for many channel pairs on identical triangle geometry."""

    G00 = (e0 * e0).sum(dim=1)
    G01 = (e0 * e1).sum(dim=1)
    G11 = (e1 * e1).sum(dim=1)
    det = (G00 * G11 - G01 * G01).clamp_min(eps)
    invG00 = G11 / det
    invG01 = -G01 / det
    invG11 = G00 / det

    b0 = h[:, 1, :] - h[:, 0, :]
    b1 = h[:, 2, :] - h[:, 0, :]

    a0 = invG00.unsqueeze(1) * b0 + invG01.unsqueeze(1) * b1
    a1 = invG01.unsqueeze(1) * b0 + invG11.unsqueeze(1) * b1

    g = a0.unsqueeze(2) * e0.unsqueeze(1) + a1.unsqueeze(2) * e1.unsqueeze(1)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


_EDGE_CACHE: dict[Tuple[int, Tuple[int, ...], str], Tuple[Tensor, Tensor]] = {}
_EDGE_CACHE_LIMIT = 16


def _build_edges_and_adjacency(faces: Tensor) -> Tuple[Tensor, Tensor]:
    """Build undirected edges and adjacency from triangle faces."""

    # Use untyped_storage() to avoid TypedStorage deprecation warnings
    key = (int(faces.untyped_storage().data_ptr()), tuple(faces.shape), faces.device.type)
    cached = _EDGE_CACHE.get(key)
    if cached is not None:
        return cached

    device = faces.device
    T = faces.shape[0]

    edges = torch.stack(
        [
            torch.stack([faces[:, 0], faces[:, 1]], dim=1),
            torch.stack([faces[:, 1], faces[:, 2]], dim=1),
            torch.stack([faces[:, 2], faces[:, 0]], dim=1),
        ],
        dim=1,
    ).reshape(-1, 2)

    edges_sorted, _ = torch.sort(edges, dim=1)
    face_indices = torch.arange(T, device=device, dtype=faces.dtype).repeat_interleave(3)

    vmax = faces.max().to(torch.int64) + 1
    edge_hash = edges_sorted[:, 0].to(torch.int64) * vmax + edges_sorted[:, 1].to(torch.int64)
    sort_hash, sort_idx = torch.sort(edge_hash)
    sorted_edges = edges_sorted[sort_idx]
    sorted_faces = face_indices[sort_idx]

    change = torch.ones(sort_hash.shape[0], dtype=torch.bool, device=device)
    change[1:] = sort_hash[1:] != sort_hash[:-1]
    change_idx = torch.nonzero(change, as_tuple=False).squeeze(1)
    change_idx = torch.cat([change_idx, sort_hash.new_tensor([sort_hash.shape[0]])])

    counts = change_idx[1:] - change_idx[:-1]
    edge_idx_unique = sorted_edges[change_idx[:-1]]

    U = edge_idx_unique.shape[0]
    edge_tris = torch.full((U, 2), -1, dtype=faces.dtype, device=device)
    first_pos = change_idx[:-1]
    edge_tris[:, 0] = sorted_faces[first_pos]
    has_second = counts >= 2
    if has_second.any():
        second_pos = first_pos[has_second] + 1
        edge_tris[has_second, 1] = sorted_faces[second_pos]

    if len(_EDGE_CACHE) >= _EDGE_CACHE_LIMIT:
        oldest_key = next(iter(_EDGE_CACHE))
        if oldest_key != key:
            _EDGE_CACHE.pop(oldest_key)
    _EDGE_CACHE[key] = (edge_idx_unique, edge_tris)
    return edge_idx_unique, edge_tris


def _all_pair_indices(C: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    if C < 2:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    if device.type == "mps":
        ii: List[int] = []
        jj: List[int] = []
        for a in range(C):
            for b in range(a + 1, C):
                ii.append(a)
                jj.append(b)
        return (
            torch.tensor(ii, dtype=torch.long, device=device),
            torch.tensor(jj, dtype=torch.long, device=device),
        )
    return torch.triu_indices(C, C, offset=1, device=device)


def _pair_lut(C: int, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """Return (ii, jj, lut) where lut[a, b] gives the pair id for unordered pair (a,b)."""

    ii, jj = _all_pair_indices(C, device)
    P = ii.numel()
    lut = -torch.ones((C, C), dtype=torch.long, device=device)
    if P > 0:
        ids = torch.arange(P, device=device, dtype=torch.long)
        lut[ii, jj] = ids
        lut[jj, ii] = ids
    return ii, jj, lut


def contour_alignment_codex(
    vertices: Tensor,
    faces: Tensor,
    f_values: Tensor,
    *,
    beta_edge: float = 12.0,
    return_stats: bool = False,
    include_triples: bool = False,
    edge_idx: Tensor | None = None,
    edge_tris: Tensor | None = None,
    pair_mask_edge: Tensor | None = None,
    eq_gate_edge: Tensor | None = None,
    use_soft_flip: bool = False,
    softflip_beta: float = 12.0,
) -> Tensor | Tuple[Tensor, Dict[str, float]]:
    """Edge-based alignment penalty encouraging smooth Voronoi seams."""

    device = f_values.device
    dtype = f_values.dtype
    _, C = f_values.shape
    zero = torch.zeros((), device=device, dtype=dtype)
    if C < 2:
        if return_stats:
            return zero, empty_align_stats()
        return zero

    if edge_idx is None or edge_tris is None:
        edge_idx, edge_tris = _build_edges_and_adjacency(faces)
    valid = (edge_tris[:, 0] >= 0) & (edge_tris[:, 1] >= 0)
    if not valid.any():
        if return_stats:
            return zero, empty_align_stats()
        return zero

    edge_idx = edge_idx[valid]
    edge_tris = edge_tris[valid]
    va = edge_idx[:, 0]
    vb = edge_idx[:, 1]
    tL = edge_tris[:, 0]
    tR = edge_tris[:, 1]

    ii, jj = torch.triu_indices(C, C, offset=1, device=device)
    P = ii.numel()
    if P == 0:
        if return_stats:
            return zero, empty_align_stats()
        return zero

    faces_L = faces[tL]
    faces_R = faces[tR]

    field_values = torch.softmax(-softflip_beta * f_values, dim=1) if use_soft_flip else f_values

    fL = field_values[faces_L]
    fR = field_values[faces_R]
    hL = fL[:, :, ii] - fL[:, :, jj]
    hR = fR[:, :, ii] - fR[:, :, jj]

    v0L = vertices[faces_L[:, 0]]
    v1L = vertices[faces_L[:, 1]]
    v2L = vertices[faces_L[:, 2]]
    v0R = vertices[faces_R[:, 0]]
    v1R = vertices[faces_R[:, 1]]
    v2R = vertices[faces_R[:, 2]]
    e0L = v1L - v0L
    e1L = v2L - v0L
    e0R = v1R - v0R
    e1R = v2R - v0R

    gL = _grad3d_intrinsic_pairs(e0L, e1L, hL)
    gR = _grad3d_intrinsic_pairs(e0R, e1R, hR)

    def normals(v0: Tensor, v1: Tensor, v2: Tensor) -> Tensor:
        e0 = v1 - v0
        e1 = v2 - v0
        n = torch.cross(e0, e1, dim=1)
        return _safe_normalize(n, dim=1)

    nL = normals(v0L, v1L, v2L)
    nR = normals(v0R, v1R, v2R)

    def proj_in_plane(g: Tensor, n: Tensor) -> Tensor:
        dot = (g * n[:, None, :]).sum(dim=2, keepdim=True)
        return g - dot * n[:, None, :]

    gL = proj_in_plane(gL, nL)
    gR = proj_in_plane(gR, nR)

    tauL = _safe_normalize(torch.cross(nL[:, None, :], gL, dim=2), dim=2)
    tauR = _safe_normalize(torch.cross(nR[:, None, :], gR, dim=2), dim=2)

    Fa = field_values[va][:, ii] - field_values[va][:, jj]
    Fb = field_values[vb][:, ii] - field_values[vb][:, jj]
    w_pairs = torch.sigmoid(-beta_edge * Fa * Fb)
    conf = 0.5 * (Fa.abs() + Fb.abs())
    w_pairs = w_pairs * torch.sigmoid(5.0 * (conf - 0.2))

    if pair_mask_edge is not None:
        mask_edge = pair_mask_edge[valid]
        if mask_edge.shape != w_pairs.shape:
            raise ValueError("pair_mask_edge has incompatible shape for filtered edges.")
        allowed = mask_edge.to(dtype=w_pairs.dtype)
        lockdown = allowed.sum(dim=1, keepdim=True) == 0
        if lockdown.any():
            allowed = torch.where(lockdown, torch.ones_like(allowed), allowed)
        w_pairs = w_pairs * allowed

    w_pairs = w_pairs.clamp_min(1.0e-6)
    pair_mix = w_pairs / (w_pairs.sum(dim=1, keepdim=True) + 1.0e-9)
    phi = 1.0 - torch.prod(1.0 - w_pairs, dim=1)

    cosang = (tauL * tauR).sum(dim=2).abs().clamp_max(1.0)
    mis = 1.0 - cosang
    mis_edge = (pair_mix * mis).sum(dim=1)

    gLm = gL.norm(dim=2).mean(dim=1) + 1.0e-8
    gRm = gR.norm(dim=2).mean(dim=1) + 1.0e-8
    grad_gate = torch.sqrt(gLm * gRm)
    finite_mask = torch.isfinite(grad_gate)
    if finite_mask.any():
        scale = torch.nanmedian(grad_gate[finite_mask]).clamp_min(1.0e-6)
    else:
        scale = grad_gate.new_tensor(1.0)
    grad_gate = (grad_gate / scale).clamp(0.0, 2.0)
    grad_gate = torch.nan_to_num(grad_gate, nan=0.0, posinf=2.0, neginf=0.0)

    edge_vec = vertices[vb] - vertices[va]
    edge_len = edge_vec.norm(dim=1)
    len_gate = (edge_len / (edge_len.mean().detach() + 1.0e-12)).clamp(0.5, 2.0)

    w_edge = phi.clamp_min(1.0e-4) * grad_gate * len_gate

    if eq_gate_edge is not None:
        gate = eq_gate_edge[valid]
        if gate.shape != w_pairs.shape:
            raise ValueError("eq_gate_edge has incompatible shape for filtered edges.")
        mix_gate = (pair_mix * gate.to(dtype)).sum(dim=1).clamp_min(1.0e-4)
        w_edge = w_edge * mix_gate

    loss_edge = torch.sqrt(mis_edge * mis_edge + 1.0e-6)
    num = (w_edge * loss_edge).sum()
    den = w_edge.sum() + 1.0e-9
    loss = num / den

    if include_triples and C >= 3:
        p_v = torch.softmax(2.0 * f_values, dim=1)
        p_t = (p_v[faces[:, 0]] + p_v[faces[:, 1]] + p_v[faces[:, 2]]) / 3.0
        top3, _ = torch.topk(p_t, k=min(3, C), dim=1)
        if top3.shape[1] == 3:
            gap = top3[:, 1] - top3[:, 2]
            tri_area = 0.5 * torch.linalg.norm(
                torch.cross(
                    vertices[faces[:, 1]] - vertices[faces[:, 0]],
                    vertices[faces[:, 2]] - vertices[faces[:, 0]],
                    dim=1,
                ),
                dim=-1,
            )
            w = tri_area / (tri_area.sum() + 1.0e-9)
            triple = (w * torch.relu(0.10 - gap)).sum()
            loss = loss + 0.1 * triple

    stats = {
        "align_pair_prior": float(w_pairs.mean().detach().item()),
        "align_edge_conf": float(phi.mean().detach().item()),
        "align_cos": float(cosang.mean().detach().item()),
        "align_mirror": float(mis.mean().detach().item()),
    }

    if return_stats:
        return loss, stats
    return loss


@torch.no_grad()
def edge_pair_mirror_scores(
    vertices: Tensor,
    faces: Tensor,
    f_values: Tensor,
    *,
    beta_edge: float = 12.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return hinge-invariant mirror scores per interior edge and candidate pair."""

    device = f_values.device
    dtype = f_values.dtype
    _, C = f_values.shape

    edge_idx, edge_tris = _build_edges_and_adjacency(faces)
    valid = (edge_tris[:, 0] >= 0) & (edge_tris[:, 1] >= 0)
    edge_idx = edge_idx[valid]
    edge_tris = edge_tris[valid]
    if edge_tris.numel() == 0:
        empty_pair = torch.empty(0, 2, dtype=torch.long, device=device)
        empty_scores = torch.zeros(edge_idx.shape[0], 0, dtype=dtype, device=device)
        empty_conf = torch.zeros(edge_idx.shape[0], dtype=dtype, device=device)
        return edge_idx, edge_tris, empty_pair, empty_scores, empty_conf

    tL = edge_tris[:, 0]
    tR = edge_tris[:, 1]

    ii, jj, _ = _pair_lut(C, device)
    P = ii.numel()
    if P == 0:
        empty_pair = torch.empty(0, 2, dtype=torch.long, device=device)
        empty_scores = torch.zeros(edge_idx.shape[0], 0, dtype=dtype, device=device)
        empty_conf = torch.zeros(edge_idx.shape[0], dtype=dtype, device=device)
        return edge_idx, edge_tris, empty_pair, empty_scores, empty_conf

    pair_ids = torch.stack([ii, jj], dim=1)

    faces_L = faces[tL]
    faces_R = faces[tR]
    v0L, v1L, v2L = vertices[faces_L[:, 0]], vertices[faces_L[:, 1]], vertices[faces_L[:, 2]]
    v0R, v1R, v2R = vertices[faces_R[:, 0]], vertices[faces_R[:, 1]], vertices[faces_R[:, 2]]
    e0L, e1L = v1L - v0L, v2L - v0L
    e0R, e1R = v1R - v0R, v2R - v0R

    F_L = f_values[faces_L]
    F_R = f_values[faces_R]
    hL = F_L[:, :, ii] - F_L[:, :, jj]
    hR = F_R[:, :, ii] - F_R[:, :, jj]

    gL = _grad3d_intrinsic_pairs(e0L, e1L, hL)
    gR = _grad3d_intrinsic_pairs(e0R, e1R, hR)

    def normals(v0: Tensor, v1: Tensor, v2: Tensor) -> Tensor:
        n = torch.cross(v1 - v0, v2 - v0, dim=1)
        return _safe_normalize(n, dim=1)

    nL = normals(v0L, v1L, v2L)
    nR = normals(v0R, v1R, v2R)

    def project(g: Tensor, n: Tensor) -> Tensor:
        return g - (g * n[:, None, :]).sum(dim=2, keepdim=True) * n[:, None, :]

    gL = project(gL, nL)
    gR = project(gR, nR)

    gR_cf = gR.permute(1, 0, 2)
    gR_rot_cf = transport_right_to_left(
        gR_cf,
        vertices=vertices,
        edge_vertices=edge_idx,
        nL=nL,
        nR=nR,
    )
    gR_rot = gR_rot_cf.permute(1, 0, 2)

    e_vec = vertices[edge_idx[:, 1]] - vertices[edge_idx[:, 0]]
    e_hat = _safe_normalize(e_vec, dim=1)[:, None, :]
    nL_bp = nL[:, None, :]
    n_in = torch.cross(e_hat, nL_bp, dim=2)

    gL_edge = (gL * e_hat).sum(dim=2)
    gR_edge = (gR_rot * e_hat).sum(dim=2)
    gL_norm = (gL * n_in).sum(dim=2)
    gR_norm = (gR_rot * n_in).sum(dim=2)

    residual = (gL_edge - gR_edge) ** 2 + (gL_norm + gR_norm) ** 2

    mag = torch.minimum(gL.norm(dim=2), gR_rot.norm(dim=2)) + 1e-12
    sigma = _robust_quantile(torch.sqrt(residual + 1e-12), 0.7).clamp(min=1e-6)
    scores = torch.exp(-residual / (sigma * sigma)) * mag

    dot = (gL * gR_rot).sum(dim=2)
    nL = gL.norm(dim=2).clamp_min(1.0e-12)
    nR = gR_rot.norm(dim=2).clamp_min(1.0e-12)
    cos = (dot / (nL * nR)).clamp(-1.0, 1.0)
    opp = 0.5 * (1.0 - cos)
    conf_edge = 1.0 - torch.prod(1.0 - torch.sigmoid(beta_edge * opp), dim=1)

    return edge_idx, edge_tris, pair_ids, scores, conf_edge


@torch.no_grad()
def edge_pair_mirror_scores_from_X(
    vertices: Tensor,
    faces: Tensor,
    X_face: Tensor,
    *,
    beta_edge: float = 12.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Mirror residual scores computed from per-seed unit fields after hinge transport.
    Returns edge indices, adjacent triangles, unordered pair ids, scores, and confidence.
    """

    device = X_face.device
    dtype = X_face.dtype
    F, C, _ = X_face.shape

    edge_idx, edge_tris = _build_edges_and_adjacency(faces)
    valid = (edge_tris[:, 0] >= 0) & (edge_tris[:, 1] >= 0)
    edge_idx = edge_idx[valid]
    edge_tris = edge_tris[valid]
    if edge_tris.numel() == 0 or C < 2:
        empty_pair = torch.empty(0, 2, dtype=torch.long, device=device)
        empty_scores = torch.zeros(edge_idx.shape[0], 0, dtype=dtype, device=device)
        empty_conf = torch.zeros(edge_idx.shape[0], dtype=dtype, device=device)
        return edge_idx, edge_tris, empty_pair, empty_scores, empty_conf

    tL = edge_tris[:, 0]
    tR = edge_tris[:, 1]
    XL = X_face[tL]  # (E, C, 3)
    XR = X_face[tR]  # (E, C, 3)

    v0L, v1L, v2L = vertices[faces[tL, 0]], vertices[faces[tL, 1]], vertices[faces[tL, 2]]
    v0R, v1R, v2R = vertices[faces[tR, 0]], vertices[faces[tR, 1]], vertices[faces[tR, 2]]
    nL = _safe_normalize(torch.cross(v1L - v0L, v2L - v0L, dim=1), dim=1)
    nR = _safe_normalize(torch.cross(v1R - v0R, v2R - v0R, dim=1), dim=1)

    XR_cf = XR.permute(1, 0, 2)
    XR_rot_cf = transport_right_to_left(
        XR_cf,
        vertices=vertices,
        edge_vertices=edge_idx,
        nL=nL,
        nR=nR,
    )
    XR_rot = XR_rot_cf.permute(1, 0, 2)

    e_vec = vertices[edge_idx[:, 1]] - vertices[edge_idx[:, 0]]
    e_hat = _safe_normalize(e_vec, dim=1)[:, None, :]
    nL_bp = nL[:, None, :]
    n_in = torch.cross(e_hat, nL_bp, dim=2)

    ii, jj, _ = _pair_lut(C, device=device)
    P = ii.numel()
    if P == 0:
        empty_pair = torch.empty(0, 2, dtype=torch.long, device=device)
        empty_scores = torch.zeros(edge_idx.shape[0], 0, dtype=dtype, device=device)
        empty_conf = torch.zeros(edge_idx.shape[0], dtype=dtype, device=device)
        return edge_idx, edge_tris, empty_pair, empty_scores, empty_conf

    pair_ids = torch.stack([ii, jj], dim=1)

    Xi = XL[:, ii, :]
    Xj = XR_rot[:, jj, :]

    Xi_edge = (Xi * e_hat).sum(dim=2)
    Xj_edge = (Xj * e_hat).sum(dim=2)
    Xi_norm = (Xi * n_in).sum(dim=2)
    Xj_norm = (Xj * n_in).sum(dim=2)

    residual = (Xi_edge - Xj_edge) ** 2 + (Xi_norm + Xj_norm) ** 2
    residual_flat = residual.reshape(-1)
    if residual_flat.numel() > 0:
        sigma = _robust_quantile(torch.sqrt(residual_flat + 1.0e-12), 0.7).clamp(min=1e-6)
    else:
        sigma = torch.tensor(1.0, dtype=dtype, device=device)
    scores = torch.exp(-residual / (sigma * sigma))

    dot = (Xi * Xj).sum(dim=2).clamp(-1.0, 1.0)
    opp = 0.5 * (1.0 - dot)
    conf_edge = 1.0 - torch.prod(1.0 - torch.sigmoid(beta_edge * opp), dim=1)

    return edge_idx, edge_tris, pair_ids, scores, conf_edge


@torch.no_grad()
def edge_pair_mirror_scores_X_adaptive(
    vertices: Tensor,
    faces: Tensor,
    X_face: Tensor,
    gradmag_face: Tensor,
    S_face: Tensor,
    *,
    K_min: int = 2,
    K_max: int = 5,
    beta_face: float = 8.0,
    beta_edge: float = 12.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Adaptive hinge-invariant mirror scores on per-seed unit directions."""

    device = S_face.device
    dtype = S_face.dtype

    edge_idx, edge_tris = _build_edges_and_adjacency(faces)
    valid = (edge_tris[:, 0] >= 0) & (edge_tris[:, 1] >= 0)
    edge_idx = edge_idx[valid]
    edge_tris = edge_tris[valid]

    if edge_tris.numel() == 0:
        empty_pairs = torch.zeros(0, 2, dtype=torch.long, device=device)
        empty_scores = torch.zeros(0, dtype=dtype, device=device)
        empty_conf = torch.zeros(0, dtype=dtype, device=device)
        return edge_idx, edge_tris, empty_pairs, empty_scores, empty_conf

    tL = edge_tris[:, 0]
    tR = edge_tris[:, 1]
    E = edge_idx.shape[0]
    _, C = S_face.shape

    if C == 0:
        empty_pairs = torch.zeros(E, 2, dtype=torch.long, device=device)
        zeros = torch.zeros(E, dtype=dtype, device=device)
        return edge_idx, edge_tris, empty_pairs, zeros, zeros

    K_cap = min(K_max, C)
    _, top_idx = torch.topk(S_face, k=K_cap, dim=1, largest=False)
    K = top_idx.shape[1]

    p_face = torch.softmax(-beta_face * S_face, dim=1)
    entropy = -(p_face * p_face.clamp_min(1e-12).log()).sum(dim=1)
    entropy = entropy / torch.log(torch.tensor(float(max(C, 2)), device=device))
    Kf = (K_min + torch.round(entropy * (K_cap - K_min))).to(torch.long)
    Kf = Kf.clamp_(K_min, K_cap)

    idxL = top_idx[tL]
    idxR = top_idx[tR]

    XL = X_face[tL].gather(1, idxL[..., None].expand(-1, -1, 3))
    XR = X_face[tR].gather(1, idxR[..., None].expand(-1, -1, 3))
    ML = gradmag_face[tL].gather(1, idxL)
    MR = gradmag_face[tR].gather(1, idxR)

    v0L = vertices[faces[tL, 0]]
    v1L = vertices[faces[tL, 1]]
    v2L = vertices[faces[tL, 2]]
    v0R = vertices[faces[tR, 0]]
    v1R = vertices[faces[tR, 1]]
    v2R = vertices[faces[tR, 2]]
    nL = _safe_normalize(torch.cross(v1L - v0L, v2L - v0L, dim=1), dim=1)
    nR = _safe_normalize(torch.cross(v1R - v0R, v2R - v0R, dim=1), dim=1)

    XR_cf = XR.permute(1, 0, 2)
    XR_rot_cf = transport_right_to_left(
        XR_cf,
        vertices=vertices,
        edge_vertices=edge_idx,
        nL=nL,
        nR=nR,
    )
    XR = XR_rot_cf.permute(1, 0, 2)

    e_vec = vertices[edge_idx[:, 1]] - vertices[edge_idx[:, 0]]
    edge_dir = _safe_normalize(e_vec, dim=1)
    n_in = torch.cross(edge_dir, nL, dim=1)

    rangeK = torch.arange(K, device=device)
    mL = rangeK[None, :] < Kf[tL][:, None]
    mR = rangeK[None, :] < Kf[tR][:, None]

    gLe = (XL * edge_dir[:, None, :]).sum(dim=2)
    gLn = (XL * n_in[:, None, :]).sum(dim=2)
    gRe = (XR * edge_dir[:, None, :]).sum(dim=2)
    gRn = (XR * n_in[:, None, :]).sum(dim=2)

    residual_samples: List[Tensor] = []
    for i in range(K):
        mask_i = mL[:, i]
        if not mask_i.any():
            continue
        for j in range(K):
            mask = mask_i & mR[:, j]
            if not mask.any():
                continue
            same = idxL[:, i] == idxR[:, j]
            mask = mask & (~same)
            if not mask.any():
                continue
            residual = (gLe[:, i] - gRe[:, j]) ** 2 + (gLn[:, i] + gRn[:, j]) ** 2
            residual_samples.append(residual[mask].detach())

    if residual_samples:
        all_res = torch.cat(residual_samples)
        sigma = _robust_quantile(torch.sqrt(all_res.clamp_min(1e-12)), 0.7).clamp(min=1e-6)
    else:
        sigma = torch.tensor(1.0, device=device, dtype=dtype)

    best_val = torch.full((E,), float("-inf"), dtype=dtype, device=device)
    best_i = torch.zeros(E, dtype=torch.long, device=device)
    best_j = torch.zeros(E, dtype=torch.long, device=device)
    conf_prod = torch.ones(E, dtype=dtype, device=device)

    for i in range(K):
        mask_i = mL[:, i]
        if not mask_i.any():
            continue
        for j in range(K):
            mask = mask_i & mR[:, j]
            if not mask.any():
                continue
            same = idxL[:, i] == idxR[:, j]
            mask = mask & (~same)
            if not mask.any():
                continue
            residual = (gLe[:, i] - gRe[:, j]) ** 2 + (gLn[:, i] + gRn[:, j]) ** 2
            mag = torch.minimum(ML[:, i], MR[:, j])
            score = torch.exp(-residual / (sigma * sigma)) * mag
            better = score > best_val
            best_val = torch.where(better, score, best_val)
            best_i = torch.where(better, idxL[:, i], best_i)
            best_j = torch.where(better, idxR[:, j], best_j)
            cos = (XL[:, i, :] * XR[:, j, :]).sum(dim=1).clamp(-1.0, 1.0)
            opp = 0.5 * (1.0 - cos)
            sig = torch.zeros_like(conf_prod)
            sig[mask] = torch.sigmoid(beta_edge * opp[mask])
            conf_prod = conf_prod * (1.0 - sig)

    no_pair = ~torch.isfinite(best_val)
    best_val = torch.where(no_pair, torch.zeros_like(best_val), best_val)
    conf = torch.where(no_pair, torch.zeros_like(conf_prod), 1.0 - conf_prod)
    best_pairs = torch.stack([best_i, best_j], dim=1)
    best_pairs[no_pair] = 0

    return edge_idx, edge_tris, best_pairs, best_val, conf
