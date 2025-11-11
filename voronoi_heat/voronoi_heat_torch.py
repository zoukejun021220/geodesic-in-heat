"""
Differentiable multi-seed heat-based Voronoi construction with PyTorch.

This module implements the blueprint from ``instruction2.md`` including:
  * Assembly of cotangent Laplacian and lumped mass matrices.
  * Implicit heat solve per seed via conjugate gradient.
  * Seam-localised loss enforcing opposing gradients across seams.
  * Optional trainable soft heat sources and inference helpers.

The implementation only depends on ``torch`` and ``numpy``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .certify import certify_segment_endpoints
from .grad_alignment import edge_pair_mirror_scores
from .mps_linear import AMatrix, cg_solve_mv, laplacian_cotan_coo_fp32, spmv_coo_multi


__all__ = [
    "VoronoiHeatModel",
    "TrainableSources",
    "train_step",
    "infer_labels_and_segments",
    "heat_only_face_loss",
    "heat_only_face_loss_vec",
    "heat_method_distances",
]

EPS = 1.0e-12
EDGE_INDEX_TEMPLATE = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)


def _dtype_for_device(device: torch.device | str) -> torch.dtype:
    dev = torch.device(device)
    return torch.float64 if dev.type == "cpu" else torch.float32


def _safe_sparse_mm(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Multiply supporting CSR fallback to COO when required."""

    try:
        return torch.sparse.mm(matrix, rhs)
    except RuntimeError as exc:
        if getattr(matrix, "layout", None) == torch.sparse_csr:
            return torch.sparse.mm(matrix.to_sparse_coo(), rhs)
        raise exc


# -----------------------------------------------------------------------------
# Small tensor utilities
# -----------------------------------------------------------------------------
def safe_norm(x: Tensor, *, dim: int = -1, keepdim: bool = True, eps: float = EPS) -> Tensor:
    """Return ||x|| with a lower bound to avoid division by zero."""

    return torch.clamp(torch.linalg.norm(x, dim=dim, keepdim=keepdim), min=eps)


def safe_normalize(x: Tensor, *, dim: int = -1, eps: float = EPS) -> Tensor:
    """Return x / ||x|| with safe denominator."""

    return x / safe_norm(x, dim=dim, keepdim=True, eps=eps)


def mean_edge_length(vertices: Tensor, faces: Tensor) -> float:
    """Mean length of unique edges in the mesh."""

    verts = vertices.detach()
    tris = faces.detach()
    edges: set[Tuple[int, int]] = set()
    for tri in tris.tolist():
        i, j, k = tri
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))
    if not edges:
        return 0.0

    lengths = []
    for i, j in edges:
        lengths.append((verts[i] - verts[j]).norm().item())
    return float(np.mean(lengths))


# -----------------------------------------------------------------------------
# Geometry & finite element operators
# -----------------------------------------------------------------------------
def build_face_geometry(V: Tensor, F: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return per-face area, unit normal, and hat function gradients."""

    vi = V[F[:, 0]]
    vj = V[F[:, 1]]
    vk = V[F[:, 2]]

    n = torch.cross(vj - vi, vk - vi, dim=1)
    area = 0.5 * torch.clamp(torch.linalg.norm(n, dim=1), min=EPS)
    n_hat = n / (2.0 * area)[:, None]

    ei = vk - vj
    ej = vi - vk
    ek = vj - vi

    denom = 2.0 * area[:, None]
    gI = torch.cross(n_hat, ei, dim=1) / denom
    gJ = torch.cross(n_hat, ej, dim=1) / denom
    gK = torch.cross(n_hat, ek, dim=1) / denom
    return area, n_hat, gI, gJ, gK


def build_mass_lumped_diag(V: Tensor, F: Tensor, A_f: Tensor, *, device: Optional[torch.device] = None) -> Tensor:
    """Assemble lumped vertex mass diagonal suitable for dense operations."""

    n_vertices = V.shape[0]
    device = device or V.device
    masses = torch.zeros(n_vertices, dtype=V.dtype, device=device)
    for corner in range(3):
        masses.index_add_(0, F[:, corner], A_f / 3.0)
    return masses


def build_cotan_laplacian_coo(V: Tensor, F: Tensor) -> Tuple[Tensor, Tensor, Tensor, int]:
    """Return cotangent Laplacian COO triplets without creating sparse tensors."""

    row, col, val, n = laplacian_cotan_coo_fp32(V, F)
    return row, col, val.to(V.dtype), n


def heat_solve_multi(
    M_diag: Tensor,
    row: Tensor,
    col: Tensor,
    val: Tensor,
    B: Tensor,
    *,
    t: float,
    tol: float = 1.0e-6,
    iters: int = 500,
    L_csr: Tensor | None = None,
    L_diag: Tensor | None = None,
) -> Tensor:
    """Solve (M + t L) U = B for one or more right-hand sides."""

    if B.ndim != 2:
        raise ValueError("B must be a 2D tensor of shape (nV, C).")
    device_type = M_diag.device.type
    if device_type != "mps":
        n = M_diag.shape[0]
        if L_csr is None:
            indices = torch.stack([row, col], dim=0)
            L_local = (
                torch.sparse_coo_tensor(indices, val.to(B.dtype), (n, n), device=M_diag.device)
                .coalesce()
                .to_sparse_csr()
            )
        else:
            L_local = L_csr.to(dtype=B.dtype)

        if L_diag is not None:
            A_diag = (M_diag + t * L_diag.to(M_diag.dtype)).to(B.dtype)
        else:
            diag_mask = row == col
            L_diag_local = torch.zeros_like(M_diag, dtype=M_diag.dtype)
            if diag_mask.any():
                L_diag_local.index_add_(0, row[diag_mask], val[diag_mask].to(M_diag.dtype))
            A_diag = (M_diag + t * L_diag_local).to(B.dtype)

        mass = M_diag.to(B.dtype)

        def matvec(X: Tensor) -> Tensor:
            return mass[:, None] * X + t * _safe_sparse_mm(L_local, X)

        return _cg_matrix_multi(matvec, B, A_diag, tol=tol, maxiter=iters)

    # MPS / fallback path using COO SpMV
    diag_mask = row == col
    L_diag_local = torch.zeros_like(M_diag, dtype=M_diag.dtype)
    if diag_mask.any():
        L_diag_local.index_add_(0, row[diag_mask], val[diag_mask].to(M_diag.dtype))
    A_diag = (M_diag + t * L_diag_local).to(B.dtype)

    mass = M_diag.to(B.dtype)

    def matvec(X: Tensor) -> Tensor:
        return mass[:, None] * X + t * spmv_coo_multi(row, col, val.to(B.dtype), X, M_diag.shape[0])

    return _cg_matrix_multi(matvec, B, A_diag, tol=tol, maxiter=iters)


def _cg_matrix_multi(matvec, B: Tensor, A_diag: Tensor, *, tol: float, maxiter: int) -> Tensor:
    """Jacobi-preconditioned CG for multi-RHS systems with per-column early stop."""

    X = torch.zeros_like(B)
    R = B - matvec(X)
    M_inv = (1.0 / A_diag.clamp_min(1.0e-12)).to(B.dtype)
    Z = M_inv[:, None] * R
    P = Z.clone()
    rz_old = (R * Z).sum(dim=0)
    active = torch.ones(B.shape[1], dtype=torch.bool, device=B.device)

    for _ in range(maxiter):
        if not active.any():
            break
        AP = matvec(P)
        denom = (P * AP).sum(dim=0)
        alpha = torch.where(
            active,
            rz_old / denom.clamp_min(1e-30),
            torch.zeros_like(rz_old),
        )
        alpha_exp = alpha.unsqueeze(0)
        X = X + P * alpha_exp
        R = R - AP * alpha_exp
        res = R.norm(dim=0)
        active = res > tol
        if not active.any():
            break
        Z = M_inv[:, None] * R
        rz_new = (R * Z).sum(dim=0)
        beta = torch.where(
            active,
            rz_new / rz_old.clamp_min(1e-30),
            torch.zeros_like(rz_old),
        )
        P = (Z + P * beta.unsqueeze(0)) * active.unsqueeze(0).to(B.dtype)
        rz_old = rz_new

    return X


def face_gradients(U: Tensor, F: Tensor, gI: Tensor, gJ: Tensor, gK: Tensor) -> Tensor:
    """Return per-face gradients of scalar fields (one column per seed)."""

    uI = U[F[:, 0]]  # (nF, C)
    uJ = U[F[:, 1]]
    uK = U[F[:, 2]]
    grad = (
        uI.unsqueeze(2) * gI.unsqueeze(1)
        + uJ.unsqueeze(2) * gJ.unsqueeze(1)
        + uK.unsqueeze(2) * gK.unsqueeze(1)
    )
    return grad


def unit_directions_from_grad(grad: Tensor, *, eps: float = EPS) -> Tensor:
    """Return -∇u / ||∇u|| for each face and seed."""

    norms = safe_norm(grad, dim=2, eps=eps)
    return -grad / norms


def scores_from_u(U: Tensor, *, eps: float = 1.0e-9) -> Tensor:
    """Convert heat potentials to scores via -log(u) with NaN/Inf guards."""
    U_safe = torch.nan_to_num(U, nan=eps, posinf=1.0, neginf=0.0)
    return -torch.log(torch.clamp(U_safe, min=eps))


def extract_segments_in_face_hard(
    V: Tensor,
    Ff: Tensor,
    S_all: Tensor,
    labels_all: Tensor,
    *,
    face_index: int,
    faces_all: Tensor,
    s_per_seed: Tensor | None = None,
    enable_certify: bool = True,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Return bisector segment within a face (two endpoints + barycentrics) if present."""

    face_idx = Ff.tolist()
    edges = ((0, 1), (1, 2), (2, 0))
    points: List[Tensor] = []
    barys: List[Tensor] = []
    host_edges: List[Tuple[int, int]] = []
    lambdas: List[float] = []
    pairs: List[Tuple[int, int]] = []
    for a, b in edges:
        va = face_idx[a]
        vb = face_idx[b]
        la = labels_all[va].item()
        lb = labels_all[vb].item()
        if la == lb:
            continue
        Da = S_all[va, la] - S_all[va, lb]
        Db = S_all[vb, la] - S_all[vb, lb]
        if (Da * Db).item() > 0:
            continue
        denom = Da - Db
        eps_tensor = denom.new_tensor(EPS)
        denom_safe = torch.where(denom.abs() > eps_tensor, denom, torch.ones_like(denom))
        lam_raw = (Da / denom_safe).clamp(0.0, 1.0)
        lam_tensor = torch.where(denom.abs() > eps_tensor, lam_raw, denom.new_tensor(0.5))
        p = (1.0 - lam_tensor) * V[va] + lam_tensor * V[vb]
        bary = torch.zeros(3, dtype=V.dtype, device=V.device)
        bary[a] = 1.0 - lam_tensor
        bary[b] = lam_tensor
        points.append(p)
        barys.append(bary)
        host_edges.append((int(va), int(vb)))
        lambdas.append(float(lam_tensor.detach().cpu().item()))
        pairs.append(tuple(sorted((int(la), int(lb)))))
    if len(points) == 2:
        if pairs[0] != pairs[1]:
            return None
        if enable_certify and s_per_seed is not None:
            pair = pairs[0]
            segment_ok = certify_segment_endpoints(
                faces_all,
                pair,
                face_index,
                (host_edges[0], host_edges[1]),
                (lambdas[0], lambdas[1]),
                s_per_seed,
            )
            if not segment_ok:
                return None
        return points[0], points[1], barys[0], barys[1]
    return None


# --- NEW: segment extraction for a specific pair (i,j) with certification ---
def extract_segment_in_face_pair(
    V: Tensor,
    Ff: Tensor,
    S_all: Tensor,
    pair: Tuple[int, int],
    *,
    face_index: int,
    faces_all: Tensor,
    s_per_seed: Tensor,
    enable_certify: bool = True,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    i, j = int(pair[0]), int(pair[1])
    verts = list(map(int, Ff.tolist()))
    deltas = [float(S_all[v, i].item() - S_all[v, j].item()) for v in verts]

    points: List[Tensor] = []
    barys: List[Tensor] = []
    host_edges: List[Tuple[int, int]] = []
    lambdas: List[float] = []
    edges = ((0, 1), (1, 2), (2, 0))

    for a, b in edges:
        Da = deltas[a]
        Db = deltas[b]
        if Da * Db > 0.0:
            continue
        denom = Da - Db
        lam = 0.5 if abs(denom) < 1e-12 else float(Da / denom)
        lam = max(0.0, min(1.0, lam))
        pa = V[verts[a]]
        pb = V[verts[b]]
        point = (1.0 - lam) * pa + lam * pb
        bary = torch.zeros(3, dtype=V.dtype, device=V.device)
        bary[a] = 1.0 - lam
        bary[b] = lam
        points.append(point)
        barys.append(bary)
        host_edges.append((verts[a], verts[b]))
        lambdas.append(lam)

    if len(points) == 2:
        if enable_certify:
            ok = certify_segment_endpoints(
                faces_all,
                (i, j),
                face_index,
                (host_edges[0], host_edges[1]),
                (lambdas[0], lambdas[1]),
                s_per_seed,
            )
            if not ok:
                return None
        return points[0], points[1], barys[0], barys[1]
    return None


# --- Edge utility helpers ----------------------------------------------------
def _build_faces_to_edges(num_faces: int, edge_tris: Tensor) -> List[List[int]]:
    faces_to_edges: List[List[int]] = [[] for _ in range(num_faces)]
    for e, (l, r) in enumerate(edge_tris.tolist()):
        if l >= 0:
            faces_to_edges[l].append(e)
        if r >= 0:
            faces_to_edges[r].append(e)
    return faces_to_edges


def _smooth_edge_scalar(
    values: Tensor,
    faces_to_edges: Sequence[Sequence[int]],
    *,
    iterations: int = 1,
    alpha: float = 0.5,
) -> Tensor:
    if values.numel() == 0 or iterations <= 0:
        return values
    val = values
    for _ in range(iterations):
        accum = torch.zeros_like(val)
        counts = torch.zeros_like(val)
        for edges in faces_to_edges:
            if len(edges) < 2:
                continue
            for e in edges:
                for nbr in edges:
                    if nbr == e:
                        continue
                    accum[e] += val[nbr]
                    counts[e] += 1
        avg = torch.where(counts > 0, accum / counts, val)
        val = alpha * val + (1.0 - alpha) * avg
    return val


def _edge_dual_neighbors(edge_tris: Tensor) -> List[List[int]]:
    """Return edge-dual adjacency: edges sharing a triangle become neighbors."""

    E = edge_tris.shape[0]
    tri2edges: Dict[int, List[int]] = {}
    for e in range(E):
        left = int(edge_tris[e, 0].item())
        right = int(edge_tris[e, 1].item())
        for tri in (left, right):
            if tri < 0:
                continue
            tri2edges.setdefault(tri, []).append(e)
    neighbors: List[List[int]] = [[] for _ in range(E)]
    for edges in tri2edges.values():
        for e in edges:
            neigh = neighbors[e]
            for other in edges:
                if other == e:
                    continue
                neigh.append(other)
    return [sorted(set(ns)) for ns in neighbors]


def _smooth_edge_pair_scores(scores: Tensor, edge_tris: Tensor, iters: int = 2, lam: float = 0.5) -> Tensor:
    """Jacobi smoothing of per-edge pair scores on the edge-dual graph."""

    if scores.numel() == 0 or iters <= 0:
        return scores
    adj = _edge_dual_neighbors(edge_tris)
    out = scores
    for _ in range(iters):
        updated = out.clone()
        for e, nbrs in enumerate(adj):
            if not nbrs:
                continue
            mean_nbr = out[nbrs].mean(dim=0)
            updated[e] = (1.0 - lam) * out[e] + lam * mean_nbr
        out = updated
    return out


def _stitch_polylines(segments: List[Tuple[np.ndarray, np.ndarray]], tol: float = 1e-6) -> List[np.ndarray]:
    if not segments:
        return []
    node_map: Dict[Tuple[int, int, int], int] = {}
    nodes: List[np.ndarray] = []
    adjacency: Dict[int, List[int]] = {}

    def node_id(p: np.ndarray) -> int:
        key = tuple(np.round(p / tol).astype(int))
        idx = node_map.get(key)
        if idx is None:
            idx = len(nodes)
            node_map[key] = idx
            nodes.append(p)
            adjacency[idx] = []
        return idx

    edges: List[Tuple[int, int]] = []
    for p0, p1 in segments:
        if np.allclose(p0, p1, atol=tol):
            continue
        i = node_id(p0)
        j = node_id(p1)
        if i == j:
            continue
        adjacency[i].append(j)
        adjacency[j].append(i)
        edges.append((i, j))

    visited: set[Tuple[int, int]] = set()
    polylines: List[np.ndarray] = []

    def append_poly(path_nodes: List[int]) -> None:
        if len(path_nodes) < 2:
            return
        coords = np.stack([nodes[n] for n in path_nodes], axis=0)
        length = np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()
        if length > tol:
            polylines.append(coords)

    # trace paths starting at nodes with degree != 2
    for start, neighbors in adjacency.items():
        if len(neighbors) != 2:
            for nxt in neighbors:
                if (start, nxt) in visited:
                    continue
                path = [start, nxt]
                visited.add((start, nxt))
                visited.add((nxt, start))
                prev, curr = start, nxt
                while len(adjacency[curr]) == 2:
                    nxt_candidates = adjacency[curr]
                    nxt = nxt_candidates[0] if nxt_candidates[0] != prev else nxt_candidates[1]
                    if (curr, nxt) in visited:
                        break
                    path.append(nxt)
                    visited.add((curr, nxt))
                    visited.add((nxt, curr))
                    prev, curr = curr, nxt
                append_poly(path)

    # handle remaining cycles
    for i, j in edges:
        if (i, j) in visited:
            continue
        path = [i, j]
        visited.add((i, j))
        visited.add((j, i))
        prev, curr = i, j
        while True:
            neighbors = [n for n in adjacency[curr] if n != prev]
            nxt = None
            for candidate in neighbors:
                if (curr, candidate) not in visited:
                    nxt = candidate
                    break
            if nxt is None:
                break
            path.append(nxt)
            visited.add((curr, nxt))
            visited.add((nxt, curr))
            prev, curr = curr, nxt
            if curr == i:
                break
        append_poly(path)

    return polylines

# -----------------------------------------------------------------------------
# Seam detection helpers
# -----------------------------------------------------------------------------
def _unordered_pair_indices(count: int, *, device: torch.device) -> Tensor:
    """Return ``(P, 2)`` tensor of unordered indices for range(count)."""

    if count < 2:
        return torch.empty(0, 2, dtype=torch.long, device=device)
    base = torch.arange(count, device=device, dtype=torch.long)
    # ``torch.combinations`` gives all unordered pairs without replacement.
    return torch.combinations(base, r=2)


def topk_pairs_per_face(
    S_face: Tensor,
    K: int = 3,
    *,
    K_min: int = 2,
    beta_face: float = 8.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return entropy-adaptive candidate indices, pair tensor, and validity mask."""

    nF, C = S_face.shape
    if C == 0:
        empty_idx = torch.empty(nF, 0, dtype=torch.long, device=S_face.device)
        empty_pairs = torch.empty(nF, 0, 2, dtype=torch.long, device=S_face.device)
        empty_mask = torch.empty(nF, 0, dtype=torch.bool, device=S_face.device)
        return empty_idx, empty_pairs, empty_mask

    K_cap = min(K, C)
    _, idx = torch.topk(S_face, k=K_cap, dim=1, largest=False)

    if C <= 1:
        empty_pairs = torch.empty(nF, 0, 2, dtype=torch.long, device=S_face.device)
        empty_mask = torch.empty(nF, 0, dtype=torch.bool, device=S_face.device)
        return idx, empty_pairs, empty_mask

    p_face = torch.softmax(-beta_face * S_face, dim=1)
    entropy = -(p_face * p_face.clamp_min(1e-12).log()).sum(dim=1)
    entropy = entropy / torch.log(torch.tensor(float(max(C, 2)), device=S_face.device))
    Kf = (K_min + torch.round(entropy * (K_cap - K_min))).to(torch.long)
    Kf = Kf.clamp_(K_min, K_cap)

    comb = _unordered_pair_indices(K_cap, device=idx.device)
    if comb.numel() == 0:
        pair_tensor = torch.empty(nF, 0, 2, dtype=torch.long, device=idx.device)
        pair_mask = torch.empty(nF, 0, dtype=torch.bool, device=idx.device)
    else:
        pair_tensor = idx[:, comb]  # (nF, P, 2)
        p0 = comb[None, :, 0]
        p1 = comb[None, :, 1]
        valid0 = p0 < Kf[:, None]
        valid1 = p1 < Kf[:, None]
        pair_mask = valid0 & valid1
    return idx, pair_tensor, pair_mask


def dominant_pairs_per_face(
    face_probs: Tensor,
    K: int = 3,
    *,
    K_min: int = 2,
) -> Tuple[Tensor, Tensor]:
    """Return unordered channel pairs ranked by dominant probability mass per face."""

    nF, C = face_probs.shape
    if C < 2:
        empty_pairs = torch.empty(nF, 0, 2, dtype=torch.long, device=face_probs.device)
        empty_mask = torch.empty(nF, 0, dtype=torch.bool, device=face_probs.device)
        return empty_pairs, empty_mask

    K_cap = min(K, C)
    if K_cap <= 0:
        empty_pairs = torch.empty(nF, 0, 2, dtype=torch.long, device=face_probs.device)
        empty_mask = torch.empty(nF, 0, dtype=torch.bool, device=face_probs.device)
        return empty_pairs, empty_mask

    _, idx = torch.topk(face_probs, k=K_cap, dim=1, largest=True)

    comb = _unordered_pair_indices(K_cap, device=face_probs.device)
    if comb.numel() == 0:
        pair_tensor = torch.empty(nF, 0, 2, dtype=torch.long, device=face_probs.device)
        pair_mask = torch.empty(nF, 0, dtype=torch.bool, device=face_probs.device)
        return pair_tensor, pair_mask

    pair_tensor = idx[:, comb]

    probs_clamped = face_probs.clamp_min(1.0e-12)
    entropy = -(probs_clamped * probs_clamped.log()).sum(dim=1)
    denom = torch.log(torch.tensor(float(max(C, 2)), dtype=face_probs.dtype, device=face_probs.device))
    entropy = entropy / denom
    Kf = (K_min + torch.round(entropy * (K_cap - K_min))).to(torch.long)
    Kf = Kf.clamp_(K_min, K_cap)

    p0 = comb[None, :, 0]
    p1 = comb[None, :, 1]
    valid0 = p0 < Kf[:, None]
    valid1 = p1 < Kf[:, None]
    pair_mask = valid0 & valid1
    return pair_tensor, pair_mask


def soft_edge_crossings_in_face(
    V: Tensor,
    Ff: Tensor,
    S_vert: Tensor,
    pair: Tensor,
    *,
    kappa: float = 20.0,
    eps: float = 1.0e-8,
) -> Tuple[Tensor, Tensor]:
    """Softly locate edge crossings for a seed pair within a face."""

    i, j = int(pair[0].item()), int(pair[1].item())
    delta = S_vert[:, i] - S_vert[:, j]
    edges = ((0, 1), (1, 2), (2, 0))
    points = []
    probs = []

    for a, b in edges:
        Da = delta[a]
        Db = delta[b]
        prod = Da * Db
        pij = torch.sigmoid(-kappa * prod)
        denom = Da - Db
        sign = torch.where(denom >= 0, denom.new_tensor(1.0), denom.new_tensor(-1.0))
        safe_denom = torch.where(torch.abs(denom) >= eps, denom, sign * eps)
        lam = (Da / safe_denom).clamp(0.0, 1.0)
        pa = V[Ff[a]]
        pb = V[Ff[b]]
        points.append((1.0 - lam) * pa + lam * pb)
        probs.append(pij)

    return torch.stack(points, dim=0), torch.stack(probs, dim=0)


def soft_segment_from_edges(
    p_edge_allpairs: List[Tensor],
    pi_allpairs: List[Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """Aggregate soft crossings across edge pairs into a single seam segment."""

    if not p_edge_allpairs:
        return (
            torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device),
            torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device),
            torch.tensor(0.0, dtype=dtype, device=device),
        )

    weights = []
    segments = []
    for p_edge, pi in zip(p_edge_allpairs, pi_allpairs):
        w01 = (pi[0] * pi[1]).unsqueeze(0)
        w12 = (pi[1] * pi[2]).unsqueeze(0)
        w20 = (pi[2] * pi[0]).unsqueeze(0)
        weights.append(torch.cat([w01, w12, w20], dim=0))
        v01 = p_edge[1] - p_edge[0]
        v12 = p_edge[2] - p_edge[1]
        v20 = p_edge[0] - p_edge[2]
        segments.append(torch.stack([v01, v12, v20], dim=0))

    W = torch.stack(weights, dim=0).sum(dim=0)
    Vseg = torch.stack(segments, dim=0).sum(dim=0)
    v = (W[0] * Vseg[0] + W[1] * Vseg[1] + W[2] * Vseg[2]) / torch.clamp(W.sum(), min=EPS)
    tau_hat = safe_normalize(v.to(dtype=dtype), dim=0).squeeze(0)
    rho = torch.clamp(W.sum() / 3.0, max=1.0)
    return tau_hat, None, rho


def heat_only_face_loss(
    grad_face: Tensor,
    S_face: Tensor,
    V: Tensor,
    F: Tensor,
    n_hat: Tensor,
    *,
    kappa: float = 20.0,
    beta_pairs: float = 10.0,
    K: int = 3,
    S_full: Optional[Tensor] = None,
    w_jump: float = 1.0,
    w_normal: float = 0.5,
    w_tan: float = 0.25,
    ignore_face_mask: Optional[Tensor] = None,
    U_full: Optional[Tensor] = None,
    use_soft_flip: bool = False,
    softflip_beta: float = 12.0,
    softflip_margin: float = 0.15,
    softflip_eta: float = 12.0,
    softflip_alpha: float = 1.0,
    softflip_kappa: Optional[float] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Mirror/normal/tangent penalties enforcing opposing gradients across seams."""

    device = V.device
    dtype = V.dtype
    nF = F.shape[0]

    def _zero() -> Tuple[Tensor, Dict[str, Tensor]]:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {"jump": zero, "normal": zero, "tan": zero}

    if S_face.numel() > 0:
        C = S_face.shape[1]
    elif S_full is not None and S_full.numel() > 0:
        C = S_full.shape[1]
    elif U_full is not None and U_full.numel() > 0:
        C = U_full.shape[1]
    else:
        C = 0

    if C < 2:
        return _zero()

    edges_idx = EDGE_INDEX_TEMPLATE.to(device)

    if S_full is not None:
        scores_vert = S_full[F]
    else:
        scores_vert = S_face[:, None, :].expand(-1, 3, -1)

    prob_full: Optional[Tensor] = None
    prob_face: Optional[Tensor] = None
    if use_soft_flip:
        if U_full is not None:
            probs_raw = torch.clamp(U_full, min=1.0e-12)
            probs_pow = torch.pow(probs_raw, softflip_beta)
            prob_full = probs_pow / probs_pow.sum(dim=1, keepdim=True).clamp_min(EPS)
        elif S_full is not None:
            prob_full = torch.softmax(-softflip_beta * S_full, dim=1)
        else:
            raise ValueError("use_soft_flip requires either S_full or U_full inputs.")
        prob_face = prob_full[F]

    if use_soft_flip and prob_face is not None:
        face_probs = prob_face.mean(dim=1)
        _, pairs, pair_mask = topk_pairs_per_face(face_probs, K=K)
    else:
        _, pairs, pair_mask = topk_pairs_per_face(S_face, K=K)

    P = pairs.shape[1]
    if P == 0:
        return _zero()

    pair_i = pairs[:, :, 0]
    pair_j = pairs[:, :, 1]
    gather_i = pair_i[:, None, :].expand(-1, 3, -1)
    gather_j = pair_j[:, None, :].expand(-1, 3, -1)

    S_vert = scores_vert
    S_vert_i = torch.gather(S_vert, 2, gather_i)
    S_vert_j = torch.gather(S_vert, 2, gather_j)
    Delta = S_vert_i - S_vert_j

    Da = Delta[:, edges_idx[:, 0], :]
    Db = Delta[:, edges_idx[:, 1], :]
    prod = Da * Db
    kappa_eff = softflip_kappa if (use_soft_flip and softflip_kappa is not None) else kappa
    pi = torch.sigmoid(-kappa_eff * prod)

    edge_gate_scalar: Optional[Tensor] = None
    pair_score_dom: Optional[Tensor] = None
    if use_soft_flip and prob_face is not None:
        prob_i = torch.gather(prob_face, 2, gather_i)
        prob_j = torch.gather(prob_face, 2, gather_j)

        Pa_i = prob_i[:, edges_idx[:, 0], :]
        Pa_j = prob_j[:, edges_idx[:, 0], :]
        Pb_i = prob_i[:, edges_idx[:, 1], :]
        Pb_j = prob_j[:, edges_idx[:, 1], :]
        pair_edge_prob = Pa_i * Pb_j + Pa_j * Pb_i  # (nF, 3, P)

        edge_prob_a = prob_face[:, edges_idx[:, 0], :]
        edge_prob_b = prob_face[:, edges_idx[:, 1], :]
        phi_edge = 1.0 - (edge_prob_a * edge_prob_b).sum(dim=2)

        if prob_full is not None:
            topk = torch.topk(prob_full, k=min(2, C), dim=1, largest=True).values
            if topk.shape[1] == 1:
                margin_full = topk[:, 0]
            else:
                margin_full = topk[:, 0] - topk[:, 1]
        else:
            margin_full = torch.ones(V.shape[0], dtype=dtype, device=device)

        margin_face = margin_full[F]
        edge_margin = torch.minimum(
            margin_face[:, edges_idx[:, 0]],
            margin_face[:, edges_idx[:, 1]],
        )
        reliability = torch.sigmoid(softflip_eta * (edge_margin - softflip_margin))
        edge_gate_scalar = phi_edge * reliability
        pi = pi * edge_gate_scalar.unsqueeze(-1)

        gate = edge_gate_scalar.unsqueeze(-1)
        denom = edge_gate_scalar.sum(dim=1, keepdim=True).clamp_min(EPS)
        pair_score_dom = (pair_edge_prob * gate).sum(dim=1) / denom

    denom = Da - Db
    sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
    safe_denom = torch.where(denom.abs() >= EPS, denom, sign * EPS)
    lam = torch.clamp(Da / safe_denom, 0.0, 1.0)

    V_face = V[F]
    Va = V_face[:, edges_idx[:, 0], :].unsqueeze(2)
    Vb = V_face[:, edges_idx[:, 1], :].unsqueeze(2)
    lam_exp = lam.unsqueeze(-1)
    p_edge = (1.0 - lam_exp) * Va + lam_exp * Vb

    w_edges = torch.stack(
        [
            pi[:, 0, :] * pi[:, 1, :],
            pi[:, 1, :] * pi[:, 2, :],
            pi[:, 2, :] * pi[:, 0, :],
        ],
        dim=1,
    )

    seg_vecs = torch.stack(
        [
            p_edge[:, 1, :, :] - p_edge[:, 0, :, :],
            p_edge[:, 2, :, :] - p_edge[:, 1, :, :],
            p_edge[:, 0, :, :] - p_edge[:, 2, :, :],
        ],
        dim=1,
    )

    W_sum = w_edges.sum(dim=2)
    V_sum = (w_edges.unsqueeze(-1) * seg_vecs).sum(dim=2)

    total_w = W_sum.sum(dim=1, keepdim=True)
    v = (
        W_sum[:, 0:1] * V_sum[:, 0, :]
        + W_sum[:, 1:2] * V_sum[:, 1, :]
        + W_sum[:, 2:3] * V_sum[:, 2, :]
    ) / torch.clamp(total_w, min=EPS)

    tau_hat = safe_normalize(v, dim=1)
    rho = torch.clamp(total_w.squeeze(1) / 3.0, max=1.0)

    if not torch.any(rho > EPS):
        return _zero()

    n_in = safe_normalize(torch.cross(n_hat, tau_hat, dim=1), dim=1)

    grad_i = torch.gather(grad_face, 1, pair_i[:, :, None].expand(-1, -1, 3))
    grad_j = torch.gather(grad_face, 1, pair_j[:, :, None].expand(-1, -1, 3))

    tau_dot_i = (grad_i * tau_hat[:, None, :]).sum(dim=2)
    tau_dot_j = (grad_j * tau_hat[:, None, :]).sum(dim=2)
    nin_dot_i = (grad_i * n_in[:, None, :]).sum(dim=2)
    nin_dot_j = (grad_j * n_in[:, None, :]).sum(dim=2)

    # Tangent penalty should drive both tangential components to zero (orthogonality),
    # not merely make them equal. Use magnitude-based penalty instead of difference.
    tan_eq = (tau_dot_i ** 2 + tau_dot_j ** 2)
    norm_opp = (nin_dot_i + nin_dot_j) ** 2

    grad_diff = grad_i - grad_j
    jump_strength = grad_diff.norm(dim=2)

    prob_face_scores = torch.softmax(-softflip_beta * S_face, dim=1)
    tie_i = torch.gather(prob_face_scores, 1, pair_i)
    tie_j = torch.gather(prob_face_scores, 1, pair_j)
    tie_gate = (4.0 * tie_i * tie_j).clamp(0.0, 1.0)

    s_face_i = torch.gather(S_face, 1, pair_i)
    s_face_j = torch.gather(S_face, 1, pair_j)
    # Prefer equality along bisectors: weight pairs by small |Si - Sj|,
    # not by small (Si + Sj), which biases near seeds.
    logits_pairs = -beta_pairs * (s_face_i - s_face_j).abs()
    if pair_mask.numel() > 0:
        logits_pairs = logits_pairs.masked_fill(~pair_mask, -1.0e9)
    w_pairs_sumS = torch.softmax(logits_pairs, dim=1)
    if pair_mask.numel() > 0:
        w_pairs_sumS = w_pairs_sumS * pair_mask.to(w_pairs_sumS.dtype)

    if use_soft_flip and pair_score_dom is not None:
        dom_logits = beta_pairs * pair_score_dom
        if pair_mask.numel() > 0:
            dom_logits = dom_logits.masked_fill(~pair_mask, -1.0e9)
        w_pairs_dom = torch.softmax(dom_logits, dim=1)
        if pair_mask.numel() > 0:
            w_pairs_dom = w_pairs_dom * pair_mask.to(w_pairs_dom.dtype)
        alpha = float(max(0.0, min(1.0, softflip_alpha)))
        if alpha <= 0.0:
            w_pairs = w_pairs_sumS
        elif alpha >= 1.0:
            w_pairs = w_pairs_dom
        else:
            base = (w_pairs_dom.clamp_min(1.0e-9) ** alpha) * (
                w_pairs_sumS.clamp_min(1.0e-9) ** (1.0 - alpha)
            )
            w_pairs = base / base.sum(dim=1, keepdim=True).clamp_min(1.0e-9)
    else:
        w_pairs = w_pairs_sumS

    weights = rho[:, None] * w_pairs * tie_gate
    if ignore_face_mask is not None and ignore_face_mask.numel() == nF:
        mask_float = (~ignore_face_mask).to(weights.dtype)
        weights = weights * mask_float[:, None]
    if pair_mask.numel() > 0:
        weights = weights * pair_mask.to(weights.dtype)

    loss_norm = torch.sum(weights * norm_opp)
    loss_tan = torch.sum(weights * tan_eq)
    weight_sum = weights.sum().clamp_min(1.0e-9)
    jump_metric = torch.sum(weights * jump_strength) / weight_sum
    jump_beta = 0.1
    loss_jump = torch.sum(weights * torch.sigmoid(-jump_beta * jump_strength)) / weight_sum

    total = w_jump * loss_jump + w_normal * loss_norm + w_tan * loss_tan
    return total, {
        "jump": jump_metric.detach(),
        "normal": loss_norm.detach(),
        "tan": loss_tan.detach(),
    }


def heat_only_face_loss_vec(
    grad_face: Tensor,
    S_face: Tensor,
    V: Tensor,
    F: Tensor,
    n_hat: Tensor,
    *,
    kappa: float = 20.0,
    beta_pairs: float = 10.0,
    K: int = 3,
    S_full: Optional[Tensor] = None,
    w_jump: float = 1.0,
    w_normal: float = 0.5,
    w_tan: float = 0.25,
    ignore_face_mask: Optional[Tensor] = None,
    U_full: Optional[Tensor] = None,
    use_soft_flip: bool = False,
    softflip_beta: float = 12.0,
    softflip_margin: float = 0.15,
    softflip_eta: float = 12.0,
    softflip_alpha: float = 1.0,
    softflip_kappa: Optional[float] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Vectorized Stage-2 loss wrapper.

    The existing implementation is already batched; this wrapper keeps a
    stable API for switching from legacy paths and allows future tuning.
    """
    return heat_only_face_loss(
        grad_face=grad_face,
        S_face=S_face,
        V=V,
        F=F,
        n_hat=n_hat,
        kappa=kappa,
        beta_pairs=beta_pairs,
        K=K,
        S_full=S_full,
        w_jump=w_jump,
        w_normal=w_normal,
        w_tan=w_tan,
        ignore_face_mask=ignore_face_mask,
        U_full=U_full,
        use_soft_flip=use_soft_flip,
        softflip_beta=softflip_beta,
        softflip_margin=softflip_margin,
        softflip_eta=softflip_eta,
        softflip_alpha=softflip_alpha,
        softflip_kappa=softflip_kappa,
    )


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------
def argmin_labels(S_vert: Tensor) -> Tensor:
    """Return hard argmin labels per vertex."""

    return torch.argmin(S_vert, dim=1)


# -----------------------------------------------------------------------------
# Gradient-only pair selection utilities
# -----------------------------------------------------------------------------
@torch.no_grad()
def _best_pair_per_face_from_edges(
    edge_tris: Tensor,
    pair_ids: Tensor,
    scores: Tensor,
    n_faces: int,
) -> Tuple[Tensor, Tensor]:
    """
    Aggregate hinge-mirror scores from incident edges to pick the best unordered
    pair per face. Returns (face_pair, face_conf), with face_pair[f] = (i,j)
    or (-1,-1) if no evidence, and face_conf the accumulated score.
    """

    device = scores.device
    dtype = scores.dtype
    if pair_ids.numel() == 0 or scores.numel() == 0:
        face_pair = torch.full((n_faces, 2), -1, dtype=torch.long, device=device)
        face_conf = torch.zeros(n_faces, dtype=dtype, device=device)
        return face_pair, face_conf

    P = pair_ids.shape[0]
    acc = torch.zeros(n_faces, P, dtype=dtype, device=device)

    valid_left = edge_tris[:, 0] >= 0
    if valid_left.any():
        acc.index_add_(0, edge_tris[valid_left, 0], scores[valid_left])
    valid_right = edge_tris[:, 1] >= 0
    if valid_right.any():
        acc.index_add_(0, edge_tris[valid_right, 1], scores[valid_right])

    face_conf, best_idx = acc.max(dim=1)
    face_pair = pair_ids[best_idx]

    no_evidence = (face_conf <= 0) | (~torch.isfinite(face_conf))
    if no_evidence.any():
        face_pair = face_pair.clone()
        face_pair[no_evidence] = -1
        face_conf = face_conf.clone()
        face_conf[no_evidence] = 0.0

    return face_pair, face_conf


@torch.no_grad()
def _extract_segment_for_pair_in_face(
    V: Tensor,
    Ff: Tensor,
    S_all: Tensor,
    pair: Tuple[int, int],
    *,
    face_index: int,
    faces_all: Tensor,
    s_per_seed: Tensor,
    enable_certify: bool = True,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    Compute the Δ_ij bisector segment inside a face for the given unordered pair.
    Returns endpoints and barycentric coordinates if a valid, certified segment
    exists; otherwise None.
    """

    i, j = int(pair[0]), int(pair[1])
    if i < 0 or j < 0 or i == j:
        return None

    delta = S_all[Ff, i] - S_all[Ff, j]  # (3,)
    edge_idx = EDGE_INDEX_TEMPLATE.to(V.device)

    Da = delta[edge_idx[:, 0]]
    Db = delta[edge_idx[:, 1]]
    edge_mask = (Da * Db) <= 0.0
    if not edge_mask.any():
        return None

    denom = Da - Db
    lam = torch.where(
        denom.abs() < 1.0e-12,
        torch.full_like(denom, 0.5),
        Da / denom,
    )
    lam = lam.clamp_(0.0, 1.0)

    valid_idx = torch.nonzero(edge_mask, as_tuple=False).squeeze(1)
    if valid_idx.numel() != 2:
        return None

    lam_sel = lam[valid_idx]
    edge_vertices = Ff[edge_idx[valid_idx]]  # (2,2)
    pa = V[edge_vertices[:, 0]]
    pb = V[edge_vertices[:, 1]]
    points = (1.0 - lam_sel)[:, None] * pa + lam_sel[:, None] * pb

    bary = torch.zeros((2, 3), dtype=V.dtype, device=V.device)
    bary[torch.arange(2, device=V.device), edge_idx[valid_idx, 0]] = 1.0 - lam_sel
    bary[torch.arange(2, device=V.device), edge_idx[valid_idx, 1]] = lam_sel

    host_edges = [
        (int(edge_vertices[k, 0].item()), int(edge_vertices[k, 1].item()))
        for k in range(2)
    ]
    lambdas = [float(lam_sel[k].item()) for k in range(2)]

    if enable_certify:
        ok = certify_segment_endpoints(
            faces_all,
            (i, j),
            face_index,
            (host_edges[0], host_edges[1]),
            (lambdas[0], lambdas[1]),
            s_per_seed,
        )
        if not ok:
            return None

    return points[0], points[1], bary[0], bary[1]


# -----------------------------------------------------------------------------
# Heat-method distance baseline helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def _divergence_from_face_field(model: VoronoiHeatModel, X_face: Tensor) -> Tensor:
    """Compute vertex divergence of a per-face vector field for each seed."""

    gI = model.gI.unsqueeze(1)  # (nF,1,3)
    gJ = model.gJ.unsqueeze(1)
    gK = model.gK.unsqueeze(1)
    area = model.A_f.unsqueeze(1).to(X_face.dtype)

    dot_I = (X_face * gI).sum(dim=2)
    dot_J = (X_face * gJ).sum(dim=2)
    dot_K = (X_face * gK).sum(dim=2)

    div = torch.zeros(model.V.shape[0], X_face.shape[1], dtype=X_face.dtype, device=X_face.device)
    weight = (-area)
    div.index_add_(0, model.F[:, 0], weight * dot_I)
    div.index_add_(0, model.F[:, 1], weight * dot_J)
    div.index_add_(0, model.F[:, 2], weight * dot_K)
    return div


def _solve_poisson_multi(model: VoronoiHeatModel, rhs: Tensor, *, tol: float, iters: int) -> Tensor:
    """Solve L phi = rhs for multiple right-hand sides via CG."""

    dtype = rhs.dtype
    if hasattr(model, "L_csr") and model.L_csr.numel() > 0:
        L_csr = model.L_csr.to(dtype=dtype)

        def matvec(X: Tensor) -> Tensor:
            return _safe_sparse_mm(L_csr, X)

    else:

        def matvec(X: Tensor) -> Tensor:
            return spmv_coo_multi(model.L_row, model.L_col, model.L_val.to(dtype), X, model.V.shape[0])

    diag = model.L_diag.to(dtype=dtype)
    return _cg_matrix_multi(matvec, rhs, diag, tol=tol, maxiter=iters)


@torch.no_grad()
def heat_method_distances(
    model: VoronoiHeatModel,
    seeds: Sequence[int],
    *,
    cg_tol: float = 1.0e-6,
    cg_iters: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute geodesic distances per seed via the heat method baseline."""

    if not seeds:
        raise ValueError("At least one seed is required for the baseline.")

    nV = model.V.shape[0]
    C = len(seeds)
    B = torch.zeros(nV, C, dtype=model.V.dtype, device=model.V.device)
    for c, s in enumerate(seeds):
        B[s, c] = model.M_diag[s]

    U, X_face, _ = model(B, cg_tol=cg_tol, cg_iters=cg_iters)

    div = _divergence_from_face_field(model, X_face)
    div = div - div.mean(dim=0, keepdim=True)

    phi = _solve_poisson_multi(model, div, tol=cg_tol, iters=cg_iters)

    for c, s in enumerate(seeds):
        phi[:, c] -= phi[s, c]
        phi[:, c] -= phi[:, c].min()

    labels = torch.argmin(phi, dim=1)
    return labels.cpu().numpy(), phi.detach().cpu().numpy()
# -----------------------------------------------------------------------------
# Main container and training utilities
# -----------------------------------------------------------------------------
class VoronoiHeatModel(torch.nn.Module):
    """Hold mesh data and provide forward pass returning (U, X, S)."""

    def __init__(self, V_np: np.ndarray, F_np: np.ndarray, t: Optional[float] = None, device: str = "cpu"):
        super().__init__()
        torch_device = torch.device(device)
        dtype = _dtype_for_device(torch_device)
        V = torch.tensor(V_np, dtype=dtype, device=torch_device)
        F = torch.tensor(F_np, dtype=torch.long, device=torch_device)
        self.register_buffer("V", V)
        self.register_buffer("F", F)

        edges_np = np.concatenate(
            [
                F_np[:, [0, 1]],
                F_np[:, [1, 2]],
                F_np[:, [2, 0]],
            ],
            axis=0,
        )
        edges_np.sort(axis=1)
        edges_np = np.unique(edges_np, axis=0)
        edges_tensor = torch.tensor(edges_np, dtype=torch.long, device=torch_device)
        self.register_buffer("edge_index", edges_tensor)

        A_f, n_hat, gI, gJ, gK = build_face_geometry(V, F)
        self.register_buffer("A_f", A_f)
        self.register_buffer("n_hat", n_hat)
        self.register_buffer("gI", gI)
        self.register_buffer("gJ", gJ)
        self.register_buffer("gK", gK)

        mass_diag = build_mass_lumped_diag(V, F, A_f, device=torch_device)
        row, col, val, _ = build_cotan_laplacian_coo(V, F)
        self.register_buffer("M_diag", mass_diag)
        self.register_buffer("L_row", row)
        self.register_buffer("L_col", col)
        self.register_buffer("L_val", val)

        diag_mask = row == col
        L_diag = torch.zeros_like(mass_diag)
        if diag_mask.any():
            L_diag.index_add_(0, row[diag_mask], val[diag_mask])
        self.register_buffer("L_diag", L_diag)

        if torch_device.type != "mps":
            indices = torch.stack([row, col], dim=0)
            L_coo = torch.sparse_coo_tensor(indices, val, (V.shape[0], V.shape[0]), device=torch_device)
            self.register_buffer("L_csr", L_coo.coalesce().to_sparse_csr())
        else:
            self.register_buffer("L_csr", torch.empty(0, device=torch_device), persistent=False)

        if t is None:
            h2 = mean_edge_length(V, F) ** 2
            t = h2
        self.t = float(t)

        # Vectorization caches (pair indices and edge topology)
        self.register_buffer("pair_idx_cache", torch.empty(0, 2, dtype=torch.long, device=torch_device), persistent=False)
        self.register_buffer("edge_index", getattr(self, "edge_index", torch.empty(0, 2, dtype=torch.long, device=torch_device)), persistent=False)
        self.register_buffer("edge_tris", torch.empty(0, 2, dtype=torch.long, device=torch_device), persistent=False)
        try:
            self._build_topology_cache()
        except Exception:
            pass

    @torch.no_grad()
    def _build_topology_cache(self) -> None:
        """Precompute unique edges and edge adjacency for vectorized ops."""
        if self.F.numel() == 0:
            return
        F = self.F.long()
        device = self.V.device
        e01 = torch.stack([F[:, 0], F[:, 1]], dim=1)
        e12 = torch.stack([F[:, 1], F[:, 2]], dim=1)
        e20 = torch.stack([F[:, 2], F[:, 0]], dim=1)
        E_all = torch.cat([e01, e12, e20], dim=0)
        E_sort, _ = torch.sort(E_all, dim=1)
        E_unique, inv = torch.unique(E_sort, dim=0, return_inverse=True)
        self.edge_index = E_unique.to(device=device)
        E = E_unique.shape[0]
        face_ids = torch.arange(F.shape[0], device=device)
        face_ids = face_ids.repeat(3)
        edge_tris = torch.full((E, 2), -1, dtype=torch.long, device=device)
        counts = torch.zeros((E,), dtype=torch.int64, device=device)
        pos = counts[inv]
        edge_tris[inv, pos] = face_ids
        counts.index_add_(0, inv, torch.ones_like(inv, dtype=torch.int64))
        self.edge_tris = edge_tris

    @torch.no_grad()
    def ensure_pair_indices(self, C: int) -> torch.Tensor:
        """Return cached unordered pair indices (P,2) for given channel count."""
        if self.pair_idx_cache.numel() == 0 or self.pair_idx_cache.shape[0] != (C * (C - 1)) // 2:
            idx = torch.combinations(torch.arange(C, device=self.V.device), r=2)
            self.pair_idx_cache = idx.to(dtype=torch.long)
        return self.pair_idx_cache

    def forward(self, B_heat: Tensor, *, cg_tol: float = 1.0e-6, cg_iters: int = 200) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform heat solve and return (U, X_face, S)."""

        if (
            self.V.device.type != "mps"
            and hasattr(self, "L_csr")
            and isinstance(self.L_csr, torch.Tensor)
            and self.L_csr.layout == torch.sparse_csr
            and self.L_csr.numel() > 0
        ):
            A_diag = (self.M_diag + self.t * self.L_diag).to(B_heat.dtype)

            def matvec(X: Tensor) -> Tensor:
                return self.M_diag[:, None] * X + self.t * _safe_sparse_mm(self.L_csr, X)

            U = _cg_matrix_multi(matvec, B_heat, A_diag, tol=cg_tol, maxiter=cg_iters)
        else:
            U = heat_solve_multi(
                self.M_diag,
                self.L_row,
                self.L_col,
                self.L_val,
                B_heat,
                t=self.t,
                tol=cg_tol,
                iters=cg_iters,
            )
        S = scores_from_u(U)
        grad = face_gradients(U, self.F, self.gI, self.gJ, self.gK)
        X = unit_directions_from_grad(grad)
        return U, X, S


class TrainableSources(torch.nn.Module):
    """Trainable soft heat sources; logits initialised around provided seeds."""

    def __init__(self, nV: int, seeds: List[int], sharpness: float = 10.0, device: str = "cpu"):
        super().__init__()
        dtype = _dtype_for_device(device)
        C = len(seeds)
        logits = -5.0 * torch.ones(C, nV, dtype=dtype, device=device)

        fixed_mask = torch.zeros_like(logits, dtype=torch.bool)
        fixed_values = torch.zeros_like(logits)

        for c, s in enumerate(seeds):
            logits[:, s] = -sharpness
            logits[c, s] = sharpness
            fixed_mask[:, s] = True
            fixed_values[:, s] = -sharpness
            fixed_values[c, s] = sharpness

        self.logits = torch.nn.Parameter(logits)
        self.register_buffer("fixed_mask", fixed_mask, persistent=False)
        self.register_buffer("fixed_values", fixed_values, persistent=False)

        if fixed_mask.any():
            def _mask_grad(grad: Tensor) -> Tensor:
                return grad.masked_fill(self.fixed_mask, 0)

            self.logits.register_hook(_mask_grad)
            self.enforce_seed_logits()

    def forward(self, M_diag: Tensor, *, temp: float = 1.0) -> Tensor:
        logits = self.logits / float(temp)
        probs = torch.softmax(logits, dim=1)  # (C, nV)
        return M_diag[:, None] * probs.t()

    def enforce_seed_logits(self) -> None:
        if hasattr(self, "fixed_mask") and self.fixed_mask is not None:
            with torch.no_grad():
                self.logits.data[self.fixed_mask] = self.fixed_values[self.fixed_mask]


def train_step(
    model: VoronoiHeatModel,
    sources: TrainableSources,
    optimizer: torch.optim.Optimizer,
    *,
    beta_pairs: float = 10.0,
    kappa: float = 20.0,
    K: int = 3,
    use_soft_flip: bool = False,
    softflip_beta: float = 12.0,
    softflip_margin: float = 0.15,
    softflip_eta: float = 12.0,
    softflip_alpha: float = 1.0,
    softflip_kappa: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """Single optimisation step updating the source logits."""

    optimizer.zero_grad()
    B_heat = sources(model.M_diag)
    U, X, S = model(B_heat)
    S_face = torch.stack([S[model.F[:, 0]], S[model.F[:, 1]], S[model.F[:, 2]]], dim=1).mean(dim=1)
    grad_face = face_gradients(S, model.F, model.gI, model.gJ, model.gK)

    loss, terms = heat_only_face_loss(
        grad_face=grad_face,
        S_face=S_face,
        V=model.V,
        F=model.F,
        n_hat=model.n_hat,
        kappa=kappa,
        beta_pairs=beta_pairs,
        K=K,
        S_full=S,
        U_full=U,
        use_soft_flip=use_soft_flip,
        softflip_beta=softflip_beta,
        softflip_margin=softflip_margin,
        softflip_eta=softflip_eta,
        softflip_alpha=softflip_alpha,
        softflip_kappa=softflip_kappa,
    )
    loss.backward()
    optimizer.step()
    return float(loss.item()), {k: float(v.item()) for k, v in terms.items()}


@torch.no_grad()
def infer_labels_and_segments(
    model: VoronoiHeatModel,
    seeds: Optional[List[int]] = None,
    B_heat: Optional[Tensor] = None,
    *,
    seed_indices: Optional[Sequence[int]] = None,
    seed_targets: Optional[Tensor] = None,
    enforce_seed_mask: bool = False,
    cg_tol: float = 1.0e-6,
    cg_iters: int = 500,
) -> Tuple[
    np.ndarray,
    List[Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Run inference with simple per-face bisector extraction."""

    nV = model.V.shape[0]
    if B_heat is not None:
        B = B_heat
    else:
        if seeds is None:
            raise ValueError("Either seeds or B_heat must be provided.")
        C = len(seeds)
        B = torch.zeros(nV, C, dtype=model.V.dtype, device=model.V.device)
        for c, s in enumerate(seeds):
            B[s, c] = model.M_diag[s]
    U, X, S = model(B, cg_tol=cg_tol, cg_iters=cg_iters)

    if enforce_seed_mask:
        if seed_indices is None or seed_targets is None:
            raise ValueError("Seed indices and targets must be provided when enforce_seed_mask=True.")
        idx_tensor = torch.as_tensor(seed_indices, device=model.V.device, dtype=torch.long)
        target_tensor = seed_targets.to(model.V.device, dtype=S.dtype)
        if target_tensor.shape != (idx_tensor.shape[0], S.shape[1]):
            raise ValueError("seed_targets must have shape (num_seeds, num_channels).")
        S = S.clone()
        S[idx_tensor] = target_tensor

    labels_tensor = argmin_labels(S)

    edge_idx, edge_tris, pair_ids, scores, conf = edge_pair_mirror_scores(
        model.V,
        model.F,
        S,
        beta_edge=12.0,
    )

    n_faces = model.F.shape[0]
    if scores.numel() > 0 and scores.shape[1] > 0 and pair_ids.numel() > 0:
        conf_weight = conf.clamp_min(1.0e-4).unsqueeze(1)
        scores = scores * conf_weight
        scores = _smooth_edge_pair_scores(scores, edge_tris, iters=3, lam=0.6)
        face_pair_tensor, face_conf = _best_pair_per_face_from_edges(edge_tris, pair_ids, scores, n_faces)

        avg_conf = torch.zeros(n_faces, dtype=conf.dtype, device=conf.device)
        counts = torch.zeros(n_faces, dtype=conf.dtype, device=conf.device)
        valid_left = edge_tris[:, 0] >= 0
        if valid_left.any():
            avg_conf.index_add_(0, edge_tris[valid_left, 0], conf[valid_left])
            counts.index_add_(0, edge_tris[valid_left, 0], torch.ones_like(conf[valid_left]))
        valid_right = edge_tris[:, 1] >= 0
        if valid_right.any():
            avg_conf.index_add_(0, edge_tris[valid_right, 1], conf[valid_right])
            counts.index_add_(0, edge_tris[valid_right, 1], torch.ones_like(conf[valid_right]))
        avg_conf = torch.where(
            counts > 0,
            avg_conf / counts.clamp_min(1.0),
            avg_conf.new_zeros(avg_conf.shape),
        )

        face_pair = face_pair_tensor
        face_mask = torch.isfinite(face_conf) & (face_conf > 0) & (avg_conf >= 1.0e-4)
        if not face_mask.any():
            face_mask = torch.isfinite(face_conf) & (face_conf > 0)
        face_iter = torch.nonzero(face_mask, as_tuple=False).squeeze(1).tolist()
    else:
        face_pair = torch.full((n_faces, 2), -1, dtype=torch.long, device=model.V.device)
        face_iter = list(range(n_faces))

    seed_face_mask = None
    if seeds:
        seed_tensor = torch.tensor(seeds, device=model.F.device, dtype=torch.long)
        if seed_tensor.numel() > 0:
            mask = torch.zeros(model.F.shape[0], dtype=torch.bool, device=model.F.device)
            for s in seed_tensor:
                mask |= (model.F == s).any(dim=1)
            seed_face_mask = mask

    raw_segments: List[Tuple[np.ndarray, np.ndarray]] = []
    faces_idx: List[int] = []
    bary_data: List[np.ndarray] = []
    s_per_seed = S.t()

    # Align face pairs with observed vertex labels to avoid inconsistent bisectors.
    if face_iter:
        face_pair = face_pair.clone()
        F_faces = model.F
        for face_id in face_iter:
            pair = tuple(face_pair[face_id].tolist())
            if pair[0] < 0 or pair[1] < 0:
                continue
            face_labels = labels_tensor[F_faces[face_id]]
            unique_labels = torch.unique(face_labels)
            if unique_labels.numel() < 2:
                continue
            combos: List[Tuple[int, int]] = []
            labels_list = [int(x.item()) for x in unique_labels]
            labels_list.sort()
            for idx_a in range(len(labels_list)):
                for idx_b in range(idx_a + 1, len(labels_list)):
                    combos.append((labels_list[idx_a], labels_list[idx_b]))
            if combos and pair not in combos:
                replacement = combos[0]
                face_pair[face_id, 0] = replacement[0]
                face_pair[face_id, 1] = replacement[1]

    for face_id in face_iter:
        if seed_face_mask is not None and seed_face_mask[face_id].item():
            continue
        pair = tuple(face_pair[face_id].tolist())
        out = _extract_segment_for_pair_in_face(
            model.V,
            model.F[face_id],
            S,
            pair,
            face_index=face_id,
            faces_all=model.F,
            s_per_seed=s_per_seed,
            enable_certify=True,
        )
        if out is None:
            continue
        p0, p1, b0, b1 = out
        raw_segments.append((p0.detach().cpu().numpy(), p1.detach().cpu().numpy()))
        faces_idx.append(int(face_id))
        bary_data.append(torch.stack([b0, b1], dim=0).cpu().numpy())

    if not raw_segments:
        for f in range(model.F.shape[0]):
            if seed_face_mask is not None and seed_face_mask[f].item():
                continue
            out = extract_segments_in_face_hard(
                model.V,
                model.F[f],
                S,
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

    stitch_tol = max(1.0e-6, 1.0e-5 * mean_edge_length(model.V, model.F))
    polylines = _stitch_polylines(raw_segments, tol=stitch_tol)
    if not polylines and raw_segments:
        polylines = [np.stack([seg[0], seg[1]], axis=0) for seg in raw_segments]

    if faces_idx:
        seg_faces_np = np.asarray(faces_idx, dtype=np.int64)
        seg_bary_np = np.asarray(bary_data, dtype=np.float32)
    else:
        seg_faces_np = np.zeros((0,), dtype=np.int64)
        seg_bary_np = np.zeros((0, 2, 3), dtype=np.float32)

    return (
        labels_tensor.cpu().numpy(),
        polylines,
        S.detach().cpu().numpy(),
        U.detach().cpu().numpy(),
        seg_faces_np,
        seg_bary_np,
    )
