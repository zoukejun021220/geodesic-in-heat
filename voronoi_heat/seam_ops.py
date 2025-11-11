"""
Seam certification and cut-mesh utilities for the Voronoi heat pipeline.

This module holds mid-level geometry helpers to keep pipeline.py high-level.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from .voronoi_heat_torch import VoronoiHeatModel


@torch.no_grad()
def certify_segments_equal_distance(
    model: VoronoiHeatModel,
    seeds: List[int],
    seg_faces: np.ndarray,
    seg_bary: np.ndarray,
    *,
    cg_tol: float,
    cg_iters: int,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Vectorized certification via baseline heat distances.

    Keeps segments where the two smallest distances at both endpoints are
    nearly equal (|d_i-d_j|<=tau) and correspond to the same seed pair.
    Returns filtered (faces, bary) and the corresponding seed pairs.
    """
    from .voronoi_heat_torch import heat_method_distances  # local import to avoid cycles

    if seg_faces.size == 0:
        return seg_faces, seg_bary, []

    _, dists_np = heat_method_distances(
        model,
        seeds,
        cg_tol=cg_tol,
        cg_iters=cg_iters,
    )

    V = model.V
    F = model.F
    device = V.device
    dtype = V.dtype

    dists = torch.from_numpy(dists_np).to(device=device, dtype=dtype)  # (nV,C)
    C = dists.shape[1]
    if C < 2:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2, 3), dtype=np.float32), []

    faces_t = torch.from_numpy(seg_faces.astype(np.int64, copy=False)).to(device)
    bary_t = torch.from_numpy(seg_bary.astype(np.float32, copy=False)).to(device=device, dtype=dtype)  # (N,2,3)
    verts = F[faces_t]                     # (N,3)
    dist_f = dists[verts]                  # (N,3,C)
    w0 = bary_t[:, 0, :].unsqueeze(-1)     # (N,3,1)
    w1 = bary_t[:, 1, :].unsqueeze(-1)     # (N,3,1)
    d0 = (dist_f * w0).sum(dim=1)          # (N,C)
    d1 = (dist_f * w1).sum(dim=1)          # (N,C)

    K = 2
    top0 = torch.topk(-d0, k=K, dim=1).indices  # (N,2)
    top1 = torch.topk(-d1, k=K, dim=1).indices  # (N,2)
    pair0 = torch.sort(top0, dim=1).values      # (N,2)
    pair1 = torch.sort(top1, dim=1).values      # (N,2)
    same_pair = (pair0 == pair1).all(dim=1)

    tol = float(tau)
    i0, j0 = pair0[:, 0], pair0[:, 1]
    eq0 = (d0.gather(1, i0.view(-1, 1)) - d0.gather(1, j0.view(-1, 1))).abs().squeeze(1) <= tol
    eq1 = (d1.gather(1, i0.view(-1, 1)) - d1.gather(1, j0.view(-1, 1))).abs().squeeze(1) <= tol
    keep_mask = same_pair & eq0 & eq1

    if not keep_mask.any():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2, 3), dtype=np.float32), []

    faces_keep = faces_t[keep_mask].detach().cpu().numpy().astype(np.int64, copy=False)
    bary_keep = bary_t[keep_mask].detach().cpu().numpy().astype(np.float32, copy=False)
    pairs_keep = pair0[keep_mask].detach().cpu().tolist()
    pairs_out = [(int(a), int(b)) for a, b in pairs_keep]
    return faces_keep, bary_keep, pairs_out


def build_cut_mesh_from_segments(
    V_np: np.ndarray,
    F_np: np.ndarray,
    seg_faces: np.ndarray,
    seg_bary: np.ndarray,
    seg_pairs: List[Tuple[int, int]],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[Tuple[int, int, int, int, int, Tuple[int, int], Tuple[int, int, int], np.ndarray, np.ndarray]],
]:
    """Build cut mesh per face segment by duplicating seam DOFs on both sides."""

    V_new = V_np.tolist()
    F_new: List[List[int]] = []
    seam_infos: List[Tuple[int, int, int, int, int, Tuple[int, int], Tuple[int, int, int], np.ndarray, np.ndarray]] = []

    def add_vertex(xyz: np.ndarray) -> int:
        V_new.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        return len(V_new) - 1

    for idx in range(seg_faces.shape[0]):
        f = int(seg_faces[idx])
        b0 = seg_bary[idx, 0]
        b1 = seg_bary[idx, 1]
        i0, i1, i2 = F_np[f]
        tri = np.array([V_np[i0], V_np[i1], V_np[i2]], dtype=np.float64)
        p = (b0[0] * tri[0] + b0[1] * tri[1] + b0[2] * tri[2]).astype(np.float64)
        q = (b1[0] * tri[0] + b1[1] * tri[1] + b1[2] * tri[2]).astype(np.float64)

        zero0 = int(np.argmin(np.abs(b0)))
        zero1 = int(np.argmin(np.abs(b1)))
        edge0 = [i0, i1, i2][:]
        edge1 = [i0, i1, i2][:]
        del edge0[zero0]
        del edge1[zero1]
        shared_candidates = set(edge0).intersection(set(edge1))
        shared = list(shared_candidates)[0] if shared_candidates else i0
        others = [v for v in [i0, i1, i2] if v != shared]
        vj, vk = int(others[0]), int(others[1])

        pL = add_vertex(p)
        qL = add_vertex(q)
        pR = add_vertex(p)
        qR = add_vertex(q)

        F_new.append([shared, pL, qL])
        faceL = len(F_new) - 1
        F_new.append([pR, vj, vk])
        F_new.append([pR, vk, qR])
        faceR = len(F_new) - 1

        seam_infos.append((
            pL, qL, faceL, pR, qR, faceR,
            (int(seg_pairs[idx][0]), int(seg_pairs[idx][1])),
            (int(i0), int(i1), int(i2)),
            b0.copy(), b1.copy(),
        ))

    return np.asarray(V_new, dtype=np.float64), np.asarray(F_new, dtype=np.int64), seam_infos


def lift_sources_to_cut(
    B_orig: Tensor,
    V_orig_np: np.ndarray,
    F_orig_np: np.ndarray,
    V_new_np: np.ndarray,
    seam_infos: List[Tuple[int, int, int, int, int, Tuple[int, int], Tuple[int, int, int], np.ndarray, np.ndarray]],
) -> Tensor:
    """Lift RHS sources to the cut mesh by barycentric interpolation on host faces."""

    device = B_orig.device
    dtype = B_orig.dtype
    nV_new = V_new_np.shape[0]
    C = B_orig.shape[1]
    B_new = torch.zeros((nV_new, C), device=device, dtype=dtype)
    nV_orig = B_orig.shape[0]
    B_new[:nV_orig, :] = B_orig

    for pL, qL, faceL, pR, qR, faceR, pair, tri, b0, b1 in seam_infos:
        i0, i1, i2 = tri
        for v_new, b in ((pL, b0), (qL, b1), (pR, b0), (qR, b1)):
            B_new[v_new] = (
                float(b[0]) * B_orig[i0] + float(b[1]) * B_orig[i1] + float(b[2]) * B_orig[i2]
            )
    return B_new


__all__ = [
    "certify_segments_equal_distance",
    "build_cut_mesh_from_segments",
    "lift_sources_to_cut",
]
