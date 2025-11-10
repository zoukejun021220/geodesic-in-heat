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
    """
    Certify segments using baseline heat distances (independent of trainable sources).
    Keep segments where the two closest distances are nearly equal at both endpoints
    and correspond to the same seed pair. Returns filtered (faces, bary) and pair list.
    """
    from .voronoi_heat_torch import heat_method_distances  # local import to avoid cycles

    if seg_faces.size == 0:
        return seg_faces, seg_bary, []

    labels_np, dists_np = heat_method_distances(
        model,
        seeds,
        cg_tol=cg_tol,
        cg_iters=cg_iters,
    )
    V = model.V
    F = model.F
    dists = torch.from_numpy(dists_np).to(V.device, dtype=V.dtype)  # (nV,C)
    C = dists.shape[1]

    keep_faces: List[int] = []
    keep_bary: List[np.ndarray] = []
    keep_pairs: List[Tuple[int, int]] = []
    tol = float(tau)

    for idx in range(seg_faces.shape[0]):
        f = int(seg_faces[idx])
        bpair = seg_bary[idx]  # (2,3)
        face = F[f]
        w0 = torch.from_numpy(bpair[0]).to(V.device, dtype=V.dtype)
        w1 = torch.from_numpy(bpair[1]).to(V.device, dtype=V.dtype)
        d0 = (dists[face, :] * w0[:, None]).sum(dim=0)
        d1 = (dists[face, :] * w1[:, None]).sum(dim=0)
        top2_0 = torch.topk(-d0, k=min(2, C), largest=True).indices.tolist()
        top2_1 = torch.topk(-d1, k=min(2, C), largest=True).indices.tolist()
        if len(top2_0) < 2 or len(top2_1) < 2:
            continue
        pair0 = tuple(sorted((top2_0[0], top2_0[1])))
        pair1 = tuple(sorted((top2_1[0], top2_1[1])))
        if pair0 != pair1:
            continue
        i, j = pair0
        if abs(float(d0[i].item() - d0[j].item())) <= tol and abs(float(d1[i].item() - d1[j].item())) <= tol:
            keep_faces.append(f)
            keep_bary.append(bpair.copy())
            keep_pairs.append((i, j))

    if not keep_faces:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2, 3), dtype=np.float32), []
    return (
        np.asarray(keep_faces, dtype=np.int64),
        np.asarray(keep_bary, dtype=np.float32),
        keep_pairs,
    )


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

