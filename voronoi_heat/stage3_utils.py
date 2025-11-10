"""
Stage-3 helpers for polyline/tangent caching and face-pair aggregation.

Moved out of pipeline to keep it high-level.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .grad_alignment import edge_pair_mirror_scores_from_X
from .voronoi_heat_torch import (
    _smooth_edge_pair_scores,
    _best_pair_per_face_from_edges,
)


@torch.no_grad()
def compute_stage3_face_pairs(
    model, X_face: Tensor, *, beta_edge: float
) -> Tuple[Tensor, Tensor]:
    device = model.V.device
    dtype = X_face.dtype
    n_faces = model.F.shape[0]
    face_pair = torch.full((n_faces, 2), -1, dtype=torch.long, device=device)
    face_conf = torch.zeros(n_faces, dtype=dtype, device=device)

    edge_idx, edge_tris, pair_ids, scores, conf = edge_pair_mirror_scores_from_X(
        model.V, model.F, X_face, beta_edge=beta_edge
    )
    if scores.numel() == 0 or pair_ids.numel() == 0:
        return face_pair, face_conf

    conf_weight = conf.clamp_min(1.0e-4).unsqueeze(1)
    scores = scores * conf_weight
    scores = _smooth_edge_pair_scores(scores, edge_tris, iters=3, lam=0.6)
    face_pair, face_conf = _best_pair_per_face_from_edges(edge_tris, pair_ids, scores, n_faces)
    return face_pair, face_conf


@torch.no_grad()
def prepare_stage3_polyline_cache(
    model,
    *,
    seg_faces: np.ndarray,
    seg_bary: np.ndarray,
    face_pair: Tensor,
    face_conf: Tensor,
    seed_face_mask: Optional[Tensor] = None,
) -> Optional[Dict[str, Tensor]]:
    if seg_faces.size == 0 or seg_bary.size == 0:
        return None

    device = model.V.device
    dtype = model.V.dtype

    faces_tensor = torch.from_numpy(np.asarray(seg_faces, dtype=np.int64)).to(device=device)
    bary_tensor = torch.from_numpy(np.asarray(seg_bary, dtype=np.float32)).to(device=device, dtype=dtype)

    if faces_tensor.numel() == 0 or bary_tensor.numel() == 0:
        return None
    if bary_tensor.ndim != 3 or bary_tensor.shape[1:] != (2, 3):
        raise ValueError("segment_bary must have shape (N, 2, 3).")

    valid_face_mask = (faces_tensor >= 0) & (faces_tensor < model.F.shape[0])
    finite_mask = torch.isfinite(bary_tensor.view(bary_tensor.shape[0], -1)).all(dim=1)
    mask = valid_face_mask & finite_mask
    if not mask.any():
        return None

    faces_tensor = faces_tensor[mask]
    bary_tensor = bary_tensor[mask]

    face_vertices = model.V[model.F[faces_tensor]]  # (N,3,3)
    points = torch.bmm(face_vertices.transpose(1, 2), bary_tensor.permute(0, 2, 1)).transpose(1, 2)
    tangents = points[:, 1, :] - points[:, 0, :]
    lengths = tangents.norm(dim=1)

    length_mask = lengths > 1.0e-9
    if not length_mask.any():
        return None

    faces_tensor = faces_tensor[length_mask]
    tangents = tangents[length_mask]
    lengths = lengths[length_mask]

    unique_faces, inverse = torch.unique(faces_tensor, return_inverse=True)
    segment_weight = lengths * face_conf.to(device=device, dtype=dtype)[faces_tensor]
    tangent_weighted = tangents * segment_weight[:, None]

    tangent_accum = torch.zeros((unique_faces.shape[0], 3), dtype=dtype, device=device)
    tangent_accum.index_add_(0, inverse, tangent_weighted)

    weight_accum = torch.zeros(unique_faces.shape[0], dtype=dtype, device=device)
    weight_accum.index_add_(0, inverse, segment_weight)

    tangent_norm = tangent_accum.norm(dim=1)
    tangent_unit = tangent_accum / tangent_norm.clamp_min(1.0e-9).unsqueeze(1)

    pair_face = face_pair.to(device=device)
    pair_valid = (pair_face[:, 0] >= 0) & (pair_face[:, 1] >= 0) & (pair_face[:, 0] != pair_face[:, 1])

    if seed_face_mask is not None and seed_face_mask.numel() == model.F.shape[0]:
        keep_seed = (~seed_face_mask.to(device=device))[unique_faces]
    else:
        keep_seed = torch.ones_like(weight_accum, dtype=torch.bool)

    keep_mask = (
        (weight_accum > 0)
        & torch.isfinite(weight_accum)
        & (tangent_norm > 1.0e-9)
        & keep_seed
        & pair_valid[unique_faces]
    )
    if not keep_mask.any():
        return None

    faces_keep = unique_faces[keep_mask]
    tangent_keep = tangent_unit[keep_mask]
    weight_keep = weight_accum[keep_mask]
    pair_keep = pair_face[faces_keep]

    return {
        "faces": faces_keep.to(torch.long),
        "tangent": tangent_keep,
        "weight": weight_keep,
        "pairs": pair_keep.to(torch.long),
    }


__all__ = [
    "compute_stage3_face_pairs",
    "prepare_stage3_polyline_cache",
]
