"""
Certification utilities for Voronoi segments using endpoint minimality (report §3.8).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


@torch.no_grad()
def certify_segment_endpoints(
    faces: Tensor,
    pair: Tuple[int, int],
    face_index: int,
    host_edges: Tuple[Tuple[int, int], Tuple[int, int]],
    lambdas: Tuple[float, float],
    s_per_seed: Tensor,
    eps_abs: float = 1e-7,
    eps_rel: float = 1e-6,
) -> bool:
    """
    True iff segment passes minimality vs all other seeds at both endpoints.
    """
    _ = faces, face_index
    i, j = pair
    C, V = s_per_seed.shape
    others = [k for k in range(C) if k != i and k != j]
    if not others:
        return True

    device = s_per_seed.device
    dtype = s_per_seed.dtype
    eps_abs_t = torch.as_tensor(eps_abs, dtype=dtype, device=device)
    eps_rel_t = torch.as_tensor(eps_rel, dtype=dtype, device=device)

    s_i = s_per_seed[i]
    s_j = s_per_seed[j]
    for lam, (a, b) in zip(lambdas, host_edges):
        lam_tensor = lam if isinstance(lam, torch.Tensor) else torch.as_tensor(lam, dtype=dtype, device=device)
        si_p = (1.0 - lam_tensor) * s_i[a] + lam_tensor * s_i[b]
        sj_p = (1.0 - lam_tensor) * s_j[a] + lam_tensor * s_j[b]
        sij = torch.minimum(si_p, sj_p)
        for k in others:
            sk = s_per_seed[k]
            sk_p = (1.0 - lam_tensor) * sk[a] + lam_tensor * sk[b]
            tol = eps_abs_t + eps_rel_t * torch.maximum(sij.abs(), sk_p.abs())
            if (sk_p + tol < sij).item():  # violated
                return False
    return True
