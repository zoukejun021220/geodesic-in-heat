"""
Hinge-parallel transport utilities (discrete Levi-Civita) for triangle meshes.
See §3.3 in the report.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _safe_norm(x: Tensor, dim: int = -1, eps: float = 1e-30) -> Tensor:
    return torch.linalg.norm(x, dim=dim, keepdim=True).clamp_min(eps)


def _rodrigues_rotate(v: Tensor, axis: Tensor, cos_theta: Tensor, sin_theta: Tensor) -> Tensor:
    axis = axis / _safe_norm(axis)
    cross = torch.cross(axis, v, dim=-1)
    dot = (axis * v).sum(dim=-1, keepdim=True)
    return v * cos_theta + cross * sin_theta + axis * dot * (1.0 - cos_theta)


@torch.no_grad()
def transport_right_to_left(
    unit_fields_right: Tensor,
    *,
    vertices: Tensor | None = None,
    faces: Tensor | None = None,
    left_faces: Tensor | None = None,
    right_faces: Tensor | None = None,
    edge_vertices: Tensor | None = None,
    nL: Tensor | None = None,
    nR: Tensor | None = None,
    axis: Tensor | None = None,
    rotate_mask: Tensor | None = None,
) -> Tensor:
    """
    Transport right-face vectors into the left-face plane across edges.
    Shapes:
      unit_fields_right: (K, N, 3)  # N edges, K seeds/channels
      left_faces,right_faces: (N,)
      edge_vertices: (N,2), vertices:(V,3), faces:(F,3)
    Returns: (K, N, 3) rotated vectors in the left plane.
    If insufficient geometry is provided, returns unit_fields_right unchanged (no-op).
    """
    if unit_fields_right.dim() != 3 or unit_fields_right.size(-1) != 3:
        return unit_fields_right

    device = unit_fields_right.device
    dtype = unit_fields_right.dtype

    # Compute face normals if needed
    if (nL is None or nR is None) and (vertices is not None and faces is not None and left_faces is not None and right_faces is not None):
        v0 = vertices[faces[:, 0].long()]
        v1 = vertices[faces[:, 1].long()]
        v2 = vertices[faces[:, 2].long()]
        e0 = v1 - v0
        e1 = v2 - v0
        n = torch.cross(e0, e1, dim=1)
        n_unit = n / _safe_norm(n)
        nL = n_unit[left_faces.long()]
        nR = n_unit[right_faces.long()]
    elif nL is None or nR is None:
        return unit_fields_right

    # Edge axis if needed
    if axis is None:
        if edge_vertices is None or vertices is None:
            return unit_fields_right
        a = vertices[edge_vertices[:, 0].long()]
        b = vertices[edge_vertices[:, 1].long()]
        axis = b - a

    axis = axis.to(device=device, dtype=dtype)
    axis = axis / _safe_norm(axis)

    nL = nL.to(device=device, dtype=dtype)
    nR = nR.to(device=device, dtype=dtype)

    # Signed dihedral: sign(θ) = sign(t ⋅ (n_R × n_L))
    cross_nr = torch.cross(nR, nL, dim=1)
    sin_theta = torch.linalg.norm(cross_nr, dim=1, keepdim=True)
    cos_theta = torch.clamp((nR * nL).sum(dim=1, keepdim=True), -1.0, 1.0)
    sign = torch.sign((axis * cross_nr).sum(dim=1, keepdim=True))
    sin_theta = sin_theta * sign

    if rotate_mask is None and right_faces is not None:
        rotate_mask = right_faces.ge(0)
    if rotate_mask is None:
        rotate_mask = torch.ones(axis.size(0), dtype=torch.bool, device=device)

    cos_term = cos_theta.to(device=device, dtype=dtype).unsqueeze(0)
    sin_term = sin_theta.to(device=device, dtype=dtype).unsqueeze(0)
    axis_exp = axis.to(device=device, dtype=dtype).unsqueeze(0)

    vR = unit_fields_right  # (K, N, 3)
    rotated = _rodrigues_rotate(vR, axis_exp, cos_term, sin_term)
    mask = rotate_mask.to(device=device).unsqueeze(0).unsqueeze(-1)
    return torch.where(mask, rotated, vR)
