"""Linear algebra helpers that avoid torch.sparse for MPS compatibility."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def laplacian_cotan_coo_fp32(V: Tensor, F: Tensor) -> Tuple[Tensor, Tensor, Tensor, int]:
    """
    Build cotangent Laplacian in COO triplets for use with MPS-safe SpMV.
    Returns (row, col, val, nV) on V.device with float32 values.
    """

    device = V.device
    dtype = torch.float32
    nV = V.shape[0]

    vi, vj, vk = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    def cot(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        u = b - a
        v = c - a
        cross = torch.linalg.norm(torch.cross(u, v, dim=-1), dim=-1).clamp_min(1e-30)
        dot = (u * v).sum(-1)
        return dot / cross

    cotA = cot(vi, vj, vk)
    cotB = cot(vj, vk, vi)
    cotC = cot(vk, vi, vj)

    w_ij = 0.5 * cotC
    w_jk = 0.5 * cotA
    w_ki = 0.5 * cotB

    ii, jj, kk = F[:, 0], F[:, 1], F[:, 2]

    rows = []
    cols = []
    vals = []

    for (a, b, w) in [
        (ii, jj, w_ij),
        (jj, ii, w_ij),
        (jj, kk, w_jk),
        (kk, jj, w_jk),
        (kk, ii, w_ki),
        (ii, kk, w_ki),
    ]:
        rows.append(a)
        cols.append(b)
        vals.append(-w)

    diag = torch.zeros(nV, device=device, dtype=dtype)
    diag.index_add_(0, ii, (w_ij + w_ki).to(dtype))
    diag.index_add_(0, jj, (w_ij + w_jk).to(dtype))
    diag.index_add_(0, kk, (w_jk + w_ki).to(dtype))
    rows.append(torch.arange(nV, device=device))
    cols.append(torch.arange(nV, device=device))
    vals.append(diag)

    row = torch.cat([r.reshape(-1) for r in rows]).to(torch.int64)
    col = torch.cat([c.reshape(-1) for c in cols]).to(torch.int64)
    val = torch.cat([v.reshape(-1) for v in vals]).to(dtype)
    return row, col, val, nV


def spmv_coo_multi(row: Tensor, col: Tensor, val: Tensor, X: Tensor, n: int) -> Tensor:
    """
    y = A * X for COO A (row, col, val). Supports multi-RHS X:(n,k).
    Implemented via gather + index_add_, which works on MPS.
    """

    VX = val[:, None] * X.index_select(dim=0, index=col)
    Y = torch.zeros((n, X.shape[1]), device=X.device, dtype=X.dtype)
    Y.index_add_(0, row, VX)
    return Y


class AMatrix:
    """Linear operator for A = M + t L in COO form on MPS."""

    def __init__(self, M_diag: Tensor, row: Tensor, col: Tensor, val: Tensor, t: float):
        self.Md = M_diag
        self.row = row
        self.col = col
        self.val = val
        self.n = M_diag.numel()
        self.t = float(t)
        diag_mask = row == col
        L_diag = torch.zeros(self.n, device=M_diag.device, dtype=M_diag.dtype)
        if diag_mask.any():
            L_diag.index_add_(0, row[diag_mask], val[diag_mask].to(M_diag.dtype))
        self.A_diag = M_diag + self.t * L_diag
        self.M_inv = 1.0 / (self.A_diag + 1e-12)

    def mv(self, X: Tensor) -> Tensor:
        return self.Md[:, None] * X + self.t * spmv_coo_multi(self.row, self.col, self.val, X, self.n)


def cg_solve_mv(A: AMatrix, B: Tensor, tol: float = 1e-6, maxit: int = 150) -> Tensor:
    """
    Conjugate Gradient with linear operator A.mv(X); supports multi-RHS B:(n,k).
    All tensors on same device/dtype (e.g., 'mps' fp32).
    """

    X = torch.zeros_like(B)
    R = B - A.mv(X)
    Z = A.M_inv[:, None] * R
    P = Z.clone()
    rz_old = (R * Z).sum(dim=0)
    active = torch.ones(B.shape[1], dtype=torch.bool, device=B.device)

    for _ in range(maxit):
        if not active.any():
            break
        AP = A.mv(P)
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
        Z = A.M_inv[:, None] * R
        rz_new = (R * Z).sum(dim=0)
        beta = torch.where(
            active,
            rz_new / rz_old.clamp_min(1e-30),
            torch.zeros_like(rz_old),
        )
        P = (Z + P * beta.unsqueeze(0)) * active.unsqueeze(0).to(B.dtype)
        rz_old = rz_new

    return X
