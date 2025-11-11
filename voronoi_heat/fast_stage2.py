import torch
from torch import Tensor

from .voronoi_heat_torch import face_gradients


@torch.no_grad()
def _topk_pairs_by_tie(Sf: Tensor, K: int) -> tuple[Tensor, Tensor]:
    """Vectorized top-K unordered pairs per face by bisector evidence |Si - Sj|.

    Sf: (F, C) face-averaged scores.
    Returns (pi, pj) each (F, K) with i<j.
    """
    F, C = Sf.shape
    if C < 2:
        z = torch.zeros((F, 1), dtype=torch.long, device=Sf.device)
        return z, z
    tri = torch.ones(C, C, device=Sf.device, dtype=torch.bool).tril(-1)
    diff = (Sf.unsqueeze(2) - Sf.unsqueeze(1)).abs()  # (F,C,C)
    masked = torch.where(tri, diff, torch.full_like(diff, float("inf")))
    flat = masked.reshape(F, -1)
    k = min(K, int(tri.sum().item()))
    neg = -flat
    idx = torch.topk(neg, k=k, dim=1).indices  # (F,K)
    pi = idx // C
    pj = idx % C
    return pi, pj


def heat_only_face_loss_fast(
    *,
    V: Tensor,
    F: Tensor,
    S_vert: Tensor,
    n_f: Tensor,
    gI: Tensor,
    gJ: Tensor,
    gK: Tensor,
    K: int,
    beta_pairs: float,
    w_normal: float,
    w_tan: float,
) -> tuple[Tensor, dict]:
    """Fully vectorized Stage-2 seam loss (no Python loops).

    Inputs:
      V: (Vtx,3), F: (F,3) long, S_vert: (Vtx,C) scores (-log U),
      n_f: (F,3) unit face normals,
      gI,gJ,gK: (F,3) per-face hat-function gradients (model buffers).
    Returns (loss, {"normal":..., "tan":..., "jump":0.0}).
    """
    device, dtype = V.device, V.dtype
    F_n = F.shape[0]
    if S_vert.numel() == 0 or F_n == 0:
        z = torch.zeros((), device=device, dtype=dtype)
        return z, {"normal": z, "tan": z, "jump": z}
    C = S_vert.shape[1]
    if C < 2:
        z = torch.zeros((), device=device, dtype=dtype)
        return z, {"normal": z, "tan": z, "jump": z}

    # Face-averaged scores and face gradients
    Sf = S_vert[F.long()].mean(dim=1)  # (F,C)
    Gf = face_gradients(S_vert, F.long(), gI, gJ, gK)  # (F,C,3)

    # Top-K pair indices per face by |Si - Sj|
    pi, pj = _topk_pairs_by_tie(Sf, K)  # (F,K)
    K_sel = pi.shape[1]
    rows = torch.arange(F_n, device=device)[:, None].expand(F_n, K_sel)

    # Gather gradients for selected pairs: Gi,Gj ∈ R^{F×K×3}
    Gi = Gf[rows, pi]
    Gj = Gf[rows, pj]
    dG = Gi - Gj

    # Seam tangent τ = normalize(n × (Gi - Gj)) (no branchy edge logic)
    nf_exp = n_f.unsqueeze(1).expand_as(dG)
    tau = torch.cross(nf_exp, dG, dim=-1)
    tau = tau / tau.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # In-plane normal to seam within face
    n_e = torch.cross(nf_exp, tau, dim=-1)
    n_e = n_e / n_e.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # Unit directions (bounded residuals on bad meshes)
    Xi = torch.nn.functional.normalize(Gi, dim=-1)
    Xj = torch.nn.functional.normalize(Gj, dim=-1)

    # Directional residuals per (F,K)
    n_term = ((Xi + Xj) * n_e).sum(dim=-1).pow(2)
    t_term = ((Xi - Xj) * tau).sum(dim=-1).pow(2)
    r_fk = w_normal * n_term + w_tan * t_term

    # Stable per-face weights: tie gate 4 p_i p_j with p=softmax(-β Sf)
    p = torch.softmax(-float(beta_pairs) * Sf, dim=1)
    piw = p.gather(1, pi)
    pjw = p.gather(1, pj)
    w_fk = (4.0 * piw * pjw).clamp_min(1e-8)

    # Aggregate (normalize to avoid tiny-sum explosions)
    num = (w_fk * r_fk).sum()
    den = w_fk.sum().clamp_min(1e-12)
    loss = num / den

    breakdown = {
        "normal": (w_fk * n_term).sum() / den,
        "tan": (w_fk * t_term).sum() / den,
        "jump": torch.zeros((), device=device, dtype=dtype),
    }
    return loss, breakdown

