"""Post-optimisation integration of Voronoi piecewise distance fields.

This module provides a CLI that takes the exported ``voronoi.vtp`` (or any
compatible VTK PolyData containing point arrays produced by the optimisation
pipeline), reconstructs the heat-based gradient field, and integrates a
piecewise scalar potential inside each Voronoi cell.

The resulting per-seed potentials are written back as a new point-data array
(``piecewise_phi`` by default) and, optionally, exported to an auxiliary NPZ
bundle for the lightweight viewer.

Example:

    python -m voronoi_heat.post_integrate \
        --input runs/voronoi_heat/final/voronoi.vtp \
        --seeds runs/voronoi_heat/seeds.txt \
        --out-vtp runs/voronoi_heat/final/voronoi_phi.vtp \
        --out-npz runs/voronoi_heat/final/piecewise_phi.npz

The script is intentionally independent of the optimisation pipeline so it can
be rerun whenever new VTP exports are produced.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:  # VTK is an optional dependency during training but required here.
    import vtk  # type: ignore
    from vtk.util import numpy_support as vtk_np  # type: ignore
except Exception as exc:  # pragma: no cover - VTK missing in minimal envs
    raise RuntimeError(
        "VTK is required to run voronoi_heat.post_integrate. "
        "Please install vtk (pip install vtk) before using this script."
    ) from exc

from .cut_by_labels import boundary_edges, cut_mesh_by_labels
from .pipeline import read_seeds
from .voronoi_heat_torch import VoronoiHeatModel, argmin_labels


@dataclass
class IntegrationResult:
    """Container bundling the outputs of the integration stage."""

    phi: np.ndarray
    heat: np.ndarray
    scores: np.ndarray


def _load_vtp(path: Path) -> Tuple[vtk.vtkPolyData, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load a VTP file and return geometry plus point-data arrays.

    Returns ``(polydata, vertices, faces, point_arrays)`` where ``faces`` are
    triangular with dtype ``int64`` and ``point_arrays`` maps array names to
    numpy views. Polyline seam data (if present) is preserved in ``polydata``
    and automatically extended when new arrays are written.
    """

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise ValueError(f"PolyData input '{path}' did not contain any points.")

    verts = vtk_np.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64, copy=False)

    polys = poly.GetPolys()
    cell_data = vtk_np.vtk_to_numpy(polys.GetData())
    if cell_data.size == 0:
        faces = np.zeros((0, 3), dtype=np.int64)
    else:
        cell_width = int(cell_data.size // polys.GetNumberOfCells())
        if cell_width != 4:
            raise ValueError(
                "Expected triangular cells in input VTP; got cell width != 4."
            )
        faces = cell_data.reshape(-1, cell_width)[:, 1:].astype(np.int64, copy=False)

    arrays: Dict[str, np.ndarray] = {}
    pd = poly.GetPointData()
    for idx in range(pd.GetNumberOfArrays()):
        arr = pd.GetArray(idx)
        if arr is None:
            continue
        name = arr.GetName() or ""
        if not name:
            continue
        np_arr = vtk_np.vtk_to_numpy(arr)
        n_comp = arr.GetNumberOfComponents()
        if n_comp > 1:
            np_arr = np_arr.reshape(-1, n_comp)
        arrays[name] = np_arr

    return poly, verts, faces, arrays


def _build_polydata(vertices: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(vtk_np.numpy_to_vtk(vertices.astype(np.float64), deep=True))
    pd.SetPoints(pts)
    cell_array = vtk.vtkCellArray()
    if faces.size > 0:
        header = np.full((faces.shape[0], 1), 3, dtype=np.int64)
        packed = np.hstack([header, faces.astype(np.int64)]).ravel()
        vtk_cells = vtk_np.numpy_to_vtk(packed, deep=True, array_type=vtk.VTK_ID_TYPE)
        cell_array.SetCells(faces.shape[0], vtk_cells)
    pd.SetPolys(cell_array)
    return pd


def _build_dirac_rhs(
    model: VoronoiHeatModel,
    seeds: List[int],
    sources_arr: Optional[np.ndarray],
) -> torch.Tensor:
    """Return the heat RHS matrix ``B`` as a tensor on the model's device."""

    nV = model.V.shape[0]
    dtype = model.V.dtype
    device = model.V.device
    C = len(seeds)

    if sources_arr is not None:
        if sources_arr.shape[0] < nV:
            raise ValueError("Sources array in VTP is shorter than vertex count.")
        if sources_arr.shape[1] != C:
            raise ValueError(
                "Sources array column count does not match number of seeds."
            )
        return torch.from_numpy(sources_arr[:nV]).to(device=device, dtype=dtype)

    rhs = torch.zeros((nV, C), device=device, dtype=dtype)
    for c, seed in enumerate(seeds):
        if seed < 0 or seed >= nV:
            raise ValueError(f"Seed index {seed} out of range for mesh (nV={nV}).")
        rhs[seed, c] = model.M_diag[seed]
    return rhs


def _unique_sorted(values: Iterable[int]) -> List[int]:
    uniq = sorted(set(int(v) for v in values))
    return uniq


@torch.no_grad()
def integrate_piecewise_phi(
    model: VoronoiHeatModel,
    X_face: torch.Tensor,
    labels: np.ndarray,
    seeds: List[int],
    *,
    scale_gradients: bool = True,
    cg_tol: float = 1.0e-8,
    cg_maxiter: int = 2000,
) -> torch.Tensor:
    """Least-squares Poisson integration per Voronoi region (Option B).

    Each channel solves ``min_phi ∑_f A_f ||∇φ - X||^2`` over faces belonging to
    that label, anchoring the solution at the corresponding seed. This matches
    Heat-Method step III applied per region.
    """

    V = model.V
    F = model.F
    A_f = model.A_f
    gI = model.gI
    gJ = model.gJ
    gK = model.gK
    dtype = V.dtype
    device = V.device

    nV = V.shape[0]
    C = len(seeds)

    labels_np = np.asarray(labels, dtype=np.int64)
    if labels_np.shape[0] != nV:
        raise ValueError("Label array must have one entry per vertex.")

    label_tensor = torch.as_tensor(labels_np, device=device, dtype=torch.long)
    label_values = _unique_sorted(val for val in labels_np if val >= 0)
    if not label_values:
        raise ValueError("No valid labels found in input VTP.")
    if len(label_values) != C:
        raise ValueError(
            "Number of unique labels does not match number of seeds. "
            f"labels={label_values}, seeds={len(seeds)}"
        )

    phi = torch.full((nV, C), float("nan"), dtype=dtype, device=device)

    G_faces = torch.stack([gI, gJ, gK], dim=1)  # (nF, 3, 3)
    gram_faces = torch.matmul(G_faces, G_faces.transpose(1, 2)) * A_f[:, None, None]

    def _solve_spd(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        try:
            chol = torch.linalg.cholesky(A)
            return torch.cholesky_solve(b.unsqueeze(1), chol).squeeze(1)
        except RuntimeError:
            x = torch.zeros_like(b)
            r = b - A @ x
            if r.norm() <= cg_tol:
                return x
            Minv = 1.0 / A.diag().clamp_min(1.0e-12)
            z = Minv * r
            p = z.clone()
            rz_old = (r * z).sum()
            for _ in range(cg_maxiter):
                Ap = A @ p
                denom = (p * Ap).sum().clamp_min(1.0e-20)
                alpha = rz_old / denom
                x = x + alpha * p
                r = r - alpha * Ap
                if r.norm() <= cg_tol:
                    break
                z = Minv * r
                rz_new = (r * z).sum()
                beta = rz_new / rz_old.clamp_min(1.0e-20)
                p = z + beta * p
                rz_old = rz_new
            return x

    for c in range(C):
        region_mask = label_tensor == c
        if not torch.any(region_mask):
            continue

        face_mask = (
            region_mask[F[:, 0]]
            & region_mask[F[:, 1]]
            & region_mask[F[:, 2]]
        )
        face_indices = torch.nonzero(face_mask, as_tuple=False).flatten()

        if face_indices.numel() == 0:
            verts_only = torch.nonzero(region_mask, as_tuple=False).flatten()
            if verts_only.numel() == 0:
                continue
            phi[verts_only, c] = torch.zeros(verts_only.numel(), dtype=dtype, device=device)
            continue

        verts = torch.unique(F[face_indices].reshape(-1))
        local_map = torch.full((nV,), -1, dtype=torch.long, device=device)
        local_map[verts] = torch.arange(verts.numel(), device=device)

        n_local = verts.numel()
        A_loc = torch.zeros((n_local, n_local), dtype=dtype, device=device)
        b_loc = torch.zeros(n_local, dtype=dtype, device=device)

        faces_subset = F[face_indices]
        gram_subset = gram_faces[face_indices]
        X_subset = X_face[face_indices, c]
        A_weights = A_f[face_indices]

        proj_subset = torch.matmul(G_faces[face_indices], X_subset.unsqueeze(2)).squeeze(2)
        proj_subset = proj_subset * A_weights[:, None]

        for idx, tri in enumerate(faces_subset):
            loc = local_map[tri]
            K_face = gram_subset[idx]
            B_face = proj_subset[idx]
            for a in range(3):
                ia = int(loc[a].item())
                b_loc[ia] += B_face[a]
                Ka = K_face[a]
                for b_idx in range(3):
                    jb = int(loc[b_idx].item())
                    A_loc[ia, jb] += Ka[b_idx]

        A_loc += torch.eye(n_local, dtype=dtype, device=device) * 1.0e-12

        seed_idx = int(seeds[c])
        if seed_idx < 0 or seed_idx >= nV:
            raise ValueError(f"Seed {seed_idx} out of bounds for region {c}.")
        if local_map[seed_idx] < 0:
            region_vertices = torch.nonzero(region_mask, as_tuple=False).flatten()
            if region_vertices.numel() == 0:
                continue
            local_seed = int(local_map[region_vertices[0]].item())
        else:
            local_seed = int(local_map[seed_idx].item())

        A_loc[local_seed, :] = 0.0
        A_loc[:, local_seed] = 0.0
        A_loc[local_seed, local_seed] = 1.0
        b_loc[local_seed] = 0.0

        phi_local = _solve_spd(A_loc, b_loc)

        if scale_gradients and face_indices.numel() > 0:
            grads = []
            for idx, tri in enumerate(faces_subset):
                loc = local_map[tri]
                grad_vec = (
                    phi_local[int(loc[0])] * gI[face_indices[idx]]
                    + phi_local[int(loc[1])] * gJ[face_indices[idx]]
                    + phi_local[int(loc[2])] * gK[face_indices[idx]]
                )
                grads.append(torch.linalg.norm(grad_vec))
            if grads:
                grad_tensor = torch.stack(grads)
                median_norm = torch.median(grad_tensor).clamp_min(1.0e-8)
                phi_local = phi_local / median_norm

        phi[verts, c] = phi_local

    return phi


def _write_vtp_with_phi(
    poly: vtk.vtkPolyData,
    output_path: Path,
    phi: np.ndarray,
    array_name: str,
    labels: Optional[np.ndarray] = None,
    cut_edges: Optional[np.ndarray] = None,
) -> None:
    """Attach potentials (and optional labels/seam edges) and write PolyData."""

    total_points = poly.GetNumberOfPoints()
    n_phi = phi.shape[0]
    if n_phi > total_points:
        raise ValueError("phi has more rows than points in the polydata.")

    phi_ext = np.full((total_points, phi.shape[1]), np.nan, dtype=np.float32)
    phi_ext[:n_phi, :] = phi.astype(np.float32, copy=False)

    vtk_arr = vtk_np.numpy_to_vtk(phi_ext, deep=True)
    vtk_arr.SetNumberOfComponents(phi.shape[1])
    vtk_arr.SetName(array_name)
    poly.GetPointData().AddArray(vtk_arr)

    if labels is not None:
        label_ext = np.full(total_points, -1, dtype=np.int32)
        label_ext[: min(n_phi, labels.shape[0])] = labels[: min(n_phi, labels.shape[0])].astype(np.int32, copy=False)
        lbl_arr = vtk_np.numpy_to_vtk(label_ext, deep=True)
        lbl_arr.SetName("labels")
        poly.GetPointData().AddArray(lbl_arr)

    if cut_edges is not None and cut_edges.size > 0:
        existing_lines = poly.GetLines()
        new_lines = vtk.vtkCellArray()
        if existing_lines is not None and existing_lines.GetNumberOfCells() > 0:
            new_lines.DeepCopy(existing_lines)
        else:
            new_lines.Initialize()
        cut_edges = np.asarray(cut_edges, dtype=np.int64)
        for u, v in cut_edges:
            line = vtk.vtkIdList()
            line.SetNumberOfIds(2)
            line.SetId(0, int(u))
            line.SetId(1, int(v))
            new_lines.InsertNextCell(line)
        poly.SetLines(new_lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(poly)
    writer.Write()


def _export_npz(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    phi: np.ndarray,
    scores: np.ndarray,
    heat: np.ndarray,
    cut_edges: Optional[np.ndarray] = None,
) -> None:
    np.savez(
        path,
        vertices=vertices.astype(np.float64, copy=True),
        faces=faces.astype(np.int64, copy=True),
        labels=labels.astype(np.float32, copy=True),
        phi=phi.astype(np.float32, copy=True),
        scores=scores.astype(np.float32, copy=True),
        heat=heat.astype(np.float32, copy=True),
        hinge_segments=np.array([], dtype=object),
        cut_boundary_edges=(cut_edges.astype(np.int64, copy=True) if cut_edges is not None else np.zeros((0, 2), dtype=np.int64)),
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate piecewise Voronoi potentials from an exported VTP."
    )
    parser.add_argument("--input", required=True, help="Path to voronoi.vtp file.")
    parser.add_argument("--seeds", required=True, help="Path to seeds.txt (one index per line).")
    parser.add_argument("--out-vtp", default=None, help="Output VTP path (default: input stem + '_phi.vtp').")
    parser.add_argument("--out-npz", default=None, help="Optional NPZ export for viz_tool.")
    parser.add_argument("--array-name", default="piecewise_phi", help="Point-data array name for the potentials.")
    parser.add_argument("--device", default="cpu", help="Torch device string (cpu/cuda/mps).")
    parser.add_argument("--t", type=float, default=None, help="Optional diffusion time override.")
    parser.add_argument("--cg-tol", type=float, default=1.0e-6, help="Tolerance for CG heat solve.")
    parser.add_argument("--cg-iters", type=int, default=500, help="Max iterations for heat solve CG.")
    parser.add_argument(
        "--outside-value",
        default="nan",
        help="Fill value outside each Voronoi region (number or 'nan').",
    )
    parser.add_argument(
        "--no-grad-scale",
        action="store_true",
        help="Disable per-region gradient median normalisation.",
    )
    parser.add_argument(
        "--cut-by-labels",
        action="store_true",
        help="Duplicate vertices along label boundaries and integrate each region independently.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> IntegrationResult:
    input_path = Path(args.input)
    seeds_path = Path(args.seeds)
    if not input_path.exists():
        raise FileNotFoundError(f"Input VTP '{input_path}' does not exist.")
    if not seeds_path.exists():
        raise FileNotFoundError(f"Seeds file '{seeds_path}' does not exist.")

    poly, vertices_all, faces, point_arrays = _load_vtp(input_path)
    if faces.size == 0:
        raise ValueError("Input VTP does not contain any triangle faces to integrate over.")

    original_vertex_count = int(faces.max()) + 1 if faces.size else vertices_all.shape[0]
    vertices = vertices_all[:original_vertex_count]

    labels_arr = point_arrays.get("labels")
    if labels_arr is None:
        raise ValueError("Input VTP missing 'labels' point array.")
    labels_initial = labels_arr[:original_vertex_count].astype(np.int64, copy=False)

    seeds_original = read_seeds(seeds_path)
    device = torch.device(args.device)

    outside_arg = args.outside_value
    if isinstance(outside_arg, str):
        outside_value = float("nan") if outside_arg.lower() == "nan" else float(outside_arg)
    else:
        outside_value = float(outside_arg)

    if args.cut_by_labels:
        vertices_cut, faces_cut, _, vertex_map = cut_mesh_by_labels(vertices, faces, labels_initial)
        seed_map: List[int] = []
        for seed in seeds_original:
            lbl = int(labels_initial[seed])
            key = (int(seed), lbl)
            if key not in vertex_map:
                raise ValueError(f"Seed {seed} (label {lbl}) not present in cut mesh.")
            seed_map.append(int(vertex_map[key]))
        model = VoronoiHeatModel(vertices_cut, faces_cut, t=args.t, device=device)
        B_heat = _build_dirac_rhs(model, seed_map, None)
        seeds_used = seed_map
        poly = _build_polydata(vertices_cut, faces_cut)
        cut_edges = boundary_edges(faces_cut)
    else:
        model = VoronoiHeatModel(vertices, faces, t=args.t, device=device)
        sources_arr = point_arrays.get("sources")
        B_heat = _build_dirac_rhs(model, seeds_original, sources_arr)
        seeds_used = [int(s) for s in seeds_original]
        cut_edges = None

    U, X, S = model(B_heat, cg_tol=args.cg_tol, cg_iters=args.cg_iters)
    labels_tensor = argmin_labels(S)
    labels_np = labels_tensor.detach().cpu().numpy().astype(np.int64, copy=False)

    phi_tensor = integrate_piecewise_phi(
        model,
        X,
        labels_np,
        seeds_used,
        scale_gradients=not args.no_grad_scale,
    )

    phi_np = phi_tensor.detach().cpu().numpy()
    if np.isnan(outside_value):
        for c in range(phi_np.shape[1]):
            phi_np[labels_np != c, c] = np.nan
    else:
        fill = float(outside_value)
        for c in range(phi_np.shape[1]):
            phi_np[labels_np != c, c] = fill

    heat_np = U.detach().cpu().numpy()
    scores_np = S.detach().cpu().numpy()
    vertices_np = model.V.detach().cpu().numpy()
    faces_np = model.F.detach().cpu().numpy()

    out_vtp = Path(args.out_vtp) if args.out_vtp else input_path.with_name(input_path.stem + "_phi.vtp")
    _write_vtp_with_phi(poly, out_vtp, phi_np, args.array_name, labels=labels_np, cut_edges=cut_edges)

    if args.out_npz:
        out_npz = Path(args.out_npz)
        _export_npz(out_npz, vertices_np, faces_np, labels_np, phi_np, scores_np, heat_np, cut_edges=cut_edges)

    return IntegrationResult(phi=phi_np, heat=heat_np, scores=scores_np)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    result = run(args)
    print(
        "Integrated potentials for",
        result.phi.shape[1],
        "regions. Output stats: phi min=%.4f max=%.4f" % (np.nanmin(result.phi), np.nanmax(result.phi)),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
