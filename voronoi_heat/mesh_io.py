"""
Mesh and artifact I/O utilities for the Voronoi heat pipeline.
Moved out of pipeline.py to keep orchestration code minimal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from seamfit.io import load_mesh as seamfit_load_mesh, _require_vtk, vtk, vtk_np


def load_obj_tri(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ reader supporting ``v`` and triangular ``f`` records."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, xs, ys, zs = line.strip().split()[:4]
                verts.append([float(xs), float(ys), float(zs)])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idxs: List[int] = []
                for token in parts:
                    idx = token.split("/")[0]
                    if not idx:
                        raise ValueError(f"Malformed face token '{token}' in {path}.")
                    idxs.append(int(idx) - 1)
                if len(idxs) < 3:
                    continue
                v0 = idxs[0]
                for a, b in zip(idxs[1:-1], idxs[2:]):
                    faces.append([v0, a, b])
    if not verts or not faces:
        raise ValueError(f"OBJ file {path} did not contain vertices/faces.")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _load_vtk_triangular(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load generic VTK/VTP meshes (polydata or unstructured) and triangulate."""
    _require_vtk()
    suffix = Path(path).suffix.lower()
    if suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        dataset = reader.GetOutput()
    elif suffix == ".vtk":
        generic = vtk.vtkGenericDataObjectReader()
        generic.SetFileName(str(path))
        generic.Update()
        dataset = generic.GetOutput()
    else:
        raise ValueError(f"Unsupported mesh extension '{suffix}' for VTK loader.")
    if dataset is None:
        raise ValueError(f"VTK reader produced no dataset for {path}.")
    if not isinstance(dataset, vtk.vtkPolyData):
        geom = vtk.vtkGeometryFilter()
        geom.SetInputData(dataset)
        geom.Update()
        dataset = geom.GetOutput()
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(dataset)
    tri_filter.PassLinesOff()
    tri_filter.PassVertsOff()
    tri_filter.Update()
    poly_tri = tri_filter.GetOutput()
    points = poly_tri.GetPoints()
    if points is None:
        raise ValueError(f"No point data found in {path}.")
    vertices_np = vtk_np.vtk_to_numpy(points.GetData()).astype(np.float64, copy=False)
    polys = poly_tri.GetPolys()
    cell_data = vtk_np.vtk_to_numpy(polys.GetData()).astype(np.int64, copy=False)
    n_cells = polys.GetNumberOfCells()
    if n_cells == 0:
        raise ValueError(f"No polygon data found in {path}.")
    cell_width = int(cell_data.size // n_cells)
    if cell_width != 4:
        raise ValueError("Triangulation produced unexpected face encoding.")
    faces_np = cell_data.reshape(n_cells, cell_width)[:, 1:]
    return vertices_np, faces_np


def load_mesh_any(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load mesh supporting OBJ (text) and VTK/VTP via seamfit."""
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        return load_obj_tri(path)
    if suffix in {".vtp", ".vtk"}:
        try:
            V_t, F_t, _ = seamfit_load_mesh(path)
        except Exception:
            V_np, F_np = _load_vtk_triangular(path)
            return V_np, F_np
        return V_t.cpu().numpy().astype(np.float64), F_t.cpu().numpy().astype(np.int64)
    raise ValueError(f"Unsupported mesh extension '{suffix}' (expected .obj/.vtp/.vtk).")


def preprocess_mesh(V_np: np.ndarray, F_np: np.ndarray, *, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Remove degenerate faces and zero-area slivers from the mesh."""
    V = torch.from_numpy(V_np.astype(np.float64, copy=False))
    F = torch.from_numpy(F_np.astype(np.int64, copy=False))
    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    distinct_mask = (i != j) & (j != k) & (k != i)
    vi = V[F[:, 0]]
    vj = V[F[:, 1]]
    vk = V[F[:, 2]]
    n = torch.cross(vj - vi, vk - vi, dim=1)
    areas = 0.5 * torch.linalg.norm(n, dim=1)
    area_eps = float(max(1.0, areas.mean().item()) * 1.0e-16)
    area_mask = areas >= area_eps
    keep_mask = distinct_mask & area_mask
    if keep_mask.all():
        if verbose:
            print("Mesh preprocessing: no degenerate faces detected.")
        return V_np, F_np
    kept = keep_mask.nonzero(as_tuple=False).reshape(-1)
    F_clean = F[kept].cpu().numpy()
    if verbose:
        dropped = F_np.shape[0] - F_clean.shape[0]
        print(
            f"Mesh preprocessing: removed {dropped} degenerate faces "
            f"(distinct={(~distinct_mask).sum().item()}, tiny={(~area_mask).sum().item()})."
        )
    return V_np, F_clean


def read_seeds(path: str | Path) -> List[int]:
    seeds: List[int] = []
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(int(line))
    if not seeds:
        raise ValueError(f"No seeds found in {path}.")
    return seeds


def write_segments_obj(path: str | Path, segments: List[np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        fh.write("# Voronoi segments\n")
        vid = 1
        for poly in segments:
            poly_arr = np.asarray(poly, dtype=np.float64)
            if poly_arr.ndim != 2 or poly_arr.shape[0] < 2:
                continue
            for i in range(poly_arr.shape[0] - 1):
                p0 = poly_arr[i]
                p1 = poly_arr[i + 1]
                fh.write(f"v {p0[0]} {p0[1]} {p0[2]}\n")
                fh.write(f"v {p1[0]} {p1[1]} {p1[2]}\n")
                fh.write(f"l {vid} {vid + 1}\n")
                vid += 2


def write_voronoi_vtp(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: Optional[np.ndarray],
    segments: List[np.ndarray],
    scores: Optional[np.ndarray] = None,
    heat: Optional[np.ndarray] = None,
    segment_faces: Optional[np.ndarray] = None,
    segment_bary: Optional[np.ndarray] = None,
    sources: Optional[np.ndarray] = None,
) -> None:
    """Write mesh, labels, and polyline segments into a single VTP file."""
    import vtk  # type: ignore
    from vtk.util import numpy_support as vtk_np  # type: ignore

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    polyline_points: List[np.ndarray] = []
    for poly in segments:
        arr = np.asarray(poly, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        polyline_points.append(arr)

    if polyline_points:
        stacked_points = np.vstack(polyline_points)
    else:
        stacked_points = np.zeros((0, 3), dtype=np.float64)

    start_idx = vertices.shape[0]
    all_points = vertices if stacked_points.size == 0 else np.vstack([vertices, stacked_points])

    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_np.numpy_to_vtk(all_points, deep=True))
    poly.SetPoints(vtk_points)

    face_cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    vtk_faces = vtk.vtkCellArray()
    vtk_faces.SetCells(faces.shape[0], vtk_np.numpy_to_vtk(face_cells, deep=True, array_type=vtk.VTK_ID_TYPE))
    poly.SetPolys(vtk_faces)

    if polyline_points:
        vtk_lines = vtk.vtkCellArray()
        offset = start_idx
        for arr in polyline_points:
            count = arr.shape[0]
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(count)
            for i in range(count):
                polyline.GetPointIds().SetId(i, offset + i)
            vtk_lines.InsertNextCell(polyline)
            offset += count
        poly.SetLines(vtk_lines)

    def add_point_array(name: str, data: Optional[np.ndarray]) -> None:
        if data is None or (hasattr(data, "size") and data.size == 0):
            return
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr_ext = np.full((all_points.shape[0],), -1.0, dtype=np.float32)
            arr_ext[: arr.shape[0]] = arr.astype(np.float32, copy=False)
            vtk_arr = vtk_np.numpy_to_vtk(arr_ext, deep=True)
        else:
            arr_ext = np.zeros((all_points.shape[0], arr.shape[1]), dtype=np.float32)
            arr_ext[: arr.shape[0], :] = arr.astype(np.float32, copy=False)
            vtk_arr = vtk_np.numpy_to_vtk(arr_ext, deep=True)
            vtk_arr.SetNumberOfComponents(arr.shape[1])
        vtk_arr.SetName(name)
        poly.GetPointData().AddArray(vtk_arr)

    add_point_array("labels", labels)
    add_point_array("scores", scores)
    add_point_array("heat", heat)
    add_point_array("sources", sources)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()


def write_viz_npz(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    segments: List[Tuple[np.ndarray, np.ndarray]],
    scores: Optional[np.ndarray] = None,
    heat: Optional[np.ndarray] = None,
    segment_faces: Optional[np.ndarray] = None,
    segment_bary: Optional[np.ndarray] = None,
    sources: Optional[np.ndarray] = None,
) -> None:
    arrays: Dict[str, np.ndarray] = {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.float32),
    }
    if segments:
        seg_list = [np.asarray(poly, dtype=np.float64) for poly in segments if np.asarray(poly).ndim == 2]
        arrays["hinge_segments"] = np.array(seg_list, dtype=object)
    if scores is not None and getattr(scores, "size", 0) > 0:
        arrays["scores"] = np.asarray(scores, dtype=np.float32)
    if heat is not None and getattr(heat, "size", 0) > 0:
        arrays["heat"] = np.asarray(heat, dtype=np.float32)
    if segment_faces is not None and getattr(segment_faces, "size", 0) > 0:
        arrays["hinge_segments_faces"] = np.asarray(segment_faces, dtype=np.int64)
    if segment_bary is not None and getattr(segment_bary, "size", 0) > 0:
        arrays["hinge_segments_bary"] = np.asarray(segment_bary, dtype=np.float32)
    if sources is not None and getattr(sources, "size", 0) > 0:
        arrays["sources"] = np.asarray(sources, dtype=np.float32)
    np.savez(path, **arrays)


__all__ = [
    "load_obj_tri",
    "load_mesh_any",
    "preprocess_mesh",
    "read_seeds",
    "write_segments_obj",
    "write_voronoi_vtp",
    "write_viz_npz",
]

