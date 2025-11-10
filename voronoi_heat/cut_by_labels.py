"""Utilities for cutting meshes along Voronoi labels."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import numpy as np


def cut_mesh_by_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], int]]:
    """Duplicate vertices so each face is label-homogeneous.

    Parameters
    ----------
    vertices : (nV, 3) ndarray
        Input mesh vertices.
    faces : (nF, 3) ndarray
        Triangle indices.
    vertex_labels : (nV,) ndarray
        Per-vertex Voronoi labels, e.g. ``argmin(phi)``.

    Returns
    -------
    vertices_cut, faces_cut, face_labels, vertex_map
        New geometry and a mapping ``(vertex, label) -> new index``.
    """

    vertex_labels = np.asarray(vertex_labels, dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertex_labels.shape[0] == 0:
        return vertices.copy(), faces.copy(), np.zeros(faces.shape[0], dtype=np.int64), {}
    if faces.size == 0:
        return vertices.copy(), faces.copy(), np.zeros(0, dtype=np.int64), {}

    face_labels = np.array(
        [Counter(vertex_labels[tri]).most_common(1)[0][0] for tri in faces],
        dtype=np.int64,
    )

    vertices_cut: list[np.ndarray] = []
    faces_cut = np.empty_like(faces)
    vertex_map: Dict[Tuple[int, int], int] = {}

    for face_id, tri in enumerate(faces):
        label = int(face_labels[face_id])
        tri_new: list[int] = []
        for v in tri:
            key = (int(v), label)
            if key not in vertex_map:
                vertex_map[key] = len(vertices_cut)
                vertices_cut.append(vertices[int(v)])
            tri_new.append(vertex_map[key])
        faces_cut[face_id] = tri_new

    return (
        np.asarray(vertices_cut, dtype=vertices.dtype),
        faces_cut,
        face_labels,
        vertex_map,
    )


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Return edges with a single adjacent face."""

    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    ).astype(np.int64, copy=False)
    edges.sort(axis=1)
    edges = np.ascontiguousarray(edges)
    structured = edges.view([("u", edges.dtype), ("v", edges.dtype)])
    unique, counts = np.unique(structured, return_counts=True)
    boundary = unique[counts == 1]
    if boundary.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((boundary["u"], boundary["v"])).astype(np.int64, copy=False)
