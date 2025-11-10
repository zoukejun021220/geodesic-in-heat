"""
Snapshot utilities for stage transitions, kept separate from pipeline orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from .voronoi_heat_torch import VoronoiHeatModel


@torch.no_grad()
def _seam_faces_from_labels(faces: Tensor, vertex_labels: Tensor) -> Tensor:
    face_labels = vertex_labels[faces]  # (nF, 3)
    same01 = face_labels[:, 0] == face_labels[:, 1]
    same12 = face_labels[:, 1] == face_labels[:, 2]
    same02 = face_labels[:, 0] == face_labels[:, 2]
    seam_mask = ~(same01 & same12 & same02)
    return torch.nonzero(seam_mask, as_tuple=False).squeeze(1)


@torch.no_grad()
def save_stage_snapshot(
    out_dir: Path,
    stage_name: str,
    model: VoronoiHeatModel,
    logits: Tensor,
) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    logits_raw = logits.detach().cpu().numpy()
    logits_vertex = logits_raw.T.astype(np.float32, copy=False)
    labels = torch.argmax(logits, dim=0)
    seam_idx = _seam_faces_from_labels(model.F, labels)
    np.save(out_dir / f"{stage_name}_logits.npy", logits_vertex)
    np.save(out_dir / f"{stage_name}_logits_raw.npy", logits_raw)
    np.save(out_dir / f"{stage_name}_labels.npy", labels.detach().cpu().numpy().astype(np.int32))
    np.save(out_dir / f"{stage_name}_seam_faces.npy", seam_idx.detach().cpu().numpy().astype(np.int64))
    return seam_idx.detach().cpu().numpy()


def write_seam_transition_report(out_dir: Path, warm_faces: np.ndarray, stage2_faces: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    warm_set = {int(x) for x in warm_faces.tolist()}
    stage2_set = {int(x) for x in stage2_faces.tolist()}
    removed = sorted(warm_set - stage2_set)
    added = sorted(stage2_set - warm_set)
    report_path = out_dir / "seam_transition_report.txt"
    with report_path.open("w", encoding="utf8") as fh:
        fh.write(f"warm_face_count={len(warm_set)}\n")
        fh.write(f"stage2_face_count={len(stage2_set)}\n")
        fh.write(f"removed_count={len(removed)}\n")
        fh.write(f"added_count={len(added)}\n")
        if removed:
            fh.write("removed_faces=" + " ".join(str(v) for v in removed) + "\n")
        if added:
            fh.write("added_faces=" + " ".join(str(v) for v in added) + "\n")
    np.save(out_dir / "seam_faces_removed.npy", np.asarray(removed, dtype=np.int64))
    np.save(out_dir / "seam_faces_added.npy", np.asarray(added, dtype=np.int64))


__all__ = ["save_stage_snapshot", "write_seam_transition_report"]

