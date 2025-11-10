"""
Schedules and logging utilities for the Voronoi heat pipeline.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch


@dataclass
class TempSchedule:
    start_beta: float = 6.0
    end_beta: float = 24.0
    start_kappa: float = 6.0
    end_kappa: float = 40.0
    start_srcT: float = 1.5
    end_srcT: float = 1.0
    total_steps: int = 1000
    mode: str = "linear"  # or "cosine"

    def at(self, step: int) -> Tuple[float, float, float]:
        if self.total_steps <= 0:
            return self.end_beta, self.end_kappa, self.end_srcT
        s = min(max(step, 0), self.total_steps)
        if self.mode == "cosine":
            w = 0.5 * (1.0 - math.cos(math.pi * s / self.total_steps))
        else:
            w = s / self.total_steps
        beta = self.start_beta + w * (self.end_beta - self.start_beta)
        kappa = self.start_kappa + w * (self.end_kappa - self.start_kappa)
        srcT = self.start_srcT + w * (self.end_srcT - self.start_srcT)
        return float(beta), float(kappa), float(srcT)


class Logger:
    """CSV + optional TensorBoard logging for training metrics."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        filename: str = "train_log.csv",
        enable_tb: bool = True,
    ) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = Path(out_dir) / filename
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf8")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(
            [
                "step",
                "loss",
                "seam",
                "jump",
                "normal",
                "tan",
                "eq",
                "margin",
                "poly",
                "smooth",
                "area",
                "align",
                "lr",
                "beta",
                "kappa",
                "srcT",
                "grad_norm",
            ]
        )
        self.csv_file.flush()
        self.tb = None
        if enable_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(Path(out_dir) / "tb"))
            except Exception:
                self.tb = None

    def log(self, step: int, scalars: Dict[str, float]) -> None:
        row = [
            step,
            scalars["loss"],
            scalars.get("seam", 0.0),
            scalars["jump"],
            scalars["normal"],
            scalars["tan"],
            scalars.get("eq", 0.0),
            scalars.get("margin", 0.0),
            scalars.get("poly", 0.0),
            scalars.get("smooth", 0.0),
            scalars.get("area", 0.0),
            scalars.get("align", 0.0),
            scalars["lr"],
            scalars["beta"],
            scalars["kappa"],
            scalars["srcT"],
            scalars["grad_norm"],
        ]
        self.writer.writerow(row)
        self.csv_file.flush()
        if self.tb:
            for key, value in scalars.items():
                self.tb.add_scalar(key, value, step)

    def close(self) -> None:
        if self.tb:
            self.tb.close()
        self.csv_file.close()


def save_checkpoint(path: str | Path, state: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path) -> Dict:
    return torch.load(path, map_location="cpu")


__all__ = [
    "TempSchedule",
    "Logger",
    "save_checkpoint",
    "load_checkpoint",
]

