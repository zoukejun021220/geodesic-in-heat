"""
Thin facade: re-export high-level orchestrators to keep pipeline minimal.

This module intentionally defines no logic; it only exposes the public API
by importing from the dedicated modules that implement the details.
"""

from __future__ import annotations

from .trainer import (
    TrainConfig,
    train,
    infer_and_export,
    heat_distance_baseline,
)

__all__ = [
    "TrainConfig",
    "train",
    "infer_and_export",
    "heat_distance_baseline",
]

