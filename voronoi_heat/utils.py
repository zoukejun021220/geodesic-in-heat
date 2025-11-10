"""
Generic utilities (device resolution, seeding) for the pipeline.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    raw = "auto" if device is None else str(device).strip()
    lowered = raw.lower()
    if lowered in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if lowered.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available on this system.")
        return torch.device(raw)
    if lowered.startswith("mps"):
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available on this system.")
        return torch.device("mps")
    if lowered == "cpu":
        return torch.device("cpu")
    return torch.device(raw)


def device_to_str(device: torch.device) -> str:
    return f"{device.type}:{device.index}" if device.index is not None else device.type


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


__all__ = ["resolve_device", "device_to_str", "set_seed"]

