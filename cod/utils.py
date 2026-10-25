from __future__ import annotations

import json
import os
import platform
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def get_device(preference: str = "auto") -> torch.device:
    """Pick cuda > mps > cpu unless a specific device is requested."""
    if preference != "auto":
        dev = torch.device(preference)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if dev.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def timer(device: torch.device, out: dict, key: str):
    sync(device)
    t = time.perf_counter()
    yield
    sync(device)
    out[key] = (time.perf_counter() - t) * 1000.0


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20  # MPS has no peak counter
    return None


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def dump_json(obj, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
