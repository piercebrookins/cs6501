from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


DeviceType = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class DeviceConfig:
    device: DeviceType
    dtype: "object"  # torch.dtype, but avoid import at module import time
    environment: Literal["colab", "local"]


def enable_mps_fallback() -> None:
    """Enable MPS fallback for unsupported ops (must be set before torch import)."""

    # Idempotent; safe to call multiple times.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def detect_environment() -> Literal["colab", "local"]:
    """Detect if we're on Colab or local."""

    try:
        import google.colab  # type: ignore

        return "colab"
    except Exception:
        return "local"


def get_device_config() -> DeviceConfig:
    """Pick the best device and dtype.

    Mirrors the logic in the provided notebook:
    - CUDA: float16
    - MPS: float32 (often faster than float16 on Apple Silicon)
    - CPU: float32
    """

    enable_mps_fallback()

    import torch

    env = detect_environment()

    if torch.cuda.is_available():
        device: DeviceType = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    return DeviceConfig(device=device, dtype=dtype, environment=env)
