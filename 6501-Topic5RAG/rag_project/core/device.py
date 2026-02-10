"""
Device detection for compute backend selection.

Priority: CUDA > MPS (Apple Silicon) > CPU
"""

import os
from typing import Tuple

import torch

# Must be set before any torch ops on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def detect_device() -> Tuple[str, torch.dtype]:
    """
    Detect the best available compute device.

    Returns:
        (device_string, recommended_dtype)
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\u2713 CUDA GPU: {name} ({mem:.1f} GB)")
        return "cuda", torch.float16

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("\u2713 Apple Silicon GPU (MPS) — using float32")
        return "mps", torch.float32

    print("\u26a0 CPU only (no GPU detected)")
    return "cpu", torch.float32
