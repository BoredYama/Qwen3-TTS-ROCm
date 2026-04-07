# coding=utf-8
# GPU-agnostic device utilities for Qwen3-TTS (NVIDIA CUDA / AMD ROCm / Apple MPS / CPU).
#
# ROCm exposes AMD GPUs through the HIP runtime, which PyTorch maps to the
# "cuda" device backend.  This means `torch.cuda.is_available()` returns True
# on both NVIDIA and AMD systems when the appropriate driver stack is installed.

import torch


def get_device() -> str:
    """Return the best available device string.

    Priority:
        1. ``cuda:0``  – works for both NVIDIA CUDA and AMD ROCm (via HIP).
        2. ``mps:0``   – Apple Metal Performance Shaders.
        3. ``cpu``      – fallback.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps:0"
    return "cpu"


def synchronize_device(device=None):
    """Synchronize the given device (or the current CUDA/ROCm device).

    This is a no-op when the device is ``cpu`` or ``mps``.
    """
    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        if dev.type == "cpu":
            return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
