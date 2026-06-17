"""Compute-device resolution with graceful CPU fallback.

The whole project is meant to run on a student laptop. GPU is used *only* if
torch reports CUDA available; otherwise everything falls back to CPU. The
result is cached so we probe torch at most once per process.
"""
from __future__ import annotations

import functools

import config
from core.logger import get_logger

log = get_logger("device")


@functools.lru_cache(maxsize=1)
def resolve_device() -> str:
    """Return the torch device string to use: ``"cpu"`` or ``"cuda:0"``.

    Honours ``config.DEVICE``:
      * ``auto``  -> cuda if available else cpu
      * ``cpu``   -> always cpu
      * ``cuda``/``0``/``1`` -> that gpu if available, else cpu (with a warning)
    """
    requested = str(config.DEVICE).strip().lower()

    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception as exc:  # torch missing or broken -> CPU
        log.warning("Could not query torch CUDA (%s); using CPU.", exc)
        return "cpu"

    if requested in {"cpu"}:
        device = "cpu"
    elif requested in {"auto", ""}:
        device = "cuda:0" if has_cuda else "cpu"
    else:
        # Explicit GPU request (e.g. "cuda", "cuda:1", "0").
        if has_cuda:
            idx = requested.replace("cuda:", "").replace("cuda", "").strip()
            device = f"cuda:{idx or '0'}"
        else:
            log.warning("DEVICE=%s requested but no CUDA GPU found; using CPU.",
                        requested)
            device = "cpu"

    log.info("Using compute device: %s", device)
    return device


def is_gpu() -> bool:
    """True if the resolved device is a CUDA GPU."""
    return resolve_device().startswith("cuda")
