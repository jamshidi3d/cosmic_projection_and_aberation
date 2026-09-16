"""CPU/GPU array-backend selection.

The geometry / deformation / overlap code is written once against a generic array
module ``xp`` (either :mod:`numpy` or :mod:`cupy`). This module resolves which one to
use and provides the few operations whose spelling differs between them.

``cupy`` is an optional dependency; nothing here imports it at module load.
"""

from __future__ import annotations

import numpy as np

_CUPY = None
_CUPY_TRIED = False


def _cupy():
    """Return the imported ``cupy`` module, or ``None`` if unavailable."""
    global _CUPY, _CUPY_TRIED
    if not _CUPY_TRIED:
        _CUPY_TRIED = True
        try:
            import cupy as cp  # noqa: F401

            _CUPY = cp
        except Exception:
            _CUPY = None
    return _CUPY


def gpu_available():
    """True if ``cupy`` imports and at least one CUDA device is visible."""
    cp = _cupy()
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def resolve_backend(name):
    """Map ``"cpu" | "gpu" | "auto"`` to a concrete ``"cpu" | "gpu"``."""
    if name not in ("cpu", "gpu", "auto"):
        raise ValueError("backend must be 'cpu', 'gpu' or 'auto'")
    if name == "auto":
        return "gpu" if gpu_available() else "cpu"
    if name == "gpu" and not gpu_available():
        raise RuntimeError(
            "backend='gpu' requested but cupy / a CUDA device is not available")
    return name


def get_xp(backend):
    """Return the array module for a resolved backend name."""
    return _cupy() if backend == "gpu" else np


def array_module(*arrays):
    """Return :mod:`cupy` if any argument is a cupy array, else :mod:`numpy`."""
    cp = _cupy()
    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):
                return cp
    return np


def to_device(a, xp, dtype=None):
    """Move / cast ``a`` onto the array module ``xp``."""
    if dtype is None:
        return xp.asarray(a)
    return xp.asarray(a, dtype=dtype)


def to_host(a):
    """Return ``a`` as a numpy array (no-op for numpy input)."""
    cp = _cupy()
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def scatter_add(target, idx, values):
    """In-place ``target[idx] += values`` with repeated indices accumulated."""
    cp = _cupy()
    if cp is not None and isinstance(target, cp.ndarray):
        cp.add.at(target, idx, values)
    else:
        np.add.at(target, idx, values)
    return target


def bincount(idx, weights, minlength, xp):
    """``xp.bincount`` with a float weight array (numpy and cupy agree here)."""
    return xp.bincount(idx, weights=weights, minlength=minlength)
