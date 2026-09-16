"""Thin wrappers over healpy -- the only module that imports it directly."""

from __future__ import annotations

import numpy as np
import healpy as hp


def infer_nside(m):
    """``nside`` of a HEALPix map array (uses the trailing axis length)."""
    return hp.npix2nside(np.asarray(m).shape[-1])


def as_ring(m, nest):
    """Return ``m`` in RING ordering given its current ordering (``nest``)."""
    return hp.reorder(m, n2r=True) if nest else np.asarray(m)


def as_nest(m, nest):
    """Return ``m`` in NEST ordering given its current ordering (``nest``)."""
    return np.asarray(m) if nest else hp.reorder(m, r2n=True)


def pixarea(nside):
    return hp.nside2pixarea(nside)


def resol(nside):
    return hp.nside2resol(nside)


def pixel_boundaries(nside, pix, step=1, nest=False):
    """Boundary vectors of pixels, shape ``(len(pix), 4 * step, 3)``."""
    pix = np.atleast_1d(pix)
    b = hp.boundaries(nside, pix, step=step, nest=nest)   # (len(pix), 3, 4*step)
    return np.moveaxis(b, -1, -2)


def _check_pow2(name, value):
    if value < 1 or (value & (value - 1)) != 0:
        raise ValueError(f"{name} must be a power of 2, got {value}")


def subpixel_centers(nside, factor, pix=None):
    """Centres of NEST sub-pixels at ``nside * factor``.

    ``pix`` (interpreted as **NEST** indices at ``nside``) selects the parent pixels;
    ``None`` covers the whole sphere. Returns ``(vecs (K, 3), parent_nest (K,))`` with
    ``K = len(parents) * factor**2``, sub-pixels grouped by parent.

    ``factor`` must be a power of 2 so that the NEST children of parent ``p`` are
    exactly ``p * factor**2 + [0 .. factor**2)``.
    """
    _check_pow2("factor", factor)
    nside_sub = nside * factor
    ratio = factor * factor
    parents = (np.arange(hp.nside2npix(nside)) if pix is None
               else np.atleast_1d(pix).astype(np.int64))
    child = (parents[:, None] * ratio + np.arange(ratio)[None, :]).reshape(-1)
    x, y, z = hp.pix2vec(nside_sub, child, nest=True)
    return np.stack([x, y, z], axis=-1), np.repeat(parents, ratio)
