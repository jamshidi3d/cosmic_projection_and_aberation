"""Feature 1: resample a deformed spherical map onto a standard HEALPix grid.

Each source pixel is a polygon on the sphere; a :class:`~cosmic_projection.deformation.
Deformation` moves its vertices. Every output pixel receives an **area-weighted mean**
of whatever source-polygon fragments land inside it::

    out[j] = sum_i (value_i * overlap_area(i, j)) / sum_i overlap_area(i, j)

Two estimators of ``overlap_area`` are provided:

``method="subpixel"`` (default, fast)
    Each source pixel is split into ``factor**2`` equal-area NEST sub-pixels; their
    centres are pushed through the deformation and binned into output pixels. Converges
    to the true area weighting as ``factor`` grows.

``method="polygon"`` (slower, more accurate on coarse maps)
    Each source pixel boundary is deformed and clipped against every overlapping output
    pixel with :func:`~cosmic_projection.overlap.spherical_polygon_overlap_area`.
    Loops over source pixels -- intended for patches or moderate ``nside``, not
    full-sky at high resolution.
"""

from __future__ import annotations

import numpy as np
import healpy as hp

from . import _healpix as hx
from . import _healpix_gpu as hg
from ._backend import get_xp, resolve_backend, to_host
from ._overlap_gpu import batched_polygon_overlap
from .geometry import normalize
from .overlap import spherical_polygon_overlap_area


def resample_deformed_map(source_map, deformation, nside_out, *,
                          method="subpixel", nest=False, factor=8, step=3,
                          pixel_indices=None, fill=hp.UNSEEN,
                          return_weights=False, chunk=4096,
                          backend="auto", dtype="float64"):
    """Resample ``source_map`` under ``deformation`` onto an ``nside_out`` HEALPix grid.

    Parameters
    ----------
    source_map : array
        HEALPix map (``nside`` inferred from its length).
    deformation : Deformation
        Callable ``vecs -> vecs`` (see :mod:`cosmic_projection.deformation`).
    nside_out : int
        Target resolution.
    method : {"subpixel", "polygon"}
    nest : bool
        Ordering of ``source_map`` **and** of the returned map / ``pixel_indices``.
    factor : int
        ``method="subpixel"``: sub-pixels per side per source pixel. Must be a power
        of 2 (uses the HEALPix nested hierarchy); ``factor**2`` samples per pixel.
    step : int
        ``method="polygon"``: boundary samples per source-pixel edge.
    pixel_indices : array of int, optional
        Restrict the input to this subset of source pixels (patch resampling).
    fill : float
        Value for output pixels that receive no coverage.
    return_weights : bool
        Also return the accumulated coverage (effective solid angle) per output pixel.
    chunk : int
        source pixels processed per batch (memory control; both methods, both backends).
    backend : {"auto", "cpu", "gpu"}
        ``"gpu"`` uses the CuPy path (``cosmic_projection._healpix_gpu`` /
        ``._overlap_gpu``); ``"auto"`` picks GPU iff cupy + a CUDA device are present.
    dtype : {"float64", "float32"}
        GPU geometry precision. Accumulators are always float64. Ignored on CPU.

    Returns
    -------
    out : ndarray  (and ``weights`` if ``return_weights``)
    """
    source_map = np.asarray(source_map, dtype=float)
    nside_in = hx.infer_nside(source_map)
    src_nest = hx.as_nest(source_map, nest)
    npix_out = hp.nside2npix(nside_out)
    backend_r = resolve_backend(backend)

    if pixel_indices is None:
        parents = np.arange(hp.nside2npix(nside_in))
    else:
        parents = np.atleast_1d(pixel_indices).astype(np.int64)
        if not nest:
            parents = hp.ring2nest(nside_in, parents)

    if method not in ("subpixel", "polygon"):
        raise ValueError("method must be 'subpixel' or 'polygon'")

    if backend_r == "cpu":
        num = np.zeros(npix_out)
        den = np.zeros(npix_out)
        if method == "subpixel":
            _accumulate_subpixel(src_nest, deformation, nside_in, nside_out, parents,
                                 factor, chunk, num, den)
        else:
            _accumulate_polygon(src_nest, deformation, nside_in, nside_out, parents,
                                step, num, den)
    else:
        xp = get_xp(backend_r)
        geom_dtype = xp.float32 if dtype == "float32" else xp.float64
        num = xp.zeros(npix_out, dtype=xp.float64)
        den = xp.zeros(npix_out, dtype=xp.float64)
        if method == "subpixel":
            _accumulate_subpixel_gpu(src_nest, deformation, nside_in, nside_out,
                                     parents, factor, chunk, num, den, xp, geom_dtype)
        else:
            _accumulate_polygon_gpu(src_nest, deformation, nside_in, nside_out,
                                    parents, step, chunk, num, den, xp, geom_dtype)
        num, den = to_host(num), to_host(den)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), fill)

    if nest:                       # num/den are indexed by RING pixel; convert if asked
        out = hp.reorder(out, r2n=True)
        den = hp.reorder(den, r2n=True)
    return (out, den) if return_weights else out


def _accumulate_subpixel(src_nest, deformation, nside_in, nside_out, parents,
                         factor, chunk, num, den):
    w = hp.nside2pixarea(nside_in * factor)
    npix_out = num.shape[0]
    for s in range(0, len(parents), chunk):
        pchunk = parents[s:s + chunk]
        vecs, parent_rep = hx.subpixel_centers(nside_in, factor, pchunk)
        dv = deformation(vecs)
        tgt = hp.vec2pix(nside_out, dv[:, 0], dv[:, 1], dv[:, 2])   # RING
        vals = src_nest[parent_rep]
        num += np.bincount(tgt, weights=vals * w, minlength=npix_out)
        den += np.bincount(tgt, weights=np.full(tgt.shape, w), minlength=npix_out)


def _accumulate_polygon(src_nest, deformation, nside_in, nside_out, parents,
                        step, num, den):
    omega_out = hp.nside2pixarea(nside_out)
    for p in parents:
        bnd = hx.pixel_boundaries(nside_in, int(p), step=step, nest=True)[0]
        dpoly = deformation(bnd)
        centroid = normalize(dpoly.mean(axis=0))
        try:
            cand = hp.query_polygon(nside_out, dpoly, inclusive=True, fact=4)
        except Exception:
            radius = float(np.max(np.arccos(np.clip(dpoly @ centroid, -1.0, 1.0))))
            cand = hp.query_disc(nside_out, centroid, radius * 1.5, inclusive=True)
        if len(cand) == 0:
            continue
        val = src_nest[p]
        cb = np.moveaxis(hp.boundaries(nside_out, cand, step=1), -1, -2)  # (n,4,3)
        for k, cpix in enumerate(cand):
            area = spherical_polygon_overlap_area(dpoly, cb[k], ref_area_b=omega_out)
            if area > 0.0:
                num[cpix] += val * area
                den[cpix] += area


# --------------------------------------------------------------------------- #
# GPU (CuPy) accumulators                                                     #
# --------------------------------------------------------------------------- #

def _accumulate_subpixel_gpu(src_nest, deformation, nside_in, nside_out, parents,
                             factor, chunk, num, den, xp, dtype):
    w = float(hp.nside2pixarea(nside_in * factor))
    npix_out = int(num.shape[0])
    src_dev = xp.asarray(src_nest, dtype=xp.float64)
    for s in range(0, len(parents), chunk):
        pc = parents[s:s + chunk]
        vecs_host, parent_rep = hx.subpixel_centers(nside_in, factor, pc)
        dv = deformation(xp.asarray(vecs_host, dtype=dtype))
        tgt = hg.vec2pix_ring(nside_out, dv)
        vals = src_dev[xp.asarray(parent_rep)]
        num += xp.bincount(tgt, weights=vals * w, minlength=npix_out)
        den += xp.bincount(tgt, weights=xp.full(tgt.shape, w, dtype=xp.float64),
                           minlength=npix_out)


def _accumulate_polygon_gpu(src_nest, deformation, nside_in, nside_out, parents,
                            step, chunk, num, den, xp, dtype):
    npix_out = int(num.shape[0])
    omega_out = float(hp.nside2pixarea(nside_out))
    pixdiag = 1.5 * float(hp.nside2resol(nside_out))
    src_dev = xp.asarray(src_nest, dtype=xp.float64)

    # exact target-pixel corners from healpy, uploaded once (RING order)
    tgt_corners = xp.asarray(
        np.moveaxis(hp.boundaries(nside_out, np.arange(npix_out), step=1), -1, -2),
        dtype=dtype)

    for s in range(0, len(parents), chunk):
        pc = parents[s:s + chunk]
        bnd = xp.asarray(
            np.moveaxis(hp.boundaries(nside_in, pc, step=step, nest=True), -1, -2),
            dtype=dtype)                                    # (Ns, 4*step, 3)
        ns, vs, _ = bnd.shape
        dpoly = deformation(bnd.reshape(-1, 3)).reshape(ns, vs, 3)

        centroid = normalize(xp.mean(dpoly, axis=1))
        cang = xp.arccos(xp.clip(xp.sum(dpoly * centroid[:, None, :], axis=-1),
                                 -1.0, 1.0))
        radius = xp.max(cang, axis=1) * 1.1 + pixdiag
        did, tgt = hg.query_disc_ring(nside_out, centroid, radius, safety=1)
        if int(tgt.shape[0]) == 0:
            continue

        areas = batched_polygon_overlap(dpoly[did], tgt_corners[tgt], omega_out)
        vals = src_dev[xp.asarray(pc)[did]]
        num += xp.bincount(tgt, weights=vals * areas, minlength=npix_out)
        den += xp.bincount(tgt, weights=areas, minlength=npix_out)
