"""Batched spherical-polygon overlap -- the GPU analogue of
:func:`cosmic_projection.overlap.spherical_polygon_overlap_area`.

Every operation is vectorised over ``P`` (source-pixel, target-pixel) pairs with fixed
array shapes: build a local gnomonic frame at each source-polygon centroid, project both
polygons, clip the (possibly non-convex) source polygon against the convex target quad
with a fixed-buffer Sutherland-Hodgman, and rescale the planar overlap to steradians by
the target pixel's known solid angle. Same maths (and same gnomonic-ratio approximation)
as the scalar CPU reference.

Runs on numpy or cupy; the array module is taken from the inputs.
"""

from __future__ import annotations

import numpy as np

from ._backend import array_module
from .geometry import batched_tangent_frame, normalize
from .overlap import gnomonic_forward

_EPS = 1e-9


def _signed_area(poly):
    """Signed shoelace area of ``poly`` ``(P, V, 2)`` -> ``(P,)`` (cyclic)."""
    xp = array_module(poly)
    x, y = poly[..., 0], poly[..., 1]
    xn, yn = xp.roll(x, -1, axis=-1), xp.roll(y, -1, axis=-1)
    return 0.5 * xp.sum(x * yn - xn * y, axis=-1)


def _ensure_ccw(quad):
    """Flip ``quad`` ``(P, 4, 2)`` to counter-clockwise where it is clockwise."""
    xp = array_module(quad)
    flip = _signed_area(quad) < 0.0
    return xp.where(flip[:, None, None], quad[:, ::-1, :], quad)


def _line_intersect(prev, cur, a, b):
    """Point where segment ``prev->cur`` meets the infinite line ``a->b``.

    All args ``(P, V, 2)`` (broadcast); returns ``(P, V, 2)``. Falls back to ``prev``
    for near-parallel configurations.
    """
    xp = array_module(prev, cur, a, b)
    d1 = cur - prev
    d2 = b - a
    denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    num = (a[..., 0] - prev[..., 0]) * d2[..., 1] - (a[..., 1] - prev[..., 1]) * d2[..., 0]
    t = xp.where(xp.abs(denom) > 1e-300, num / xp.where(xp.abs(denom) > 1e-300, denom, 1.0), 0.0)
    return prev + t[..., None] * d1


def batched_sutherland_hodgman(subject, clip):
    """Clip ``subject`` ``(P, Vs, 2)`` against the convex quad ``clip`` ``(P, 4, 2)``.

    Returns ``(poly (P, Vmax, 2), valid (P, Vmax))`` with ``Vmax = Vs + 4``. After each
    edge the valid vertices are compacted to the front and trailing slots hold a copy of
    the last valid vertex, so a plain cyclic shoelace over the whole buffer yields the
    correct clipped area.
    """
    xp = array_module(subject, clip)
    P, Vs, _ = subject.shape
    vmax = Vs + 4
    clip = _ensure_ccw(clip)

    poly = xp.concatenate([subject, xp.repeat(subject[:, -1:, :], 4, axis=1)], axis=1)
    nvalid = xp.full(P, Vs, dtype=xp.int64)
    slot = xp.arange(vmax)
    ramp = xp.arange(2 * vmax, dtype=xp.int64)

    for k in range(4):
        a = clip[:, k, :][:, None, :]                       # (P, 1, 2)
        b = clip[:, (k + 1) % 4, :][:, None, :]
        edge = b - a

        valid = slot[None, :] < nvalid[:, None]             # (P, Vmax) real vertices
        is_last = slot[None, :] == (nvalid - 1)[:, None]
        nxt = xp.where(is_last[..., None], poly[:, 0:1, :], xp.roll(poly, -1, axis=1))

        s0 = edge[..., 0] * (poly[..., 1] - a[..., 1]) - edge[..., 1] * (poly[..., 0] - a[..., 0])
        s1 = edge[..., 0] * (nxt[..., 1] - a[..., 1]) - edge[..., 1] * (nxt[..., 0] - a[..., 0])
        in0 = s0 >= -1e-12
        in1 = s1 >= -1e-12
        inter = _line_intersect(poly, nxt, a, b)            # per edge (start slot -> next)

        emit = xp.empty((P, 2 * vmax, 2), dtype=poly.dtype)
        emit[:, 0::2, :] = inter
        emit[:, 1::2, :] = nxt
        emit_valid = xp.empty((P, 2 * vmax), dtype=bool)
        emit_valid[:, 0::2] = valid & (in0 != in1)          # crossing -> intersection
        emit_valid[:, 1::2] = valid & in1                   # end vertex kept if inside

        key = xp.where(emit_valid, ramp[None, :], ramp[None, :] + 10 * vmax)
        order = xp.argsort(key, axis=1)
        emit = xp.take_along_axis(emit, xp.broadcast_to(order[..., None], emit.shape), axis=1)
        emit_valid = xp.take_along_axis(emit_valid, order, axis=1)

        nvalid = xp.clip(emit_valid.sum(axis=1), 0, vmax)
        poly = emit[:, :vmax, :]
        keep = slot[None, :] < nvalid[:, None]
        last = emit[xp.arange(P), xp.clip(nvalid - 1, 0, vmax - 1)]
        poly = xp.where(keep[..., None], poly, last[:, None, :])

    return poly, slot[None, :] < nvalid[:, None]


def batched_polygon_overlap(subject_xyz, clip_xyz, ref_area):
    """Spherical overlap area (steradians) for a batch of polygon pairs.

    ``subject_xyz`` ``(P, Vs, 3)`` deformed source polygons; ``clip_xyz`` ``(P, 4, 3)``
    target pixel corners; ``ref_area`` scalar or ``(P,)`` -- the target pixel solid
    angle. Pairs with any vertex on the far hemisphere, or a degenerate target, get 0.
    """
    xp = array_module(subject_xyz, clip_xyz)
    subject_xyz = xp.asarray(subject_xyz, dtype=xp.float64)
    clip_xyz = xp.asarray(clip_xyz, dtype=xp.float64)

    centroid = normalize(xp.mean(subject_xyz, axis=1))
    frames = batched_tangent_frame(centroid)

    a_xy, a_w = gnomonic_forward(subject_xyz, frames)
    b_xy, b_w = gnomonic_forward(clip_xyz, frames)
    bad = (xp.any(a_w <= 1e-6, axis=1) | xp.any(b_w <= 1e-6, axis=1))

    clipped, _ = batched_sutherland_hodgman(a_xy, b_xy)
    area_clip = xp.abs(_signed_area(clipped))
    area_b = xp.abs(_signed_area(b_xy))

    ref = xp.asarray(ref_area, dtype=xp.float64)
    frac = xp.where(area_b > _EPS, area_clip / xp.where(area_b > _EPS, area_b, 1.0), 0.0)
    return xp.where(bad | (area_b <= _EPS), 0.0, frac * ref)


def self_intersecting(poly_xyz):
    """Cheap flag: True where a source polygon's 2-D gnomonic image is non-simple.

    Uses the sign of consecutive edge cross-products at its own centroid frame; a simple
    (convex or gently non-convex) polygon keeps a consistent winding. Diagnostic only.
    """
    xp = array_module(poly_xyz)
    poly_xyz = xp.asarray(poly_xyz, dtype=xp.float64)
    frames = batched_tangent_frame(normalize(xp.mean(poly_xyz, axis=1)))
    xy, _ = gnomonic_forward(poly_xyz, frames)
    e = xp.roll(xy, -1, axis=1) - xy
    c = e[..., 0] * xp.roll(e[..., 1], -1, axis=1) - e[..., 1] * xp.roll(e[..., 0], -1, axis=1)
    pos = xp.sum(c > 0, axis=1)
    neg = xp.sum(c < 0, axis=1)
    return (pos > 0) & (neg > 0)
