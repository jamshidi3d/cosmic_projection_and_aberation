"""Exact-ish spherical polygon overlap via a local gnomonic tangent plane.

The overlap of two small spherical polygons is computed by projecting both into the
gnomonic (tangent-plane) chart centred on the first polygon, clipping there with the
Sutherland-Hodgman algorithm (the clip polygon must be convex -- HEALPix pixels are),
and rescaling the planar overlap area to steradians using the known solid angle of the
clip polygon. Gnomonic area distortion largely cancels in that ratio for pixel-sized
polygons; see ``FUTURE_PROJECTIONS.md`` for the exact-clip alternative.

This module is pure numpy and does not import healpy.
"""

from __future__ import annotations

import numpy as np

from ._backend import array_module
from .geometry import as_vec3, normalize, spherical_polygon_area, tangent_frame


def _frame_axes(frame):
    """Split a frame into ``(e_x, e_y, n)``, each broadcastable against ``(..., 3)`` vecs.

    ``frame`` is either a single ``(3, 3)`` (rows ``[e_x, e_y, n]``) or a batch
    ``(P, 3, 3)`` -- in the batch case an extra axis is inserted so each axis vector
    broadcasts against per-frame vector stacks of shape ``(P, V, 3)``.
    """
    ex, ey, n = frame[..., 0, :], frame[..., 1, :], frame[..., 2, :]
    if frame.ndim == 3:
        ex, ey, n = ex[:, None, :], ey[:, None, :], n[:, None, :]
    return ex, ey, n


def gnomonic_forward(vecs, frame):
    """Project unit vectors ``(..., 3)`` into plane coords ``(..., 2)``.

    Returns ``(xy, w)`` where ``w = vecs . n``; points with ``w <= 0`` are on the far
    hemisphere and their ``xy`` is not meaningful. ``frame`` rows are ``[e_x, e_y, n]``;
    it may be a single ``(3, 3)`` or a batch ``(P, 3, 3)`` (with ``vecs`` ``(P, V, 3)``).
    """
    xp = array_module(vecs, frame)
    vecs = normalize(as_vec3(vecs))
    ex, ey, n = _frame_axes(frame)
    w = xp.sum(vecs * n, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = vecs / w[..., None]
    return xp.stack([xp.sum(p * ex, axis=-1), xp.sum(p * ey, axis=-1)], axis=-1), w


def gnomonic_inverse(xy, frame):
    """Map plane coords ``(..., 2)`` back to unit vectors ``(..., 3)``."""
    xp = array_module(xy, frame)
    xy = xp.asarray(xy, dtype=float)
    ex, ey, n = _frame_axes(frame)
    p = n + xy[..., 0:1] * ex + xy[..., 1:2] * ey
    return normalize(p)


def _signed_area_2d(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def polygon_area_2d(poly):
    """Unsigned area of a simple 2-D polygon ``(N, 2)`` (shoelace)."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return 0.0
    return abs(_signed_area_2d(poly))


def _ensure_ccw(poly):
    return poly[::-1].copy() if _signed_area_2d(poly) < 0 else poly


def _line_intersect(a, b, p, q):
    """Intersection of infinite line ``a-b`` with segment ``p-q`` (assumed to cross)."""
    d1 = b - a
    d2 = q - p
    denom = d2[0] * d1[1] - d2[1] * d1[0]
    if abs(denom) < 1e-300:
        return q
    u = ((a[0] - p[0]) * d1[1] - (a[1] - p[1]) * d1[0]) / denom
    return p + u * d2


def sutherland_hodgman(subject, clip):
    """Clip ``subject`` ``(N, 2)`` against the **convex** polygon ``clip`` ``(M, 2)``.

    Returns the clipped polygon ``(K, 2)`` (``K`` may be 0).
    """
    subject = np.asarray(subject, dtype=float)
    clip = _ensure_ccw(np.asarray(clip, dtype=float))
    output = [np.asarray(p, dtype=float) for p in subject]
    m = len(clip)
    for i in range(m):
        if not output:
            break
        a, b = clip[i], clip[(i + 1) % m]
        edge = b - a

        def inside(pt, a=a, edge=edge):
            return edge[0] * (pt[1] - a[1]) - edge[1] * (pt[0] - a[0]) >= -1e-12

        prev = output[-1]
        prev_in = inside(prev)
        new_output = []
        for cur in output:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    new_output.append(_line_intersect(a, b, prev, cur))
                new_output.append(cur)
            elif prev_in:
                new_output.append(_line_intersect(a, b, prev, cur))
            prev, prev_in = cur, cur_in
        output = new_output
    return np.array(output, dtype=float).reshape(-1, 2)


def spherical_polygon_overlap_area(poly_a, poly_b, ref_area_b=None):
    """Approximate spherical overlap area (steradians) of polygons ``A`` and ``B``.

    Each polygon is ``(N, 3)`` ordered unit vectors. ``B`` is assumed convex. If
    ``ref_area_b`` is given it is used as ``B``'s solid angle (e.g. the exact HEALPix
    pixel area from ``healpy.nside2pixarea``); otherwise it is computed from ``B``.
    Returns ``0.0`` when the polygons do not overlap or lie on opposite hemispheres.
    """
    poly_a = normalize(as_vec3(poly_a))
    poly_b = normalize(as_vec3(poly_b))
    centroid = normalize(poly_a.mean(axis=0))
    frame = tangent_frame(centroid, 0.0, view="inside")

    a2d, wa = gnomonic_forward(poly_a, frame)
    b2d, wb = gnomonic_forward(poly_b, frame)
    if np.any(wa <= 1e-6) or np.any(wb <= 1e-6):
        return 0.0

    clipped = sutherland_hodgman(a2d, b2d)
    if len(clipped) < 3:
        return 0.0

    area_b_plane = polygon_area_2d(b2d)
    if area_b_plane <= 0.0:
        return 0.0
    omega_b = (ref_area_b if ref_area_b is not None
               else abs(spherical_polygon_area(poly_b)))
    return polygon_area_2d(clipped) / area_b_plane * omega_b
