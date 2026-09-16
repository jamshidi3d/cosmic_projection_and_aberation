"""Shared spherical-geometry primitives (fully vectorized).

Works transparently on :mod:`numpy` or :mod:`cupy` arrays -- each function picks its
array module from its inputs (see :func:`cosmic_projection._backend.array_module`).
Python scalars / lists are treated as numpy.

Conventions
-----------
* Unit vectors have their 3 Cartesian components on the trailing axis, shape ``(..., 3)``.
* ``lon``/``lat`` are in **degrees**, following the healpy ``lonlat=True`` convention:
  ``lon`` measured about ``+z`` starting from ``+x``, ``lat`` measured from the equator.
* Angles passed to :func:`rotation_matrix` are in **radians**.
"""

from __future__ import annotations

import numpy as np

from ._backend import array_module

DEG = np.pi / 180.0
_POLE_TOL = 1e-8


def as_vec3(v):
    """Return ``v`` as a float array (numpy or cupy), checking the trailing axis is 3."""
    xp = array_module(v)
    v = xp.asarray(v, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError(f"expected trailing axis of size 3, got shape {v.shape}")
    return v


def normalize(v, axis=-1):
    """Normalize ``v`` to unit length along ``axis`` (zero vectors map to zero)."""
    xp = array_module(v)
    v = xp.asarray(v, dtype=float)
    n = xp.linalg.norm(v, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return xp.where(n > 0, v / xp.where(n > 0, n, 1), 0.0)


def lonlat2vec(lon, lat):
    """``(lon, lat)`` in degrees -> unit vector(s) with shape ``(..., 3)``."""
    xp = array_module(lon, lat)
    lon = xp.asarray(lon, dtype=float) * DEG
    lat = xp.asarray(lat, dtype=float) * DEG
    clat = xp.cos(lat)
    return xp.stack([clat * xp.cos(lon), clat * xp.sin(lon), xp.sin(lat)], axis=-1)


def vec2lonlat(v):
    """Unit vector(s) ``(..., 3)`` -> ``(lon, lat)`` in degrees, ``lon`` in ``[0, 360)``."""
    xp = array_module(v)
    v = as_vec3(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    lon = xp.mod(xp.arctan2(y, x) / DEG, 360.0)
    lat = xp.arctan2(z, xp.sqrt(x * x + y * y)) / DEG
    return lon, lat


def rotation_matrix(axis, angle):
    """Right-handed rotation by ``angle`` radians about ``axis`` (Rodrigues' formula).

    ``axis`` is a 3-vector (need not be unit). ``angle`` may be a scalar (returns a
    ``(3, 3)`` matrix) or an array (returns ``(..., 3, 3)``).
    """
    xp = array_module(axis, angle)
    ax, ay, az = normalize(xp.asarray(axis, dtype=float))
    K = xp.asarray([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
    angle = xp.asarray(angle, dtype=float)
    s, c = xp.sin(angle), xp.cos(angle)
    eye = xp.eye(3)
    if angle.ndim == 0:
        return eye + s * K + (1.0 - c) * (K @ K)
    return eye + s[..., None, None] * K + (1.0 - c)[..., None, None] * (K @ K)


def great_circle_distance(v1, v2):
    """Angular separation (radians) between unit vectors, stable near 0 and pi."""
    xp = array_module(v1, v2)
    v1 = normalize(as_vec3(v1))
    v2 = normalize(as_vec3(v2))
    sin_a = xp.linalg.norm(xp.cross(v1, v2), axis=-1)
    cos_a = xp.sum(v1 * v2, axis=-1)
    return xp.arctan2(sin_a, cos_a)


def _tri_solid_angle(a, b, c):
    """Signed solid angle of the spherical triangle ``(a, b, c)`` (unit vecs, ``(..., 3)``).

    Van Oosterom & Strackee ``atan2`` form; sign follows the orientation ``a -> b -> c``
    about the outward normal.
    """
    xp = array_module(a, b, c)
    triple = xp.sum(a * xp.cross(b, c), axis=-1)
    denom = (1.0 + xp.sum(a * b, axis=-1) + xp.sum(b * c, axis=-1)
             + xp.sum(c * a, axis=-1))
    return 2.0 * xp.arctan2(triple, denom)


def spherical_polygon_area(verts):
    """Signed spherical polygon area (solid angle, steradians).

    ``verts`` has shape ``(..., N, 3)`` with ``N >= 3`` ordered unit vectors. Positive
    for counter-clockwise winding as seen from **outside** the sphere. Fan triangulation
    from the first vertex -- exact for convex polygons and polygons star-shaped w.r.t.
    ``verts[..., 0, :]`` (HEALPix pixels, deformed pixels, clipped overlaps all qualify).
    """
    xp = array_module(verts)
    verts = normalize(as_vec3(verts))
    n = verts.shape[-2]
    if n < 3:
        raise ValueError("need at least 3 vertices")
    v0 = verts[..., 0, :]
    total = xp.zeros(verts.shape[:-2])
    for i in range(1, n - 1):
        total = total + _tri_solid_angle(v0, verts[..., i, :], verts[..., i + 1, :])
    return total


def tangent_frame(center_vec, roll_deg=0.0, view="inside"):
    """Orthonormal screen basis for a tangent plane at ``center_vec``.

    Returns an array of shape ``(3, 3)`` whose rows are ``[e_x, e_y, n]``:

    * ``n`` -- plane normal. ``view="inside"`` -> ``n = +center``; ``"outside"`` -> ``-center``.
    * ``e_y`` -- sky north (``+z``) projected into the plane (falls back to ``+x`` at a pole).
    * ``e_x`` -- ``e_y x n`` (right-handed: ``e_x x e_y = n``).
    * ``roll_deg`` -- right-handed rotation of ``(e_x, e_y)`` about ``n``.
    """
    xp = array_module(center_vec)
    c = normalize(as_vec3(center_vec))
    if view == "inside":
        n = c
    elif view == "outside":
        n = -c
    else:
        raise ValueError("view must be 'inside' or 'outside'")

    up = xp.asarray([0.0, 0.0, 1.0])
    ey = up - xp.dot(up, n) * n
    if float(xp.linalg.norm(ey)) < _POLE_TOL:
        ref = xp.asarray([1.0, 0.0, 0.0])
        ey = ref - xp.dot(ref, n) * n
    ey = normalize(ey)
    ex = normalize(xp.cross(ey, n))

    if roll_deg:
        r = rotation_matrix(n, xp.asarray(np.deg2rad(roll_deg)))
        ex = r @ ex
        ey = r @ ey
    return xp.stack([ex, ey, n], axis=0)


def batched_tangent_frame(centers, view="inside"):
    """Vectorized :func:`tangent_frame` (``roll_deg=0``) for ``centers`` ``(P, 3)``.

    Returns ``(P, 3, 3)`` with rows ``[e_x, e_y, n]`` per centre.
    """
    xp = array_module(centers)
    c = normalize(as_vec3(centers))
    n = c if view == "inside" else -c
    nz = n[..., 2:3]                                   # z-component of each normal
    ey = -nz * n                                       # (+z) projected into the plane
    ey = ey + xp.concatenate([xp.zeros_like(nz), xp.zeros_like(nz),
                              xp.ones_like(nz)], axis=-1)
    bad = xp.linalg.norm(ey, axis=-1) < _POLE_TOL
    nx = n[..., 0:1]
    ref = -nx * n                                      # (+x) projected into the plane
    ref = ref + xp.concatenate([xp.ones_like(nx), xp.zeros_like(nx),
                                xp.zeros_like(nx)], axis=-1)
    ey = normalize(xp.where(bad[..., None], ref, ey))
    ex = normalize(xp.cross(ey, n))
    return xp.stack([ex, ey, n], axis=-2)
