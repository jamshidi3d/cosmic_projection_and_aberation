"""Closed-form HEALPix RING primitives, written against a generic array module.

These run on numpy **or** cupy (no healpy) so the GPU resampling path needs no host
round-trips for pixel lookup. The heavy pixel *geometry* (source-pixel boundaries,
target-pixel corners, NEST sub-pixel centres) is still produced with healpy on the host
and uploaded -- only ``vec2pix_ring`` and the vectorised ``query_disc_ring`` are
reimplemented here, because those are the ones in the hot loop.

Algorithms follow Gorski et al. 2005 and the HEALPix C++ ``loc2pix`` / disc-query
logic. Everything is branchless (``xp.where``) and validated against healpy in
``tests/test_gpu.py``.
"""

from __future__ import annotations

import numpy as np

from ._backend import array_module

_HALFPI = 0.5 * np.pi
_TWOPI = 2.0 * np.pi


def _ifloor(x):
    xp = array_module(x)
    return xp.floor(x).astype(xp.int64)


def _isqrt(n):
    """Elementwise floor(sqrt(n)) for non-negative integer arrays, exact."""
    xp = array_module(n)
    n = n.astype(xp.int64)
    r = xp.floor(xp.sqrt(n.astype(xp.float64))).astype(xp.int64)
    r = xp.where((r + 1) * (r + 1) <= n, r + 1, r)
    r = xp.where(r * r > n, r - 1, r)
    return r


# --------------------------------------------------------------------------- #
# direction -> RING pixel                                                     #
# --------------------------------------------------------------------------- #

def ang2pix_ring(nside, theta, phi):
    """``(theta, phi)`` in radians -> RING pixel index (int64), vectorised."""
    xp = array_module(theta, phi)
    nside = int(nside)
    z = xp.cos(theta)
    return _zphi2pix_ring(nside, z, xp.mod(phi, _TWOPI), xp)


def vec2pix_ring(nside, xyz):
    """Unit vectors ``(..., 3)`` -> RING pixel index (int64), vectorised."""
    xp = array_module(xyz)
    xyz = xp.asarray(xyz, dtype=xp.float64)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    phi = xp.mod(xp.arctan2(y, x), _TWOPI)
    norm = xp.sqrt(x * x + y * y + z * z)
    return _zphi2pix_ring(nside, z / norm, phi, xp)


def _zphi2pix_ring(nside, z, phi, xp):
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    za = xp.abs(z)
    tt = xp.mod(phi * (1.0 / _HALFPI), 4.0)                 # in [0, 4)

    # ---- equatorial belt (|z| <= 2/3) ----
    temp1 = nside * (0.5 + tt)
    temp2 = nside * z * 0.75
    jp = _ifloor(temp1 - temp2)
    jm = _ifloor(temp1 + temp2)
    ir = nside + 1 + jp - jm                                # ring, counted from z = 2/3
    kshift = 1 - (ir & 1)
    ip_eq = (jp + jm - nside + kshift + 1) // 2
    ip_eq = xp.mod(ip_eq, 4 * nside)
    pix_eq = ncap + (ir - 1) * 4 * nside + ip_eq

    # ---- polar caps (|z| > 2/3) ----
    tp = tt - xp.floor(tt)
    tmp = nside * xp.sqrt(xp.clip(3.0 * (1.0 - za), 0.0, None))
    jp_c = _ifloor(tp * tmp)
    jm_c = _ifloor((1.0 - tp) * tmp)
    ir_c = jp_c + jm_c + 1                                  # ring from the nearest pole
    ip_c = _ifloor(tt * ir_c)
    ip_c = xp.mod(ip_c, xp.maximum(4 * ir_c, 1))
    pix_north = 2 * ir_c * (ir_c - 1) + ip_c
    pix_south = npix - 2 * ir_c * (ir_c + 1) + ip_c
    pix_cap = xp.where(z > 0, pix_north, pix_south)

    return xp.where(za <= 2.0 / 3.0, pix_eq, pix_cap).astype(xp.int64)


# --------------------------------------------------------------------------- #
# RING pixel -> direction / corners                                          #
# --------------------------------------------------------------------------- #

def _pix2zphi_ring(nside, ipix, xp):
    nside = int(nside)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    fact2 = 1.0 / (3.0 * nside * nside)
    fact1 = 2.0 * nside * fact2                             # = 2 / (3 nside)
    ipix = ipix.astype(xp.int64)

    north = ipix < ncap
    south = ipix >= npix - ncap

    # north cap
    iring_n = (1 + _isqrt(1 + 2 * ipix)) // 2
    iphi_n = (ipix + 1) - 2 * iring_n * (iring_n - 1)
    z_n = 1.0 - iring_n * iring_n * fact2
    phi_n = (iphi_n - 0.5) * _HALFPI / xp.maximum(iring_n, 1)

    # south cap
    ip_s = npix - ipix
    iring_s = (1 + _isqrt(2 * ip_s - 1)) // 2
    iphi_s = 4 * iring_s + 1 - (ip_s - 2 * iring_s * (iring_s - 1))
    z_s = iring_s * iring_s * fact2 - 1.0
    phi_s = (iphi_s - 0.5) * _HALFPI / xp.maximum(iring_s, 1)

    # equatorial belt
    ip_e = ipix - ncap
    iring_e = ip_e // (4 * nside) + nside
    iphi_e = xp.mod(ip_e, 4 * nside) + 1
    fodd = xp.where(((iring_e + nside) & 1) == 1, 1.0, 0.5)
    z_e = (2 * nside - iring_e) * fact1
    phi_e = (iphi_e - fodd) * (_HALFPI / nside)

    z = xp.where(north, z_n, xp.where(south, z_s, z_e))
    phi = xp.where(north, phi_n, xp.where(south, phi_s, phi_e))
    return z, xp.mod(phi, _TWOPI)


def pix2vec_ring(nside, ipix):
    """RING pixel index -> unit vector of the pixel centre, shape ``(..., 3)``."""
    xp = array_module(ipix)
    z, phi = _pix2zphi_ring(nside, xp.asarray(ipix), xp)
    sth = xp.sqrt(xp.clip(1.0 - z * z, 0.0, 1.0))
    return xp.stack([sth * xp.cos(phi), sth * xp.sin(phi), z], axis=-1)


def _ring_info(nside, iring, xp):
    """For ring index ``iring`` in ``[1, 4 nside - 1]`` return ``(z, nr, phi0, sth)``.

    ``nr`` = pixels in the ring; ``phi0`` = azimuth of pixel 0; ``sth`` = sin(colat).
    Also valid for *fractional* ``iring`` (used for half-ring pixel corners).
    """
    nside = int(nside)
    fact2 = 1.0 / (3.0 * nside * nside)
    fact1 = 2.0 * nside * fact2
    north = iring < nside
    south = iring > 3 * nside
    z = xp.where(north, 1.0 - iring * iring * fact2,
                 xp.where(south, (4 * nside - iring) ** 2 * fact2 - 1.0,
                          (2 * nside - iring) * fact1))
    nr = xp.where(north, 4 * iring,
                  xp.where(south, 4 * (4 * nside - iring), 4 * nside)).astype(xp.float64)
    # phase of pixel 0: caps -> 0.5 dphase; equatorial -> (1 - fodd) dphase
    fodd = xp.where(((xp.floor(iring).astype(xp.int64) + nside) & 1) == 1, 1.0, 0.5)
    frac0 = xp.where(north | south, 0.5, 1.0 - fodd)
    phi0 = frac0 * (_TWOPI / nr)
    sth = xp.sqrt(xp.clip(1.0 - z * z, 0.0, 1.0))
    return z, nr, phi0, sth


def _ring_start(nside, iring, xp):
    """Global RING index of pixel 0 in integer ring ``iring`` (1-indexed from north)."""
    nside = int(nside)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    north = iring <= nside
    south = iring > 3 * nside
    start_n = 2 * iring * (iring - 1)
    start_e = ncap + (iring - nside) * 4 * nside
    ii = 4 * nside - iring
    start_s = npix - 2 * ii * (ii + 1)
    return xp.where(north, start_n, xp.where(south, start_s, start_e)).astype(xp.int64)


def _z_to_ring(nside, z, xp):
    """Inverse of ring -> z: the (fractional) ring index for colatitude ``arccos z``."""
    nside = int(nside)
    north = z > 2.0 / 3.0
    south = z < -2.0 / 3.0
    ir_n = nside * xp.sqrt(xp.clip(3.0 * (1.0 - z), 0.0, None))
    ir_s = 4 * nside - nside * xp.sqrt(xp.clip(3.0 * (1.0 + z), 0.0, None))
    ir_e = nside * (2.0 - 1.5 * z)
    return xp.where(north, ir_n, xp.where(south, ir_s, ir_e))


def pix2corners_ring(nside, ipix):
    """Approximate 4 corners (N, E, S, W) of a RING pixel, shape ``(..., 4, 3)``.

    Diamond model: E/W corners sit half a ring-pixel of azimuth either side of the
    centre at the ring's own ``z``; N/S corners sit at the same azimuth on the
    half-ring boundaries. Exact in the equatorial belt; sub-pixel error in the caps.
    For the reference geometry use ``healpy.boundaries`` on the host instead.
    """
    xp = array_module(ipix)
    ipix = xp.asarray(ipix).astype(xp.int64)
    zc, phic = _pix2zphi_ring(nside, ipix, xp)
    ir = xp.round(_z_to_ring(nside, zc))                    # nearest integer ring
    _, nr, _, _ = _ring_info(nside, ir, xp)
    dphi = _TWOPI / nr / 2.0                                # half azimuthal pixel width
    z_n, _, _, _ = _ring_info(nside, ir - 0.5, xp)
    z_s, _, _, _ = _ring_info(nside, ir + 0.5, xp)

    def _vec(z, phi):
        sth = xp.sqrt(xp.clip(1.0 - z * z, 0.0, 1.0))
        return xp.stack([sth * xp.cos(phi), sth * xp.sin(phi),
                         xp.broadcast_to(z, phi.shape)], axis=-1)

    n = _vec(z_n, phic)
    s = _vec(z_s, phic)
    e = _vec(zc, phic + dphi)
    w = _vec(zc, phic - dphi)
    return xp.stack([n, e, s, w], axis=-2)


# --------------------------------------------------------------------------- #
# vectorised disc query                                                       #
# --------------------------------------------------------------------------- #

def query_disc_ring(nside, centers, radii, safety=1):
    """RING pixels whose centres lie within ``radii`` of ``centers``.

    ``centers`` ``(D, 3)`` unit vectors, ``radii`` ``(D,)`` radians. Returns flat
    ``(disc_id (K,), ipix (K,))`` -- like calling ``healpy.query_disc`` per disc and
    concatenating, minus ordering guarantees. ``safety`` inflates radii by that many
    ring spacings before selecting (over-inclusion is harmless downstream).
    """
    xp = array_module(centers, radii)
    nside = int(nside)
    npix = 12 * nside * nside
    centers = xp.asarray(centers, dtype=xp.float64)
    radii = xp.asarray(radii, dtype=xp.float64)
    D = centers.shape[0]

    z0 = xp.clip(centers[:, 2], -1.0, 1.0)
    phi0 = xp.mod(xp.arctan2(centers[:, 1], centers[:, 0]), _TWOPI)
    theta0 = xp.arccos(z0)
    pad = safety * (np.pi / (2 * nside))
    r = radii + pad
    z_hi = xp.cos(xp.clip(theta0 - r, 0.0, np.pi))          # larger z -> smaller ring idx
    z_lo = xp.cos(xp.clip(theta0 + r, 0.0, np.pi))

    ir_lo = xp.clip(xp.ceil(_z_to_ring(nside, z_hi, xp)), 1, 4 * nside - 1).astype(xp.int64)
    ir_hi = xp.clip(xp.floor(_z_to_ring(nside, z_lo, xp)), 1, 4 * nside - 1).astype(xp.int64)
    n_rings = xp.clip(ir_hi - ir_lo + 1, 0, None)

    grp, within = _pair_expand(n_rings, xp)
    if grp.shape[0] == 0:
        z64 = xp.zeros(0, dtype=xp.int64)
        return z64, z64
    ring = ir_lo[grp] + within                              # (T,)

    zr, nr, phi_r0, sth_r = _ring_info(nside, ring.astype(xp.float64), xp)
    r_p = r[grp]
    z0_p = z0[grp]
    phi0_p = phi0[grp]
    sth0_p = xp.sqrt(xp.clip(1.0 - z0_p * z0_p, 0.0, 1.0))

    denom = sth0_p * sth_r
    cos_dphi = xp.where(denom > 1e-12,
                        (xp.cos(r_p) - z0_p * zr) / xp.where(denom > 1e-12, denom, 1.0),
                        -2.0)
    dphi = xp.where(cos_dphi <= -1.0, np.pi,
                    xp.where(cos_dphi >= 1.0, 0.0, xp.arccos(xp.clip(cos_dphi, -1.0, 1.0))))

    dphase = _TWOPI / nr
    jc = (phi0_p - phi_r0) / dphase
    jw = dphi / dphase
    j_first = xp.ceil(jc - jw).astype(xp.int64)
    j_last = xp.floor(jc + jw).astype(xp.int64)
    count = xp.where(dphi >= np.pi, nr.astype(xp.int64),
                     xp.clip(j_last - j_first + 1, 0, nr.astype(xp.int64)))

    pair, within2 = _pair_expand(count, xp)
    if pair.shape[0] == 0:
        z64 = xp.zeros(0, dtype=xp.int64)
        return z64, z64
    j_local = j_first[pair] + within2
    nr_i = nr[pair].astype(xp.int64)
    ipix_in_ring = xp.mod(j_local, nr_i)
    ipix = _ring_start(nside, ring[pair], xp) + ipix_in_ring
    disc_id = grp[pair]
    return disc_id.astype(xp.int64), xp.clip(ipix, 0, npix - 1).astype(xp.int64)


def _ring_above(nside, z, xp):
    """Index of the ring immediately north of colatitude ``arccos z`` (0..4 nside)."""
    nside = int(nside)
    az = xp.abs(z)
    eq = nside * (2.0 - 1.5 * z)
    ic = nside * xp.sqrt(xp.clip(3.0 * (1.0 - az), 0.0, None))
    cap = xp.where(z > 0, ic, 4 * nside - ic - 1)
    return xp.floor(xp.where(az <= 2.0 / 3.0, eq, cap)).astype(xp.int64)


def get_interp_weights_ring(nside, theta, phi):
    """Bilinear interpolation stencil on the RING grid.

    Returns ``(pix (4, K), wgt (4, K))`` -- the 4 pixels bracketing each ``(theta, phi)``
    in ``(z, phi)`` and their weights (sum to 1). This is *a* bilinear scheme on the
    HEALPix grid, close to but not bit-identical with ``healpy.get_interp_weights``;
    within one ring of a pole it degrades to linear-in-phi on the nearest ring.
    """
    xp = array_module(theta, phi)
    nside = int(nside)
    z = xp.cos(theta)
    phi = xp.mod(phi, _TWOPI)

    ir1 = xp.clip(_ring_above(nside, z, xp), 1, 4 * nside - 1)
    ir2 = xp.clip(ir1 + 1, 1, 4 * nside - 1)
    z1, nr1, phi01, _ = _ring_info(nside, ir1.astype(xp.float64), xp)
    z2, nr2, phi02, _ = _ring_info(nside, ir2.astype(xp.float64), xp)
    start1 = _ring_start(nside, ir1, xp)
    start2 = _ring_start(nside, ir2, xp)

    same = ir1 == ir2
    dz = xp.where(same, 1.0, z1 - z2)
    wz = xp.clip(xp.where(same, 1.0, (z - z2) / dz), 0.0, 1.0)   # weight on the north ring

    def _in_ring(nr, phi0, start):
        nr_i = nr.astype(xp.int64)
        t = (phi - phi0) / (_TWOPI / nr)
        j0 = xp.floor(t).astype(xp.int64)
        fj = t - j0
        jA = start + xp.mod(j0, nr_i)
        jB = start + xp.mod(j0 + 1, nr_i)
        return jA, jB, fj

    a1, b1, f1 = _in_ring(nr1, phi01, start1)
    a2, b2, f2 = _in_ring(nr2, phi02, start2)
    pix = xp.stack([a1, b1, a2, b2], axis=0)
    wgt = xp.stack([wz * (1.0 - f1), wz * f1, (1.0 - wz) * (1.0 - f2),
                    (1.0 - wz) * f2], axis=0)
    return pix.astype(xp.int64), wgt


def _pair_expand(counts, xp):
    """Return ``(group_id, within_group_index)`` flat arrays for per-group ``counts``."""
    counts = counts.astype(xp.int64)
    total = int(counts.sum())
    if total == 0:
        z = xp.zeros(0, dtype=xp.int64)
        return z, z
    group = xp.repeat(xp.arange(counts.shape[0], dtype=xp.int64), counts)
    ends = xp.cumsum(counts)
    starts = ends - counts
    ramp = xp.arange(total, dtype=xp.int64)
    within = ramp - xp.repeat(starts, counts)
    return group, within
