"""Feature 2: gnomonic (tangent-plane) cutouts of a HEALPix map.

:class:`TangentPlane` describes a square grid of near-equal-area pixels tangent to the
sphere at a chosen direction, with the plane normal along (``view="inside"``) or
opposite (``view="outside"``) that direction and a free ``roll_deg`` about the normal.
:meth:`TangentPlane.project_map` samples a HEALPix map onto it; :func:`flat_sky_
diagnostics` quantifies how good the flat-sky (gnomonic) approximation is over the patch.

Only the gnomonic projection is implemented. Orthographic / stereographic / reverse
(plane -> HEALPix) approaches are written up in ``FUTURE_PROJECTIONS.md``.
"""

from __future__ import annotations

import numpy as np
import healpy as hp

from . import _healpix as hx
from . import _healpix_gpu as hg
from ._backend import get_xp, resolve_backend, to_host
from .geometry import (as_vec3, great_circle_distance, lonlat2vec, normalize,
                       spherical_polygon_area, tangent_frame)
from .overlap import gnomonic_forward, gnomonic_inverse

_SUPPORTED_PROJECTIONS = ("gnomonic",)


class TangentPlane:
    """A square gnomonic cutout grid.

    Parameters
    ----------
    center : (lon_deg, lat_deg) or 3-vector
        Direction the plane is tangent to.
    size_deg : float
        Full angular width of the (square) patch.
    npix : int
        Pixels per side.
    roll_deg : float
        Right-handed rotation of the grid about the plane normal.
    view : {"inside", "outside"}
        Normal along ``+center`` (observer at the origin) or ``-center``.
    projection : {"gnomonic"}
    """

    def __init__(self, center, size_deg, npix, roll_deg=0.0, view="inside",
                 projection="gnomonic"):
        if projection not in _SUPPORTED_PROJECTIONS:
            raise ValueError(
                f"projection {projection!r} not implemented; supported: "
                f"{_SUPPORTED_PROJECTIONS}. See FUTURE_PROJECTIONS.md.")
        center = np.asarray(center, dtype=float)
        if center.shape == (2,):
            self.center_vec = lonlat2vec(center[0], center[1])
        elif center.shape == (3,):
            self.center_vec = normalize(center)
        else:
            raise ValueError("center must be (lon, lat) in degrees or a 3-vector")

        self.size_deg = float(size_deg)
        self.npix = int(npix)
        self.roll_deg = float(roll_deg)
        self.view = view
        self.projection = projection
        self.frame = tangent_frame(self.center_vec, self.roll_deg, view=view)

    @classmethod
    def matching_healpix(cls, center, nside, size_deg, **kw):
        """Build a plane whose pixel scale matches ``healpy.nside2resol(nside)``."""
        res_deg = np.degrees(hp.nside2resol(nside))
        npix = max(1, int(round(size_deg / res_deg)))
        return cls(center, size_deg, npix, **kw)

    @property
    def res_deg(self):
        return self.size_deg / self.npix

    def __repr__(self):
        return (f"TangentPlane(size_deg={self.size_deg}, npix={self.npix}, "
                f"roll_deg={self.roll_deg}, view={self.view!r})")

    # -- geometry -------------------------------------------------------------

    def _edges_1d(self):
        """Tangent-plane coordinate of the ``npix + 1`` pixel edges (``tan`` of angle)."""
        half = np.deg2rad(self.size_deg) / 2.0
        return np.tan(np.linspace(-half, half, self.npix + 1))

    def _centers_1d(self):
        e = self._edges_1d()
        return 0.5 * (e[:-1] + e[1:])

    def pixel_centers_vec(self):
        """Unit vectors of pixel centres, shape ``(npix, npix, 3)``.

        Row 0 is the top of the image (largest ``e_y``); column 0 is the left
        (smallest ``e_x``).
        """
        g = self._centers_1d()
        xx, yy = np.meshgrid(g, g[::-1], indexing="xy")     # yy decreases with row
        return gnomonic_inverse(np.stack([xx, yy], axis=-1), self.frame)

    def pixel_corners_vec(self):
        """Unit vectors of the 4 corners of every pixel, shape ``(npix, npix, 4, 3)``.

        Corners are ordered TL, TR, BR, BL in image space.
        """
        e = self._edges_1d()
        n = self.npix
        xl, xr = e[:-1][None, :], e[1:][None, :]            # (1, n)  left / right
        yt, yb = e[::-1][:-1][:, None], e[::-1][1:][:, None]  # (n, 1) top / bottom
        xl, xr = np.broadcast_to(xl, (n, n)), np.broadcast_to(xr, (n, n))
        yt, yb = np.broadcast_to(yt, (n, n)), np.broadcast_to(yb, (n, n))
        cx = np.stack([xl, xr, xr, xl], axis=-1)
        cy = np.stack([yt, yt, yb, yb], axis=-1)
        return gnomonic_inverse(np.stack([cx, cy], axis=-1), self.frame)

    # -- projection --------------------------------------------------------------

    def project_map(self, hpx_map, *, method="bilinear", nest=False, oversample=4,
                    fill=hp.UNSEEN, backend="cpu"):
        """Sample ``hpx_map`` onto this plane, returning an ``(npix, npix)`` image.

        ``method``:
          * ``"nearest"``  -- nearest HEALPix pixel at each cell centre.
          * ``"bilinear"`` -- ``healpy.get_interp_val`` at each cell centre.
          * ``"area"``     -- mean of ``oversample**2`` bilinear sub-samples per cell
            (sub-samples equally weighted; residual gnomonic area distortion is small
            over a patch for which the flat-sky approximation is being used at all).

        ``backend``: ``"cpu"`` (default) or ``"gpu"``/``"auto"``. The GPU path uses the
        closed-form RING primitives in ``cosmic_projection._healpix_gpu``; its
        ``"nearest"`` matches the CPU result exactly, while ``"bilinear"`` / ``"area"``
        use an independent bilinear stencil (close to, not identical with, healpy's).
        """
        if resolve_backend(backend) == "gpu":
            return self._project_map_gpu(hpx_map, method, nest, oversample, fill)

        hpx_map = np.asarray(hpx_map, dtype=float)
        nside = hp.npix2nside(hpx_map.shape[-1])
        n = self.npix

        if method == "nearest":
            v = self.pixel_centers_vec().reshape(-1, 3)
            pix = hp.vec2pix(nside, v[:, 0], v[:, 1], v[:, 2], nest=nest)
            img = hpx_map[pix].reshape(n, n)
        elif method == "bilinear":
            v = self.pixel_centers_vec().reshape(-1, 3)
            th, ph = hp.vec2ang(v)
            img = hp.get_interp_val(hpx_map, th, ph, nest=nest).reshape(n, n)
        elif method == "area":
            fine = TangentPlane(self.center_vec, self.size_deg, n * oversample,
                                roll_deg=self.roll_deg, view=self.view,
                                projection=self.projection)
            v = fine.pixel_centers_vec().reshape(-1, 3)
            th, ph = hp.vec2ang(v)
            vals = hp.get_interp_val(hpx_map, th, ph, nest=nest)
            img = vals.reshape(n, oversample, n, oversample).mean(axis=(1, 3))
        else:
            raise ValueError("method must be 'nearest', 'bilinear' or 'area'")

        bad = ~np.isfinite(img)
        if bad.any():
            img = np.where(bad, fill, img)
        return img

    def _project_map_gpu(self, hpx_map, method, nest, oversample, fill):
        xp = get_xp("gpu")
        hpx_map = np.asarray(hpx_map, dtype=float)
        nside = hp.npix2nside(hpx_map.shape[-1])
        n = self.npix
        m_dev = xp.asarray(hx.as_ring(hpx_map, nest))          # GPU works in RING

        def _sample(vecs):                                     # vecs (K, 3) on device
            if method == "nearest":
                return m_dev[hg.vec2pix_ring(nside, vecs)]
            th = xp.arccos(xp.clip(vecs[:, 2], -1.0, 1.0))
            ph = xp.arctan2(vecs[:, 1], vecs[:, 0])
            pix, wgt = hg.get_interp_weights_ring(nside, th, ph)
            return xp.sum(m_dev[pix] * wgt, axis=0)

        if method in ("nearest", "bilinear"):
            v = xp.asarray(self.pixel_centers_vec().reshape(-1, 3))
            img = _sample(v).reshape(n, n)
        elif method == "area":
            fine = TangentPlane(self.center_vec, self.size_deg, n * oversample,
                                roll_deg=self.roll_deg, view=self.view,
                                projection=self.projection)
            v = xp.asarray(fine.pixel_centers_vec().reshape(-1, 3))
            img = _sample(v).reshape(n, oversample, n, oversample).mean(axis=(1, 3))
        else:
            raise ValueError("method must be 'nearest', 'bilinear' or 'area'")

        img = to_host(img)
        bad = ~np.isfinite(img)
        return np.where(bad, fill, img) if bad.any() else img


def flat_sky_diagnostics(plane, tol=0.01, n_probe=None):
    """Quantify the gnomonic (flat-sky) approximation over ``plane``.

    Gnomonic scale factors at angular distance ``c`` from the tangent point are
    ``sec c`` (tangential), ``sec^2 c`` (radial) and ``sec^3 c`` (area magnification:
    plane area per unit sky solid angle). All grow with ``c``, so the worst case is
    the patch corner at ``c_max = arctan(sqrt(2) * tan(size/2))``.

    Returns a dict with:

    * ``area_distortion_max``  -- ``sec^3(c_max) - 1``, the point area magnification.
    * ``anisotropy_max``       -- ``sec(c_max) - 1``, the radial/tangential shear.
    * ``max_distance_error``   -- ``(tan c_max - c_max) / c_max``, the fractional error
      from equating projected radius with true angular distance.
    * ``empirical_solidangle_range`` -- ``max/min - 1`` of the true pixel solid angle
      over the actual grid (real non-uniformity of the sky sampling).
    * ``empirical_distance_error``   -- ``max |rho - c| / c`` sampled on the grid.
    * ``ok`` -- ``area_distortion_max <= tol``.

    (The closed-form and empirical figures agree only to leading order in ``c_max``;
    the empirical ones describe the finite pixel grid that :meth:`project_map` uses.)
    """
    n = plane.npix if n_probe is None else int(n_probe)
    half = np.deg2rad(plane.size_deg) / 2.0

    c_max = np.arctan(np.sqrt(2.0) * np.tan(half))
    sec = 1.0 / np.cos(c_max)
    area_dist_max = sec ** 3 - 1.0
    aniso_max = sec - 1.0
    dist_err_cf = (np.tan(c_max) - c_max) / c_max

    probe = TangentPlane(plane.center_vec, plane.size_deg, n, roll_deg=plane.roll_deg,
                         view=plane.view, projection=plane.projection)
    solid = np.abs(spherical_polygon_area(probe.pixel_corners_vec()))
    emp_sa_range = float(solid.max() / solid.min() - 1.0)

    centers = probe.pixel_centers_vec().reshape(-1, 3)
    gc = great_circle_distance(plane.center_vec, centers)
    rho = np.linalg.norm(gnomonic_forward(centers, probe.frame)[0], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        emp_dist_err = float(np.max(np.abs(rho - gc) / np.where(gc > 0, gc, 1.0)))

    ok = bool(area_dist_max <= tol)
    return {
        "size_deg": plane.size_deg,
        "npix": plane.npix,
        "res_arcmin": plane.res_deg * 60.0,
        "c_max_deg": float(np.degrees(c_max)),
        "area_distortion_max": float(area_dist_max),
        "anisotropy_max": float(aniso_max),
        "max_distance_error": float(dist_err_cf),
        "empirical_solidangle_range": emp_sa_range,
        "empirical_distance_error": emp_dist_err,
        "tol": tol,
        "ok": ok,
        "recommendation": ("flat-sky approximation OK at this patch size" if ok else
                           "flat-sky distortion exceeds tol -- shrink size_deg or use "
                           "a curved-sky method"),
    }


def flat_sky_table(sizes_deg=(1, 2, 5, 10, 20, 30)):
    """Closed-form corner distortion vs patch size, for choosing ``size_deg``."""
    table = {}
    for s in sizes_deg:
        c = np.arctan(np.sqrt(2.0) * np.tan(np.deg2rad(s) / 2.0))
        table[s] = {
            "c_max_deg": float(np.degrees(c)),
            "area_distortion_max": float(1.0 / np.cos(c) ** 3 - 1.0),
            "anisotropy_max": float(1.0 / np.cos(c) - 1.0),
        }
    return table


TangentPlane.flat_sky_table = staticmethod(flat_sky_table)
