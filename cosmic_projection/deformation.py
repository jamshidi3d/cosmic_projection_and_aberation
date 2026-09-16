"""Deformation maps: continuous bijections of the unit sphere onto itself.

A :class:`Deformation` is a callable ``vecs (..., 3) -> (..., 3)`` on unit vectors. It
works on numpy or cupy arrays -- the array module is taken from the input.

Forward convention
------------------
A deformation maps a direction *as labelled on the source map* to the direction where
that patch of sky actually lies on the standard sphere. :func:`cosmic_projection.
resample.resample_deformed_map` "pushes" each source pixel polygon through it.
"""

from __future__ import annotations

from ._backend import array_module
from .geometry import as_vec3, lonlat2vec, normalize, vec2lonlat


class Deformation:
    """Base class. Subclasses implement ``__call__(vecs) -> vecs``."""

    def __call__(self, vecs):  # pragma: no cover - abstract
        raise NotImplementedError

    def inverse_map(self):
        """Return the inverse deformation, if one is available analytically."""
        raise NotImplementedError("no analytic inverse for this deformation")


class IdentityDeformation(Deformation):
    """The identity map (baseline / testing)."""

    def __call__(self, vecs):
        return normalize(as_vec3(vecs))

    def inverse_map(self):
        return self

    def __repr__(self):
        return "IdentityDeformation()"


class CallableDeformation(Deformation):
    """Wrap a user function.

    ``kind="vec"``    -> ``fn(vecs (...,3)) -> (...,3)`` (result is renormalized).
    ``kind="lonlat"`` -> ``fn(lon_deg, lat_deg) -> (lon_deg, lat_deg)``.
    """

    def __init__(self, fn, kind="vec"):
        if kind not in ("vec", "lonlat"):
            raise ValueError("kind must be 'vec' or 'lonlat'")
        self.fn = fn
        self.kind = kind

    def __call__(self, vecs):
        vecs = normalize(as_vec3(vecs))
        if self.kind == "vec":
            return normalize(as_vec3(self.fn(vecs)))
        lon, lat = vec2lonlat(vecs)
        out_lon, out_lat = self.fn(lon, lat)
        return lonlat2vec(out_lon, out_lat)

    def __repr__(self):
        return f"CallableDeformation({self.fn!r}, kind={self.kind!r})"


class AberrationDeformation(Deformation):
    """Relativistic aberration of photon arrival directions.

    For a boost of speed ``beta`` (units of ``c``) toward ``direction``, a viewing
    direction at angle ``theta`` from the boost axis is mapped to ``theta'`` with::

        cos theta' = (cos theta + beta) / (1 + beta * cos theta)

    azimuth about the axis unchanged. ``inverse=True`` applies the de-boosting transform.

    Purely geometric: the Doppler intensity modulation (``~ (1 + beta cos theta)``
    brightness factor) is *not* applied. See ``FUTURE_PROJECTIONS.md`` / ``GPU_NOTES.md``.
    """

    def __init__(self, direction, beta, inverse=False, kind="vec"):
        if kind == "lonlat":
            direction = lonlat2vec(direction[0], direction[1])
        self.axis = normalize(as_vec3(direction))
        self.beta = float(beta)
        if not -1.0 < self.beta < 1.0:
            raise ValueError("beta must lie in (-1, 1)")
        self.inverse = bool(inverse)

    def __call__(self, vecs):
        xp = array_module(vecs)
        vecs = normalize(as_vec3(vecs))
        axis = xp.asarray(self.axis)                      # match the input's module
        beta = -self.beta if self.inverse else self.beta

        mu = xp.sum(vecs * axis, axis=-1)                 # cos(theta)
        mu_p = (mu + beta) / (1.0 + beta * mu)            # cos(theta')

        perp = vecs - mu[..., None] * axis
        perp_norm = xp.linalg.norm(perp, axis=-1, keepdims=True)
        perp_hat = xp.where(perp_norm > 1e-12,
                            perp / xp.where(perp_norm > 1e-12, perp_norm, 1.0), 0.0)
        sin_p = xp.sqrt(xp.clip(1.0 - mu_p * mu_p, 0.0, 1.0))[..., None]
        return normalize(mu_p[..., None] * axis + sin_p * perp_hat)

    def inverse_map(self):
        return AberrationDeformation(self.axis, self.beta, inverse=not self.inverse)

    def __repr__(self):
        return (f"AberrationDeformation(axis={list(self.axis)}, beta={self.beta}, "
                f"inverse={self.inverse})")
