"""cosmic_projection -- area-weighted 3-D projection of spherical maps.

Two operations:

1. :func:`resample_deformed_map` -- push a deformed spherical map onto a standard
   HEALPix grid with area-weighted accumulation (see :mod:`cosmic_projection.resample`).
2. :class:`TangentPlane` -- gnomonic square cutouts of a HEALPix map, plus
   :func:`flat_sky_diagnostics` to assess the flat-sky approximation
   (see :mod:`cosmic_projection.plane`).
"""

from __future__ import annotations

from ._backend import gpu_available, resolve_backend
from .geometry import (great_circle_distance, lonlat2vec, rotation_matrix,
                       spherical_polygon_area, tangent_frame, vec2lonlat)
from .deformation import (AberrationDeformation, CallableDeformation, Deformation,
                          IdentityDeformation)
from .overlap import (gnomonic_forward, gnomonic_inverse, polygon_area_2d,
                      spherical_polygon_overlap_area, sutherland_hodgman)
from .resample import resample_deformed_map
from .plane import TangentPlane, flat_sky_diagnostics, flat_sky_table

__version__ = "0.1.0"

__all__ = [
    "resample_deformed_map",
    "TangentPlane",
    "flat_sky_diagnostics",
    "flat_sky_table",
    "Deformation",
    "IdentityDeformation",
    "CallableDeformation",
    "AberrationDeformation",
    "great_circle_distance",
    "spherical_polygon_area",
    "spherical_polygon_overlap_area",
    "tangent_frame",
    "lonlat2vec",
    "vec2lonlat",
    "rotation_matrix",
    "gnomonic_forward",
    "gnomonic_inverse",
    "sutherland_hodgman",
    "polygon_area_2d",
    "gpu_available",
    "resolve_backend",
]
