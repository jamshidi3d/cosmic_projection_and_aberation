"""Aberrate a HEALPix map and resample it back onto the standard grid.

Run from the project root (with healpy available):

    python -m cosmic_projection.examples.demo_aberration_resample
"""

import numpy as np
import healpy as hp

from cosmic_projection import AberrationDeformation, resample_deformed_map

NSIDE = 64
BETA = 1.23e-3                     # ~ CMB dipole speed, v/c
BOOST_DIR = hp.ang2vec(np.deg2rad(90 - 48.0), np.deg2rad(264.0))   # ~ CMB dipole apex


def make_map(nside):
    rng = np.random.default_rng(42)
    lmax = 3 * nside - 1
    cl = 1.0 / (np.arange(lmax + 1) + 10.0) ** 2
    m = hp.synfast(cl, nside, new=True)
    return m - m.min() + 1.0


def main():
    m = make_map(NSIDE)
    defm = AberrationDeformation(BOOST_DIR, BETA)

    fast, w = resample_deformed_map(m, defm, NSIDE, method="subpixel", factor=8,
                                    return_weights=True)
    exact = resample_deformed_map(m, defm, NSIDE, method="polygon", step=3)

    good = (fast != hp.UNSEEN) & (exact != hp.UNSEEN)
    flux_in = m.sum() * hp.nside2pixarea(NSIDE)
    flux_out = (fast[good] * w[good]).sum()

    print(f"nside={NSIDE}  beta={BETA}")
    print(f"covered fraction         : {good.mean():.4f}")
    print(f"flux in / out            : {flux_in:.6e} / {flux_out:.6e}  "
          f"(rel diff {abs(flux_out / flux_in - 1):.2e})")
    print(f"subpixel vs polygon RMS  : {np.sqrt(np.mean((fast[good]-exact[good])**2)):.3e}")
    print(f"max |aberrated - original|: {np.max(np.abs(fast[good]-m[good])):.3e}")

    np.save("demo_aberration_subpixel.npy", fast)
    np.save("demo_aberration_polygon.npy", exact)
    print("saved demo_aberration_subpixel.npy, demo_aberration_polygon.npy")


if __name__ == "__main__":
    main()
