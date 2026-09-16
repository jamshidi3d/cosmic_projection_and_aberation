"""Gnomonic square cutout of a HEALPix map, with a flat-sky assessment.

Run from the project root (with healpy available):

    python -m cosmic_projection.examples.demo_flatsky_cutout
"""

import numpy as np
import healpy as hp

from cosmic_projection import (TangentPlane, flat_sky_diagnostics, flat_sky_table,
                               lonlat2vec)

NSIDE = 128
CENTER = lonlat2vec(80.0, 20.0)


def make_map(nside):
    rng = np.random.default_rng(7)
    lmax = 3 * nside - 1
    cl = 1.0 / (np.arange(lmax + 1) + 5.0) ** 2.5
    return hp.synfast(cl, nside, new=True)


def main():
    m = make_map(NSIDE)

    print("flat-sky corner distortion vs patch size:")
    for size, row in flat_sky_table((2, 5, 10, 20, 30)).items():
        print(f"  {size:>2} deg : area x{1 + row['area_distortion_max']:.4f}  "
              f"(c_max {row['c_max_deg']:.1f} deg)")

    for size, roll in [(5.0, 0.0), (10.0, 30.0)]:
        plane = TangentPlane.matching_healpix(CENTER, NSIDE, size_deg=size,
                                              roll_deg=roll)
        img = plane.project_map(m, method="area", oversample=3)
        diag = flat_sky_diagnostics(plane)
        print(f"\npatch {size} deg, roll {roll} deg -> {img.shape} image, "
              f"res {plane.res_deg * 60:.2f} arcmin")
        print(f"  flat-sky ok={diag['ok']}  area_distortion_max="
              f"{diag['area_distortion_max']:.2e}  "
              f"max_distance_error={diag['max_distance_error']:.2e}")
        fname = f"demo_cutout_{int(size)}deg.npy"
        np.save(fname, img)
        print(f"  saved {fname}")


if __name__ == "__main__":
    main()
