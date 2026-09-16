import numpy as np
import pytest

hp = pytest.importorskip("healpy")

from cosmic_projection.geometry import lonlat2vec, great_circle_distance
from cosmic_projection.plane import TangentPlane, flat_sky_diagnostics, flat_sky_table


def _smooth_map(nside, seed=0):
    rng = np.random.default_rng(seed)
    lmax = 16
    size = hp.Alm.getsize(lmax)
    alm = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(complex)
    return hp.alm2map(alm, nside, lmax=lmax)


def test_center_pixel_points_at_plane_center():
    c = lonlat2vec(40.0, 25.0)
    p = TangentPlane(c, size_deg=10.0, npix=51)      # odd -> a pixel sits on centre
    centers = p.pixel_centers_vec()
    mid = centers[25, 25]
    assert great_circle_distance(c, mid) < np.deg2rad(0.2)


def test_constant_map_gives_constant_image():
    nside = 32
    m = np.full(hp.nside2npix(nside), 3.0)
    p = TangentPlane(lonlat2vec(10.0, -20.0), size_deg=8.0, npix=40)
    for method in ("nearest", "bilinear", "area"):
        img = p.project_map(m, method=method)
        np.testing.assert_allclose(img, 3.0, atol=1e-9)


def test_matching_healpix_pixel_scale():
    nside = 64
    p = TangentPlane.matching_healpix(lonlat2vec(0.0, 0.0), nside, size_deg=10.0)
    assert p.res_deg == pytest.approx(np.degrees(hp.nside2resol(nside)), rel=0.1)


def test_roll_180_point_reflects_image():
    nside = 64
    m = _smooth_map(nside, seed=1)
    c = lonlat2vec(30.0, 40.0)
    base = TangentPlane(c, size_deg=6.0, npix=60, roll_deg=0.0).project_map(m)
    rolled = TangentPlane(c, size_deg=6.0, npix=60, roll_deg=180.0).project_map(m)
    # roll by 180 deg flips e_x and e_y -> image is point-reflected through its centre
    np.testing.assert_allclose(rolled, base[::-1, ::-1], atol=5e-3, rtol=5e-3)


def test_roll_90_is_a_quarter_turn():
    nside = 64
    m = _smooth_map(nside, seed=1)
    c = lonlat2vec(30.0, 40.0)
    base = TangentPlane(c, size_deg=6.0, npix=60, roll_deg=0.0).project_map(m)
    rolled = TangentPlane(c, size_deg=6.0, npix=60, roll_deg=90.0).project_map(m)
    # a pure frame rotation -> the image is a quarter turn of the roll=0 image
    # (resampling onto a rotated grid leaves ~1% interpolation residual)
    err = [np.max(np.abs(rolled - np.rot90(base, k))) for k in (1, -1)]
    assert min(err) < 0.05 * (base.max() - base.min())


def test_flat_sky_diagnostics_small_patch_ok():
    p = TangentPlane(lonlat2vec(0.0, 0.0), size_deg=0.5, npix=64)
    d = flat_sky_diagnostics(p)
    assert d["ok"]
    assert d["area_distortion_max"] < 1e-4


def test_flat_sky_diagnostics_large_patch_flagged():
    p = TangentPlane(lonlat2vec(0.0, 0.0), size_deg=30.0, npix=64)
    d = flat_sky_diagnostics(p)
    assert not d["ok"]
    # corner distance for a 30 deg square: c_max = arctan(sqrt2 * tan 15 deg) ~ 20.75 deg
    assert d["c_max_deg"] == pytest.approx(20.75, abs=0.05)
    assert d["area_distortion_max"] == pytest.approx(1 / np.cos(np.deg2rad(20.75)) ** 3 - 1,
                                                     rel=2e-3)
    assert d["anisotropy_max"] == pytest.approx(1 / np.cos(np.deg2rad(20.75)) - 1,
                                                rel=2e-3)


def test_flat_sky_diagnostics_distance_error_closed_form_vs_empirical():
    p = TangentPlane(lonlat2vec(0.0, 0.0), size_deg=20.0, npix=80)
    d = flat_sky_diagnostics(p)
    # (tan c - c)/c and the grid-sampled |rho - c|/c agree to leading order
    assert d["empirical_distance_error"] == pytest.approx(d["max_distance_error"],
                                                          rel=0.1)
    assert d["empirical_solidangle_range"] > 0.0


def test_flat_sky_diagnostics_metrics_shrink_with_patch():
    small = flat_sky_diagnostics(TangentPlane(lonlat2vec(0.0, 0.0), 2.0, 40))
    big = flat_sky_diagnostics(TangentPlane(lonlat2vec(0.0, 0.0), 20.0, 40))
    for key in ("area_distortion_max", "anisotropy_max", "max_distance_error",
                "empirical_solidangle_range", "empirical_distance_error"):
        assert small[key] < big[key]


def test_flat_sky_table_monotonic():
    t = flat_sky_table((1, 5, 10, 20))
    vals = [t[s]["area_distortion_max"] for s in (1, 5, 10, 20)]
    assert all(x < y for x, y in zip(vals, vals[1:]))


def test_project_matches_direct_healpy_interp_at_center():
    nside = 64
    m = _smooth_map(nside, seed=2)
    c = lonlat2vec(-15.0, 55.0)
    p = TangentPlane(c, size_deg=4.0, npix=41)
    img = p.project_map(m, method="bilinear")
    th, ph = hp.vec2ang(c.reshape(1, 3))
    assert img[20, 20] == pytest.approx(float(hp.get_interp_val(m, th, ph)[0]), rel=1e-3)
