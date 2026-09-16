import numpy as np
import pytest

from cosmic_projection.geometry import lonlat2vec, spherical_polygon_area
from cosmic_projection.overlap import (
    gnomonic_forward, gnomonic_inverse, polygon_area_2d,
    spherical_polygon_overlap_area, sutherland_hodgman,
)


def _square(cx, cy, h):
    return np.array([[cx - h, cy - h], [cx + h, cy - h],
                     [cx + h, cy + h], [cx - h, cy + h]])


def test_sutherland_hodgman_identical_squares():
    s = _square(0, 0, 1)
    clipped = sutherland_hodgman(s, s)
    assert polygon_area_2d(clipped) == pytest.approx(4.0, rel=1e-9)


def test_sutherland_hodgman_half_overlap():
    subj = _square(0, 0, 1)
    clip = _square(1, 0, 1)          # overlap is x in [0,1], y in [-1,1] -> area 2
    clipped = sutherland_hodgman(subj, clip)
    assert polygon_area_2d(clipped) == pytest.approx(2.0, rel=1e-9)


def test_sutherland_hodgman_disjoint():
    clipped = sutherland_hodgman(_square(0, 0, 1), _square(10, 0, 1))
    assert polygon_area_2d(clipped) == pytest.approx(0.0, abs=1e-12)


def test_sutherland_hodgman_clip_inside_subject():
    clipped = sutherland_hodgman(_square(0, 0, 5), _square(0, 0, 1))
    assert polygon_area_2d(clipped) == pytest.approx(4.0, rel=1e-9)


def test_gnomonic_roundtrip():
    from cosmic_projection.geometry import tangent_frame
    frame = tangent_frame([0.3, -0.4, 0.6])
    xy = np.array([[0.0, 0.0], [0.1, 0.05], [-0.2, 0.15], [0.05, -0.3]])
    v = gnomonic_inverse(xy, frame)
    xy2, w = gnomonic_forward(v, frame)
    np.testing.assert_allclose(xy2, xy, atol=1e-12)
    assert np.all(w > 0)


def test_gnomonic_center_maps_to_origin():
    from cosmic_projection.geometry import tangent_frame
    c = lonlat2vec(42.0, -13.0)
    frame = tangent_frame(c)
    xy, w = gnomonic_forward(c, frame)
    np.testing.assert_allclose(xy, [0.0, 0.0], atol=1e-12)
    assert w == pytest.approx(1.0)


def test_overlap_identical_small_polygons():
    verts = lonlat2vec(np.array([-1.0, 1.0, 1.0, -1.0]), np.array([-1.0, -1.0, 1.0, 1.0]))
    omega = abs(spherical_polygon_area(verts))
    got = spherical_polygon_overlap_area(verts, verts)
    assert got == pytest.approx(omega, rel=1e-6)


def test_overlap_disjoint_small_polygons():
    a = lonlat2vec(np.array([-1.0, 1.0, 1.0, -1.0]), np.array([-1.0, -1.0, 1.0, 1.0]))
    b = lonlat2vec(np.array([9.0, 11.0, 11.0, 9.0]), np.array([-1.0, -1.0, 1.0, 1.0]))
    assert spherical_polygon_overlap_area(a, b) == pytest.approx(0.0, abs=1e-12)


def test_overlap_small_polygon_fully_inside_large():
    big = lonlat2vec(np.array([-5.0, 5.0, 5.0, -5.0]), np.array([-5.0, -5.0, 5.0, 5.0]))
    small = lonlat2vec(np.array([-1.0, 1.0, 1.0, -1.0]), np.array([-1.0, -1.0, 1.0, 1.0]))
    omega_small = abs(spherical_polygon_area(small))
    # clip big by small -> the small polygon; ref area = small's solid angle
    got = spherical_polygon_overlap_area(big, small)
    assert got == pytest.approx(omega_small, rel=1e-4)


def test_overlap_half():
    a = lonlat2vec(np.array([-2.0, 2.0, 2.0, -2.0]), np.array([-2.0, -2.0, 2.0, 2.0]))
    b = lonlat2vec(np.array([0.0, 4.0, 4.0, 0.0]), np.array([-2.0, -2.0, 2.0, 2.0]))
    omega_b = abs(spherical_polygon_area(b))
    got = spherical_polygon_overlap_area(a, b)
    assert got == pytest.approx(0.5 * omega_b, rel=2e-3)
