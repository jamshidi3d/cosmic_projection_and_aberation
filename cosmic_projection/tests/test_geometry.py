import numpy as np
import pytest

from cosmic_projection.geometry import (
    great_circle_distance, lonlat2vec, rotation_matrix, spherical_polygon_area,
    tangent_frame, vec2lonlat,
)


def test_lonlat_vec_roundtrip():
    lon = np.array([0.0, 30.0, 123.4, 359.9, 200.0])
    lat = np.array([0.0, -45.0, 80.0, -10.0, 12.5])
    v = lonlat2vec(lon, lat)
    np.testing.assert_allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-12)
    lon2, lat2 = vec2lonlat(v)
    np.testing.assert_allclose(lon2, lon, atol=1e-9)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)


def test_great_circle_distance_known_values():
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    assert great_circle_distance(x, x) == pytest.approx(0.0, abs=1e-12)
    assert great_circle_distance(x, y) == pytest.approx(np.pi / 2, abs=1e-12)
    assert great_circle_distance(x, -x) == pytest.approx(np.pi, abs=1e-9)


def test_spherical_polygon_area_octant():
    octant = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert abs(spherical_polygon_area(octant)) == pytest.approx(np.pi / 2, rel=1e-12)


def _area_via_interior_angles(verts):
    """Independent area estimate for a convex spherical polygon: spherical excess."""
    n = len(verts)
    total = 0.0
    for i in range(n):
        a, b, c = verts[i - 1], verts[i], verts[(i + 1) % n]
        ta = a - np.dot(a, b) * b
        tc = c - np.dot(c, b) * b
        ta /= np.linalg.norm(ta)
        tc /= np.linalg.norm(tc)
        total += np.arccos(np.clip(np.dot(ta, tc), -1.0, 1.0))
    return total - (n - 2) * np.pi


def test_spherical_polygon_area_matches_excess_formula():
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    lat = np.full(4, 45.0)
    verts = lonlat2vec(lon, lat)
    assert abs(spherical_polygon_area(verts)) == pytest.approx(
        _area_via_interior_angles(verts), rel=1e-9)


def test_spherical_polygon_area_sign_flips_with_winding():
    octant = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    a = spherical_polygon_area(octant)
    b = spherical_polygon_area(octant[::-1])
    assert np.sign(a) == -np.sign(b)


def test_rotation_matrix_orthogonal_and_angle():
    r = rotation_matrix([0, 0, 1], np.pi / 2)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(r @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)


def test_rotation_matrix_batched():
    ang = np.array([0.0, np.pi / 2, np.pi])
    r = rotation_matrix([0, 0, 1], ang)
    assert r.shape == (3, 3, 3)
    np.testing.assert_allclose(r[0], np.eye(3), atol=1e-12)


def test_tangent_frame_orthonormal_right_handed():
    for center in ([1.0, 0, 0], [0, 0, 1], [1, 1, 1], [0.2, -0.9, 0.3]):
        f = tangent_frame(center, roll_deg=0.0)
        np.testing.assert_allclose(f @ f.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.cross(f[0], f[1]), f[2], atol=1e-12)


def test_tangent_frame_normal_direction_by_view():
    c = np.array([0.3, 0.4, np.sqrt(1 - 0.25)])
    fin = tangent_frame(c, view="inside")
    fout = tangent_frame(c, view="outside")
    np.testing.assert_allclose(fin[2], c, atol=1e-12)
    np.testing.assert_allclose(fout[2], -c, atol=1e-12)


def test_tangent_frame_roll_composes_additively():
    c = [0.2, -0.9, 0.3]
    f0 = tangent_frame(c, roll_deg=0.0)
    f30 = tangent_frame(c, roll_deg=30.0)
    f = tangent_frame(c, roll_deg=70.0)
    f_expected_ex = rotation_matrix(f0[2], np.deg2rad(70.0)) @ f0[0]
    np.testing.assert_allclose(f[0], f_expected_ex, atol=1e-12)
    # 30 then 40 == 70
    ex_2step = rotation_matrix(f0[2], np.deg2rad(40.0)) @ f30[0]
    np.testing.assert_allclose(f[0], ex_2step, atol=1e-12)
