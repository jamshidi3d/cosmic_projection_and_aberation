import numpy as np
import pytest

hp = pytest.importorskip("healpy")

from cosmic_projection.deformation import AberrationDeformation, IdentityDeformation
from cosmic_projection.resample import resample_deformed_map


def _smooth_map(nside, seed=0):
    """A band-limited map (ell <= 8) so neighbouring pixels are similar."""
    rng = np.random.default_rng(seed)
    lmax = 8
    alm = rng.normal(size=hp.Alm.getsize(lmax)) + 1j * rng.normal(size=hp.Alm.getsize(lmax))
    m = hp.alm2map(alm.astype(complex), nside, lmax=lmax)
    return m - m.mean() + 1.0     # mean ~ 1, some structure


def test_identity_same_nside_subpixel_is_near_exact():
    nside = 16
    m = _smooth_map(nside)
    out = resample_deformed_map(m, IdentityDeformation(), nside, method="subpixel",
                                factor=4)
    np.testing.assert_allclose(out, m, atol=1e-12)


def test_identity_downgrade_matches_ud_grade():
    nside = 16
    m = _smooth_map(nside)
    out = resample_deformed_map(m, IdentityDeformation(), nside // 2, method="subpixel",
                                factor=8)
    ref = hp.ud_grade(m, nside // 2)
    np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)


def test_identity_same_nside_polygon_reproduces_smooth_map():
    nside = 8
    m = _smooth_map(nside)
    out = resample_deformed_map(m, IdentityDeformation(), nside, method="polygon",
                                step=3)
    good = out != hp.UNSEEN
    assert good.mean() > 0.99
    np.testing.assert_allclose(out[good], m[good], atol=1.5e-2)


@pytest.mark.parametrize("method", ["subpixel", "polygon"])
def test_constant_map_stays_constant_under_aberration(method):
    nside = 8
    m = np.full(hp.nside2npix(nside), 2.5)
    defm = AberrationDeformation(direction=[0, 0, 1], beta=1.23e-3)
    kw = dict(factor=4) if method == "subpixel" else dict(step=3)
    out = resample_deformed_map(m, defm, nside, method=method, **kw)
    good = out != hp.UNSEEN
    np.testing.assert_allclose(out[good], 2.5, atol=1e-9)


def test_flux_conserved_small_beta():
    nside = 16
    m = _smooth_map(nside, seed=3)
    defm = AberrationDeformation(direction=[1, 0, 0], beta=1e-3)
    out, w = resample_deformed_map(m, defm, nside, method="subpixel", factor=8,
                                   return_weights=True)
    good = out != hp.UNSEEN
    flux_in = m.sum() * hp.nside2pixarea(nside)
    flux_out = (out[good] * w[good]).sum()
    assert flux_out == pytest.approx(flux_in, rel=3e-3)


def test_subpixel_and_polygon_agree_on_smooth_map():
    nside = 8
    m = _smooth_map(nside, seed=7)
    defm = AberrationDeformation(direction=[0.3, 0.2, 0.9], beta=0.05)
    a = resample_deformed_map(m, defm, nside, method="subpixel", factor=16)
    b = resample_deformed_map(m, defm, nside, method="polygon", step=4)
    good = (a != hp.UNSEEN) & (b != hp.UNSEEN)
    assert good.mean() > 0.95
    np.testing.assert_allclose(a[good], b[good], atol=2e-2, rtol=2e-2)


def test_pixel_indices_restricts_to_patch():
    nside = 16
    m = _smooth_map(nside)
    disc = hp.query_disc(nside, hp.ang2vec(np.pi / 2, 0.0), np.deg2rad(20))
    out = resample_deformed_map(m, IdentityDeformation(), nside, method="subpixel",
                                factor=4, pixel_indices=disc)
    covered = out != hp.UNSEEN
    # coverage should be localised near the requested disc, not the whole sky
    assert covered.sum() < hp.nside2npix(nside) // 4
    assert covered.sum() >= len(disc) - 5
