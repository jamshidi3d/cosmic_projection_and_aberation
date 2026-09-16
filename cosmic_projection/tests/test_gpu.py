"""GPU (CuPy) backend tests.

Skipped unless both ``cupy`` and ``healpy`` import and a CUDA device is visible. Run
from the WSL conda env:  ``python -m pytest cosmic_projection/tests/test_gpu.py -q``
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
hp = pytest.importorskip("healpy")

from cosmic_projection._backend import gpu_available
from cosmic_projection import _healpix_gpu as hg
from cosmic_projection.deformation import AberrationDeformation, IdentityDeformation
from cosmic_projection.geometry import lonlat2vec
from cosmic_projection.plane import TangentPlane
from cosmic_projection.resample import resample_deformed_map

if not gpu_available():
    pytest.skip("no CUDA device visible to cupy", allow_module_level=True)

NSIDES = [8, 16, 32, 64]


def _smooth_map(nside, seed=0, lmax=None):
    rng = np.random.default_rng(seed)
    lmax = lmax or (2 * nside)
    size = hp.Alm.getsize(lmax)
    alm = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(complex)
    m = hp.alm2map(alm, nside, lmax=lmax)
    return m - m.mean() + 1.0


# --------------------------------------------------------------------------- #
# closed-form primitives vs healpy                                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("nside", NSIDES)
def test_vec2pix_ring_matches_healpy(nside):
    rng = np.random.default_rng(nside)
    v = rng.normal(size=(100_000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ref = hp.vec2pix(nside, v[:, 0], v[:, 1], v[:, 2], nest=False)
    got = cp.asnumpy(hg.vec2pix_ring(nside, cp.asarray(v)))
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("nside", NSIDES)
def test_pix2vec_ring_matches_healpy(nside):
    ip = np.arange(hp.nside2npix(nside))
    ref = np.array(hp.pix2vec(nside, ip, nest=False)).T
    got = cp.asnumpy(hg.pix2vec_ring(nside, cp.asarray(ip)))
    np.testing.assert_allclose(got, ref, atol=1e-12)


@pytest.mark.parametrize("nside", NSIDES)
def test_pix2vec_vec2pix_roundtrip(nside):
    ip = cp.arange(hp.nside2npix(nside))
    assert cp.array_equal(hg.vec2pix_ring(nside, hg.pix2vec_ring(nside, ip)), ip)


@pytest.mark.parametrize("nside", [16, 64])
def test_query_disc_ring_brackets_healpy(nside):
    rng = np.random.default_rng(nside)
    centers = rng.normal(size=(40, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers[0] = [0, 0, 1.0]          # north pole
    centers[1] = [0, 0, -1.0]         # south pole
    radii = np.deg2rad(rng.uniform(0.5, 60.0, size=40))

    did, ipx = hg.query_disc_ring(nside, cp.asarray(centers), cp.asarray(radii),
                                  safety=0)
    did = cp.asnumpy(did)
    ipx = cp.asnumpy(ipx)
    for d in range(len(centers)):
        got = set(ipx[did == d].tolist())
        strict = set(hp.query_disc(nside, centers[d], radii[d], inclusive=False))
        loose = set(hp.query_disc(nside, centers[d], radii[d], inclusive=True, fact=4))
        assert strict <= got <= loose, f"disc {d}: |got|={len(got)} not bracketed"


@pytest.mark.parametrize("nside", [16, 64])
def test_get_interp_weights_ring_reasonable(nside):
    m = _smooth_map(nside, seed=1)
    rng = np.random.default_rng(nside)
    th = np.arccos(rng.uniform(-0.9, 0.9, size=5000))
    ph = rng.uniform(0, 2 * np.pi, size=5000)
    pix, wgt = hg.get_interp_weights_ring(nside, cp.asarray(th), cp.asarray(ph))
    wgt = cp.asnumpy(wgt)
    np.testing.assert_allclose(wgt.sum(axis=0), 1.0, atol=1e-10)
    assert np.all(wgt >= -1e-9)
    got = (cp.asnumpy(cp.asarray(m)[pix]) * wgt).sum(axis=0)
    ref = hp.get_interp_val(m, th, ph)
    assert np.corrcoef(got, ref)[0, 1] > 0.999
    assert np.max(np.abs(got - ref)) < 0.05 * (m.max() - m.min())


# --------------------------------------------------------------------------- #
# end-to-end GPU vs CPU                                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("beta", [0.05, 0.3, 0.7])
def test_polygon_gpu_matches_cpu(beta):
    nside = 64
    m = _smooth_map(nside, seed=2)
    defm = AberrationDeformation([0.3, 0.2, 0.9], beta)
    cpu = resample_deformed_map(m, defm, nside, method="polygon", step=3, backend="cpu")
    gpu = resample_deformed_map(m, defm, nside, method="polygon", step=3, backend="gpu")
    good = (cpu != hp.UNSEEN) & (gpu != hp.UNSEEN)
    assert good.mean() > 0.98
    assert np.max(np.abs(cpu[good] - gpu[good])) < 1e-6


def test_polygon_gpu_matches_cpu_upgrade():
    m = _smooth_map(64, seed=5)
    defm = AberrationDeformation([1, 0, 0], 0.2)
    cpu = resample_deformed_map(m, defm, 128, method="polygon", step=3, backend="cpu")
    gpu = resample_deformed_map(m, defm, 128, method="polygon", step=3, backend="gpu")
    good = (cpu != hp.UNSEEN) & (gpu != hp.UNSEEN)
    assert np.max(np.abs(cpu[good] - gpu[good])) < 1e-6


def test_subpixel_gpu_matches_cpu():
    nside = 64
    m = _smooth_map(nside, seed=3)
    defm = AberrationDeformation([0, 0, 1], 0.1)
    cpu = resample_deformed_map(m, defm, nside, method="subpixel", factor=8, backend="cpu")
    gpu = resample_deformed_map(m, defm, nside, method="subpixel", factor=8, backend="gpu")
    good = (cpu != hp.UNSEEN) & (gpu != hp.UNSEEN)
    assert np.max(np.abs(cpu[good] - gpu[good])) < 1e-9


@pytest.mark.parametrize("method", ["subpixel", "polygon"])
def test_constant_map_stays_constant_on_gpu(method):
    nside = 32
    m = np.full(hp.nside2npix(nside), 2.5)
    defm = AberrationDeformation([0, 0, 1], 0.5)
    kw = dict(factor=4) if method == "subpixel" else dict(step=3)
    out = resample_deformed_map(m, defm, nside, method=method, backend="gpu", **kw)
    good = out != hp.UNSEEN
    np.testing.assert_allclose(out[good], 2.5, atol=1e-9)


def test_gpu_coverage_matches_cpu():
    nside = 32
    m = _smooth_map(nside, seed=8)
    defm = AberrationDeformation([0.2, 0.9, 0.1], 0.4)
    _, w_cpu = resample_deformed_map(m, defm, nside, method="polygon", step=3,
                                     backend="cpu", return_weights=True)
    _, w_gpu = resample_deformed_map(m, defm, nside, method="polygon", step=3,
                                     backend="gpu", return_weights=True)
    np.testing.assert_allclose(w_gpu, w_cpu, rtol=1e-4, atol=1e-6 * hp.nside2pixarea(nside))


def test_identity_gpu_polygon_reproduces_map():
    nside = 32
    m = _smooth_map(nside, seed=9)
    out = resample_deformed_map(m, IdentityDeformation(), nside, method="polygon",
                                step=3, backend="gpu")
    good = out != hp.UNSEEN
    assert good.mean() > 0.99
    np.testing.assert_allclose(out[good], m[good], atol=1e-2)


# --------------------------------------------------------------------------- #
# TangentPlane.project_map on GPU                                             #
# --------------------------------------------------------------------------- #

def test_project_map_nearest_gpu_exact():
    nside = 64
    m = _smooth_map(nside, seed=4)
    plane = TangentPlane(lonlat2vec(30.0, 20.0), size_deg=6.0, npix=48, roll_deg=15.0)
    cpu = plane.project_map(m, method="nearest", backend="cpu")
    gpu = plane.project_map(m, method="nearest", backend="gpu")
    np.testing.assert_array_equal(cpu, gpu)


def test_project_map_bilinear_gpu_close():
    nside = 64
    m = _smooth_map(nside, seed=6)
    plane = TangentPlane(lonlat2vec(-40.0, 55.0), size_deg=5.0, npix=50)
    cpu = plane.project_map(m, method="bilinear", backend="cpu")
    gpu = plane.project_map(m, method="bilinear", backend="gpu")
    assert np.max(np.abs(cpu - gpu)) < 0.02 * (m.max() - m.min())


def test_project_map_constant_gpu():
    nside = 32
    m = np.full(hp.nside2npix(nside), 3.0)
    plane = TangentPlane(lonlat2vec(10.0, -20.0), size_deg=8.0, npix=40)
    for method in ("nearest", "bilinear", "area"):
        img = plane.project_map(m, method=method, backend="gpu")
        np.testing.assert_allclose(img, 3.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# opt-in benchmark                                                           #
# --------------------------------------------------------------------------- #

@pytest.mark.skip(reason="benchmark: run explicitly with -k benchmark --no-header -s")
def test_benchmark_polygon_speedup():
    import time
    for nside in (128, 256):
        m = _smooth_map(nside, seed=0)
        defm = AberrationDeformation([0, 0, 1], 0.3)
        t0 = time.perf_counter()
        a = resample_deformed_map(m, defm, nside, method="polygon", backend="cpu")
        t_cpu = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = resample_deformed_map(m, defm, nside, method="polygon", backend="gpu")
        cp.cuda.Device().synchronize()
        t_gpu = time.perf_counter() - t0
        good = (a != hp.UNSEEN) & (b != hp.UNSEEN)
        print(f"\nnside={nside}: CPU {t_cpu:.2f}s  GPU {t_gpu:.2f}s  "
              f"speedup {t_cpu / t_gpu:.1f}x  maxdiff {np.max(np.abs(a[good]-b[good])):.2e}")
