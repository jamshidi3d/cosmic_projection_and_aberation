"""CPU vs GPU timing for the exact (polygon) resampler, with a correctness check.

Run from the project root in the WSL env (healpy + cupy):

    python -m cosmic_projection.examples.demo_gpu_benchmark
"""

import time

import numpy as np
import healpy as hp

from cosmic_projection import (AberrationDeformation, gpu_available,
                               resample_deformed_map)

BETA = 0.3
BOOST = hp.ang2vec(np.deg2rad(55.0), np.deg2rad(120.0))


def make_map(nside):
    cl = 1.0 / (np.arange(3 * nside) + 10.0) ** 2
    m = hp.synfast(cl, nside, new=True)
    return m - m.min() + 1.0


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main():
    if not gpu_available():
        print("no CUDA device visible to cupy -- GPU path unavailable")
        return

    defm = AberrationDeformation(BOOST, BETA)
    print(f"exact (polygon) resample, aberration beta={BETA}\n")
    print(f"{'nside':>6} {'CPU (s)':>10} {'GPU (s)':>10} {'speedup':>9} {'max|diff|':>12}")

    for nside in (64, 128, 256):
        m = make_map(nside)
        cpu, t_cpu = timed(lambda: resample_deformed_map(
            m, defm, nside, method="polygon", step=3, backend="cpu"))
        gpu, t_gpu = timed(lambda: resample_deformed_map(
            m, defm, nside, method="polygon", step=3, backend="gpu"))
        good = (cpu != hp.UNSEEN) & (gpu != hp.UNSEEN)
        d = float(np.max(np.abs(cpu[good] - gpu[good])))
        print(f"{nside:>6} {t_cpu:>10.2f} {t_gpu:>10.2f} {t_cpu / t_gpu:>8.1f}x {d:>12.2e}")

    # a larger run, GPU only
    nside = 512
    m = make_map(nside)
    gpu, t_gpu = timed(lambda: resample_deformed_map(
        m, defm, nside, method="polygon", step=3, backend="gpu", dtype="float32"))
    print(f"\nGPU-only  nside={nside} (float32 tables): {t_gpu:.2f} s, "
          f"covered {(gpu != hp.UNSEEN).mean():.3f}")


if __name__ == "__main__":
    main()
