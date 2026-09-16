# skewness_abberation_relation

PhD research code on the **skewness–aberration relation** (CMB aberration / Doppler-boosting
studies).

## Contents

- [`cosmic_projection/`](cosmic_projection/README.md) — area-weighted 3-D projection of
  spherical (HEALPix) maps: resampling a *deformed* sphere map (e.g. under relativistic
  aberration) onto a standard HEALPix grid, and gnomonic square cutouts of a HEALPix map
  with a flat-sky accuracy assessment. Includes an optional CuPy GPU backend for the
  resampler. See that package's README for the API and usage.

## Install

```bash
pip install -e .
```

See [`cosmic_projection/README.md`](cosmic_projection/README.md) for dependencies
(`numpy`, `healpy`; optional `cupy` for the GPU backend) and environment notes.
