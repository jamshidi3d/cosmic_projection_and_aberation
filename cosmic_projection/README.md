# cosmic_projection

Area-weighted 3-D projection of spherical maps. The mental model: the pixels of a
spherical map are polygons of a mesh on the sphere; projecting a map means moving those
polygons and re-integrating them onto a new pixel grid, weighting by the **area** each
moved polygon contributes to each target pixel.

Two operations:

1. **Deformed sphere → standard HEALPix** — `resample_deformed_map`. A source map is
   pushed through a deformation `D: S² → S²` (e.g. relativistic aberration). Each source
   pixel polygon is moved by `D`; every output HEALPix pixel gets the area-weighted mean
   of the source fragments landing inside it:

   ```
   out[j] = Σ_i value_i · overlap_area(D(pixel_i), pixel_j) / Σ_i overlap_area(D(pixel_i), pixel_j)
   ```

2. **HEALPix → flat square plane** — `TangentPlane`. A square grid of near-equal-area
   pixels tangent to the sphere at a chosen direction, with the plane normal along (or
   opposite) the line of sight and a free roll angle about the normal. Plus
   `flat_sky_diagnostics` to quantify how good the flat-sky (gnomonic) approximation is
   over the patch.

Only the **gnomonic** projection is implemented. Orthographic / stereographic / the
reverse plane→HEALPix map are specified in [`FUTURE_PROJECTIONS.md`](FUTURE_PROJECTIONS.md).

---

## Install

```bash
pip install -r cosmic_projection/requirements.txt      # numpy, healpy
# or, from the repo root, as an editable package:
pip install -e .
```

`healpy` is a hard dependency. It installs cleanly on Linux / WSL; on Windows use a
recent wheel or run from WSL. `numpy` is the only other runtime requirement; `pytest`
is needed for the test suite.

**Optional GPU backend** (`resample_deformed_map(..., backend="gpu")`): install a CuPy
wheel matching your CUDA runtime, e.g. `pip install cupy-cuda12x` (works with CUDA 13
too), or `pip install -e ".[gpu]"`. See [`GPU_NOTES.md`](GPU_NOTES.md).

---

## Conventions

| Thing | Convention |
|---|---|
| Unit vectors | Cartesian components on the trailing axis, shape `(..., 3)`. |
| `lon`, `lat` | **degrees**, healpy `lonlat=True` convention (`lon` about `+z` from `+x`; `lat` from the equator). Helpers: `lonlat2vec`, `vec2lonlat`. |
| Rotation angles | **radians** (`rotation_matrix`). |
| HEALPix ordering | Public functions take/return **RING** by default; pass `nest=True` to switch. Internally the resampler works in NEST. |
| Uncovered pixels | Filled with `healpy.UNSEEN` unless you pass `fill=`. |

A `Deformation` is any callable `vecs (...,3) -> (...,3)` on unit vectors. Its **forward
direction** maps a direction *as labelled on the source map* to where that patch of sky
lands on the standard sphere (the resampler "pushes" source polygons through it).

---

## Feature 1 — resample a deformed map

```python
import numpy as np, healpy as hp
from cosmic_projection import AberrationDeformation, resample_deformed_map

m = hp.read_map("sky.fits")
nside = hp.get_nside(m)

# boost toward the CMB dipole apex, v/c = 1.23e-3
boost = AberrationDeformation(direction=hp.ang2vec(np.deg2rad(90 - 48.0),
                                                   np.deg2rad(264.0)),
                              beta=1.23e-3)

# fast: dense sub-pixel sampling (converges to area weighting as `factor` grows)
out = resample_deformed_map(m, boost, nside_out=nside, method="subpixel", factor=8)

# accurate: exact spherical-polygon overlap (slower; best for coarse maps / patches)
out2 = resample_deformed_map(m, boost, nside_out=nside, method="polygon", step=3)

# restrict the input to a patch of source pixels, and get the coverage weights back
patch = hp.query_disc(nside, hp.ang2vec(np.pi/2, 0.0), np.deg2rad(10))
out3, weight = resample_deformed_map(m, boost, nside_out=256, pixel_indices=patch,
                                     return_weights=True)
```

### `resample_deformed_map(source_map, deformation, nside_out, *, ...)`

| arg | default | meaning |
|---|---|---|
| `method` | `"subpixel"` | `"subpixel"` or `"polygon"` (see below). |
| `nest` | `False` | ordering of `source_map`, the returned map, and `pixel_indices`. |
| `factor` | `8` | `method="subpixel"`: sub-pixels per side per source pixel. **Must be a power of 2** (uses the HEALPix nested hierarchy); `factor²` samples per pixel. |
| `step` | `3` | `method="polygon"`: boundary samples per source-pixel edge. |
| `pixel_indices` | `None` | restrict the input to this subset of source pixels. |
| `fill` | `healpy.UNSEEN` | value for output pixels with no coverage. |
| `return_weights` | `False` | also return the accumulated coverage (effective solid angle) per output pixel — doubles as a hit mask. |
| `chunk` | `4096` | source pixels processed per batch (memory control; both methods, both backends). |
| `backend` | `"auto"` | `"cpu"`, `"gpu"`, or `"auto"` (GPU iff CuPy + a CUDA device are present). |
| `dtype` | `"float64"` | GPU geometry-table precision (`"float32"` halves their VRAM footprint; the overlap math stays float64). Ignored on CPU. |

**`method="subpixel"`** — each source pixel is split into `factor²` equal-area NEST
sub-pixels; their centres are pushed through the deformation and binned into output
pixels. Fast, vectorized, area-weighted in the limit of large `factor`. Good default for
full-sky work.

**`method="polygon"`** — each source pixel boundary is deformed and clipped against
every overlapping output pixel with `spherical_polygon_overlap_area` (Sutherland–Hodgman
in a local gnomonic chart, rescaled to steradians by the exact target-pixel solid
angle). More accurate on coarse maps where a pixel spans a large angle. The CPU path
loops over source pixels (use it on patches / moderate `nside`); the **GPU path**
(`backend="gpu"`) vectorizes the whole thing — candidate search via a batched
`query_disc`, all clips in one batched kernel — and stays valid for large β. GPU vs CPU
agree to ≲ 1e-6. See [`GPU_NOTES.md`](GPU_NOTES.md).

### Deformations

```python
from cosmic_projection import (Deformation, IdentityDeformation,
                               CallableDeformation, AberrationDeformation)

# wrap a function on vectors ...
d1 = CallableDeformation(lambda v: v @ R.T)               # kind="vec" (default)
# ... or on (lon_deg, lat_deg)
d2 = CallableDeformation(my_lonlat_shift, kind="lonlat")

# relativistic aberration of arrival directions:
#   cos θ' = (cos θ + β) / (1 + β cos θ)         (θ from the boost axis)
ab = AberrationDeformation(direction=[0, 0, 1], beta=1e-3)
ab_inv = ab.inverse_map()                                 # the de-boosting map

# custom: subclass Deformation and implement __call__(vecs) -> vecs
```

`AberrationDeformation` is **purely geometric** — it does not apply the Doppler
intensity modulation (`~ (1 + β cos θ)` brightness factor). A hook for that is sketched
in `FUTURE_PROJECTIONS.md`.

---

## Feature 2 — flat cutout

```python
from cosmic_projection import TangentPlane, flat_sky_diagnostics, flat_sky_table, lonlat2vec

# pixel scale matched to an nside=128 map, 10° square, rolled 30° about the normal
plane = TangentPlane.matching_healpix(lonlat2vec(80.0, 20.0), nside=128,
                                      size_deg=10.0, roll_deg=30.0, view="inside")

img = plane.project_map(m, method="area", oversample=3)   # -> (npix, npix) ndarray

diag = flat_sky_diagnostics(plane)
print(diag["ok"], diag["area_distortion_max"], diag["max_distance_error"])

# corner distortion vs patch size, for choosing size_deg up front
for size, row in flat_sky_table((2, 5, 10, 20, 30)).items():
    print(size, row["area_distortion_max"])
```

### `TangentPlane(center, size_deg, npix, roll_deg=0.0, view="inside", projection="gnomonic")`

| arg | meaning |
|---|---|
| `center` | `(lon_deg, lat_deg)` or a 3-vector — the tangent point. |
| `size_deg` | full angular width of the (square) patch. |
| `npix` | pixels per side. |
| `roll_deg` | right-handed rotation of the grid about the plane normal. |
| `view` | `"inside"` → normal along `+center` (observer at the origin looking out); `"outside"` → normal along `-center`. |
| `projection` | `"gnomonic"` only (others raise). |

- `TangentPlane.matching_healpix(center, nside, size_deg, **kw)` — sets
  `npix = round(size_deg / healpy.nside2resol(nside))` so flat pixels ≈ HEALPix pixel
  scale.
- `plane.res_deg` — pixel scale.
- `plane.pixel_centers_vec()` → `(npix, npix, 3)` unit vectors; row 0 is the top of the
  image, column 0 the left.
- `plane.pixel_corners_vec()` → `(npix, npix, 4, 3)` (corners TL, TR, BR, BL) — for
  area weighting or feeding an external 3-D / Blender mesh.
- `plane.project_map(hpx_map, *, method="bilinear", nest=False, oversample=4,
  fill=UNSEEN, backend="cpu")` → `(npix, npix)`:
  - `"nearest"` — nearest HEALPix pixel at each cell centre;
  - `"bilinear"` — `healpy.get_interp_val` at each cell centre;
  - `"area"` — mean of `oversample²` bilinear sub-samples per cell.
  - `backend="gpu"` (opt-in, not `"auto"`): `"nearest"` matches the CPU result exactly;
    `"bilinear"` / `"area"` use an independent on-device bilinear stencil (~1e-2 of the
    CPU result), so request `backend="cpu"` when you need healpy-identical interpolation.

### `flat_sky_diagnostics(plane, tol=0.01, n_probe=None)` → dict

Gnomonic scale factors at angular distance `c` from the tangent point are `sec c`
(tangential), `sec² c` (radial), `sec³ c` (area). All grow with `c`, so the worst case
is the patch corner, `c_max = arctan(√2 · tan(size/2))`.

| key | meaning |
|---|---|
| `c_max_deg` | corner angular distance. |
| `area_distortion_max` | `sec³(c_max) − 1` — point area magnification (plane area per unit sky solid angle). |
| `anisotropy_max` | `sec(c_max) − 1` — radial/tangential shear. |
| `max_distance_error` | `(tan c_max − c_max) / c_max` — error from equating projected radius with true angle. |
| `empirical_solidangle_range` | `max/min − 1` of the true pixel solid angle over the actual grid. |
| `empirical_distance_error` | `max |ρ − c| / c` sampled on the grid. |
| `ok` | `area_distortion_max ≤ tol`. |
| `recommendation` | one-line human-readable verdict. |
| `size_deg`, `npix`, `res_arcmin`, `tol` | echoed inputs. |

The closed-form and empirical figures agree only to leading order in `c_max`; the
empirical ones describe the finite pixel grid `project_map` actually uses.

`flat_sky_table(sizes_deg=(1, 2, 5, 10, 20, 30))` → `{size: {c_max_deg, area_distortion_max,
anisotropy_max}}`, closed-form only, for picking a patch size before building anything.

---

## Module layout

```
cosmic_projection/
  __init__.py            public API re-exports
  geometry.py            vec<->angle, rotations, great-circle dist, spherical polygon area, tangent frame   (numpy | cupy)
  overlap.py             gnomonic forward/inverse, Sutherland-Hodgman clip, spherical overlap area          (numpy | cupy)
  deformation.py         Deformation base + Identity / Callable / Aberration                                 (numpy | cupy)
  _backend.py            CPU/GPU array-module selection, host<->device, scatter-add
  _healpix.py            thin wrappers over healpy  (the only module that imports it)
  _healpix_gpu.py        closed-form RING primitives (vec2pix, query_disc, ...), numpy | cupy, no healpy
  _overlap_gpu.py        batched spherical-polygon overlap (GPU analogue of overlap.py)
  resample.py            Feature 1: resample_deformed_map  (CPU + GPU accumulators)
  plane.py               Feature 2: TangentPlane, flat_sky_diagnostics, flat_sky_table
  GPU_NOTES.md           GPU backend design, precision, memory, method-accuracy caveats
  FUTURE_PROJECTIONS.md  notes for orthographic / stereographic / reverse map / Doppler hook / rect planes
  requirements.txt
  tests/                 test_geometry.py  test_overlap.py  test_resample.py  test_plane.py  test_gpu.py
  examples/              demo_aberration_resample.py  demo_flatsky_cutout.py  demo_gpu_benchmark.py
```

---

## Tests

```bash
python -m pytest cosmic_projection/tests -q
```

- `test_geometry.py`, `test_overlap.py` — pure numpy, run anywhere.
- `test_resample.py`, `test_plane.py` — `importorskip("healpy")`, so they skip cleanly
  where healpy is unavailable (e.g. a Windows dev box) and run in WSL / Linux.
- `test_gpu.py` — `importorskip("cupy")` + a visible CUDA device; validates the
  closed-form RING primitives against healpy and the GPU resampler against the CPU one.

Invariants exercised: octant polygon area `= π/2`; `lonlat`↔`vec` round-trip; tangent
frame orthonormal & right-handed; Sutherland–Hodgman areas (identical / half / disjoint
/ nested); aberration Jacobian integrates to 2 (area preservation); identity resample
reproduces the input; downgrade matches `healpy.ud_grade`; **a constant map stays
constant under aberration for both methods** (the core area-weighting invariant);
subpixel vs polygon agree on a smooth map; flux is conserved for small β; gnomonic
centre pixel points at the tangent point; `roll_deg=180°` point-reflects the image;
`flat_sky_diagnostics` matches the closed-form scale factors at the corner.

## Examples

```bash
python -m cosmic_projection.examples.demo_aberration_resample   # aberrate + resample, save .npy, print flux conservation
python -m cosmic_projection.examples.demo_flatsky_cutout        # gnomonic cutouts + flat-sky report, save .npy
python -m cosmic_projection.examples.demo_gpu_benchmark         # CPU vs GPU polygon-resample timing + max|diff|
```
