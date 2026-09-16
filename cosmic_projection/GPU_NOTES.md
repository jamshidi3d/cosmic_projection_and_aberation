# GPU (CuPy) backend — design notes

`resample_deformed_map(..., backend="gpu"|"auto")` and
`TangentPlane.project_map(..., backend="gpu")` run the resampling arithmetic on an
NVIDIA GPU through CuPy. The CPU path is unchanged and remains the numerical reference.

## Architecture

| module | role |
|---|---|
| `_backend.py` | `resolve_backend("cpu"/"gpu"/"auto")`, `get_xp`, `array_module`, host↔device transfer, `scatter_add`. |
| `_healpix_gpu.py` | closed-form **RING** primitives that run on numpy *or* cupy: `vec2pix_ring`, `ang2pix_ring`, `pix2vec_ring`, `pix2corners_ring` (approx), `query_disc_ring` (vectorised over many discs), `get_interp_weights_ring`. No healpy. |
| `_overlap_gpu.py` | `batched_polygon_overlap` — gnomonic frame + projection + fixed-buffer Sutherland–Hodgman + masked shoelace, fully vectorised over `P` (source, target) pairs. |
| `geometry.py`, `deformation.py`, `overlap.py` | refactored to take their array module from the input (`array_module(...)`), so one code path serves numpy and cupy. |

`geometry.py` / `deformation.py` / `overlap.py` import no healpy and no cupy; cupy is
imported lazily on first GPU use.

### What still runs on the host (healpy), per `resample_deformed_map` GPU call

* Source-pixel boundaries: `hp.boundaries(nside_in, parents_chunk, step)` — per chunk,
  uploaded. (Correct geometry by construction; cheap and vectorised.)
* Target-pixel corners: `hp.boundaries(nside_out, arange(npix_out), step=1)` — **once**,
  uploaded, then gathered on-device by candidate index.
* NEST sub-pixel centres for `method="subpixel"`: `hx.subpixel_centers` — per chunk.
* I/O reordering (`nest=True`): `hp.reorder` on the final map.

Only `vec2pix_ring` (sample → pixel) and `query_disc_ring` (candidate search) are
reimplemented on-device, because those are the ones inside the hot loop. They are
validated against healpy in `tests/test_gpu.py`.

## The one novel primitive: `query_disc_ring`

Vectorised disc query over `D` discs at once, working entirely in RING `(z, phi)` space
(no face/`xyf` machinery):

1. Per disc, the intersected rings span a contiguous index range from
   `z = cos(theta0 ∓ r)`.
2. Flatten `(disc, ring)` pairs (`repeat` + within-group `arange`).
3. Per pair, the disc cuts the ring in `|phi − phi0| ≤ dphi` with
   `cos dphi = (cos r − z0·z_ring) / (sin theta0 · sin theta_ring)` (clamp → whole ring
   at the poles / large `r`).
4. That maps to a contiguous (wrapping) pixel-index window in the ring; expand
   `(disc, pixel)` pairs with a second `repeat`.

Two ragged expansions, both standard CSR-style GPU patterns. Handles arbitrary radius
and pole-centred discs — which is what makes **large β** tractable (deformed polygons
move tens of degrees; a fixed neighbour stencil would miss them).

`safety=` inflates the radius by that many ring spacings before selecting. Over-inclusion
is harmless: a non-overlapping candidate simply contributes `area = 0`.

## Precision

* `dtype="float64"` (default) or `"float32"`. This controls the **uploaded geometry
  tables** (source boundaries, target corners) — the memory-heavy part.
* The polygon-overlap math in `_overlap_gpu.batched_polygon_overlap` always upcasts to
  **float64** (small op count per pair, precision-sensitive shoelace of near-equal
  areas). So `dtype="float32"` trades table VRAM, not overlap accuracy.
* `num` / `den` accumulators are always float64.
* Expected GPU-vs-CPU agreement: `method="polygon"` ≲ 1e-6 abs on an O(1) map;
  `method="subpixel"` ≲ 1e-9; `project_map(method="nearest")` exact; `"bilinear"` /
  `"area"` ~1e-2 (independent bilinear stencil, see below).

## Determinism

`xp.bincount` scatter is deterministic on CuPy (it is a sorted reduction, not raw
atomics), so runs are bit-reproducible. If a future fused `RawKernel` uses `atomicAdd`,
add a `deterministic=` sorted-segment path.

## Memory

The uploaded target-corner table is `npix_out · 4 · 3 · itemsize`:

| nside_out | float64 | float32 |
|---|---|---|
| 256  | 226 MB | 113 MB |
| 512  | 905 MB | 452 MB |
| 1024 | 3.6 GB | 1.8 GB |

On an 8 GB card, `nside_out ≤ 1024` is comfortable with `dtype="float32"`; above that,
stream the table (or fall back to the approximate on-device `pix2corners_ring`, whose
cap-pixel error is sub-percent of a pixel). `chunk=` bounds the per-batch pair buffer.

Large-β pair counts are **unbalanced**: source pixels compress toward the boost
direction and expand opposite it, so a source pixel in the anti-boost hemisphere can
overlap many target pixels. Lower `chunk` if you hit an OOM during
`batched_polygon_overlap`.

## `project_map` on GPU

`backend="gpu"` must be requested explicitly (default stays `"cpu"`), because:

* `"nearest"` — exact match to the CPU result.
* `"bilinear"` / `"area"` — use `get_interp_weights_ring`, *a* bilinear scheme on the
  HEALPix grid, close to but **not** bit-identical with `healpy.get_interp_weights`;
  within one ring of a pole it degrades to linear-in-phi. Use `backend="cpu"` when you
  need healpy-identical interpolation.

## Not implemented / future

* **Fused `RawKernel`** — one thread per (source, target) pair, frame + projection +
  clip in registers, `atomicAdd` accumulation. Removes the per-op kernel launches of
  the vectorised CuPy version; worthwhile if profiling shows launch/bandwidth bound.
* **JAX rewrite** — `vmap` + `lax.fori_loop` + `segment_sum` would make the whole
  resampler differentiable w.r.t. `beta` / the boost direction (a differentiable
  forward model for a skewness–aberration fit).
* **Neighbour-stencil fast path** — for the small-β (β ≈ 10⁻³) regime, replace
  `query_disc_ring` with a precomputed source-pixel + N-ring neighbour table.
* **On-device source/target geometry** — a fully validated closed-form
  `pix2corners_ring` and `boundaries` to drop the host round-trips entirely.

## Accuracy of the polygon method itself (independent of CPU/GPU)

Real-space area-weighted pixel remapping is a legitimate low-ℓ / large-β / independent
-check tool, but it is not the precision instrument for CMB aberration:

* It is donor-cell (piecewise-constant source) → first-order, diffusive → shows up as
  high-ℓ suppression, worst toward the boost direction where sub-pixel structure is
  averaged away and no fixed rest-frame pixelisation recovers it.
* The single local-gnomonic clip + `Ω_B / area2d(B)` rescale in `overlap.py` picks up
  real chart distortion for the multi-degree polygons that occur in the anti-boost
  hemisphere at large β (error ∝ polygon angular size²).
* It only redistributes existing pixel values — it cannot inject the high-ℓ power
  aberration pumps into the boost region.

For high-ℓ fidelity use a non-perturbative harmonic boost kernel (Chluba 2011 recursion;
CosmoBoost, Yasini & Pierpaoli 2017/2020; Dai & Chluba 2014 for polarization /
spectral) or a finite-angle deflection remap with an exact SHT-at-points engine
(`ducc0` synthesis / NUFFT, `lenspyx`), applying the Doppler modulation `D(n)^p`
analytically. Keep `AberrationDeformation` geometry-only and multiply `D^p` separately.

Ways to reduce the spurious mode-coupling if staying in real space, by payoff:

1. Higher-order conservative reconstruction — give each source pixel a linear/parabolic
   profile (van Leer / PPM) from its neighbours and integrate *that* over each overlap
   polygon. Keeps exact conservation, removes most numerical diffusion.
2. Sample the band-limited field, not the pixel value — evaluate the source at polygon
   sub-samples via interpolation / SHT; polygon areas then only handle target binning.
3. Adaptive source resolution — pre-upsample the source where the local magnification
   `1/D²` is large: `nside_in ≳ nside_out · max(D²)`.
4. Exact spherical-polygon clipping — great-circle edge clipping + `spherical_polygon_
   area` directly, instead of the gnomonic-plane clip + area-ratio rescale. At high β
   this is no longer optional.
5. Adaptive `step` + interior triangulation — scale boundary sampling with local shear.
6. Anti-alias the target deposit / supersample-then-downgrade — soften the top-hat
   polygon-in-pixel window that causes sinc ringing in ℓ.
7. Measure and divide out the (anisotropic) transfer function `T_ℓ(β, direction)` by
   pushing pure `Y_ℓm` through the remap, and calibrate against an exact method.
