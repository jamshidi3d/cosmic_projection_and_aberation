# Future projection work

Notes for extensions deliberately left out of the first version. The current code
implements only the **gnomonic** tangent-plane projection and the two resampling
methods in `resample.py`.

## 1. Additional azimuthal projections for `TangentPlane`

All azimuthal projections share the same tangent frame (`geometry.tangent_frame`) and
differ only in the radial map between angular distance `c` from the centre and plane
radius `rho`. Refactor `plane.py` to take a small `projection` object:

```python
class Projection:
    def radial(self, c):        ...   # c (rad) -> rho
    def inverse_radial(self, rho): ... # rho -> c (rad)
```

`gnomonic_forward` / `gnomonic_inverse` in `overlap.py` become
`azimuthal_forward(vecs, frame, projection)` etc.: keep the `p = vecs - (vecs.n) n`
decomposition to get the azimuth, and swap the radial function.

| projection    | `rho(c)`            | `c(rho)`            | notes |
|---------------|---------------------|--------------------|-------|
| gnomonic (TAN)| `tan c`             | `arctan rho`       | great circles -> straight lines; valid `c < 90 deg`; area factor `1/cos^3 c` |
| stereographic (STG) | `2 tan(c/2)`  | `2 arctan(rho/2)`  | conformal (no shape distortion); whole sphere except antipode; area factor `(2/(1+cos c))^2` |
| orthographic (SIN)  | `sin c`       | `arcsin rho`       | view from infinity; one hemisphere only (`c <= 90 deg`); strong radial compression near the limb |

`flat_sky_diagnostics` should branch on the projection for the closed-form scale
factors; the empirically-sampled `empirical_area_distortion` / `max_distance_error`
metrics already work for any projection because they use `pixel_corners_vec`.

## 2. Reverse map: plane image -> HEALPix (`image_to_map`)

Mirror of `resample._accumulate_subpixel`:

1. Build the plane at `oversample` x resolution, get sub-pixel centre vectors.
2. `pix = healpy.vec2pix(nside, ...)` for each sub-pixel.
3. `num = np.bincount(pix, weights=image_oversampled.ravel(), minlength=npix)`,
   `den = np.bincount(pix, minlength=npix)`.
4. `out = np.where(den > 0, num / den, UNSEEN)`.

Sub-pixel weights should be the gnomonic cell solid angle
(`spherical_polygon_area` of `pixel_corners_vec`) if the plane is large enough for the
area distortion to matter.

## 3. Exact spherical-polygon clipping (instead of gnomonic-plane clipping)

`overlap.spherical_polygon_overlap_area` currently clips in a local gnomonic chart and
rescales by `Omega_B / area2d(B)`. For coarse maps / large pixels, replace with true
spherical clipping: represent each polygon edge as the great circle with pole
`n_i = normalize(v_i x v_{i+1})`, clip vertex-by-vertex against each half-space
`p . n_i >= 0`, computing edge/great-circle intersections as
`normalize(n_edge x n_clip)` with the sign chosen to lie on the subject edge. Then use
`geometry.spherical_polygon_area` on the result directly (no rescaling).

## 4. Doppler intensity modulation for `AberrationDeformation`

`AberrationDeformation` is purely geometric. The brightness change from a boost is a
separate multiplicative field. Add an opt-in `intensity` hook that
`resample_deformed_map` applies to `value_i` before accumulation, e.g. for a
thermodynamic-temperature CMB map to first order `T -> T (1 + beta cos theta)` with
`theta` the *pre*-aberration angle from the boost axis (or the full frequency-dependent
boost kernel). Keep it out of the `Deformation` object; pass it alongside.

## 5. Rectangular / anisotropic planes

`TangentPlane` is square (`size_deg`, `npix`). Generalise to `(size_x, size_y)` and
`(npix_x, npix_y)`; `_edges_1d` becomes per-axis. `matching_healpix` then rounds each
axis independently.
