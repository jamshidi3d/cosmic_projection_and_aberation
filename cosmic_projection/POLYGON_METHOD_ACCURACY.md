# Polygon remap vs. "proper" aberration — saved Q&A

Answers to two `/btw` questions asked while building the GPU backend (2026-08-27).
Kept here for future reference. Short version also lives in `GPU_NOTES.md`
("Accuracy of the polygon method itself").

---

## Q1. Is `method="polygon"` the most precise way to apply aberration?

**No.** Real-space pixel remapping is not the precision tool for CMB aberration. It is
intuitive and fine for visualization, low-ℓ effects, mesh work, and cross-checks, but it
is interpolation/pixelization-limited: aberration is a smooth ~1.23×10⁻³ rad deflection
whose observable effect is a coupling between neighbouring multipoles that pixel
binning/interpolation damps at high ℓ.

### What the literature uses instead

- **First-order harmonic-space coupling** (standard for analysis/forecasts): write
  aberration + Doppler modulation as an operator on `a_ℓm` coupling ℓ↔ℓ±1 with analytic
  coefficients. Challinor & van Leeuwen (2002); Kosowsky & Kahniashvili (2011); Amendola
  et al. (2011); Planck 2013 XXVII (Doppler-boosting detection). No pixelization error.

- **Non-perturbative boost kernels** (most precise in practice): first order is
  insufficient at Planck resolution (ℓ ≳ 1500–2000) because the coherent displacement
  reaches arcminute scale. Chluba (2011) gives a numerically exact recursion in β;
  Dai & Chluba (2014) extend to polarization and frequency-dependent (spectral) boost;
  **CosmoBoost** (Yasini & Pierpaoli 2017/2020) is the public reference implementation,
  computing the kernel to arbitrary order in a Doppler-weighted spherical-harmonic basis.

- **Real-space done right**: treat aberration as a fixed dipolar deflection field
  `d(n̂) = β sinθ ê_θ` (formally identical to a lensing remap) and evaluate the field at
  deflected positions with an exact SHT-at-arbitrary-points engine — `ducc0`
  synthesis/NUFFT, `lenspyx` — which preserve power to ℓ ~ 4000+. This is the accurate
  way to produce a boosted *map*; nobody uses pixel-polygon overlap for it.

**Bottom line:** `method="polygon"` is a legitimate independent-check / low-ℓ /
visualization tool, not a substitute for harmonic-space boost kernels (CosmoBoost,
Chluba 2011) or a `ducc`/`lenspyx` deflection remap when high-ℓ fidelity matters.

---

## Q2. For **high β**, compare `method="polygon"` against other approaches, and list ways to reduce the spurious numerical mode-coupling ("decoupling") it introduces.

At large β the first-order tool is gone, so the comparison changes.

| Approach | High-β behaviour |
|---|---|
| **First-order harmonic ℓ↔ℓ±1** (Challinor–van Leeuwen 2002; Kosowsky–Kahniashvili 2011) | **Invalid.** Truncation error ~O(β²) per order; tens of percent at β = 0.3–0.7. |
| **Non-perturbative harmonic boost kernel** (Chluba 2011 recursion; CosmoBoost, Yasini–Pierpaoli 2017/20; Dai–Chluba 2014 for pol/spectral) | **Reference-grade.** Exact in β for a band-limited field. Cost: kernel widens in ℓ; must raise `ℓ_max` in the boosted frame by ~the Doppler factor `D ≈ γ(1+β)` or you alias / lose power near the boost pole. |
| **Finite-angle deflection remap + exact SHT-at-points** (`ducc0` synthesis/NUFFT, `lenspyx`) | **Cleanest for a map, best ground truth.** Use the exact `n → n′` relation (not `d = β sinθ ê_θ`), evaluate rest-frame `a_ℓm` at deflected points, apply Doppler modulation `D^p` analytically. Exact to the rest-frame band limit, no pixelization. |
| **This module — area-weighted polygon remap** | **Beats first-order harmonic** (no β expansion; flux/area conserved by construction → monopole and low-ℓ power unbiased). **Loses** to the two above, and the gap grows with β: (1) toward the boost direction many source pixels compress into one target → sub-pixel structure averaged away, effective resolution collapses and no fixed rest-frame pixelization recovers it; (2) in the anti-boost hemisphere deformed polygons span many degrees, so the single local-gnomonic clip + `Ω_B/area2d(B)` rescale in `overlap.py` picks up real chart distortion, error ∝ (polygon angular size)²; (3) it only redistributes existing pixel values, so it cannot inject the high-ℓ power aberration should pump into the boost region. |
| Nearest / bilinear pixel remap | Strictly worse — non-conservative, biases even the monopole. |

**Net:** the polygon method is the right cheap independent check and is unbiased at low
ℓ, but for anything resolution- or high-ℓ-sensitive use CosmoBoost/Chluba or a
`ducc`/`lenspyx` deflection remap. Expect sub-percent `Cl` agreement at ℓ ≲ few×10, a
few-percent mid-ℓ deficit growing toward the boost pole, order-unity high-ℓ suppression
near it.

### Benchmark protocol

Take a `ducc0` deflection remap (exact angle + synthesis at deflected points + analytic
`D^p`) as truth; cross-check with CosmoBoost at high order, `ℓ_max × D`. Report per-ℓ
fractional `Cl` error and real-space residual maps **split by hemisphere** — the
transfer function is anisotropic here, so an isotropic average hides the dominant error.

### Ways to improve the decoupling (ranked by payoff)

1. **Higher-order conservative reconstruction (biggest single win).** The scheme is
   donor-cell (piecewise-constant source) → first-order, diffusive → shows up as
   high-ℓ suppression / broadband coupling. Give each source pixel a linear or parabolic
   profile from its neighbours (van Leer / PPM / WENO) and integrate *that* over each
   overlap polygon. Keeps exact conservation, removes most numerical diffusion.
2. **Sample the band-limited field, not the pixel value.** Evaluate the source at
   polygon sub-samples via `get_interp_val`, or better via SHT interpolation of the
   source `a_ℓm`. Turns "pixel remap" into a band-limited deflection remap; polygon
   areas then only handle target binning.
3. **Adaptive source resolution vs. local magnification.** The dominant loss is
   compression toward the boost pole. Pre-upsample the source (SHT to higher `nside`)
   where local magnification `1/D²` is large: choose `nside_in ≳ nside_out · max(D²)`.
4. **Exact spherical-polygon clipping.** Replace the local-gnomonic clip + area-ratio
   rescale with true great-circle edge clipping, then `spherical_polygon_area` directly.
   Kills the large-polygon distortion term that grows with β (noted as optional in
   `FUTURE_PROJECTIONS.md` — at high β it isn't optional).
5. **Raise `step` + triangulate the polygon interior, adaptively.** The deflection is
   nonlinear across a pixel at high β; more edge samples and interior sub-triangles cut
   the "great-circle between deformed corners" error. Scale `step` with local shear.
6. **Apply Doppler modulation `D(n)^p` analytically, separate from the geometry.** It's
   a smooth closed-form factor — never run it through a pixel operation, that only adds
   coupling. Keep `AberrationDeformation` geometry-only and multiply `D^p` in closed
   form afterward.
7. **Anti-alias the target deposit.** Hard polygon-in-pixel weighting is a top-hat
   window → sinc ringing in ℓ. Deposit each fragment with a smooth compact kernel
   (~1-pixel Gaussian or SPH kernel) and deconvolve the known window in ℓ. Standard
   particle-mesh trick, cheap.
8. **Supersample the target then downgrade.** Remap onto `nside_out × (2–4)`,
   `ud_grade` down with the proper window; softens hard target-bin-edge coupling at
   fixed final resolution.
9. **Measure and divide out the transfer function.** Push pure `Y_ℓm` inputs (or
   delta-`Cl` spectra) through the remap, form `T_ℓ = Cl_out/Cl_in` as a function of β
   **and direction**, use it as the coupling-quality metric, and deconvolve it;
   calibrate against an exact method on a few realizations.

---

## References

- Challinor & van Leeuwen 2002, PRD 65, 103001 — first-order aberration/modulation of the CMB.
- Kosowsky & Kahniashvili 2011, PRL 106, 191301 — dipolar `a_ℓm` coupling from the boost.
- Amendola et al. 2011, JCAP 07, 027.
- Planck 2013 results XXVII, A&A 571, A27 — Doppler-boosting detection.
- Chluba 2011, MNRAS 415, 3227 — non-perturbative aberration-kernel recursion in β.
- Dai & Chluba 2014, PRD 89, 123504 — polarization + spectral (frequency-dependent) boost.
- Yasini & Pierpaoli 2017 (PRD 96, 023021) / CosmoBoost (2020) — public boost-kernel code.
- `ducc0` (SHTs / NUFFT at arbitrary points), `lenspyx` (deflection remaps).
