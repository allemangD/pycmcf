# Extended varifold: fixes

This document records the bugs found and fixed in the `ExtendedVarifold` attachment in our
Deformetrica fork. It also covers the changes that let the attachment run in R^6 for the
backward flow ([backward-flow-ablation.md](backward-flow-ablation.md)).

The attachment is eqs. 1–6 of `devel/allemang-2026.pdf`: the varifold inner product, with an
extra feature kernel ψ on the mean curvature field H⁰.

    ⟨S, T⟩ = Σ_x Σ_y  a_x a_y  ρ(x, y)  γ(x, y)  ψ(x, y)
    ρ = exp(−|x − y|² / σ²),   γ = ⟨n_x, n_y⟩²,   ψ = exp(−|H⁰_x − H⁰_y|² / w²)

Here a are the vertex (Voronoi) areas and n the unit vertex normals.

Code locations:

- `ext/deformetrica/src/deformetrica/core/model_tools/attachments/multi_object_attachment.py`
  (`extended_varifold_distance`)
- `ext/deformetrica/src/deformetrica/support/kernels/keops_kernel.py` (the KeOps reduction)
- `src/chordal/differentiable_geometry.py` (normals, areas, H and tangent 2-vectors, all
  differentiable)

Helper scripts are in [`aux/`](aux/). Each script states its usage in its docstring.

| # | Fix | Commit |
|---|---|---|
| 1 | Gradient did not flow through normals, areas and H | `e5270a14` |
| 2 | Feature kernel used the spatial width (ψ ≈ 1) | `4420a203` |
| 3 | Source H recomputed on the deformed points instead of fixed H⁰ | `e863ea43`, `39681a98` |
| 4 | Device assertions never ran | `4420a203` |
| 5 | `Varifold` restored to upstream semantics | `76ffbf7a` |
| 6 | Generalisation to R^d (areas, tangent planes, KeOps per dimension) | `4420a203`, `e863ea43`, `39681a98`; docstring `2b9f323e` |
| 7 | NaN gradient at degenerate triangles (regression from 6); additive epsilon rejected | `2b9f323e` |
| 8 | Stopping on a single L-BFGS step; windowed relative-improvement rule | `358c27b7` |

## 1. Gradient did not flow through normals, areas and H

Found and diagnosed in the earlier Claude session; see the Claude transcript PDF in `devel/`.

**Bug.** `extended_varifold_distance` computed the source normals, areas and H with igl on
`pa.detach()`. The value the optimiser saw depended on all three, but the gradient it got
only saw point positions. With the normal and area terms missing, the source self-term
⟨S, S⟩ becomes a plain repulsion between nearby source points with similar orientation. That
pushes sulcal fundi and the banks of narrow sulci into the white matter. `varifold_distance`
in the fork had the same issue; see fix 5.

**Evidence (from the transcript).** The transcript computed the attachment gradient at the
white surface, smoothed it with the 2.75 mm deformation kernel, and compared it with the
chords Deformetrica produced:

- The detached gradient predicted 88% of the vertices whose chords go inward, with a +0.89
  correlation to the observed outward motion.
- The full gradient predicted 32%, with a +0.41 correlation, and did not push fundi inward.
- The detached gradient matched the true gradient with only 0.71 cosine similarity, and its
  self-term part pointed the opposite way.

**Fix.** `differentiable_geometry.py` provides torch versions of the three igl calls. They
match igl to round-off and keep the autograd graph.

**Outcome.** These are `chord_diagnostics.py` results reported in the transcript, on the
`oasis4` decimated pair. The "after" run is `register.py` with the patched attachment, with
the template still unfrozen at that point.

| Measure | Before | After |
|---|---|---|
| Reconstruction vertices > 0.5 mm inside the white surface | 2.35% | 0.15% |
| Chords heading inward | 10.5% | 1.0% |
| Chords leaving the cortical ribbon | 15.0% | 5.4% |
| Reconstruction-to-pial mean distance | 1.89 mm | 0.92 mm |
| Median chord length (white-to-pial mean 2.84 mm) | 1.69 mm | 2.42 mm |
| Fundi: deep crossings | 5.7% | 0.02% |

## 2. Feature kernel used the spatial width

**Bug.** The KeOps formula used the spatial parameter `G = 1/σ²` for the H term too:
`Exp(-WeightedSqDist(G, Hx, Hy))`. That is eq. 6 as written, but H has units of 1/mm, so
|ΔH|²/σ² mixes units. At σ = 4 mm, ψ was close to 1 everywhere, and the attachment was
almost a plain varifold. The formula also declared and passed an unused parameter `E`
(`self.epsilon = 1e-3`).

| Mesh | H p5 / p50 / p95 (1/mm) | ψ(p5, p95), old: w = σ | ψ(p5, p95), new: w = 1/σ |
|---|---|---|---|
| `oasis4-inner-decimated` | −0.633 / 0.013 / 0.927 | 0.859 | 1.2e−17 |
| `oasis4-outer-decimated` | −0.966 / 0.088 / 0.709 | 0.839 | 3.2e−20 |

So a strong gyral crown and a strong sulcal fundus were 84–86% "the same feature". The table
comes from [`aux/check_attachment.py`](aux/check_attachment.py).

**Fix: commensurate units.** The same σ can address both terms if the curvature term is
scaled the other way: σ² |ΔH|² is dimensionless, like |x − y|²/σ². The feature width now
defaults to `w = 1/σ` (`KeopsKernel(feature_width=None)`), and `E = 1/w²` is the parameter
in the formula. An explicit `feature_width` overrides it.

**Open.** The commensurate term is sharp. At σ = 4 mm, ΔH = 0.25/mm already gives ψ ≈ 0.37.
H on decimated meshes is noisy (the transcript smoothed it over 3 rings for remeshing), so
this may be too strict. Not yet evaluated on a full registration.

## 3. Source H recomputed on the deformed points

**Bug.** The paper computes H once, at the first timestep (H⁰), and holds it fixed (§4.1).
The differentiable-geometry refactor (fix 1) instead recomputed the source H from the
*deformed* points at every evaluation, and differentiated through it. The target's H was
cached, so only the source side was affected. This changed what the method matches: the
current curvature rather than the original feature labels.

**Fix.** H⁰ is a per-object constant: either the object's `features` array (n, k), or H of
the object's own initial points, computed once and stored as `features`. No gradient flows
through it; normals and areas still carry gradients. The same mechanism carries
`[H⁰_white, H⁰_pial]` per chord in R^6.

**Check** ([`aux/check_attachment.py`](aux/check_attachment.py)):

- The directional derivative from autograd (−3.153771578) matches a central difference
  (−3.153771395).
- Source H⁰ is unchanged after moving the points.

## 4. Device assertions never ran

`assert (x.device == t.device for t in ...)` asserts a generator object, which is always
true. This was in both the `varifold` and `extended_varifold` branches of
`KeopsKernel.convolve`. Fixed with `all(...)`.

## 5. `Varifold` restored to upstream

The fork's `varifold_distance` had been rewritten to the per-vertex form with igl on
`pa.detach()`. That has the bug from fix 1 and differs from upstream's per-face
discretisation. It was restored byte-for-byte to upstream, so `"Varifold"` means what it
means in Deformetrica. `"ExtendedVarifold"` keeps the per-vertex form.

## 6. Generalisation to R^d

The backward-flow baseline runs the attachment on the chord surface c = (u, v) in R^6. Three
places relied on R^3:

| Quantity | R^3 | Any dimension |
|---|---|---|
| Triangle area, and the cotangent denominators in the Voronoi areas | \|a × b\| | sqrt(\|a\|²\|b\|² − (a·b)²) (Lagrange identity), same value |
| Tangent plane in γ | unit normal a × b | unit 2-vector a ∧ b: Plücker coordinates a_i b_j − a_j b_i, i < j (3 in R^3, 15 in R^6) |
| Mean curvature H = ⟨Lx / a, n⟩ | needs the normal | not computed; H⁰ is carried as `features` (fix 3) |

**Tangent 2-vectors.** In R^3 the 2-vector is the normal written in a different basis (Hodge
duality), so the kernel is unchanged. In general, for unit 2-vectors of two planes,
t_x · t_y = cos θ₁ cos θ₂, where θ₁ and θ₂ are the principal angles between the planes. In
R^3 two planes share a line, so θ₁ = 0, and this reduces to n_x · n_y. In R^6 both angles are
free: the kernel is 1 only for identical planes and falls as either angle grows.

A vertex's 2-vector is the area-weighted sum of its faces' 2-vectors, normalised, just like
vertex normals. In R^3 any such sum is again a plane. In R^4 and up it need not be, so at
strongly curved vertices it only approximates a tangent plane. The kernel is still a valid
positive-definite kernel. All of this is in the `vertex_tangent_planes` docstring.

**KeOps.** Reductions were compiled only for d ∈ {2, 3}. They are now compiled on first use
for whatever dimensions the tensors have (point, 2-vector and feature dimensions), and
cached. Net −21 lines.

**Checks.**

| Check | Result |
|---|---|
| R^3 kernel, 2-vectors vs normals (max difference over 250k pairs) | 1.0e−15 |
| R^3 Voronoi areas vs `igl.massmatrix` (max difference) | 1.9e−14 |
| R^6 lift (x, 2x): area scale / mean \|t\|² | 5.000000 (expect 5) / 1.000000 |
| R^6 \|t·t′\| vs cos θ₁ cos θ₂ (random planes) | 0.052656380534 / 0.052656380534 |
| R^6 `torch.autograd.gradcheck` (2-vectors and areas) | pass |
| 3D attachment value, before vs after the refactor (float32, `oasis4` pair, σ = 4 mm) | 6.367820e6 / 6.367821e6, gradient norm 2.328775e4 both |

The first five rows come from [`aux/check_geometry.py`](aux/check_geometry.py). The last row
was measured with the H recomputation and width of that time (before fixes 2 and 3), by
running [`aux/check_attachment.py`](aux/check_attachment.py) from a worktree of the previous
commit and from the new tree.

## 7. NaN gradient at degenerate triangles (regression from 6)

**Bug.** `torch.cross(...).norm()` has a zero (sub)gradient when a triangle collapses. Its
replacement `sqrt(|a|²|b|² − (a·b)²)` has an infinite one, because d sqrt / dx is infinite
at 0. If a line search briefly collapses a single triangle, the whole gradient becomes NaN,
and L-BFGS never recovers. A `register()` run on the new decimated `oasis4` pair wrote
all-NaN momenta and reconstruction (`data/outputs`, stopped at iteration 72).

**Fix.** Clamp inside the square root: `(…).clamp_min(1e-30).sqrt()` (`2b9f323e`).

**Reproduction** ([`aux/register_repro.py`](aux/register_repro.py), 150 iterations,
identical log-likelihoods up to the failure):

| Code | Result |
|---|---|
| `45133d1f` (no clamp) | NaN after ~40 iterations; every later line search is NaN |
| with the clamp | 150 iterations, no NaN; log-likelihood −1.005e6 (attachment −8.42e5, regularity −1.63e5) |

The forward-flow states contain truly degenerate triangles in R^6: state 25 has 2 vertices
with zero area. So the clamp also matters for the backward flow.

### Rejected alternative: additive epsilon

A regularised area, `sqrt(s + EPS)` with `EPS = 1e-12` mm⁴, looked more parsimonious. It is
smooth everywhere, and it would have made the separate cotangent clamps in `voronoi_areas` and
`cotan_laplacian_apply` redundant. It was tried and rejected.

**Static checks passed.** Areas matched igl to 8.2e−13 (1.9e−14 with the clamp), H matched to
5e−11, and the attachment value and gradient norm agreed with the clamp in every printed digit.

**Registrations did not.** Each `register()` iteration logs its line-search trials; the last
value in a block is the accepted point ([`aux/plot_register_loss.py`](aux/plot_register_loss.py)
counts them).

| Run | Iterations | Stopped by | Final log-likelihood | Line-search trials | Trials > 2× accepted | Largest trial / accepted |
|---|---|---|---|---|---|---|
| clamp, `ftol` 1e−8, cap 150 | 150 | cap | −1.005e6 | 3 | 0 | 1.3 |
| clamp, `ftol` 1e−8, cap 150 (repeat) | 150 | cap | −1.001e6 | 3 | 0 | 1.3 |
| clamp, `ftol` 1e−8 | 500 | cap | −9.28e5 | 8 | 0 | 1.3 |
| clamp, `ftol` 1e−6 | 497 | `ftol` | −9.23e5 | 3 | 0 | 1.3 |
| epsilon, `ftol` 1e−8 | 70 | `ftol` | −1.104e6 | 38 | 1 | 2.9e23 |
| epsilon, `ftol` 1e−8 (repeat) | 70 | `ftol` | −1.106e6 | 10 | 2 | 9.8e20 |
| epsilon, `ftol` 1e−6 | 132 | `ftol` | −1.024e6 | 42 | 3 | 5.1e10 |

Every clamp run accepts nearly every first step. Every epsilon run proposes steps that blow
the objective up by 10¹⁰–10²³, then stalls and stops. Reconstructions against the 500-iteration
clamp run ([`aux/compare_recon.py`](aux/compare_recon.py)) track the iteration count, not the
formula:

| Reconstruction | Iterations | Distance from the 500-iteration clamp run, p50 / p95 / max (mm) | Distance to pial, p50 / p95 (mm) |
|---|---|---|---|
| clamp, `ftol` 1e−8 (reference) | 500 | — | 0.368 / 1.173 |
| clamp, `ftol` 1e−6 | 497 | 0.235 / 1.02 / 6.9 | 0.367 / 1.158 |
| clamp, cap 150 | 150 | 0.738 / 2.13 / 6.4 | 0.384 / 1.211 |
| clamp, cap 150 (repeat) | 150 | 0.727 / 2.09 / 7.7 | 0.382 / 1.217 |
| epsilon, `ftol` 1e−6 | 132 | 0.843 / 2.41 / 7.3 | 0.386 / 1.225 |

Two clamp runs stopped at the same iteration differ by 0.16 mm at the median (150 vs 150).

**Gradient spikes.** Recording the gradient with respect to the momenta at every accepted
iteration ([`aux/register_trajectory.py`](aux/register_trajectory.py),
[`aux/analyze_trajectory.py`](aux/analyze_trajectory.py); both runs at `ftol` 1e−6):

| Run | Iteration | Largest per-vertex gradient (median) | Smallest doubled area at that vertex (mm²) | \|Σn\| / Σ\|n\| at that vertex | Smallest doubled area in the mesh (mm²) |
|---|---|---|---|---|---|
| clamp | 51 | 3.1e3 (26) | 3.6 | 0.70 | 1.3e−2 |
| clamp | 189 | 1.4e4 (16) | 2.0 | 0.88 | 6.8e−3 |
| clamp | 307 | 3.0e3 (14) | 0.36 | 0.02 | 4.5e−3 |
| epsilon | 92 | 9.4e4 (21) | 1.2 | 0.99 | 7.6e−3 |
| epsilon | 103 | 2.5e5 (20) | 1.3e−2 | 0.89 | 1.0e−2 |
| epsilon | 114 | 2.3e5 (19) | 2.5 | 0.92 | 1.3e−2 |

The epsilon spikes are 10–100× larger and recur until the fit stops. But neither guard is
ever active. The smallest doubled area anywhere is at least 4e−3 mm², thousands of times
above √EPS = 1e−6 mm², so the clamp never produces its exact zeros either. The spike
vertices are mostly not degenerate. (The gradient at a vertex's momentum also collects
attachment forces from about 4 mm around it, so the source may be a neighbour.)

**Conclusion.** The root cause is not established. The working intuition is that the tiny
bias epsilon applies to every area occasionally steers the flow toward an instability. Each
triangle is a separate chance for that to happen, so a rare per-triangle effect becomes
likely over a whole mesh and a whole fit. This is treated as a dead end: the clamp is kept,
and no epsilon code remains.

## 8. Stopping on a single L-BFGS step

**Problem.** `register()` stopped through scipy's `ftol`, a test on a *single* step: stop
when one iteration improves the objective by less than `ftol` (relative). Late in a fit,
single steps occasionally improve almost nothing (a line-search stall), so where a run stops
was partly luck: the runs in fix 7 stopped at 70, 132 and 497 iterations. The default
`ftol = 1e−8` is also below what the float32 objective can resolve. At about 4e6 a float32
value moves in steps of 0.25, while 1e−8 asks for improvements of 0.04.

**Precision is not the fix.** Measured on the 3D attachment for the `oasis4` pair
(σ = 4 mm), against a float64 reference:

| Variant | Error vs float64 | Time (s) |
|---|---|---|
| float32, as now (5 repeat evaluations bit-identical) | 0.33 (8e−8 relative) | attachment 0.016, shooting 0.17 |
| float32 inputs, float64 KeOps accumulators (`dtype_acc`) | 0.33 | — |
| float32 inputs, float64 final dot product | 0.16 | — |
| float64 throughout | 0 | attachment 0.34 (22×), shooting 5.5 (32×) |

The objective is deterministic; accumulators are not the noise. The limits are the float32
scalar and the float32 coordinates: at about 100 mm from the origin, positions move in steps
of about 7.6e−6 mm. For a 1e−5 mm step along the negative gradient, float64 sees an
improvement of 0.126 and the float32 path (with a float64 dot) sees 0.006. Float64 inputs
fix that, at 20–30× the cost on an RTX 4090. A stopping rule that does not depend on single
steps makes the float32 resolution irrelevant.

**The fit converges geometrically long before the objective stops falling.** One clamp run
(`ftol` 1e−8, 500 iterations), with measures against the pial surface:

| Iteration | Objective | Distance to pial, p50 / p95 (mm) | H⁰ mismatch, p50 / p95 | H sign agreement | Normal agreement (cos), p50 |
|---|---|---|---|---|---|
| 0 (white surface) | 4.42e6 | 2.883 / 4.969 | 0.511 / 1.613 | 0.593 | 0.818 |
| 150 | ~1.00e6 | 0.383 / 1.214 | 0.306 / 1.107 | 0.668 | 0.885 |
| 500 | 9.28e5 | 0.368 / 1.173 | 0.303 / 1.099 | 0.672 | 0.888 |

(H⁰ mismatch: each vertex's H⁰ against the H of the nearest pial vertex. Iteration-150 values
are the mean of two runs.)

| Iterations | Median relative improvement per iteration |
|---|---|
| 1–20 | 2.7e−2 |
| 20–50 | 5.5e−3 |
| 50–100 | 1.8e−3 |
| 100–150 | 9.7e−4 |
| 150–250 | 4.1e−4 |
| 250–500 | 1.1e−4 |

**But the vertices never stop sliding.** Distance from each iterate to the same run's
iteration-500 positions:

| At | Iteration | Distance to iteration-500 positions, p50 / p95 / max (mm) | Gradient per vertex, p50 / max | Step per iteration, p50 / max (mm) |
|---|---|---|---|---|
| objective 2e6 | 6 | 2.79 / 6.63 / 18.8 | 88 / 943 | 0.70 / 4.8 |
| objective 1.1e6 | 66 | 1.20 / 3.23 / 8.9 | 19 / 1070 | 0.029 / 0.41 |
| iteration 100 | 100 | 0.95 / 2.65 / 8.8 | 21 / 399 | 0.035 / 0.77 |
| objective 1e6 | 159 | 0.71 / 2.00 / 5.4 | 17 / 459 | 0.018 / 0.24 |
| iteration 200 | 200 | 0.60 / 1.67 / 5.0 | 16 / 413 | 0.017 / 0.40 |
| end | 500 | 0 | 14 / 735 | 0.0025 / 0.049 |

The median distance falls steadily all the way to 500, with no plateau. The median gradient
drops about 10× in the first 50 iterations and then stays at 15–20 while the steps shrink, the
signature of an ill-conditioned problem. The extra sliding barely changes the fit, H⁰ or the
normals, so the objective does not pin correspondence along the surface. Since correspondence
*is* the chord, stopping early (e.g. at objective 1.1e6, 1.2 mm median from the later
positions) is not acceptable.

**Rule.** `ScipyOptimize` (fork) takes `improvement_window` and `improvement_tolerance`. After
each accepted iteration, it stops once the objective's relative improvement over the last
`window` iterations, averaged per iteration, falls below `tolerance`. The rule is off unless
both are set. `register()` sets `improvement_window=5, improvement_tolerance=1e−4` and keeps
`ftol` (1e−8) and `max_iterations` (500) as backstops. The method is fixed; the window and
tolerance are values to tune for a problem, not for one subject.

Simulated on the recorded clamp trajectory:

| Window | Tolerance per iteration | Would stop at | Distance to iteration-500 positions, p50 / p95 (mm) | Distance to pial, p50 / p95 (mm) |
|---|---|---|---|---|
| 3 or 5 | 1e−3 | 109 | 0.91 / 2.56 | 0.389 / 1.215 |
| 3 or 5 | 5e−4 | 155 | 0.72 / 2.03 | 0.381 / 1.209 |
| 3 or 5 | 2e−4 | 191–193 | 0.61 / 1.73 | 0.383 / 1.204 |
| 3 | 1e−4 | 192 | 0.61 / 1.74 | 0.383 / 1.204 |
| **5** | **1e−4** | **438** | **0.14 / 0.31** | 0.371 / 1.167 |

The fit stalls for a few steps around iteration 190, at the one clamp gradient spike in the
table above. A 3-iteration window mistakes the stall for convergence; 5 rides through it. At
iteration 438 the remaining motion (0.14 mm median) is below the 0.16 mm by which two
identical runs differ, so stopping there loses nothing that is reproducible anyway. On this
subject it saves little time over the 500 cap, but the stop no longer depends on one step.

**Verified** ([`aux/register_repro.py`](aux/register_repro.py) with the new defaults, a fresh
run): it stopped by the window rule at iteration 392 (relative improvement 9.9e−5 per
iteration over the last 5), at log-likelihood −9.355e5, with no NaN. That is earlier than the
simulated 438 because each run follows its own trajectory. Its reconstruction is 0.257 /
1.02 / 5.1 mm (p50 / p95 / max) from the 500-iteration clamp run, about the same as the
497-iteration clamp run (0.235 / 1.02 / 6.9 mm), and it fits the pial surface as well
(0.371 / 1.171 mm against 0.368 / 1.173 mm).

**Open.** The objective does not pin correspondence along the surface: vertices still slide
about 0.2 mm per 100 iterations near the end. For chord selection this matters more than
precision.

