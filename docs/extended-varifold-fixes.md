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
| 7 | NaN gradient at degenerate triangles (regression from 6) | `2b9f323e` |

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
