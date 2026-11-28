# Backward flow and the annular s-rep: initial implementation

This document records the decisions made, and the questions answered, while building a baseline backward flow and the
annular s-rep (`devel/allemang-2026.pdf`, §3.1 and §7 "future work"). It also sets out the plan for the four-case
ablation, which has not been run yet. The attachment fixes it depends on are
in [extended-varifold-fixes.md](extended-varifold-fixes.md).

**Status.** The initial implementation is complete:

- the forward flow stops while the chord surface is an ellipsoid (experiment 5);
- the surface s-rep is a frozen (θ1, θ2) grid on the fitted ellipsoid, evaluated on the original chords with no backward
  flow (`srep.surface_srep`, experiment 5);
- the backward flow reproduces the forward states (R6, 0.06 / 0.30 mm at state 0). In R3 it can carry gray-matter points
  together with the s-rep's surface points, as one diffeomorphism, but with the 5 mm kernel those points drift towards
  the pial side and up to 27% leave the gray matter (experiment 6, negative).

**Main caveat: chord selection.** The chords these results use slide along the surfaces and are noisy
(see [Chord selection](#chord-selection-caveat)). The forward flow regularises them, but the backward flow restores the
noise, and the surface s-rep inherits it directly. Improving chord selection is the next step.

Code: `src/flow.py` (forward flow, saves its states), `src/backflow.py` (backward flow), `src/srep.py` (s-rep),
`examples/main-6d-flow.py` and `examples/main-backflow.py`. Helper scripts are in [`util`](util/);
see [Helper scripts](#helper-scripts).

## Setup

**Ground truth.** `flow.flow()` keeps the triangulation fixed, so chord i of every forward state corresponds to chord i
of the original chords c₀ = (u, v). A backward flow can therefore be scored per chord against the truth. The
registrations themselves never see vertex indices; they only see the attachment, as in `register()`.

**Metrics** (`backflow.chord_errors`, per chord):

- `white_error` and `pial_error`: distance from each end to its true position (mm);
- `chord_length_error`: |v − u| − |v₀ − u₀| (mm);
- `chord_angle`: angle between v − u and v₀ − u₀ (degrees).

**Data** for every result below: `oasis4-inner-decimated.vtk` and `oasis4-outer-decimated-recon.vtk`, copied from
`data/` before the NaN registration
of [fix 7](extended-varifold-fixes.md#7-nan-gradient-at-degenerate-triangles-regression-from-6). These chords therefore
come from a registration made before attachment fixes 2, 3 and 7; the exact code version was not recorded. 19,775
chords. The scratch outputs lived in `/tmp/bfcheck/` and are not tracked.

**Forward flow schedule:** `rate_0 = 0.05`, `growth = 1.4`, `rate_max = 500`, `epsilon = 1e-4`. Experiments 1–4 used
`stop_speed = 1e-4`: 59 iterations, giving 61 states (the input, 59 iterates, and the final chord-length projection Q).
Their backward flows started from the Q state, so their first step undid Q. Experiment 5 changed the default to
`stop_speed = 1e-3` (35 iterations, 37 states), and `backflow()` now starts from the last iterate and ignores the Q
state. Up to state 35 the two runs are identical.

**Backward flow defaults (`backflow()`):**

- dense LDDMM, with the control points at the template vertices;
- deformation and data kernels both 5 mm;
- feature width commensurate (1/5 mm⁻¹), `data_noise_std = 1`;
- 5 time points with RK2;
- L-BFGS-B, at most 100 iterations per step (`ftol = 1e-8`, memory 15).

## Decisions

1. **Features on the chord surface: carried H⁰.** In R^6 a surface has no normal, so a signed H isn't defined. Each
   chord carries `[H⁰_white, H⁰_pial]`, computed once from state 0, on both the template and the target. This follows
   the paper's intent that H⁰ is fixed at the first timestep. It makes the baseline **optimistic**: a backward flow from
   a fitted s-rep would not have identical labels on both sides. (Alternatives considered: recompute H separately on
   each half of the moving surface, or use no H at all; the second becomes case 3 of
   the [plan](#plan-four-case-ablation-not-run).)
2. **Tangent planes in R^6: unit 2-vectors.** These reduce exactly to normals in R^3;
   see [fix 6](extended-varifold-fixes.md#6-generalisation-to-rd).
3. **Driver.** Deformetrica's `Exponential` does the shooting, `MultiObjectAttachment` the attachment, and scipy
   L-BFGS-B optimises the deterministic-atlas objective `attachment / noise_std² + |p|²_K`. `DeterministicAtlas` itself
   is not used, because its file input and output only handle 2D and 3D.
4. **Stage through the forward-flow states** rather than registering the final state to c₀ in one go (experiment 1
   below). `flow()` saves every iterate to `<white stem>-flow-states.npz`, with `faces` (m, 3), `states` (K+2, n, 6)
   (the input, K iterates and Q) and `rates` (K). This is also the intent of the evolutionary s-rep architecture.
5. **Chain the steps.** Step k → k−1 starts from the previous step's *result*, not from the true state k, so errors
   accumulate as they would in a real backward flow. Every result is scored against the true state it targets.
6. **All pairs, for now.** Every consecutive pair is fitted. Whether states can be skipped is open.
7. **R6 over R3 for the surfaces (experiment 4).** Fitting white→white and pial→pial with one deformation of R^3 loses
   the chord coupling. R3 is kept for points inside the gray matter, which R6 cannot represent, though not yet usable
   for them (experiment 6).

## Experiment 1: single-shot coarse-to-fine (rejected)

The final forward state was registered directly onto c₀, through coarse-to-fine stages, each starting from the previous
result: (deformation, data) widths 40, 20, 10, then 5 mm. Each stage ran 200 L-BFGS iterations, with feature width 0.5
mm⁻¹ (before the commensurate fix), 11 time points and noise std 1. The flowed chords start a median 40 mm (max 90 mm)
from c₀ in R^6, after aligning centroids.

| After stage | White error p50 / p95 (mm) | Pial error p50 / p95 (mm) | Chord length error p50 (mm) | Chord angle p50 / p95 (°) |
|---|---|---|---|---|
| start | 26.7 / 48.0 | 29.6 / 52.4 | 0.16 | 57.4 / 140.4 |
| 40 mm | 45.5 / 82.4 | 48.1 / 86.6 | 1.69 | 79.1 / 150.2 |
| 20 mm | 37.7 / 75.8 | 40.0 / 79.8 | 1.51 | 79.3 / 142.7 |
| 10 mm | 37.1 / 71.8 | 39.6 / 75.9 | 1.47 | 78.6 / 148.7 |
| 5 mm | 36.7 / 71.4 | 39.4 / 75.5 | 1.41 | 80.7 / 149.7 |

The attachment fell from 2e8 to 8e6, while regularity stayed between about 6e4 and 3e5. So the shape is being matched,
but points slide along it and correspondence gets *worse*. Every stage hit its iteration cap, so the fit is also
unconverged and poorly balanced. This motivated decision 4.

## Experiment 2: chained staged backward flow in R6 (baseline)

60 steps, 61 → 0 states. 29 minutes on an RTX 4090; 49 of 60 steps hit the 100-iteration cap; no NaNs.

| State | White error p50 / p95 (mm) | Pial error p50 / p95 (mm) | Chord length error p95 (mm) | Chord angle p50 / p95 (°) |
|---|---|---|---|---|
| 59 (after undoing Q) | 0.15 / 0.41 | 0.15 / 0.40 | 0.76 | 0.68 / 2.15 |
| 50 | 0.09 / 0.20 | 0.09 / 0.21 | 0.37 | 0.44 / 1.72 |
| 40 | 0.12 / 0.32 | 0.13 / 0.33 | 0.34 | 0.59 / 1.77 |
| 30 | 0.26 / 1.12 | 0.27 / 1.21 | 0.31 | 0.61 / 3.04 |
| 28 | 0.26 / 1.16 | 0.27 / 1.24 | 0.28 | 0.75 / 6.75 |
| 25 | 0.12 / 0.48 | 0.13 / 0.53 | 0.20 | 0.68 / 4.78 |
| 20 | 0.05 / 0.14 | 0.05 / 0.15 | 0.12 | 0.37 / 2.64 |
| 10 | 0.07 / 0.21 | 0.07 / 0.21 | 0.12 | 0.51 / 2.09 |
| 5 | 0.08 / 0.30 | 0.07 / 0.26 | 0.14 | 0.45 / 2.35 |
| **0** | **0.06 / 0.31** | **0.05 / 0.25** | **0.14** | **0.35 / 2.26** |

At the final state: max white error 2.53 mm, max pial error 4.42 mm, 0.2% of chords more than 10° off. The median chord
edge length in R^6 is about 4.3 mm.

- **Error does not compound.** Each step fits against the true shape of the next state, so the error stays bounded. It
  peaks around states 28–30 and recovers.
- **The Q step costs the most chord length** (p95 0.76 mm). The later steps recover most of it.
- **The max error grows in the last steps**, from 0.9 mm at state 15 to 2.5 mm at state 0. These outliers have not been
  located yet.

**Single step from the true state** ([`util`](util/backflow_step.py)), for comparison:

| Step | White error p50 / p95 before → after (mm) | Chord angle p50 before → after (°) |
|---|---|---|
| 28 → 27 | 4.43 / 7.13 → 0.18 / 0.69 | 7.05 → 0.30 |
| 26 → 25 | 2.77 / 5.80 → 0.09 / 0.39 | 6.11 → 0.17 |

## Experiment 3: forward-flow rate schedule

**Why the error peaks around state 28.** The rate is 0.05 · 1.4^k, which reaches 500 at k = ln(10⁴) / ln(1.4) ≈ 27.4. So
it is clipped from iteration 28 on. Below the clip the forward steps grow with the rate; above it they shrink as the
surface smooths. `flow()` now logs each iteration's rate, whether it was clipped, and the step sizes.

| Iteration | 20 | 22 | 24 | 26 | 27 | 28 | 30 | 32 | 34 | 36 |
|---|---|---|---|---|---|---|---|---|---|---|
| Forward step p95 (mm, R^6) | 3.0 | 4.9 | 7.4 | 9.6 | **10.4** | 9.4 (clipped) | 6.4 | 3.7 | 2.1 | 1.9 |

**Schedule sweep** ([`util`](util/flow_schedule_sweep.py)). The worst step excludes the final Q step. The last column
compares each schedule's final pre-Q state with the default's.

| Schedule | Iterations | Clipped from | Worst step p95 / max (mm) | Final vs default p50 / p95 (mm) |
|---|---|---|---|---|
| default (0.05, ×1.4, max 500) | 59 | 28 | 10.40 / 11.46 | — |
| growth 1.2 | 79 | 51 | 5.86 / 6.35 | 0.24 / 0.45 |
| growth 1.1 | 118 | 97 | 3.07 / 3.34 | 0.43 / 0.81 |
| base 0.5, growth 1.2 | 66 | 38 | 5.86 / 6.36 | 0.23 / 0.45 |
| base 2.0, growth 1.2 | 59 | 31 | 5.92 / 6.40 | 0.24 / 0.45 |
| max rate 100 | 154 | 23 | 5.69 / 7.59 | 0.47 / 0.82 |
| max rate 50 | 275 | 21 | 3.37 / 6.12 | 0.57 / 0.97 |
| base 0.5, growth 1.2, max 100 | 158 | 30 | 4.46 / 5.34 | 0.49 / 0.84 |

**Backward flow with base 2.0, growth 1.2.** This schedule has the same number of steps as the default, and its worst
step is halved. The result was *worse* (32 minutes; 48 of 60 steps capped):

| Schedule | Final white error p50 / p95 / max (mm) | Final chord angle p50 / p95 (°) | Final chord length error p50 / p95 (mm) | Worst state, p95 (mm) |
|---|---|---|---|---|
| default | **0.06 / 0.31 / 2.53** | **0.35 / 2.26** | **0.02 / 0.14** | 1.26 (state 29) |
| base 2.0, growth 1.2 | 0.14 / 0.48 / 3.16 | 0.85 / 3.83 | 0.05 / 0.24 | 1.11 (state 29) |

| State | 0 | 1 | 2 | 3 | 4 | … | 11 |
|---|---|---|---|---|---|---|---|
| Forward step p95, default (mm) | 0.03 | 0.04 | 0.06 | 0.08 | 0.10 | | 0.62 |
| Error p95, default (mm) | 0.31 | 0.31 | 0.31 | 0.31 | 0.30 | | 0.19 |
| Forward step p95, base 2.0 (mm) | 0.79 | 0.80 | 0.86 | 0.93 | 1.00 | | 1.50 |
| Error p95, base 2.0 (mm) | 0.48 | 0.33 | 0.23 | 0.18 | 0.16 | | 0.12 |

The base-2.0 run is more accurate through most of the flow, then loses it in the last two backward steps. With a
starting rate of 2.0, the first forward steps erase the finest folds, and the 5 mm kernels cannot rebuild sub-millimetre
detail in one step.

**Finding:** what a step removes matters as much as how far it moves. The early forward steps remove fine detail, so
they must stay small. Raising the base rate to stand in for skipping states works against this.

Candidates not yet run:

- growth 1.2 with base 0.05: small early steps, halved peak, 79 states;
- kernel widths that follow the flow: narrow near c₀, wider later.

## Experiment 4: R6 vs R3 (white→white and pial→pial)

`backflow(..., space="R3")` uses one deformation of R^3 acting on both surfaces, with all 2n vertices as control points,
and two extended varifold attachments, white→white and pial→pial. Both carry the same per-chord H⁰ pair. Everything else
matches experiment 2.

**Surface crossings in the forward states**
([`util`](util/surface_crossings.py)):

| State | 0 | 10 | 20 | 30 … 60 |
|---|---|---|---|---|
| Pial vertices inside the white surface | 0.25% | 0.04% | 0 | 0 |
| White vertices outside the pial surface | 0.19% | 0.03% | 0 | 0 |

These crossings come from chord selection; the forward flow untangles them. Both backward flows end with about the same
fraction (R6 0.22%, R3 0.24%). A diffeomorphism keeps the continuous surfaces disjoint, but the meshes are flat
triangles between vertices and can still cross. So this is not a floor on the R3 error, despite an earlier claim that it
was.

**Single step 28 → 27 from the true state:**

| Space | White error p50 / p95 (mm) | Pial error p50 / p95 (mm) | Chord angle p50 / p95 (°) |
|---|---|---|---|
| R6 | 0.18 / 0.69 | 0.19 / 0.75 | 0.30 / 3.04 |
| R3 | **0.12 / 0.38** | 0.27 / 0.83 | 3.57 / 28.1 |

**Full chained run** (about 35 minutes):

| State | R6 white p50 / p95 (mm) | R3 white p50 / p95 (mm) | R6 chord angle p50 / p95 (°) | R3 chord angle p50 / p95 (°) |
|---|---|---|---|---|
| 59 | 0.15 / 0.41 | 0.37 / 0.91 | 0.68 / 2.15 | 4.99 / 18.30 |
| 40 | 0.12 / 0.32 | 0.37 / 0.79 | 0.59 / 1.77 | 4.56 / 19.25 |
| 28 | 0.26 / 1.16 | 0.50 / 1.15 | 0.75 / 6.75 | 9.41 / 45.46 |
| 20 | 0.05 / 0.14 | 0.61 / 2.23 | 0.37 / 2.64 | 8.14 / 77.73 |
| **0** | **0.06 / 0.31** | 0.65 / 2.02 | **0.35 / 2.26** | 7.73 / 44.49 |

| Final state | R6 | R3 |
|---|---|---|
| Max white / pial error (mm) | 2.53 / 4.42 | 11.43 / 10.74 |
| Chord length error p95 (mm) | 0.14 | 1.49 |
| Chords more than 10° off | 0.2% | 37.4% |

At the shape level the two look alike in ParaView (R3's median error is 0.65 mm against ~4 mm edges). The difference is
in correspondence. In R3 the two ends of a chord slide independently along their own surfaces, and the error builds up
through the second half of the backward flow. R6 moves each chord as one point, which keeps its length and direction
together.

## Experiment 5: annular s-rep on the ellipsoid

Code: `src/srep.py` (`surface_srep`); `examples/main-backflow.py`. Scripts: [
`ellipsoid_states.py`](util/ellipsoid_states.py) (fits per state), [`srep_schemes.py`](util/srep_schemes.py) (schemes A
and B, failure analysis), [`plot_latitude_area.py`](util/plot_latitude_area.py) (latitudes).

**Stop the forward flow while the surface is an ellipsoid.** An ellipsoid fit to each forward state (x = μ + A d, unit
d, A of size 6 × 3), on the run of experiment 2. Skeleton semi-axes are those of the chord midpoints, in mm; residuals
are midpoint residuals.

| State | Skeleton semi-axes (mm) | Midpoint residual p50 / p95 (mm) | R^6 residual p50 / p95 (mm) |
|---|---|---|---|
| 30 | 68.8 / 44.6 / 32.2 | 1.18 / 3.76 | 1.68 / 5.33 |
| 35 | 62.2 / 41.2 / 37.4 | 0.09 / 0.31 | 0.13 / 0.43 |
| 40 | 58.9 / 47.7 / 46.4 | 0.04 / 0.12 | 0.06 / 0.17 |
| 59 (last iterate at stop 1e-4) | 56.0 / 55.9 / 55.3, a sphere | 0.01 / 0.06 | 0.02 / 0.08 |

The surface is ellipsoidal around states 34–40, then rounds into a sphere, as the paper expects. The default
`stop_speed = 1e-4` ran 59 iterations, far past this. Mean speed is not monotone (7.8e-4 at state 36, 8.5e-4 at 40), and
`stop_speed = 1e-3` stops at state 35. With the same parameters the states up to 35 are identical bit for bit, so this
is only a change of value. The backward flow from state 35 gives the same mesh result as from state 60 (white error
0.06 / 0.30 mm at state 0) in 35 steps, 17 minutes.

The long axis is L/R (x, in the LSA frame of the data), the middle A/P (z) and the short S/I (y); each lines up with its
coordinate axis to within 0.06. The middle and short axes are only ~10% apart at state 35, so they cannot orient the
s-rep.

**Frame convention.** θ1 is the polar angle from the end of the long axis towards `pole` (default L); θ2 = 0 lies
towards `meridian` (default S). Only the long axis comes from the fit, so only two sign conventions are needed, not a
fine pre-alignment.

**Placing the samples.** Each grid direction is located in the sphere mesh (the triangulation at the fitted d of each
chord): a radial projection, in the ellipsoid's normalised coordinates, onto the chord surface. The face and barycentric
weights then give the sample in every state, since `flow()` keeps the triangulation. Four schemes were compared (R6,
uniform 32 × 64 grid, white error at state 0 against the barycentric point of state 0):

| Scheme | Start (state 35) | Back to state 0 | White error p50 / p95 (mm) | Chord angle p50 / p95 (°) |
|---|---|---|---|---|
| A | the ellipsoid point | shot through each step, Q at the end | 0.61 / 6.9 | 4.4 / 44 |
| B | the barycentric point of the chord surface | shot through each step, Q at the end | 0.26 / 1.25 | 2.7 / 17 |
| C | — | barycentric point of the backward-flowed mesh | 0.03 / 0.14 | — |
| **D** | — | **barycentric point of the original mesh (no backward flow)** | **0 (exact)** | **0** |

- **A.** Ellipsoid points sit off the surface by the fit residual (median 0.22 mm, up to 1.5 mm), and the backward flow,
  fitted only to the surface, amplifies the offset. The final error grows with it (correlation 0.76): p50 0.33 mm below
  0.1 mm offset, 9.2 mm for 1–2 mm.
- **B.** 69 of 1,986 samples (3.5%) still fail by more than 2 mm, up to 20 mm, and leave the surface (median 2.45 mm
  from the original white surface). All lie in the equatorial belt (θ1 between 56° and 124°), on the faces that CMCF
  *dilates* most (original area 1/35 of the median per unit of sphere area), around the midline at the centre of the
  brain. They are not on the compression hot spot (25 mm away) or its tube artefacts. Going back, those triangles shrink
  by up to about 6× linearly, and points inside them follow the smooth velocity field, not the triangle. Elsewhere the
  error p95 is 0.6–0.8 mm.
- **C and D** differ only by the backward flow's vertex error. With the triangulation kept, the forward mesh already
  carries the correspondence, so the backward flow adds nothing to surface samples.

Shot points are carried as extra control points with zero momenta, so they share the mesh's integrator exactly (a
separate non-dense shoot integrates differently: 0.02 mm per step).

**Decision: D for the surface s-rep** (`srep.surface_srep`). Its chords have their true length, so no Q is needed. The
backward flow is kept for what the mesh cannot give: points off the surface. In R6 a point is a whole chord, so depth
along it is linear by construction; gray-matter points need the R3 flow, carried together with the s-rep's surface
points so that both come from one diffeomorphism. Applications that need only surface points use D; those that need both
should use the shared R3 diffeomorphism, once it resolves the thickness (experiment 6: with the 5 mm kernel it does
not).

**Latitude spacing.** CMCF compresses each hemisphere's lateral surface into a cap of about 35° around its L/R pole, at
3.3–3.7 times the uniform density, and stretches the midline into an equatorial belt at 0.12–0.2 times, per unit sphere
area ([plot](srep-latitude-area.png): pdf per degree of θ1, raw and floored cdfs, ring latitudes;
`docs/util/plot_latitude_area.py`). Equal steps in θ1 therefore put most samples where there is least cortex (cortical
area per sample varies 13 times). Rings are placed at quantiles of the cortical area over θ1, averaged over the dataset
and frozen (`equal_area_latitudes`, `save_sampling`). Every case uses the same Cartesian (θ1, θ2) grid; cases added
later reuse the frozen sampling. For this subject, with 32 bands, the rings fall at 11.1, 15.3, 18.8, …, 54.2, 65.4, 90,
114.6, 125.8, … 168.9°.

Smooth forms for the latitudes, symmetric with fixed poles:

| Form | Max deviation from quantiles | Monotone | Cortical area per ring, min / max (equal = 1) |
|---|---|---|---|
| uniform | 25.7° | yes | 0.18 / 2.40 |
| odd cubic / quintic | 18.2° / 13.7° | no | — |
| odd 9th degree | 8.9° | yes | 0.14 / 1.82 |
| π s + 2 sine terms | 10.9° | yes | 0.06 / 1.96 |
| π s + 4 sine terms | 6.0° | yes | 0.36 / 1.58 |

The quantile function is nearly a step across the belt, so no low-order curve is simpler in practice than the table of
latitudes.

**A floor in the belt.** The most compressed 5% of faces lie in the belt, around (1, −22, −52), and straddle the
midline: the sulcus of the corpus callosum. Their extreme compression is a segmentation artefact: the imaging does not
separate the sulcal banks that run parallel to the corpus callosum, so the mesh has long thin tubes there. The sulcus
itself is real, so the belt should be sampled a little more densely than the quantiles give.
`equal_area_latitudes(..., uniform_fraction=α)` mixes the area CDF with a uniform one: a fraction α of the rings is
spaced evenly in θ1. For this subject, with 32 bands:

| α | Rings in 60–120° | Ring spacing at the equator | Cortical area per ring, min / max (equal = 1) |
|---|---|---|---|
| 0 (quantiles) | 3 | 24.6° | 1.00 / 1.00 |
| 0.2 (example default) | 5 | 15.7° | 0.55 / 1.14 |
| 0.3 | 5 | 13.0° | 0.43 / 1.22 |
| 0.5 | 7 | 9.5° | 0.31 / 1.42 |
| 1 (uniform) | 11 | 5.6° | 0.18 / 2.40 |

The floor also smooths the quantile function: with α = 0.2, π s + 2 sine terms fit the latitudes to within 4.3° and 4
terms to within 2.5° (10.9° and 6.0° without it).

**Symmetric about the equator.** The area CDF is symmetrised, so with an even number of bands the middle ring lies at
exactly 90° and separates the hemispheres: θ1 < 90° is L (the pole side), θ1 > 90° is R. The hemispheres do differ, but
little. Without symmetrisation, 49.96% of the cortex lies in the L half, the CDF is within 0.0035 of its symmetrised
form, and no ring moves by more than 0.5° (0.2° with α = 0.2), so the clean hemisphere split is worth more.

## Experiment 6: gray-matter points through the R3 backward flow (negative)

[`interior_r3.py`](util/interior_r3.py). The s-rep grid (32 bands, α = 0.2, 1,986 samples) on the last CMCF iterate,
with points at depths τ = 0, 0.25, 0.5, 0.75 and 1 along each sample chord, all carried by one R3 backward flow
(`backflow(..., space="R3", carry=...)`; 35 steps, all at the iteration cap, ~20 minutes, no NaNs; mesh white error at
state 0 0.55 / 1.43 mm). Compared at state 0 with the straight chords of the surface s-rep (D).

| Depth τ | Carried vs straight, p50 / p95 / max (mm) | Inside white, carried / straight | Outside pial, carried / straight |
|---|---|---|---|
| 0 (white) | 0.52 / 1.28 / 4.64 | — | — |
| 0.25 | 2.92 / 6.95 / 10.94 | 5.4% / 1.4% | 3.5% / 0.15% |
| 0.5 | 3.15 / 6.68 / 10.13 | 1.9% / 0.55% | 12.5% / 0.40% |
| 0.75 | 2.12 / 4.31 / 7.75 | 0.81% / 0.15% | 27.5% / 0.81% |
| 1 (pial) | 0.53 / 1.30 / 3.29 | — | — |

**The interior points do not hold their depth.** Along each carried chord (s = 0 at the white end, 1 at the pial end),
they drift towards the pial side while the cortex thickens back from 2.2 mm (state 35) to 4.8 mm, mostly between states
30 and 10:

| State | Chord p50 | τ = 0.25: s p50 [p5, p95] | τ = 0.5 | τ = 0.75 | Off the chord p50 (τ = 0.5) |
|---|---|---|---|---|---|
| 35 | 2.19 mm | 0.25 | 0.50 | 0.75 | 0 |
| 30 | 2.26 mm | 0.24 [0.20, 0.29] | 0.49 [0.43, 0.56] | 0.74 [0.69, 0.79] | 0.33 mm |
| 20 | 3.58 mm | 0.40 [0.16, 0.78] | 0.68 [0.37, 0.95] | 0.87 [0.64, 1.00] | 1.70 mm |
| 0 | 4.83 mm | 0.37 [−0.16, 1.20] | 0.63 [0.00, 1.34] | 0.84 [0.34, 1.27] | 2.33 mm |

The 5 mm deformation kernel is wider than the cortex is thick (2–5 mm), so the velocity field cannot make the white and
pial surfaces move differently across the gap without large, nearly cancelling momenta. Between the surfaces it is then
a poor interpolation of their motion, not a map of the columns. Points on the surfaces are pinned by the attachment;
points between them are not. The intuition that interior points would be better conditioned holds only if the kernel
resolves the thickness.

The prism test of the script (inverted tetrahedra between depth layers: 0% at state 35, 19.8% carried and 2.7% straight
at state 0) is not a valid fold detector here: the grid triangles are ~8 mm wide, and in folded cortex a diffeomorphism
can invert straight-edged tetrahedra of that size.

**Status.** The machinery stays (`carry`, `srep.depth_points`, the script), but carried gray-matter points are not
usable with these settings. Candidates: a deformation kernel narrower than the thickness (1–2 mm), at least for the
steps where the thickness changes (states 30 → 10); more time points per step; or a column construction that does not
rely on the flow (e.g. Laplace streamlines between the surfaces).

## Chord selection (caveat)

[`chord_quality.py`](util/chord_quality.py), on the chords of experiment 2 (from a registration before fixes 2, 3 and
7). Prisms are between the depth layers 0, 0.5 and 1 over each mesh face; "leaves gray matter" means a point at depth
0.25, 0.5 or 0.75 lies inside the white surface or outside the pial surface.

| Chords at state 0 | Length p50 | Angle to white normal p50 / p90 | Neighbour spread p50 / p90 | Leaves gray matter | Faces with an inverted prism |
|---|---|---|---|---|---|
| registration (ours) | 4.67 mm | 46° / 73° | 29° / 52° | 3.9% | 12.0% |
| nearest pial point (geometry only) | 2.90 mm | 27° / 76° | 68° / 140° | 5.6% | 24.1% |
| R6 backward flow | 4.67 mm | 46° / 73° | 29° / 52° | 3.3% | 11.6% |
| R3 backward flow | 4.81 mm | 45° / 74° | 29° / 51° | 3.0% | 13.4% |

- **Mostly chord selection.** The chords are 60% longer than the distance to the nearest pial point and lean 46° from
  the normal: the registration slides correspondence along the surfaces. The chords that leave the gray matter lean most
  (median 76°). The surfaces cross at only 0.25% of vertices, which accounts for 10% of them; for the rest both ends are
  on the right side and the straight segment between them is not.
- **Not entirely.** Nearest-point chords are worse still. A straight segment cannot follow a curved column, so in folds
  tighter than the thickness neighbouring chords fan out or cross whatever the correspondence, and the 90% decimation
  (edges of 2–5 mm, about the thickness) adds to it.
- **The forward flow regularises the chords; the backward flow restores the noise.** Against the midpoint-surface
  normal:

  | State | 0 | 10 | 20 | 30 | 35 |
  |---|---|---|---|---|---|
  | Angle p50 / p90 | 43° / 66° | 37° / 58° | 10° / 21° | 24° / 43° | 21° / 34° |
  | Neighbour spread p50 / p90 | 29° / 52° | 22° / 41° | 6° / 13° | 1.7° / 3.9° | 1.3° / 3.7° |

  Both backward flows end with chords as noisy as the originals, R3 included, although its attachments never see the
  pairing. The composed backward deformation must rebuild millimetre folds, so its derivative varies at that scale and
  turns the smooth state-35 chords back into noisy ones. The staged steps are too small for the attachment and
  regularity to choose another pairing, which is why both reproduce the forward correspondence (0.06 mm in R6, 0.65 mm
  in R3). So the backward flow does not regularise chord selection; that has to happen at the registration.

## Plan: four-case ablation (not run)

A progression from the upstream baseline to the final method. Rows 1 and 4 are the endpoints that matter; rows 2 and 3
are intermediate controls.

| # | Space | Attachment | H⁰ | Status / change needed |
|---|---|---|---|---|
| 1 | R3 | `Varifold` (upstream, per face) | no | `backflow` needs a switch that builds `SurfaceMesh` objects and calls `varifold_distance` (~10 lines) |
| 2 | R3 | `ExtendedVarifold` | yes | runs today (`space="R3"`); experiment 4 |
| 3 | R6 | `ExtendedVarifold` | no | constant features (ψ ≡ 1), via a flag in `backflow` (a few lines) |
| 4 | R6 | `ExtendedVarifold` | yes | runs today (`space="R6"`); experiment 2 |

Hold fixed across all four: forward states (default schedule, unless experiment 3 changes it), kernel widths, noise std,
time points, iteration cap, metrics. Upstream `Varifold` is not ported to R6; case 3 isolates the effect of H⁰ in R6
instead. Case 1 against case 2 differs in both per-face vs per-vertex and H⁰.

Before running, re-derive the chords from a registration with fixes 2, 3 and 7 in place, so the ablation does not
inherit the older chords.

## Open questions

- **Gray-matter points.** Experiment 6 lists candidates: a deformation kernel narrower than the thickness, more time
  points, or columns from outside the flow.
- **Chord selection.** Re-register with the current `register()` (fixes 2, 3 and 7), and perhaps a less decimated mesh
  or stronger regularity, then rerun [`chord_quality.py`](util/chord_quality.py).
  See [Chord selection](#chord-selection-caveat).
- **Segmentation tubes.** The callosal sulcus has thin tube artefacts (experiment 5). A numerical detector: inward
  thickness (shape diameter) under ~1 mm, convex across with radius about half that, near-zero curvature along, an
  elongated connected component, and a normal outward gap. The 90% decimated mesh smooths them below these thresholds (≤
  2% of hot-spot vertices under 1 mm), so calibrate on the full-resolution surfaces.
- **Skipping states.** The early steps (states 0–20, under ~1 mm each) contribute ~0.05 mm of error and most of the run
  time. Experiment 3 suggests merging them by raising the base rate is the wrong approach.
- **Outliers.** Which vertices reach the 2.5 mm max error in the last ten steps, and why? The per-vertex errors are in
  the output VTKs.
- **Convergence.** Most steps hit the 100-iteration cap. Does a higher cap, or another noise std, change the picture?
- **Kernel widths.** Fixed at 5 mm. Detail is removed fine to coarse, so the widths should probably follow the flow.
- **Optimism.** The targets are the exact forward states, and H⁰ is identical on both sides. A backward flow for a case
  without its own forward states (e.g. from a population s-rep) would have neither advantage.

## Helper scripts

All in [`util`](util/); run from the repository root (they put `src/` on the path themselves).

| Script | Purpose | Used for |
|---|---|---|
| `common.py` | VTK reader, path setup | — |
| `check_geometry.py` | 2-vectors vs normals, areas vs igl, R^6 lift, principal angles, gradcheck | fix 6 |
| `check_attachment.py` | H range and ψ, attachment value and gradient, directional derivative check, H⁰ fixed | fixes 2, 3, 6 |
| `register_repro.py` | `register()` into a scratch directory | fix 7 |
| `flow_schedule_sweep.py` | forward-flow rate schedules; writes each variant's states | experiment 3 |
| `surface_crossings.py` | white/pial crossings per forward state | experiment 4 |
| `backflow_step.py` | one backward step from the true state, R6 or R3 | experiments 2, 4 |
| `compare_backflow.py` | per-state errors of one or more backward runs | experiments 2–4 |
| `plot_latitude_area.py` | cortical area density over θ1 and ring latitudes, over one or more cases | experiment 5 |
| `ellipsoid_states.py` | ellipsoid fit and mean speed per forward state | experiment 5 |
| `srep_schemes.py` | s-rep samples shot through the R6 backward flow (schemes A, B), failure analysis, compression VTKs | experiment 5 |
| `interior_r3.py` | s-rep surface and gray-matter points carried by the R3 backward flow, vs straight chords | experiment 6 |
| `chord_quality.py` | chords as columns: obliquity, neighbour spread, leaving the gray matter, inverted prisms; forward-flow regularisation | chord selection |

Example, from the repository root:

    python docs/util/flow_schedule_sweep.py data/oasis4-inner-decimated.vtk \
        data/oasis4-outer-decimated-recon.vtk /tmp/sweep
    python -c "import sys; sys.path.insert(0, 'src'); from pathlib import Path; \
        from backflow import backflow; backflow(Path('/tmp/sweep/default/oasis4-inner-decimated-flow-states.npz'))"
    python docs/util/compare_backflow.py /tmp/sweep/default/oasis4-inner-decimated-flow-states.npz \
        /tmp/sweep/default/oasis4-inner-decimated-flow-states-back-R6.npz

The full backward runs use `src/backflow.py` directly. `examples/main-backflow.py` freezes the s-rep sampling, writes
each case's surface s-rep, and runs the R6 and R3 backward flows on the states written by `examples/main-6d-flow.py`.
