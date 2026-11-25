# Backward flow: baseline and ablation (work in progress)

This document records the decisions made, and the questions answered, while building a
baseline backward flow for the chordal s-rep (`devel/allemang-2026.pdf`, §7 "future work").
It also sets out the plan for the four-case ablation, which has not been run yet. The
attachment fixes it depends on are in
[extended-varifold-fixes.md](extended-varifold-fixes.md).

Code: `src/flow.py` (forward flow, saves its states), `src/backflow.py` (backward flow) and
`examples/main-backflow.py`. Helper scripts are in [`util`](util/); see
[Helper scripts](#helper-scripts).

## Setup

**Ground truth.** `flow.flow()` keeps the triangulation fixed, so chord i of every forward
state corresponds to chord i of the original chords c₀ = (u, v). A backward flow can
therefore be scored per chord against the truth. The registrations themselves never see
vertex indices; they only see the attachment, as in `register()`.

**Metrics** (`backflow.chord_errors`, per chord):

- `white_error` and `pial_error`: distance from each end to its true position (mm);
- `chord_length_error`: |v − u| − |v₀ − u₀| (mm);
- `chord_angle`: angle between v − u and v₀ − u₀ (degrees).

**Data** for every result below: `oasis4-inner-decimated.vtk` and
`oasis4-outer-decimated-recon.vtk`, copied from `data/` before the NaN registration of
[fix 7](extended-varifold-fixes.md#7-nan-gradient-at-degenerate-triangles-regression-from-6).
These chords therefore come from a registration made before attachment fixes 2, 3 and 7; the
exact code version was not recorded. 19,775 chords. The scratch outputs lived in
`/tmp/bfcheck/` and are not tracked.

**Forward flow, default schedule:** `rate_0 = 0.05`, `growth = 1.4`, `rate_max = 500`,
`stop_speed = 1e-4`, `epsilon = 1e-4`. This takes 59 iterations, giving 61 states: the input,
59 iterates, and the final chord-length projection Q.

**Backward flow defaults (`backflow()`):**

- dense LDDMM, with the control points at the template vertices;
- deformation and data kernels both 5 mm;
- feature width commensurate (1/5 mm⁻¹), `data_noise_std = 1`;
- 5 time points with RK2;
- L-BFGS-B, at most 100 iterations per step (`ftol = 1e-8`, memory 15).

## Decisions

1. **Features on the chord surface: carried H⁰.** In R^6 a surface has no normal, so a
   signed H isn't defined. Each chord carries `[H⁰_white, H⁰_pial]`, computed once from
   state 0, on both the template and the target. This follows the paper's intent that H⁰ is
   fixed at the first timestep. It makes the baseline **optimistic**: a backward flow from a
   fitted s-rep would not have identical labels on both sides.
   (Alternatives considered: recompute H separately on each half of the moving surface, or
   use no H at all; the second becomes case 3 of the [plan](#plan-four-case-ablation-not-run).)
2. **Tangent planes in R^6: unit 2-vectors.** These reduce exactly to normals in R^3; see
   [fix 6](extended-varifold-fixes.md#6-generalisation-to-rd).
3. **Driver.** Deformetrica's `Exponential` does the shooting, `MultiObjectAttachment` the
   attachment, and scipy L-BFGS-B optimises the deterministic-atlas objective
   `attachment / noise_std² + |p|²_K`. `DeterministicAtlas` itself is not used, because its
   file input and output only handle 2D and 3D.
4. **Stage through the forward-flow states** rather than registering the final state to c₀ in
   one go (experiment 1 below). `flow()` saves every iterate to
   `<white stem>-flow-states.npz`, with `faces` (m, 3), `states` (K+1, n, 6) and `rates` (K).
   This is also the intent of the evolutionary s-rep architecture.
5. **Chain the steps.** Step k → k−1 starts from the previous step's *result*, not from the
   true state k, so errors accumulate as they would in a real backward flow. Every result is
   scored against the true state it targets.
6. **All pairs, for now.** Every consecutive pair is fitted. Whether states can be skipped is
   open.
7. **R6 over R3 (experiment 4).** Fitting white→white and pial→pial with one deformation of
   R^3 loses the chord coupling.

## Experiment 1: single-shot coarse-to-fine (rejected)

The final forward state was registered directly onto c₀, through coarse-to-fine stages, each
starting from the previous result: (deformation, data) widths 40, 20, 10, then 5 mm. Each
stage ran 200 L-BFGS iterations, with feature width 0.5 mm⁻¹ (before the commensurate fix),
11 time points and noise std 1. The flowed chords start a median 40 mm (max 90 mm) from c₀ in
R^6, after aligning centroids.

| After stage | White error p50 / p95 (mm) | Pial error p50 / p95 (mm) | Chord length error p50 (mm) | Chord angle p50 / p95 (°) |
|---|---|---|---|---|
| start | 26.7 / 48.0 | 29.6 / 52.4 | 0.16 | 57.4 / 140.4 |
| 40 mm | 45.5 / 82.4 | 48.1 / 86.6 | 1.69 | 79.1 / 150.2 |
| 20 mm | 37.7 / 75.8 | 40.0 / 79.8 | 1.51 | 79.3 / 142.7 |
| 10 mm | 37.1 / 71.8 | 39.6 / 75.9 | 1.47 | 78.6 / 148.7 |
| 5 mm | 36.7 / 71.4 | 39.4 / 75.5 | 1.41 | 80.7 / 149.7 |

The attachment fell from 2e8 to 8e6, while regularity stayed between about 6e4 and 3e5. So
the shape is being matched, but points slide along it and correspondence gets *worse*. Every
stage hit its iteration cap, so the fit is also unconverged and poorly balanced. This
motivated decision 4.

## Experiment 2: chained staged backward flow in R6 (baseline)

60 steps, 61 → 0 states. 29 minutes on an RTX 4090; 49 of 60 steps hit the 100-iteration
cap; no NaNs.

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

At the final state: max white error 2.53 mm, max pial error 4.42 mm, 0.2% of chords more than
10° off. The median chord edge length in R^6 is about 4.3 mm.

- **Error does not compound.** Each step fits against the true shape of the next state, so
  the error stays bounded. It peaks around states 28–30 and recovers.
- **The Q step costs the most chord length** (p95 0.76 mm). The later steps recover most of
  it.
- **The max error grows in the last steps**, from 0.9 mm at state 15 to 2.5 mm at state 0.
  These outliers have not been located yet.

**Single step from the true state** ([`util`](util/backflow_step.py)), for
comparison:

| Step | White error p50 / p95 before → after (mm) | Chord angle p50 before → after (°) |
|---|---|---|
| 28 → 27 | 4.43 / 7.13 → 0.18 / 0.69 | 7.05 → 0.30 |
| 26 → 25 | 2.77 / 5.80 → 0.09 / 0.39 | 6.11 → 0.17 |

## Experiment 3: forward-flow rate schedule

**Why the error peaks around state 28.** The rate is 0.05 · 1.4^k, which reaches 500 at
k = ln(10⁴) / ln(1.4) ≈ 27.4. So it is clipped from iteration 28 on. Below the clip the
forward steps grow with the rate; above it they shrink as the surface smooths. `flow()` now
logs each iteration's rate, whether it was clipped, and the step sizes.

| Iteration | 20 | 22 | 24 | 26 | 27 | 28 | 30 | 32 | 34 | 36 |
|---|---|---|---|---|---|---|---|---|---|---|
| Forward step p95 (mm, R^6) | 3.0 | 4.9 | 7.4 | 9.6 | **10.4** | 9.4 (clipped) | 6.4 | 3.7 | 2.1 | 1.9 |

**Schedule sweep** ([`util`](util/flow_schedule_sweep.py)). The worst
step excludes the final Q step. The last column compares each schedule's final pre-Q state
with the default's.

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

**Backward flow with base 2.0, growth 1.2.** This schedule has the same number of steps as
the default, and its worst step is halved. The result was *worse* (32 minutes; 48 of 60
steps capped):

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

The base-2.0 run is more accurate through most of the flow, then loses it in the last two
backward steps. With a starting rate of 2.0, the first forward steps erase the finest folds,
and the 5 mm kernels cannot rebuild sub-millimetre detail in one step.

**Finding:** what a step removes matters as much as how far it moves. The early forward
steps remove fine detail, so they must stay small. Raising the base rate to stand in for
skipping states works against this.

Candidates not yet run:

- growth 1.2 with base 0.05: small early steps, halved peak, 79 states;
- kernel widths that follow the flow: narrow near c₀, wider later.

## Experiment 4: R6 vs R3 (white→white and pial→pial)

`backflow(..., space="R3")` uses one deformation of R^3 acting on both surfaces, with all 2n
vertices as control points, and two extended varifold attachments, white→white and
pial→pial. Both carry the same per-chord H⁰ pair. Everything else matches experiment 2.

**Surface crossings in the forward states**
([`util`](util/surface_crossings.py)):

| State | 0 | 10 | 20 | 30 … 60 |
|---|---|---|---|---|
| Pial vertices inside the white surface | 0.25% | 0.04% | 0 | 0 |
| White vertices outside the pial surface | 0.19% | 0.03% | 0 | 0 |

These crossings come from chord selection; the forward flow untangles them. Both backward
flows end with about the same fraction (R6 0.22%, R3 0.24%). A diffeomorphism keeps the
continuous surfaces disjoint, but the meshes are flat triangles between vertices and can
still cross. So this is not a floor on the R3 error, despite an earlier claim that it was.

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

At the shape level the two look alike in ParaView (R3's median error is 0.65 mm against
~4 mm edges). The difference is in correspondence. In R3 the two ends of a chord slide
independently along their own surfaces, and the error builds up through the second half of
the backward flow. R6 moves each chord as one point, which keeps its length and direction
together.

## Plan: four-case ablation (not run)

A progression from the upstream baseline to the final method. Rows 1 and 4 are the endpoints
that matter; rows 2 and 3 are intermediate controls.

| # | Space | Attachment | H⁰ | Status / change needed |
|---|---|---|---|---|
| 1 | R3 | `Varifold` (upstream, per face) | no | `backflow` needs a switch that builds `SurfaceMesh` objects and calls `varifold_distance` (~10 lines) |
| 2 | R3 | `ExtendedVarifold` | yes | runs today (`space="R3"`); experiment 4 |
| 3 | R6 | `ExtendedVarifold` | no | constant features (ψ ≡ 1), via a flag in `backflow` (a few lines) |
| 4 | R6 | `ExtendedVarifold` | yes | runs today (`space="R6"`); experiment 2 |

Hold fixed across all four: forward states (default schedule, unless experiment 3 changes
it), kernel widths, noise std, time points, iteration cap, metrics. Upstream `Varifold` is not
ported to R6; case 3 isolates the effect of H⁰ in R6 instead. Case 1 against case 2 differs
in both per-face vs per-vertex and H⁰.

Before running, re-derive the chords from a registration with fixes 2, 3 and 7 in place, so
the ablation does not inherit the older chords.

## Open questions

- **Skipping states.** The early steps (states 0–20, under ~1 mm each) contribute ~0.05 mm of
  error and most of the run time. Experiment 3 suggests merging them by raising the base rate
  is the wrong approach.
- **Outliers.** Which vertices reach the 2.5 mm max error in the last ten steps, and why? The
  per-vertex errors are in the output VTKs.
- **Convergence.** Most steps hit the 100-iteration cap. Does a higher cap, or another noise
  std, change the picture?
- **Kernel widths.** Fixed at 5 mm. Detail is removed fine to coarse, so the widths should
  probably follow the flow.
- **Optimism.** The targets are the exact forward states, and H⁰ is identical on both sides.
  A backward flow from a fitted s-rep will have neither advantage.

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

Example, from the repository root:

    python docs/util/flow_schedule_sweep.py data/oasis4-inner-decimated.vtk \
        data/oasis4-outer-decimated-recon.vtk /tmp/sweep
    python -c "import sys; sys.path.insert(0, 'src'); from pathlib import Path; \
        from backflow import backflow; backflow(Path('/tmp/sweep/default/oasis4-inner-decimated-flow-states.npz'))"
    python docs/util/compare_backflow.py /tmp/sweep/default/oasis4-inner-decimated-flow-states.npz \
        /tmp/sweep/default/oasis4-inner-decimated-flow-states-back-R6.npz

The full backward runs use `src/backflow.py` directly; `examples/main-backflow.py` runs R6 and
R3 on the states written by `examples/main-6d-flow.py`.
