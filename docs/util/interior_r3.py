"""
Gray-matter points through the R3 backward flow (docs/backward-flow-ablation.md, experiment 6).

Places the s-rep grid (equal-area latitudes with a 20% uniform floor, from this case) on the
last CMCF iterate, makes points at depths tau along each sample chord, and carries all of them
(tau = 0 and 1 are the s-rep's surface points) through backflow(..., space="R3"). Compares the
carried points at state 0 with the straight chords of the surface s-rep (srep.surface_srep):

  - surface points (tau = 0, 1): distance to the exact s-rep points;
  - interior points: distance to the same depth on the straight chord;
  - containment: fraction of interior points inside the white surface or outside the pial
    surface (generalised winding number against the original surfaces);
  - folding: prisms between consecutive depth layers over the grid triangles, split into
    three tetrahedra; fraction with the opposite orientation to the majority. The grid
    triangles are ~8 mm wide, so in folded cortex even a diffeomorphism can invert these
    straight-edged tetrahedra: this is not a fold detector at this resolution;
  - drift: per state, each interior point's relative position s along its carried chord
    (0 = white end, 1 = pial end) and its distance off that chord.

Usage: python docs/util/interior_r3.py states.npz out_dir [--evaluate-only]
"""
import shutil
import sys
from pathlib import Path

import igl
import numpy as np

import common  # noqa: F401  (puts src/ on the path)
from backflow import backflow
from srep import (
    depth_points,
    equal_area_latitudes,
    latitude_cdf,
    sampling,
    surface_srep,
)

DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)

states_path, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
local = out / states_path.name
if not local.exists():
    shutil.copy(states_path, local)
z = np.load(local)
S, F = z["states"], z["faces"].astype(np.int64)

latitudes = equal_area_latitudes([latitude_cdf(S[-2], F, S[0])], 32, uniform_fraction=0.2)
srep, exact = surface_srep(S, F, sampling(latitudes, n_phi=64))  # L, S in LSA
m = len(exact)
start = depth_points(srep.points, DEPTHS)  # at the last CMCF iterate, (len(DEPTHS), m, 3)

result = out / f"{local.stem}-back-R3.npz"
if "--evaluate-only" not in sys.argv:
    import logging

    logging.basicConfig(level=logging.INFO)
    backflow(local, space="R3", carry=start.reshape(-1, 3))
carried = np.load(result)["carried"]  # (K+1, len(DEPTHS) * m, 3), [0] is state 0
end = carried[0].reshape(len(DEPTHS), m, 3)
straight = depth_points(exact, DEPTHS)


def pct(x, q=(50, 95)):
    return " / ".join(f"{v:.2f}" for v in np.percentile(x, q)) + f" / max {x.max():.2f}"


print(f"{m} s-rep samples, depths {DEPTHS}")
print("carried vs exact (straight chord of the surface s-rep), mm, p50 / p95 / max:")
for i, tau in enumerate(DEPTHS):
    print(f"  tau {tau:.2f}: {pct(np.linalg.norm(end[i] - straight[i], axis=1))}")

white = np.ascontiguousarray(S[0][:, :3])
pial = np.ascontiguousarray(S[0][:, 3:])


def outside_gray(points):
    w_white = np.abs(igl.winding_number(white, F, np.ascontiguousarray(points)))
    w_pial = np.abs(igl.winding_number(pial, F, np.ascontiguousarray(points)))
    return np.mean(w_white > 0.5), np.mean(w_pial < 0.5)


print("interior points inside white / outside pial (state 0 surfaces):")
for i, tau in enumerate(DEPTHS[1:-1], 1):
    cw, cp = outside_gray(end[i])
    sw, sp = outside_gray(straight[i])
    print(f"  tau {tau:.2f}: carried {cw:.2%} / {cp:.2%}   straight {sw:.2%} / {sp:.2%}")


def inverted(layers):
    """Fraction of prism tetrahedra (between consecutive layers) against the majority."""
    G = srep.grid_faces
    vols = []
    for lo, hi in zip(layers[:-1], layers[1:]):
        a, b, c = (lo[G[:, k]] for k in range(3))
        A, B, C = (hi[G[:, k]] for k in range(3))
        for p, q, r, s in ((a, b, c, A), (b, c, A, B), (c, A, B, C)):
            vols.append(np.einsum("ij,ij->i", q - p, np.cross(r - p, s - p)))
    vols = np.concatenate(vols)
    return min(np.mean(vols < 0), np.mean(vols > 0))


print("inverted prism tetrahedra between depth layers:")
print(f"  start (last CMCF iterate): {inverted(start):.3%}")
print(f"  straight chords, state 0:  {inverted(straight):.3%}")
print(f"  carried, state 0:          {inverted(end):.3%}")


print("drift of interior points along their carried chord (s: 0 = white, 1 = pial):")
for k in sorted({len(carried) - 1, 30, 25, 20, 10, 0} & set(range(len(carried))), reverse=True):
    X = carried[k].reshape(len(DEPTHS), m, 3)
    u, v = X[0], X[-1]
    d = v - u
    L2 = np.sum(d * d, 1)
    row = []
    for i, tau in enumerate(DEPTHS[1:-1], 1):
        q = X[i] - u
        s_ = np.sum(q * d, 1) / L2
        off = np.linalg.norm(q - s_[:, None] * d, axis=1)
        row.append(f"tau {tau}: s {np.median(s_):.2f} [{np.percentile(s_, 5):.2f}, "
                   f"{np.percentile(s_, 95):.2f}] off {np.median(off):.2f} mm")
    print(f"  state {k:2d}, chord {np.median(np.sqrt(L2)):.2f} mm | " + " | ".join(row))
