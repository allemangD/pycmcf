"""
Ellipsoid samples shot back through the backward flow (scheme A of
docs/backward-flow-ablation.md, experiment 5), against the barycentric s-rep (scheme D) and
the full-resolution surfaces.

    python docs/util/srep_shoot.py states.npz white.vtk pial.vtk GRID [GRID ...]
        [--space R6|R3] [--evaluate-only]

GRID is N_THETAxN_PHI, naming the <states dir>/srep-<GRID>-sampling.npz and srep-<GRID>.npz
written by docs/util/srep_grid.py. Each grid's samples start on the fitted ellipsoid at the
last CMCF iterate (exactly the ellipsoid, not the chord surface). All the grids are carried in
one backflow(..., space=SPACE, carry=...) run, which writes <states stem>-back-<SPACE>* next
to the states. In R6 each sample is carried as a chord; in R3 its white end and pial end are
carried as separate points by the same R3 diffeomorphism (the one that takes white to white
and pial to pial). No Q: the carried chords keep the length the flow gives them.

white.vtk and pial.vtk are the full-resolution surfaces (before decimation). Writes, per
grid, srep-<GRID>-shot-<SPACE>-{white,pial,skeleton}.vtk with theta1, theta2 and, on white
and pial,
to_full (distance to the full-resolution surface, mm), to_D (distance to the scheme D point,
mm) and normal_angle (degrees between the s-rep mesh's vertex normal and the normal of the
nearest full-resolution face, sign ignored: the faceting). The scheme D points get the same
to_full and normal_angle as srep-<GRID>-D-{white,pial}.vtk.
"""
import argparse
from pathlib import Path

import igl
import numpy as np

import common  # noqa: F401  (puts src/ on the path)
from backflow import _write, backflow
from common import read_vtk
from srep import fit_srep, load_sampling

parser = argparse.ArgumentParser()
parser.add_argument("states", type=Path)
parser.add_argument("white", type=Path)
parser.add_argument("pial", type=Path)
parser.add_argument("grids", nargs="+")
parser.add_argument("--space", choices=("R6", "R3"), default="R6")
parser.add_argument("--evaluate-only", action="store_true")
args = parser.parse_args()

z = np.load(args.states)
S, F = z["states"], z["faces"].astype(np.int64)
root = args.states.parent
fits, starts = {}, []
for g in args.grids:
    fit = fit_srep(S[-2], F, load_sampling(root / f"srep-{g}-sampling.npz"))
    fits[g] = fit
    starts.append(fit.mu + fit.directions @ fit.axes.T)
sizes = np.cumsum([0] + [len(s) for s in starts])

result = root / f"{args.states.stem}-back-{args.space}.npz"
carry = np.concatenate(starts)
if args.space == "R3":
    carry = np.concatenate([carry[:, :3], carry[:, 3:]])
if not args.evaluate_only:
    import logging

    logging.basicConfig(level=logging.INFO)
    backflow(args.states, space=args.space, carry=carry)
shot_all = np.load(result)["carried"][0]
if args.space == "R3":
    shot_all = np.hstack(np.split(shot_all, 2))

full = {name: read_vtk(p) for name, p in (("white", args.white), ("pial", args.pial))}


def to_full(name, points, faces):
    """Distance to the full-resolution surface (mm) and normal angle to it (degrees)."""
    V, Ff = full[name]
    points = np.ascontiguousarray(points)
    d2, nearest, _ = igl.point_mesh_squared_distance(points, V, Ff)
    n_full = igl.per_face_normals(V, Ff, np.zeros(3))[nearest]
    n_srep = igl.per_vertex_normals(points, faces)
    cos = np.abs(np.sum(n_full * n_srep, axis=1))
    return np.sqrt(d2), np.degrees(np.arccos(np.clip(cos, 0, 1)))


def pct(x):
    return " / ".join(f"{v:.2f}" for v in np.percentile(x, (50, 95, 99))) + f" / max {x.max():.2f}"


print(f"{args.space}: distance to the full-resolution surface, mm, p50 / p95 / p99 / max")
for i, g in enumerate(args.grids):
    fit = fits[g]
    shot = shot_all[sizes[i]:sizes[i + 1]]
    D = np.load(root / f"srep-{g}.npz")["chords"]
    print(f"{g} ({len(shot)} samples)")
    theta = {"theta1": fit.theta[:, 0], "theta2": fit.theta[:, 1]}
    for name, cols in (("white", slice(0, 3)), ("pial", slice(3, 6))):
        (dA, nA), (dD, nD) = (to_full(name, X[:, cols], fit.grid_faces) for X in (shot, D))
        off = np.linalg.norm(shot[:, cols] - D[:, cols], axis=1)
        print(f"  {name:5s} distance  A (shot) {pct(dA)}   D (barycentric) {pct(dD)}   "
              f"|A - D| {pct(off)}")
        print(f"  {name:5s} normal angle (deg)  A {pct(nA)}   D {pct(nD)}")
        _write(root / f"srep-{g}-shot-{args.space}-{name}.vtk", shot[:, cols], fit.grid_faces,
               {**theta, "to_full": dA, "to_D": off, "normal_angle": nA})
        _write(root / f"srep-{g}-D-{name}.vtk", D[:, cols], fit.grid_faces,
               {**theta, "to_full": dD, "normal_angle": nD})
    _write(root / f"srep-{g}-shot-{args.space}-skeleton.vtk", (shot[:, :3] + shot[:, 3:]) / 2,
           fit.grid_faces, theta)
