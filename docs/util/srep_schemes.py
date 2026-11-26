"""
Shooting s-rep samples through the R6 backward flow (docs/backward-flow-ablation.md,
experiment 5, schemes A and B), and where it fails.

    python docs/util/srep_schemes.py states.npz out_dir A|B [--evaluate-only]

Uniform 32 x 64 grid. A starts each sample on the fitted ellipsoid, B on the chord surface
(the barycentric point); both are carried by backflow(..., space="R6", carry=...). Q with
the thickness of state 0 at each sample is applied at the end, as for those runs. Scored
against the barycentric point of state 0 (scheme D), with C (the barycentric point of the
backward-flowed mesh) for comparison. Then, for samples more than 2 mm off: polar angle,
CMCF compression of their faces (original mid-surface area per unit of fitted-sphere area,
median face = 1), distance to the most compressed 5% of faces, and distance to the original
white surface. Writes out_dir/compression-{white,pial}.vtk (log10 compression per vertex).

Note: the documented A run used an earlier carrying method (a separate non-dense shoot,
about 0.02 mm per step from the zero-momentum control points used here).
"""
import shutil
import sys
from pathlib import Path

import igl
import numpy as np
from scipy.spatial import cKDTree

import common  # noqa: F401  (puts src/ on the path)
from backflow import _summary, _write, backflow, chord_errors
from flow import chord_length_projection
from srep import fit_ellipsoid, fit_srep, interpolate, sampling, uniform_latitudes

states_path, out, scheme = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
local = out / f"{states_path.stem}-{scheme}.npz"
if not local.exists():
    shutil.copy(states_path, local)
z = np.load(local)
S, F = z["states"], z["faces"].astype(np.int64)

srep = fit_srep(S[-2], F, sampling(uniform_latitudes(32), n_phi=64))
start = srep.points if scheme == "B" else srep.mu + srep.directions @ srep.axes.T
truth = interpolate(S[0], F, srep.face, srep.bary)
thickness = np.linalg.norm(truth[:, 3:] - truth[:, :3], axis=1, keepdims=True)

result = out / f"{local.stem}-back-R6.npz"
if "--evaluate-only" not in sys.argv:
    import logging

    logging.basicConfig(level=logging.INFO)
    backflow(local, space="R6", carry=start)
back = np.load(result)
y = chord_length_projection(back["carried"][0], thickness)
c_scheme = interpolate(back["states"][0], F, srep.face, srep.bary)
print(f"scheme {scheme}, {len(y)} samples; start offset from the chord surface p50/p95/max "
      f"{np.percentile(np.linalg.norm(start - srep.points, axis=1), [50, 95, 100]).round(3)}")
print(f"  {scheme} vs D: {_summary(chord_errors(y, truth))}")
print(f"  C vs D: {_summary(chord_errors(c_scheme, truth))}")

err = np.linalg.norm(y[:, :3] - truth[:, :3], axis=1)
t1 = np.degrees(srep.theta[:, 0])
fail = err > 2
print(f"  > 2 mm: {fail.sum()} samples, polar angle {t1[fail].min():.0f}-{t1[fail].max():.0f} deg"
      if fail.any() else "  > 2 mm: none")
for lo in range(0, 180, 30):
    m = (t1 >= lo) & (t1 < lo + 30)
    print(f"    polar {lo:3d}-{lo + 30:3d}: white error p50 {np.median(err[m]):.2f} "
          f"p95 {np.percentile(err[m], 95):.2f}")

# CMCF compression per face, against the sphere of the ellipsoid fit
_, _, d = fit_ellipsoid(S[-2])
m0 = (S[0, :, :3] + S[0, :, 3:]) / 2


def area(x):
    return 0.5 * np.linalg.norm(np.cross(x[F[:, 1]] - x[F[:, 0]], x[F[:, 2]] - x[F[:, 0]]), axis=1)


comp = area(m0) / area(d)
comp /= np.median(comp)
hot = comp > np.percentile(comp, 95)
mid = (truth[:, :3] + truth[:, 3:]) / 2
to_hot = cKDTree(m0[F[hot]].mean(1)).query(mid)[0]
to_white = np.sqrt(igl.point_mesh_squared_distance(
    np.ascontiguousarray(y[:, :3]), np.ascontiguousarray(S[0, :, :3]), F)[0])
for name, g in (("failing", fail), ("others", ~fail)):
    if g.any():
        print(f"  {name}: compression p50 {np.median(comp[srep.face[g]]):.3f}; "
              f"on top-5% compressed faces {np.mean(hot[srep.face[g]]):.0%}; "
              f"distance to them p50 {np.median(to_hot[g]):.1f} mm; "
              f"to the original white surface p50 {np.median(to_white[g]):.2f} mm")
print(f"  most compressed 5% of faces: centroid {m0[F[hot]].mean((0, 1)).round(1)}; "
      f"failing samples: centroid {mid[fail].mean(0).round(1) if fail.any() else '-'}")

lc, w = np.log10(comp), area(m0)
num, den = np.zeros(len(m0)), np.zeros(len(m0))
for k in range(3):
    np.add.at(num, F[:, k], w * lc)
    np.add.at(den, F[:, k], w)
for name, cols in (("white", slice(0, 3)), ("pial", slice(3, 6))):
    _write(out / f"compression-{name}.vtk", S[0, :, cols], F, {"log10_compression": num / den})
