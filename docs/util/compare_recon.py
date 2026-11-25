"""Compares register() reconstructions with a reference reconstruction.

    python docs/util/compare_recon.py pial.vtk reference-recon.vtk recon.vtk [recon.vtk ...]

All reconstructions deform the same template, so they share vertices. For each one:
per-vertex displacement from the reference (p50 / p95 / max, mm), and its fit to the pial
surface (unsigned distance from each reconstruction vertex to the pial mesh, p50 / p95, mm),
with the reference's fit for scale.
"""

import sys

import igl
import numpy as np
from common import read_vtk

P, FP = read_vtk(sys.argv[1])
R, _ = read_vtk(sys.argv[2])


def fit(V):
    d2, _, _ = igl.point_mesh_squared_distance(V, P, FP)
    return np.sqrt(d2)


def q(a, qs=(50, 95)):
    return " / ".join(f"{np.percentile(a, x):.3f}" for x in qs)


print(f"reference {sys.argv[2]}\n  fit to pial p50 / p95: {q(fit(R))}")
for path in sys.argv[3:]:
    V, _ = read_vtk(path)
    d = np.linalg.norm(V - R, axis=1)
    print(
        f"{path}\n  displacement from reference p50 / p95 / max: "
        f"{q(d)} / {d.max():.3f}   fit to pial p50 / p95: {q(fit(V))}"
    )
