"""Checks the dimension-free geometry in src/chordal/differentiable_geometry.py.

  python docs/util/check_geometry.py data/oasis4-inner-decimated.vtk

- R^3: the tangent 2-vector kernel (t.t')^2 equals the normal kernel (n.n')^2;
- R^3: Voronoi areas equal igl.massmatrix;
- R^6: lifting the surface to (x, 2x) scales areas by 5 and keeps unit 2-vectors;
- R^6: t.t' equals cos(theta_1) cos(theta_2) of the principal angles between two planes;
- R^6: torch.autograd.gradcheck of tangent planes and areas on a small patch.
"""

import sys

import igl
import numpy as np
import torch
from common import read_vtk

from chordal.differentiable_geometry import (
    vertex_normals,
    vertex_tangent_planes,
    voronoi_areas,
)

V, F = read_vtk(sys.argv[1])
x, F = torch.from_numpy(V), torch.from_numpy(F)

n, t = vertex_normals(x, F), vertex_tangent_planes(x, F)
d = ((n[:500] @ n[500:1000].T) ** 2 - (t[:500] @ t[500:1000].T) ** 2).abs().max()
print(f"R3 normal vs 2-vector kernel, max difference: {d.item():.1e}")
d = np.abs(voronoi_areas(x, F).numpy() - igl.massmatrix(V, F.numpy()).diagonal()).max()
print(f"R3 Voronoi areas vs igl.massmatrix, max difference: {d:.1e}")

x6 = torch.cat([x, 2 * x], 1)
scale = (voronoi_areas(x6, F) / voronoi_areas(x, F)).median().item()
norm = (vertex_tangent_planes(x6, F) ** 2).sum(1).mean().item()
print(f"R6 lift (x, 2x): area scale {scale:.6f} (expect 5), mean |t|^2 {norm:.6f}")

torch.manual_seed(0)
A, B = torch.randn(6, 2, dtype=torch.float64), torch.randn(6, 2, dtype=torch.float64)


def plane(M):
    p = torch.zeros(3, 6, dtype=torch.float64)
    p[1:] = M.T
    return vertex_tangent_planes(p, torch.tensor([[0, 1, 2]]))[0]


cosines = torch.linalg.svdvals(torch.linalg.qr(A)[0].T @ torch.linalg.qr(B)[0])
print(
    f"R6 |t.t'| {abs(plane(A) @ plane(B)).item():.12f}"
    f" vs cos(theta_1) cos(theta_2) {(cosines[0] * cosines[1]).item():.12f}"
)

Fs = F[:30]
idx = torch.unique(Fs)
remap = torch.full((len(x),), -1)
remap[idx] = torch.arange(len(idx))
Fs = remap[Fs]
xs = (x6[idx] + 0.1 * torch.randn(len(idx), 6, dtype=torch.float64)).requires_grad_()
ok = torch.autograd.gradcheck(
    lambda y: (vertex_tangent_planes(y, Fs).sum(), voronoi_areas(y, Fs).sum()), (xs,)
)
print(f"R6 gradcheck: {ok}")
