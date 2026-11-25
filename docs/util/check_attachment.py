"""Evaluates the 3D extended varifold attachment between two meshes.

    python docs/util/check_attachment.py data/oasis4-inner-decimated.vtk \
        data/oasis4-outer-decimated.vtk [kernel_width=4.0]

Prints the H range of both meshes and the feature-kernel weight psi between a p5 and a p95
curvature (old convention: width = kernel_width; commensurate: width = 1 / kernel_width),
then the attachment value, its gradient norm, an autograd vs central-difference directional
derivative, and whether the source H0 stays fixed after moving the points.

To compare two code versions (e.g. HEAD against the working tree), check one out with
`git worktree add /tmp/head HEAD` and run this script from both trees.
"""

import sys

import numpy as np
import torch
from common import read_vtk
from deformetrica.core.model_tools.attachments.multi_object_attachment import (
    MultiObjectAttachment,
)
from deformetrica.core.observations import SurfaceMesh
from deformetrica.support.kernels.keops_kernel import KeopsKernel

from backflow import mean_curvature

width = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
(u, Fu), (v, Fv) = read_vtk(sys.argv[1]), read_vtk(sys.argv[2])

for name, (V, F) in (("source", (u, Fu)), ("target", (v, Fv))):
    p5, p50, p95 = np.percentile(mean_curvature(V, F), [5, 50, 95])
    gap = p95 - p5
    print(
        f"{name} H p5/p50/p95 {p5:.3f}/{p50:.3f}/{p95:.3f}  psi(p5, p95): "
        f"old {np.exp(-(gap**2) / width**2):.3f}  "
        f"commensurate {np.exp(-(gap**2) * width**2):.1e}"
    )

kernel = KeopsKernel(kernel_width=width)
source, target = SurfaceMesh(u, Fu), SurfaceMesh(v, Fv)


def distance(y):
    return MultiObjectAttachment.extended_varifold_distance(y, source, target, kernel)


x = torch.tensor(u, dtype=torch.float64, requires_grad=True)
d = distance(x)
d.backward()
h0 = torch.as_tensor(source.features).clone()
torch.manual_seed(0)
e = torch.randn_like(x)
e /= e.norm()
eps = 1e-3
with torch.no_grad():
    fd = (distance(x + eps * e) - distance(x - eps * e)).item() / (2 * eps)
print(f"distance {d.item():.6e}  gradient norm {x.grad.norm().item():.6e}")
print(
    f"directional derivative: autograd {(x.grad * e).sum().item():.9f}  central {fd:.9f}"
)
print(
    f"source H0 unchanged after moving: {torch.equal(h0, torch.as_tensor(source.features))}"
)
