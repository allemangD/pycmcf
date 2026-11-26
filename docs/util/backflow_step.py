"""Fits one backward step k -> k-1 from the true state k (no accumulated error).

    python docs/util/backflow_step.py states.npz k [R6|R3]

Prints the fit summary and the chord errors before and after, against state k-1.
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import common  # noqa: F401  (puts src/ on the path)
import numpy as np

import backflow as bf

logging.basicConfig(level=logging.WARNING)
z = np.load(Path(sys.argv[1]))
F, S = z["faces"].astype(np.int64), z["states"]
k = int(sys.argv[2])
space = sys.argv[3] if len(sys.argv) > 3 else "R6"
n = S.shape[1]

h0 = np.stack([bf.mean_curvature(S[0, :, :3], F), bf.mean_curvature(S[0, :, 3:], F)], 1)
template = SimpleNamespace(connectivity=F, features=h0)


def target(p):
    return SimpleNamespace(points=p, connectivity=F, features=h0, norm=None)


if space == "R6":
    x0, objects = S[k], [(slice(None), template, target(S[k - 1]))]
else:
    x0 = np.concatenate([S[k, :, :3], S[k, :, 3:]])
    objects = [
        (slice(0, n), template, target(S[k - 1, :, :3])),
        (slice(n, 2 * n), template, target(S[k - 1, :, 3:])),
    ]
x, _, info = bf._fit_step(
    x0,
    objects,
    deformation_kernel_width=5.0,
    data_kernel_width=5.0,
    feature_kernel_width=None,
    data_noise_std=1.0,
    number_of_time_points=5,
    use_rk2=True,
    max_iterations=100,
    convergence_tolerance=1e-8,
    memory_length=15,
)
c = x if space == "R6" else np.hstack([x[:n], x[n:]])
print(f"{space} {k} -> {k - 1}: {info}")
print("  before", bf._summary(bf.chord_errors(S[k], S[k - 1])))
print("  after ", bf._summary(bf.chord_errors(c, S[k - 1])))
