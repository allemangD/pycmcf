"""Runs register() and records the optimiser's trajectory.

    python docs/util/register_trajectory.py white.vtk pial.vtk out_dir \
        [max_iterations=500] [convergence_tolerance=1e-6]

Wraps ScipyOptimize from outside (no library change): at every accepted L-BFGS iteration it
records the objective (-log-likelihood), the momenta, and the gradient with respect to the
momenta. Writes out_dir/trajectory.npz with
    iteration (k,), objective (k,), momenta (k, n, 3) float32, gradient (k, n, 3) float32,
plus the usual register() outputs. Analyse with analyze_trajectory.py.
"""

import logging
import sys
from pathlib import Path

import common  # noqa: F401  (puts src/ on the path)
import numpy as np
from deformetrica.core.estimators.scipy_optimize import ScipyOptimize

from register import register

logging.basicConfig(level=logging.INFO)
white, pial, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 500
tolerance = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-6
out.mkdir(parents=True, exist_ok=True)

record = {"iteration": [], "objective": [], "momenta": [], "gradient": []}
cost_and_derivative, callback = (
    ScipyOptimize._cost_and_derivative,
    ScipyOptimize._callback,
)


def recording_cost_and_derivative(self, x):
    cost, gradient = cost_and_derivative(self, x)
    self._last_evaluation = (x.copy(), cost, gradient)
    return cost, gradient


def recording_callback(self, x):
    # L-BFGS-B calls back with the accepted point, normally its last evaluation
    x_last, cost, gradient = self._last_evaluation
    if not np.array_equal(x, x_last):
        cost, gradient = cost_and_derivative(self, x)
    record["iteration"].append(self.current_iteration + 1)
    record["objective"].append(cost)
    record["momenta"].append(x.reshape(-1, 3).astype(np.float32))
    record["gradient"].append(gradient.reshape(-1, 3).astype(np.float32))
    return callback(self, x)


ScipyOptimize._cost_and_derivative = recording_cost_and_derivative
ScipyOptimize._callback = recording_callback
try:
    register(
        white,
        pial,
        output_dir=out,
        max_iterations=iterations,
        convergence_tolerance=tolerance,
    )
finally:
    np.savez(out / "trajectory.npz", **{k: np.array(v) for k, v in record.items()})
    print(f"recorded {len(record['iteration'])} iterations to {out / 'trajectory.npz'}")
