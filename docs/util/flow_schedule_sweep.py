"""Sweeps forward-flow rate schedules and reports the per-iteration step sizes.

    python docs/util/flow_schedule_sweep.py white.vtk pial-recon.vtk out_dir

Each variant runs flow() on copies of the inputs in out_dir/<variant>/ (which also keeps
its -flow-states.npz for backflow). Reports: iterations, the iteration where the rate is
first clipped to rate_max, the worst per-iteration step (p95 and max over vertices, in R^6,
excluding the final chord-length projection), and the distance of the final pre-projection
state from the default schedule's.
"""

import logging
import shutil
import sys
from pathlib import Path

import common  # noqa: F401  (puts src/ on the path)
import numpy as np

from flow import flow

logging.basicConfig(level=logging.WARNING)
VARIANTS = {
    "default": {},
    "growth1.2": {"growth": 1.2},
    "growth1.1": {"growth": 1.1},
    "r0=0.5,g1.2": {"rate_0": 0.5, "growth": 1.2},
    "r0=2,g1.2": {"rate_0": 2.0, "growth": 1.2},
    "rmax100": {"rate_max": 100},
    "rmax50": {"rate_max": 50},
    "r0=0.5,g1.2,rmax100": {"rate_0": 0.5, "growth": 1.2, "rate_max": 100},
}

white, pial, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
reference = None
print(f"{'variant':22s} iters clip@  worst step p95/max  final vs default p50/p95")
for name, kwargs in VARIANTS.items():
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    u, v = shutil.copy(white, d), shutil.copy(pial, d)
    # stop_speed 1e-4: the run to near-convergence of experiment 3 (flow() now stops at 1e-3)
    *_, states = flow(Path(u), Path(v), max_iters=400, stop_speed=1e-4, **kwargs)
    z = np.load(states)
    S, rates = z["states"], z["rates"]
    steps = np.linalg.norm(np.diff(S[:-1], axis=0), axis=2)
    p95 = np.percentile(steps, 95, axis=1)
    clipped = rates == rates.max()
    clip = int(np.argmax(clipped)) if clipped.sum() > 1 else -1
    if reference is None:
        reference = S[-2]
    shift = np.linalg.norm(S[-2] - reference, axis=1)
    print(
        f"{name:22s} {len(rates):5d} {clip:5d}  {p95.max():6.2f} / {steps.max():6.2f}"
        f"       {np.median(shift):5.2f} / {np.percentile(shift, 95):5.2f}"
    )
