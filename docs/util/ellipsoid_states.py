"""
Ellipsoid fit at each forward-flow state, and the mean speed that flow() stops on
(docs/backward-flow-ablation.md, experiment 5).

    python docs/util/ellipsoid_states.py states.npz [first_state]

Per state: the skeleton (chord midpoint) semi-axes of the fitted ellipsoid, midpoint and R^6
residuals, and the mean speed of the iteration that produced the state. The speed needs the
rates saved by flow(); older states files without them use the default schedule.
"""
import sys

import numpy as np

import common  # noqa: F401  (puts src/ on the path)
from srep import fit_ellipsoid

z = np.load(sys.argv[1])
S = z["states"][:-1]  # drop the Q state
first = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rates = z["rates"] if "rates" in z.files else np.minimum(0.05 * 1.4 ** np.arange(len(S)), 500)

print("state  mean speed  skeleton semi-axes (mm)   midpoint resid p50/p95   R6 resid p50/p95")
for k in range(first, len(S)):
    mu, axes, d = fit_ellipsoid(S[k])
    e = S[k] - mu - d @ axes.T
    em = np.linalg.norm((e[:, :3] + e[:, 3:]) / 2, axis=1)
    er = np.linalg.norm(e, axis=1)
    skeleton = np.linalg.norm((axes[:3] + axes[3:]) / 2, axis=0)
    speed = np.sqrt(np.mean((S[k] - S[k - 1]) ** 2)) / rates[k - 1]
    print(
        f"{k:5d}  {speed:9.2e}   {skeleton[0]:6.1f} {skeleton[1]:6.1f} {skeleton[2]:6.1f}"
        f"      {np.median(em):5.2f} / {np.percentile(em, 95):5.2f}"
        f"       {np.median(er):5.2f} / {np.percentile(er, 95):5.2f}"
    )
