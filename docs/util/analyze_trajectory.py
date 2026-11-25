"""Analyses register_trajectory.py recordings: distance to the endpoint, gradients, steps.

    python docs/util/analyze_trajectory.py white.vtk out.png label=run_dir [label=run_dir ...]

run_dir is the register_trajectory.py output directory (trajectory.npz, and the
reconstruction for the sanity check). Re-shoots the recorded momenta with register()'s
defaults (dense mode, control points = template vertices, 4 mm kernel, 11 time points, RK2),
then per iteration: per-vertex distance to the run's own final positions, per-vertex norm of
the gradient with respect to the momenta, and per-vertex displacement from the previous
iteration. Prints a table at objective levels and fixed iterations, and plots the three
measures (p50 and max) per run.
"""

import sys
from pathlib import Path

import common  # noqa: F401  (puts src/ on the path)
import matplotlib
import numpy as np
import torch
from common import read_vtk
from deformetrica.core import default
from deformetrica.core.model_tools.deformations.exponential import Exponential
from deformetrica.support import utilities
from deformetrica.support.kernels.keops_kernel import KeopsKernel

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
LEVELS = [2e6, 1.1e6, 1e6]
FIXED = [100, 200]

template, _ = read_vtk(sys.argv[1])
out = sys.argv[2]
runs = [a.split("=", 1) for a in sys.argv[3:]]

kernel = KeopsKernel(kernel_width=4.0)
device, _ = utilities.get_best_device(kernel.gpu_mode)
cp = utilities.move_data(template, device=device, dtype=default.tensor_scalar_type)


@torch.no_grad()
def shoot(momenta):
    e = Exponential(
        dense_mode=True,
        kernel=kernel,
        number_of_time_points=11,
        use_rk2_for_shoot=True,
        use_rk2_for_flow=True,
    )
    e.set_initial_control_points(cp)
    e.set_initial_template_points({"landmark_points": cp})
    e.set_initial_momenta(
        utilities.move_data(momenta, device=device, dtype=default.tensor_scalar_type)
    )
    e.update()
    return e.get_template_points()["landmark_points"].cpu().numpy().astype(np.float64)


def pct(a, axis=1):
    return np.percentile(a, 50, axis=axis), np.max(a, axis=axis)


results = {}
for label, run_dir in runs:
    z = np.load(Path(run_dir) / "trajectory.npz")
    X = np.stack([shoot(m) for m in z["momenta"]])  # (k, n, 3)
    recon = next(Path(run_dir).parent.glob("*-recon.vtk"), None)
    if recon is not None:
        d = np.linalg.norm(X[-1] - read_vtk(recon)[0], axis=1).max()
        print(f"{label}: re-shot final vs register() reconstruction, max {d:.2e} mm")
    results[label] = {
        "objective": z["objective"],
        "dist": np.linalg.norm(X - X[-1], axis=2),
        "grad": np.linalg.norm(z["gradient"], axis=2),
        "step": np.linalg.norm(np.diff(X, axis=0), axis=2),
    }

print(
    "\nrun          point            iteration  objective   distance to end p50/p95/max (mm)"
    "   |grad| p50/max   step p50/max (mm)"
)
for label, r in results.items():
    f = r["objective"]
    points = [
        (f"objective <= {lv:.1e}", int(np.argmax(f <= lv)))
        for lv in LEVELS
        if f.min() <= lv
    ]
    points += [(f"iteration {k}", k - 1) for k in FIXED if k <= len(f)]
    points.append(("end", len(f) - 1))
    for name, k in points:
        d, g = r["dist"][k], r["grad"][k]
        s = r["step"][k - 1] if k > 0 else np.zeros(1)
        print(
            f"{label:12s} {name:18s} {k + 1:5d}  {f[k]:.4e}   "
            f"{np.median(d):.3f} / {np.percentile(d, 95):.3f} / {d.max():.3f}"
            f"              {np.median(g):.2e}/{g.max():.2e}   "
            f"{np.median(s):.4f}/{s.max():.3f}"
        )

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "text.color": INK})
fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True, facecolor=SURFACE)
panels = [
    ("dist", "distance to own endpoint (mm)"),
    ("grad", "|gradient| per vertex (momenta)"),
    ("step", "displacement per iteration (mm)"),
]
for ax, (key, title) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    for k in FIXED:
        ax.axvline(k, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    for color, (label, r) in zip(SERIES, results.items()):
        med, mx = pct(r[key])
        it = np.arange(1, len(med) + 1) + (1 if key == "step" else 0)
        ax.plot(it, med, color=color, linewidth=2, label=f"{label} p50")
        ax.plot(
            it,
            mx,
            color=color,
            linewidth=1,
            linestyle=(0, (4, 2)),
            label=f"{label} max",
        )
        ax.annotate(
            label,
            (it[-1], med[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            va="center",
        )
    ax.set_yscale("log")
    ax.set_title(title, loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8)
axes[-1].set_xlabel("L-BFGS iteration", color=MUTED)
fig.tight_layout()
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"\nwrote {out}")
