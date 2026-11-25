"""Plots register() objective curves from Deformetrica logs (as written by register_repro.py).

    python docs/util/plot_register_loss.py out.png label=run.log [label=run.log ...]

Top: objective (-log-likelihood) per iteration, log scale, one line per run (at most 4).
Bottom: attachment and regularity terms of the first run. Each iteration block logs its
line-search trials; the last value is the accepted point, which is what is plotted. Also
prints, per run, the median relative reduction per iteration in windows, and the line-search
trials: how many, how many exceed 2x the accepted value, and the largest ratio.
"""

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference categorical palette (dataviz skill), fixed order, validated on the light surface
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
LINE = re.compile(
    r"Log-likelihood = (\S+)\s+\[ attachment = (\S+) ; regularity = (\S+) \]"
)


def read(path):
    rows = []
    with open(path) as f:
        text = f.read()
    trials = []
    for block in text.split("Iteration:")[1:]:
        found = LINE.findall(block)
        if found:
            values = [[-float(v) for v in row] for row in found]
            rows.append(values[-1])
            trials += [v[0] / values[-1][0] for v in values[:-1]]
    # objective, attachment, regularity (all positive); trial / accepted ratios
    return np.array(rows), np.array(trials)


out, runs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
data, trials = {}, {}
for label, path in runs:
    data[label], trials[label] = read(path)

for label, d in data.items():
    rel = -np.diff(d[:, 0]) / d[1:, 0]
    windows = [(1, 20), (20, 50), (50, 100), (100, 150), (150, 250), (250, 500)]
    parts = [
        f"{a}-{b}: {np.median(rel[a - 1 : b - 1]):.1e}"
        for a, b in windows
        if len(rel) >= b - 1
    ]
    print(
        f"{label} ({len(d)} iterations) median relative reduction: " + "  ".join(parts)
    )
    t = trials[label]
    print(
        f"  line-search trials {len(t)}, > 2x accepted {np.sum(t > 2)}, "
        f"largest ratio {t.max() if len(t) else 1:.1e}"
    )

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "text.color": INK})
fig, (top, bottom) = plt.subplots(
    2, 1, figsize=(8, 7), height_ratios=[3, 2], sharex=True, facecolor=SURFACE
)
for ax in (top, bottom):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.axvline(150, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))

ends = []  # (iteration, value) of labels already placed, to stagger collisions
for color, (label, d) in zip(SERIES, data.items()):
    it = np.arange(1, len(d) + 1)
    top.plot(it, d[:, 0], color=color, linewidth=2, label=label)
    near = sum(abs(i - it[-1]) < 40 and abs(v / d[-1, 0] - 1) < 0.05 for i, v in ends)
    ends.append((it[-1], d[-1, 0]))
    top.annotate(
        f"{label}, {len(d)} its",
        (it[-1], d[-1, 0]),
        xytext=(4, 11 * near),
        textcoords="offset points",
        fontsize=8,
        color=INK,
        va="center",
    )
top.set_yscale("log")
top.set_ylabel("objective (−log-likelihood)", color=MUTED)
top.set_title("register(): objective per iteration", loc="left", color=INK)
top.text(152, top.get_ylim()[1], "150", color=MUTED, fontsize=8, va="top")
top.legend(frameon=False, loc="upper right")

first_label, first = next(iter(data.items()))
it = np.arange(1, len(first) + 1)
for color, j, name in ((SERIES[0], 1, "attachment"), (SERIES[1], 2, "regularity")):
    bottom.plot(it, first[:, j], color=color, linewidth=2, label=name)
    bottom.annotate(
        name,
        (it[-1], first[-1, j]),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK,
        va="center",
    )
bottom.set_ylim(0, None)
bottom.set_ylabel("term value", color=MUTED)
bottom.set_xlabel("L-BFGS iteration", color=MUTED)
bottom.set_title(f"{first_label}: attachment and regularity", loc="left", color=INK)
bottom.legend(frameon=False, loc="upper right")

fig.tight_layout()
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
