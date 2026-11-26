"""
Plot how the cortical area is distributed over theta1 (polar angle from the L/R pole of the
fitted ellipsoid), and the ring latitudes it implies (docs/backward-flow-ablation.md,
experiment 5). One column with a shared theta1 axis:

  1. pdf: fraction of the cortex per degree of theta1, raw and with the uniform floor;
  2. raw cdf, with the 32-band ring latitudes (pure quantiles);
  3. cdf with the floor (uniform_fraction), with its ring latitudes; raw cdf for reference.

Usage: python docs/util/plot_latitude_area.py out.png states.npz [states.npz ...]
           [--alpha 0.2] [--fits]
(the forward-flow states written by flow.flow(); several cases are averaged, as in
srep.equal_area_latitudes). --fits adds smooth sine-series fits of the latitudes,
theta1(s) = pi s + sum_k b_k sin(2 pi k s), to the cdf panels.
"""
import argparse

import matplotlib
import numpy as np

import common  # noqa: F401  (puts src/ on the path)
from srep import equal_area_latitudes, latitude_cdf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("states", nargs="+")
parser.add_argument("--alpha", type=float, default=0.2, help="uniform_fraction")
parser.add_argument("--bands", type=int, default=32)
parser.add_argument("--fits", action="store_true", help="add sine-series fits")
args = parser.parse_args()

cdfs = []
for path in args.states:
    z = np.load(path)
    S, F = z["states"], z["faces"].astype(np.int64)
    cdfs.append(latitude_cdf(S[-2], F, S[0]))  # last CMCF iterate, before Q
raw = np.mean(cdfs, axis=0)
grid = np.linspace(0, np.pi, len(raw))
deg = np.degrees(grid)
biased = (1 - args.alpha) * raw + args.alpha * grid / np.pi
lat_raw = np.degrees(equal_area_latitudes(cdfs, args.bands))
lat_biased = np.degrees(equal_area_latitudes(cdfs, args.bands, uniform_fraction=args.alpha))


def sine_fit(cdf, n_terms):
    """Least-squares fit of the quantile function: theta1(s) = pi s + sum b_k sin(2 pi k s)."""
    s = np.linspace(0, 1, 2001)
    q = np.interp(s, cdf, grid)
    B = np.stack([np.sin(2 * np.pi * k * s) for k in range(1, n_terms + 1)], 1)
    b = np.linalg.lstsq(B, q - np.pi * s, rcond=None)[0]
    fit = np.pi * s + B @ b
    return np.degrees(fit), s, np.degrees(np.abs(fit - q).max())


# pdf per degree of theta1, in 2-degree bins
h = 2.0
edges = np.arange(0, 180 + h, h)
centres = edges[:-1] + h / 2
pdf_raw = np.diff(np.interp(edges, deg, raw)) / h
pdf_biased = np.diff(np.interp(edges, deg, biased)) / h

ink, muted, rule = "#1f1f1e", "#6b6a64", "#e6e5e0"
blue, orange, aqua, yellow = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update(
    {"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
     "xtick.color": muted, "ytick.color": muted}
)
fig, ax = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
cases = f"{len(args.states)} case{'s' * (len(args.states) > 1)}"
floor = f"{args.alpha:.0%} uniform floor"

a = ax[0]
a.plot(centres, 100 * pdf_raw, color=blue, lw=2, label="raw")
a.plot(centres, 100 * pdf_biased, color=orange, lw=2, label=f"with {floor}")
a.axhline(100 / 180, color=muted, lw=1, ls=(0, (3, 3)), label="uniform in θ1")
a.set_ylabel("% of cortex per degree")
a.set_title(f"pdf of cortical area over θ1 ({cases})", loc="left", color=ink)
a.legend(frameon=False, fontsize=9, loc="upper center")


def cdf_panel(a, cdf, lat, color, title, reference=None):
    if reference is not None:
        a.plot(deg, reference, color=blue, lw=1.2, alpha=0.45, label="raw cdf")
    a.plot(deg, cdf, color=color, lw=2, label="cdf")
    a.plot(deg, deg / 180, color=muted, lw=1, ls=(0, (3, 3)), label="uniform in θ1")
    # each ring: the latitude where the cdf crosses k / bands
    levels = np.arange(1, args.bands) / args.bands
    a.vlines(lat, 0, levels, color=color, lw=0.6, alpha=0.6)
    a.plot(lat, levels, "o", color=color, ms=3)
    a.plot(lat, np.zeros_like(lat), "|", color=color, ms=10, mew=1.5)
    if args.fits:
        for n_terms, c in ((2, aqua), (4, yellow)):
            fit, s, dev = sine_fit(cdf, n_terms)
            a.plot(fit, s, color=c, lw=1.5, label=f"{n_terms} sine terms (max dev {dev:.1f}°)")
    a.set_ylim(0, 1)
    a.set_ylabel("fraction of cortex")
    a.set_title(title, loc="left", color=ink)
    a.legend(frameon=False, fontsize=9, loc="upper left")


cdf_panel(ax[1], raw, lat_raw, blue,
          f"raw cdf: {args.bands} equal-area bands, {np.sum((lat_raw > 60) & (lat_raw < 120))} rings in 60–120°")
cdf_panel(ax[2], biased, lat_biased, orange,
          f"cdf with {floor}: {np.sum((lat_biased > 60) & (lat_biased < 120))} rings in 60–120°",
          reference=raw)

ax[2].set_xlim(0, 180)
ax[2].set_xticks(range(0, 181, 30))
ax[2].set_xlabel("θ1, polar angle from the L/R pole (°); ticks on the axis = ring latitudes")
for a in ax:
    a.grid(color=rule, lw=0.8)
    a.set_axisbelow(True)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(args.out, dpi=130)
print(args.out)
