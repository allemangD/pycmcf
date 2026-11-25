"""Per-state errors of one or more backward-flow runs against the forward states.

    python docs/util/compare_backflow.py states.npz run-a-back.npz [run-b-back.npz ...]

For each run (its backward states, aligned with the forward ones): white and pial error
p50/p95, chord length error p95 and chord angle p50/p95 at selected states, then at the final
state the max errors, the fraction of chords more than 10 degrees off, and the fraction of
pial vertices inside the white surface (with the forward state 0 value for reference).
"""

import sys
from pathlib import Path

import igl
import numpy as np
from common import percentiles

from backflow import chord_errors

z = np.load(sys.argv[1])
F, S = z["faces"].astype(np.int64), z["states"]
runs = {Path(p).stem: np.load(p)["states"] for p in sys.argv[2:]}
K = len(S) - 1
ks = sorted({K - 1, 50, 40, 30, 28, 25, 20, 10, 5, 0} & set(range(K + 1)), reverse=True)

for name, B in runs.items():
    print(f"== {name}")
    print("state  white p50/p95  pial p50/p95  length p95  angle p50/p95")
    for k in ks:
        e = chord_errors(B[k], S[k])
        print(
            f"{k:5d}  {percentiles(e['white_error']):>13s}  {percentiles(e['pial_error']):>12s}"
            f"  {np.percentile(np.abs(e['chord_length_error']), 95):10.2f}"
            f"  {percentiles(e['chord_angle'], (50, 95)):>13s}"
        )
    e = chord_errors(B[0], S[0])
    inside = np.mean(np.abs(igl.winding_number(B[0][:, :3], F, B[0][:, 3:])) > 0.5)
    truth = np.mean(np.abs(igl.winding_number(S[0][:, :3], F, S[0][:, 3:])) > 0.5)
    print(
        f"final: max white {e['white_error'].max():.2f}  max pial {e['pial_error'].max():.2f}"
        f"  chords > 10 deg {np.mean(e['chord_angle'] > 10):.3f}"
        f"  pial inside white {inside:.4f} (state 0: {truth:.4f})"
    )
