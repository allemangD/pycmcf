"""Runs register() on a white/pial pair into a scratch output directory.

    python docs/util/register_repro.py white.vtk pial.vtk out_dir [max_iterations=150] \
        [convergence_tolerance=1e-8]

Used to reproduce the NaN registration and to compare precision and tolerance settings
(docs/extended-varifold-fixes.md): run it from the working tree and from a worktree of
another commit, and compare the logged log-likelihoods and reconstructions.
Note register() writes <pial stem>-recon.vtk next to the pial input, so pass copies.
"""

import logging
import sys
from pathlib import Path

import common  # noqa: F401  (puts src/ on the path)

from register import register

logging.basicConfig(level=logging.INFO)
iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 150
tolerance = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-8
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
register(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    output_dir=out,
    max_iterations=iterations,
    convergence_tolerance=tolerance,
)
