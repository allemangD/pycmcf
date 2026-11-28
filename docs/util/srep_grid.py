"""
Surface s-rep at a chosen grid resolution, and how the grid cells look on the original
surfaces.

    python docs/util/srep_grid.py states.npz out_dir N_THETA N_PHI [--uniform-fraction 0.2]

N_THETA latitude bands (N_THETA - 1 rings) and N_PHI longitudes. Latitudes are the
equal-area quantiles of this case (srep.latitude_cdf at the last CMCF iterate, with the
uniform floor), pole and meridian L and S (LSA), as in examples/main-backflow.py.
Writes, with prefix <out_dir>/srep-<N_THETA>x<N_PHI>:

  - -sampling.npz (srep.save_sampling) and .npz (chords, theta, faces, face, bary, ellipsoid);
  - -{white,pial,skeleton}.vtk with theta1, theta2 and, per point, the cell aspect.

Prints the spacing of the grid on the original surfaces: per quad (ring i to i+1, longitude
j to j+1), the latitude edge, the longitude edge, and the aspect log2(latitude / longitude),
against the median edge of the original mesh.
"""
import argparse
from pathlib import Path

import numpy as np

import common  # noqa: F401  (puts src/ on the path)
from backflow import _write
from srep import equal_area_latitudes, latitude_cdf, sampling, save_sampling, surface_srep

parser = argparse.ArgumentParser()
parser.add_argument("states", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("n_theta", type=int)
parser.add_argument("n_phi", type=int)
parser.add_argument("--uniform-fraction", type=float, default=0.2)
args = parser.parse_args()

z = np.load(args.states)
S, F = z["states"], z["faces"].astype(np.int64)
latitudes = equal_area_latitudes(
    [latitude_cdf(S[-2], F, S[0])], args.n_theta, uniform_fraction=args.uniform_fraction
)
s = sampling(latitudes, n_phi=args.n_phi)  # L, S in LSA
fit, chords = surface_srep(S, F, s)

prefix = args.out / f"srep-{args.n_theta}x{args.n_phi}"
args.out.mkdir(parents=True, exist_ok=True)
save_sampling(f"{prefix}-sampling.npz", s)
np.savez(f"{prefix}.npz", chords=chords, theta=fit.theta, faces=fit.grid_faces, face=fit.face,
         bary=fit.bary, ellipsoid_mu=fit.mu, ellipsoid_axes=fit.axes)

parts = {"white": chords[:, :3], "pial": chords[:, 3:],
         "skeleton": (chords[:, :3] + chords[:, 3:]) / 2}
mesh_edge = np.median(np.linalg.norm(S[0][F[:, 1], :3] - S[0][F[:, 0], :3], axis=1))
print(f"{args.n_theta} x {args.n_phi} grid, {len(chords)} samples; "
      f"original white mesh median edge {mesh_edge:.2f} mm")
print("  quad edges p5/p50/p95 (mm), aspect log2(lat/lon) p5/p50/p95:")
for name, V in parts.items():
    rings = V[1:-1].reshape(len(latitudes), args.n_phi, 3)
    lat = np.linalg.norm(rings[1:] - rings[:-1], axis=2)  # (n_theta - 1, n_phi)
    lon = np.linalg.norm(np.roll(rings, -1, axis=1) - rings, axis=2)[:-1]
    aspect = np.log2(np.maximum(lat, 1e-9) / np.maximum(lon, 1e-9))
    q = (5, 50, 95)
    print(f"  {name:9s} lat {np.percentile(lat, q).round(2)}  lon {np.percentile(lon, q).round(2)}"
          f"  aspect {np.percentile(aspect, q).round(2)}")
    per_point = np.zeros(len(V))
    per_point[1:-1] = np.vstack([aspect, aspect[-1:]]).ravel()  # last ring reuses its neighbour
    _write(Path(f"{prefix}-{name}.vtk"), V, fit.grid_faces,
           {"theta1": fit.theta[:, 0], "theta2": fit.theta[:, 1], "aspect": per_point})
