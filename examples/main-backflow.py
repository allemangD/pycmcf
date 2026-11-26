import logging
from pathlib import Path

import numpy as np

from backflow import _write, backflow
from srep import (
    equal_area_latitudes,
    latitude_cdf,
    load_sampling,
    sampling,
    save_sampling,
    surface_srep,
)

logging.basicConfig(level=logging.INFO)

# forward-flow states written by flow() in main-6d-flow.py, one per case
cases = [Path("data/oasis4-inner-decimated-flow-states.npz")]
print("inputs:\n  " + "\n  ".join(map(str, cases)))

# s-rep sampling: frozen once for the dataset and reused for cases added later. Ring
# latitudes split the population's cortical area equally; theta1 = 0 at L and longitude 0
# at S (for the LSA frame of the OASIS-4 data).
sampling_path = Path("data/srep-sampling.npz")
if sampling_path.exists():
    srep = load_sampling(sampling_path)
else:
    cdfs = []
    for path in cases:
        z = np.load(path)
        F, S = z["faces"].astype(np.int64), z["states"]
        cdfs.append(latitude_cdf(S[-2], F, S[0]))  # last CMCF iterate, before Q
    # 20% of the rings uniform: keeps 5 rings (not 3) in the belt around the callosal sulcus
    latitudes = equal_area_latitudes(cdfs, n_theta=32, uniform_fraction=0.2)
    pole, meridian = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)  # L, S in LSA
    srep = sampling(latitudes, n_phi=64, pole=pole, meridian=meridian)
    save_sampling(sampling_path, srep)
print(f"s-rep sampling:\n  {sampling_path}")

# surface s-rep: the grid on the original chords; no backward flow needed
for path in cases:
    z = np.load(path)
    F, S = z["faces"].astype(np.int64), z["states"]
    fit, chords = surface_srep(S, F, srep)
    out = path.with_name(f"{path.stem}-srep.npz")
    np.savez(out, chords=chords, theta=fit.theta, faces=fit.grid_faces,
             face=fit.face, bary=fit.bary, ellipsoid_mu=fit.mu, ellipsoid_axes=fit.axes)
    arrays = {"theta1": fit.theta[:, 0], "theta2": fit.theta[:, 1]}
    parts = {"white": chords[:, :3], "pial": chords[:, 3:],
             "skeleton": (chords[:, :3] + chords[:, 3:]) / 2}
    for name, V in parts.items():
        _write(out.with_name(f"{out.stem}-{name}.vtk"), V, fit.grid_faces, arrays)
    print(f"surface s-rep:\n  {out}")

# backward flow of the mesh: R6 is the baseline; R3 (white -> white and pial -> pial) is
# an ablation. For gray-matter points, carry them in R3 together with the s-rep's surface
# points (docs/util/interior_r3.py).
for path in cases:
    for space in ("R6", "R3"):
        outputs = backflow(path, space=space)
        print(f"backflow {space} outputs:\n  " + "\n  ".join(map(str, outputs)))
