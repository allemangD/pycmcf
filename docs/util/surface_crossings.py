"""Fraction of pial vertices inside the white surface (and white vertices outside the pial
surface) at selected states, by generalised winding number.

    python docs/util/surface_crossings.py states.npz [states...]

Any non-zero value at a state means a diffeomorphism of R^3 cannot carry disjoint
continuous surfaces onto it (the meshes themselves can still cross between vertices).
"""

import sys

import igl
import numpy as np

S_npz = np.load(sys.argv[1])
F, S = S_npz["faces"].astype(np.int64), S_npz["states"]
ks = [int(k) for k in sys.argv[2:]] or sorted(
    {0, 10, 20, 30, 40, 50, len(S) - 2, len(S) - 1}
)
print("state  pial inside white  white outside pial")
for k in ks:
    u, v = S[k, :, :3], S[k, :, 3:]
    inside = np.mean(np.abs(igl.winding_number(u, F, v)) > 0.5)
    outside = np.mean(np.abs(igl.winding_number(v, F, u)) < 0.5)
    print(f"{k:5d}  {inside:17.4f}  {outside:18.4f}")
