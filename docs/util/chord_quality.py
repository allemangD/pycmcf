"""
Quality of the chords as cortical columns (docs/backward-flow-ablation.md, chord selection).

    python docs/util/chord_quality.py states.npz [run-back.npz ...]

For the chords at state 0, a nearest-point reference (each white vertex to its closest pial
point), and the state 0 of each backward run given:

  - angle between each chord and the white surface normal, and the largest angle to its
    mesh neighbours' chords;
  - chord length;
  - straight segment leaving the gray matter: a point at depth 0.25, 0.5 or 0.75 inside the
    white surface or outside the pial surface (winding numbers against that chord set's own
    surfaces);
  - mesh faces whose prism between the depth layers 0, 0.5 and 1 has an inverted tetrahedron;
  - for backward runs, distance to the original surfaces and to the original vertices.

Then the chord direction against the chord-midpoint surface normal through the forward
states, to show how the forward flow regularises it.
"""
import sys
from pathlib import Path

import igl
import numpy as np

import common  # noqa: F401  (puts src/ on the path)

z = np.load(sys.argv[1])
S, F = z["states"], z["faces"].astype(np.int64)
u0, v0 = np.ascontiguousarray(S[0, :, :3]), np.ascontiguousarray(S[0, :, 3:])
adj = [np.array(a) for a in igl.adjacency_list(F)]


def outward_normals(x):
    n = igl.per_vertex_normals(x, F)
    return -n if np.mean(np.sum(n * (x - x.mean(0)), 1)) < 0 else n


def neighbour_spread(dn):
    return np.array(
        [np.degrees(np.arccos(np.clip(dn[j] @ dn[i], -1, 1))).max() for i, j in enumerate(adj)]
    )


def inverted_prisms(u, v):
    layers = [(1 - t) * u + t * v for t in (0, 0.5, 1)]
    vols = []
    for lo, hi in zip(layers[:-1], layers[1:]):
        a, b, c = (lo[F[:, k]] for k in range(3))
        A, B, C = (hi[F[:, k]] for k in range(3))
        for p, q, r, s in ((a, b, c, A), (b, c, A, B), (c, A, B, C)):
            vols.append(np.einsum("ij,ij->i", q - p, np.cross(r - p, s - p)))
    V = np.stack(vols, 1)
    return np.mean(np.any(V * np.sign(np.median(V)) < 0, axis=1))


def quality(u, v, white, pial):
    d = v - u
    L = np.linalg.norm(d, axis=1)
    dn = d / np.maximum(L, 1e-12)[:, None]
    ang = np.degrees(np.arccos(np.clip(np.sum(dn * outward_normals(u), 1), -1, 1)))
    leave = np.zeros(len(u), bool)
    for tau in (0.25, 0.5, 0.75):
        p = np.ascontiguousarray((1 - tau) * u + tau * v)
        leave |= (np.abs(igl.winding_number(white, F, p)) > 0.5) | (
            np.abs(igl.winding_number(pial, F, p)) < 0.5
        )
    nb = neighbour_spread(dn)
    return (
        f"angle to normal p50/p90 {np.median(ang):3.0f}/{np.percentile(ang, 90):3.0f} deg; "
        f"neighbour spread p50/p90 {np.median(nb):3.0f}/{np.percentile(nb, 90):3.0f} deg; "
        f"length p50 {np.median(L):.2f} mm; leaves gray matter {leave.mean():.2%}; "
        f"inverted prisms {inverted_prisms(u, v):.1%}"
    )


print(f"{'original chords':24s} {quality(u0, v0, u0, v0)}")
_, _, nearest = igl.point_mesh_squared_distance(u0, v0, F)
print(f"{'nearest pial point':24s} {quality(u0, np.ascontiguousarray(nearest), u0, v0)}")
for path in sys.argv[2:]:
    c = np.load(path)["states"][0]
    u, v = np.ascontiguousarray(c[:, :3]), np.ascontiguousarray(c[:, 3:])
    dw = np.sqrt(igl.point_mesh_squared_distance(u, u0, F)[0])
    dp = np.sqrt(igl.point_mesh_squared_distance(v, v0, F)[0])
    cw, cp = np.linalg.norm(u - u0, axis=1), np.linalg.norm(v - v0, axis=1)
    print(f"{Path(path).stem:24s} {quality(u, v, u, v)}")
    print(
        f"{'':24s} to original surface white / pial p50/p95 {np.median(dw):.2f}/"
        f"{np.percentile(dw, 95):.2f} / {np.median(dp):.2f}/{np.percentile(dp, 95):.2f} mm; "
        f"from original vertex {np.median(cw):.2f}/{np.percentile(cw, 95):.2f} / "
        f"{np.median(cp):.2f}/{np.percentile(cp, 95):.2f} mm"
    )

print("\nforward flow: chord against the midpoint-surface normal, and neighbour spread (deg)")
for k in sorted({0, 10, 20, 30, 35, len(S) - 2}):
    u, v = S[k, :, :3], S[k, :, 3:]
    nm = igl.per_vertex_normals(np.ascontiguousarray((u + v) / 2), F)
    d = v - u
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.abs(np.clip(np.sum(dn * nm, 1), -1, 1))))
    nb = neighbour_spread(dn)
    print(
        f"  state {k:2d}: angle p50/p90 {np.median(ang):4.1f}/{np.percentile(ang, 90):4.1f}; "
        f"neighbour spread p50/p90 {np.median(nb):4.1f}/{np.percentile(nb, 90):4.1f}"
    )
