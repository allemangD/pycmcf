"""
Backward flow without fitting: each stage's velocity field is the Gaussian-kernel (RBF)
interpolant of the forward flow's own vertex displacements, in R3.

    python docs/util/kernel_backflow.py states.npz out.npz --width W --lam L --mode euler|shoot
        [--time-points 5] [--grid 160x256]

Stage k + 1 -> k: centres x = the white and pial vertices of state k + 1 (2n points, R3),
displacements delta = state k - state k + 1, momenta a from (K + lam I) a = delta with
K = exp(-|x_i - x_j|^2 / W^2) (deformetrica's convention). With lam > 0 this is the
minimiser of |v|_V^2 + |v(x) - delta|^2 / lam: one-step LDDMM with a point-correspondence
data term. The stage then moves the carried points by
  - euler: p + v(p), one explicit step (not guaranteed invertible), or
  - shoot: geodesic shooting of (x, a) for unit time with RK2 and --time-points points,
    carrying p along (a diffeomorphism; vertices miss their targets by the linearisation).

Carried from the last CMCF iterate: the mesh vertices (white, pial) and the s-rep grid
(equal-area latitudes, 20% floor, L/S) at depths 0, 0.25, 0.5, 0.75, 1 along its chords.
Per stage, the Jacobian determinant of the stage map at every carried point (euler:
det(I + Dv); shoot: det of the Jacobian integrated along the path).

Prints one summary line and writes out.npz: vertices (n, 6) and depth points (5, m, 3) at
state 0, the s-rep's D chords and grid faces, and per stage the minimum determinant and the
number of points with det <= 0.
"""
import argparse
import time
from pathlib import Path

import igl
import numpy as np
import torch
from pykeops.torch import Vi, Vj

import common  # noqa: F401  (puts src/ on the path)
from srep import depth_points, equal_area_latitudes, fit_srep, interpolate, latitude_cdf, sampling

DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)

parser = argparse.ArgumentParser()
parser.add_argument("states", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--width", type=float, required=True)
parser.add_argument("--lam", type=float, required=True)
parser.add_argument("--mode", choices=("euler", "shoot"), required=True)
parser.add_argument("--time-points", type=int, default=5)
parser.add_argument("--grid", default="160x256")
args = parser.parse_args()

z = np.load(args.states)
S, F = z["states"][:-1], z["faces"].astype(np.int64)  # drop the Q state
n = S.shape[1]
n_theta, n_phi = map(int, args.grid.split("x"))
lat = equal_area_latitudes([latitude_cdf(S[-1], F, S[0])], n_theta, uniform_fraction=0.2)
fit = fit_srep(S[-1], F, sampling(lat, n_phi=n_phi))  # fit.points: chords at the last iterate
D = interpolate(S[0], F, fit.face, fit.bary)  # scheme D
m = len(D)

dev = torch.device("cuda")
center = S[-1].reshape(-1, 3).mean(0)


def T(a):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=dev)


def r3(c):
    """(n, 6) chords -> (2n, 3) points, centred."""
    return np.concatenate([c[:, :3], c[:, 3:]]) - center


s2 = args.width ** 2


def velocity(p, x, a):
    return ((-Vi(p).sqdist(Vj(x)) / s2).exp() * Vj(a)).sum(1)


def jacobian(p, x, a):
    """Dv at p, (len(p), 3, 3): -2/W^2 sum_j K(p, x_j) a_j (p - x_j)^T."""
    v = velocity(p, x, a)
    ax = (a[:, :, None] * x[:, None, :]).reshape(-1, 9)
    M = ((-Vi(p).sqdist(Vj(x)) / s2).exp() * Vj(ax)).sum(1)
    return -2 / s2 * (v[:, :, None] * p[:, None, :] - M.reshape(-1, 3, 3))


def momentum_force(x, a):
    """da/dt = -d/dx (1/2 sum_ij a_i.a_j K_ij) = 2/W^2 sum_j (a_i.a_j) K_ij (x_i - x_j)."""
    K = (-Vi(x).sqdist(Vj(x)) / s2).exp() * (Vi(a) * Vj(a)).sum()
    return 2 / s2 * (K * (Vi(x) - Vj(x))).sum(1)


def stage(p, x, delta):
    """Moves points p by the stage map; returns (new p, det of its Jacobian at p)."""
    a = (-Vi(x).sqdist(Vj(x)) / s2).exp().solve(Vi(delta), alpha=args.lam)
    I3 = torch.eye(3, dtype=p.dtype, device=dev)
    if args.mode == "euler":
        return p + velocity(p, x, a), torch.linalg.det(I3 + jacobian(p, x, a))
    J = I3.expand(len(p), 3, 3).clone()
    h = 1.0 / (args.time_points - 1)
    for _ in range(args.time_points - 1):  # Heun on (x, a, p, J)
        k1 = (velocity(x, x, a), momentum_force(x, a), velocity(p, x, a), jacobian(p, x, a) @ J)
        x2, a2, p2, J2 = x + h * k1[0], a + h * k1[1], p + h * k1[2], J + h * k1[3]
        k2 = (velocity(x2, x2, a2), momentum_force(x2, a2), velocity(p2, x2, a2),
              jacobian(p2, x2, a2) @ J2)
        x, a = x + h / 2 * (k1[0] + k2[0]), a + h / 2 * (k1[1] + k2[1])
        p, J = p + h / 2 * (k1[2] + k2[2]), J + h / 2 * (k1[3] + k2[3])
    return p, torch.linalg.det(J)


start = np.concatenate([r3(S[-1]), depth_points(fit.points, DEPTHS).reshape(-1, 3) - center])
p = T(start)
min_det, n_bad = [], []
t0 = time.perf_counter()
for k in range(len(S) - 1, 0, -1):
    p, det = stage(p, T(r3(S[k])), T(r3(S[k - 1]) - r3(S[k])))
    min_det.append(det.min().item())
    n_bad.append(int((det <= 0).sum().item()))
elapsed = time.perf_counter() - t0
p = p.cpu().numpy() + center
verts = np.hstack(np.split(p[:2 * n], 2))
depth = p[2 * n:].reshape(len(DEPTHS), m, 3)

# scores at state 0
white, pial = np.ascontiguousarray(S[0][:, :3]), np.ascontiguousarray(S[0][:, 3:])
ev = np.concatenate([np.linalg.norm(verts[:, :3] - white, axis=1),
                     np.linalg.norm(verts[:, 3:] - pial, axis=1)])
es = np.concatenate([np.linalg.norm(depth[0] - D[:, :3], axis=1),
                     np.linalg.norm(depth[-1] - D[:, 3:], axis=1)])
inner = np.ascontiguousarray(depth[1:-1].reshape(-1, 3))
inside_white = np.mean(np.abs(igl.winding_number(white, F, inner)) > 0.5)
outside_pial = np.mean(np.abs(igl.winding_number(pial, F, inner)) < 0.5)
stages_bad = sum(b > 0 for b in n_bad)


def q(x):
    return f"{np.median(x):.3f}/{np.percentile(x, 95):.3f}/{x.max():.2f}"


print(f"{args.states.parent.name:10s} stages {len(S) - 1:4d}  W {args.width:4.1f}  lam {args.lam:7.0e}  "
      f"{args.mode:5s}  {elapsed:6.0f} s | vertex err p50/p95/max {q(ev)} | s-rep vs D {q(es)} | "
      f"interior in-white {inside_white:.2%} out-pial {outside_pial:.2%} | "
      f"min det {min(min_det):.3f}, stages with det<=0 {stages_bad}, worst {max(n_bad)} pts",
      flush=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, vertices=verts, depth=depth, D=D, grid_faces=fit.grid_faces,
         min_det=np.array(min_det[::-1]), n_bad=np.array(n_bad[::-1]))
