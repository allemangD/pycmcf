#!/usr/bin/env python
"""
Run cMCF on a closed genus-0 mesh while carrying (a) a planar grid through the centre,
with finite-difference offsets for Jacobians, and (b) random points just inside and just
outside the surface. Reports Jacobian statistics and whether points stayed on their
side of the surface; saves a figure and the flowed surface.

  python demo_cmcf_extend.py --mesh oasis4-inner.vtk --decimate 0.9 --keops --device cuda
  python demo_cmcf_extend.py --synthetic folded       # small CPU test
"""
import argparse, json, logging, math, os, time
import numpy as np
import torch
import vtk
from vtk.util import numpy_support as ns
from cmcf_extend import LocalLinearExtension, flow_3d_extended, median_edge


def to_pd(V, F):
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(V, float), deep=True))
    ca = vtk.vtkCellArray()
    ca.SetData(ns.numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(F) + 1, 3, dtype=np.int64), deep=True),
               ns.numpy_to_vtkIdTypeArray(np.ascontiguousarray(F, np.int64).ravel(), deep=True))
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetPolys(ca)
    return pd


def from_pd(pd):
    return (np.asarray(pd.points.data, float),
            np.asarray(pd.polys.connectivity_array).reshape(-1, 3).astype(np.int64))


def load(path, reduction):
    r = vtk.vtkPolyDataReader(); r.SetFileName(path)
    r = vtk.vtkTriangleFilter(input_connection=r.output_port)
    if reduction > 0:
        r = vtk.vtkQuadricDecimation(input_connection=r.output_port, target_reduction=reduction)
    r.Update()
    return from_pd(r.output)


def synthetic(kind):
    src = vtk.vtkPlatonicSolidSource(); src.SetSolidTypeToIcosahedron()
    sub = vtk.vtkLoopSubdivisionFilter(); sub.SetInputConnection(src.GetOutputPort())
    sub.SetNumberOfSubdivisions(4); sub.Update()
    U, F = from_pd(sub.GetOutput()); U /= np.linalg.norm(U, axis=1, keepdims=True)
    if kind == "ellipsoid":
        return U * np.array([40.0, 25.0, 20.0]), F
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(7, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    f = np.tanh(1.2 * np.cos(6.0 * U @ dirs.T + rng.uniform(0, 2 * np.pi, 7)).sum(1) / math.sqrt(7))
    return U * (30.0 + 5.0 * f)[:, None] * np.array([1.3, 1.0, 0.85]), F


def inside(V, F, P):
    s = vtk.vtkSelectEnclosedPoints(); s.SetSurfaceData(to_pd(V, F)); s.SetTolerance(1e-6)
    pd = vtk.vtkPolyData(); pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(P, float), deep=True)); pd.SetPoints(pts)
    s.SetInputData(pd); s.Update()
    return ns.vtk_to_numpy(s.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)


def surface_distance(V, F, P):
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(to_pd(V, F))
    return np.array([abs(f.EvaluateFunction(*p)) for p in P])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh"); ap.add_argument("--synthetic", choices=["ellipsoid", "folded"])
    ap.add_argument("--decimate", type=float, default=0.0)
    ap.add_argument("--out", default="cmcf_demo")
    ap.add_argument("--grid", type=float, default=None, help="slice grid spacing (mm)")
    ap.add_argument("--n-near", type=int, default=1000)
    ap.add_argument("--near-band", type=float, default=3.0, help="mm either side of the surface")
    ap.add_argument("--max-iters", type=int, default=100)
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--n-correct", type=int, default=1)
    ap.add_argument("--s0-factor", type=float, default=1.0)
    ap.add_argument("--keops", action="store_true"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(a.out, exist_ok=True)

    V, F = synthetic(a.synthetic) if a.synthetic else load(a.mesh, a.decimate)
    h = median_edge(V, F)
    print(f"{len(V)} vertices, median edge {h:.2f} mm")
    rng = np.random.default_rng(0)

    # (a) slice grid through the centre, normal to the shortest bounding-box axis
    lo, hi = V.min(0) - 8, V.max(0) + 8
    ax = int(np.argmin(hi - lo)); u_ax, v_ax = [k for k in range(3) if k != ax]
    g = a.grid or max(1.0, (hi - lo).max() / 60)
    gu, gv = np.arange(lo[u_ax], hi[u_ax], g), np.arange(lo[v_ax], hi[v_ax], g)
    UU, VV = np.meshgrid(gu, gv, indexing="xy")
    P = np.zeros((UU.size, 3)); P[:, u_ax], P[:, v_ax], P[:, ax] = UU.ravel(), VV.ravel(), V[:, ax].mean()
    eps = 0.1 * g
    # (b) random points within near_band of the surface, labelled inside / outside
    vn = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3): np.add.at(vn, F[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    idx = rng.integers(0, len(V), a.n_near)
    Pn = V[idx] + vn[idx] * rng.uniform(-a.near_band, a.near_band, (a.n_near, 1))
    in0 = inside(V, F, Pn)
    keep = surface_distance(V, F, Pn) > 0.25 * h            # avoid points on the surface itself
    Pn, in0 = Pn[keep], in0[keep]
    in_grid0 = inside(V, F, P)
    Q = np.concatenate([P] + [P + eps * np.eye(3)[k] for k in range(3)] + [Pn])
    print(f"{len(P)} grid points (x4 for Jacobians), {len(Pn)} near-surface points")

    ext = LocalLinearExtension(kappa=a.kappa, ridge=a.ridge, n_correct=a.n_correct,
                               keops=a.keops, device=a.device,
                               dtype=torch.float32 if a.device != "cpu" else torch.float64)
    t = time.time()
    Vf, Qf, info = flow_3d_extended(V, F, Q, max_iters=a.max_iters, substeps=a.substeps,
                                    s0_factor=a.s0_factor, ext=ext)
    secs = time.time() - t
    n = len(P)
    Y = Qf[:n]
    J = np.stack([(Qf[(k + 1) * n:(k + 2) * n] - Y) / eps for k in range(3)], -1)
    det = np.linalg.det(J)
    in1 = inside(Vf, F, Qf[4 * n:])
    mc = Vf.mean(0)       # flow_3d recentres; express the input in the same frame for |u|
    disp = np.linalg.norm(Y - (P - V.mean(0)), axis=1)
    res = dict(vertices=len(V), iters=info["iters"], seconds=secs,
               det_min=float(det.min()), det_nonpos_pct=float(100 * np.mean(det <= 0)),
               det_min_inside=float(det[in_grid0].min()) if in_grid0.any() else None,
               det_p1_inside=float(np.percentile(det[in_grid0], 1)) if in_grid0.any() else None,
               near_inside_kept_pct=float(100 * np.mean(in1[in0])),
               near_outside_kept_pct=float(100 * np.mean(~in1[~in0])),
               final_radius_cv=float(np.std(np.linalg.norm(Vf - mc, axis=1)) /
                                     np.mean(np.linalg.norm(Vf - mc, axis=1))))
    print(json.dumps(res, indent=1))
    json.dump(dict(res, **{k: info[k] for k in ("rate", "s0", "speed")}),
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    w = vtk.vtkPolyDataWriter(); w.SetFileName(os.path.join(a.out, "flowed.vtk"))
    w.SetInputData(to_pd(Vf, F)); w.SetFileTypeToBinary(); w.Write()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    shp = UU.shape
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ext_ = [gu[0], gu[-1], gv[0], gv[-1]]
    im = axs[0].imshow(disp.reshape(shp), origin="lower", extent=ext_, cmap="magma")
    fig.colorbar(im, ax=axs[0], fraction=0.046, label="|total displacement| (mm)")
    im = axs[1].imshow(np.log2(np.clip(det, 1e-3, None)).reshape(shp), origin="lower",
                       extent=ext_, cmap="RdBu_r", vmin=-2, vmax=2)
    fig.colorbar(im, ax=axs[1], fraction=0.046, label="log2 det J (initial -> final)")
    for axx in axs[:2]:
        axx.contour(UU, VV, in_grid0.reshape(shp), [0.5], colors="c", linewidths=0.8)
        axx.set_title(["displacement", "volume change"][list(axs).index(axx)])
    # warped grid lines (final positions, in the two in-plane coordinates)
    Yg = Y.reshape(shp + (3,))
    step = max(1, shp[0] // 30)
    for i in range(0, shp[0], step):
        axs[2].plot(Yg[i, :, u_ax], Yg[i, :, v_ax], lw=0.4, c="0.3")
    for j in range(0, shp[1], step):
        axs[2].plot(Yg[:, j, u_ax], Yg[:, j, v_ax], lw=0.4, c="0.3")
    cut = np.abs(Vf[:, ax] - np.median(Yg[..., ax])) < 1.5 * h
    axs[2].scatter(Vf[cut, u_ax], Vf[cut, v_ax], s=1, c="c")
    axs[2].set_aspect("equal"); axs[2].set_title("slice grid after the flow (final surface near it: cyan)")
    for axx in axs: axx.set_xticks([]); axx.set_yticks([])
    fig.suptitle(f"cMCF + local-linear extension: {info['iters']} iterations, "
                 f"min det J {res['det_min']:.2f}, {res['det_nonpos_pct']:.2f}% det J <= 0", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "slice.png"), dpi=120)


if __name__ == "__main__":
    main()
