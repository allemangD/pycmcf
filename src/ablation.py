#!/usr/bin/env python
"""
Kernel / attachment ablation for white -> pial LDDMM.

Real data (GPU + KeOps):
  python ablation.py --source oasis4-inner.vtk --target oasis4-outer.vtk \
                     --device cuda --keops --out results_oasis4
Synthetic smoke test (CPU, dense torch):
  python ablation.py --synthetic rigid --out smoke_rigid --iters 30 --steps 6 \
                     --attach-reduction 0.75 --dtype float32

Arms (all share control points, optimizer, kernel width, and regularization):
  gauss_vf        Gaussian kernel + Deformetrica varifold        (baseline)
  gauss_ovf       Gaussian kernel + oriented varifold
  norm_ovf        normalized Gaussian, fixed width
  nvarLO_ovf      normalized, width grows with distance (kappa = --kappa-lo)
  nvarHI_ovf      normalized, width grows faster        (kappa = --kappa-hi)
  nvarLOaff_ovf   nvarLO + global affine kernel term
  nvarLO_vf       nvarLO with the unoriented varifold (kernel x attachment interaction)
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import vtk
import vtk.util.numpy_support as ns
from scipy import ndimage
from scipy.spatial import cKDTree

from lddmm_core import (
    DefKernel,
    Varifold,
    centers_normals,
    decimate,
    flow,
    from_polydata,
    load_mesh,
    make_synthetic,
    save_mesh,
    shoot,
    signed_volume,
    to_polydata,
    vertex_normals,
)


def arm_table(a):
    lo, hi, g = a.kappa_lo, a.kappa_hi, a.affine_gain
    return {
        "gauss_vf": dict(kind="gauss", kappa=0.0, aff=0.0, oriented=False),
        "gauss_ovf": dict(kind="gauss", kappa=0.0, aff=0.0, oriented=True),
        "norm_ovf": dict(kind="norm", kappa=0.0, aff=0.0, oriented=True),
        "nvarLO_ovf": dict(kind="norm", kappa=lo, aff=0.0, oriented=True),
        "nvarHI_ovf": dict(kind="norm", kappa=hi, aff=0.0, oriented=True),
        "nvarLOaff_ovf": dict(kind="norm", kappa=lo, aff=g, oriented=True),
        "nvarLO_vf": dict(kind="norm", kappa=lo, aff=0.0, oriented=False),
    }


# ============================================================== geometry utilities
def outward(V, F):
    """Flip winding if needed so every mesh is outward-oriented (consistent varifolds)."""
    return (V, F) if signed_volume(V, F) > 0 else (V, F[:, ::-1].copy())


def convexity(V, F, iters=60):
    """sulc-like convexity: offset of each vertex from a heavily smoothed copy, along the
    smoothed outward normal. Negative = fundus, positive = crown."""
    sm = vtk.vtkWindowedSincPolyDataFilter()
    sm.SetInputData(to_polydata(V, F))
    sm.SetNumberOfIterations(iters)
    sm.SetPassBand(0.0005)
    sm.NonManifoldSmoothingOn()
    sm.NormalizeCoordinatesOn()
    sm.Update()
    Vs, _ = from_polydata(sm.GetOutput())
    return np.einsum("ij,ij->i", V - Vs, vertex_normals(Vs, F))


def surface_distances(Va, Fa, Vb, Fb):
    """Point-to-surface distances a->b (per vertex of a) and b->a."""
    f = vtk.vtkDistancePolyDataFilter()
    f.SetInputData(0, to_polydata(Va, Fa))
    f.SetInputData(1, to_polydata(Vb, Fb))
    f.SignedDistanceOff()
    f.ComputeSecondDistanceOn()
    if hasattr(f, "ComputeCellCenterDistanceOff"):
        f.ComputeCellCenterDistanceOff()
    f.Update()
    dab = ns.vtk_to_numpy(f.GetOutput().GetPointData().GetArray("Distance"))
    dba = ns.vtk_to_numpy(
        f.GetSecondDistanceOutput().GetPointData().GetArray("Distance")
    )
    return np.abs(dab), np.abs(dba)


class Grid:
    def __init__(self, lo, hi, h):
        self.h, self.origin = h, np.asarray(lo, float)
        self.dims = np.ceil((np.asarray(hi) - lo) / h).astype(int) + 1  # (nx, ny, nz)
        nx, ny, nz = self.dims
        z, y, x = np.meshgrid(
            np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
        )
        self.pts = np.stack([x, y, z], -1) * h + self.origin  # (nz, ny, nx, 3)

    def voxelize(self, V, F):
        img = vtk.vtkImageData()
        img.SetOrigin(*self.origin)
        img.SetSpacing(self.h, self.h, self.h)
        img.SetDimensions(*[int(d) for d in self.dims])
        img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        img.GetPointData().GetScalars().Fill(1)
        st = vtk.vtkPolyDataToImageStencil()
        st.SetInputData(to_polydata(V, F))
        st.SetOutputOrigin(*self.origin)
        st.SetOutputSpacing(self.h, self.h, self.h)
        st.SetOutputWholeExtent(img.GetExtent())
        st.Update()
        cut = vtk.vtkImageStencil()
        cut.SetInputData(img)
        cut.SetStencilConnection(st.GetOutputPort())
        cut.SetBackgroundValue(0)
        cut.Update()
        nx, ny, nz = self.dims
        a = ns.vtk_to_numpy(cut.GetOutput().GetPointData().GetScalars())
        return a.reshape(nz, ny, nx).astype(bool)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else float("nan")


def edt(m, h):
    return ndimage.distance_transform_edt(m, sampling=h)


# ============================================================== fitting
def fit_arm(spec, c0, Xs, Fs, target_cn, args, dev, dt, gdeg_ref, ell):
    kern = DefKernel(
        spec["kind"],
        args.sigma,
        kappa=spec["kappa"],
        affine_gain=spec["aff"],
        keops=args.keops,
    )
    kern.calibrate(c0)
    att = Varifold(target_cn, args.sigma_w, spec["oriented"], args.sigma_n, args.keops)
    Fs_t = torch.as_tensor(Fs, device=dev)
    with torch.no_grad():
        A0 = float(att(centers_normals(Xs, Fs_t)))
    H_ref = len(c0) * ell**2 / gdeg_ref
    a0 = torch.zeros_like(c0, requires_grad=True)
    opt = torch.optim.LBFGS(
        [a0],
        lr=1.0,
        max_iter=args.iters,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )
    trace = []

    def closure():
        opt.zero_grad()
        x1, H0, _ = shoot(kern, c0, a0, Xs, args.steps, create_graph=True)
        A = att(centers_normals(x1, Fs_t)) / A0
        loss = A + args.reg * H0 / H_ref
        loss.backward()
        trace.append((loss.item(), A.item(), (H0 / H_ref).item()))
        return loss

    t = time.time()
    opt.step(closure)
    a_best = a0.detach()
    _, _, traj = shoot(kern, c0, a_best, None, args.steps, create_graph=False)
    return (
        kern,
        traj,
        dict(
            fit_seconds=time.time() - t,
            n_evals=len(trace),
            A_rel=trace[-1][1],
            H_rel=trace[-1][2],
            loss=trace[-1][0],
            trace=trace,
        ),
    )


# ============================================================== metrics
class Evaluator:
    def __init__(self, S, T, args, dev, dt, truth=None):
        self.S, self.T, self.args, self.dev, self.dt, self.truth = (
            S,
            T,
            args,
            dev,
            dt,
            truth,
        )
        (Vs, Fs), (Vt, Ft) = S, T
        h, pad = args.voxel, args.close_radius + 6.0
        lo = np.minimum(Vs.min(0), Vt.min(0)) - pad
        hi = np.maximum(Vs.max(0), Vt.max(0)) + pad
        self.g = g = Grid(lo, hi, h)
        self.inS, self.inT = g.voxelize(Vs, Fs), g.voxelize(Vt, Ft)
        dS = np.where(self.inS, edt(self.inS, h), edt(~self.inS, h))
        dT = np.where(self.inT, edt(self.inT, h), edt(~self.inT, h))
        r = args.close_radius
        self.closeT = edt(edt(~self.inT, h) <= r, h) > r  # morphological closing
        self.openT = edt(~(edt(self.inT, h) > r), h) <= r  # morphological opening
        k = args.jac_stride
        sl = (slice(0, None, k),) * 3
        self.jac_pts = g.pts[sl].reshape(-1, 3)
        deep = (self.inS & (dS >= args.deep))[sl].ravel()
        ext = (~self.inT & (dT >= args.deep))[sl].ravel()
        near = ((dS < args.band) | (dT < args.band))[sl].ravel()
        self.regions = dict(
            deep_interior=deep, near_band=near & ~deep & ~ext, exterior=ext
        )
        # source-surface descriptors
        self.cvx = convexity(Vs, Fs)
        q = np.quantile(self.cvx, [1 / 3, 2 / 3])
        self.bins = dict(
            fundus=self.cvx < q[0],
            wall=(self.cvx >= q[0]) & (self.cvx < q[1]),
            crown=self.cvx >= q[1],
        )
        self.nT = vertex_normals(Vt, Ft)
        self.treeT = cKDTree(Vt)
        # slice for the figure: perpendicular to the longest axis, through the center
        self.slice_axis = int(np.argmax(g.dims))  # 0=x,1=y,2=z
        self.slice_index = int(g.dims[self.slice_axis] // 2)

    def _t(self, X):
        return torch.as_tensor(X, device=self.dev, dtype=self.dt)

    def slice_pts(self):
        g, ax, i = self.g, self.slice_axis, self.slice_index
        arr_axis = 2 - ax  # pts is (nz, ny, nx, 3)
        return np.take(g.pts, i, axis=arr_axis), arr_axis

    def mask_slice(self, m):
        return np.take(m, self.slice_index, axis=2 - self.slice_axis)

    def jac(self, push, P, eps=0.25):
        X = np.concatenate([P] + [P + eps * e for e in np.eye(3)])
        Y = push(X)
        n = len(P)
        J = np.stack(
            [(Y[(k + 1) * n : (k + 2) * n] - Y[:n]) / eps for k in range(3)], -1
        )
        return Y[:n] - P, np.linalg.det(J)

    def evaluate(self, push, pull=None):
        (Vs, Fs), (Vt, Ft) = self.S, self.T
        args, out = self.args, {}
        Vd = push(Vs)
        # --- surface fit
        dst, dts = surface_distances(Vd, Fs, Vt, Ft)
        out["fit_mean_mm"] = float(0.5 * (dst.mean() + dts.mean()))
        out["fit_p95_mm"] = float(max(np.percentile(dst, 95), np.percentile(dts, 95)))
        for b, m in self.bins.items():
            out[f"fit_{b}_mm"] = float(dst[m].mean())
        # --- wrong-bank matches
        nD = vertex_normals(Vd, Fs)
        _, idx = self.treeT.query(Vd)
        out["wrong_bank_pct"] = float(
            100 * np.mean(np.einsum("ij,ij->i", nD, self.nT[idx]) < 0)
        )
        # --- Dice family
        inD = self.g.voxelize(Vd, Fs)
        out["dice_volume"] = dice(inD, self.inT)
        out["dice_sulcal"] = dice(self.closeT & ~inD, self.closeT & ~self.inT)
        out["dice_gyral"] = dice(inD & ~self.openT, self.inT & ~self.openT)
        # --- ambient field
        u_surf = np.linalg.norm(Vd - Vs, axis=1)
        out["disp_surface_median_mm"] = float(np.median(u_surf))
        u, det = self.jac(push, self.jac_pts)
        um = np.linalg.norm(u, axis=1)
        out["jac_min"] = float(det.min())
        out["jac_nonpos_pct"] = float(100 * np.mean(det <= 0))
        for r, m in self.regions.items():
            if m.any():
                out[f"{r}_disp_median_mm"] = float(np.median(um[m]))
                out[f"{r}_jac_p1"] = float(np.percentile(det[m], 1))
                out[f"{r}_jac_p99"] = float(np.percentile(det[m], 99))
        out["deep_to_surface_disp_ratio"] = out.get(
            "deep_interior_disp_median_mm", 0
        ) / max(out["disp_surface_median_mm"], 1e-9)
        # --- synthetic interior ground truth
        if self.truth is not None:
            out.update(self.interior_truth(push, pull))
        # --- slice for figure
        P, _ = self.slice_pts()
        shp = P.shape[:2]
        us, dets = self.jac(push, P.reshape(-1, 3))
        extras = dict(
            Vd=Vd,
            fit=dst,
            slice_u=np.linalg.norm(us, axis=1).reshape(shp),
            slice_logjac=np.log2(np.clip(dets, 1e-3, None)).reshape(shp),
            slice_inD=self.mask_slice(inD),
        )
        return out, extras

    def interior_truth(self, push, pull):
        R, t = self.truth["R"], self.truth["t"]
        c, ax = np.array([8.0, 5.0, -4.0]), np.array([7.0, 4.0, 5.0])
        inE = lambda X: (((X - c) / ax) ** 2).sum(1) <= 1
        T = lambda X: X @ R.T + t
        # target-space voxels around the true image of the ellipsoid
        h = self.args.voxel / 2
        lo, hi = T(c) - ax.max() - 6, T(c) + ax.max() + 6
        gg = np.stack(
            np.meshgrid(*[np.arange(l, u, h) for l, u in zip(lo, hi)], indexing="ij"),
            -1,
        ).reshape(-1, 3)
        m_true = inE((gg - t) @ R)
        res = {}
        if pull is not None:
            res["dice_interior_truth"] = dice(inE(pull(gg)), m_true)
        rng = np.random.default_rng(0)
        Xe = c + ax * rng.uniform(-1, 1, (20000, 3))
        Xe = Xe[inE(Xe)][:3000]
        res["interior_err_mean_mm"] = float(
            np.linalg.norm(push(Xe) - T(Xe), axis=1).mean()
        )
        res["interior_truth_disp_mm"] = float(np.linalg.norm(T(Xe) - Xe, axis=1).mean())
        return res


# ============================================================== report
COLS = [
    ("fit_mean_mm", "fit mean"),
    ("fit_p95_mm", "fit p95"),
    ("fit_fundus_mm", "fundus"),
    ("fit_crown_mm", "crown"),
    ("wrong_bank_pct", "wrong-bank %"),
    ("dice_volume", "Dice vol"),
    ("dice_sulcal", "Dice sulc"),
    ("dice_gyral", "Dice gyr"),
    ("deep_to_surface_disp_ratio", "deep/surf |u|"),
    ("exterior_disp_median_mm", "ext |u|"),
    ("jac_min", "min detJ"),
    ("deep_interior_jac_p1", "deep detJ p1"),
    ("near_band_jac_p1", "band detJ p1"),
    ("dice_interior_truth", "Dice interior"),
    ("interior_err_mean_mm", "interior err"),
    ("A_rel", "A/A0"),
    ("H_rel", "H/Href"),
    ("fit_seconds", "fit s"),
]


def write_report(results, path):
    cols = [c for c in COLS if any(c[0] in r for r in results.values())]
    lines = [
        "| arm | " + " | ".join(c[1] for c in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for name, r in results.items():
        cells = [
            ("%.3g" % r[k]) if isinstance(r.get(k), (int, float)) else ""
            for k, _ in cols
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    open(path, "w").write("\n".join(lines) + "\n")
    return "\n".join(lines)


def make_figure(ev, extras, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(extras)
    fig, axs = plt.subplots(
        2, len(names), figsize=(3.2 * len(names), 6.4), squeeze=False
    )
    sS, sT = ev.mask_slice(ev.inS), ev.mask_slice(ev.inT)
    vmax = max(np.percentile(e["slice_u"], 99) for e in extras.values())
    for j, n in enumerate(names):
        e = extras[n]
        for i, (img, kw) in enumerate(
            [
                (e["slice_u"], dict(cmap="magma", vmin=0, vmax=vmax)),
                (e["slice_logjac"], dict(cmap="RdBu_r", vmin=-1, vmax=1)),
            ]
        ):
            ax = axs[i, j]
            im = ax.imshow(img, origin="lower", **kw)
            ax.contour(sS, [0.5], colors="c", linewidths=0.6)
            ax.contour(sT, [0.5], colors="w" if i == 0 else "k", linewidths=0.6)
            ax.contour(
                e["slice_inD"], [0.5], colors="lime", linewidths=0.6, linestyles="--"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(n, fontsize=9)
        if j == len(names) - 1:
            fig.colorbar(
                axs[0, j].images[0], ax=axs[0, j], fraction=0.046, label="|u| (mm)"
            )
            fig.colorbar(
                axs[1, j].images[0], ax=axs[1, j], fraction=0.046, label="log2 det J"
            )
    fig.suptitle(
        "cyan: source  white/black: target  lime dashed: deformed source", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ============================================================== main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source")
    p.add_argument("--target")
    p.add_argument("--synthetic", choices=["thicken", "rigid"])
    p.add_argument("--out", default="ablation_out")
    p.add_argument(
        "--arms", default="all", help="all | none (report only) | comma list"
    )
    p.add_argument("--ctrl-reduction", type=float, default=0.9)
    p.add_argument("--attach-reduction", type=float, default=0.75)
    p.add_argument(
        "--sigma", type=float, default=5.0, help="deformation kernel width (mm)"
    )
    p.add_argument("--kappa-lo", type=float, default=0.25)
    p.add_argument("--kappa-hi", type=float, default=1.0)
    p.add_argument("--affine-gain", type=float, default=10.0)
    p.add_argument(
        "--sigma-w", type=lambda s: [float(v) for v in s.split(",")], default=[4.0, 2.0]
    )
    p.add_argument("--sigma-n", type=float, default=0.5)
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--keops", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--voxel", type=float, default=1.0)
    p.add_argument("--jac-stride", type=int, default=3)
    p.add_argument("--deep", type=float, default=5.0)
    p.add_argument("--band", type=float, default=3.0)
    p.add_argument("--close-radius", type=float, default=6.0)
    p.add_argument("--no-meshes", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)

    truth = None
    if args.synthetic:
        S, T, truth = make_synthetic(args.synthetic)
    else:
        S, T = load_mesh(args.source), load_mesh(args.target)
    S, T = outward(*S), outward(*T)
    print(
        f"source {len(S[0])} v / {len(S[1])} f   target {len(T[0])} v / {len(T[1])} f"
    )

    # control points: decimated source + decimated target
    c_np = np.concatenate(
        [decimate(*S, args.ctrl_reduction)[0], decimate(*T, args.ctrl_reduction)[0]]
    )
    c0 = torch.as_tensor(c_np, device=dev, dtype=dt)
    Sa, Ta = decimate(*S, args.attach_reduction), decimate(*T, args.attach_reduction)
    Sa, Ta = outward(*Sa), outward(*Ta)
    Xs = torch.as_tensor(Sa[0], device=dev, dtype=dt)
    tgt_cn = centers_normals(
        torch.as_tensor(Ta[0], device=dev, dtype=dt), torch.as_tensor(Ta[1], device=dev)
    )
    ell = float(np.median(cKDTree(Ta[0]).query(Sa[0])[0]))
    gdeg = DefKernel("gauss", args.sigma, keops=args.keops).calibrate(c0)
    print(
        f"{len(c0)} control points, attachment mesh {len(Sa[0])} v / {len(Ta[1])} target faces, "
        f"median source->target {ell:.2f} mm, Gaussian degree {gdeg:.1f}"
    )

    t = time.time()
    ev = Evaluator(S, T, args, dev, dt, truth)
    print(f"evaluator ready ({time.time() - t:.0f}s), grid {tuple(ev.g.dims)}")
    res_path = os.path.join(args.out, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    ident = lambda X: np.asarray(X, float)
    results["identity"], ex = ev.evaluate(ident, ident)
    save_slices(args.out, "identity", ex)

    table = arm_table(args)
    names = (
        []
        if args.arms == "none"
        else list(table)
        if args.arms == "all"
        else args.arms.split(",")
    )
    for name in names:
        spec = table[name]
        print(f"--- {name}: {spec}", flush=True)
        kern, traj, info = fit_arm(
            spec, c0, Xs, Sa[1], tgt_cn, args, dev, dt, gdeg, ell
        )
        push = lambda X: flow(kern, traj, ev._t(X)).double().cpu().numpy()
        pull = lambda X: (
            flow(kern, traj, ev._t(X), backward=True).double().cpu().numpy()
        )
        m, ex = ev.evaluate(push, pull)
        m.update({k: v for k, v in info.items() if k != "trace"})
        json.dump(
            info["trace"], open(os.path.join(args.out, f"trace_{name}.json"), "w")
        )
        save_slices(args.out, name, ex)
        if not args.no_meshes:
            save_mesh(
                ex["Vd"],
                S[1],
                os.path.join(args.out, f"deformed_{name}.vtk"),
                {
                    "fit_distance": ex["fit"],
                    "displacement": np.linalg.norm(ex["Vd"] - S[0], axis=1),
                    "convexity": ev.cvx,
                },
            )
        print(
            {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, flush=True
        )
        results = json.load(open(res_path)) if os.path.exists(res_path) else results
        results[name] = m
        json.dump(results, open(res_path, "w"), indent=1)

    # report everything collected so far in this --out directory (any process)
    order = ["identity"] + [n for n in table if n in results]
    results = {n: results[n] for n in order}
    json.dump(results, open(res_path, "w"), indent=1)
    print(write_report(results, os.path.join(args.out, "results.md")))
    extras = {
        n: dict(np.load(os.path.join(args.out, f"slice_{n}.npz")))
        for n in order
        if os.path.exists(os.path.join(args.out, f"slice_{n}.npz"))
    }
    make_figure(ev, extras, os.path.join(args.out, "slices.png"))


def save_slices(out, name, ex):
    np.savez_compressed(
        os.path.join(out, f"slice_{name}.npz"),
        slice_u=ex["slice_u"],
        slice_logjac=ex["slice_logjac"],
        slice_inD=ex["slice_inD"],
    )


if __name__ == "__main__":
    main()
