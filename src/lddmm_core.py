"""
Core pieces for the kernel/attachment ablation: mesh IO, synthetic meshes,
deformation kernels, geodesic shooting, varifold attachments.

Deformation kernels (all symmetric PD, so H = 1/2 a^T K(c) a is a valid metric):
  gauss    K = exp(-|x-y|^2 / s^2)                         (Deformetrica convention)
  norm     K = dbar^2 * G(x,y) / (d(x) d(y)),  G Gaussian, d(x) = sum_j G(x, c_j)
  normvar  as norm, with G the Paciorek-Schervish kernel whose width grows with
           distance from the control points: s(x)^2 = s^2 + kappa^2 softdist^2(x)
  +affine  adds lam * (1 + (x-o).(y-o)/L^2) I  (global affine velocities)
dbar = median d(c0) is a constant, so normalized kernels are ~Gaussian inside the
band and their energies are on the same scale as the baseline.
All normalization is done in log space; the far field never underflows.
"""

import math

import numpy as np
import torch
import vtk
import vtk.util.numpy_support as ns

try:
    from pykeops.torch import LazyTensor
except ImportError:  # dense torch fallback only
    LazyTensor = None


# ============================================================== mesh IO
def to_polydata(V, F):
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(V, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    F = np.ascontiguousarray(F, dtype=np.int64)
    offsets = np.arange(0, 3 * len(F) + 1, 3, dtype=np.int64)
    cells.SetData(
        ns.numpy_to_vtkIdTypeArray(offsets, deep=True),
        ns.numpy_to_vtkIdTypeArray(F.ravel(), deep=True),
    )
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    return pd


def from_polydata(pd):
    V = np.asarray(pd.points.data, dtype=np.float64)
    F = np.asarray(pd.polys.connectivity_array).reshape(-1, 3).astype(np.int64)
    return V, F


def load_mesh(path):
    r = vtk.vtkPolyDataReader()
    r.SetFileName(path)
    r = vtk.vtkTriangleFilter(input_connection=r.output_port)
    r.Update()
    return from_polydata(r.output)


def decimate(V, F, reduction):
    if reduction <= 0:
        return V.copy(), F.copy()
    d = vtk.vtkQuadricDecimation()
    d.SetInputData(to_polydata(V, F))
    d.SetTargetReduction(reduction)
    d.Update()
    return from_polydata(d.GetOutput())


def save_mesh(V, F, path, point_arrays=None):
    pd = to_polydata(V, F)
    for name, arr in (point_arrays or {}).items():
        a = ns.numpy_to_vtk(np.ascontiguousarray(arr, dtype=np.float32), deep=True)
        a.SetName(name)
        pd.GetPointData().AddArray(a)
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(pd)
    w.SetFileTypeToBinary()
    w.Write()


def signed_volume(V, F):
    return (
        np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0
    )


def vertex_normals(V, F, outward=True):
    """Area-weighted unit vertex normals; flipped to point outward if requested."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    if outward and signed_volume(V, F) < 0:
        vn = -vn
    return vn


# ============================================================== synthetic meshes
def icosphere(level):
    src = vtk.vtkPlatonicSolidSource()
    src.SetSolidTypeToIcosahedron()
    sub = vtk.vtkLoopSubdivisionFilter()
    sub.SetInputConnection(src.GetOutputPort())
    sub.SetNumberOfSubdivisions(level)
    sub.Update()
    V, F = from_polydata(sub.GetOutput())
    return V / np.linalg.norm(V, axis=1, keepdims=True), F


def rotation(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def make_synthetic(scenario="thicken", level=4, R=30.0, fold=5.0, T0=2.5, seed=0):
    """Folded 'white' sphere and a 'pial' surface = white offset along its normals by a
    thickness T(u) (fatter gyri, narrower sulci), re-meshed by decimation so the two
    meshes differ in vertex count. scenario='rigid' also applies a known rigid motion
    to the pial surface; that motion is the interior ground truth."""
    rng = np.random.default_rng(seed)
    U, F = icosphere(level)
    dirs = rng.normal(size=(7, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ph = rng.uniform(0, 2 * np.pi, 7)
    f = np.cos(6.0 * U @ dirs.T + ph).sum(1) / math.sqrt(7)
    f = np.tanh(1.2 * f)  # flatter crowns, sharper creases
    Vw = U * (R + fold * f)[:, None]
    n = vertex_normals(Vw, F)
    Vp = Vw + (T0 * (1 + 0.3 * f))[:, None] * n
    # sanity: offset must not fold triangles
    fw = np.cross(Vw[F[:, 1]] - Vw[F[:, 0]], Vw[F[:, 2]] - Vw[F[:, 0]])
    fp = np.cross(Vp[F[:, 1]] - Vp[F[:, 0]], Vp[F[:, 2]] - Vp[F[:, 0]])
    assert (np.einsum("ij,ij->i", fw, fp) > 0).all(), "offset folded; reduce T0/fold"
    Vp, Fp = decimate(Vp, F, 0.2)
    truth = None
    if scenario == "rigid":
        Rm, t = rotation([1, 1, 0.3], 20.0), np.array([5.0, -3.0, 2.0])
        Vp = Vp @ Rm.T + t
        truth = dict(R=Rm, t=t)
    return (Vw, F), (Vp, Fp), truth


# ============================================================== kernel helpers
def _i(t, keops):
    return LazyTensor(t[:, None, :]) if keops else t[:, None, :]


def _j(t, keops):
    return LazyTensor(t[None, :, :]) if keops else t[None, :, :]


def _sqdist(a, b, keops):
    return ((a - b) ** 2).sum(-1) if keops else ((a - b) ** 2).sum(-1, keepdim=True)


def _dot(a, b, keops):
    return (a * b).sum(-1) if keops else (a * b).sum(-1, keepdim=True)


def _lse(t, keops):
    return t.logsumexp(dim=1) if keops else torch.logsumexp(t, dim=1)


class DefKernel:
    def __init__(self, kind, sigma, kappa=0.0, tau=None, affine_gain=0.0, keops=False):
        assert kind in ("gauss", "norm")
        self.kind, self.sigma, self.kappa, self.tau = kind, sigma, kappa, tau
        self.affine_gain, self.keops = affine_gain, keops
        self.logdbar, self.lam, self.o, self.L2 = 0.0, 0.0, None, 1.0

    # ---- widths / log kernels (normalized family)
    def log_width(self, x, c):
        if self.kappa == 0:
            return None
        D = _sqdist(_i(x, self.keops), _j(c, self.keops), self.keops)
        lse = _lse(-D / self.tau**2, self.keops)
        softdist2 = self.tau**2 * torch.nn.functional.softplus(-lse)
        return 0.5 * torch.log(self.sigma**2 + self.kappa**2 * softdist2)

    def log_kernel(self, x, lsx, c, lsc):
        k = self.keops
        D = _sqdist(_i(x, k), _j(c, k), k)
        if self.kappa == 0:
            return -D / self.sigma**2
        li, lj = _i(lsx, k), _j(lsc, k)
        ssum = (2 * li).exp() + (2 * lj).exp()
        return 1.5 * (math.log(2.0) + li + lj - ssum.log()) - 2 * D / ssum

    def prep(self, c):
        if self.kind == "gauss":
            return {}
        lsc = self.log_width(c, c)
        ldc = _lse(self.log_kernel(c, lsc, c, lsc), self.keops)
        return dict(lsc=lsc, ldc=ldc)

    def velocity(self, x, c, a, cache, x_is_c=False):
        k = self.keops
        if self.kind == "gauss":
            D = _sqdist(_i(x, k), _j(c, k), k)
            v = ((-D / self.sigma**2).exp() * _j(a, k)).sum(dim=1)
        else:
            lsc, ldc = cache["lsc"], cache["ldc"]
            if x_is_c:
                lsx, ldx = lsc, ldc
            else:
                lsx = self.log_width(x, c)
                ldx = _lse(self.log_kernel(x, lsx, c, lsc), k)
            logK = (
                self.log_kernel(x, lsx, c, lsc)
                + 2 * self.logdbar
                - _i(ldx, k)
                - _j(ldc, k)
            )
            v = (logK.exp() * _j(a, k)).sum(dim=1)
        if self.lam > 0:
            v = v + self.lam * (
                a.sum(0) + (x - self.o) @ ((c - self.o).T @ a) / self.L2
            )
        return v

    def calibrate(self, c0):
        """Fix the constants that depend on the initial control points."""
        from scipy.spatial import cKDTree

        with torch.no_grad():
            if self.tau is None:  # soft-distance scale ~ control-point spacing
                cn = c0.detach().cpu().numpy()
                self.tau = float(np.median(cKDTree(cn).query(cn, k=2)[0][:, 1]))
            gdeg = self._gauss_degree(c0)
            if self.kind == "norm":
                self.logdbar = float(self.prep(c0)["ldc"].median())
            self.o = c0.mean(0)
            self.L2 = float(((c0 - self.o) ** 2).sum(1).mean())
            self.lam = self.affine_gain * gdeg / len(c0)
            return gdeg

    def _gauss_degree(self, c0):
        k = self.keops
        D = _sqdist(_i(c0, k), _j(c0, k), k)
        return float(torch.exp(_lse(-D / self.sigma**2, k)).median())


# ============================================================== shooting
def _rhs(kern, c, a, create_graph):
    """dc/dt = dH/da = K(c) a,  da/dt = -dH/dc. Returns (vc, -dHdc, cache, H)."""
    if not create_graph:
        with torch.enable_grad():
            cc = c.detach().requires_grad_(True)
            cache = kern.prep(cc)
            vc = kern.velocity(cc, cc, a.detach(), cache, x_is_c=True)
            H = 0.5 * (a.detach() * vc).sum()
            (gc,) = torch.autograd.grad(H, cc)
        cache = {k: (v.detach() if v is not None else None) for k, v in cache.items()}
        return vc.detach(), -gc, cache, H.detach()
    cache = kern.prep(c)
    vc = kern.velocity(c, c, a, cache, x_is_c=True)
    H = 0.5 * (a * vc).sum()
    (gc,) = torch.autograd.grad(H, c, create_graph=True)
    return vc, -gc, cache, H


def shoot(kern, c0, a0, x0, n_steps=10, create_graph=True):
    """RK2 (midpoint) geodesic shooting carrying points x. Returns x1, H0, trajectory."""
    dt = 1.0 / n_steps
    c = (
        c0
        if (c0.requires_grad or not create_graph)
        else c0.detach().requires_grad_(True)
    )
    a, x, H0 = a0, x0, None
    traj = dict(ends=[(c.detach(), a.detach())], mids=[])
    for _ in range(n_steps):
        dc, da, cache, H = _rhs(kern, c, a, create_graph)
        H0 = H if H0 is None else H0
        xm = x + 0.5 * dt * kern.velocity(x, c, a, cache) if x is not None else None
        cm, am = c + 0.5 * dt * dc, a + 0.5 * dt * da
        dc, da, cache_m, _ = _rhs(kern, cm, am, create_graph)
        if x is not None:
            x = x + dt * kern.velocity(xm, cm, am, cache_m)
        c, a = c + dt * dc, a + dt * da
        traj["mids"].append((cm.detach(), am.detach()))
        traj["ends"].append((c.detach(), a.detach()))
    return x, H0, traj


@torch.no_grad()
def flow(kern, traj, X, backward=False, chunk=None):
    """Push points forward (or pull back) through a stored shooting trajectory."""
    n = len(traj["mids"])
    dt = 1.0 / n
    N = len(traj["ends"][0][0])
    chunk = chunk or (len(X) if kern.keops else max(256, int(4e6 // N)))
    caches = {}

    def vel(Y, state):
        key = id(state[0])
        if key not in caches:
            caches[key] = kern.prep(state[0])
        return torch.cat(
            [
                kern.velocity(Y[s : s + chunk], state[0], state[1], caches[key])
                for s in range(0, len(Y), chunk)
            ]
        )

    Y = X.clone()
    steps = range(n - 1, -1, -1) if backward else range(n)
    for k in steps:
        if backward:
            Ym = Y - 0.5 * dt * vel(Y, traj["ends"][k + 1])
            Y = Y - dt * vel(Ym, traj["mids"][k])
        else:
            Ym = Y + 0.5 * dt * vel(Y, traj["ends"][k])
            Y = Y + dt * vel(Ym, traj["mids"][k])
    return Y


# ============================================================== attachments
def centers_normals(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    N = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=1)
    area = N.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (v0 + v1 + v2) / 3, N / area, area


class Varifold:
    """Squared varifold distance to a fixed target mesh (centers, unit normals, areas).
    oriented=False: Deformetrica's varifold, exp(-|x-y|^2/s^2) <n,m>^2, area weighted.
    oriented=True : exp(-|x-y|^2/s^2) exp(2(<n,m>-1)/sig_n^2)  (Kaltenmark et al. 2017).
    Several widths in `sigmas` are summed (multi-scale attachment)."""

    def __init__(self, Y, sigmas, oriented, sig_n=0.5, keops=False):
        self.Y, self.sigmas, self.oriented, self.sig_n, self.keops = (
            Y,
            sigmas,
            oriented,
            sig_n,
            keops,
        )
        with torch.no_grad():
            self.yy = self._pair(Y, Y)

    def _pair(self, P, Q):
        k_ = self.keops
        (p, pn, pa), (q, qn, qa) = P, Q
        D = _sqdist(_i(p, k_), _j(q, k_), k_)
        A = _dot(_i(pn, k_), _j(qn, k_), k_)
        tot = 0.0
        for s in self.sigmas:
            k = (
                (-D / s**2 + 2 * (A - 1) / self.sig_n**2).exp()
                if self.oriented
                else (-D / s**2).exp() * A**2
            )
            tot = tot + (pa * (k * _j(qa, k_)).sum(dim=1)).sum()
        return tot

    def __call__(self, X):
        return self._pair(X, X) - 2 * self._pair(X, self.Y) + self.yy
