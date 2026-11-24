"""
Conformalized mean curvature flow (cMCF) whose motion is extended, step by step,
to arbitrary points in and around the surface with a local-linear kernel fit.

Each cMCF step (implicit solve, then the recentre/rescale of flow_3d) moves the
vertices by d = X_{k+1} - X_k. Folding the normalization into d matters: the implicit
step alone can shrink the surface by a large factor, while d after normalization is
only the change of shape. The surface moves along the straight path X_k + s d; each of
`substeps` increments d/substeps is extended from the surface where that increment
starts, by area-weighted local-linear regression, and added to the query points. The
step map for the query points is a composition of `substeps` near-identity maps.

Local-linear extension at a query point q:
    min_{A,b} sum_j w_j(q) |b + A (x_j - q) - d_j|^2,   v(q) = b
    w_j(q) = a_j exp(-|q - x_j|^2 / s(q)^2),  s(q)^2 = s0^2 + kappa^2 softdist(q)^2
a_j = vertex areas (mass-matrix diagonal), so the weights do not depend on sampling.
Reproduces affine displacement fields exactly; a small ridge on A falls back toward
the zeroth-order (Nadaraya-Watson) average where the local fit is ill-conditioned.
Weights are normalized in log space, so points far from the surface never underflow.
`n_correct` residual passes push the fit toward exact interpolation at the vertices.
"""
import logging
import math

import igl
import numpy as np
import scipy.sparse as sps
import torch

try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- sparse solve
def cho_factor(A):
    """Cholesky via CHOLMOD when available, else sparse LU (same .solve interface)."""
    try:
        from sksparse.cholmod import cho_factor
        return cho_factor(A)
    except ImportError:
        lu = sps.linalg.splu(sps.csc_matrix(A))
        return type("F", (), {"solve": staticmethod(lambda b: lu.solve(np.asarray(b)))})()


# ---------------------------------------------------------------- extension
class LocalLinearExtension:
    def __init__(self, kappa=0.5, tau=None, ridge=1e-6, n_correct=1, keops=False,
                 device="cpu", dtype=torch.float64, max_pairs=2e7):
        self.kappa, self.tau, self.ridge, self.n_correct = kappa, tau, ridge, n_correct
        self.keops, self.device, self.dtype, self.max_pairs = keops, device, dtype, max_pairs

    def _t(self, a):
        return torch.as_tensor(np.asarray(a), device=self.device, dtype=self.dtype)

    # --- moments: normalized weighted sums of [y, y y^T, d, y d^T], y = x_j - q
    def _moments_dense(self, Q, X, D, la, s0, tau):
        out = []
        B = max(1, int(self.max_pairs // len(X)))
        for i in range(0, len(Q), B):
            q = Q[i:i + B]
            y = X[None] - q[:, None]                                   # (b, n, 3)
            D2 = (y ** 2).sum(-1)
            lse = torch.logsumexp(-D2 / tau ** 2, 1)
            s2 = s0 ** 2 + self.kappa ** 2 * tau ** 2 * torch.nn.functional.softplus(-lse)
            w = torch.softmax(la[None] - D2 / s2[:, None], 1)             # (b, n)
            S1 = torch.einsum("bn,bnk->bk", w, y)
            S2 = torch.einsum("bn,bnk,bnl->bkl", w, y, y)
            T0 = w @ D
            T1 = torch.einsum("bn,bnk,nl->bkl", w, y, D)
            out.append((S1, S2, T0, T1, s2))
        return [torch.cat(z) for z in zip(*out)]

    def _moments_keops(self, Q, X, D, la, s0, tau):
        qi, xj = LazyTensor(Q[:, None, :]), LazyTensor(X[None, :, :])
        D2 = ((qi - xj) ** 2).sum(-1)
        lse = (-D2 / tau ** 2).logsumexp(dim=1)
        s2 = s0 ** 2 + self.kappa ** 2 * tau ** 2 * torch.nn.functional.softplus(-lse)
        logw = LazyTensor(la[None, :, None]) - D2 / LazyTensor(s2[:, None, :])
        L = logw.logsumexp(dim=1)
        w = (logw - LazyTensor(L[:, None, :])).exp()
        y = xj - qi
        dj = LazyTensor(D[None, :, :])
        M = (w * y.concat(y.tensorprod(y)).concat(dj).concat(y.tensorprod(dj))).sum(dim=1)
        n = len(Q)
        return (M[:, 0:3], M[:, 3:12].reshape(n, 3, 3), M[:, 12:15],
                M[:, 15:24].reshape(n, 3, 3), s2[:, 0])

    def _once(self, Q, X, D, la, s0, tau):
        f = self._moments_keops if self.keops else self._moments_dense
        S1, S2, T0, T1, s2 = f(Q, X, D, la, s0, tau)
        n = len(Q)
        G = torch.zeros(n, 4, 4, device=Q.device, dtype=Q.dtype)
        G[:, 0, 0] = 1.0
        G[:, 0, 1:] = S1
        G[:, 1:, 0] = S1
        G[:, 1:, 1:] = S2 + self.ridge * s2[:, None, None] * torch.eye(3, device=Q.device, dtype=Q.dtype)
        R = torch.cat([T0[:, None, :], T1], 1)                          # (n, 4, 3)
        return torch.linalg.solve(G, R)[:, 0, :]                        # b = v(q)

    def __call__(self, X, D, area, Q, s0):
        """Extend vertex displacements D (n,3) at vertices X with areas `area` to points Q."""
        X, D, Q = self._t(X), self._t(D), self._t(Q)
        la = torch.log(self._t(area).clamp_min(1e-30))
        tau = self.tau or s0 / 2
        V = self._once(Q, X, D, la, s0, tau)
        R = D
        for _ in range(self.n_correct):        # extend the residual left at the vertices
            R = R - self._once(X, X, R, la, s0, tau)
            V = V + self._once(Q, X, R, la, s0, tau)
        return V.cpu().numpy()


# ---------------------------------------------------------------- flow
def median_edge(V, F):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return float(np.median(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)))


def flow_3d_extended(V, F, Q, *, rate_0=0.05, growth=1.4, rate_max=500, max_iters=100,
                     stop_speed=1e-4, epsilon=1e-4, substeps=2, s0_factor=1.0,
                     s0_min_edges=1.5, ext=None, callback=None):
    """cMCF of (V, F) as in flow_3d, carrying the extra points Q along.
    Returns (V_final, Q_final, info). V, Q are in the input coordinates; like flow_3d,
    the surface is recentred to the origin and kept at its initial RMS radius."""
    ext = ext or LocalLinearExtension()
    cc = np.array(V, dtype=np.float64)
    qq = np.array(Q, dtype=np.float64)
    ff = np.asarray(F, dtype=np.int64)

    m = cc.mean(0, keepdims=True)
    cc -= m
    qq -= m
    cc_norm = np.sqrt(np.mean(np.square(cc)))
    s0_min = s0_min_edges * median_edge(cc, ff)

    ll = igl.cotmatrix(cc, ff)
    ii = epsilon * sps.eye(len(cc))
    rate, info = rate_0, dict(iters=0, s0=[], rate=[], speed=[])
    for it in range(max_iters):
        mm = igl.massmatrix(cc, ff)
        prev = cc.copy()
        new = cho_factor(mm + ii - rate * ll).solve((mm + ii) @ cc)
        # similarity normalization (as in flow_3d), folded into the step's displacement:
        # the extension reproduces similarities exactly, and the displacement it has to
        # extend is then only the change of shape, not the large implicit-step shrink.
        new -= new.mean(0, keepdims=True)
        new *= cc_norm / np.sqrt(np.mean(np.square(new)))
        d = new - prev
        area = mm.diagonal()
        s0 = float(np.clip(s0_factor * math.sqrt(rate), s0_min, 0.25 * cc_norm))
        for k in range(substeps):              # straight surface path, consistent sub-steps:
            Xs = prev + k / substeps * d       # each increment extended from the surface
            qq += ext(Xs, d / substeps, area, qq, s0)   # where that increment starts
        cc = new
        speed = np.sqrt(np.mean(np.square(cc - prev))) / rate
        info["s0"].append(s0); info["rate"].append(rate); info["speed"].append(speed)
        info["iters"] = it + 1
        logger.info(f"{it=} rate={rate:.3g} s0={s0:.3g} speed={speed:.3e}")
        if callback:
            callback(it, cc, qq)
        if speed < stop_speed:
            break
        rate = float(np.clip(rate * growth, 0, rate_max))
    return cc, qq, info
