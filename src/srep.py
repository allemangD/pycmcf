"""
Annular s-rep on the simplified chord surface (devel/allemang-2026.pdf, sections 3.1 and 8).

The forward flow takes the chord surface c = (u, v) in R^6 towards an ellipsoid. We fit that
ellipsoid, x = mu + axes @ d for unit d in R^3, with axes (6, 3). The fit maps every chord to
a direction d_i on the unit sphere. The s-rep coordinates are spherical coordinates of d:
theta1 is the polar angle from the longest axis, and theta2 the longitude.

The frame is fixed by convention, so that the same (theta1, theta2) means the same place in
every subject: theta1 = 0 is the end of the long axis towards `pole`, and theta2 = 0 is the
direction towards `meridian`, both given in the data frame. Only the long axis is taken from
the fit, so no fine pre-alignment is needed, only the two sign conventions. The other two
semi-axes are nearly equal by the time the surface is ellipsoidal (about 10% apart at the
default stop), so their order cannot orient anything. Longitude 90 degrees lies towards
pole x meridian. The defaults, L and S, are for the OASIS-4 data (LSA); data that is not
pre-aligned needs directions that do not depend on alignment, e.g. from extents.

The ring latitudes are a population choice (equal_area_latitudes). CMCF compresses the two
caps around the L/R poles about 3.5 times and stretches the belt around the equator, so equal
steps in theta1 put most samples where there is least cortex. The belt still holds real
structure (the callosal sulcus), so a fraction of the rings can be kept uniform as a floor.
The latitudes are symmetric about the equator, so with an even number of bands the middle
ring lies at 90 degrees and separates the hemispheres. The workflow is: forward flow every
case; take latitude_cdf of each and freeze equal_area_latitudes of them all (save_sampling);
then build each case's s-rep with that sampling. Cases added later reuse the saved sampling:
they do not change it, but stay commensurate with it. Every case shares one Cartesian
(theta1, theta2) grid, so sample (i, j) means the same place in every case.

The ellipsoid only provides the parameterisation. Each sample direction is located in the
sphere mesh (the triangulation at the d_i) as a face and barycentric weights: a radial
projection, in the ellipsoid's normalised coordinates, onto the triangulated chord surface.
flow() keeps the triangulation, so the same face and weights give the sample in every state.

surface_srep evaluates them on the original chords (state 0). This needs no backward flow
and is exact: the forward mesh already carries the correspondence. Its midpoint is a skeletal
point and its two halves are the spokes, and the chord has its true length.

For points inside the gray matter as well, the s-rep's surface points and the interior points
(depth_points at the last CMCF iterate) can be carried together through the R3 backward flow
(backflow(..., space="R3", carry=...)), as images of one diffeomorphism. With the 5 mm
deformation kernel this does not yet work: the kernel is wider than the cortex is thick, and
interior points drift towards the pial surface (docs/backward-flow-ablation.md, experiment
6). Shooting points of the chord surface in R6 fails where CMCF dilated small triangles
(experiment 5).
"""

import logging
from types import SimpleNamespace

import igl
import numpy as np

logger = logging.getLogger(__name__)


def fit_ellipsoid(c, *, max_iterations=1000, tolerance=1e-8):
    """Fits c (n, k) ~ mu + d @ axes.T with unit d. Returns mu (k,), axes (k, 3), d (n, 3).

    Alternates d_i = normalize(pinv(A) (c_i - mu)) with a linear least-squares fit of
    (mu, A), starting from PCA. axes = U S of the SVD A = U S V^T, and d is expressed in
    the same frame, so d[:, 0] runs along the longest axis.
    """
    mu = c.mean(0)
    _, sv, vt = np.linalg.svd(c - mu, full_matrices=False)
    a = vt[:3].T * (sv[:3] * np.sqrt(3 / len(c)))  # RMS * sqrt(3) for a uniform sphere
    ones = np.ones((len(c), 1))
    for _ in range(max_iterations):
        d = (c - mu) @ np.linalg.pinv(a).T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        coef, *_ = np.linalg.lstsq(np.hstack([ones, d]), c, rcond=None)
        change = np.abs(coef[1:].T - a).max()
        mu, a = coef[0], coef[1:].T
        if change < tolerance:
            break
    else:
        logger.warning(f"ellipsoid fit did not converge (last change {change:.2e})")
    d = (c - mu) @ np.linalg.pinv(a).T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return mu, u * s, d @ vt.T


def sampling(latitudes, *, n_phi=64, pole=(1.0, 0.0, 0.0), meridian=(0.0, 1.0, 0.0)):
    """The s-rep sampling convention: ring latitudes theta1 (interior rings, radians,
    increasing in (0, pi)), n_phi points per ring, and the pole and meridian directions in the
    data frame (defaults L and S in LSA)."""
    latitudes = np.asarray(latitudes, float)
    if not (np.all(np.diff(latitudes) > 0) and 0 < latitudes[0] and latitudes[-1] < np.pi):
        raise ValueError("latitudes must increase strictly inside (0, pi)")
    return dict(
        latitudes=latitudes, n_phi=int(n_phi),
        pole=np.asarray(pole, float), meridian=np.asarray(meridian, float),
    )


def uniform_latitudes(n_theta):
    """n_theta - 1 rings at theta1 = pi i / n_theta."""
    return np.pi * np.arange(1, n_theta) / n_theta


def save_sampling(path, sampling):
    np.savez(path, **sampling)


def load_sampling(path):
    z = np.load(path)
    return sampling(z["latitudes"], n_phi=z["n_phi"], pole=z["pole"], meridian=z["meridian"])


def latitude_cdf(c, F, c0, *, grid_size=721):
    """Fraction of the original cortex (mid-surface area of c0) within theta1 of the pole, on
    a uniform theta1 grid over [0, pi], for the ellipsoid fitted to c.

    Symmetrised about the equator, so that with an even number of bands the middle ring lies
    at 90 degrees and separates the hemispheres (theta1 < 90 is the pole's side). The
    hemispheres differ, but little: in oasis4 the CDF is within 0.0035 of its symmetrised
    form and no ring moves more than 0.5 degrees. Being symmetric, it does not depend on the
    sign of the pole."""
    _, _, d = fit_ellipsoid(c)
    m = (c0[:, :3] + c0[:, 3:]) / 2
    area = np.linalg.norm(np.cross(m[F[:, 1]] - m[F[:, 0]], m[F[:, 2]] - m[F[:, 0]]), axis=1)
    dc = d[F].sum(1)
    t = np.arccos(np.clip(dc[:, 0] / np.linalg.norm(dc, axis=1), -1, 1))
    order = np.argsort(t)
    grid = np.linspace(0, np.pi, grid_size)
    cdf = np.interp(grid, t[order], np.cumsum(area[order]) / area.sum(), left=0, right=1)
    return (cdf + 1 - cdf[::-1]) / 2


def equal_area_latitudes(cdfs, n_theta, *, uniform_fraction=0.0):
    """n_theta - 1 ring latitudes that split the mean of the cases' latitude_cdf into n_theta
    bands of equal cortical area. The mean is taken over cases, so each case counts equally.
    n_theta must be even: the middle ring then lies at 90 degrees, between the hemispheres.

    uniform_fraction: a floor on the ring density. This fraction of the rings is spaced
    evenly in theta1 and the rest by cortical area (the two CDFs are mixed). Pure quantiles
    leave the belt around the equator, where CMCF stretches the midline, with few rings.
    """
    if n_theta % 2:
        raise ValueError(f"n_theta must be even, for a ring at 90 degrees (got {n_theta})")
    grid = np.linspace(0, np.pi, len(cdfs[0]))
    cdf = (1 - uniform_fraction) * np.mean(cdfs, axis=0) + uniform_fraction * grid / np.pi
    return np.interp(np.arange(1, n_theta) / n_theta, cdf, grid)


def sphere_grid(latitudes, n_phi):
    """Latitude-longitude grid on the unit sphere, pole on the first axis: the two poles and
    a ring of n_phi points at each latitude. Returns directions (m, 3), (theta1, theta2)
    (m, 2) and triangles over the grid."""
    t1, t2 = np.meshgrid(latitudes, 2 * np.pi * np.arange(n_phi) / n_phi, indexing="ij")
    t1 = np.concatenate([[0.0], t1.ravel(), [np.pi]])
    t2 = np.concatenate([[0.0], t2.ravel(), [0.0]])
    d = np.stack([np.cos(t1), np.sin(t1) * np.cos(t2), np.sin(t1) * np.sin(t2)], 1)

    ring = 1 + np.arange(len(latitudes) * n_phi).reshape(len(latitudes), n_phi)
    nxt = np.roll(ring, -1, axis=1)
    south = len(d) - 1
    faces = [np.stack([np.zeros(n_phi, int), ring[0], nxt[0]], 1)]
    for i in range(len(latitudes) - 1):
        faces += [
            np.stack([ring[i], ring[i + 1], nxt[i + 1]], 1),
            np.stack([ring[i], nxt[i + 1], nxt[i]], 1),
        ]
    faces.append(np.stack([np.full(n_phi, south), nxt[-1], ring[-1]], 1))
    return d, np.stack([t1, t2], 1), np.concatenate(faces)


def fit_srep(c, F, sampling):
    """Fits the ellipsoid to chords c (n, 6) on faces F and places the (theta1, theta2) grid
    of `sampling` (see sampling()) on the chord surface.

    Returns a namespace with points (m, 6) on the chord surface, directions (m, 3) (unit, in
    the frame of axes), theta (m, 2), grid_faces, face and bary (each sample's location in
    F), mu and axes (columns: the long axis towards pole, then the longitude 0 and 90 degree
    directions), residual (|c - ellipsoid| per chord, mm) and offset (|ellipsoid - surface|
    per sample, mm).
    """
    mu, axes, d = fit_ellipsoid(c)
    residual = np.linalg.norm(c - mu - d @ axes.T, axis=1)

    # Orient: d-space rotation R with R[:, 0] the long axis towards pole and R[:, 1] the
    # equatorial direction whose skeletal (midpoint) displacement best matches meridian.
    skeleton = (axes[:3] + axes[3:]) / 2
    long = np.array([1.0, 0, 0]) * np.sign(skeleton[:, 0] @ sampling["pole"])
    up = sampling["meridian"]
    up = up - (up @ skeleton[:, 0]) / (skeleton[:, 0] @ skeleton[:, 0]) * skeleton[:, 0]
    e = np.linalg.lstsq(skeleton[:, 1:], up, rcond=None)[0]
    e = np.array([0.0, *e]) / np.linalg.norm(e)
    R = np.stack([long, e, np.cross(long, e)], 1)
    axes, d = axes @ R, d @ R

    # The sphere mesh must be an embedding for the sample location to be unique: count the
    # faces against the majority orientation.
    orientation = np.einsum("ij,ij->i", d[F[:, 0]], np.cross(d[F[:, 1]], d[F[:, 2]]))
    flipped = int(min(np.sum(orientation < 0), np.sum(orientation > 0)))
    if flipped:
        logger.warning(f"{flipped} faces fold over on the fitted sphere")

    directions, theta, grid_faces = sphere_grid(sampling["latitudes"], sampling["n_phi"])
    _, face, closest = igl.point_mesh_squared_distance(directions, d, F)
    bary = igl.barycentric_coordinates(
        closest, *(np.ascontiguousarray(d[F[face, k]]) for k in range(3))
    )
    points = interpolate(c, F, face, bary)
    offset = np.linalg.norm(mu + directions @ axes.T - points, axis=1)
    logger.info(
        f"ellipsoid semi-axes {np.linalg.norm(axes, axis=0).round(2)}  "
        f"residual p50 {np.median(residual):.3f} p95 {np.percentile(residual, 95):.3f} "
        f"max {residual.max():.3f}  folded faces {flipped}  samples {len(directions)}"
    )
    return SimpleNamespace(
        points=points, directions=directions, theta=theta, grid_faces=grid_faces,
        face=face, bary=bary, mu=mu, axes=axes, residual=residual, offset=offset,
        flipped=flipped,
    )


def surface_srep(states, F, sampling):
    """The s-rep on the original surfaces, from the forward-flow states (K+2, n, 6) of
    flow.flow(): fit at the last CMCF iterate, evaluate at state 0. Returns (fit_srep's
    namespace, chords (m, 6) at state 0)."""
    srep = fit_srep(states[-2], F, sampling)
    return srep, interpolate(states[0], F, srep.face, srep.bary)


def depth_points(chords, depths):
    """Points (1 - tau) u + tau v (len(depths), m, 3) along chords (m, 6), tau = 0 at white."""
    tau = np.asarray(depths, float)[:, None, None]
    return (1 - tau) * chords[None, :, :3] + tau * chords[None, :, 3:]


def interpolate(c, F, face, bary):
    """Values at the samples (face, bary) of the per-vertex field c on faces F."""
    return np.einsum("mk,mkj->mj", bary, c[F[face]])
