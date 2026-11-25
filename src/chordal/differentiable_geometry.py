"""
Differentiable surface geometry for the extended varifold attachment (deformetrica fork,
MultiObjectAttachment.extended_varifold_distance).

Torch versions of igl.per_vertex_normals, igl.massmatrix (Voronoi) and igl.cotmatrix, plus the
mean curvature H = <(L x) / a, n>, that keep the autograd graph. The attachment's gradient
must flow through normals and areas, not only point positions; without those terms the source
self-term becomes a plain repulsion that pushes sulcal fundi into the white matter (see
docs/extended-varifold-fixes.md, fix 1). Checked on oasis4-inner-decimated: normals, areas,
cotangent Laplacian and H match igl to round-off.

Normals and H are R^3 only. Areas, cotangents and tangent 2-vectors work in any dimension
(used in R^6 by the backward flow). Degenerate triangles are handled by clamping inside the
square root (see _double_area and docs/extended-varifold-fixes.md, fix 7).
"""
import torch


def vertex_normals(x, F):
    """Area-weighted unit vertex normals (igl.per_vertex_normals default weighting)."""
    v0, v1, v2 = x[F[:, 0]], x[F[:, 1]], x[F[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)            # 2 * face area * unit normal
    vn = torch.zeros_like(x)
    for k in range(3):
        vn = vn.index_add(0, F[:, k], fn)
    return vn / vn.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _double_area(a, b):
    """|a x b| = 2 * triangle area, in any dimension (Lagrange identity)."""
    aa, bb, ab = (a * a).sum(1), (b * b).sum(1), (a * b).sum(1)
    return (aa * bb - ab * ab).clamp_min(1e-30).sqrt()  # finite gradient at 0


def vertex_tangent_planes(x, F):
    """Area-weighted unit tangent 2-vectors, in any ambient dimension d.

    A face with edges a, b spans the 2-vector a ^ b, stored as its d(d-1)/2 Pluecker
    coordinates a_i b_j - a_j b_i (i < j): 3 in R^3, 15 in R^6. Its norm is |a x b|, twice
    the face area, and it plays the role of the face normal a x b. In R^3 the two are the same
    vector up to sign and ordering of components (Hodge duality), so every inner product, and
    hence the varifold kernel, is unchanged.

    The varifold term (n_x . n_y)^2 becomes (t_x . t_y)^2. For unit 2-vectors of two planes,
    t_x . t_y = cos(theta_1) cos(theta_2), where theta_1, theta_2 are the principal angles
    between the planes. In R^3 two planes always share a line, so theta_1 = 0 and this is the
    cosine of the dihedral angle, n_x . n_y. In R^6 both angles are free: the kernel is 1 only
    for identical planes and falls as either angle grows.

    The vertex value sums its faces' 2-vectors (weighted by area, as for vertex normals) and
    normalises. In R^3 any sum of 2-vectors is again a plane; in R^4 and up it need not be (it
    can mix two planes), so at strongly curved vertices t is a unit 2-vector that only
    approximates a tangent plane. The kernel stays a valid positive-definite kernel.
    """
    v0, v1, v2 = x[F[:, 0]], x[F[:, 1]], x[F[:, 2]]
    a, b = v1 - v0, v2 - v0
    i, j = torch.triu_indices(x.shape[1], x.shape[1], 1, device=x.device)
    ft = a[:, i] * b[:, j] - a[:, j] * b[:, i]           # 2 * face area * unit 2-vector
    vt = torch.zeros(len(x), len(i), dtype=x.dtype, device=x.device)
    for k in range(3):
        vt = vt.index_add(0, F[:, k], ft)
    return vt / vt.norm(dim=1, keepdim=True).clamp_min(1e-12)


def voronoi_areas(x, F):
    """Mixed Voronoi vertex areas, as igl.massmatrix(V, F) (MASSMATRIX_TYPE_DEFAULT = Voronoi):
    Voronoi region for non-obtuse triangles, area/2 to the obtuse corner and area/4 to the others."""
    v = [x[F[:, k]] for k in range(3)]
    e = [v[2] - v[1], v[0] - v[2], v[1] - v[0]]          # edge opposite corner k
    l2 = [(ek ** 2).sum(1) for ek in e]
    dbl = _double_area(e[2], -e[1])
    area = 0.5 * dbl
    cot = [((-e[(k + 2) % 3]) * e[(k + 1) % 3]).sum(1) / dbl.clamp_min(1e-30) for k in range(3)]
    # Voronoi share of corner k: (l2 of the two edges at k, weighted by cot of the opposite corners) / 8
    vor = [(l2[(k + 1) % 3] * cot[(k + 1) % 3] + l2[(k + 2) % 3] * cot[(k + 2) % 3]) / 8 for k in range(3)]
    obtuse = [cot[k] < 0 for k in range(3)]
    any_obt = obtuse[0] | obtuse[1] | obtuse[2]
    out = torch.zeros(len(x), dtype=x.dtype, device=x.device)
    for k in range(3):
        share = torch.where(any_obt, torch.where(obtuse[k], area / 2, area / 4), vor[k])
        out = out.index_add(0, F[:, k], share)
    return out


def cotan_laplacian_apply(x, F, y=None):
    """(L y)_i = sum_j 0.5 (cot a_ij + cot b_ij) (y_j - y_i), i.e. igl.cotmatrix(x, F) @ y,
    with cotangent weights computed from x (differentiable). y defaults to x."""
    y = x if y is None else y
    out = torch.zeros_like(y)
    v = [x[F[:, k]] for k in range(3)]
    for k in range(3):                                   # corner k, opposite edge (i, j)
        i, j = F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        a, b = v[(k + 1) % 3] - v[k], v[(k + 2) % 3] - v[k]
        cot = (a * b).sum(1) / torch.cross(a, b, dim=1).norm(dim=1).clamp_min(1e-30)
        w = (0.5 * cot)[:, None]
        out = out.index_add(0, i, w * (y[j] - y[i]))
        out = out.index_add(0, j, w * (y[i] - y[j]))
    return out


def normals_areas_curvature(x, F, curvature=True):
    """Returns (unit vertex normals, Voronoi areas, H) with H = <(L x)/a, n>, as in the fork."""
    n = vertex_normals(x, F)
    a = voronoi_areas(x, F)
    if not curvature:
        return n, a, None
    h = (cotan_laplacian_apply(x, F) / a[:, None] * n).sum(1)
    return n, a, h
