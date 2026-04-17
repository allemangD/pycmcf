import collections
import hashlib
import inspect
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor

import scipy as sp
from tqdm import tqdm
import math
import types
from pathlib import Path

import igl
import numpy as np
import vtk
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

np.set_printoptions(suppress=True)


def normalize(v):
    center = np.mean(v, axis=0, keepdims=True)
    v -= center

    scale = np.sqrt(np.mean(np.square(v), keepdims=True))
    v /= scale

    def denormalize(v):
        return np.asarray(v) * scale + center

    return denormalize


class cached_access:
    def __init__(self, func: types.FunctionType, force: bool):
        self.func = func
        self.force = force

        source, lineno = inspect.getsourcelines(self.func)
        self.code = hashlib.md5("".join(source).encode()).hexdigest()
        self.name = self.func.__name__
        self.link = Path(".cache", f"{self.name}.vtk")
        self.path = Path(".cache", f".{self.code}.vtk")
        self.path.parent.mkdir(exist_ok=True)

        self.deps = list[cached_access]()
        locs = inspect.currentframe().f_back.f_back.f_locals  # the decorator context.
        for ref in self.func.__code__.co_names:
            if isinstance(acc := locs.get(ref), cached_access):
                self.deps.append(acc)

    def __call__(self) -> vtk.vtkPolyData:
        mtime = self.mtime

        if math.isinf(mtime) or (
            self.deps and mtime < max(dep.mtime for dep in self.deps)
        ):
            print(f"invoke {self.name}!")
            output = self.func()
            assert isinstance(output, vtk.vtkPolyData)
            writer = vtk.vtkPolyDataWriter(file_name=str(self.path))
            writer.input_data = output
            writer.Update()
        else:
            print(f"cached {self.name} {self.path.name}!")
            reader = vtk.vtkPolyDataReader(file_name=str(self.path))
            reader.Update()
            output = reader.output

        self.link.unlink(missing_ok=True)
        self.link.hardlink_to(self.path)

        return output

    @property
    def mtime(self):
        return self.path.stat().st_mtime if self.path.exists() else float("inf")


class cached:
    def __init__(self, force=False):
        self.force = force

    def __call__(self, func: types.FunctionType) -> cached_access:
        return cached_access(func, self.force)


@cached()
def original():
    pipe = vtk.vtkAppendPolyData()

    for path in [
        "devel/OASIS4-00010_0102_00-00_BRAIN-T1-SYNTH-3D-SYNTH-PRE_hdrfix_n4_reg_haca3_lesionfilled_slant_wlesions_surface-inner.vtk",
        "devel/OASIS4-00010_0102_00-00_BRAIN-T1-SYNTH-3D-SYNTH-PRE_hdrfix_n4_reg_haca3_lesionfilled_slant_wlesions_surface-outer.vtk",
    ]:
        read = vtk.vtkPolyDataReader(file_name=path)
        pipe.AddInputConnection(read.output_port)

    pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
    pipe = vtk.vtkCleanPolyData(input_connection=pipe.output_port)
    pipe.convert_lines_to_points = False
    pipe.convert_polys_to_lines = False
    pipe.convert_strips_to_polys = False
    pipe.Update()

    output = pipe.output

    V = np.asarray(output.points)
    V -= np.mean(V, axis=0, keepdims=True)
    V /= np.sqrt(np.mean(np.square(V)))
    output.points = V

    return output


@cached()
def pre_smooth():
    """Slight presmoothing."""

    data = original()

    rate = 2e-3

    V = np.asarray(data.points, copy=True)
    F = np.reshape(data.polys.connectivity_array, (-1, 3), copy=True)

    L = igl.cotmatrix(V, F)

    for _ in range(1):
        M = igl.massmatrix(V, F)

        V = spsolve(M - rate * L, M @ V)
        V -= V.mean(axis=0, keepdims=True)
        V /= np.sqrt(np.mean(np.square(V)))

    data.points = V

    return data


@cached()
def link():
    data = pre_smooth()

    V = np.asarray(data.points)
    F = np.reshape(data.polys.connectivity_array, (-1, 3))

    A = igl.adjacency_matrix(F)
    Cn, C, Ck = igl.connected_components(A)
    data.point_data["C"] = C
    # component 1 is the inner; component 0 is the outer.

    N = igl.per_vertex_normals(V, F)
    N[C == 1] *= -1  # invert the inner component's normals.

    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F)
    Minv = M.power(-1)  # diagonal matrix.

    HN = -Minv @ L @ V

    FACTOR = 0.8

    # TODO run in batches. should be possible to call query_tree and query_

    for c in range(Cn):
        (idxs,) = np.nonzero(C == c)
        tree = cKDTree(V[idxs])

        (idxs_neg,) = np.nonzero(C != c)
        tree_neg = cKDTree(V[idxs_neg])

        # mask = np.zeros((len(idxs), len(idxs_neg)), dtype=bool)

        dists, _ = tree_neg.query(tree.data, workers=-1)
        rmaxs = np.multiply(dists, 1.5)

        jdxss = tree_neg.query_ball_point(
            tree.data, r=rmaxs, workers=-1, return_sorted=True
        )

        global get_links

        def get_links(idx, jdxs):
            jdxs = np.take(idxs_neg, jdxs)
            vecs = V[jdxs] - V[idx]
            lens = np.linalg.norm(vecs, axis=1)

            jdxs = np.compress(
                np.einsum("ic, c -> i", vecs, N[idx]) < (-lens * 0.2), jdxs
            )
            vecs = V[jdxs] - V[idx]
            lens = np.linalg.norm(vecs, axis=1)

            jdxs = np.compress(
                np.einsum("ic, ic -> i", vecs, N[jdxs]) > (lens * 0.2), jdxs
            )
            vecs = V[jdxs] - V[idx]
            lens = np.linalg.norm(vecs, axis=1)

            dist_cost = lens
            norm_cost = np.einsum("ic, c -> i", N[jdxs], N[idx])  # ideal -1

            arg = np.argsort(
                dist_cost + norm_cost,
            )

            jdxs = jdxs[arg[:2]]

            # N[jdxs] @ N[idx]
            # V[jdxs] - V[idx]

            return [(idx, jdx) for jdx in jdxs]

        with ProcessPoolExecutor() as ex:
            for links in tqdm(
                ex.map(get_links, idxs, jdxss, chunksize=512),
                total=len(idxs),
            ):
                for link in links:
                    data.lines.InsertNextCell(2, link)

    data.SetPolys(None)

        # lens = []
        #
        # for idx, jdxs in zip(tqdm(idxs), jdxss):
        #     jdxs = np.take(idxs_neg, jdxs)
        #     jdxs = np.compress(
        #         np.einsum("ic, c -> i", V[jdxs] - V[idx], N[idx]) < 0, jdxs
        #     )
        #     jdxs = np.compress(
        #         np.einsum("ic, ic -> i", V[jdxs] - V[idx], N[jdxs]) > 0, jdxs
        #     )
        #     if not len(jdxs):
        #         continue
        #
        #     lens.append(len(jdxs))
        #     for jdx in jdxs:
        #         data.lines.InsertNextCell(2, [idx, jdx])

        # for idx, rmax in zip(tqdm(idxs), rmaxs):
        #     ball = tree_neg.query_ball_point()

        # ress = tree_neg.query_ball_point(
        #     tree.data, r=np.multiply(d, 1.5), return_sorted=True, workers=-1
        # )
        # print(len(ress))
        # print(np.quantile([len(res) for res in ress], [0, 0.5, 1.0]))

        # for idx in tqdm(idxs):
        #     pass
        # D = (V[idx] - V[idxs]) @ N[idx] < 0

        # out = np.einsum('ic, jc -> ij', V[idxs], V[idxs_neg])
        # print(out.shape)

        # V[idxs]

        # tree.query_ball_tree()
        # tree.query()
        # tree.query(tree_neg.data)

        # print(c)
        # d, ress = tree_neg.query(tree.data, k=3, workers=-1)
        # ress = np.take(idxs_neg, ress)
        #
        # ns = N[idxs]
        # hs = HN[idxs]
        # ps = V[idxs]
        # qs = V[ress]
        #
        # for i, js in zip(idxs, ress):
        #     for j in js:
        #         data.lines.InsertNextCell(2, [i, j])

        # print(np.einsum("ij,ij->i", hs, ps - qs)[:10])

        # print(d.min(), d.max())
        # ress = tree_neg.query_ball_point(tree.data, r=np.multiply(d, 1.25), workers=-1)
        # tree_neg.query_ball_tree()
        # ress = tree.query_ball_tree(tree_neg, r=d.max(), )
        # print(len(ress), len(d))
        # print(sorted([len(res) for res in ress])[:10])

    # data.GetLines().GetData().SetScalars(C)
    # data.GetLines().

    # data.SetPolys(None)
    # data.cell_data['F'] = C

    # for i in tqdm(range(0, len(V), 8)):
    #     N[i]
    #     V[i]
    #     C[i]

    # E = []
    #
    # for i in tqdm(range(0, len(V), 8)):
    #     v = V[i]
    #     n = N[i]
    #     c = C[i]
    #     hn = HN[i]
    #
    #     d, _ = C_tree_neg[c].query(v)
    #     opts = C_tree_neg[c].query_ball_point(v, d * 1.5)
    #     iopts = C_idxs_neg[c][opts]
    #
    #     vs = V[iopts]
    #     ns = N[iopts]
    #     hns = HN[iopts]
    #
    #     ds = vs - v
    #
    #     ucost = np.einsum("c,ic->i", n, ds)  # want this negative
    #     vcost = np.einsum("ic,ic->i", ns, ds)  # want this positive
    #
    #     dcost = np.einsum("ic,ic->i", ds, ds)  # want this small.
    #
    #     cost = np.log(dcost) * (ucost - vcost)
    #
    #     opt = np.argmin(cost)
    #     j = C_idxs_neg[c][opts[opt]]
    #
    #     E.append([i, j])
    #
    # E = np.array(E)
    #
    # L = igl.cotmatrix(V, F)
    # M = igl.massmatrix(V, F)
    #
    # u, v = np.unstack(V[E], axis=1)
    # D = v - u
    # D /= np.linalg.norm(D, axis=1, keepdims=True)
    # D = sp.sparse.linalg.spsolve(M - 1e-2 * L, M @ D)
    #
    # E = []
    # for i in tqdm(range(0, len(V), 8)):
    #     v = V[i]
    #     c = C[i]
    #
    #     k = D[i]
    #
    #     d, _ = C_tree_neg[c].query(v)
    #     opts = C_tree_neg[c].query_ball_point(v, d * 1.5)
    #     iopts = C_idxs_neg[c][opts]
    #
    #     vs = V[iopts]
    #     ds = vs - v
    #
    #     dcost = np.einsum("ic,ic->i", ds, ds)  # want this small.
    #     ncost = np.einsum("c,ic->i", k, ds)  # want this close to 1.
    #
    #     # np.log(1 - dcost) + ncost
    #     cost = dcost * ncost
    #     opt = np.argmin(cost)
    #     j = C_idxs_neg[c][opts[opt]]
    #
    #     E.append([i, j])
    #
    #     data.lines.InsertNextCell(2, [i, j])

    return data


@cached()
def flow():
    data = link()

    # V = np.asarray(data.points)
    # F = np.reshape(data.polys.connectivity_array, (-1, 3))
    # E = np.reshape(data.lines.connectivity_array, (-1, 2))
    #
    # u, v = np.stack(np.unstack(V[E], axis=1), axis=-1)
    # D = u - v
    # D /= np.linalg.norm(D, axis=1)
    #
    # L = igl.cotmatrix(D, F)
    # M = igl.massmatrix(V, F)
    #
    # D = sp.sparse.linalg.spsolve(M - 1e-2 * L, M @ D)
    #
    # data.SetLines(vtk.vtkCellArray())

    return data

    V = np.asarray(data.points)
    F = np.reshape(data.polys.connectivity_array, (-1, 3))
    E = np.reshape(data.lines.connectivity_array, (-1, 2))
    N = np.asarray(data.point_data["N"])

    # C = np.reshape(V[E], (-1, 6))  # chords
    # Minv = M.power(-1)  # diagonal matrix inverse.

    u, v = np.unstack(V[E], axis=1)
    vec = u - v
    trad = np.linalg.norm(vec, axis=1)  # target distances.

    rate = 1e-2

    L = igl.cotmatrix(V, F)

    from scipy.sparse.linalg import minres, cg

    for i in range(10):
        print(i, rate)

        M = igl.massmatrix(V, F)

        Q = M - rate * L
        P = sp.sparse.diags(1 / Q.diagonal())

        for comp in np.unstack(V, axis=1):
            comp[...], ret = minres(Q, M @ comp, x0=comp, maxiter=100, rtol=1e-3, M=P)
            # comp[...], ret = cg(Q, M @ comp, x0=comp, maxiter=100, rtol=1e-4, M=P)
            print(ret)

        V -= np.mean(V, axis=0, keepdims=True)
        V /= np.sqrt(np.mean(np.square(V)))

        rate *= 1.25

    data.points = V

    return data

    # solver = sp.linalg.factorized(Q[free, :][:, free])
    # verts[free, :] = solver(B[free, :] - Q[free, :][:, ~free] @ verts[~free, :])
    # # verts[free, :] = solver(B[free, :] - Q[free, :][:, ~free] @ verts[~free, :])

    # # Gauss-Newton Distance Constraints
    # vec = np.subtract(verts[links], np.expand_dims(verts, 1))
    # rad = np.linalg.norm(vec, axis=-1)
    #
    # # verts -= np.mean(verts, axis=0)
    # # verts /= np.sqrt(np.mean(verts * verts))
    #
    # verts -= np.mean(verts[~free], axis=0)
    # verts /= np.sqrt(np.mean(verts[~free] * verts[~free]))
    #
    # pdata.points = verts

    return data

    for _ in range(3):
        V = np.asarray(data.points, copy=True)
        F = np.reshape(data.polys.connectivity_array, (-1, 3), copy=True)

        print(V.shape, F.shape)

        rate = 1e-4
        L = igl.cotmatrix(V, F)

        for _ in range(3):
            M = igl.massmatrix(V, F)
            V = spsolve(M - rate * L, M @ V)
            V -= np.mean(V, axis=0, keepdims=True)
            V /= np.sqrt(np.mean(np.square(V)))

        data.points = V

        pipe = vtk.vtkDecimatePro()
        pipe.input_data = data
        pipe.preserve_topology = True
        pipe.target_reduction = 0.05
        pipe.Update()
        data = pipe.output

    print("final", V.shape, F.shape)
    return data


# flow()
flow()


# # %%
#
#
# def foo():
#     print(original)
#     x = 4
#     print(x)
#     print(cached)
#
#
# # foo()
#
#
# def bar():
#     import inspect
#
#     print(foo.__code__.co_names)
#     for name in foo.__code__.co_names:
#         loc = inspect.currentframe().f_back.f_locals
#         print(name, type(loc.get(name)))
#         loc = inspect.currentframe().f_back.f_globals
#         print(name, type(loc.get(name)))
#
#
# bar()
#
# # %%
#
# # initial smoothing. one stage of (C)MCF.
# verts = np.asarray(pipe.output.points, copy=True)
# faces = np.reshape(pipe.output.polys.connectivity_array, (-1, 3), copy=True)
# denorm = normalize(verts)
#
# rate = 5e-3
# L = igl.cotmatrix(verts, faces)
# M = igl.massmatrix(verts, faces)
# verts = sp.sparse.linalg.spsolve(M - rate * L, M @ verts, use_umfpack=False)
#
# L = igl.cotmatrix(verts, faces)
# M = igl.massmatrix(verts, faces)
# Minv = sp.sparse.diags(1 / M.diagonal())
# N = igl.per_vertex_normals(verts, faces)
# K = Minv @ igl.gaussian_curvature(verts, faces)
# H = Minv @ np.einsum("id,id->i", N, L @ verts)
#
# # print(K.min(), K.max())
# km = min(np.abs(np.quantile(K, [0.05, 0.95])))
#
# original.points = verts
# original.point_data["K"] = np.clip(K, -km, km)
# original.point_data["H"] = np.clip(H, *np.quantile(H, [0.05, 0.95]))
#
# if __debug__:
#     tmp = vtk.vtkPolyData()
#     tmp.DeepCopy(original)
#     tmp.points = denorm(verts)
#
#     writer = vtk.vtkPolyDataWriter(file_name="devel/smoothed.vtk")
#     writer.input_data = tmp
#     writer.Update()
#
# print(f"{original.GetNumberOfPoints() = }")
#
# # %%
#
# verts = np.copy(original.points)
#
# L = igl.cotmatrix(verts, faces)
# M = igl.massmatrix(verts, faces)
# Minv = sp.sparse.diags(1 / M.diagonal())
#
# N = igl.per_vertex_normals(verts, faces)
# K = Minv @ igl.gaussian_curvature(verts, faces)
# H = Minv @ np.einsum("id,id->i", N, L @ verts)
#
# # QL = L.T @ (Minv @ L)
# # QH = igl.hessian_energy(verts, faces)
# # QcH = igl.curved_hessian_energy(verts, faces)
# # sp.sparse.linalg.cg
#
# # eps = 0
# # K2 = sp.sparse.linalg.spsolve(eps * QL + (1 - eps) * M, eps * M @ K)
# # print('K2', np.quantile(K2, [0.01, 0.99]))
#
# q = 1e-1
#
# diam = 1.0
# C = np.abs(K + (diam / 2) ** 2) - np.abs(H * diam)
# # C = np.clip(C, -50, 50)
# C = np.clip(C, -q, q)
#
# q = 10
# H = np.clip(H, -q, q)
#
# s = 1.0
# L = igl.cotmatrix(verts, faces)
# for _ in range(3):
#     M = igl.massmatrix(verts, faces)
#     verts = sp.sparse.linalg.spsolve(M - s * L, M @ verts)
#     normalize(verts)
#
# if __debug__:
#     tmp = vtk.vtkPolyData()
#     tmp.DeepCopy(original)
#     tmp.points = denorm(verts)
#     tmp.point_data["K"] = K
#     tmp.point_data["H"] = H
#     tmp.point_data["C"] = C
#
#     writer = vtk.vtkPolyDataWriter(file_name="devel/smoothed2.vtk")
#     writer.input_data = tmp
#     writer.Update()
#
# # %%
#
# # Decimate mesh for flow.
# pipe = vtk.vtkDecimatePro(input_data=original)
# pipe.preserve_topology = True
# pipe.target_reduction = 0.75
# pipe.maximum_error = 1e-2
# pipe.Update()
# pdata: vtk.vtkPolyData = pipe.output
#
# print(f"{pdata.GetNumberOfPoints() = }")
#
# verts = np.asarray(pdata.points, copy=True)
# faces = np.reshape(pipe.output.polys.connectivity_array, (-1, 3), copy=True)
# denorm = normalize(verts)
#
# if __debug__:
#     tmp = vtk.vtkPolyData()
#     tmp.DeepCopy(pdata)
#     tmp.points = denorm(tmp.points)
#
#     writer = vtk.vtkPolyDataWriter(file_name="devel/decimated.vtk")
#     writer.input_data = tmp
#     writer.Update()
#
# # %%
#
# M = igl.massmatrix(verts, faces)
# Minv = M.power(-1)  # invert diagonal matrix
#
# L = igl.cotmatrix(verts, faces)
#
# N = igl.per_vertex_normals(verts, faces)
#
# K = Minv @ igl.gaussian_curvature(verts, faces)
# HN = -Minv @ L @ verts
# # H = np.dot(HN, N)
# H = np.einsum("vi, vi -> v", HN, N)
#
# print(np.quantile(K, [0, 1]))
# print(np.quantile(H, [0, 1]))
#
# print(np.quantile(K, [0.05, 0.95]))
# print(np.quantile(H, [0.05, 0.95]))
#
# K = np.clip(K, -40, 40)
# H = np.clip(H, -15, 15)
#
# if __debug__:
#     tmp = vtk.vtkPolyData()
#     tmp.DeepCopy(pdata)
#     tmp.points = denorm(tmp.points)
#
#     tmp.point_data["Gaus"] = K
#     tmp.point_data["Mean"] = H
#
#     writer = vtk.vtkPolyDataWriter(file_name="devel/curvatures.vtk")
#     writer.input_data = tmp
#     writer.Update()
#
# # %%
#
# mesh = vtk.vtkPolyData()
# mesh.DeepCopy(pdata)
# mesh.point_data.RemoveArray("Normals")
#
# verts = np.asarray(mesh.points)
# faces = np.reshape(mesh.polys.connectivity_array, (-1, 3))
#
# center = np.mean(verts, axis=0)
# verts -= center
# scale = np.sqrt(np.mean(np.square(verts)))
# verts /= scale
#
# rate = 1e-2
# L = igl.cotmatrix(verts, faces)
# for it in range(10):
#     print(it)
#     M = igl.massmatrix(verts, faces)
#
#     verts = sp.sparse.linalg.factorized(M - rate * L)(M @ verts)
#     verts -= np.mean(verts, axis=0)
#     verts /= np.sqrt(np.mean(np.square(verts)))
#
#     rate *= 1.5
#     rate = np.clip(rate, 1e-5, 1e1)
#
# verts *= scale
# verts += center
#
# mesh.points = verts
#
# writer = vtk.vtkPolyDataWriter(file_name="devel/wip.vtk")
# writer.SetInputData(mesh)
# writer.Update()
