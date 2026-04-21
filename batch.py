import hashlib
import inspect
import math
import shutil
import types
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

import igl
import numpy as np
import scipy as sp
import vtk
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from tqdm import tqdm

from vtk import vtkPolyData

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
            writer.SetFileTypeToBinary()
            writer.Update()
        else:
            print(f"cached {self.name} {self.path.name}!")
            reader = vtk.vtkPolyDataReader(file_name=str(self.path))
            reader.Update()
            output = reader.output

        self.link.unlink(missing_ok=True)
        shutil.copy(self.path, self.link)

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
def decimate():
    data: vtkPolyData = original()

    RATE = 1e-3  # should be small, on order of 5e-4 to 2e-3.
    ITER = 3  # should be small, but *not* one. more iterations with smaller rate yields better results for much longer runtime.
    MERGE_TOL = 1e-4  # should be less than half the smallest features to preserve. on order of 1e-3, 1e-4
    DECIMATE = 0.75  # target polygon reduction. with sufficient smoothing, order of 0.5 to 0.9 is probably reasonable.

    V = np.asarray(data.points, copy=True)
    F = np.reshape(data.polys.connectivity_array, (-1, 3))
    L = igl.cotmatrix(V, F)

    for _ in tqdm(range(ITER), desc="decimate"):
        M = igl.massmatrix(V, F)

        V = sp.sparse.linalg.factorized(M - RATE * L)(M @ V)
        V -= V.mean(axis=0, keepdims=True)
        V /= np.sqrt(np.mean(np.square(V)))

        data.points = V

    pipe = vtk.vtkCleanPolyData(input_data=data)
    pipe.SetTolerance(MERGE_TOL)
    pipe = vtk.vtkQuadricDecimation(input_connection=pipe.output_port)
    pipe.SetTargetReduction(DECIMATE)
    pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
    pipe.Update()
    data = pipe.output

    return data


@cached()
def link():
    data = decimate()

    V = np.asarray(data.points)
    F = np.reshape(data.polys.connectivity_array, (-1, 3))

    A = igl.adjacency_matrix(F)
    Cn, C, Ck = igl.connected_components(A)
    data.point_data["C"] = C
    # component 1 is the inner; component 0 is the outer.

    assert Cn == 2, "There must be two connected components."

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

            jdxs = jdxs[arg[:1]]

            # N[jdxs] @ N[idx]
            # V[jdxs] - V[idx]

            return [(idx, jdx) for jdx in jdxs]

        with ProcessPoolExecutor() as ex:
            for links in tqdm(
                ex.map(get_links, idxs, jdxss, chunksize=512),
                total=len(idxs),
                desc="link",
            ):
                for link in links:
                    data.lines.InsertNextCell(2, link)

    return data


@cached()
def flow():
    data = link()

    V = np.asarray(data.points, copy=True)
    F = np.reshape(data.polys.connectivity_array, (-1, 3), copy=True)
    E = np.reshape(data.lines.connectivity_array, (-1, 2), copy=True)

    A = igl.adjacency_matrix(F)
    Cn, C, Ck = igl.connected_components(A)

    OUT = np.nonzero(C == 0)

    print(E.shape)
    print(F.shape)
    print(V.shape)

    print(np.unique(E[:, 0]).shape)

    V -= np.mean(V[OUT], axis=0, keepdims=True)
    V /= np.sqrt(np.mean(np.square(V[OUT])))

    L0 = igl.cotmatrix(V, F)


    i, j = np.unstack(E[np.random.random(len(E)) < 0.05], axis=1)

    P = np.power(np.linalg.norm(V[i] - V[j], axis=1), -1)
    P /= np.mean(P)

    A = sp.sparse.dok_matrix(L0.shape)
    A[i, j] = P
    A[j, i] = P

    G = sp.sparse.csgraph.laplacian(A)

    rate = 5e-3
    relax = 0.5

    for _ in tqdm(range(6)):
        M = igl.massmatrix(V, F)
        energy = relax * G - L0

        V = spsolve(M + rate * energy, M @ V)

        V -= np.mean(V[OUT], axis=0, keepdims=True)
        V /= np.sqrt(np.mean(np.square(V[OUT])))

    V -= np.mean(V, axis=0, keepdims=True)
    V /= np.sqrt(np.mean(np.square(V)))
    data.points = V

    return data


flow()
