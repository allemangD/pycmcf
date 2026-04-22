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
from scipy.spatial import cKDTree
from sksparse.cholmod import cho_solve
from tqdm import tqdm
from vtk import (
    vtkPolyData,
    vtkInformation,
    vtkInformationVector,
    vtkStreamingDemandDrivenPipeline,
)
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

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


def save_anim(
    file_name,
    verts: list[np.ndarray],
    faces: np.ndarray,
    edges: np.ndarray | None,
):
    inds = np.arange(len(verts))

    A = igl.adjacency_matrix(faces)
    Cn, C, Ck = igl.connected_components(A)

    if faces is not None:
        poly_array = vtk.vtkCellArray()
        for i, j, k in faces:
            poly_array.InsertNextCell(3, (i, j, k))
    else:
        poly_array = None

    if edges is not None:
        line_array = vtk.vtkCellArray()
        for i, j in edges:
            line_array.InsertNextCell(2, (i, j))
    else:
        line_array = None

    class Source(VTKPythonAlgorithmBase):
        def __init__(self, **kwargs):
            VTKPythonAlgorithmBase.__init__(
                self, nInputPorts=0, nOutputPorts=1, outputType="vtkPolyData"
            )
            for name, value in kwargs.items():
                setattr(self, name, value)

        def RequestInformation(
            self,
            request: vtkInformation,
            ins_infos: tuple[()],
            out_infos: vtkInformationVector,
        ):
            out_info = out_infos.GetInformationObject(0)

            out_info.Set(vtkStreamingDemandDrivenPipeline.TIME_STEPS(), inds, len(inds))
            out_info.Set(
                vtkStreamingDemandDrivenPipeline.TIME_RANGE(), [inds[0], inds[-1]], 2
            )

            return 1

        def RequestData(
            self,
            request: vtkInformation,
            ins_infos: tuple[()],
            out_infos: vtkInformationVector,
        ):
            out_info = out_infos.GetInformationObject(0)
            i = int(out_info.Get(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()))

            out_data = vtkPolyData.GetData(out_info)

            out_data.points = verts[i]
            out_data.point_data["C"] = C
            out_data.polys = poly_array
            out_data.lines = line_array

            progress.update(1)
            return 1

    with tqdm(total=len(verts), desc=f"write {file_name}") as progress:
        pipe = Source()
        pipe = vtk.vtkHDFWriter(input_connection=pipe.output_port, file_name=file_name)
        pipe.write_all_time_steps = True
        pipe.Write()


@cached()
def flow_cmcf():
    data = link()

    V = np.asarray(data.points, copy=True)
    F = np.reshape(data.polys.connectivity_array, (-1, 3), copy=True)
    E = np.reshape(data.lines.connectivity_array, (-1, 2), copy=True)
    E_sub = E[np.random.random(len(E)) < 0.05]

    A = igl.adjacency_matrix(F)
    Cn, C, Ck = igl.connected_components(A)
    OUT = np.nonzero(C == 1)

    V = V - np.mean(V[OUT])
    V = V / np.sqrt(np.mean(np.square(V[OUT])))

    rate = 5e-4
    grow = 1.1
    rmax = 1e-1

    L0 = igl.cotmatrix(V, F)

    I = 1e-5 * sp.sparse.eye(len(V))

    Vs = [V]

    for _ in tqdm(range(80), desc="cmcf"):
        M = igl.massmatrix(V, F)

        V = cho_solve(
            M + I - rate * L0,
            (M + I) @ V,
        )

        V -= np.mean(V[OUT], axis=0, keepdims=True)
        V /= np.sqrt(np.mean(np.square(V[OUT])))

        Vs.append(V)

        rate *= grow
        rate = np.clip(rate, 0, rmax)

    save_anim("devel/anim-cmcf.hdf", Vs, F, E_sub)

    data.points = V
    return data


@cached()
def flow_link():
    data = link()

    V = np.asarray(data.points, copy=True)
    F = np.reshape(data.polys.connectivity_array, (-1, 3), copy=True)
    E = np.reshape(data.lines.connectivity_array, (-1, 2), copy=True)
    E_sub = E[np.random.random(len(E)) < 0.05]

    A = igl.adjacency_matrix(F)
    Cn, C, Ck = igl.connected_components(A)
    OUT = np.nonzero(C == 1)

    V = V - np.mean(V[OUT])
    V = V / np.sqrt(np.mean(np.square(V[OUT])))

    i, j = np.unstack(E, axis=1)
    P = np.power(np.linalg.norm(V[i] - V[j], axis=1), -1)
    P /= np.mean(P)
    A = sp.sparse.dok_matrix((len(V), len(V)))
    A[i, j] = P
    A[j, i] = P
    G = sp.sparse.csgraph.laplacian(A)

    rate = 5e-4
    grow = 1.1
    rmax = 1e-1

    relax = 0.9

    L0 = igl.cotmatrix(V, F)

    I = 1e-5 * sp.sparse.eye(len(V))

    Vs = [V]

    for _ in tqdm(range(80), desc="link"):
        M = igl.massmatrix(V, F)

        V = cho_solve(
            M + I - rate * L0 + relax * G,
            (M + I + relax * G) @ V,
        )

        V -= np.mean(V[OUT], axis=0, keepdims=True)
        V /= np.sqrt(np.mean(np.square(V[OUT])))

        Vs.append(V)

        rate *= grow
        rate = np.clip(rate, 0, rmax)

    save_anim(f"devel/anim-link-{int(relax * 100)}.hdf", Vs, F, E_sub)

    data.points = V
    return data


@cached()
def flow_pred_corr():
    data = link()

    V = np.asarray(data.points, copy=True)
    F = np.reshape(data.polys.connectivity_array, (-1, 3), copy=True)
    E = np.reshape(data.lines.connectivity_array, (-1, 2), copy=True)
    E_sub = E[np.random.random(len(E)) < 0.05]

    A = igl.adjacency_matrix(F)
    Cn, C, Ck = igl.connected_components(A)
    OUT = np.nonzero(C == 1)

    V = V - np.mean(V[OUT])
    V = V / np.sqrt(np.mean(np.square(V[OUT])))

    i, j = np.unstack(E, axis=1)
    true_dist = np.linalg.norm(V[i] - V[j], axis=1, keepdims=True)

    rate = 5e-4
    grow = 1.1
    rmax = 1e-1

    relax = 0.9

    L0 = igl.cotmatrix(V, F)

    I = 1e-5 * sp.sparse.eye(len(V))

    Vs = [V]

    for _ in tqdm(range(80), desc="corr"):
        M = igl.massmatrix(V, F)

        V = cho_solve(
            M + I - rate * L0,
            (M + I) @ V,
        )

        V -= np.mean(V[OUT], axis=0, keepdims=True)
        V /= np.sqrt(np.mean(np.square(V[OUT])))

        diff = V[i] - V[j]
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        norm = diff / dist
        extra = dist - true_dist
        fixup = -norm * extra / 2  # amount to push 'i' vertex.

        accum = np.zeros((len(V), 3))
        weight = np.zeros((len(V), 1))

        accum[i] += fixup
        weight[i] += 1
        mask = weight > 0

        np.divide(accum, weight, out=accum, where=mask)
        np.add(V, accum * relax, out=V, where=mask)

        Vs.append(V)

        rate *= grow
        rate = np.clip(rate, 0, rmax)

    save_anim(f"devel/anim-pc-{int(relax * 100)}.hdf", Vs, F, E_sub)

    data.points = V
    return data


flow_cmcf()
flow_link()
flow_pred_corr()
