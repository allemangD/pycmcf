from pathlib import Path

import igl
import numpy as np
import scipy as sp
import vtk
from sksparse.cholmod import cho_solve
from tqdm import tqdm

RATE = 0.05
GROW = 1.4  # geometric growth factor. maybe replace with sigmoid? we're kinda going for roughly constant rms here.
RMAX = 500.0

MAX_ITER = 50
STOP = 1e-4

REGULARIZE = 1e-4

INNER_PATH = Path("data/inner.vtk")
OUTER_PATH = Path("output/DeterministicAtlas__Reconstruction__surf__subject_outer.vtk")

pipe = vtk.vtkPolyDataReader(file_name=INNER_PATH)
pipe.Update()
inner: vtk.vtkPolyData = pipe.output

ALPHA = 1e-4

v = np.asarray(inner.points)
f = np.asarray(inner.polys.connectivity_array).reshape((-1, 3))
m = igl.massmatrix(v, f)
l = igl.cotmatrix(v, f)
ql = l.T @ m.power(-1) @ l
n = igl.per_vertex_normals(v, f)
h = np.vecdot((l @ v) / np.expand_dims(m.diagonal(), 1), n)
zh = cho_solve(ALPHA * ql + (1 - ALPHA) * m, ALPHA * m * h) / ALPHA
inner.point_data["H"] = zh
print(zh.min(), zh.max())

pipe = vtk.vtkPolyDataReader(file_name=OUTER_PATH)
pipe.Update()
outer: vtk.vtkPolyData = pipe.output

v = np.asarray(outer.points)
f = np.asarray(outer.polys.connectivity_array).reshape((-1, 3))
m = igl.massmatrix(v, f)
l = igl.cotmatrix(v, f)
ql = l.T @ m.power(-1) @ l
n = igl.per_vertex_normals(v, f)
h = np.vecdot((l @ v) / np.expand_dims(m.diagonal(), 1), n)
zh = cho_solve(ALPHA * ql + (1 - ALPHA) * m, ALPHA * m * h) / ALPHA
outer.point_data["H"] = zh
print(zh.min(), zh.max())

vu = np.asarray(inner.points, copy=True)
vv = np.asarray(outer.points, copy=True)
assert vu.shape == vv.shape
V = np.concatenate([vu, vv], axis=1)

diff = vu - vv
true_dist = np.linalg.norm(diff, axis=1, keepdims=True)

Fu = np.asarray(inner.polys.connectivity_array, copy=True).reshape((-1, 3))
Fv = np.asarray(outer.polys.connectivity_array, copy=True).reshape((-1, 3))
assert np.array_equal(Fu, Fv)
F = Fu

Vcent = np.mean(V, axis=0, keepdims=True)
V -= Vcent
Vnorm = np.sqrt(np.mean(np.square(V)))

L0 = igl.cotmatrix(V, F)

I = REGULARIZE * sp.sparse.eye(len(V))

rate = RATE

output = Path("output-flow")
output.mkdir(exist_ok=True)
for f in output.glob("inner-*.vtk"):
    f.unlink()
for f in output.glob("outer-*.vtk"):
    f.unlink()

links = vtk.vtkPolyData()
links.SetPoints(vtk.vtkPoints())
links.SetLines(vtk.vtkCellArray())
links.points = np.concatenate([vu, vv], axis=0)
for i in range(len(vu)):
    links.InsertNextCell(vtk.VTK_LINE, 2, [i, i + len(vu)])

for it in tqdm(range(MAX_ITER), desc="corr"):
    M = igl.massmatrix(V, F)

    _prev = V.copy()

    V = cho_solve(
        M + I - rate * L0,
        (M + I) @ V,
    )
    V -= np.mean(V, axis=0, keepdims=True)
    V *= Vnorm / np.sqrt(np.mean(np.square(V)))

    vu = V[..., :3]
    vv = V[..., 3:]

    diff = vu - vv
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    norm = diff / dist
    extra = dist - true_dist
    fixup = -norm * extra / 2

    vu += fixup
    vv -= fixup

    rms_delta = np.sqrt(np.mean(np.square(V - _prev))) / rate

    inner.points = vu
    outer.points = vv
    links.points = np.concatenate([vu, vv], axis=0)

    pipe = vtk.vtkPolyDataNormals()
    pipe.input_data = inner
    pipe = vtk.vtkPolyDataWriter(input_connection=pipe.output_port)
    pipe.file_name = output.joinpath(f"{it:03}-inner.vtk")
    pipe.SetFileTypeToBinary()
    pipe.Update()

    pipe = vtk.vtkPolyDataNormals()
    pipe.input_data = outer
    pipe = vtk.vtkPolyDataWriter(input_connection=pipe.output_port)
    pipe.file_name = output.joinpath(f"{it:03}-outer.vtk")
    pipe.SetFileTypeToBinary()
    pipe.Update()

    pipe = vtk.vtkPolyDataWriter()
    pipe.input_data = links
    pipe.file_name = output.joinpath(f"{it:03}-links.vtk")
    pipe.SetFileTypeToBinary()
    pipe.Update()

    if rms_delta < STOP:
        break

    rate *= GROW
    rate = np.clip(rate, 0, RMAX)
