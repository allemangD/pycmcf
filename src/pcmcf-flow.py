from pathlib import Path

import igl
import numpy as np
import scipy as sp
import vtk
from sksparse.cholmod import cho_solve
from tqdm import tqdm

RATE = 0.02
GROW = 1.4  # geometric growth factor. maybe replace with sigmoid? we're kinda going for roughly constant rms here.
RMAX = 500.0

MAX_ITER = 50
STOP = 1e-4

REGULARIZE = 1e-4

INNER_PATH = Path("data/inner.vtk")
OUTER_PATH = Path("output/DeterministicAtlas__Reconstruction__surf__subject_outer.vtk")

pipe = vtk.vtkPolyDataReader(file_name=INNER_PATH)
pipe = vtk.vtkCleanPolyData(input_connection=pipe.output_port)
pipe.SetTolerance(1e-4)
pipe = vtk.vtkCurvatures(input_connection=pipe.output_port)
pipe.SetCurvatureTypeToMean()
pipe.Update()
inner: vtk.vtkPolyData = pipe.output

pipe = vtk.vtkPolyDataReader(file_name=OUTER_PATH)
pipe = vtk.vtkCleanPolyData(input_connection=pipe.output_port)
pipe.SetTolerance(1e-4)
pipe = vtk.vtkCurvatures(input_connection=pipe.output_port)
pipe.SetCurvatureTypeToMean()
pipe.Update()
outer: vtk.vtkPolyData = pipe.output

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
for f in output.glob('inner-*.vtk'):
    f.unlink()
for f in output.glob('outer-*.vtk'):
    f.unlink()

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

    pipe = vtk.vtkPolyDataNormals()
    pipe.input_data = inner
    pipe = vtk.vtkPolyDataWriter(input_connection=pipe.output_port)
    pipe.file_name = output.joinpath(f"inner-{it:03}.vtk")
    pipe.SetFileTypeToBinary()
    pipe.Update()

    pipe = vtk.vtkPolyDataNormals()
    pipe.input_data = outer
    pipe = vtk.vtkPolyDataWriter(input_connection=pipe.output_port)
    pipe.file_name = output.joinpath(f"outer-{it:03}.vtk")
    pipe.SetFileTypeToBinary()
    pipe.Update()

    if rms_delta < STOP:
        break

    rate *= GROW
    rate = np.clip(rate, 0, RMAX)
