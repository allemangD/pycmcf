from pathlib import Path

import igl
import numpy as np
import scipy as sp
import vtk
from sksparse.cholmod import cho_solve
from tqdm import tqdm

for name, path in {
    (
        "inner",
        "data/OASIS4-00010_0102_00-00_BRAIN-T1-SYNTH-3D-SYNTH-PRE_hdrfix_n4_reg_haca3_lesionfilled_slant_wlesions_surface-inner.vtk",
    ),
    (
        "outer",
        "data/OASIS4-00010_0102_00-00_BRAIN-T1-SYNTH-3D-SYNTH-PRE_hdrfix_n4_reg_haca3_lesionfilled_slant_wlesions_surface-outer.vtk",
    ),
}:
    print(name)
    pipe = vtk.vtkPolyDataReader(file_name=path)
    pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
    pipe = vtk.vtkCleanPolyData(input_connection=pipe.output_port)
    pipe.convert_lines_to_points = False
    pipe.convert_polys_to_lines = False
    pipe.convert_strips_to_polys = False
    pipe.Update()

    data = pipe.output

    V = np.asarray(data.points)
    V -= np.mean(V, axis=0, keepdims=True)
    V /= np.sqrt(np.mean(np.square(V)))

    RATE = 1e-3  # should be small, on order of 5e-4 to 2e-3.
    ITER = 3  # should be small, but *not* one. more iterations with smaller rate yields better results for much longer runtime.
    MERGE_TOL = 1e-4  # should be less than half the smallest features to preserve. on order of 1e-3, 1e-4
    DECIMATE = 0.75  # target polygon reduction. with sufficient smoothing, order of 0.5 to 0.9 is probably reasonable.

    V = np.asarray(data.points, copy=True)
    F = np.reshape(data.polys.connectivity_array, (-1, 3))
    L = igl.cotmatrix(V, F)
    I = 1e-5 * sp.sparse.eye(len(V))

    for _ in tqdm(range(ITER), desc="decimate"):
        M = igl.massmatrix(V, F)

        V = cho_solve(
            M + I - RATE * L,
            (M + I) @ V,
        )
        V -= V.mean(axis=0, keepdims=True)
        V /= np.sqrt(np.mean(np.square(V)))

    data.points = V

    pipe = vtk.vtkCleanPolyData(input_data=data)
    pipe.SetTolerance(MERGE_TOL)
    pipe = vtk.vtkQuadricDecimation(input_connection=pipe.output_port)
    pipe.SetTargetReduction(DECIMATE)
    pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
    pipe = vtk.vtkPolyDataNormals(input_connection=pipe.output_port)
    pipe = vtk.vtkPolyDataWriter(input_connection=pipe.output_port)
    pipe.file_name = str(Path(path).with_stem(name))
    pipe.SetFileTypeToBinary()
    pipe.Update()

"""
then register with

~/src/pycmcf/ $ deformetrica estimate src/model.xml src/data-set.xml -p src/optimization-parameters.xml
"""
