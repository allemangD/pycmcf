from pathlib import Path

import igl
import numpy as np
import scipy as sp
import vtk
from sksparse.cholmod import cho_solve
from tqdm import tqdm

RATE = 0.2  # mm/k/t  # T should be small, on order of 5e-3 to 2e-2 mm/u.
ITER = 4  # N # should be small, but *not* one. more iterations with smaller rate yields better results for much longer runtime.
DECIMATE = 0.60  # frac  # target polygon reduction. with sufficient smoothing, order of 0.5 to 0.9 is probably reasonable.
REGULARIZE = 1e-4
MERGE_TOL = 1e-3  # merge tol for coincident points

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
    pipe = decm = vtk.vtkQuadricDecimation(input_connection=pipe.output_port)
    pipe.SetTargetReduction(1 - np.sqrt(1 - DECIMATE))
    pipe.Update()
    print(f"pre {decm.actual_reduction = }")

    data = pipe.output

    V = np.asarray(data.points)

    V = np.asarray(data.points, copy=True)
    F = np.reshape(data.polys.connectivity_array, (-1, 3))
    L = igl.cotmatrix(V, F)
    I = REGULARIZE * sp.sparse.eye(len(V))

    for _ in tqdm(range(ITER), desc="decimate"):
        M = igl.massmatrix(V, F)

        V = cho_solve(
            M + I - RATE * L,
            (M + I) @ V,
        )

    print("presmoothing bounds", data.bounds)
    print("smoothing rms", np.sqrt(np.mean((data.points - V) ** 2)))
    data.points = V
    print("postsmoothing bounds", data.bounds)

    pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
    pipe = decm = vtk.vtkQuadricDecimation(input_connection=pipe.output_port)
    pipe.SetTargetReduction(1 - np.sqrt(1 - DECIMATE))
    norm = pipe = vtk.vtkPolyDataNormals(input_connection=pipe.output_port)
    pipe.SplittingOff()
    pipe = vtk.vtkPolyDataWriter(input_connection=pipe.output_port)
    pipe.file_name = str(Path(path).with_stem(name))
    pipe.SetFileTypeToBinary()
    pipe.Update()

    print(f"post {decm.actual_reduction = }")

    F = np.reshape(norm.output.polys.connectivity_array, (-1, 3))
    V = np.asarray(norm.output.points)
    M = igl.massmatrix(V, F).diagonal()
    print(M.min(), M.max())


"""
then register with

~/src/pycmcf/ $ deformetrica estimate src/model.xml src/data-set.xml -p src/optimization-parameters.xml
"""
