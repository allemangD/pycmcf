from pathlib import Path
from subprocess import Popen

import igl
import numpy as np
import scipy as sp
import sksparse.cholmod
import vtk
from tqdm import tqdm
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import vtkInformation, vtkInformationVector
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline


def cho_solve(A, b):
    return sksparse.cholmod.cholesky(A)(b)


dst_path = Path("~/src/pycmcf/devel/temp.vtk").expanduser()

src_path = Path(
    "~/src/pycmcf/deformetrica-geodesic-args/output/PrincipalGeodesicAnalysis__Reconstruction__surf__subject_inner.vtk",
).expanduser()
pipe = vtk.vtkPolyDataReader(file_name=src_path)
pipe.Update()
inner: vtk.vtkPolyData = pipe.output

src_path = Path(
    "~/src/pycmcf/deformetrica-geodesic-args/output/PrincipalGeodesicAnalysis__Reconstruction__surf__subject_outer.vtk",
).expanduser()
pipe = vtk.vtkPolyDataReader(file_name=src_path)
pipe.Update()
outer: vtk.vtkPolyData = pipe.output

mesh = vtk.vtkPolyData()
mesh.points = np.concatenate([inner.points, outer.points], axis=0)

mesh.lines = vtk.vtkCellArray()
for i in range(len(inner.points)):
    mesh.lines.InsertNextCell(2, [i, i + len(inner.points)])

mesh.polys = vtk.vtkCellArray()
F = np.asarray(inner.polys.connectivity_array).reshape((-1, 3))
for tup in F:
    mesh.polys.InsertNextCell(len(tup), tup)
    mesh.polys.InsertNextCell(len(tup), tup + len(inner.points))

F = np.asarray(inner.polys.connectivity_array).reshape((-1, 3))

C = np.stack([inner.points, outer.points], axis=1)

U, V = np.unstack(C, axis=1)
C -= np.mean(C, axis=0, keepdims=True)
C /= np.sqrt(np.mean(np.square(C[:, 0, :])))
true_dist = np.linalg.norm(U - V, axis=1, keepdims=True)

time_sequence = [0.0]
points_sequence = [np.concatenate([U, V], axis=0)]

rate = 1e-2
grow = 1.2
rmax = 1e-1

L0 = igl.cotmatrix(C.reshape((-1, 6)), F)
I = 1e-5 * sp.sparse.eye(len(C.reshape((-1, 6))))
for _ in tqdm(range(80)):
    M = igl.massmatrix(C.reshape((-1, 6)), F)

    C = cho_solve(M + I - rate * L0, (M + I) @ C.reshape(-1, 6)).reshape(C.shape)
    U, V = np.unstack(C, axis=1)
    C -= np.mean(C, axis=0, keepdims=True)
    C /= np.sqrt(np.mean(np.square(U)))

    diff = U - V
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    norm = diff / dist
    extra = dist - true_dist
    fixup = -norm * extra / 2

    U += fixup
    V -= fixup

    points_sequence.append(np.concatenate([U, V], axis=0))
    time_sequence.append(time_sequence[-1] + rate)
    rate = np.clip(rate * grow, 0, rmax)

U, V = np.unstack(C, axis=1)
mesh.points = np.concatenate([U, V], axis=0)

# pipe = vtk.vtkTrivialProducer(output=mesh)
# pipe = vtk.vtkPolyDataWriter(file_name=dst_path, input_connection=pipe.output_port)
# pipe.SetFileTypeToBinary()
# pipe.Update()

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

        out_info.Set(
            vtkStreamingDemandDrivenPipeline.TIME_STEPS(),
            list(range(len(points_sequence))),
            len(points_sequence),
        )
        out_info.Set(
            vtkStreamingDemandDrivenPipeline.TIME_RANGE(),
            [0, len(points_sequence) - 1],
            2,
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

        out_data.points = points_sequence[i]
        # out_data.point_data["C"] = C
        # for name, data in point_data.items():
        #     out_data.point_data[name] = data
        out_data.polys = mesh.polys
        out_data.lines = mesh.lines

        return 1

pipe = Source()

writer = vtk.vtkHDFWriter(input_connection=pipe.output_port, file_name='devel/test-animation.hdf')
writer.write_all_time_steps = True
writer.Write()
