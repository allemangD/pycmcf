# %%
import igl
import numpy as np
import scipy.sparse as sp
from vtkmodules.vtkCommonCore import vtkInformationVector

np.set_printoptions(suppress=True)

import vtk
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtk import vtkStreamingDemandDrivenPipeline, vtkPolyData


class CMCF(VTKPythonAlgorithmBase):
    L: sp.csc_matrix
    T: float

    def __init__(self, **kwargs):
        VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=1,
            inputType="vtkPolyData",
            nOutputPorts=1,
            outputType="vtkPolyData",
        )
        for name, value in kwargs.items():
            setattr(self, name, value)

    def RequestInformation(
        self,
        request: vtk.vtkInformation,
        ins_infos: tuple[vtkInformationVector, ...],
        out_infos: vtk.vtkInformationVector,
    ):
        in_info = ins_infos[0].GetInformationObject(0)
        out_info = out_infos.GetInformationObject(0)

        T = np.array([0, 2e-3])
        out_info.Set(vtkStreamingDemandDrivenPipeline.TIME_STEPS(), T, len(T))
        out_info.Set(vtkStreamingDemandDrivenPipeline.TIME_RANGE(), [T[0], T[-1]], 2)

        return 1

    def RequestData(
        self,
        request: vtk.vtkInformation,
        ins_infos: tuple[vtkInformationVector, ...],
        out_infos: vtk.vtkInformationVector,
    ):
        in_info = ins_infos[0].GetInformationObject(0)
        in_data = vtkPolyData.GetData(in_info)

        out_info = out_infos.GetInformationObject(0)
        out_data = vtkPolyData.GetData(out_info)

        t = out_info.Get(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())
        print(t)

        out_data.ShallowCopy(in_data)

        if t == 0.0:
            self.T = t
            self.V = np.asarray(out_data.points, copy=True)
            self.F = np.reshape(out_data.polys.connectivity_array, (-1, 3))
            self.L = igl.cotmatrix(self.V, self.F)
        else:
            dt, self.T = t - self.T, t
            self.M = igl.massmatrix(self.V, self.F)
            self.V = sp.linalg.spsolve(self.M - dt * self.L, self.M @ self.V)

        self.V -= np.mean(self.V, axis=0, keepdims=True)
        self.V /= np.sqrt(np.mean(np.square(self.V)))

        out_data.points = self.V

        return 1


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

pipe = CMCF(input_connection=pipe.output_port)

writer = vtk.vtkHDFWriter(
    file_name="devel/anim.hdf",
    input_connection=pipe.output_port,
)

writer.write_all_time_steps = True
writer.Update()
