from pathlib import Path

import vtk
import numpy as np
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import vtkInformation, vtkInformationVector
from vtkmodules.vtkCommonDataModel import vtkPolyData

FILE_INNER = Path(
    "data/OASIS4-00010_0102_00-00_BRAIN-T1-SYNTH-3D-SYNTH-PRE_hdrfix_n4_reg_haca3_lesionfilled_slant_wlesions_surface-inner.vtk"
)
FILE_OUTER = Path(
    "data/OASIS4-00010_0102_00-00_BRAIN-T1-SYNTH-3D-SYNTH-PRE_hdrfix_n4_reg_haca3_lesionfilled_slant_wlesions_surface-outer.vtk"
)


class Normalize(VTKPythonAlgorithmBase):
    def __init__(self, **kwargs):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType='vtkPolyData')
        for name, value in kwargs.items():
            setattr(self, name, value)

    def RequestInformation(self, request: vtkInformation, ins_infos: tuple[vtkInformationVector],
                           out_infos: vtkInformationVector):
        return 1

    def RequestData(self, request: vtkInformation, ins_infos: tuple[vtkInformationVector],
                    out_infos: vtkInformationVector):
        in_info = ins_infos[0].GetInformationObject(0)
        out_info = out_infos.GetInformationObject(0)

        src = vtkPolyData.GetData(in_info)
        dst = vtkPolyData.GetData(out_info)

        dst.ShallowCopy(src)

        dst.points = dst.points - np.mean(dst.points, axis=0, keepdims=True)
        dst.points = dst.points / np.sqrt(np.mean(np.square(dst.points)))

        return 1


pipe = vtk.vtkPolyDataReader()
pipe.SetFileName(str(FILE_INNER))
pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
pipe = vtk.vtkCleanPolyData(input_connection=pipe.output_port)
pipe.ConvertLinesToPointsOff()
pipe.ConvertPolysToLinesOff()
pipe.ConvertStripsToPolysOff()
pipe = Normalize(input_connection=pipe.output_port)
writer = vtk.vtkPolyDataWriter(input_connection=pipe.output_port)
writer.SetFileName('devel/inner.norm.vtk')
writer.SetFileTypeToBinary()

pipe = vtk.vtkPolyDataReader()
pipe.SetFileName(str(FILE_OUTER))
pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
pipe = vtk.vtkCleanPolyData(input_connection=pipe.output_port)
pipe.ConvertLinesToPointsOff()
pipe.ConvertPolysToLinesOff()
pipe.ConvertStripsToPolysOff()
pipe = Normalize(input_connection=pipe.output_port)
writer = vtk.vtkPolyDataWriter(input_connection=pipe.output_port)
writer.SetFileName('devel/outer.norm.vtk')
writer.SetFileTypeToBinary()

writer.Update()
