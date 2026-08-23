from pathlib import Path

from vtkmodules.vtkFiltersCore import vtkTriangleFilter, vtkQuadricDecimation
from vtkmodules.vtkIOLegacy import vtkPolyDataReader, vtkPolyDataWriter


def decimate(src: Path, dst: Path, *, target_reduction: float):
    """Downsample large meshes."""

    pipe = vtkPolyDataReader()
    pipe.file_name = src
    pipe = vtkTriangleFilter(input_connection=pipe.output_port)
    pipe = vtkQuadricDecimation(input_connection=pipe.output_port)
    pipe.SetTargetReduction(target_reduction)
    pipe = vtkPolyDataWriter(input_connection=pipe.output_port)
    pipe.file_name = dst
    pipe.SetFileTypeToBinary()
    pipe.Update()
