from pathlib import Path

from vtkmodules.vtkFiltersCore import vtkTriangleFilter, vtkQuadricDecimation
from vtkmodules.vtkIOLegacy import vtkPolyDataReader, vtkPolyDataWriter


def decimate(src: Path, *, target_reduction: float):
    """Downsample large meshes."""

    dst = src.with_stem(f'{src.stem}-decimated')
    print(f"downsampling {src} -> {dst}")

    pipe = vtkPolyDataReader()
    pipe.file_name = src
    pipe = vtkTriangleFilter(input_connection=pipe.output_port)
    pipe = vtkQuadricDecimation(input_connection=pipe.output_port)
    pipe.SetTargetReduction(target_reduction)
    pipe = vtkPolyDataWriter(input_connection=pipe.output_port)
    pipe.file_name = dst
    pipe.SetFileTypeToBinary()
    pipe.Update()

    return dst
