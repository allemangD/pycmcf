import logging
from pathlib import Path

from vtkmodules.vtkFiltersCore import vtkQuadricDecimation, vtkTriangleFilter
from vtkmodules.vtkIOLegacy import vtkPolyDataReader, vtkPolyDataWriter

logger = logging.getLogger(__name__)


def decimate(
    src: Path,
    *,
    target_reduction: float = 0.5,
):
    """Downsample large meshes."""

    dst = src.with_stem(f"{src.stem}-decimated")

    pipe = vtkPolyDataReader()
    pipe.file_name = src
    pipe = vtkTriangleFilter(input_connection=pipe.output_port)
    pipe = decm = vtkQuadricDecimation(input_connection=pipe.output_port)
    pipe.SetTargetReduction(target_reduction)
    pipe = vtkPolyDataWriter(input_connection=pipe.output_port)
    pipe.file_name = dst
    pipe.SetFileTypeToBinary()
    pipe.Update()

    logger.info(f"{src} actual reduction: {decm.actual_reduction}")

    return dst
