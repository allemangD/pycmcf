from pathlib import Path

import itk
import vtk
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkIOLegacy import VTK_BINARY, vtkPolyDataWriter

for p in Path("../data/hip").glob("*.gipl.gz"):
    print(p, end='... ')
    im = itk.imread(p)
    im: vtk.vtkImageData = itk.vtk_image_from_image(im)

    pipe = vtkFlyingEdges3D(
        input_data=im,
        compute_normals=True,
        compute_gradients=True,
    )
    pipe.SetValue(0, 128)

    pipe = vtkPolyDataWriter(
        input_connection=pipe.output_port,
        file_name=str(p.with_suffix('').with_suffix(".vtk")),
        file_type=VTK_BINARY,
    )
    pipe.Update()
    print(pipe.file_name)
