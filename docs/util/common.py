"""Shared helpers for the docs/util scripts: puts src/ on the path and reads VTK meshes."""

import sys
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOLegacy import vtkPolyDataReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def read_vtk(path):
    """(points (n, 3) float64, faces (m, 3) int64) of a triangle mesh."""
    r = vtkPolyDataReader()
    r.SetFileName(str(path))
    r.Update()
    pd = r.GetOutput()
    V = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    F = vtk_to_numpy(pd.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return V, F.astype(np.int64)


def percentiles(a, qs=(50, 95)):
    return "/".join(f"{np.percentile(np.abs(a), q):.2f}" for q in qs)
