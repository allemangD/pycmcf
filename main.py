import contextlib
import weakref

import f3d
import numpy as np
import vtk

f3d.Engine.autoload_plugins()


@contextlib.contextmanager
def Engine():
    instance = f3d.Engine.create()
    yield weakref.ref(instance)


# with Engine() as eng:
#     eng().scene.add('devel/pancake.stl')
#     eng().interactor.start()


pipe = vtk.vtkSTLReader(file_name='devel/pancake.stl')
pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
pipe = vtk.vtkCleanPolyData(input_connection=pipe.output_port)

pipe = fnorms = vtk.vtkTriangleMeshPointNormals(input_connection=pipe.output_port)
pipe = fedges = vtk.vtkExtractEdges(input_connection=pipe.output_port)
pipe.UseAllPointsOn()

pipe.Update()

verts = fnorms.output.points
norms = fnorms.output.point_data['Normals']
polys = np.reshape(fnorms.output.polys.connectivity_array, (-1, 3), order='C')
edges = np.reshape(fedges.output.lines.connectivity_array, (-1, 2), order='C')

# %%

edge_affinity = np.sum(np.prod(norms[edges], axis=1), axis=1)
print(edge_affinity)

# u, v = np.moveaxis(norms[edges], source=1, destination=0)
# print(np.dot(u, v).shape)
edge_affinity = np.einsum('ijk -> i', norms[edges])
print(edge_affinity)
# out = np.einsum('ij, ik -> i', u, v)
# print(out.shape)
# print(u, v)

# verts

# print(edge_affinity)


# points: vtk.vtkPoints = pd.GetPoints()
# print(pd.GetNumberOfPolys())
#
# print('normals:')
# print(fnorms.output.GetNumberOfPolys())
# print(fnorms.output.GetNumberOfPoints())
#
# print('edges:')
# print(fedges.output.GetNumberOfPolys())
# print(fedges.output.GetNumberOfPoints())

# points_vals: vtk.vtkDataArray = points.GetData()
# print(np.array(points_vals))

# print(np.array(pd.GetPolys().GetConnectivityArray())[:10])
# trias = np.array(pd.GetPolys().GetConnectivityArray()).reshape((3, -1), order='F')
# print(trias)

# print(pd.GetLines().GetNumberOfCells())
