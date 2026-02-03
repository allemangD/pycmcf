import contextlib
import weakref

import f3d
import numpy as np
import vtk
from scipy.spatial import cKDTree
from tqdm import tqdm

f3d.Engine.autoload_plugins()


@contextlib.contextmanager
def Engine():
    instance = f3d.Engine.create()
    yield weakref.ref(instance)


pipe = vtk.vtkSTLReader(file_name='devel/pancake.stl')
pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
pipe = original = vtk.vtkCleanPolyData(input_connection=pipe.output_port)

pipe = fnorms = vtk.vtkTriangleMeshPointNormals(input_connection=pipe.output_port)
pipe = fedges = vtk.vtkExtractEdges(input_connection=pipe.output_port)
pipe.UseAllPointsOn()

pipe.Update()

verts = fnorms.output.points
norms = fnorms.output.point_data['Normals']
polys = np.reshape(fnorms.output.polys.connectivity_array, (-1, 3), order='C')
edges = np.reshape(fedges.output.lines.connectivity_array, (-1, 2), order='C')

# %%

N = len(verts)
r_max = 0.10
tree = cKDTree(verts)

indexes = tree.query_ball_point(verts, r_max)

links = []

for vert, norm, idxs in tqdm(zip(verts, norms, indexes), total=N):
    side_mask = norms[idxs] @ norm < 0  # restrict to points whose normals point "down".
    idxs = np.compress(side_mask, idxs)
    # side_mask tends to remove more vertices, so do that first

    down_mask = (verts[idxs] - vert) @ norm < 0  # restrict to points whose location is "down".
    idxs = np.compress(down_mask, idxs)

    fwd = idxs  # need to be able to convert the query result back to global indexes.
    _, idxs = cKDTree(verts[idxs]).query(vert, k=3)
    links.append(np.take(fwd, idxs))

pdata = vtk.vtkPolyData()
pdata.SetPoints(original.output.GetPoints())
pdata.SetPolys(original.output.GetPolys())
pdata.SetLines(vtk.vtkCellArray())

for src, dsts in enumerate(links):
    for dst in dsts:
        pdata.InsertNextCell(vtk.VTK_LINE, 2, [src, dst])

writer = vtk.vtkPolyDataWriter(file_name='devel/with-links.vtk', input_data=pdata)
writer.Update()

# %%


# %%

import f3d

try:
    eng = f3d.Engine.create()

    mesh = f3d.Mesh(
        points=verts.ravel(),
        face_indices=polys.ravel(),
        face_sides=[3] * len(polys)
    )
    eng.scene.add(mesh)

    # eng.options['render.show_edges'] = True
    # eng.options['render.line_width'] = 1

    eng.options['render.effect.blending.enable'] = True
    eng.options['model.color.opacity'] = 0.4

    # eng.options['render.effect.ambient_occlusion'] = True

    eng.options['interactor.trackball'] = True
    eng.options['scene.up_direction'] = [0, 1, 1]

    eng.interactor.start()
finally:
    del eng

# %%

# igl.cotmatrix()
# igl.massmatrix()

# # %%
#
# edge_affinity = np.sum(np.prod(norms[edges], axis=1), axis=1)
# print(edge_affinity)
#
# # u, v = np.moveaxis(norms[edges], source=1, destination=0)
# # print(np.dot(u, v).shape)
# edge_affinity = np.einsum('ijk -> i', norms[edges])
# print(edge_affinity)
# # out = np.einsum('ij, ik -> i', u, v)
# # print(out.shape)
# # print(u, v)
#
# # verts
#
# # print(edge_affinity)
#
#
# # points: vtk.vtkPoints = pd.GetPoints()
# # print(pd.GetNumberOfPolys())
# #
# # print('normals:')
# # print(fnorms.output.GetNumberOfPolys())
# # print(fnorms.output.GetNumberOfPoints())
# #
# # print('edges:')
# # print(fedges.output.GetNumberOfPolys())
# # print(fedges.output.GetNumberOfPoints())
#
# # points_vals: vtk.vtkDataArray = points.GetData()
# # print(np.array(points_vals))
#
# # print(np.array(pd.GetPolys().GetConnectivityArray())[:10])
# # trias = np.array(pd.GetPolys().GetConnectivityArray()).reshape((3, -1), order='F')
# # print(trias)
#
# # print(pd.GetLines().GetNumberOfCells())
