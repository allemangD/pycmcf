# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import asyncio
import gc

import igl
import numpy as np
import vtk
from scipy.sparse.linalg import cg, LinearOperator
from scipy.spatial import cKDTree
from tqdm import tqdm

# %%
r = vtk.vtkRenderer()

# %%
r.use_ssao = False
r.use_depth_peeling = True
r.use_fxaa = True


# %%
async def _loop():
    rwi = vtk.vtkRenderWindowInteractor()
    rwi.SetRenderWindow(rw := vtk.vtkRenderWindow())
    rw.AddRenderer(r)

    rwi.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

    rwi.Initialize()

    while not rwi.done:
        # print('running!')
        rwi.ProcessEvents()
        rwi.Render()
        await asyncio.sleep(1 / 20)

    del rw
    del rwi

    gc.collect()


_loop_task = asyncio.ensure_future(_loop())


# %%
pipe = vtk.vtkSTLReader(file_name='devel/pancake.stl')
pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
pipe = original = vtk.vtkCleanPolyData(input_connection=pipe.output_port)

pipe = fnorms = vtk.vtkTriangleMeshPointNormals(input_connection=pipe.output_port)
pipe = fedges = vtk.vtkExtractEdges(input_connection=pipe.output_port)
pipe.use_all_points = True

pipe.Update()

r.RemoveAllViewProps()

m = vtk.vtkPolyDataMapper()
a = vtk.vtkActor(mapper=m)
r.AddActor(a)

# %%
a.GetProperty().opacity = 0.2
a.GetProperty().line_width = 1
a.GetProperty().edge_color = (1.0, 1.0, 1.0)

# %%

verts = fnorms.output.points
norms = fnorms.output.point_data['Normals']
polys = np.reshape(fnorms.output.polys.connectivity_array, (-1, 3), order='C')
edges = np.reshape(fedges.output.lines.connectivity_array, (-1, 2), order='C')

print(verts.shape)

N = len(verts)
K = 2  # number of links for each vertex
r_max = 0.30

links = []
tree = cKDTree(verts)
indexes = tree.query_ball_point(verts, r_max)

for vert, norm, idxs in tqdm(zip(verts, norms, indexes), total=N):
    side_mask = norms[idxs] @ norm < 0  # restrict to points whose normals point "down".
    idxs = np.compress(side_mask, idxs)
    # side_mask tends to remove more vertices, so do that first

    down_mask = (verts[idxs] - vert) @ norm < 0  # restrict to points whose location is "down".
    idxs = np.compress(down_mask, idxs)

    fwd = idxs  # need to be able to convert the query result back to global indexes.
    _, idxs = cKDTree(verts[idxs]).query(vert, k=K)
    links.append(np.take(fwd, np.atleast_1d(idxs)))

# %%

pdata = vtk.vtkPolyData()

pdata.SetLines(vtk.vtkCellArray())
for src, dsts in enumerate(links):
    for dst in dsts:
        pdata.InsertNextCell(vtk.VTK_LINE, 2, [src, dst])

verts = np.array(fnorms.output.points)  # copy to avoid overwriting fnorms data
verts -= np.mean(verts, axis=0)
verts /= np.sqrt(np.mean(verts * verts))
pdata.points = verts

L = igl.cotmatrix(verts, polys)

m.input_data = pdata


# %%
rate = 0.1

M = igl.massmatrix(verts, polys, igl.MASSMATRIX_TYPE_BARYCENTRIC)

# verts = spsolve(M - rate * L, M * verts, "MMD_AT_PLUS_A")

Q = M - rate * L
prec = LinearOperator(Q.shape, matvec=Q.diagonal().__rtruediv__)

for i in range(3):
    verts[..., i], code = cg(Q, M * verts[..., i], rtol=1e-5, x0=verts[..., i], M=prec)

verts -= np.mean(verts, axis=0)
verts /= np.sqrt(np.mean(verts * verts))

pdata.points = verts

