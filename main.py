# %%
import asyncio
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor

import igl
import numpy as np
import scipy.sparse as sp
import vtk
from scipy.spatial import cKDTree
from tqdm import tqdm

# %%
globals().setdefault('rwis', vtk.vtkInteractorStyleSwitch())
rwis: vtk.vtkInteractorStyleSwitch

globals().setdefault('rwi', vtk.vtkRenderWindowInteractor())
rwi: vtk.vtkRenderWindowInteractor

globals().setdefault('rw', vtk.vtkRenderWindow())
rw: vtk.vtkRenderWindow

globals().setdefault('r', vtk.vtkRenderer())
r: vtk.vtkRenderer

rwi.SetInteractorStyle(rwis)
rwi.SetRenderWindow(rw)
rw.AddRenderer(r)

rwis.SetCurrentStyleToTrackballCamera()


async def loop():
    rwi.Initialize()
    rwi.done = False
    while not rwi.done:
        rwi.ProcessEvents()
        rwi.Render()
        try:
            await asyncio.sleep(1 / 30)
        except asyncio.CancelledError:
            rwi.done = True


if f := globals().get('_future'):
    f.cancel()

_future = asyncio.ensure_future(loop())

# %%

r.UseDepthPeelingOn()
r.UseFXAAOn()
r.UseSSAOOn()


# %%
pipe = vtk.vtkSTLReader(file_name='devel/pancake.stl')
pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
pipe = vtk.vtkDecimatePro(input_connection=pipe.output_port)
pipe.preserve_topology = True
pipe.target_reduction = 0.2
pipe = original = vtk.vtkCleanPolyData(input_connection=pipe.output_port)

pipe = fnorms = vtk.vtkTriangleMeshPointNormals(input_connection=pipe.output_port)
pipe = fedges = vtk.vtkExtractEdges(input_connection=pipe.output_port)
pipe.use_all_points = True

pipe.Update()

# %%

verts = fnorms.output.points
norms = fnorms.output.point_data['Normals']
polys = np.reshape(fnorms.output.polys.connectivity_array, (-1, 3), order='C')
edges = np.reshape(fedges.output.lines.connectivity_array, (-1, 2), order='C')

N = len(verts)
K = 2  # number of links for each vertex
r_max = 0.30

print(verts.shape)

assert K == 2, "K > 2 causes runaway where relation is asymmetric."

links = []
tree = cKDTree(verts)


def f(vert, norm):
    idxs = tree.query_ball_point(vert, r_max)
    idxs = np.asarray(idxs)

    vecs = verts[idxs] - vert

    mask = vecs @ norm <= 0
    idxs = np.compress(mask, idxs, axis=0)
    vecs = np.compress(mask, vecs, axis=0)

    vecs *= np.expand_dims(np.exp(norms[idxs] @ norm), 1)

    _, subs = cKDTree(vecs).query([0, 0, 0], k=K)
    subs = np.atleast_1d(subs)
    return np.take(idxs, subs)

pdata = vtk.vtkPolyData()
pdata.points = verts
pdata.SetLines(vtk.vtkCellArray())

m = vtk.vtkPolyDataMapper()
m.SetInputData(pdata)
a = vtk.vtkActor()
a.SetMapper(m)
a.GetProperty().color = (1.0, 1.0, 1.0)
a.GetProperty().opacity = 0.1

r.RemoveAllViewProps()
r.AddActor(a)

# with ThreadPoolExecutor() as tx:
r.RemoveAllViewProps()
r.AddActor(a)

with ProcessPoolExecutor() as px:
    idxss = []
    for src, *dsts in tqdm(px.map(f, verts, norms, chunksize=32), total=N):
        idxss.append((src, *dsts))
        for dst in dsts:
            pdata.InsertNextCell(vtk.VTK_LINE, 2, [src, dst])
        m.Modified()
        r.Modified()
        r.ResetCameraClippingRange()
        rwi.Modified()
        await asyncio.sleep(0)

idxss = np.array(idxss)


# %%

verts = np.array(fnorms.output.points)

verts -= np.mean(verts, axis=0)
verts /= np.sqrt(np.mean(verts * verts))

free = np.linalg.norm(verts, axis=1) < 2.5

verts -= np.mean(verts[~free], axis=0)
verts /= np.sqrt(np.mean(verts[~free] * verts[~free]))

pdata.points = verts

vec = np.subtract(verts[links], np.expand_dims(verts, 1))
target_lengths = np.linalg.norm(vec, axis=2)

# L = igl.cotmatrix(verts, polys)

chords = verts[idxss, :].reshape((len(verts), -1))
L = igl.cotmatrix(chords, polys)

# %%

free = ~free


# %%
rate = 2.04

# M = igl.massmatrix(np.asarray(verts), np.asarray(polys), igl.MASSMATRIX_TYPE_BARYCENTRIC)

chords = verts[idxss, :].reshape((len(verts), -1))
M = igl.massmatrix(np.asarray(chords), np.asarray(polys), igl.MASSMATRIX_TYPE_BARYCENTRIC)

Q = M - rate * L
B = M * verts

solver = sp.linalg.factorized(Q[free, :][:, free])
verts[free, :] = solver(B[free, :] - Q[free, :][:, ~free] @ verts[~free, :])
# verts[free, :] = solver(B[free, :] - Q[free, :][:, ~free] @ verts[~free, :])

# solver = sp.linalg.factorized(Q)
# verts[...] = solver(B)

# Gauss-Newton Distance Constraints

vec = np.subtract(verts[links], np.expand_dims(verts, 1))
rad = np.linalg.norm(vec, axis=-1)

drad = (target_lengths - rad) / 2 * 0.01
dvec = (np.linalg.pinv(vec) @ (rad * drad)[..., np.newaxis]).squeeze(-1)
verts -= dvec

# verts -= np.mean(verts, axis=0)
# verts /= np.sqrt(np.mean(verts * verts))

verts -= np.mean(verts[~free], axis=0)
verts /= np.sqrt(np.mean(verts[~free] * verts[~free]))


pdata.points = verts

