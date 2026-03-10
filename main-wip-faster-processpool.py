# %%
import asyncio
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from os import cpu_count

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

# r.UseDepthPeelingOn()
# r.UseFXAAOn()
# r.UseSSAOOn()

r.UseDepthPeelingForVolumesOff()
r.SetMaximumNumberOfPeels(2)
r.UseFXAAOff()
r.UseSSAOOff()




# %%

outer = vtk.vtkPolyDataReader(
    file_name='devel/OASIS4-00010_0102_00-00_BRAIN-T1-SYNTH-3D-SYNTH-PRE_hdrfix_n4_reg_haca3_lesionfilled_slant_wlesions_surface-outer.vtk')
inner = vtk.vtkPolyDataReader(
    file_name='devel/OASIS4-00010_0102_00-00_BRAIN-T1-SYNTH-3D-SYNTH-PRE_hdrfix_n4_reg_haca3_lesionfilled_slant_wlesions_surface-inner.vtk')

# pipe = vtk.vtkSTLReader(file_name='devel/oasout')
pipe = vtk.vtkAppendPolyData()
pipe.AddInputConnection(outer.output_port)
pipe.AddInputConnection(inner.output_port)
pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
# pipe = vtk.vtkDecimatePro(input_connection=pipe.output_port)
# pipe.preserve_topology = True
# pipe.target_reduction = 0.2
pipe = original = vtk.vtkCleanPolyData(input_connection=pipe.output_port)

pipe = fnorms = vtk.vtkTriangleMeshPointNormals(input_connection=pipe.output_port)
pipe = fedges = vtk.vtkExtractEdges(input_connection=pipe.output_port)
pipe.use_all_points = True

pipe.Update()

# %%

m = vtk.vtkPolyDataMapper()
m.SetInputData(pipe.output)
a = vtk.vtkActor()
a.SetMapper(m)
a.GetProperty().color = (1.0, 1.0, 1.0)
a.GetProperty().opacity = 0.1

r.RemoveAllViewProps()
r.AddActor(a)


# %%

verts = fnorms.output.points
norms = fnorms.output.point_data['Normals']
polys = np.reshape(fnorms.output.polys.connectivity_array, (-1, 3), order='C')
edges = np.reshape(fedges.output.lines.connectivity_array, (-1, 2), order='C')

N = len(verts)
K = 2  # number of links for each vertex
# r_max = 0.30
r_max = 10

print(verts.shape)

assert K == 2, "K > 2 causes runaway where relation is asymmetric."

links = []
tree = cKDTree(verts)

print('beginning query')


def f(vert, norm):
    idxs = tree.query_ball_point(vert, r_max)
    idxs = np.asarray(idxs)

    vecs = verts[idxs] - vert

    mask = vecs @ norm <= 0
    idxs = np.compress(mask, idxs, axis=0)
    vecs = np.compress(mask, vecs, axis=0)

    if len(idxs) < K:
        return None

    vecs *= np.expand_dims(np.exp(norms[idxs] @ norm), 1)

    _, subs = cKDTree(vecs).query([0, 0, 0], k=K)
    subs = np.atleast_1d(subs)

    if len(subs) < K:
        return None

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

with ProcessPoolExecutor(4) as px:
    idxss = []
    for link in tqdm(px.map(f, list(verts), list(norms), chunksize=2048), total=N):
        if link is None:
            continue
        src, *dsts = link
        idxss.append((src, *dsts))
        for dst in dsts:
            pdata.InsertNextCell(vtk.VTK_LINE, 2, [src, dst])
        # m.Modified()
        # r.Modified()
        # r.ResetCameraClippingRange()
        # rwi.Modified()
        # await asyncio.sleep(0)

idxss = np.array(idxss)
print(idxss.shape)


# %%
def f(vert): return vert


with ProcessPoolExecutor() as px:
    x = list(tqdm(px.map(f, list(verts), chunksize=4096)))
    # for k in px.map(f, verts):
    #     print(k)

# %%

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

idxss = []
for link in idxss:
    src, *dsts = link
    for dst in dsts:
        pdata.InsertNextCell(vtk.VTK_LINE, 2, [src, dst])

# pdata.Modified()
# m.Update()
# r.Modified()



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

Lv = igl.cotmatrix(verts, polys)

chords = verts[idxss, :].reshape((len(verts), -1))
Lc = igl.cotmatrix(chords, polys)


# %%
r.ResetCamera()

# %%
rate = 0.02

M = igl.massmatrix(np.asarray(verts), np.asarray(polys), igl.MASSMATRIX_TYPE_BARYCENTRIC)

Q = M - rate * Lc
# Q = sp.eye(N) + 1 * M - rate * Lc
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


# %%
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future
from scipy.spatial import cKDTree
from os import cpu_count
from tqdm import tqdm

N = 12345
arr = np.zeros((N, 3), dtype=np.float64)
tree = cKDTree(arr)

res = np.zeros((N, 2), dtype=np.uint32)
PITCH = 1024


def f(root):
    _, r = tree.query(arr[root:][:PITCH], k=2)
    return r


workers = min(cpu_count() - 1, 1)
with ProcessPoolExecutor(workers) as px:
    for out in px.map(f, range(0, N, PITCH)):
        pass

    # for out in px.map(f, np.array_split(arr, workers), np.array_split(res, workers)):
    #     pass

    # print(res)

    # print(len(ou))

    # for q in tqdm(px.map(f, arr[:200000], chunksize=1000)):
    #     pass
    # fut: Future = px.submit(f, 99)
    # print(fut.result())

# %%
verts = fnorms.output.points
norms = fnorms.output.point_data['Normals']
polys = np.reshape(fnorms.output.polys.connectivity_array, (-1, 3), order='C')
edges = np.reshape(fedges.output.lines.connectivity_array, (-1, 2), order='C')

tree = cKDTree(verts)

print(verts.shape)
N = len(verts)

r_max = 10
pitch = 128


def f(idx):
    vs = verts[idx:idx + pitch]
    ns = norms[idx:idx + pitch]

    idxs = tree.query_ball_point(vs, r_max)

    ress = []

    for idx, vert, norm in zip(idxs, vs, ns):
        # vec -= vert
        vec = verts[idx] - vert
        mask = vec @ norm < 0
        idx = np.compress(mask, idx, axis=0)
        vec = np.compress(mask, vec, axis=0)
        if len(vec) < 2:
            ress.append(None)
            continue
        vec *= np.expand_dims(np.exp(norms[idx] @ norm), 1)
        _, sub = cKDTree(vec).query([0,0,0], k=2)
        if len(sub) < 2:
            ress.append(None)
            continue
        ress.append(idx[sub])

    return ress

    # _, idxs = tree.query(batch, k=2)
    # return idxs

    # idxs = tree.query_ball_point(vert, r_max)
    # idxs = np.asarray(idxs)
    #
    # vecs = verts[idxs] - vert
    #
    # mask = vecs @ norm <= 0
    # idxs = np.compress(mask, idxs, axis=0)
    # vecs = np.compress(mask, vecs, axis=0)
    #
    # if len(idxs) < K:
    #     return None
    #
    # vecs *= np.expand_dims(np.exp(norms[idxs] @ norm), 1)
    #
    # _, subs = cKDTree(vecs).query([0, 0, 0], k=K)
    # subs = np.atleast_1d(subs)
    #
    # if len(subs) < K:
    #     return None
    #
    # return np.take(idxs, subs)



with ProcessPoolExecutor() as ex:
    # balls = list(ex.map(f, range(0, N, pitch)))
    links = (link for links in ex.map(f, range(0, N, pitch)) for link in links)
    links = list(tqdm(links, total=N))
        # print(ball)
    # print(balls)
        # break
    # for idxs in ex.map(f, range(0, N, PITCH)):
    #     print(idxs)
    #     pass


# %%
