# %%
import igl
from PIL.Image import Image
import f3d
import numpy as np
import vtk
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from tqdm import tqdm

f3d.Engine.autoload_plugins()
eng = f3d.Engine.create(offscreen=True)

pipe = vtk.vtkSTLReader(file_name='devel/pancake.stl')
pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
pipe = vtk.vtkDecimatePro(input_connection=pipe.output_port)
pipe.preserve_topology = True
pipe.target_reduction = 0.3
pipe = original = vtk.vtkCleanPolyData(input_connection=pipe.output_port)

pipe = fnorms = vtk.vtkTriangleMeshPointNormals(input_connection=pipe.output_port)
pipe = fedges = vtk.vtkExtractEdges(input_connection=pipe.output_port)
pipe.use_all_points = True

pipe.Update()

verts = fnorms.output.points
norms = fnorms.output.point_data['Normals']
polys = np.reshape(fnorms.output.polys.connectivity_array, (-1, 3), order='C')
edges = np.reshape(fedges.output.lines.connectivity_array, (-1, 2), order='C')

print(verts.shape)

N = len(verts)
K = 1  # number of links for each vertex
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
idxss = np.stack([links, np.expand_dims(np.arange(len(links)), 1)], axis=1)
# chords = verts[idxss, :].reshape((-1, 6))

# L = igl.cotmatrix(np.asarray(chords), np.asarray(polys))
# M = igl.massmatrix(np.asarray(chords), np.asarray(polys), igl.MASSMATRIX_TYPE_BARYCENTRIC)


# %%

pdata = vtk.vtkPolyData()
pdata.SetLines(vtk.vtkCellArray())

for src, dsts in enumerate(links):
    for dst in dsts:
        pdata.InsertNextCell(vtk.VTK_LINE, 2, [src, dst])

m = vtk.vtkPolyDataMapper()
m.input_data = pdata

a = vtk.vtkActor()
a.SetMapper(m)
a.GetProperty().color = (1.0, 1.0, 1.0)
a.GetProperty().opacity = 0.8

r = vtk.vtkRenderer()
r.AddActor(a)

rw = vtk.vtkRenderWindow()
rw.off_screen_rendering = True
rw.AddRenderer(r)

rw.SetSize(1280, 720)

r.use_fxaa = False
r.use_ssao = False

cam: vtk.vtkCamera = r.active_camera
cam.focal_point = (0, 0, 0)
cam.position = (10, 10, 0)
cam.view_angle = 10
cam.view_up = (0, 0, 1)

f = vtk.vtkWindowToImageFilter()
f.input = rw


# %%
verts = np.array(fnorms.output.points)

verts -= np.mean(verts, axis=0)
verts /= np.sqrt(np.mean(verts * verts))

# L = igl.cotmatrix(verts, polys)

chords = verts[idxss, :].reshape((-1, 6))
L = igl.cotmatrix(np.asarray(chords), np.asarray(polys))


# %%
rate = 0.2

# print(chords[0])

# M = igl.massmatrix(verts, polys, igl.MASSMATRIX_TYPE_BARYCENTRIC)

chords = verts[idxss, :].reshape((-1, 6))
M = igl.massmatrix(np.asarray(chords), np.asarray(polys), igl.MASSMATRIX_TYPE_BARYCENTRIC)

from scipy.sparse.linalg import cg, LinearOperator

# verts = spsolve(M - rate * L, M * verts, "MMD_AT_PLUS_A")

Q = M - rate * L
prec = LinearOperator(Q.shape, matvec=Q.diagonal().__rtruediv__)

for i in range(3):
    verts[..., i], code = cg(Q, M * verts[..., i], rtol=1e-5, x0=verts[..., i], M=prec)
    print(i, code)

# scipy.sparse.linalg.cg

# Need to use Gauss-Newton for this as an adjustment. The full workflow ought to be something like:
#
#    target_lengths = ... # compute target chordal lengths from the original (centered-and-scaled) mesh
#
#    verts = scipy.sparse.linalg.spsolve(M - rate * L, M * verts, "MMD_AT_PLUS_A")
#
#    J = ... # Jacobian. The direction of each link. (It is the gradient of the distance function for each link.)
#    # TODO I think J can be computed something simple like "verts[links] - verts" but I'm not sure...

#    # J = verts[links] - verts  # TODO something like this??? And then some suitable reshaping / sparsening.
#    # actual_lengths = np.linalg.norm(J, axis=1)
#    # J /= actual_lengths

#    actual_lengths = ... # The length of each current link.
#    # Use lsqr here for softish constraint, especially when "K" link count is high.
#    # TODO Check if the sign is wrong.
#    verts += scipy.sparse.linalg.spsolve(J, actual_lengths - target_lengths)
#
#    Then recenter and rescale.
#    verts -= np.mean(verts, axis=0)
#    verts /= np.sqrt(np.mean(verts * verts))

print(np.mean(verts, axis=0))
# verts -= np.mean(verts, axis=0)
verts /= np.sqrt(np.mean(verts * verts))

pdata.points = verts
m.Update()

f = vtk.vtkWindowToImageFilter()
f.input = rw
rw.Render()
f.Update()
res: vtk.vtkImageData = f.output

arr = np.reshape(res.point_data.scalars, (*res.dimensions, -1), order='F').squeeze()
arr = np.transpose(arr, (1, 0, 2))
arr = np.flip(arr, 0)

import PIL.Image

PIL.Image.fromarray(arr)

# import wand.image
# img = wand.image.Image.from_array(arr)
# img


# %%
# pdata.GetPolys().SetNumberOfCells(0)
pdata.polys = fnorms.output.polys
pdata.lines = vtk.vtkCellArray()

# %%
ww = vtk.vtkPolyDataWriter(input_data=pdata, file_name='devel/wip.vtk')
ww.Update()
