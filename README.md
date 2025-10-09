# PyCMCF

Python implementation of Conformalized Mean Curvature Flow as described by Kazhdan, Solomon, Ben-Chen (2012). Based on
C++ implementation from https://github.com/allemangd/cmcf.

https://doi.org/10.48550/arXiv.1203.6819

### Installation

Install via pip:

```
pip install git+https://github.com/allemangd/pycmcf.git
```

Or, for local development:

```
git clone https://github.com/allemangd/pycmcf.git
cd pycmcf
pip install -e .
```

### Future work and notes

- Alternative solvers. Each stage of flow performs a direct sparse linear solve via `scipy.sparse.linalg.spsolve`. It is
  very possible that faster solution could be achieved by tuning parameters of some iterative solver, but the direct
  solver is fast enough that I haven't bothered to work on this yet. Alternatively, one could install `scikit-umfpack`
  via conda for potential speedup with the direct solver.

- Better rate scaling and tuning. From trial-and-error, to get cortexes the ellipsoid domain, parameters
  `-n 8 -r 1e-2 -f 4` seem to do fairly well, but this destroys all the fine detail in the first few stages of flow. For
  visualization and shape representation purposes, a much slower rate (on the order of `1e-4`) is better.

- The script `extras/f3d-render.py` provides some utility to render out batches of meshes to images for animations.

- Boundary conditions. The paper https://doi.org/10.48550/arXiv.1203.6819 describes how mesh boundaries may be fixed,
  but I believe this requires modifying the mass matrix `M` and stiffness matrix `L` and is not implemented here. As
  such this implementation currently only supports closed meshes with spherical topology. In the context of cortex
  research, I believe that implementing these boundary conditions will support a superior disc-based representation.

### Command-Line Interface

```
usage: pycmcf [-h] -n STEPS -r RATE_0 [-f RATE_FACTOR] [--strip_normals | --no-strip_normals] 
    [-o OUTPUT] [--progress | --no-progress] mesh

positional arguments:
  mesh

options:
  -h, --help            show this help message and exit
  -n, --steps STEPS     Number of steps to run CMCF.
  -r, --rate RATE_0     Initial timestep for CMCF flow.
  -f, --rate-factor RATE_FACTOR
                        Common factor for rate acceleration. Set to 1 to use constant rate.
  --strip_normals, --no-strip_normals
                        Remove normals from mesh (default).
  -o, --output OUTPUT   Results are placed in this directory with the name format: <filename>.cmcf-<stage>.vtk
  --progress, --no-progress
                        Show a progress bar (default).
```

### Python Interface

```python
from pycmcf.method import flow

verts, faces = ...  # acquire mesh data

for stage in flow(verts, faces, steps=25, rate_0=1e-2, rate_common_factor=4):
    ... # visualize

final = stage
... # work in ellipsoid domain
```
