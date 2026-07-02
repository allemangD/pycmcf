import halide as hl
import numpy as np

u = hl.InputBuffer(hl.Float(32), 2)
A = hl.InputBuffer(hl.Float(32), 2)
v = hl.InputBuffer(hl.Float(32), 2)


# computing (i,m) @ (m,n) @ (n,j) -> (i,j)

r = hl.OutputBuffer(hl.Float(32), 2)
i = hl.Var("i")
j = hl.Var("j")
mn = hl.RDom(A)

r[i, j] = hl.sum(mn, u[mn.x, i] * A[mn.y, mn.x] * v[j, mn.y])

adj = hl.propagate_adjoints(r)
du = adj[u, -1]
dv = adj[v, -1]

out = hl.Pipeline([r, du, dv]).compile_to_callable([u, A, v])


def wrap(u, A, v):
    i, m = u.shape
    n, j = v.shape
    assert A.shape == (m, n)

    r = np.empty((i, j), np.float32)
    du = np.empty((i, m), np.float32)
    dv = np.empty((n, j), np.float32)

    out(u, A, v, r, du, dv)

    return r, du, dv


u = np.random.normal(size=(1, 3)).astype(np.float32)
A = np.random.normal(size=(3, 4)).astype(np.float32)
v = np.random.normal(size=(4, 1)).astype(np.float32)

r, du, dv = wrap(u, A, v)
print(r)
print(du)
print(dv)

import torch

print()

u = torch.tensor(u, requires_grad=True)
A = torch.tensor(A)
v = torch.tensor(v, requires_grad=True)
r = u @ A @ v
r.backward()

print(r.detach().numpy())
print(u.grad.detach().numpy())
print(v.grad.detach().numpy())

