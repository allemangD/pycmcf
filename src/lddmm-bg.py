import numpy as np
import torch
import vtk
from pykeops.torch import LazyTensor
from tqdm import tqdm


# ==========================================
# 1. VTK I/O & GEOMETRY PROCESSING
# ==========================================
def load_and_decimate_surface(filepath, reduction=None):
    """Loads a VTK PolyData file, guarantees triangles, and optionally decimates it."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()

    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputConnection(reader.GetOutputPort())
    tri_filter.Update()
    full_res = tri_filter.GetOutput()

    if reduction is not None and reduction > 0.0:
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputData(full_res)
        decimator.SetTargetReduction(reduction)
        decimator.Update()
        decimated = decimator.GetOutput()
        return full_res, decimated

    return full_res, full_res


def vtk_surface_to_torch(polydata, device="cpu"):
    # Extract vertices
    points_np = np.asarray(polydata.points.data)
    vertices = torch.tensor(points_np, dtype=torch.float32, device=device)

    # Extract faces (assumes pure triangle mesh)
    polys_np = np.asarray(polydata.polys.connectivity_array)
    faces = torch.tensor(polys_np.reshape(-1, 3), dtype=torch.long, device=device)

    # Compute Triangle face normals and areas
    A = vertices[faces[:, 0]]
    B = vertices[faces[:, 1]]
    C = vertices[faces[:, 2]]

    cross = torch.cross(B - A, C - A, dim=1)
    face_areas = 0.5 * torch.norm(cross, dim=1)
    face_normals = cross / (2.0 * face_areas.unsqueeze(1) + 1e-8)

    # Distribute 1/3 of face area to each vertex
    point_areas = torch.zeros(vertices.shape[0], dtype=torch.float32, device=device)
    point_areas.scatter_add_(0, faces[:, 0], face_areas / 3.0)
    point_areas.scatter_add_(0, faces[:, 1], face_areas / 3.0)
    point_areas.scatter_add_(0, faces[:, 2], face_areas / 3.0)

    # Compute Point Normals (area-weighted average of incident face normals)
    weighted_normals = face_normals * face_areas.unsqueeze(1)
    point_normals = torch.zeros_like(vertices)
    for i in range(3):
        point_normals.scatter_add_(
            0, faces[:, i].unsqueeze(1).expand(-1, 3), weighted_normals
        )

    # Normalize the point normals
    point_normals = point_normals / (
        torch.norm(point_normals, dim=1, keepdim=True) + 1e-8
    )

    return vertices, faces, point_normals, point_areas


def save_vtk(filepath, polydata, vertices_tensor):
    """DeepCopies the original polydata geometry and replaces the point coordinates."""
    result = vtk.vtkPolyData()
    result.DeepCopy(polydata)
    np.copyto(result.points.data, vertices_tensor.detach().cpu().numpy())
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(result)
    writer.SetFileTypeToBinary()
    writer.Update()


# ==========================================
# 2. KERNEL FACTORY & SHOOTING
# ==========================================
def make_deformation_kernel(kernel_flags, sigma_base=10.0, gamma_angle=1.0):

    def get_K_base(x, y, sx=None, sy=None, nx=None, ny=None):
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(y[None, :, :])
        D2_ij = ((x_i - y_j) ** 2).sum(-1)

        if "spatially_varying" in kernel_flags and sx is not None and sy is not None:
            s_i = LazyTensor(sx[:, None, :])
            s_j = LazyTensor(sy[None, :, :])
            s2_sum = s_i**2 + s_j**2
            prefactor = (2 * s_i * s_j / s2_sum) ** 1.5
            K_mat = prefactor * (-D2_ij / s2_sum).exp()
        else:
            K_mat = (-D2_ij / (sigma_base**2)).exp()

        if "varifold" in kernel_flags and nx is not None and ny is not None:
            n_i = LazyTensor(nx[:, None, :])
            n_j = LazyTensor(ny[None, :, :])
            dot_ij = (n_i * n_j).sum(-1)
            K_mat = K_mat * (dot_ij**2)

        if "von_mises" in kernel_flags and nx is not None and ny is not None:
            n_i = LazyTensor(nx[:, None, :])
            n_j = LazyTensor(ny[None, :, :])
            dot_ij = (n_i * n_j).sum(-1)
            K_mat = K_mat * ((dot_ij - 1.0) / (gamma_angle**2)).exp()

        return K_mat

    def kernel_eval(x, y, p, nx=None, ny=None, sx=None, sy=None):
        """Cross-evaluates the kernel map from control points (y) onto ambient targets (x)."""
        K_mat = get_K_base(x, y, sx, sy, nx, ny)

        if "normalized" in kernel_flags:
            ones_y = LazyTensor(torch.ones_like(p[None, :, 0:1]))
            rho_x = (K_mat * ones_y).sum(dim=1)

            # Optimization: If evaluating q on q, rho_y is identical to rho_x
            if x is y:
                rho_y = rho_x
            else:
                K_yy = get_K_base(y, y, sy, sy, ny, ny)
                rho_y = (K_yy * ones_y).sum(dim=1)

            rho_x_lz = LazyTensor(rho_x[:, None, :])
            rho_y_lz = LazyTensor(rho_y[None, :, :])
            K_mat = K_mat / (rho_x_lz * rho_y_lz).sqrt()

        p_j = LazyTensor(p[None, :, :])
        return (K_mat * p_j).sum(dim=1)

    return kernel_eval


def hamiltonian_system(q, p, kernel_fn, normals, sigmas):
    v = kernel_fn(q, q, p, normals, normals, sigmas, sigmas)
    H = 0.5 * (p * v).sum()
    gp, gq = torch.autograd.grad(H, (p, q), create_graph=True)
    return gp, -gq


def shoot_euler(
    q0,
    p0,
    kernel_fn,
    normals=None,
    sigmas=None,
    nt=10,
    x0=None,
    x_normals=None,
    x_sigmas=None,
):
    """Integrates the Hamiltonian system using Forward Euler (1st order)."""
    dt = 1.0 / nt
    q, p = q0, p0
    x = x0

    for _ in range(nt):
        # 1. Evaluate derivatives at current state
        dq, dp = hamiltonian_system(q, p, kernel_fn, normals, sigmas)
        if x is not None:
            dx = kernel_fn(x, q, p, x_normals, normals, x_sigmas, sigmas)

        # 2. Update state
        q = q + dt * dq
        p = p + dt * dp
        if x is not None:
            x = x + dt * dx

    if x is not None:
        return q, x
    return q


def shoot_rk2(
    q0,
    p0,
    kernel_fn,
    normals=None,
    sigmas=None,
    nt=10,
    x0=None,
    x_normals=None,
    x_sigmas=None,
):
    """Integrates the Hamiltonian system using RK2 / Midpoint Method (2nd order)."""
    dt = 1.0 / nt
    q, p = q0, p0
    x = x0

    for _ in range(nt):
        # 1. Half-step (Euler) to the midpoint
        k1_q, k1_p = hamiltonian_system(q, p, kernel_fn, normals, sigmas)
        if x is not None:
            k1_x = kernel_fn(x, q, p, x_normals, normals, x_sigmas, sigmas)

        q2 = q + 0.5 * dt * k1_q
        p2 = p + 0.5 * dt * k1_p
        x2 = x + 0.5 * dt * k1_x if x is not None else None

        # 2. Full-step using the derivatives at the midpoint
        k2_q, k2_p = hamiltonian_system(q2, p2, kernel_fn, normals, sigmas)
        if x is not None:
            k2_x = kernel_fn(x2, q2, p2, x_normals, normals, x_sigmas, sigmas)

        q = q + dt * k2_q
        p = p + dt * k2_p
        if x is not None:
            x = x + dt * k2_x

    if x is not None:
        return q, x
    return q


def shoot_rk4(
    q0,
    p0,
    kernel_fn,
    normals=None,
    sigmas=None,
    nt=10,
    x0=None,
    x_normals=None,
    x_sigmas=None,
):
    dt = 1.0 / nt
    q, p = q0, p0
    x = x0

    for _ in range(nt):
        # k1
        k1_q, k1_p = hamiltonian_system(q, p, kernel_fn, normals, sigmas)
        if x is not None:
            k1_x = kernel_fn(x, q, p, x_normals, normals, x_sigmas, sigmas)

        # k2
        q2, p2 = q + 0.5 * dt * k1_q, p + 0.5 * dt * k1_p
        x2 = x + 0.5 * dt * k1_x if x is not None else None

        k2_q, k2_p = hamiltonian_system(q2, p2, kernel_fn, normals, sigmas)
        if x is not None:
            k2_x = kernel_fn(x2, q2, p2, x_normals, normals, x_sigmas, sigmas)

        # k3
        q3, p3 = q + 0.5 * dt * k2_q, p + 0.5 * dt * k2_p
        x3 = x + 0.5 * dt * k2_x if x is not None else None

        k3_q, k3_p = hamiltonian_system(q3, p3, kernel_fn, normals, sigmas)
        if x is not None:
            k3_x = kernel_fn(x3, q3, p3, x_normals, normals, x_sigmas, sigmas)

        # k4
        q4, p4 = q + dt * k3_q, p + dt * k3_p
        x4 = x + dt * k3_x if x is not None else None

        k4_q, k4_p = hamiltonian_system(q4, p4, kernel_fn, normals, sigmas)
        if x is not None:
            k4_x = kernel_fn(x4, q4, p4, x_normals, normals, x_sigmas, sigmas)

        # Merge
        q = q + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        p = p + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        if x is not None:
            x = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)

    if x is not None:
        return q, x
    return q


# ==========================================
# 3. AREA-WEIGHTED VARIFOLD ATTACHMENT
# ==========================================
def varifold_attachment(
    q, target_q, q_normals, target_normals, q_areas, target_areas, sigma_W=5.0
):
    """Area-weighted varifold distance for surface point clouds."""

    def inner_prod(x, y, nx, ny, ax, ay):
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(y[None, :, :])
        nx_i = LazyTensor(nx[:, None, :])
        ny_j = LazyTensor(ny[None, :, :])
        ax_i = LazyTensor(ax[:, None, None])
        ay_j = LazyTensor(ay[None, :, None])

        D2 = ((x_i - y_j) ** 2).sum(-1)
        K = (-D2 / (sigma_W**2)).exp()
        dot2 = (nx_i * ny_j).sum(-1) ** 2

        # 1. KeOps reduction over j (returns a PyTorch tensor of shape [N, 1])
        lazy_matrix = K * dot2 * ax_i * ay_j
        row_sums = lazy_matrix.sum(dim=1)

        # 2. PyTorch reduction over i (returns a pure scalar)
        return row_sums.sum()

    return (
        inner_prod(q, q, q_normals, q_normals, q_areas, q_areas)
        + inner_prod(
            target_q,
            target_q,
            target_normals,
            target_normals,
            target_areas,
            target_areas,
        )
        - 2 * inner_prod(q, target_q, q_normals, target_normals, q_areas, target_areas)
    )


# ==========================================
# 4. OPTIMIZATION LOOP
# ==========================================
def fit_lddmm(
    source_q,
    target_q,
    source_normals,
    target_normals,
    source_areas,
    target_areas,
    source_sigmas,
    flags,
    shoot=shoot_rk4,
    num_epochs=20,
    nt=4,
):

    # Enable grad on the initial coordinates so autograd can compute dH/dq
    source_areas = source_areas.clone().requires_grad_(True)
    source_normals = source_normals.clone().requires_grad_(True)
    source_q = source_q.clone().requires_grad_(True)
    p0 = torch.zeros_like(source_q, requires_grad=True)

    optimizer = torch.optim.LBFGS([p0], max_iter=10, line_search_fn="strong_wolfe")
    kernel_fn = make_deformation_kernel(flags, sigma_base=10.0, gamma_angle=1.0)

    pbar = tqdm(range(1, num_epochs + 1), desc="Shooting Optimization")

    def closure():
        optimizer.zero_grad()

        q1 = shoot(source_q, p0, kernel_fn, source_normals, source_sigmas, nt=nt)

        loss_data = varifold_attachment(
            q1,
            target_q,
            source_normals,
            target_normals,
            source_areas,
            target_areas,
            sigma_W=5.0,
        )

        v0 = kernel_fn(
            source_q,
            source_q,
            p0,
            source_normals,
            source_normals,
            source_sigmas,
            source_sigmas,
        )
        loss_reg = 0.5 * (p0 * v0).sum()

        loss = loss_data + 1.0 * loss_reg
        loss.backward()

        pbar.set_postfix(loss=f"{loss.item():.4e}")
        return loss

    for epoch in pbar:
        optimizer.step(closure)
    pbar.close()

    # Pass the kernel_fn back so we can easily advect the ambient mesh with it
    return p0, kernel_fn


# ==========================================
# 5. EXECUTION HARNESS
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    print("Loading and decimating data...")
    source_full, source_dec = load_and_decimate_surface(
        "oasis4-inner.vtk", reduction=0.85
    )
    target_full, target_dec = load_and_decimate_surface(
        "oasis4-outer.vtk", reduction=0.80
    )

    # Decimated variables act as our control points
    source_v, source_f, source_n, source_a = vtk_surface_to_torch(
        source_dec, device=device
    )
    target_v, target_f, target_n, target_a = vtk_surface_to_torch(
        target_dec, device=device
    )

    # Full resolution source variables will act as our ambient space advection target
    source_full_v, _, source_full_n, _ = vtk_surface_to_torch(
        source_full, device=device
    )

    print(f"Control Points (Decimated Source): {source_v.shape}")
    print(f"Target Points (Decimated Target): {target_v.shape}")
    print(f"Full-Res Source to Advect: {source_full_v.shape}")

    source_sigmas = torch.full((source_v.shape[0], 1), 10.0, device=device)
    source_full_sigmas = torch.full((source_full_v.shape[0], 1), 10.0, device=device)

    flags = ["normalized", "varifold"]

    print(f"\nStarting registration with flags: {flags}...")
    p_opt, kernel_fn = fit_lddmm(
        source_v,
        target_v,
        source_n,
        target_n,
        source_a,
        target_a,
        source_sigmas,
        flags,
        shoot=shoot_rk2,
        nt=3,
    )
    print("Optimization complete.")

    print("Shooting full-resolution surface...")
    _, x_deformed = shoot_rk4(
        q0=source_v.clone().requires_grad_(True),
        p0=p_opt,
        kernel_fn=kernel_fn,
        normals=source_n,
        sigmas=source_sigmas,
        nt=8,
        x0=source_full_v,
        x_normals=source_full_n,
        x_sigmas=source_full_sigmas,
    )

    print("Saving assets to disk...")
    save_vtk("decimated_source.vtk", source_dec, source_v)
    save_vtk("decimated_target.vtk", target_dec, target_v)
    save_vtk("full_res_deformed.vtk", source_full, x_deformed)
    print("Done!")
