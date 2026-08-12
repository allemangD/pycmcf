from __future__ import annotations

import logging
import random
from logging import INFO
from pathlib import Path
from typing import TYPE_CHECKING

import igl
import numpy as np
import torch
from deformetrica.core import GpuMode
from deformetrica.core.model_tools.attachments import MultiObjectAttachment
from deformetrica.core.model_tools.deformations import Exponential
from deformetrica.core.model_tools.manifolds.exponential_interface import ExponentialInterface
from deformetrica.core.models.deterministic_atlas import DeterministicAtlas
from deformetrica.core.models.model_functions import create_regular_grid_of_points
from deformetrica.core.observations import SurfaceMesh
from deformetrica.in_out.dataset_functions import create_dataset
from deformetrica.support import utilities
from deformetrica.support.kernels.keops_kernel import KeopsKernel
from scipy.optimize import minimize

if TYPE_CHECKING:
    from scipy.optimize._minimize import _MinimizeOptions

logging.basicConfig(level=INFO)

output_dir = Path("./output-rework")
output_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

DIMENSION = 3
GPU_MODE = GpuMode.NONE
DEVICE, DEVICE_ID = utilities.get_best_device(gpu_mode=GPU_MODE)
TIMESTEPS = 10

template_specifications = {
    "surf": {
        "kernel_width": 0.10,
        "attachment_type": "extendedvarifold",
        "deformable_object_type": "surfacemesh",
        "filename": "../data/inner.vtk",
        "kernel_device": "auto",
        "kernel_type": "keops",
        "noise_std": 0.005,
        "noise_variance_prior_normalized_dof": 0.01,
        "noise_variance_prior_scale_std": None,
    }
}

dataset = create_dataset(
    template_specifications,
    dimension=DIMENSION,
    dataset_filenames=[[{"surf": "../data/outer.vtk"}]],
    subject_ids=["outer"],
    visit_ages=[[]],
)
assert dataset.is_cross_sectional(), "cannot estimate an atlas from a non-cross-sectional dataset."

model = DeterministicAtlas(
    template_specifications,
    dataset.number_of_subjects,
    deformation_kernel_type="keops",
    deformation_kernel_width=0.15,
    dense_mode=True,
    dimension=DIMENSION,
    gpu_mode=GpuMode.NONE,
    initial_control_points=None,
    initial_cp_spacing=0.15,
    initial_momenta=None,
    freeze_control_points=True,
    freeze_momenta=False,
    freeze_template=True,
    number_of_processes=1,
    number_of_time_points=TIMESTEPS,
    process_per_gpu=1,
    shoot_kernel_type="keops",
    smoothing_kernel_width=0.15,
    tensor_integer_type=torch.LongTensor,
    tensor_scalar_type=torch.FloatTensor,
    use_rk2_for_flow=False,
    use_rk2_for_shoot=False,
    use_sobolev_gradient=True,
)
model.initialize_noise_variance(dataset)

verbose = 2
options: _MinimizeOptions = dict(
    maxiter=100 + 10,  # total iterations
    maxls=10,  # line search
    ftol=1e-3,  # convergence tolerance
    maxcor=1,  # l-bfcg-c memory
)


exp_kernel = KeopsKernel(
    gpu_mode=GPU_MODE,
    kernel_width=0.15,
)

sob_kernel = KeopsKernel(
    gpu_mode=GPU_MODE,
    kernel_width=0.15,
)

var_kernel = KeopsKernel(
    gpu_mode=GPU_MODE,
    kernel_width=0.15,
)


source: SurfaceMesh = model.template[0]
target: SurfaceMesh = dataset.deformable_objects[0][0][0]
noise_std = 0.005

pts_ = source.points
print("source:", source.points.shape)
print("target:", target.points.shape)

bbox = np.stack([np.min(pts_, axis=0), np.max(pts_, axis=0)], axis=1)
print(bbox.shape)
print(bbox)

cps_ = create_regular_grid_of_points(bbox, 0.15, DIMENSION)

# cps_ = pts_  # dense mode

# _, _, cps_ = igl.random_points_on_mesh(10000, source.points, source.connectivity, seed=1)
# cps_ = np.array(random.sample(list(source.points), k=len(source.points) // 5))

parameters_initial = {
    # "landmark_points": pts_,
    # "control_points": cps_,
    "momenta": np.zeros_like(cps_),
}

parameters_shape = {key: value.shape for key, value in parameters_initial.items()}
parameters_order = [key for key in parameters_initial]


def serialize(parameters):
    return np.concatenate([parameters[key].flatten() for key in parameters_order])


def deserialize(x):
    parameters = {}
    cursor = 0
    for key in parameters_order:
        length = np.prod(parameters_shape[key])
        parameters[key] = x[cursor : cursor + length].reshape(parameters_shape[key])
        cursor += length
    return parameters


def cost_with_grad(x):
    effects = deserialize(x)

    pts0 = torch.from_numpy(pts_).to(DEVICE, torch.float32).requires_grad_(False)
    cps0 = torch.from_numpy(cps_).to(DEVICE, torch.float32).requires_grad_(False)

    # pts0 = (
    #     torch.from_numpy(effects["landmark_points"]).to(DEVICE, torch.float32).requires_grad_(True)
    # )
    # cps0 = (
    #     torch.from_numpy(effects["control_points"]).to(DEVICE, torch.float32).requires_grad_(True)
    # )
    mom0 = torch.from_numpy(effects["momenta"]).to(DEVICE, torch.float32).requires_grad_(True)

    dt = 1.0 / TIMESTEPS
    series = [(cps0, mom0, pts0)]
    for _ in range(TIMESTEPS):
        cps, mom, pts = series[-1]
        cps, mom, pts = (
            cps + dt * exp_kernel.convolve(cps, cps, mom),
            mom - dt * exp_kernel.convolve_gradient(mom, cps),
            pts + dt * exp_kernel.convolve(pts, cps, mom),
        )
        series.append((cps, mom, pts))

    distance = MultiObjectAttachment.extended_varifold_distance(
        pts,
        source,
        target,
        var_kernel,
    )
    attachment = torch.sum(distance / (noise_std**2))
    regularity = torch.sum(mom * exp_kernel.convolve(cps, cps, mom))

    total_loss = attachment + regularity
    total_loss.backward()

    gradient = {
        # "landmark_points": (
        #     sob_kernel.convolve(
        #         pts0.detach(),
        #         pts0.detach(),
        #         pts0.grad.detach(),
        #     )
        #     .cpu()
        #     .numpy()
        # ),
        # "control_points": cps0.grad.detach().cpu().numpy(),
        "momenta": mom0.grad.detach().cpu().numpy(),
    }

    logger.info(f"{total_loss = :.3e} ({attachment = :.3e}, {regularity = :.3e})")
    return total_loss.detach().cpu().numpy(), serialize(gradient)


result = minimize(
    cost_with_grad,
    serialize(parameters_initial),
    method="L-BFGS-B",
    jac=True,
    options=options,
)
msg = result.message
if msg == "ABNORMAL_TERMINATION_IN_LNSRCH":
    logger.info(">> Number of line search loops exceeded. Stopping.")
else:
    logger.info(">> " + msg)
res = deserialize(result.x)

pts0 = torch.from_numpy(pts_).to(DEVICE, torch.float32).requires_grad_(False)
cps0 = torch.from_numpy(cps_).to(DEVICE, torch.float32).requires_grad_(False)

# pts0 = (
#     torch.from_numpy(effects["landmark_points"]).to(DEVICE, torch.float32).requires_grad_(True)
# )
# cps0 = (
#     torch.from_numpy(effects["control_points"]).to(DEVICE, torch.float32).requires_grad_(True)
# )
mom0 = torch.from_numpy(res["momenta"]).to(DEVICE, torch.float32).requires_grad_(True)

dt = 1.0 / TIMESTEPS
series = [(cps0, mom0, pts0)]
for _ in range(TIMESTEPS):
    cps, mom, pts = series[-1]
    cps, mom, pts = (
        cps + dt * exp_kernel.convolve(cps, cps, mom),
        mom - dt * exp_kernel.convolve_gradient(mom, cps),
        pts + dt * exp_kernel.convolve(pts, cps, mom),
    )
    series.append((cps, mom, pts))

model.set_fixed_effects(
    {
        "landmark_points": pts0,
        "control_points": cps0,
        "momenta": res["momenta"],
    }
)
model.write(dataset, 0, 0, output_dir)
