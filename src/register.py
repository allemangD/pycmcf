import logging
from logging import INFO
from pathlib import Path

from deformetrica.core import GpuMode
from deformetrica.core.estimators.scipy_optimize import ScipyOptimize
from deformetrica.core.models import DeterministicAtlas
from deformetrica.in_out.dataset_functions import create_dataset
from sympy.printing.pytorch import torch

logging.basicConfig(level=INFO)

output_dir = Path("./output-rework")
output_dir.mkdir(exist_ok=True)

model_options = {
    "initial_cp_spacing": 0.05,
    "smoothing_kernel_width": 0.05,
    "deformation_kernel_width": 0.05,
    "concentration_of_time_points": 10,
    "deformation_kernel_device": "auto",
    "deformation_kernel_type": "keops",
    "dense_mode": False,
    "dimension": 3,
    "downsampling_factor": 1,
    "dtype": "float32",
    "freeze_control_points": False,
    "freeze_momenta": False,
    "freeze_noise_variance": False,
    "freeze_template": True,
    "gpu_mode": GpuMode.NONE,
    "initial_acceleration_variance": None,
    "initial_control_points": None,
    "initial_modulation_matrix": None,
    "initial_momenta": None,
    "initial_time_shift_variance": None,
    "number_of_processes": 1,
    "number_of_sources": None,
    "number_of_time_points": 2,
    "random_seed": None,
    "sobolev_kernel_width_ratio": 1,
    "t0": None,
    "tensor_integer_type": torch.LongTensor,
    "tensor_scalar_type": torch.FloatTensor,
    "use_rk2_for_flow": False,
    "use_rk2_for_shoot": False,
    "use_sobolev_gradient": True,
}

template_specifications = {
    "surf": {
        "kernel_width": 0.05,
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

estimator_options = {
    "convergence_tolerance": 1e-3,
    "freeze_template": False,
    "gpu_mode": GpuMode.NONE,
    "load_state_file": False,
    "max_iterations": 100,
    "max_line_search_iterations": 10,
    "memory_length": 1,
    "optimization_method_type": "scipylbfgs",
    "optimized_log_likelihood": "complete",
    "print_every_n_iters": 1,
    "save_every_n_iters": 50,
    "state_file": output_dir.joinpath("deformetrica-state.p"),
    "verbose": 2,
}

dataset_specifications = {
    "dataset_filenames": [[{"surf": "../data/outer.vtk"}]],
    "subject_ids": ["outer"],
    "visit_ages": [[]],
}

dataset = create_dataset(
    template_specifications,
    dimension=model_options["dimension"],
    **dataset_specifications,
)
assert dataset.is_cross_sectional(), (
    "cannot estimate an atlas from a non-cross-sectional dataset."
)

model = DeterministicAtlas(
    template_specifications,
    dataset.number_of_subjects,
    **model_options,
)
model.initialize_noise_variance(dataset)

estimator = ScipyOptimize(model, dataset, output_dir=output_dir, **estimator_options)
estimator.update()
estimator.write()
