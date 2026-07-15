import logging
from pathlib import Path

import vtk

from deformetrica.core.estimators import ScipyOptimize
from deformetrica.core.models import DeterministicAtlas
from deformetrica.in_out.dataset_functions import create_dataset

logging.basicConfig(level=logging.INFO)

PATHS = {
    "moving": Path("data/oasis4-inner.vtk"),
    "fixed": Path("data/oasis4-outer.vtk"),
}

DEFORMETRICA_OUTPUTS = Path("data/outputs/")
DEFORMETRICA_OUTPUTS.mkdir(exist_ok=True)

DECIMATE = 0.95

# downsample large meshes
for key in list(PATHS):
    src = PATHS[key]
    dst = src.with_stem(f"{src.stem}-decimated")
    print(f"downsampling {src} -> {dst}")

    pipe = vtk.vtkPolyDataReader()
    pipe.file_name = src
    pipe = vtk.vtkTriangleFilter(input_connection=pipe.output_port)
    pipe = vtk.vtkQuadricDecimation(input_connection=pipe.output_port)
    pipe.SetTargetReduction(DECIMATE)
    pipe = vtk.vtkPolyDataWriter(input_connection=pipe.output_port)
    pipe.file_name = dst
    pipe.SetFileTypeToBinary()
    pipe.Update()

    PATHS[key] = dst

# invoke deformetrica
MODEL_OPTIONS = {
    "dense_mode": True,
    "deformation_kernel_type": "keops",
    "dimension": 3,
    "freeze_control_points": False,
    "freeze_momenta": False,
    "freeze_template": False,
    "number_of_time_points": 5,
    "use_sobolev_gradient": False,
    "use_rk2_for_flow": False,
    "use_rk2_for_shoot": False,
}

ESTIMATOR_OPTIONS = {
    "convergence_tolerance": 1e-6,
    "max_iterations": 1,
    "max_line_search_iterations": 20,
    "memory_length": 20,
    "optimization_method_type": "scipylbfgs",
    "optimized_log_likelihood": "complete",
    "verbose": 2,
}

TEMPLATE_OPTIONS = {
    "surface": {
        "deformable_object_type": "surfacemesh",
        "filename": str(PATHS["moving"].absolute()),
        "attachment_type": "extendedvarifold",
        "kernel_type": "keops",
        "kernel_width": 4.0,  # mm
        "noise_std": 8.5,
    }
}

DATASET_OPTIONS = {
    "dataset_filenames": [[{"surface": str(PATHS["fixed"].absolute())}]],
    "subject_ids": ["fixed"],
    "visit_ages": [[]],
}

print("loading deformetrica dataset")
dataset = create_dataset(
    TEMPLATE_OPTIONS,
    dimension=MODEL_OPTIONS["dimension"],
    **DATASET_OPTIONS,
)
assert dataset.is_cross_sectional(), (
    "cannot estimate an atlas from a non-cross-sectional dataset."
)

print("loading deformetrica model")
model = DeterministicAtlas(
    TEMPLATE_OPTIONS,
    dataset.number_of_subjects,
    **MODEL_OPTIONS,
)
model.initialize_noise_variance(dataset)

print("loading deformetrica estimator")
estimator = ScipyOptimize(
    model,
    dataset,
    output_dir=str(DEFORMETRICA_OUTPUTS.absolute()),
    state_file=str(DEFORMETRICA_OUTPUTS.joinpath("deformetrica-state.p").absolute()),
    **ESTIMATOR_OPTIONS,
)

print("registering")
estimator.update()
print("writing outputs")
estimator.write()

PATHS["fixed"] = DEFORMETRICA_OUTPUTS.joinpath(
    "DeterministicAtlas__Reconstruction__surface__subject_fixed.vtk"
)
assert PATHS["fixed"].exists()
