import logging
import shutil
from pathlib import Path

from deformetrica.core.estimators import ScipyOptimize
from deformetrica.core.models import DeterministicAtlas
from deformetrica.in_out.dataset_functions import create_dataset

logger = logging.getLogger(__name__)


def register(
    moving: Path,
    fixed: Path,
    *,
    output_dir: Path,
    number_of_time_points=5,
    dense_mode=True,
    use_sobolev=False,
    use_rk2=False,
    convergence_tolerance=1e-6,
    max_iterations=100,
    max_line_search_iterations=20,
    memory_length=20,
    data_kernel_width=4.0,
    data_noise_std=8.5,
    deformation_kernel_width=2.75,
    verbose=2,
):
    """Invoke Deformetrica DeterministicAtlas."""

    TEMPLATE_OPTIONS = {
        "surface": {
            "deformable_object_type": "surfacemesh",
            "filename": str(moving.absolute()),
            "attachment_type": "extendedvarifold",
            "kernel_type": "keops",
            "kernel_width": data_kernel_width,  # mm
            "noise_std": data_noise_std,
        }
    }

    DATASET_OPTIONS = {
        "dataset_filenames": [[{"surface": str(fixed.absolute())}]],
        "subject_ids": ["fixed"],
        "visit_ages": [[]],
    }

    dataset = create_dataset(
        TEMPLATE_OPTIONS,
        dimension={
            "dense_mode": dense_mode,
            "deformation_kernel_type": "keops",
            "dimension": 3,
            "freeze_control_points": False,
            "freeze_momenta": False,
            "freeze_template": False,
            "number_of_time_points": number_of_time_points,
            "use_sobolev_gradient": use_sobolev,
            "use_rk2_for_flow": use_rk2,
            "use_rk2_for_shoot": use_rk2,
            "deformation_kernel_width": deformation_kernel_width,
        }["dimension"],
        **DATASET_OPTIONS,
    )
    assert dataset.is_cross_sectional(), (
        "cannot estimate an atlas from a non-cross-sectional dataset."
    )

    model = DeterministicAtlas(
        TEMPLATE_OPTIONS,
        dataset.number_of_subjects,
        dense_mode=dense_mode,
        deformation_kernel_type="keops",
        dimension=3,
        freeze_control_points=False,
        freeze_momenta=False,
        freeze_template=False,
        number_of_time_points=number_of_time_points,
        use_sobolev_gradient=use_sobolev,
        use_rk2_for_flow=use_rk2,
        use_rk2_for_shoot=use_rk2,
        deformation_kernel_width=deformation_kernel_width,
    )
    model.initialize_noise_variance(dataset)

    estimator = ScipyOptimize(
        model,
        dataset,
        output_dir=str(output_dir.absolute()),
        state_file=str(output_dir.joinpath("deformetrica-state.p").absolute()),
        convergence_tolerance=convergence_tolerance,
        max_iterations=max_iterations,
        max_line_search_iterations=max_line_search_iterations,
        memory_length=memory_length,
        optimization_method_type="scipylbfgs",
        optimized_log_likelihood="complete",
        verbose=verbose,
    )

    estimator.update()
    estimator.write()

    recon = output_dir.joinpath(
        "DeterministicAtlas__Reconstruction__surface__subject_fixed.vtk"
    )
    assert recon.exists()

    fixed = fixed.with_stem(f"{fixed.stem}-recon")
    shutil.copy(recon, fixed)

    return moving, fixed
