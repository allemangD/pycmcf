import logging
import shutil
from pathlib import Path

import numpy as np
from deformetrica.core.estimators import ScipyOptimize
from deformetrica.core.estimators.scipy_optimize import logger
from deformetrica.core.models import DeterministicAtlas
from deformetrica.in_out.dataset_functions import create_dataset
from deformetrica.support import utilities
from scipy.optimize import minimize

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
            "freeze_control_points": True,
            "freeze_momenta": False,
            "freeze_template": True,
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
        freeze_control_points=True,
        freeze_momenta=False,
        freeze_template=True,
        number_of_time_points=number_of_time_points,
        use_sobolev_gradient=use_sobolev,
        use_rk2_for_flow=use_rk2,
        use_rk2_for_shoot=use_rk2,
        deformation_kernel_width=deformation_kernel_width,
    )

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

    target = dataset.deformable_objects[0][0]

    estimator.print()

    def _cost_and_derivative(x):
        parameters = {}
        shape = estimator.parameters_shape["momenta"]
        length = np.prod(shape)
        parameters["momenta"] = x[:length].reshape(shape)
        model.set_momenta(parameters["momenta"])

        device, _device_id = utilities.get_best_device(gpu_mode=model.gpu_mode)
        template_data, template_points, control_points, momenta = (
            model._fixed_effects_to_torch_tensors(True, device=device)
        )

        momenta1 = momenta[0]

        model.exponential.set_initial_template_points(template_points)
        model.exponential.set_initial_control_points(control_points)
        model.exponential.set_initial_momenta(momenta1)
        model.exponential.move_data_to_(device=device)
        model.exponential.update()

        deformed_points = model.exponential.get_template_points()
        deformed_data = model.template.get_deformed_data(deformed_points, template_data)
        attachment = -model.multi_object_attachment.compute_weighted_distance(
            deformed_data, model.template, target, model.objects_noise_variance
        )
        regularity = -model.exponential.get_norm_squared()

        total_for_subject = attachment + regularity
        total_for_subject.backward()

        gradient = {}
        gradient["momenta"] = momenta.grad.detach().cpu().numpy()

        attachment, regularity, gradient = (
            attachment.detach().cpu().numpy(),
            regularity.detach().cpu().numpy(),
            gradient,
        )

        logger.info(
            f">> Log-likelihood = {attachment + regularity:.3E} \t [ attachment = {attachment:.3E} ; regularity = {regularity:.3E} ]"
        )

        cost = -attachment - regularity
        gradient = -np.concatenate(
            [gradient[key].flatten() for key in estimator.parameters_order]
        )

        return cost, gradient

    def _callback(_):
        estimator.current_iteration += 1
        estimator.print()

    result = minimize(
        _cost_and_derivative,
        estimator.x0,
        method="L-BFGS-B",
        jac=True,
        callback=_callback,
        options={
            "maxiter": estimator.max_iterations + 10,
            "maxls": estimator.max_line_search_iterations,
            "ftol": estimator.convergence_tolerance,
            "maxcor": estimator.memory_length,
        },
    )
    logger.info(f"minimize terminated: {result.message}")

    model.write(
        estimator.dataset,
        estimator.population_RER,
        estimator.individual_RER,
        estimator.output_dir,
    )

    recon = output_dir.joinpath(
        "DeterministicAtlas__Reconstruction__surface__subject_fixed.vtk"
    )
    assert recon.exists()

    fixed = fixed.with_stem(f"{fixed.stem}-recon")
    shutil.copy(recon, fixed)

    return moving, fixed
