"""
Baseline backward flow: LDDMM through the saved forward-flow states, last to first.

flow.flow() keeps the triangulation and saves every iterate, so chord i of every state is known
to correspond to chord i of the original. That gives the baseline a ground truth at every step;
the registrations themselves never see vertex indices, only the extended varifold attachment,
as in register().

  - steps: K -> K-1 -> ... -> 0 over the saved states. Each step starts from the previous
    step's result, not from the true state, so errors accumulate as in a real backward flow;
    each result is scored against the true state it targets;
  - features: H0 = [H(u), H(v)] per chord from state 0, carried unchanged on both sides;
  - deformation: dense LDDMM (control points = template vertices), shot with deformetrica's
    Exponential and fitted with scipy L-BFGS-B on the deterministic-atlas objective
    attachment / noise_std^2 + |p|_K^2. The composition of the steps is a diffeomorphism.

Two spaces, compared as an ablation (outputs are tagged with the space):

  - "R6" (baseline): one diffeomorphism of R^6 acting on the chord surface c = (u, v); one
    extended varifold attachment, with unit tangent 2-vectors as tangent planes;
  - "R3" (ablation): one diffeomorphism of R^3 acting on the white and pial surfaces together
    (2n control points); two attachments, white -> white and pial -> pial, both carrying the
    same per-chord H0 pair. The two surfaces must then stay disjoint in R^3 at every state.
"""

import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy as sp
import torch
from deformetrica.core import default
from deformetrica.core.model_tools.attachments.multi_object_attachment import (
    MultiObjectAttachment,
)
from deformetrica.core.model_tools.deformations.exponential import Exponential
from deformetrica.support import utilities
from deformetrica.support.kernels.keops_kernel import KeopsKernel
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter

from chordal.differentiable_geometry import normals_areas_curvature

logger = logging.getLogger(__name__)


def _write(path: Path, V, F, arrays):
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(V, dtype=np.float64), deep=True))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(
            np.arange(0, 3 * len(F) + 1, 3, dtype=np.int64), deep=True
        ),
        numpy_to_vtkIdTypeArray(
            np.ascontiguousarray(F, dtype=np.int64).ravel(), deep=True
        ),
    )
    pd = vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(cells)
    for name, a in arrays.items():
        arr = numpy_to_vtk(np.ascontiguousarray(a, dtype=np.float64), deep=True)
        arr.SetName(name)
        pd.GetPointData().AddArray(arr)
    pipe = vtkPolyDataWriter()
    pipe.file_name = str(path)
    pipe.SetFileTypeToBinary()
    pipe.input_data = pd
    pipe.Update()


def mean_curvature(V, F):
    """H as the attachment computes it in R^3: <(L x) / a, n>."""
    with torch.no_grad():
        _, _, h = normals_areas_curvature(torch.from_numpy(V), torch.from_numpy(F))
    return h.numpy()


def chord_errors(c, c0):
    """Per-chord errors of c against the ground truth c0, both (n, 6)."""
    d, d0 = c[:, 3:] - c[:, :3], c0[:, 3:] - c0[:, :3]
    length, length0 = np.linalg.norm(d, axis=1), np.linalg.norm(d0, axis=1)
    cos = np.vecdot(d, d0) / np.maximum(length * length0, 1e-12)
    return {
        "white_error": np.linalg.norm(c[:, :3] - c0[:, :3], axis=1),
        "pial_error": np.linalg.norm(c[:, 3:] - c0[:, 3:], axis=1),
        "chord_length_error": length - length0,
        "chord_angle": np.degrees(np.arccos(np.clip(cos, -1, 1))),
    }


def _summary(errors):
    return "  ".join(
        f"{k} p50 {np.median(np.abs(e)):.3f} p95 {np.percentile(np.abs(e), 95):.3f}"
        for k, e in errors.items()
    )


def _fit_step(
    x0,
    objects,
    *,
    deformation_kernel_width,
    data_kernel_width,
    feature_kernel_width,
    data_noise_std,
    number_of_time_points,
    use_rk2,
    max_iterations,
    convergence_tolerance,
    memory_length,
):
    """One dense LDDMM fit from points x0 (m, d); returns the deformed points.

    objects: (rows of x0, template, target) per attachment; the distances are summed.
    """
    deformation = KeopsKernel(kernel_width=deformation_kernel_width)
    data = KeopsKernel(
        kernel_width=data_kernel_width, feature_width=feature_kernel_width
    )
    device, _ = utilities.get_best_device(deformation.gpu_mode)
    dtype = default.tensor_scalar_type
    cp = utilities.move_data(x0, device=device, dtype=dtype)

    exponential = Exponential(
        dense_mode=True,
        kernel=deformation,
        number_of_time_points=number_of_time_points,
        use_rk2_for_shoot=use_rk2,
        use_rk2_for_flow=use_rk2,
    )

    def shoot(momenta):
        exponential.set_initial_control_points(cp)
        exponential.set_initial_template_points({"landmark_points": cp})
        exponential.set_initial_momenta(momenta)
        exponential.update()
        return exponential.get_template_points()["landmark_points"]

    def objective(flat):
        momenta = utilities.move_data(
            flat.reshape(x0.shape), device=device, dtype=dtype
        )
        momenta.requires_grad_(True)
        x = shoot(momenta)
        attachment = (
            sum(
                MultiObjectAttachment.extended_varifold_distance(
                    x[rows], template, target, data
                )
                for rows, template, target in objects
            )
            / data_noise_std**2
        )
        regularity = exponential.get_norm_squared()
        loss = attachment + regularity
        loss.backward()
        objective.last = attachment.item(), regularity.item()
        return loss.item(), momenta.grad.detach().cpu().numpy().ravel().astype(
            np.float64
        )

    start = time.perf_counter()
    result = sp.optimize.minimize(
        objective,
        np.zeros(x0.size),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_iterations,
            "maxcor": memory_length,
            "ftol": convergence_tolerance,
        },
    )
    attachment, regularity = objective.last
    info = (
        f"{result.nit} its {time.perf_counter() - start:.0f} s  "
        f"attachment {attachment:.3e} regularity {regularity:.3e}"
    )
    if not result.success:
        info += f" ({result.message})"

    momenta = utilities.move_data(
        result.x.reshape(x0.shape), device=device, dtype=dtype
    )
    with torch.no_grad():
        return shoot(momenta).detach().cpu().numpy().astype(np.float64), info


def backflow(
    states_path: Path,
    *,
    space="R6",
    deformation_kernel_width=5.0,
    data_kernel_width=5.0,
    feature_kernel_width=None,
    data_noise_std=1.0,
    number_of_time_points=5,
    use_rk2=True,
    max_iterations=100,
    convergence_tolerance=1e-8,
    memory_length=15,
):
    """Flow the last saved forward-flow state back to the first, one saved step at a time.

    states_path: the .npz written by flow.flow() (faces (m, 3), states (K+1, n, 6)).
    space: "R6" (baseline) or "R3" (ablation); see the module docstring.
    feature_kernel_width: width of the H0 kernel in 1/mm; None for the commensurate
        1 / data kernel width, as in register().
    Writes <stem>-back-<space>.npz (the backward states as chords, aligned with the forward
    ones) and <stem>-back-<space>-{white,pial}.vtk (the final result, with per-vertex errors against state 0:
    white_error, pial_error, chord_length_error, chord_angle). Returns the three paths.
    """
    z = np.load(states_path)
    F, S = z["faces"].astype(np.int64), z["states"]
    h0 = np.stack(
        [mean_curvature(S[0, :, :3], F), mean_curvature(S[0, :, 3:], F)], axis=1
    )
    template = SimpleNamespace(connectivity=F, features=h0)
    n = S.shape[1]

    def target(points):
        return SimpleNamespace(points=points, connectivity=F, features=h0, norm=None)

    if space == "R6":

        def objects(c):
            return [(slice(None), template, target(c))]

        def to_points(c):
            return c

        def to_chords(x):
            return x
    elif space == "R3":

        def objects(c):
            return [
                (slice(0, n), template, target(c[:, :3])),
                (slice(n, 2 * n), template, target(c[:, 3:])),
            ]

        def to_points(c):
            return np.concatenate([c[:, :3], c[:, 3:]])

        def to_chords(x):
            return np.hstack([x[:n], x[n:]])
    else:
        raise ValueError(f"space must be 'R6' or 'R3', not {space!r}")

    c = S[-1].copy()
    back = [c]
    for k in range(len(S) - 1, 0, -1):
        x, info = _fit_step(
            to_points(c),
            objects(S[k - 1]),
            deformation_kernel_width=deformation_kernel_width,
            data_kernel_width=data_kernel_width,
            feature_kernel_width=feature_kernel_width,
            data_noise_std=data_noise_std,
            number_of_time_points=number_of_time_points,
            use_rk2=use_rk2,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            memory_length=memory_length,
        )
        c = to_chords(x)
        back.append(c)
        logger.info(
            f"{space} {k} -> {k - 1}: {info}\n  {_summary(chord_errors(c, S[k - 1]))}"
        )

    stem = f"{states_path.stem}-back-{space}"
    back_path = states_path.with_name(f"{stem}.npz")
    np.savez(back_path, faces=F, states=np.stack(back[::-1]))
    errors = chord_errors(c, S[0])
    u_out = states_path.with_name(f"{stem}-white.vtk")
    v_out = states_path.with_name(f"{stem}-pial.vtk")
    _write(u_out, c[:, :3], F, errors)
    _write(v_out, c[:, 3:], F, errors)
    return back_path, u_out, v_out
