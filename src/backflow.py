"""
Baseline backward flow: LDDMM in R^6 from the flowed chords back to the original chords.

The forward flow (flow.flow) keeps the triangulation, so chord i of the flowed surface is known
to correspond to chord i of the original. That gives the baseline a ground truth: after the
backward flow, chord i should sit on chord i. The registration itself never sees vertex
indices; it only sees the extended varifold attachment, as in register().

  - template: flowed chords c_f = (u_f, v_f), translated so its centroid matches the original
    (flow() returns centred surfaces);
  - target:   original chords c_0 = (u, v);
  - features: H0 = [H(u), H(v)] per chord from the original surfaces, carried unchanged on both
    sides (the paper's H0 is fixed at the first timestep);
  - tangent planes: unit 2-vectors in R^6 (normals have no meaning there);
  - deformation: dense LDDMM (control points = template vertices), shot with deformetrica's
    Exponential and fitted with scipy L-BFGS-B on the deterministic-atlas objective
    attachment / noise_std^2 + |p|_K^2.

The flowed chords are tens of mm from the originals, beyond any single fine kernel, so the fit
runs as coarse-to-fine stages, each starting where the previous one ended. The composition is
still a diffeomorphism of R^6.
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
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkIOLegacy import vtkPolyDataReader, vtkPolyDataWriter

from chordal.differentiable_geometry import normals_areas_curvature

logger = logging.getLogger(__name__)


def _read(path: Path):
    pipe = vtkPolyDataReader()
    pipe.file_name = str(path)
    pipe.Update()
    pd = pipe.output
    V = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    F = vtk_to_numpy(pd.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return pd, V, F.astype(np.int64)


def _write(path: Path, pd, V, arrays):
    pd.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(V), deep=True))
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


def _summary(label, errors):
    logger.info(
        f"{label}: "
        + "  ".join(
            f"{k} p50 {np.median(np.abs(e)):.3f} p95 {np.percentile(np.abs(e), 95):.3f}"
            for k, e in errors.items()
        )
    )


def _fit_stage(
    x0,
    template,
    target,
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
    """One dense LDDMM fit from points x0 (n, 6); returns the deformed points."""
    deformation = KeopsKernel(kernel_width=deformation_kernel_width)
    data = KeopsKernel(
        kernel_width=data_kernel_width, feature_width=feature_kernel_width
    )
    device, _ = utilities.get_best_device(deformation.gpu_mode)
    dtype = default.tensor_scalar_type
    cp = utilities.move_data(x0, device=device, dtype=dtype)
    template.points = x0

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
        attachment = (
            MultiObjectAttachment.extended_varifold_distance(
                shoot(momenta), template, target, data
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
    logger.info(
        f"stage deformation {deformation_kernel_width} mm, data {data_kernel_width} mm: "
        f"{result.nit} iterations in {time.perf_counter() - start:.0f} s, "
        f"attachment {attachment:.4e}  regularity {regularity:.4e}  ({result.message})"
    )

    momenta = utilities.move_data(
        result.x.reshape(x0.shape), device=device, dtype=dtype
    )
    with torch.no_grad():
        return shoot(momenta).detach().cpu().numpy().astype(np.float64)


def backflow(
    u: Path,
    v: Path,
    u_flow: Path,
    v_flow: Path,
    *,
    stages=((40.0, 40.0), (20.0, 20.0), (10.0, 10.0), (5.0, 5.0)),
    feature_kernel_width=None,
    data_noise_std=1.0,
    number_of_time_points=11,
    use_rk2=True,
    max_iterations=200,
    convergence_tolerance=1e-8,
    memory_length=15,
):
    """Register the flowed chords (u_flow, v_flow) back onto the original chords (u, v).

    stages: (deformation kernel width, data kernel width) in mm, coarse to fine.
    feature_kernel_width: width of the H0 kernel in 1/mm; None for the commensurate
        1 / data kernel width, as in register().
    Returns the paths of the reconstructed white and pial surfaces; each carries per-vertex
    errors against the known correspondence (white_error, pial_error, chord_length_error,
    chord_angle).
    """
    pd_u, Vu, F = _read(u)
    _, Vv, Fv = _read(v)
    _, Vuf, Fuf = _read(u_flow)
    _, Vvf, Fvf = _read(v_flow)
    assert all(np.array_equal(F, f) for f in (Fv, Fuf, Fvf)), "meshes must share faces"

    c0 = np.hstack([Vu, Vv])
    c = np.hstack([Vuf, Vvf])
    c += c0.mean(axis=0) - c.mean(axis=0)
    h0 = np.stack([mean_curvature(Vu, F), mean_curvature(Vv, F)], axis=1)

    template = SimpleNamespace(points=c, connectivity=F, features=h0, norm=None)
    target = SimpleNamespace(points=c0, connectivity=F, features=h0, norm=None)

    _summary("before", chord_errors(c, c0))
    for deformation_kernel_width, data_kernel_width in stages:
        c = _fit_stage(
            c,
            template,
            target,
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
        target.norm = target._extended_varifold = (
            None  # the norm depends on the data kernel
        )
        _summary(f"after {deformation_kernel_width} mm", errors := chord_errors(c, c0))

    u_out = u_flow.with_stem(f"{u_flow.stem}-back")
    v_out = v_flow.with_stem(f"{v_flow.stem}-back")
    _write(u_out, pd_u, c[:, :3], errors)
    _write(v_out, pd_u, c[:, 3:], errors)
    return u_out, v_out
