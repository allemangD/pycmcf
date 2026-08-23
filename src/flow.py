import logging
from pathlib import Path

import igl
import numpy as np
import scipy as sp
from sksparse.cholmod import cho_factor
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkIOLegacy import vtkPolyDataReader, vtkPolyDataWriter

logger = logging.getLogger(__name__)


def flow(
    u_path: Path,
    v_path: Path,
    *,
    rate_0=0.05,
    growth=1.4,
    rate_max=500,
    max_iters=100,
    stop_speed=1e-4,
    epsilon=1e-4,
):
    pipe = vtkPolyDataReader()
    pipe.file_name = u_path
    pipe.Update()
    u: vtkPolyData = pipe.output

    pipe = vtkPolyDataReader()
    pipe.file_name = v_path
    pipe.Update()
    v: vtkPolyData = pipe.output

    assert len(u.points) == len(v.points)
    f_u = np.asarray(u.polys.connectivity_array).reshape((-1, 3))
    f_v = np.asarray(v.polys.connectivity_array).reshape((-1, 3))
    assert np.array_equal(f_u, f_v)
    ff = f_u

    cc = np.concatenate(
        [
            u.points,
            v.points,
        ],
        axis=1,
    )
    uu = cc[..., :3]
    vv = cc[..., 3:]

    cc_cent = np.mean(cc, axis=0, keepdims=True)
    cc[...] -= cc_cent
    cc_norm = np.sqrt(np.mean(np.square(cc)))

    ll = igl.cotmatrix(cc, ff)

    ii = epsilon * sp.sparse.eye(len(cc))

    true_dist = np.linalg.norm(uu - vv, axis=1, keepdims=True)

    rate = rate_0

    prev = np.empty_like(cc)
    for it in range(max_iters):
        logger.debug(f"{it=}: {rate=:.3e}")
        mm = igl.massmatrix(cc, ff)

        np.copyto(prev, cc)

        cc[...] = cho_factor(
            mm + ii - rate * ll,
        ).solve(
            (mm + ii) @ cc,
        )

        cc[...] -= np.mean(cc, axis=0, keepdims=True)
        cc[...] *= cc_norm / np.sqrt(np.mean(np.square(cc)))

        diff = uu - vv
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        norm = diff / dist
        extra = dist - true_dist
        fixup = -norm * extra / 2

        uu[...] += fixup
        vv[...] -= fixup

        mean_speed = np.sqrt(np.mean(np.square(cc - prev))) / rate
        logger.debug(f"{mean_speed=:.3e}")
        if mean_speed < stop_speed:
            logger.info("reached stop condition.")
            break
        rate = np.clip(rate * growth, 0, rate_max)
    else:
        logger.info("maximum iterations exceeded.")

    u_path = u_path.with_stem(f"{u_path.stem}-flow")
    u.points = uu
    pipe = vtkPolyDataWriter()
    pipe.file_name = u_path
    pipe.SetFileTypeToBinary()
    pipe.input_data = u
    pipe.Update()

    v_path = v_path.with_stem(f"{v_path.stem}-flow")
    v.points = vv
    pipe = vtkPolyDataWriter()
    pipe.file_name = v_path
    pipe.SetFileTypeToBinary()
    pipe.input_data = v
    pipe.Update()

    return u_path, v_path
