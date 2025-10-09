from itertools import count
from types import EllipsisType

import numpy as np
import igl
from scipy.sparse.linalg import spsolve
from typing import Generator, NamedTuple, Callable


class Mesh(NamedTuple):
    verts: np.ndarray
    faces: np.ndarray


Norm = Callable[[Mesh], float]
Mean = Callable[[Mesh], np.ndarray]


def root_mean_square(mesh: Mesh) -> float:
    return np.sqrt(np.mean(mesh.verts * mesh.verts))


def mean_length(mesh: Mesh) -> float:
    return np.mean(np.linalg.norm(mesh.verts, axis=1))


def center_of_mass(mesh: Mesh) -> np.ndarray:
    return np.mean(mesh.verts, axis=0)


def flow(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    steps: int | EllipsisType = ...,
    rate_0: float = 1e-4,
    rate_common_factor: float = 1.4,
    norm: Norm = root_mean_square,
    mean: Mean = center_of_mass,
) -> Generator[Mesh, None, None]:
    """
    Apply CMCF to a triangular mesh. A generator which yields successive stages
    of flow. Note stages are computed using normalized values, but the yielded
    meshes are always un-normalized to return to the original scale.

    :param verts: Floating-point ndarray with shape (v, 3) holding vertex
    coordinates.
    :param faces: Integral ndarray with shape (f, 3) holding triangle indices.
    :param steps: The number of steps to compute, or ... to generate
    indefinitely.
    :param rate_0: The initial flow rate.
    :param rate_common_factor: Increase the flow rate by this factor after each
    stage. Set to 1 to use a constant flow rate.
    :param norm: Function used to compute the "scale" of the mesh.
    :param mean: Function used to compute the "center" of the mesh.
    """

    rate = rate_0

    verts = np.copy(verts)

    # Initial normalization. Record these so subsequent stages can be re-emitted in common coordinates.

    verts -= (mean_0 := mean(Mesh(verts, faces)))
    verts /= (norm_0 := norm(Mesh(verts, faces)))

    # Stiffness matrix per Kazhdan
    L = igl.cotmatrix(verts, faces)

    yield Mesh(verts * norm_0 + mean_0, faces)

    for _ in range(steps) if steps is not ... else count():
        # Mass matrix per Kazhdan
        M = igl.massmatrix(verts, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC)

        verts = spsolve(M - rate * L, M * verts, "MMD_AT_PLUS_A")

        # Stage normalization.
        verts -= mean(Mesh(verts, faces))
        verts /= norm(Mesh(verts, faces))

        yield Mesh(verts * norm_0 + mean_0, faces)

        rate *= rate_common_factor
