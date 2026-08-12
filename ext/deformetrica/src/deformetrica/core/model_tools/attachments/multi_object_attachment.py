import logging

import igl
import numpy as np
import torch
from torch import Tensor

from deformetrica.core.observations import SurfaceMesh
from deformetrica.support import utilities
from deformetrica.support.kernels import AbstractKernel

logger = logging.getLogger(__name__)


class MultiObjectAttachment:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, attachment_types, kernels):
        # List of strings, e.g. 'varifold' or 'current'.
        self.attachment_types = attachment_types
        # List of kernel objects.
        self.kernels = kernels

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_weighted_distance(self, data, multi_obj1, multi_obj2, inverse_weights):
        """
        Takes two multiobjects and their new point positions to compute the distances
        """
        distances = self.compute_distances(data, multi_obj1, multi_obj2)
        assert distances.size()[0] == len(inverse_weights)
        device = next(iter(data.values())).device  # deduce device from template_data
        dtype = next(iter(data.values())).dtype  # deduce dtype from template_data
        inverse_weights_torch = utilities.move_data(
            inverse_weights, device=device, dtype=dtype
        )
        return torch.sum(distances / inverse_weights_torch)

    def compute_distances(self, data, multi_obj1, multi_obj2):
        """
        Takes two multiobjects and their new point positions to compute the distances.
        """
        assert len(multi_obj1.object_list) == len(multi_obj2.object_list), (
            "Cannot compute distance between multi-objects which have different number of objects"
        )
        device = next(iter(data.values())).device  # deduce device from template_data
        dtype = next(iter(data.values())).dtype  # deduce dtype from template_data
        distances = torch.zeros(
            (len(multi_obj1.object_list),), device=device, dtype=dtype
        )

        pos = 0
        for i, obj1 in enumerate(multi_obj1.object_list):
            obj2 = multi_obj2.object_list[i]

            if self.attachment_types[i].lower() == "current":
                distances[i] = self.current_distance(
                    data["landmark_points"][pos : pos + obj1.get_number_of_points()],
                    obj1,
                    obj2,
                    self.kernels[i],
                )
                pos += obj1.get_number_of_points()

            elif self.attachment_types[i].lower() == "pointcloud":
                distances[i] = self.point_cloud_distance(
                    data["landmark_points"][pos : pos + obj1.get_number_of_points()],
                    obj1,
                    obj2,
                    self.kernels[i],
                )
                pos += obj1.get_number_of_points()

            elif self.attachment_types[i].lower() == "varifold":
                distances[i] = self.varifold_distance(
                    data["landmark_points"][pos : pos + obj1.get_number_of_points()],
                    obj1,
                    obj2,
                    self.kernels[i],
                )
                pos += obj1.get_number_of_points()

            elif self.attachment_types[i].lower() == "extendedvarifold":
                distances[i] = self.extended_varifold_distance(
                    data["landmark_points"][pos : pos + obj1.get_number_of_points()],
                    obj1,
                    obj2,
                    self.kernels[i],
                )
                pos += obj1.get_number_of_points()

            elif self.attachment_types[i].lower() == "landmark":
                distances[i] = self.landmark_distance(
                    data["landmark_points"][pos : pos + obj1.get_number_of_points()],
                    obj2,
                )
                pos += obj1.get_number_of_points()

            elif self.attachment_types[i].lower() == "l2":
                assert obj1.type.lower() == "image" and obj2.type.lower() == "image"
                distances[i] = self.L2_distance(data["image_intensities"], obj2)

            else:
                assert False, (
                    f"Please implement the distance {self.attachment_types[i]} you are trying to use :)"
                )

        return distances

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    @staticmethod
    def current_distance(points, source, target, kernel):
        """
        Compute the current distance between source and target, assuming points are the new points of the source
        We assume here that the target never moves.
        """
        device, _ = utilities.get_best_device(kernel.gpu_mode)
        c1, n1, c2, n2 = (
            MultiObjectAttachment.__get_source_and_target_centers_and_normals(
                points, source, target, device=device
            )
        )

        def current_scalar_product(points_1, points_2, normals_1, normals_2):
            assert (
                points_1.device
                == points_2.device
                == normals_1.device
                == normals_2.device
            ), "tensors must be on the same device"
            return torch.dot(
                normals_1.view(-1),
                kernel.convolve(points_1, points_2, normals_2).view(-1),
            )

        if target.norm is None:
            target.norm = current_scalar_product(c2, c2, n2, n2)

        return (
            current_scalar_product(c1, c1, n1, n1)
            + target.norm.to(c1.device)
            - 2 * current_scalar_product(c1, c2, n1, n2)
        )

    @staticmethod
    def point_cloud_distance(points, source, target, kernel):
        """
        Compute the point cloud distance between source and target, assuming points are the new points of the source
        We assume here that the target never moves.
        """
        device, _ = utilities.get_best_device(kernel.gpu_mode)
        c1, n1, c2, n2 = (
            MultiObjectAttachment.__get_source_and_target_centers_and_normals(
                points, source, target, device=device
            )
        )

        def point_cloud_scalar_product(points_1, points_2, normals_1, normals_2):
            return torch.dot(
                normals_1.view(-1),
                kernel.convolve(points_1, points_2, normals_2, mode="pointcloud").view(
                    -1
                ),
            )

        if target.norm is None:
            target.norm = point_cloud_scalar_product(c2, c2, n2, n2)

        return (
            point_cloud_scalar_product(c1, c1, n1, n1)
            + target.norm
            - 2 * point_cloud_scalar_product(c1, c2, n1, n2)
        )

    @staticmethod
    def varifold_distance(
        points: Tensor,
        source: SurfaceMesh,
        target: SurfaceMesh,
        kernel: AbstractKernel,
    ):
        """
        Returns the varifold distance between the 3D meshes
        source and target are SurfaceMesh objects
        points are source points (torch tensor)
        """
        device, _ = utilities.get_best_device(kernel.gpu_mode)
        dtype = points.dtype

        pa = points
        fa = source.connectivity
        na = igl.per_vertex_normals(pa.detach(), fa)
        aa = igl.massmatrix(pa.detach(), fa).diagonal()

        pb = target.points
        fb = target.connectivity
        nb = igl.per_vertex_normals(pb, fb)
        ab = igl.massmatrix(pb, fb).diagonal()

        def wrap(t):
            if isinstance(t, np.ndarray):
                return torch.from_numpy(t)
            else:
                return t

        def varifold_scalar_product(x, y):
            px, ax, nx = (
                wrap(t).type(dtype, non_blocking=True).to(device, non_blocking=True)
                for t in x
            )
            py, ay, ny = (
                wrap(t).type(dtype, non_blocking=True).to(device, non_blocking=True)
                for t in y
            )
            return torch.dot(
                ax.view(-1),
                kernel.convolve(
                    (px, nx),
                    (py, ny),
                    ay.view(-1, 1),
                    mode="varifold",
                ).view(-1),
            )

        a = pa, aa, na
        b = pb, ab, nb

        if target.norm is None:
            target.norm = varifold_scalar_product(b, b)

        return (
            varifold_scalar_product(a, a)
            + target.norm
            - 2 * varifold_scalar_product(a, b)
        )

    @staticmethod
    def extended_varifold_distance(
        points: Tensor,
        source: SurfaceMesh,
        target: SurfaceMesh,
        kernel: AbstractKernel,
    ):
        """
        Returns the extended_varifold distance between the 3D meshes
        source and target are SurfaceMesh objects
        points are source points (torch tensor)
        """
        device, _ = utilities.get_best_device(kernel.gpu_mode)
        dtype = points.dtype

        pa = points
        fa = source.connectivity
        na = igl.per_vertex_normals(pa.detach(), fa)
        aa = igl.massmatrix(pa.detach(), fa).diagonal()
        if hasattr(source, "_h_field"):
            ha = source._h_field
        else:
            hna = (igl.cotmatrix(pa.detach(), fa) * pa.detach()) / np.expand_dims(
                aa, axis=1
            )
            ha = np.vecdot(hna, na, axis=1)

        pb = target.points
        fb = target.connectivity
        nb = igl.per_vertex_normals(pb, fb)
        ab = igl.massmatrix(pb, fb).diagonal()
        if hasattr(target, "_h_field"):
            hb = target._h_field
        else:
            hnb = (igl.cotmatrix(pb, fb) * pb) / np.expand_dims(ab, axis=1)
            hb = target._h_field = np.vecdot(hnb, nb, axis=1)

        def wrap(t):
            if isinstance(t, np.ndarray):
                return torch.from_numpy(t)
            else:
                return t

        def extended_varifold_scalar_product(x, y):
            px, ax, nx, hx = (
                wrap(t).type(dtype, non_blocking=True).to(device, non_blocking=True)
                for t in x
            )
            py, ay, ny, hy = (
                wrap(t).type(dtype, non_blocking=True).to(device, non_blocking=True)
                for t in y
            )
            return torch.dot(
                ax.view(-1),
                kernel.convolve(
                    (px, nx, hx),
                    (py, ny, hy),
                    ay.view(-1, 1),
                    mode="extended_varifold",
                ).view(-1),
            )

        a = pa, aa, na, ha
        b = pb, ab, nb, hb

        if target.norm is None:
            target.norm = extended_varifold_scalar_product(b, b)

        return (
            extended_varifold_scalar_product(a, a)
            + target.norm
            - 2 * extended_varifold_scalar_product(a, b)
        )

    @staticmethod
    def landmark_distance(points, target):
        """
        Point correspondance distance
        """
        target_points = utilities.move_data(
            target.get_points(), dtype=str(points.type()), device=points.device
        )
        assert points.device == target_points.device, (
            "tensors must be on the same device"
        )
        return torch.sum(
            (points.contiguous().view(-1) - target_points.contiguous().view(-1)) ** 2
        )

    @staticmethod
    def L2_distance(intensities, target):
        """
        L2 image distance.
        """
        # if not isinstance(intensities, torch.Tensor):
        #     target_intensities = target.get_intensities_torch(tensor_scalar_type=intensities.type(), device=intensities.device)
        # else:
        #     target_intensities = intensities

        assert isinstance(intensities, torch.Tensor)

        target_intensities = utilities.move_data(
            target.get_intensities(),
            dtype=intensities.type(),
            device=intensities.device,
        )
        # target_intensities = target.get_intensities_torch(tensor_scalar_type=intensities.type(), device=intensities.device)
        assert intensities.device == target_intensities.device, (
            "tensors must be on the same device"
        )
        return torch.sum(
            (
                intensities.contiguous().view(-1)
                - target_intensities.contiguous().view(-1)
            )
            ** 2
        )

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    @staticmethod
    def __get_source_and_target_centers_and_normals(
        points, source, target, device=None
    ):
        if device is None:
            device = points.device

        dtype = str(points.dtype)

        c1, n1 = source.get_centers_and_normals(
            points,
            tensor_scalar_type=utilities.get_torch_scalar_type(dtype=dtype),
            tensor_integer_type=utilities.get_torch_integer_type(dtype=dtype),
            device=device,
        )
        c2, n2 = target.get_centers_and_normals(
            tensor_scalar_type=utilities.get_torch_scalar_type(dtype=dtype),
            tensor_integer_type=utilities.get_torch_integer_type(dtype=dtype),
            device=device,
        )

        assert c1.device == n1.device == c2.device == n2.device, (
            "all tensors must be on the same device, c1.device="
            + str(c1.device)
            + ", n1.device="
            + str(n1.device)
            + ", c2.device="
            + str(c2.device)
            + ", n2.device="
            + str(n2.device)
        )
        return c1, n1, c2, n2
