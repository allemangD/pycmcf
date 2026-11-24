import functools
import logging

import torch
from pykeops.torch import Genred

from deformetrica.core import GpuMode, default
from deformetrica.support.kernels import AbstractKernel

logger = logging.getLogger(__name__)


@functools.cache
def _genred(formula, aliases):
    """Compile (once per formula and dimensions) a KeOps sum reduction over j."""
    return Genred(formula, list(aliases), reduction_op="Sum", axis=1)


class KeopsKernel(AbstractKernel):
    def __init__(
        self,
        gpu_mode=default.gpu_mode,
        kernel_width=None,
        feature_width=None,
        **kwargs,
    ):
        super().__init__("keops", gpu_mode, kernel_width)

        self.gamma = 1.0 / default.tensor_scalar_type([self.kernel_width**2])
        # Extended-varifold feature kernel exp(-|dH|^2 / w^2). H is in 1/length, so the
        # commensurate width is w = 1 / kernel_width: the term becomes
        # kernel_width^2 |dH|^2, dimensionless like |x - y|^2 / kernel_width^2.
        if feature_width is None:
            feature_width = 1.0 / self.kernel_width
        self.feature_gamma = 1.0 / default.tensor_scalar_type([feature_width**2])

    @staticmethod
    def _dims(**dims):
        """KeOps alias list; dims maps each variable name to (category, dimension)."""
        return tuple(f"{k} = {cat}({d})" for k, (cat, d) in dims.items())

    def __eq__(self, other):
        return AbstractKernel.__eq__(self, other)

    def convolve(self, x, y, p, mode="gaussian"):
        if mode == "gaussian":
            assert isinstance(x, torch.Tensor), "x variable must be a torch Tensor"
            assert isinstance(y, torch.Tensor), "y variable must be a torch Tensor"
            assert isinstance(p, torch.Tensor), "p variable must be a torch Tensor"

            # move tensors with respect to gpu_mode
            x, y, p = (
                self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, y, p]
            )
            assert x.device == y.device == p.device, (
                "tensors must be on the same device. x.device="
                + str(x.device)
                + ", y.device="
                + str(y.device)
                + ", p.device="
                + str(p.device)
            )

            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)

            device_id = x.device.index if x.device.index is not None else -1
            res = _genred(
                "Exp(-G*SqDist(X,Y)) * P",
                self._dims(
                    G=("Pm", 1), X=("Vi", d), Y=("Vj", d), P=("Vj", p.size(1))
                ),
            )(
                gamma,
                x.contiguous(),
                y.contiguous(),
                p.contiguous(),
                device_id=device_id,
            )
            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

        elif mode == "pointcloud":
            assert isinstance(x, torch.Tensor), "x variable must be a torch Tensor"
            assert isinstance(y, torch.Tensor), "y variable must be a torch Tensor"
            assert isinstance(p, torch.Tensor), "p variable must be a torch Tensor"

            # move tensors with respect to gpu_mode
            x, y, p = (
                self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, y, p]
            )
            assert x.device == y.device == p.device, (
                "tensors must be on the same device. x.device="
                + str(x.device)
                + ", y.device="
                + str(y.device)
                + ", p.device="
                + str(p.device)
            )

            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)

            device_id = x.device.index if x.device.index is not None else -1
            res = _genred(
                "Exp(-G*SqDist(X,Y)) * P",
                self._dims(G=("Pm", 1), X=("Vi", d), Y=("Vj", d), P=("Vj", 1)),
            )(
                gamma,
                x.contiguous(),
                y.contiguous(),
                p.contiguous(),
                device_id=device_id,
            )
            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

        elif mode == "varifold":
            assert isinstance(x, tuple), "x must be a tuple"
            assert len(x) == 2, "tuple length must be 2"
            assert isinstance(y, tuple), "y must be a tuple"
            assert len(y) == 2, "tuple length must be 2"

            # tuples are immutable, mutability is needed to mode to device
            x, nx = x
            y, ny = y

            # move tensors with respect to gpu_mode
            x, nx, y, ny, p = (
                self._move_to_device(t, gpu_mode=self.gpu_mode)
                for t in (x, nx, y, ny, p)
            )
            assert all(x.device == t.device for t in (nx, y, ny, p)), (
                "tensors must be on the same device"
            )

            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)

            device_id = x.device.index if x.device.index is not None else -1
            res = _genred(
                "Exp(-(WeightedSqDist(G, X, Y))) * Square((Nx|Ny)) * P",
                self._dims(
                    G=("Pm", 1),
                    X=("Vi", d),
                    Y=("Vj", d),
                    Nx=("Vi", nx.size(1)),
                    Ny=("Vj", ny.size(1)),
                    P=("Vj", 1),
                ),
            )(
                gamma,
                x.contiguous(),
                y.contiguous(),
                nx.contiguous(),
                ny.contiguous(),
                p.contiguous(),
                device_id=device_id,
            )
            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

        elif mode == "extended_varifold":
            assert isinstance(x, tuple), "x must be a tuple"
            assert len(x) == 3, "tuple length must be 3"
            assert isinstance(y, tuple), "y must be a tuple"
            assert len(y) == 3, "tuple length must be 3"

            # tuples are immutable, mutability is needed to mode to device
            x, nx, hx = x
            y, ny, hy = y

            # move tensors with respect to gpu_mode
            x, nx, hx, y, ny, hy, p = (
                self._move_to_device(t, gpu_mode=self.gpu_mode)
                for t in (x, nx, hx, y, ny, hy, p)
            )
            assert all(x.device == t.device for t in (nx, hx, y, ny, hy, p)), (
                "tensors must be on the same device"
            )

            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)
            feature_gamma = self.feature_gamma.to(x.device, dtype=x.dtype)
            hx, hy = hx.reshape(len(hx), -1), hy.reshape(len(hy), -1)

            device_id = x.device.index if x.device.index is not None else -1
            res = _genred(
                "Exp(-WeightedSqDist(G, X, Y)) * Exp(-WeightedSqDist(E, Hx, Hy))"
                " * Square((Nx|Ny)) * P",
                self._dims(
                    E=("Pm", 1),
                    G=("Pm", 1),
                    X=("Vi", d),
                    Y=("Vj", d),
                    Nx=("Vi", nx.size(1)),
                    Ny=("Vj", ny.size(1)),
                    Hx=("Vi", hx.size(1)),
                    Hy=("Vj", hy.size(1)),
                    P=("Vj", 1),
                ),
            )(
                feature_gamma,
                gamma,
                x.contiguous(),
                y.contiguous(),
                nx.contiguous(),
                ny.contiguous(),
                hx.contiguous(),
                hy.contiguous(),
                p.contiguous(),
                device_id=device_id,
            )
            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

        else:
            raise RuntimeError("Unknown kernel mode.")

    def convolve_gradient(self, px, x, y=None, py=None, mode="gaussian"):
        if y is None:
            y = x
        if py is None:
            py = px

        assert isinstance(px, torch.Tensor), "px variable must be a torch Tensor"
        assert isinstance(x, torch.Tensor), "x variable must be a torch Tensor"
        assert isinstance(y, torch.Tensor), "y variable must be a torch Tensor"
        assert isinstance(py, torch.Tensor), "py variable must be a torch Tensor"

        # move tensors with respect to gpu_mode
        x, px, y, py = (
            self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, px, y, py]
        )
        assert px.device == x.device == y.device == py.device, (
            "tensors must be on the same device"
        )

        d = x.size(1)
        gamma = self.gamma.to(x.device, dtype=x.dtype)

        device_id = x.device.index if x.device.index is not None else -1
        res = (
            -2
            * gamma
            * _genred(
                "(Px|Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
                self._dims(
                    G=("Pm", 1),
                    X=("Vi", d),
                    Y=("Vj", d),
                    Px=("Vi", px.size(1)),
                    Py=("Vj", py.size(1)),
                ),
            )(
                gamma, x, y, px, py, device_id=device_id
            )
        )
        return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res
