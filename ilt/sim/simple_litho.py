"""
Minimal differentiable lithography proxy (SOCS-style blur + resist), inspired by
Hopkins / ICCAD13-style flows used in OpenILT, LithoBench, and lithosim - kept tiny for L2O demos.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_2d(sigma: float, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ax = torch.arange(-(size // 2), size // 2 + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum().clamp(min=1e-12)
    return k.view(1, 1, size, size)


class SimplifiedLitho(nn.Module):
    """Coherent-sum-of-kernels optical model + sigmoid resist; optional defocus corners.

    Forward: mask in ``(B, 1, H, W)`` in ``[0, 1]`` -> nominal / max / min printed images.
    This mirrors the ``printedNom, printedMax, printedMin`` split used in OpenILT's ``LithoSim``.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        sigmas: tuple[float, float, float] = (1.2, 2.0, 3.0),
        weights: tuple[float, float, float] = (0.5, 0.35, 0.15),
        defocus_scale: float = 1.15,
        resist_gamma: float = 12.0,
        resist_threshold: float = 0.35,
    ):
        super().__init__()
        assert len(sigmas) == len(weights) == 3
        self.kernel_size = kernel_size
        self.resist_gamma = resist_gamma
        self.resist_threshold = resist_threshold
        self.padding = kernel_size // 2

        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum()
        self.register_buffer("socs_weights", w)

        # Nominal SOCS kernels
        for i, s in enumerate(sigmas):
            self.register_buffer(f"kern_nom_{i}", _gaussian_kernel_2d(s, kernel_size, torch.device("cpu"), torch.float32))

        # Slightly wider / narrower blur for simple process-window proxies
        for i, s in enumerate(sigmas):
            smin = max(0.4, s / defocus_scale)
            smax = s * defocus_scale
            self.register_buffer(f"kern_min_{i}", _gaussian_kernel_2d(smin, kernel_size, torch.device("cpu"), torch.float32))
            self.register_buffer(f"kern_max_{i}", _gaussian_kernel_2d(smax, kernel_size, torch.device("cpu"), torch.float32))

    def _socs_stack(self, mask: torch.Tensor, prefix: str) -> torch.Tensor:
        """Sum weighted convolutions (Hopkins SOCS intuition, heavily simplified)."""
        acc = None
        device, dtype = mask.device, mask.dtype
        for i in range(3):
            k = getattr(self, f"kern_{prefix}_{i}").to(device=device, dtype=dtype)
            term = F.conv2d(mask, k, padding=self.padding)
            wterm = self.socs_weights[i] * term
            acc = wterm if acc is None else acc + wterm
        assert acc is not None
        return acc

    def aerial_and_printed(self, mask: torch.Tensor, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        aerial = self._socs_stack(mask, prefix)
        printed = torch.sigmoid(self.resist_gamma * (aerial - self.resist_threshold))
        return aerial, printed

    def forward(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(printed_nom, printed_max, printed_min)`` each ``(B, 1, H, W)``."""
        _, p_nom = self.aerial_and_printed(mask, "nom")
        _, p_max = self.aerial_and_printed(mask, "max")
        _, p_min = self.aerial_and_printed(mask, "min")
        return p_nom, p_max, p_min
