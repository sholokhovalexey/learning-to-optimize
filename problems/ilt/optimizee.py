"""Inverse lithography optimizee."""

from __future__ import annotations

import torch
import torch.nn as nn

from l2o.core import BaseOptimizee


class ILTOptimizee(BaseOptimizee):
    """Inverse lithography objective over mask logits ``x`` (flattened ``H*W``)."""

    def __init__(
        self,
        target: torch.Tensor,
        litho: nn.Module | None = None,
        *,
        pvb_weight: float = 0.08,
    ):
        """Initialize ILT objective from target layouts and lithography proxy.

        Args:
            target: Tensor of shape ``(B, H, W)`` with target patterns.
            litho: Optional lithography simulator returning nominal/max/min prints.
            pvb_weight: Weight for PV-band robustness term.
        """
        super().__init__()
        if target.dim() != 3:
            raise ValueError("target must be (B, H, W)")
        self.register_buffer("target", target)
        self.pvb_weight = pvb_weight
        if litho is None:
            from ilt.sim.simple_litho import SimplifiedLitho as _SimplifiedLitho

            litho = _SimplifiedLitho()
        self.litho = litho
        self._hw = (int(target.shape[1]), int(target.shape[2]))

    def loss_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-row mean L2 (nominal), PV-band proxy, and weighted total."""
        b, d = x.shape
        h, w = self._hw
        if b != self.target.shape[0] or d != h * w:
            raise ValueError(f"x must be (B={self.target.shape[0]}, {h * w}), got {x.shape}")
        m = torch.sigmoid(x).view(b, h, w)
        mask = m.unsqueeze(1)
        p_nom, p_max, p_min = self.litho(mask)
        t = self.target.unsqueeze(1)
        l2 = ((p_nom - t) ** 2).mean(dim=(1, 2, 3))
        pvb = ((p_max - p_min) ** 2).mean(dim=(1, 2, 3))
        total = l2 + self.pvb_weight * pvb
        return total, l2, pvb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-row weighted ILT objective values at logits ``x``."""
        total, _, _ = self.loss_components(x)
        return total

