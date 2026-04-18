"""Helpers for visualizing ILT masks and printed images (notebook / inference)."""

from __future__ import annotations

import torch


@torch.no_grad()
def mask_and_printed_nominal(optimizee: torch.nn.Module, x: torch.Tensor) -> tuple:
    """Return (mask HxW, printed_nominal HxW, target HxW) as CPU float tensors."""
    h, w = optimizee._hw  # type: ignore[attr-defined]
    m = torch.sigmoid(x).view(-1, h, w)
    mask_b = m.unsqueeze(1)
    p_nom, _, _ = optimizee.litho(mask_b)  # type: ignore[attr-defined]
    tgt = optimizee.target  # type: ignore[attr-defined]
    return m[0].cpu(), p_nom[0, 0].cpu(), tgt[0].cpu()
