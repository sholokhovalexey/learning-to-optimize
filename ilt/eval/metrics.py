"""LithoBench-style ILT metrics (Zheng et al., NeurIPS 2023): binarized images, fixed eval resolution.

The LithoBench paper evaluates L2 and PVB after **binarizing** masks and printed images, with outputs
interpolated to **2048x2048** before metric computation. We apply the same protocol to this repo's
toy litho stack so comparisons use a **fixed optimization budget** (inner steps) and **reported**
metrics that match common academic reporting - still on the **same differentiable proxy** as training,
not full industrial lithography.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from problems.ilt import ILTOptimizee


@dataclass
class ILTReportedMetricMeans:
    """Mean L2 / PVB / weighted total on binarized, upsampled tensors (LithoBench-style)."""

    total: float
    l2: float
    pvb: float


def lithobench_style_metrics(
    optee: ILTOptimizee,
    x: torch.Tensor,
    *,
    eval_size: int = 2048,
    binarize_threshold: float = 0.5,
) -> ILTReportedMetricMeans:
    """L2 (nominal print vs target) and PVB proxy on **binarized** images at ``eval_size``².

    Pipeline: sigmoid mask → litho forward → upsample continuous images to ``eval_size``²
    (bilinear) → threshold at ``binarize_threshold`` → mean squared L2 and mean squared PV band.
    """
    b, d = x.shape
    h, w = optee._hw  # type: ignore[attr-defined]
    if b != optee.target.shape[0] or d != h * w:
        raise ValueError(f"x must match optimizee layout, got {x.shape}")
    m = torch.sigmoid(x).view(b, 1, h, w)
    p_nom, p_max, p_min = optee.litho(m)
    t = optee.target.unsqueeze(1)

    size = (eval_size, eval_size)

    def _up(u: torch.Tensor) -> torch.Tensor:
        return F.interpolate(u.float(), size=size, mode="bilinear", align_corners=False)

    t_up = _up(t)
    pn = _up(p_nom)
    px = _up(p_max)
    pnin = _up(p_min)

    def _bin(u: torch.Tensor) -> torch.Tensor:
        return (u > binarize_threshold).to(dtype=u.dtype)

    t_b = _bin(t_up)
    pn_b = _bin(pn)
    px_b = _bin(px)
    mn_b = _bin(pnin)

    l2 = ((pn_b - t_b) ** 2).mean(dim=(1, 2, 3))
    pvb = ((px_b - mn_b) ** 2).mean(dim=(1, 2, 3))
    total = l2 + float(optee.pvb_weight) * pvb
    return ILTReportedMetricMeans(
        total=float(total.mean().item()),
        l2=float(l2.mean().item()),
        pvb=float(pvb.mean().item()),
    )
