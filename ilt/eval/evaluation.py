"""Inner-loop unroll for fair optimizer comparisons on ILT tasks.

For **README / ICCAD tables**, use :func:`ilt.eval.metrics.lithobench_style_metrics` (binarized
images, fixed eval resolution). Training-time smooth objectives come from :meth:`ILTOptimizee.forward`
/ :meth:`ILTOptimizee.loss_components`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from optimization import OptimizerAdaGrad, OptimizerAdam, OptimizerRMSprop


def run_inner_optimization(
    optimizer: nn.Module,
    optee: nn.Module,
    x0: torch.Tensor,
    n_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    """Unrolled inner loop; returns final ``x`` and mean-loss curve."""
    x = x0.clone().to(device)
    optee = optee.to(device)
    if isinstance(optimizer, OptimizerAdam):
        optimizer.reset(device=device)
    elif isinstance(optimizer, (OptimizerAdaGrad, OptimizerRMSprop)):
        optimizer.reset(device=device)
    else:
        optimizer.reset()
    curve: list[float] = []
    for t in range(n_steps):
        loss_tensor, grad = optee.loss_and_grad(x)
        g = grad.detach()
        step = torch.tensor(float(t), device=device, dtype=torch.float32)
        try:
            delta = optimizer(g, loss=loss_tensor, step=step)
        except TypeError:
            delta = optimizer(g)
        x = x + delta
        with torch.no_grad():
            curve.append(float(optee(x).mean().item()))
    return x, curve
