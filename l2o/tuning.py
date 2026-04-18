"""Learning-rate tuning helpers for analytic optimizers."""

from __future__ import annotations

import torch
import torch.nn as nn

from optimization import ADAM_EPS



def tune_adam_learning_rate(
    lrs: list[float],
    optimizee: nn.Module,
    x0: torch.Tensor,
    n_steps: int,
    device: torch.device | str,
    eps: float = ADAM_EPS,
) -> float:
    """Pick best Adam LR on a single task by final mean loss."""
    device = torch.device(device) if isinstance(device, str) else device
    from optimization import OptimizerAdam

    dim = x0.shape[-1]
    best_lr = lrs[0]
    best_final = float("inf")
    for lr in lrs:
        opt_adam = OptimizerAdam(dim, lr=lr, beta1=0.9, beta2=0.999, eps=eps).to(device)
        x = x0.clone().to(device)
        last = 0.0
        for _ in range(n_steps):
            _, grad = optimizee.loss_and_grad(x)
            with torch.no_grad():
                delta = opt_adam(grad.detach())
                x = x + delta
                last = optimizee(x).mean().item()
        if last < best_final:
            best_final = last
            best_lr = lr
    return best_lr


def tune_gd_learning_rate(
    lrs: list[float],
    optimizee: nn.Module,
    x0: torch.Tensor,
    n_steps: int,
    device: torch.device | str,
) -> float:
    """Pick best plain GD LR on a single task by final mean loss."""
    device = torch.device(device) if isinstance(device, str) else device
    from optimization import OptimizerGD

    dim = x0.shape[-1]
    best_lr = lrs[0]
    best_final = float("inf")
    for lr in lrs:
        opt_gd = OptimizerGD(dim, lr=lr).to(device)
        x = x0.clone().to(device)
        last = 0.0
        for _ in range(n_steps):
            _, grad = optimizee.loss_and_grad(x)
            with torch.no_grad():
                delta = opt_gd(grad.detach())
                x = x + delta
                last = optimizee(x).mean().item()
        if last < best_final:
            best_final = last
            best_lr = lr
    return best_lr


def tune_adagrad_learning_rate(
    lrs: list[float],
    optimizee: nn.Module,
    x0: torch.Tensor,
    n_steps: int,
    device: torch.device | str,
    eps: float = 1e-6,
) -> float:
    """Pick best AdaGrad LR on a single task by final mean loss."""
    device = torch.device(device) if isinstance(device, str) else device
    from optimization import OptimizerAdaGrad

    dim = x0.shape[-1]
    best_lr = lrs[0]
    best_final = float("inf")
    for lr in lrs:
        opt = OptimizerAdaGrad(dim, lr=lr, eps=eps).to(device)
        opt.reset(device=device)
        x = x0.clone().to(device)
        last = 0.0
        for _ in range(n_steps):
            _, grad = optimizee.loss_and_grad(x)
            with torch.no_grad():
                delta = opt(grad.detach())
                x = x + delta
                last = optimizee(x).mean().item()
        if last < best_final:
            best_final = last
            best_lr = lr
    return best_lr


def tune_rmsprop_learning_rate(
    lrs: list[float],
    optimizee: nn.Module,
    x0: torch.Tensor,
    n_steps: int,
    device: torch.device | str,
    eps: float = 1e-6,
    beta: float = 0.99,
) -> float:
    """Pick best RMSprop LR on a single task by final mean loss."""
    device = torch.device(device) if isinstance(device, str) else device
    from optimization import OptimizerRMSprop

    dim = x0.shape[-1]
    best_lr = lrs[0]
    best_final = float("inf")
    for lr in lrs:
        opt = OptimizerRMSprop(dim, lr=lr, beta=beta, eps=eps).to(device)
        opt.reset(device=device)
        x = x0.clone().to(device)
        last = 0.0
        for _ in range(n_steps):
            _, grad = optimizee.loss_and_grad(x)
            with torch.no_grad():
                delta = opt(grad.detach())
                x = x + delta
                last = optimizee(x).mean().item()
        if last < best_final:
            best_final = last
            best_lr = lr
    return best_lr

