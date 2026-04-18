"""Core abstractions for problem-agnostic Learning-to-Optimize code."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.autograd
import torch.nn as nn


class BaseOptimizee(nn.Module, ABC):
    """Base class for inner optimization objectives."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-row objective values for parameter batch ``x``."""

    def compute_loss_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional fast-path hook for analytic gradients.

        Implement this in optimizees that can compute gradients more efficiently than autograd.
        Return ``(loss, grad)`` where ``loss`` is scalar or per-row and ``grad`` matches ``x.shape``.
        """
        raise NotImplementedError

    def loss_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Primary gradient API used by L2O loops.

        Default behavior:
        - use ``compute_loss_and_grad`` if implemented by subclass
        - otherwise compute gradient with autograd wrt ``x`` only
        """
        x = x.detach()
        try:
            loss_out, grad_out = self.compute_loss_and_grad(x)
        except NotImplementedError:
            loss_out, grad_out = None, None
        if loss_out is not None and grad_out is not None:
            loss = loss_out if loss_out.ndim == 0 else loss_out.sum()
            if grad_out.shape != x.shape:
                raise ValueError(
                    f"compute_loss_and_grad grad must match x.shape {x.shape}, got {grad_out.shape}"
                )
            return loss.detach(), grad_out.detach()

        x_req = x.detach().requires_grad_(True)
        out = self(x_req)
        loss = out if out.ndim == 0 else out.sum()
        grad = torch.autograd.grad(loss, x_req, create_graph=False, retain_graph=False, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(x_req)
        return loss.detach(), grad.detach()

