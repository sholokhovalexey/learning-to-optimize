"""Quadratic least-squares optimizee."""

from __future__ import annotations

import torch

from l2o.core import BaseOptimizee


class QuadraticOptimizee(BaseOptimizee):
    """Batched linear least squares: ``0.5 * ||A x - b||^2`` per row."""

    def __init__(self, A: torch.Tensor, b: torch.Tensor):
        """Store batched matrix/vector parameters for LS objectives."""
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-row least-squares objective values at ``x``."""
        e = torch.bmm(self.A, x.unsqueeze(-1)).squeeze(-1) - self.b
        return 0.5 * torch.sum(e**2, dim=-1)

    def compute_loss_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytic per-row loss and gradient for batched least squares."""
        e = torch.bmm(self.A, x.unsqueeze(-1)).squeeze(-1) - self.b
        loss = 0.5 * torch.sum(e**2, dim=-1)
        grad = torch.bmm(self.A.transpose(1, 2), e.unsqueeze(-1)).squeeze(-1)
        return loss, grad

