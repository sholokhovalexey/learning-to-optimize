"""Quadratic datasets and collate utilities."""

from __future__ import annotations

import torch
import torch.utils.data

from l2o.datasets import BaseTaskDataset

from .optimizee import QuadraticOptimizee
from .utils import random_ill_conditioned_A


def collate_quadratic_batch(
    batch: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]],
) -> tuple[torch.Tensor, QuadraticOptimizee]:
    """Stack batched quadratic samples into one :class:`QuadraticOptimizee`."""
    xs = torch.stack([b[0] for b in batch])
    a_stack = torch.stack([b[1][0] for b in batch])
    b_stack = torch.stack([b[1][1] for b in batch])
    return xs, QuadraticOptimizee(a_stack, b_stack)


class RandomQuadraticFunctionsDataset(BaseTaskDataset[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]):
    """Random batched quadratics; pair with :func:`collate_quadratic_batch`."""

    def __init__(
        self,
        size: int,
        x_dim: int,
        y_dim: int,
        device: str = "cpu",
        seed: int = 42,
        noise_std: float = 0.1,
        ill_conditioned: bool = True,
        condition_number: float = 1e3,
    ):
        """Sample random least-squares tasks and initial points.

        Each item returns ``(x_init, (A, b))`` for objective
        ``0.5 * ||A x - b||^2``.
        """
        super().__init__()
        self.size = size
        torch.manual_seed(seed)
        self.x_init = torch.randn(size, x_dim, device=device)
        self.A = torch.empty(size, y_dim, x_dim, device=device)
        for i in range(size):
            if ill_conditioned:
                self.A[i] = random_ill_conditioned_A(y_dim, x_dim, torch.device(device), condition_number)
            else:
                self.A[i] = torch.randn(y_dim, x_dim, device=device)
        self.b = torch.bmm(self.A, self.x_init.unsqueeze(-1)).squeeze()
        self.b = self.b + torch.randn_like(self.b) * noise_std

    def __len__(self) -> int:
        """Number of pre-generated tasks."""
        return self.size

    def __getitem__(self, idx: int):
        """Return one pre-generated quadratic task and its initialization."""
        return self.x_init[idx], (self.A[idx], self.b[idx])



