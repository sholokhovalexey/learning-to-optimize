"""Quadratic-task generators and analysis helpers."""

from __future__ import annotations

import math

import torch

from .optimizee import QuadraticOptimizee


def random_ill_conditioned_A(
    y_dim: int,
    x_dim: int,
    device: torch.device,
    condition_number: float = 1e3,
) -> torch.Tensor:
    """Sample matrix ``A`` with controlled singular-value spread.

    Uses SVD re-synthesis with log-spaced singular values to target
    an approximate condition number.
    """
    a0 = torch.randn(y_dim, x_dim, device=device)
    u, _, vh = torch.linalg.svd(a0, full_matrices=False)
    k = min(y_dim, x_dim)
    lo = -0.5 * math.log10(condition_number)
    hi = 0.5 * math.log10(condition_number)
    exponents = torch.linspace(lo, hi, k, device=device)
    s = 10.0**exponents
    return (u * s.unsqueeze(0)) @ vh


def quadratic_ls_minimizer(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Least-squares minimizer of ``0.5 * ||A x - b||^2`` for one task."""
    if A.dim() == 3:
        A = A[0]
    if b.dim() == 2:
        b = b[0]
    return torch.linalg.pinv(A) @ b


def make_aligned_quadratic_problem(
    y_dim: int,
    x_dim: int,
    device: torch.device | str,
    *,
    seed: int = 0,
    noise_std: float = 0.01,
    ill_conditioned: bool = False,
    condition_number: float = 1e3,
) -> tuple[QuadraticOptimizee, torch.Tensor]:
    """Build one quadratic task matching :class:`RandomQuadraticFunctionsDataset`.

    Returns optimizee and the latent ``x_star`` used to synthesize ``b``.
    """
    device = torch.device(device) if isinstance(device, str) else device
    torch.manual_seed(seed)
    x_star = torch.randn(1, x_dim, device=device)
    if ill_conditioned:
        a = random_ill_conditioned_A(y_dim, x_dim, device, condition_number).unsqueeze(0)
    else:
        a = torch.randn(1, y_dim, x_dim, device=device)
    b = torch.bmm(a, x_star.unsqueeze(-1)).squeeze()
    b = b + torch.randn_like(b) * noise_std
    return QuadraticOptimizee(a, b), x_star


def distance_to_solution(x_path: list, x_opt: torch.Tensor) -> list[float]:
    """Compute Euclidean distance to ``x_opt`` for each iterate in ``x_path``."""
    ref = x_opt.detach()
    out: list[float] = []
    for x in x_path:
        xc = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        ref_on = ref.to(device=xc.device, dtype=xc.dtype)
        out.append(torch.norm(xc - ref_on).item())
    return out

