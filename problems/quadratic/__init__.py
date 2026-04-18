"""Quadratic least-squares problem family."""

from .datasets import (
    RandomQuadraticFunctionsDataset,
    collate_quadratic_batch,
)
from .optimizee import QuadraticOptimizee
from .utils import (
    distance_to_solution,
    make_aligned_quadratic_problem,
    quadratic_ls_minimizer,
    random_ill_conditioned_A,
)

__all__ = [
    "QuadraticOptimizee",
    "collate_quadratic_batch",
    "RandomQuadraticFunctionsDataset",
    "random_ill_conditioned_A",
    "quadratic_ls_minimizer",
    "make_aligned_quadratic_problem",
    "distance_to_solution",
]

