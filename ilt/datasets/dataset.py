"""PyTorch dataset + collate for ICCAD ``.glp`` targets (OpenILT / ICCAD 2013 benchmark)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.utils.data

from ilt.io.glp_raster import load_glp_path
from ilt.paths import repo_root


def benchmark_iccad_glp_paths(root: str | Path | None = None) -> list[Path]:
    """Sorted ``M1_test*.glp`` under ``<repo>/benchmarks/iccad2013/``."""
    r = Path(root) if root is not None else repo_root()
    d = r / "benchmarks" / "iccad2013"
    return sorted(d.glob("M1_test*.glp"))


class ICCADGlpDataset(torch.utils.data.Dataset):
    """One meta-training task per file: random init ``x_init`` and rasterized binary target.

    Parameters
    ----------
    glp_paths :
        Paths to ``.glp`` files (e.g. from :func:`download_iccad_glp_files`).
    grid_hw :
        Raster resolution ``(H, W)`` - mask parameters are ``H * W`` scalars.
    init_noise :
        Std of Gaussian noise for ``x_init`` (mask logits initialized near 0 -> sigmoid ~ 0.5).
    seed :
        Per-index deterministic noise if set.
    """

    def __init__(
        self,
        glp_paths: list[str | Path],
        grid_hw: tuple[int, int] = (32, 32),
        init_noise: float = 0.35,
        seed: int | None = 42,
    ):
        self.paths = [Path(p) for p in glp_paths]
        self.H, self.W = grid_hw
        self.init_noise = init_noise
        self.seed = seed
        if not self.paths:
            raise ValueError("glp_paths is empty")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tgt = load_glp_path(str(self.paths[idx]), self.H, self.W)
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(self.seed + idx * 10007)
        x_init = torch.randn(self.H * self.W, generator=gen) * self.init_noise
        return x_init, tgt


def collate_ilt_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.nn.Module]:
    """Stack tasks into one batched :class:`~problems.ilt.ILTOptimizee`."""
    from problems.ilt import ILTOptimizee

    xs = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    return xs, ILTOptimizee(targets)
