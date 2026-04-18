"""LithoBench-style layout targets (PNG) for :class:`~problems.ilt.ILTOptimizee`.

Upstream LithoBench stores MetalSet (and other splits) under ``work/<BenchmarkName>/`` with
subfolders ``target/``, ``pixelILT/``, ``glp/`` - see ``lithobench/dataset.py`` in the official repo.
For L2O we only need **target** prints resized to the demo grid.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch.utils.data

from ilt.paths import repo_root


def discover_metalset_dir(root: Path | None) -> Path:
    """Resolve a directory that contains ``target/*.png`` (and optionally ``pixelILT/``, ``glp/``).

    Tries, in order:

    * ``root`` if ``root/target`` exists
    * ``root/MetalSet``
    * ``root/work/MetalSet``
    * Default repo ``data/MetalSet`` (typical unpacked LithoBench tarball layout)
    * ``LITHOBENCH_ROOT`` / legacy ``data/lithobench/work`` when ``root is None``
    """
    repo = repo_root()
    if root is None:
        env = os.environ.get("LITHOBENCH_ROOT", "").strip()
        if env:
            root = Path(env).expanduser()
        else:
            root = repo / "data"
    root = root.expanduser().resolve()
    candidates = [
        root,
        root / "MetalSet",
        root / "work" / "MetalSet",
        repo / "data" / "MetalSet",
        repo / "data",
        repo / "data" / "lithobench" / "work",
        repo / "data" / "lithobench" / "work" / "MetalSet",
    ]
    seen: set[Path] = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    for c in ordered:
        tg = c / "target"
        if tg.is_dir() and list(tg.glob("*.png")):
            return c
    raise FileNotFoundError(
        "Could not find LithoBench MetalSet-style folder with target/*.png. "
        "Unpack the tarball under ./data so that ./data/MetalSet/target/*.png exists, "
        "or set LITHOBENCH_ROOT / --lithobench-root."
    )


def list_target_pngs(metalset_dir: Path) -> list[Path]:
    d = metalset_dir / "target"
    if not d.is_dir():
        raise FileNotFoundError(f"Missing target directory: {d}")
    return sorted(d.glob("*.png"))


def load_png_target(path: str | Path, height: int, width: int) -> torch.Tensor:
    """Load a grayscale PNG and resize to ``(height, width)`` in ``[0, 1]``."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).view(1, 1, arr.shape[0], arr.shape[1])
    t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
    return t[0, 0].contiguous()


class LithoBenchTargetDataset(torch.utils.data.Dataset):
    """One meta-training task per PNG: random init ``x_init`` and raster target (same contract as :class:`ICCADGlpDataset`)."""

    def __init__(
        self,
        target_paths: list[str | Path],
        grid_hw: tuple[int, int] = (32, 32),
        init_noise: float = 0.35,
        seed: int | None = 42,
    ):
        self.paths = [Path(p) for p in target_paths]
        self.H, self.W = grid_hw
        self.init_noise = init_noise
        self.seed = seed
        if not self.paths:
            raise ValueError("target_paths is empty")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tgt = load_png_target(self.paths[idx], self.H, self.W)
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(self.seed + idx * 10007)
        x_init = torch.randn(self.H * self.W, generator=gen) * self.init_noise
        return x_init, tgt


def train_val_split_paths(
    paths: list[Path],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Deterministic shuffle then split (e.g. ``train_ratio=0.95``)."""
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError("train_ratio must be in (0, 1]")
    rng = paths[:]
    random.Random(seed).shuffle(rng)
    n = len(rng)
    n_train = max(1, int(n * train_ratio)) if train_ratio < 1.0 else n
    return rng[:n_train], rng[n_train:]
