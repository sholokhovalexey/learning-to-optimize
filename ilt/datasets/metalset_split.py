"""Persisted train/val split for LithoBench MetalSet ``target/*.png`` (reproducible ILT L2O training).

LithoBench does not ship a public train/val file list for MetalSet mask optimization in this repo's
format. We record **basenames** only so the same split works across machines; regenerate with
``force=True`` if you change ``max_samples`` or the shuffle seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ilt.datasets.lithobench_loader import list_target_pngs, train_val_split_paths
from ilt.paths import repo_root


SPLIT_VERSION = 1


def default_metalset_split_path(root: Path | None = None) -> Path:
    """Default JSON path: ``<repo>/data/metalset_train_val_split.json``."""
    r = root if root is not None else repo_root()
    return r / "data" / "metalset_train_val_split.json"


def load_or_create_metalset_split(
    metalset_dir: Path,
    split_path: Path,
    *,
    train_ratio: float,
    shuffle_seed: int,
    max_samples: int | None = None,
    force: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Load train/val PNG paths from ``split_path`` or create, save, and return them.

    Args:
        metalset_dir: Directory returned by :func:`~ilt.datasets.lithobench_loader.discover_metalset_dir`
            (contains ``target/``).
        split_path: JSON file with ``train`` / ``val`` basename lists.
        train_ratio: Fraction of (capped) tiles assigned to training after shuffling.
        shuffle_seed: Seed for :func:`train_val_split_paths`.
        max_samples: If set, only the first ``max_samples`` files (sorted order) enter the split.
        force: If True, ignore existing file and rewrite.

    Returns:
        ``(train_paths, val_paths)`` as absolute :class:`Path` objects under ``metalset_dir/target``.
    """
    metalset_dir = metalset_dir.resolve()
    split_path = split_path.resolve()
    target_dir = metalset_dir / "target"
    all_paths = list_target_pngs(metalset_dir)
    if max_samples is not None:
        all_paths = all_paths[: max_samples]

    if split_path.is_file() and not force:
        data = json.loads(split_path.read_text(encoding="utf-8"))
        if int(data.get("version", 0)) != SPLIT_VERSION:
            raise ValueError(f"Unsupported split version in {split_path}")
        if data.get("shuffle_seed") != shuffle_seed:
            raise ValueError(
                f"{split_path}: shuffle_seed mismatch (file {data.get('shuffle_seed')} vs {shuffle_seed}); "
                "use --regenerate-metalset-split or delete the file."
            )
        if abs(float(data.get("train_ratio", 0)) - train_ratio) > 1e-9:
            raise ValueError(
                f"{split_path}: train_ratio mismatch; use --regenerate-metalset-split or delete the file."
            )
        if data.get("max_samples") != max_samples:
            raise ValueError(
                f"{split_path}: max_samples mismatch (file {data.get('max_samples')} vs {max_samples}); "
                "use --regenerate-metalset-split or delete the file."
            )
        if int(data.get("n_total_before_split", -1)) != len(all_paths):
            raise ValueError(
                f"{split_path}: dataset size changed ({data.get('n_total_before_split')} vs {len(all_paths)}); "
                "use --regenerate-metalset-split or delete the file."
            )
        train_paths, val_paths = _resolve_split_file(split_path, target_dir, all_paths)
        return train_paths, val_paths

    train_paths, val_paths = train_val_split_paths(
        all_paths, train_ratio=train_ratio, seed=shuffle_seed
    )
    if not val_paths and train_paths:
        val_paths = [train_paths[-1]]
        train_paths = train_paths[:-1] or train_paths[:1]

    payload: dict[str, Any] = {
        "version": SPLIT_VERSION,
        "metalset_dir": str(metalset_dir),
        "shuffle_seed": shuffle_seed,
        "train_ratio": train_ratio,
        "max_samples": max_samples,
        "n_total_before_split": len(all_paths),
        "train": [p.name for p in train_paths],
        "val": [p.name for p in val_paths],
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return train_paths, val_paths


def _resolve_split_file(
    split_path: Path,
    target_dir: Path,
    expected_pool: list[Path],
) -> tuple[list[Path], list[Path]]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    pool_names = {p.name for p in expected_pool}
    train_names: list[str] = data["train"]
    val_names: list[str] = data["val"]
    missing = [n for n in train_names + val_names if n not in pool_names]
    if missing:
        raise ValueError(
            f"Split file {split_path} lists PNGs not in current MetalSet pool. "
            f"Missing (sample): {missing[:5]}"
        )
    train_paths = [target_dir / n for n in train_names]
    val_paths = [target_dir / n for n in val_names]
    return train_paths, val_paths


def train_basenames_for_tune(split_path: Path | None, n: int) -> list[str] | None:
    """First ``n`` training basenames from a split file, or ``None`` if missing/empty."""
    if split_path is None or not split_path.is_file():
        return None
    data = json.loads(split_path.read_text(encoding="utf-8"))
    names: list[str] = data.get("train", [])[:n]
    return names if names else None
