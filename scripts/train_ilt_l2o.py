#!/usr/bin/env python3
"""Meta-train the ILT L2O checkpoint used for README figures (`ilt_l2o.pt`).

**Default training:** ICCAD-style **synthetic** ``.glp`` clips under ``data/synthetic_glp_train/``
(same raster path as ``benchmarks/iccad2013/M1_test*.glp``). Generate them with
``scripts/prepare_ilt_training_data.py synthetic``.

**ICCAD eval:** ``benchmarks/iccad2013/M1_test*.glp`` (separate ``eval_ilt_benchmark_table.py``).

Example::

    python scripts/prepare_ilt_training_data.py synthetic --n 256 --out data/synthetic_glp_train
    python scripts/train_ilt_l2o.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from l2o.training import train_optimizer
from l2o.models import OptimizerNeuralCoordinatewiseGradEnc

from ilt.datasets.dataset import ICCADGlpDataset, collate_ilt_batch
from ilt.datasets.lithobench_loader import LithoBenchTargetDataset, discover_metalset_dir
from ilt.datasets.metalset_split import default_metalset_split_path, load_or_create_metalset_split


def _glp_train_val_paths(
    paths: list[Path],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if not paths:
        raise ValueError("empty .glp path list")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = random.Random(seed)
    shuf = paths[:]
    rng.shuffle(shuf)
    n_train = max(1, int(len(shuf) * (1.0 - val_fraction)))
    train_paths = shuf[:n_train]
    val_paths = shuf[n_train:]
    if not val_paths:
        val_paths = [train_paths[-1]]
        train_paths = train_paths[:-1]
    if not train_paths:
        raise ValueError("train split empty")
    return train_paths, val_paths


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train ILT L2O (default: synthetic .glp; optional --metalset for LithoBench PNGs)"
    )
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "ilt_l2o.pt")
    p.add_argument(
        "--metalset",
        action="store_true",
        help="Train on LithoBench MetalSet PNGs instead of synthetic .glp",
    )
    p.add_argument(
        "--glp-train-dir",
        type=Path,
        default=ROOT / "data" / "synthetic_glp_train",
        help="Directory of ICCAD-style .glp files (default: data/synthetic_glp_train)",
    )
    p.add_argument(
        "--lithobench-root",
        type=str,
        default=None,
        help="(With --metalset) directory containing MetalSet/target/*.png",
    )
    p.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="(With --metalset) train/val split JSON (default: data/metalset_train_val_split.json)",
    )
    p.add_argument(
        "--regenerate-metalset-split",
        action="store_true",
        help="(With --metalset) ignore existing split JSON and rewrite",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Validation fraction (synthetic: shuffled .glp list; metalset: split JSON)",
    )
    p.add_argument(
        "--grid",
        type=int,
        default=32,
        help="H=W; match eval_ilt_benchmark_table.py / gen_readme_figures.py",
    )
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument(
        "--inner-steps",
        type=int,
        default=512,
        help="Stored as max_unroll_steps / inner_eval (match benchmark eval).",
    )
    p.add_argument(
        "--meta-unroll",
        type=int,
        default=128,
        help="Truncated BPTT unroll per meta step.",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--meta-lr", type=float, default=0.0025)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=2048,
        help="(With --metalset) cap MetalSet PNGs before split.0 = all.",
    )
    p.add_argument(
        "--max-train-files",
        type=int,
        default=None,
        help="Cap number of .glp files used (after sort; default: all in --glp-train-dir)",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-outer-chunks", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save intermediate checkpoints every N epochs (0 = only final --out)",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints",
        help="Directory for periodic checkpoints when --checkpoint-every > 0",
    )
    p.add_argument(
        "--val-max-batches",
        type=int,
        default=None,
        help="Cap validation batches per metric",
    )
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    pin = torch.cuda.is_available()
    ck_every = int(args.checkpoint_every)
    n_unroll = max(1, min(int(args.meta_unroll), int(args.inner_steps)))
    dim = args.grid * args.grid
    learned = OptimizerNeuralCoordinatewiseGradEnc(
        dim, hidden_size=args.hidden, max_unroll_steps=float(args.inner_steps)
    ).to(device)

    if args.metalset:
        ms = Path(args.lithobench_root) if args.lithobench_root else None
        max_train_samples = args.max_train_samples
        if max_train_samples == 0:
            max_train_samples = None
        split_path = args.split_file if args.split_file is not None else default_metalset_split_path(ROOT)
        train_ratio = 1.0 - float(args.val_fraction)
        if not 0.0 < train_ratio < 1.0:
            raise SystemExit("--val-fraction must be in (0, 1)")
        try:
            metal = discover_metalset_dir(ms)
        except FileNotFoundError as e:
            raise SystemExit(
                "LithoBench MetalSet not found. Unpack under ./data/MetalSet/ or set LITHOBENCH_ROOT, "
                "or drop --metalset to use synthetic .glp.\n"
                f"Detail: {e}"
            ) from e
        train_paths, val_paths = load_or_create_metalset_split(
            metal,
            split_path,
            train_ratio=train_ratio,
            shuffle_seed=args.seed,
            max_samples=max_train_samples,
            force=bool(args.regenerate_metalset_split),
        )
        if not val_paths:
            raise SystemExit("Validation split is empty; increase MetalSet size or lower --val-fraction.")
        train_ds = LithoBenchTargetDataset(
            train_paths, grid_hw=(args.grid, args.grid), init_noise=0.4, seed=args.seed
        )
        val_ds = LithoBenchTargetDataset(
            val_paths, grid_hw=(args.grid, args.grid), init_noise=0.4, seed=args.seed + 1
        )
        ck_meta = {
            "architecture": "gradenc",
            "dim": dim,
            "grid": args.grid,
            "hidden_size": args.hidden,
            "max_unroll_steps": float(args.inner_steps),
            "inner_eval_steps": args.inner_steps,
            "train_source": "lithobench_metalset_png",
            "n_train_clips": len(train_paths),
            "max_train_samples": args.max_train_samples,
            "metalset_split_file": str(split_path.resolve()),
        }
        print(
            f"[train] MetalSet | split: {split_path} | train: {len(train_paths)}, val: {len(val_paths)}",
            flush=True,
        )
    else:
        train_dir = args.glp_train_dir.expanduser().resolve()
        paths = sorted(train_dir.glob("*.glp"))
        if args.max_train_files is not None:
            paths = paths[: int(args.max_train_files)]
        if len(paths) < 16:
            raise SystemExit(
                f"Need synthetic .glp clips in {train_dir} (got {len(paths)}). Run:\n"
                f"  python scripts/prepare_ilt_training_data.py synthetic "
                f"--n 256 --out {train_dir} --seed {args.seed}"
            )
        train_paths, val_paths = _glp_train_val_paths(paths, val_fraction=args.val_fraction, seed=args.seed)
        train_ds = ICCADGlpDataset(
            train_paths, grid_hw=(args.grid, args.grid), init_noise=0.4, seed=args.seed
        )
        val_ds = ICCADGlpDataset(
            val_paths, grid_hw=(args.grid, args.grid), init_noise=0.4, seed=args.seed + 1
        )
        ck_meta = {
            "architecture": "gradenc",
            "dim": dim,
            "grid": args.grid,
            "hidden_size": args.hidden,
            "max_unroll_steps": float(args.inner_steps),
            "inner_eval_steps": args.inner_steps,
            "train_source": "synthetic_iccad_style_glp",
            "n_train_clips": len(train_paths),
            "glp_train_dir": str(train_dir),
        }
        print(
            f"[train] synthetic .glp | dir: {train_dir} | train: {len(train_paths)}, val: {len(val_paths)}",
            flush=True,
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ilt_batch,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ilt_batch,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    train_optimizer(
        learned,
        train_dl,
        device,
        n_epochs=args.epochs,
        meta_lr=args.meta_lr,
        n_outer_chunks=args.n_outer_chunks,
        max_unroll=n_unroll,
        fixed_unroll=n_unroll,
        grad_scale_augment=True,
        meta_weight_decay=0.0,
        extra_init_noise_scale=1.0,
        val_dataloader=val_dl if ck_every > 0 else None,
        checkpoint_every=ck_every,
        checkpoint_dir=args.checkpoint_dir if ck_every > 0 else None,
        checkpoint_meta=ck_meta if ck_every > 0 else None,
        val_max_batches=args.val_max_batches,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **ck_meta,
            "state_dict": learned.state_dict(),
            "meta_epochs": args.epochs,
        },
        args.out,
    )
    print("Saved", args.out)


if __name__ == "__main__":
    main()
