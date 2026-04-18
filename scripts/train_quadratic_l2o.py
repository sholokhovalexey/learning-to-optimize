#!/usr/bin/env python3
"""Meta-train the quadratic README L2O model and save a checkpoint.

Figure generation (`gen_readme_figures.py`) loads this checkpoint only - no training.

    python scripts/train_quadratic_l2o.py

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from l2o.training import train_optimizer
from problems.quadratic import RandomQuadraticFunctionsDataset, collate_quadratic_batch
from l2o.models import OptimizerNeuralCoordinatewiseGradEnc


def main() -> None:
    p = argparse.ArgumentParser(description="Meta-train quadratic L2O and save checkpoints/quadratic_l2o.pt")
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "quadratic_l2o.pt", help="Checkpoint path")
    p.add_argument("--meta-epochs", type=int, default=100, help="Full-scale default matches README eval recipe")
    p.add_argument("--n-tasks", type=int, default=512)
    p.add_argument("--inner-steps", type=int, default=64, help="Unroll horizon (stored in checkpoint; eval can differ)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--x-dim", type=int, default=8)
    p.add_argument("--y-dim", type=int, default=8)
    p.add_argument("--meta-lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=1, help="Dataset seed")
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--condition-number", type=float, default=1e3)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save checkpoint every N epochs (0 = off); filename includes epoch and val loss",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Hold out this fraction of tasks for validation metrics (min 1 task)",
    )
    p.add_argument("--val-max-batches", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    ds = RandomQuadraticFunctionsDataset(
        args.n_tasks,
        args.x_dim,
        args.y_dim,
        device=device,
        seed=args.seed,
        noise_std=args.noise_std,
        ill_conditioned=False,
        condition_number=args.condition_number,
    )
    ck_every = int(args.checkpoint_every)
    if ck_every > 0 and args.n_tasks < 2:
        raise SystemExit("--checkpoint-every requires --n-tasks >= 2 for a validation split")
    if ck_every > 0:
        n_val = min(max(1, int(round(args.n_tasks * args.val_fraction))), max(0, args.n_tasks - 1))
        n_train = args.n_tasks - n_val
        train_dl = DataLoader(
            Subset(ds, range(n_train)),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_quadratic_batch,
        )
        val_dl = DataLoader(
            Subset(ds, range(n_train, args.n_tasks)),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_quadratic_batch,
        )
    else:
        train_dl = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_quadratic_batch
        )
        val_dl = None
    opt_learned = OptimizerNeuralCoordinatewiseGradEnc(
        args.x_dim, hidden_size=args.hidden, max_unroll_steps=float(args.inner_steps)
    ).to(device)
    ck_meta = {
        "architecture": "gradenc",
        "dim": args.x_dim,
        "hidden_size": args.hidden,
        "max_unroll_steps": float(args.inner_steps),
        "inner_eval_steps": args.inner_steps,
        "y_dim": args.y_dim,
        "meta_dataset": "RandomQuadraticFunctionsDataset",
        "extra_init_noise_scale": 1.0,
    }
    train_optimizer(
        opt_learned,
        train_dl,
        device,
        n_epochs=args.meta_epochs,
        meta_lr=args.meta_lr,
        n_outer_chunks=1,
        grad_scale_augment=False,
        max_unroll=args.inner_steps,
        fixed_unroll=args.inner_steps,
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
            "state_dict": opt_learned.state_dict(),
            "architecture": "gradenc",
            "dim": args.x_dim,
            "hidden_size": args.hidden,
            "max_unroll_steps": float(args.inner_steps),
            "meta_epochs": args.meta_epochs,
            "n_tasks": args.n_tasks,
            "inner_eval_steps": args.inner_steps,
            "meta_dataset": "RandomQuadraticFunctionsDataset",
            "extra_init_noise_scale": 1.0,
            "y_dim": args.y_dim,
        },
        args.out,
    )
    print("Saved", args.out)


if __name__ == "__main__":
    main()
