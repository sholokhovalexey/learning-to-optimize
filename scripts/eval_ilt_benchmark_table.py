#!/usr/bin/env python3
"""Tune handcrafted optimizers, then benchmark AdaGrad / RMSprop / Adam / L2O on ICCAD clips.

**Evaluation protocol** (LithoBench-style, Zheng et al. NeurIPS 2023): fixed inner unroll
(``--inner-steps``, default 512) for **all** methods; reported **L2** and **PVB** are computed on
**binarized** nominal / corner prints and target after bilinear upsampling to ``--eval-size`` (default
2048), then :math:`\\mathcal{L} = \\mathrm{L2} + \\lambda\\mathrm{PVB}` with the same
``pvb_weight`` as ``ILTOptimizee``. Uses the repo's toy SOCS litho proxy (not full industrial
simulation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimization import OptimizerAdaGrad, OptimizerAdam, OptimizerRMSprop
from problems.ilt import ILTOptimizee

from l2o.checkpoint import load_learned_optimizer_checkpoint
from ilt.datasets.dataset import benchmark_iccad_glp_paths
from ilt.datasets.lithobench_loader import discover_metalset_dir, list_target_pngs, load_png_target
from ilt.datasets.metalset_split import train_basenames_for_tune
from ilt.eval.evaluation import run_inner_optimization
from ilt.eval.metrics import lithobench_style_metrics
from ilt.io.glp_raster import load_glp_path


def _load_ckpt(path: Path, device: torch.device):
    return load_learned_optimizer_checkpoint(path, device, eval_mode=True)


def _build_tune_tasks_from_paths(
    paths: list[Path],
    grid: int,
    n_tasks: int,
    init_noise: float,
    seed: int,
    device: torch.device,
) -> tuple[list[ILTOptimizee], list[torch.Tensor]]:
    paths = paths[:n_tasks]
    if len(paths) < n_tasks:
        print(f"Warning: only {len(paths)} tune tasks available (requested {n_tasks}).", file=sys.stderr)
    optimizees: list[ILTOptimizee] = []
    x0s: list[torch.Tensor] = []
    torch.manual_seed(seed)
    dim = grid * grid
    for i, p in enumerate(paths):
        tgt = load_png_target(p, grid, grid).to(device)
        optee = ILTOptimizee(tgt.unsqueeze(0)).to(device)
        optimizees.append(optee)
        g = torch.Generator(device=device)
        g.manual_seed(seed + i * 9973)
        x0s.append(torch.randn(1, dim, device=device, generator=g) * init_noise)
    return optimizees, x0s


def _build_tune_tasks(
    metalset_dir: Path | None,
    grid: int,
    n_tasks: int,
    init_noise: float,
    seed: int,
    device: torch.device,
) -> tuple[list[ILTOptimizee], list[torch.Tensor]]:
    ms = discover_metalset_dir(metalset_dir)
    paths = list_target_pngs(ms)[:n_tasks]
    return _build_tune_tasks_from_paths(paths, grid, n_tasks, init_noise, seed, device)


def _build_tune_tasks_from_glp(
    glp_dir: Path,
    grid: int,
    n_tasks: int,
    init_noise: float,
    seed: int,
    device: torch.device,
) -> tuple[list[ILTOptimizee], list[torch.Tensor]]:
    """LR tuning on ICCAD-style ``.glp`` clips (e.g. synthetic training set)."""
    paths = sorted(glp_dir.glob("*.glp"))[:n_tasks]
    if len(paths) < n_tasks:
        print(f"Warning: only {len(paths)} .glp tune tasks in {glp_dir} (requested {n_tasks}).", file=sys.stderr)
    optimizees: list[ILTOptimizee] = []
    x0s: list[torch.Tensor] = []
    torch.manual_seed(seed)
    dim = grid * grid
    for i, p in enumerate(paths):
        tgt = load_glp_path(str(p), grid, grid).to(device)
        optee = ILTOptimizee(tgt.unsqueeze(0)).to(device)
        optimizees.append(optee)
        g = torch.Generator(device=device)
        g.manual_seed(seed + i * 9973)
        x0s.append(torch.randn(1, dim, device=device, generator=g) * init_noise)
    return optimizees, x0s


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark ILT optimizers (AdaGrad, RMSprop, Adam, learned L2O) on ICCAD .glp clips."
    )
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "ilt_l2o.pt")
    p.add_argument("--lithobench-root", type=str, default=None)
    p.add_argument(
        "--metalset-split-file",
        type=Path,
        default=None,
        help="Train/val split JSON: use **train** basenames for LR tuning (default: data/metalset_train_val_split.json if present)",
    )
    p.add_argument(
        "--tune-glp-dir",
        type=Path,
        default=None,
        help="If set (or when MetalSet is missing), tune Adam/GD LR on .glp clips here (e.g. data/synthetic_glp_train).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--grid", type=int, default=32)
    p.add_argument(
        "--inner-steps",
        type=int,
        default=512,
        help="Inner unroll steps for every optimizer (same budget; LithoBench-style fair comparison)",
    )
    p.add_argument(
        "--inner-steps-adagrad",
        type=int,
        default=None,
        help="Override AdaGrad steps (default: same as --inner-steps)",
    )
    p.add_argument(
        "--inner-steps-rmsprop",
        type=int,
        default=None,
        help="Override RMSprop steps (default: same as --inner-steps)",
    )
    p.add_argument(
        "--tune-steps",
        type=int,
        default=None,
        help="Steps for LR grid search (default: min(128, --inner-steps); use --inner-steps for full-budget tuning)",
    )
    p.add_argument(
        "--eval-size",
        type=int,
        default=2048,
        help="LithoBench-style metric resolution (binarized images upsampled to this side length)",
    )
    p.add_argument("--tune-tasks", type=int, default=8, help="Number of MetalSet targets for LR tuning")
    p.add_argument("--init-noise", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", type=Path, default=ROOT / "docs" / "ilt_benchmark_metrics.json")
    p.add_argument(
        "--markdown-out",
        type=Path,
        default=ROOT / "docs" / "ilt_benchmark_table.md",
        help="Write a GitHub-flavored markdown table (for README excerpts).",
    )
    args = p.parse_args()
    inner_steps_val = int(args.inner_steps)
    tune_steps_eff = int(args.tune_steps) if args.tune_steps is not None else min(128, inner_steps_val)

    device = torch.device(args.device)
    ms_arg = Path(args.lithobench_root) if args.lithobench_root else None
    split_candidate = args.metalset_split_file
    if split_candidate is None:
        split_candidate = ROOT / "data" / "metalset_train_val_split.json"
    tune_glp = args.tune_glp_dir
    if tune_glp is None:
        syn = ROOT / "data" / "synthetic_glp_train"
        if syn.is_dir() and list(syn.glob("*.glp")):
            tune_glp = syn

    tune_optees: list[ILTOptimizee] = []
    tune_x0s: list[torch.Tensor] = []
    explicit_glp_tune = args.tune_glp_dir is not None

    # Explicit --tune-glp-dir wins (e.g. synthetic .glp matching train distribution).
    if explicit_glp_tune and tune_glp is not None and Path(tune_glp).is_dir():
        tune_optees, tune_x0s = _build_tune_tasks_from_glp(
            Path(tune_glp).resolve(), args.grid, args.tune_tasks, args.init_noise, args.seed, device
        )
        print(f"[tune] Using .glp directory for LR search: {tune_glp} ({len(tune_optees)} tasks)", flush=True)

    split_for_tune = split_candidate if split_candidate.is_file() else None
    train_names = train_basenames_for_tune(split_for_tune, args.tune_tasks)
    if not tune_optees:
        try:
            if train_names is not None:
                ms = discover_metalset_dir(ms_arg)
                paths = [ms / "target" / n for n in train_names]
                tune_optees, tune_x0s = _build_tune_tasks_from_paths(
                    paths, args.grid, args.tune_tasks, args.init_noise, args.seed, device
                )
                print(
                    f"[tune] MetalSet train split from {split_candidate} ({len(tune_optees)} tasks)",
                    flush=True,
                )
            else:
                raise FileNotFoundError("no split file")
        except FileNotFoundError:
            try:
                tune_optees, tune_x0s = _build_tune_tasks(
                    ms_arg, args.grid, args.tune_tasks, args.init_noise, args.seed, device
                )
                print(f"[tune] Using LithoBench MetalSet PNG targets ({len(tune_optees)} tasks)")
            except FileNotFoundError:
                tune_optees, tune_x0s = [], []
    if not tune_optees and not explicit_glp_tune and tune_glp is not None and Path(tune_glp).is_dir():
        tune_optees, tune_x0s = _build_tune_tasks_from_glp(
            Path(tune_glp).resolve(), args.grid, args.tune_tasks, args.init_noise, args.seed, device
        )
        print(f"[tune] Using .glp directory for LR search: {tune_glp} ({len(tune_optees)} tasks)")
    if not tune_optees:
        raise SystemExit(
            "No LR-tuning tasks: unpack LithoBench MetalSet (./data/MetalSet/target/*.png) or set "
            "LITHOBENCH_ROOT / --lithobench-root, or pass --tune-glp-dir with synthetic .glp clips "
            "(e.g. data/synthetic_glp_train/)."
        )
    if not tune_optees:
        raise SystemExit("No tune tasks for LR search.")

    print(
        f"[tune] LR search unroll: {tune_steps_eff} steps (eval uses {inner_steps_val}; "
        "set --tune-steps to match --inner-steps for full-budget hyperparameter search)",
        flush=True,
    )

    dim = args.grid * args.grid
    lrs = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.1,
        0.12,
        0.15,
        0.2,
    ]

    def _tune_pick_lr(which: str) -> float:
        best_lr = lrs[0]
        best_score = float("inf")
        for lr in lrs:
            finals: list[float] = []
            for optee, x0 in zip(tune_optees, tune_x0s, strict=True):
                x = x0.clone()
                if which == "adam":
                    opt = OptimizerAdam(dim, lr=lr).to(device)
                    opt.reset(device=device)
                elif which == "adagrad":
                    opt = OptimizerAdaGrad(dim, lr=lr).to(device)
                    opt.reset(device=device)
                elif which == "rmsprop":
                    opt = OptimizerRMSprop(dim, lr=lr).to(device)
                    opt.reset(device=device)
                else:
                    raise ValueError(which)
                for _ in range(tune_steps_eff):
                    _, grad = optee.loss_and_grad(x)
                    with torch.no_grad():
                        x = x + opt(grad.detach())
                finals.append(float(optee(x).mean().item()))
            worst = max(finals)
            if worst < best_score:
                best_score = worst
                best_lr = lr
        return best_lr

    lr_adam = _tune_pick_lr("adam")
    lr_ada = _tune_pick_lr("adagrad")
    lr_rms = _tune_pick_lr("rmsprop")
    print(f"[tune] Adam lr (minimax over {len(tune_optees)} tune tasks): {lr_adam}")
    print(f"[tune] AdaGrad lr (minimax): {lr_ada}")
    print(f"[tune] RMSprop lr (minimax): {lr_rms}")

    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint {args.checkpoint}")

    learned, ckpt = _load_ckpt(args.checkpoint, device)
    learned.eval()
    inner_base = int(args.inner_steps)
    inner_adam = inner_base
    inner_l2o = inner_base
    inner_ada = int(args.inner_steps_adagrad) if args.inner_steps_adagrad is not None else inner_base
    inner_rms = int(args.inner_steps_rmsprop) if args.inner_steps_rmsprop is not None else inner_base
    eval_sz = int(args.eval_size)

    bench_paths = benchmark_iccad_glp_paths(ROOT)
    if not bench_paths:
        raise SystemExit(f"No benchmark .glp under {ROOT / 'benchmarks' / 'iccad2013'}")

    rows: list[dict[str, float | str]] = []
    torch.manual_seed(args.seed + 999)

    for pth in bench_paths:
        tgt = load_glp_path(str(pth), args.grid, args.grid).to(device)
        x0 = torch.randn(1, dim, device=device) * args.init_noise
        optee = ILTOptimizee(tgt.unsqueeze(0)).to(device)

        adam = OptimizerAdam(dim, lr=lr_adam).to(device)
        x_a, _ = run_inner_optimization(adam, optee, x0, inner_adam, device)
        m_a = lithobench_style_metrics(optee, x_a, eval_size=eval_sz)

        optee_ada = ILTOptimizee(tgt.unsqueeze(0)).to(device)
        adag = OptimizerAdaGrad(dim, lr=lr_ada).to(device)
        x_ada, _ = run_inner_optimization(adag, optee_ada, x0, inner_ada, device)
        m_ada = lithobench_style_metrics(optee_ada, x_ada, eval_size=eval_sz)

        optee_rms = ILTOptimizee(tgt.unsqueeze(0)).to(device)
        rms = OptimizerRMSprop(dim, lr=lr_rms).to(device)
        x_rms, _ = run_inner_optimization(rms, optee_rms, x0, inner_rms, device)
        m_rms = lithobench_style_metrics(optee_rms, x_rms, eval_size=eval_sz)

        optee_l = ILTOptimizee(tgt.unsqueeze(0)).to(device)
        learned.reset()
        x = x0.clone()
        for t in range(inner_l2o):
            loss_tensor, grad = optee_l.loss_and_grad(x)
            g = grad.detach()
            step = torch.tensor(float(t), device=device, dtype=torch.float32)
            x = x + learned(g, loss=loss_tensor, step=step)
        m_l = lithobench_style_metrics(optee_l, x, eval_size=eval_sz)

        rows.append(
            {
                "clip": pth.name,
                "adagrad_total": m_ada.total,
                "adagrad_l2": m_ada.l2,
                "adagrad_pvb": m_ada.pvb,
                "rmsprop_total": m_rms.total,
                "rmsprop_l2": m_rms.l2,
                "rmsprop_pvb": m_rms.pvb,
                "adam_total": m_a.total,
                "adam_l2": m_a.l2,
                "adam_pvb": m_a.pvb,
                "l2o_total": m_l.total,
                "l2o_l2": m_l.l2,
                "l2o_pvb": m_l.pvb,
            }
        )

    def _mean(key: str) -> float:
        return sum(float(r[key]) for r in rows) / len(rows)

    summary = {
        "adagrad_mean_total": _mean("adagrad_total"),
        "rmsprop_mean_total": _mean("rmsprop_total"),
        "adam_mean_total": _mean("adam_total"),
        "l2o_mean_total": _mean("l2o_total"),
        "adagrad_mean_l2": _mean("adagrad_l2"),
        "rmsprop_mean_l2": _mean("rmsprop_l2"),
        "adam_mean_l2": _mean("adam_l2"),
        "l2o_mean_l2": _mean("l2o_l2"),
        "adagrad_mean_pvb": _mean("adagrad_pvb"),
        "rmsprop_mean_pvb": _mean("rmsprop_pvb"),
        "adam_mean_pvb": _mean("adam_pvb"),
        "l2o_mean_pvb": _mean("l2o_pvb"),
        "lr_adam": lr_adam,
        "lr_adagrad": lr_ada,
        "lr_rmsprop": lr_rms,
        "checkpoint": str(args.checkpoint),
        "inner_steps": inner_base,
        "inner_steps_adam": inner_adam,
        "inner_steps_l2o": inner_l2o,
        "inner_steps_adagrad": inner_ada,
        "inner_steps_rmsprop": inner_rms,
        "tune_steps_lr_search": tune_steps_eff,
        "eval_size_bin_bilinear": eval_sz,
        "methodology": (
            "LithoBench-style reported metrics: bilinear upsample to eval_size, threshold 0.5, "
            "then mean L2 / PVB; same inner unroll for all optimizers; toy SOCS litho in ILTOptimizee."
        ),
        "grid": args.grid,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"per_clip": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"Wrote {args.json_out}")

    md_lines = [
        "<!-- Generated by: python scripts/eval_ilt_benchmark_table.py -->",
        "",
        "### Mean metrics on ICCAD 2013 benchmark clips (toy litho, LithoBench-style reporting)",
        "",
        f"| Optimizer | Mean total $\\mathcal{{L}}$ | Mean L2 (bin., {eval_sz} x {eval_sz}) | Mean PVB (bin.) |",
        "|-----------|------------------------------|---------------------------|-----------------|",
        f"| AdaGrad (lr={lr_ada:g}) | {summary['adagrad_mean_total']:.6f} | {summary['adagrad_mean_l2']:.6f} | {summary['adagrad_mean_pvb']:.6f} |",
        f"| RMSprop (lr={lr_rms:g}) | {summary['rmsprop_mean_total']:.6f} | {summary['rmsprop_mean_l2']:.6f} | {summary['rmsprop_mean_pvb']:.6f} |",
        f"| Adam (lr={lr_adam:g}) | {summary['adam_mean_total']:.6f} | {summary['adam_mean_l2']:.6f} | {summary['adam_mean_pvb']:.6f} |",
        f"| Learned L2O | {summary['l2o_mean_total']:.6f} | {summary['l2o_mean_l2']:.6f} | {summary['l2o_mean_pvb']:.6f} |",
        "",
        f"Fixed **{inner_base}** inner steps per optimizer on each clip; LR tuning uses **{tune_steps_eff}** steps (see `--tune-steps`). "
        "Metrics: binarized resist/target at `eval_size` (see `--eval-size`).",
        "",
    ]
    md_text = "\n".join(md_lines)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(md_text, encoding="utf-8")
        print(f"Wrote {args.markdown_out}")

    print("\n" + md_text)


if __name__ == "__main__":
    main()
