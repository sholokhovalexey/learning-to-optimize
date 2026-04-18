#!/usr/bin/env python3
"""Generate PNGs under docs/images/ for README using **pre-trained** checkpoints.

Training is separate - run once, then redraw figures with different inference/plot settings:

    python scripts/train_quadratic_l2o.py
    python scripts/train_ilt_l2o.py
    python scripts/gen_readme_figures.py --only-quadratic
    python scripts/gen_readme_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from l2o.tuning import tune_adagrad_learning_rate, tune_adam_learning_rate, tune_rmsprop_learning_rate
from problems.quadratic import make_aligned_quadratic_problem, quadratic_ls_minimizer
from l2o.models import OptimizerNeuralCoordinatewiseGradEnc
from optimization import ADAM_EPS, OptimizerAdaGrad, OptimizerAdam, OptimizerRMSprop
from problems.ilt import ILTOptimizee

from l2o.checkpoint import load_learned_optimizer_checkpoint
from ilt.datasets.download import default_benchmark_iccad_dir, download_iccad_glp_files
from ilt.io.glp_raster import load_glp_path
from ilt.viz.plotting import mask_and_printed_nominal

BENCH_ICCAD_DIR = ROOT / "benchmarks" / "iccad2013"


def _load_learned_optimizer_from_ckpt(
    ckpt_path: Path, device: torch.device, train_hint: str
) -> tuple[torch.nn.Module, dict]:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing {ckpt_path}. Train first: {train_hint}")
    net, ckpt = load_learned_optimizer_checkpoint(ckpt_path, device, eval_mode=True)
    return net, ckpt


def load_quadratic_l2o(ckpt_path: Path, device: torch.device) -> tuple[OptimizerNeuralCoordinatewiseGradEnc, dict]:
    return _load_learned_optimizer_from_ckpt(ckpt_path, device, "python scripts/train_quadratic_l2o.py")


def load_ilt_l2o(ckpt_path: Path, device: torch.device) -> tuple[OptimizerNeuralCoordinatewiseGradEnc, dict]:
    return _load_learned_optimizer_from_ckpt(ckpt_path, device, "python scripts/train_ilt_l2o.py")


def fig_quadratic_l2o(
    out: Path,
    ckpt_path: Path,
    *,
    device: torch.device | None = None,
    inner_eval_steps: int | None = None,
    eval_optee_seed: int = 42,
    x0_seed: int = 12345,
    tune_optee_seed: int = 999,
    tune_x0_seed: int = 54321,
    dpi: int = 150,
    grid_n: int = 90,
) -> None:
    """2D projection (loss level sets) + log-loss curves; loads a trained quadratic L2O checkpoint."""
    device = device or torch.device("cpu")

    learned, ckpt = load_quadratic_l2o(ckpt_path, device)
    x_dim = int(ckpt["dim"])
    dim = x_dim
    y_dim = int(ckpt.get("y_dim", x_dim))
    if dim < 2:
        raise ValueError(
            f"Trajectory projection plot needs inner dimension >= 2 (uses first two coordinates); got dim={dim} from checkpoint."
        )
    steps = int(inner_eval_steps) if inner_eval_steps is not None else int(ckpt["inner_eval_steps"])
    n_tasks = int(ckpt.get("n_tasks", 512))

    # Evaluation task (reported trajectories / losses).
    optee, _ = make_aligned_quadratic_problem(y_dim, x_dim, device, seed=eval_optee_seed)
    torch.manual_seed(x0_seed)
    x0 = torch.randn(1, dim, device=device) * 0.5

    # Tune Adam's lr on a *different* random quadratic so we do not oracle-tune on the eval task.
    optee_tune, _ = make_aligned_quadratic_problem(y_dim, x_dim, device, seed=tune_optee_seed)
    torch.manual_seed(tune_x0_seed)
    x0_tune = torch.randn(1, dim, device=device) * 0.5

    def trajectory_and_losses(opt: torch.nn.Module) -> tuple[np.ndarray, np.ndarray]:
        x = x0.clone()
        if isinstance(opt, OptimizerAdam):
            opt.reset(device=device)
        else:
            opt.reset()
        pts = [x[0, :2].detach().cpu().numpy().copy()]
        losses: list[float] = [optee(x).mean().item()]
        for t in range(steps):
            lt, g = optee.loss_and_grad(x)
            st = torch.tensor(float(t), device=device, dtype=torch.float32)
            x = x + opt(g.detach(), loss=lt, step=st)
            pts.append(x[0, :2].detach().cpu().numpy().copy())
            losses.append(optee(x).mean().item())
        return np.stack(pts, axis=0), np.asarray(losses, dtype=np.float64)

    best_lr = tune_adam_learning_rate(
        [0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4],
        optee_tune,
        x0_tune,
        n_steps=steps,
        device=device,
        eps=ADAM_EPS,
    )
    adam = OptimizerAdam(dim, lr=best_lr, eps=ADAM_EPS).to(device)
    learned = learned.to(device)
    learned.eval()
    learned.reset()

    p_adam, loss_adam = trajectory_and_losses(adam)
    learned.reset()
    p_l2o, loss_l2o = trajectory_and_losses(learned)

    final_adam, final_l2o = float(loss_adam[-1]), float(loss_l2o[-1])
    print(
        f"Final mean loss (eval task) - Adam: {final_adam:.6g}, Learned L2O: {final_l2o:.6g} "
        f"(Adam lr tuned on a different quadratic; eval optee seed={eval_optee_seed}; "
        f"meta-train={ckpt.get('meta_dataset', '?')} {n_tasks} tasks from checkpoint)"
    )
    if final_l2o >= final_adam:
        print("Warning: learned optimizer did not beat Adam final loss; try raising meta_epochs or hidden_size.")

    # True LS minimizer (first two coords) for the eval task; differs from planted draw when b is noisy.
    x_ls = quadratic_ls_minimizer(optee.A, optee.b)

    # Level sets in the (x1, x2) plane with x3..xd fixed to the **LS minimizer** tail so this slice
    # contains the global optimum; then the 2D minimum is (x_ls[0], x_ls[1]) and contours surround it.
    # (Fixing the tail to x0 instead would show a different quadratic slice whose minimum is not the
    # projected LS point - misleading next to the green LS marker.)
    x_tail = x_ls[2:].detach()
    xy_stack = np.vstack(
        [p_adam[:, :2], p_l2o[:, :2], x0[0, :2].detach().cpu().numpy(), x_ls[:2].detach().cpu().numpy()]
    )
    span = float(np.max(xy_stack.max(axis=0) - xy_stack.min(axis=0)))
    pad = max(0.2 * span, 0.25)
    xmin, xmax = float(xy_stack[:, 0].min() - pad), float(xy_stack[:, 0].max() + pad)
    ymin, ymax = float(xy_stack[:, 1].min() - pad), float(xy_stack[:, 1].max() + pad)
    gx = np.linspace(xmin, xmax, grid_n)
    gy = np.linspace(ymin, ymax, grid_n)
    Xg, Yg = np.meshgrid(gx, gy)
    flat = grid_n * grid_n
    x_grid = torch.zeros(flat, dim, device=device)
    x_grid[:, 0] = torch.from_numpy(Xg.ravel()).to(device=device, dtype=torch.float32)
    x_grid[:, 1] = torch.from_numpy(Yg.ravel()).to(device=device, dtype=torch.float32)
    x_grid[:, 2:] = x_tail.unsqueeze(0).expand(flat, -1)
    with torch.no_grad():
        # Same loss as ``optee(x)``; expand A,b to match grid batch (QuadraticOptimizee uses bmm).
        A = optee.A.expand(flat, -1, -1)
        bb = optee.b.expand(flat, -1)
        e = torch.bmm(A, x_grid.unsqueeze(-1)).squeeze(-1) - bb
        Z = (0.5 * (e**2).sum(-1)).cpu().numpy().reshape(grid_n, grid_n)

    fig, ax = plt.subplots(figsize=(5.4, 5))
    n_levels = 18
    zmin, zmax = float(Z.min()), float(Z.max())
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        zmax = zmin + 1e-12
    lev = np.linspace(zmin, zmax, n_levels + 2)[1:-1]
    # Filled level sets + line contours so $L$ is visible at README thumbnail size.
    ax.contourf(
        Xg,
        Yg,
        Z,
        levels=np.linspace(zmin, zmax, n_levels + 1),
        cmap="Blues",
        alpha=0.55,
        zorder=0,
    )
    ax.contour(
        Xg,
        Yg,
        Z,
        levels=lev,
        colors="0.2",
        linewidths=1.0,
        alpha=0.95,
        zorder=1,
    )
    ax.plot(p_adam[:, 0], p_adam[:, 1], "b.-", alpha=0.85, label="Adam (tuned lr)", zorder=2)
    ax.plot(p_l2o[:, 0], p_l2o[:, 1], "r.-", alpha=0.85, label="Learned", zorder=2)
    ax.scatter([x_ls[0].item()], [x_ls[1].item()], c="green", s=85, zorder=5, label=r"LS optimum $(x_1,x_2)$")
    ax.scatter([x0[0, 0].item()], [x0[0, 1].item()], c="k", s=45, zorder=4, label="init")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Quadratic inner problem")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close()
    print("Wrote", out)

    steps_axis = np.arange(len(loss_adam))
    fig2, ax2 = plt.subplots(figsize=(5.8, 4.2))
    ax2.semilogy(steps_axis, np.maximum(loss_adam, 1e-16), "b.-", alpha=0.85, label="Adam (tuned lr)", markevery=max(1, len(steps_axis) // 16))
    ax2.semilogy(steps_axis, np.maximum(loss_l2o, 1e-16), "r.-", alpha=0.85, label="Learned", markevery=max(1, len(steps_axis) // 16))
    ax2.set_xlabel("Inner step")
    ax2.set_ylabel(r"Mean inner loss $L(\theta_t)$ (log scale)")
    ax2.set_title("Convergence: same task and init")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    loss_out = out.parent / "l2o_quadratic_loss_log.png"
    fig2.savefig(loss_out, dpi=dpi)
    plt.close()
    print("Wrote", loss_out)


def fig_ilt_benchmark_four_way(
    out: Path,
    ckpt_path: Path,
    *,
    device: torch.device | None = None,
    inner_steps: int | None = None,
    inner_steps_adagrad: int | None = None,
    inner_steps_rmsprop: int | None = None,
    inner_steps_adam: int | None = None,
    inner_steps_l2o: int | None = None,
    tune_steps: int | None = None,
    handcrafted_lrs: list[float] | None = None,
    init_seed: int = 7,
    glp_name: str = "M1_test9.glp",
    dpi: int = 140,
) -> None:
    """AdaGrad, RMSprop, Adam, and learned L2O on one ICCAD benchmark clip (``./benchmarks/iccad2013``).

    Uses the **same inner unroll** for every method (LithoBench-style fair comparison; default 512 steps).
    Learning rates for handcrafted optimizers are chosen on a **short** unroll (``tune_steps``) unless
    overridden.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(init_seed)
    glp_path = BENCH_ICCAD_DIR / glp_name
    if not glp_path.is_file():
        raise FileNotFoundError(
            f"Missing benchmark layout {glp_path} (run: ./get_benchmarks.sh or python -m ilt.datasets.download)"
        )

    learned, ckpt = _load_learned_optimizer_from_ckpt(
        ckpt_path,
        device,
        "python scripts/train_ilt_l2o.py",
    )
    dim = int(ckpt["dim"])
    base_inner = int(inner_steps) if inner_steps is not None else int(ckpt.get("inner_eval_steps", 512))
    n_unified = max(1, base_inner)
    n_ada = int(inner_steps_adagrad) if inner_steps_adagrad is not None else n_unified
    n_rms = int(inner_steps_rmsprop) if inner_steps_rmsprop is not None else n_unified
    n_adam = int(inner_steps_adam) if inner_steps_adam is not None else n_unified
    n_l2o = int(inner_steps_l2o) if inner_steps_l2o is not None else n_unified
    dim_meta = int(ckpt["dim"])
    grid = int(ckpt.get("grid") or round(dim_meta**0.5))

    target_hw = load_glp_path(str(glp_path), grid, grid).to(device)
    optee = ILTOptimizee(target_hw.unsqueeze(0)).to(device)
    x0 = torch.randn(1, dim, device=device) * 0.4

    lrs = handcrafted_lrs or [
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
    ts = tune_steps if tune_steps is not None else min(128, max(n_unified, 1))
    best_adam = tune_adam_learning_rate(lrs, optee, x0.clone(), n_steps=ts, device=device)
    best_ada = tune_adagrad_learning_rate(lrs, optee, x0.clone(), n_steps=ts, device=device)
    best_rms = tune_rmsprop_learning_rate(lrs, optee, x0.clone(), n_steps=ts, device=device)
    print(
        f"[ilt figure] LR search (tune_steps={ts}): AdaGrad={best_ada:g}, RMSprop={best_rms:g}, "
        f"Adam={best_adam:g}; inner_steps (all) ada={n_ada}, rms={n_rms}, adam={n_adam}, l2o={n_l2o}",
        flush=True,
    )
    adam = OptimizerAdam(dim, lr=best_adam).to(device)
    adag = OptimizerAdaGrad(dim, lr=best_ada).to(device)
    rms = OptimizerRMSprop(dim, lr=best_rms).to(device)
    learned.eval()

    def run_loop(optimizer: torch.nn.Module, x_init: torch.Tensor, n_iter: int) -> torch.Tensor:
        x = x_init.clone().to(device)
        if isinstance(optimizer, OptimizerAdam):
            optimizer.reset(device=device)
        elif isinstance(optimizer, (OptimizerAdaGrad, OptimizerRMSprop)):
            optimizer.reset(device=device)
        else:
            optimizer.reset()
        for t in range(n_iter):
            loss_tensor, grad = optee.loss_and_grad(x)
            g = grad.detach()
            step = torch.tensor(float(t), device=device, dtype=torch.float32)
            try:
                x = x + optimizer(g, loss=loss_tensor, step=step)
            except TypeError:
                x = x + optimizer(g)
        return x

    x_ada = run_loop(adag, x0, n_ada)
    x_rms = run_loop(rms, x0, n_rms)
    x_adam = run_loop(adam, x0, n_adam)
    learned.reset()
    x = x0.clone()
    for t in range(n_l2o):
        loss_tensor, grad = optee.loss_and_grad(x)
        g = grad.detach()
        step = torch.tensor(float(t), device=device, dtype=torch.float32)
        x = x + learned(g, loss=loss_tensor, step=step)
    x_l2o = x

    m_ada, p_ada, tgt = mask_and_printed_nominal(optee, x_ada)
    m_rms, p_rms, _ = mask_and_printed_nominal(optee, x_rms)
    m_a, p_a, _ = mask_and_printed_nominal(optee, x_adam)
    m_l, p_l, _ = mask_and_printed_nominal(optee, x_l2o)

    fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.2))
    row0 = [
        ("Target (desired print)", tgt.numpy()),
        (f"Mask - AdaGrad", m_ada.numpy()),
        (f"Mask - RMSprop", m_rms.numpy()),
        (f"Mask - Adam", m_a.numpy()),
        (f"Mask - L2O ({n_l2o} steps)", m_l.numpy()),
    ]

    def _bin(u: torch.Tensor) -> torch.Tensor:
        return (u > 0.5).to(dtype=u.dtype)

    p_ada = _bin(p_ada)
    p_rms = _bin(p_rms)
    p_a = _bin(p_a)
    p_l = _bin(p_l)

    row1 = [
        ("Target (same)", tgt.numpy()),
        ("Printed (nom) - AdaGrad", p_ada.numpy()),
        ("Printed (nom) - RMSprop", p_rms.numpy()),
        ("Printed (nom) - Adam", p_a.numpy()),
        ("Printed (nom) - L2O", p_l.numpy()),
    ]
    for ax, (title, img) in zip(axes[0], row0):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=15)
        ax.axis("off")
    for ax, (title, img) in zip(axes[1], row1):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=15)
        ax.axis("off")
    fig.suptitle(
        f"ILT toy model (ICCAD 2013 clip): same inner budget ({n_adam} steps) for all optimizers",
        fontsize=20,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close()
    print("Wrote", out)


def fig_ilt_notebook_style(
    out: Path,
    ckpt_path: Path,
    *,
    device: torch.device | None = None,
    inner_steps: int | None = None,
    init_seed: int = 7,
    glp_name: str = "M1_test9.glp",
    dpi: int = 140,
) -> None:
    """Adam vs learned on ILTOptimizee; loads a pre-trained ILT README checkpoint."""
    device = device or torch.device("cpu")
    torch.manual_seed(init_seed)
    iccad_dir = default_benchmark_iccad_dir(ROOT)
    paths = download_iccad_glp_files(iccad_dir)
    assert paths, "No .glp files; run ./get_benchmarks.sh or python -m ilt.datasets.download"

    learned, ckpt = load_ilt_l2o(ckpt_path, device)
    dim = int(ckpt["dim"])
    INNER = int(inner_steps) if inner_steps is not None else int(ckpt["inner_eval_steps"])

    glp_path = iccad_dir / glp_name
    if not glp_path.is_file():
        raise FileNotFoundError(f"Missing {glp_path}")
    grid = int(ckpt["grid"])
    target_hw = load_glp_path(str(glp_path), grid, grid).to(device)
    optee = ILTOptimizee(target_hw.unsqueeze(0)).to(device)
    x0 = torch.randn(1, dim, device=device) * 0.4

    tune_steps = max(1, min(30, INNER // 2))
    best_lr = tune_adam_learning_rate([0.02, 0.05, 0.1, 0.2, 0.4], optee, x0.clone(), n_steps=tune_steps, device=device)
    adam = OptimizerAdam(dim, lr=best_lr).to(device)
    learned.eval()

    def run_loop(optimizer: torch.nn.Module, x_init: torch.Tensor):
        x = x_init.clone().to(device)
        if isinstance(optimizer, OptimizerAdam):
            optimizer.reset(device=device)
        else:
            optimizer.reset()
        for t in range(INNER):
            loss_tensor, grad = optee.loss_and_grad(x)
            g = grad.detach()
            step = torch.tensor(float(t), device=device, dtype=torch.float32)
            x = x + optimizer(g, loss=loss_tensor, step=step)
        return x

    x_adam = run_loop(adam, x0)
    x_l2o = run_loop(learned, x0)

    m_a, p_a, tgt = mask_and_printed_nominal(optee, x_adam)
    m_l, p_l, _ = mask_and_printed_nominal(optee, x_l2o)

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2))
    titles = [
        ("Target (desired print)", tgt.numpy()),
        ("Mask - Adam", m_a.numpy()),
        ("Printed (nominal) - Adam", p_a.numpy()),
        ("Target (same)", tgt.numpy()),
        ("Mask - learned L2O", m_l.numpy()),
        ("Printed (nominal) - L2O", p_l.numpy()),
    ]
    for ax, (title, img) in zip(axes.ravel(), titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        "ILT toy model: same init, Adam vs learned optimizer (fallback panel when benchmark .glp or checkpoint is missing)",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close()
    print("Wrote", out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate README figures (loads pre-trained checkpoints; use train_*.py to train first)"
    )
    p.add_argument(
        "--only-quadratic",
        action="store_true",
        help="Only quadratic L2O figures (skip ILT; no ICCAD .glp required).",
    )
    p.add_argument(
        "--only-ilt",
        action="store_true",
        help="Only ILT benchmark figure (skip quadratic panels; needs checkpoint + benchmarks/iccad2013).",
    )
    p.add_argument(
        "--quadratic-checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "quadratic_l2o.pt",
        help="README L2O (e.g. 8D meta-train) from scripts/train_quadratic_l2o.py",
    )
    p.add_argument(
        "--ilt-checkpoint",
        type=Path,
        default=None,
        help="ILT checkpoint (default: first existing metalset_best / metalset / ilt_l2o)",
    )
    p.add_argument(
        "--quadratic-inner-steps",
        type=int,
        default=None,
        help="Override inner unroll for quadratic eval/plots (default: from checkpoint)",
    )
    p.add_argument("--eval-optee-seed", type=int, default=42)
    p.add_argument("--x0-seed", type=int, default=12345, help="Init for quadratic eval trajectory")
    p.add_argument("--tune-optee-seed", type=int, default=999)
    p.add_argument("--tune-x0-seed", type=int, default=54321)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--grid-n", type=int, default=90, help="Contour grid resolution (quadratic 2D plot)")
    p.add_argument(
        "--ilt-inner-steps",
        type=int,
        default=512,
        help="Inner unroll for every ILT method in the figure (LithoBench-style same budget; default 512)",
    )
    p.add_argument("--ilt-glp", type=str, default="M1_test9.glp", help="Basename under ICCAD data dir")
    p.add_argument("--ilt-init-seed", type=int, default=7)
    p.add_argument(
        "--ilt-inner-steps-adagrad",
        type=int,
        default=None,
        help="Override AdaGrad steps (default: same as --ilt-inner-steps)",
    )
    p.add_argument(
        "--ilt-inner-steps-rmsprop",
        type=int,
        default=None,
        help="Override RMSprop steps (default: same as --ilt-inner-steps)",
    )
    p.add_argument(
        "--ilt-inner-steps-adam",
        type=int,
        default=None,
        help="Override Adam steps (default: same as --ilt-inner-steps)",
    )
    p.add_argument(
        "--ilt-inner-steps-l2o",
        type=int,
        default=None,
        help="Override learned L2O steps (default: same as --ilt-inner-steps)",
    )
    p.add_argument(
        "--ilt-tune-steps",
        type=int,
        default=None,
        help="Steps for handcrafted-optimizer LR grid search in ILT figure (default: scaled from inner unroll)",
    )
    p.add_argument(
        "--ilt-lr-grid",
        type=str,
        default=None,
        help="Comma-separated LRs for AdaGrad/RMSprop/Adam tuning, e.g. 0.01,0.02,0.05,0.1",
    )
    args = p.parse_args()

    img = ROOT / "docs" / "images"
    if args.only_quadratic and args.only_ilt:
        p.error("Use only one of --only-quadratic and --only-ilt")
    if not args.only_ilt:
        fig_quadratic_l2o(
            img / "l2o_quadratic_2d.png",
            args.quadratic_checkpoint,
            inner_eval_steps=args.quadratic_inner_steps,
            eval_optee_seed=args.eval_optee_seed,
            x0_seed=args.x0_seed,
            tune_optee_seed=args.tune_optee_seed,
            tune_x0_seed=args.tune_x0_seed,
            dpi=args.dpi,
            grid_n=args.grid_n,
        )
    if not args.only_quadratic:
        ck_ilt = args.ilt_checkpoint
        if ck_ilt is None or not Path(ck_ilt).is_file():
            for cand in (
                ROOT / "checkpoints" / "ilt_l2o_metalset_best.pt",
                ROOT / "checkpoints" / "ilt_l2o_metalset.pt",
                ROOT / "checkpoints" / "ilt_l2o.pt",
            ):
                if cand.is_file():
                    ck_ilt = cand
                    break
            if ck_ilt is None or not Path(ck_ilt).is_file():
                ck_ilt = ROOT / "checkpoints" / "ilt_l2o.pt"
        bench_glp = BENCH_ICCAD_DIR / args.ilt_glp
        if bench_glp.is_file() and Path(ck_ilt).is_file():
            lr_grid = None
            if args.ilt_lr_grid:
                lr_grid = [float(x.strip()) for x in args.ilt_lr_grid.split(",") if x.strip()]
            fig_ilt_benchmark_four_way(
                img / "ilt_inference_comparison.png",
                ck_ilt,
                inner_steps=args.ilt_inner_steps,
                inner_steps_adagrad=args.ilt_inner_steps_adagrad,
                inner_steps_rmsprop=args.ilt_inner_steps_rmsprop,
                inner_steps_adam=args.ilt_inner_steps_adam,
                inner_steps_l2o=args.ilt_inner_steps_l2o,
                tune_steps=args.ilt_tune_steps,
                handcrafted_lrs=lr_grid,
                init_seed=args.ilt_init_seed,
                glp_name=args.ilt_glp,
                dpi=max(args.dpi, 140),
            )
        else:
            print(
                f"[ilt figure] Using legacy 2×3 panel (benchmark clip or checkpoint missing: {bench_glp}, {ck_ilt})",
                file=sys.stderr,
            )
            fig_ilt_notebook_style(
                img / "ilt_inference_comparison.png",
                ck_ilt,
                inner_steps=args.ilt_inner_steps,
                init_seed=args.ilt_init_seed,
                glp_name=args.ilt_glp,
                dpi=max(args.dpi, 140),
            )


if __name__ == "__main__":
    main()

