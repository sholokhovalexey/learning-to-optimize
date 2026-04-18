"""Meta-training loop for learned optimizers."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm


def _inner_unroll_length(epoch: int, max_unroll: int, fixed_unroll: int | None) -> int:
    if fixed_unroll is not None:
        return min(max_unroll, int(fixed_unroll))
    return min(max_unroll, int(5 + 3 * math.log(1 + epoch)))


def meta_validation_loss(
    optimizer: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    epoch_for_schedule: int,
    max_unroll: int,
    fixed_unroll: int | None,
    max_batches: int | None = None,
    extra_init_noise_scale: float = 0.0,
) -> float:
    """Mean final inner objective on ``val_dataloader`` without meta-gradients.

    Runs the same unroll length schedule as training (via ``epoch_for_schedule`` and
    ``fixed_unroll`` / ``max_unroll``). Optimizer forward runs under ``torch.no_grad()``
    so no graph is built through learned parameters.
    """
    was_training = optimizer.training
    optimizer.eval()
    n_unroll = _inner_unroll_length(epoch_for_schedule, max_unroll, fixed_unroll)
    total = 0.0
    n_batches = 0
    device_t = device
    for batch in val_dataloader:
        if max_batches is not None and n_batches >= max_batches:
            break
        x_init, optimizee = batch
        x_init = x_init.to(device_t)
        optimizee = optimizee.to(device_t)
        if extra_init_noise_scale > 0:
            x = x_init.clone() + extra_init_noise_scale * torch.randn_like(x_init)
        else:
            x = x_init.clone()
        optimizer.reset(device=device_t)
        for i in range(n_unroll):
            loss_tensor, grad = optimizee.loss_and_grad(x)
            step_feat = torch.tensor(float(i), device=device_t)
            with torch.no_grad():
                delta = optimizer(grad, loss=loss_tensor, step=step_feat)
            x = x + delta
        total += float(optimizee(x).mean().item())
        n_batches += 1
    if was_training:
        optimizer.train()
    else:
        optimizer.eval()
    return total / max(1, n_batches)


def _safe_checkpoint_stem(epoch: int, val_loss: float) -> str:
    """Stem with epoch and validation loss."""
    return f"learner_epoch-{epoch:03d}_val-{val_loss:.4f}"


def train_optimizer(
    optimizer: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
    n_epochs: int = 100,
    meta_lr: float = 0.01,
    n_outer_chunks: int = 5,
    grad_scale_augment: bool = True,
    max_unroll: int = 64,
    fixed_unroll: int | None = None,
    meta_weight_decay: float = 1e-4,
    extra_init_noise_scale: float = 1.0,
    use_amp: bool = False,
    val_dataloader: torch.utils.data.DataLoader | None = None,
    checkpoint_every: int = 0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_meta: dict | None = None,
    val_max_batches: int | None = None,
) -> nn.Module:
    """Meta-train a learned optimizer with truncated unrolled inner loops.

    ``dataloader`` must yield ``(x_init, optimizee)`` where ``optimizee`` implements
    ``loss_and_grad``. The routine performs outer-loop Adam updates on optimizer
    parameters using averaged inner losses over ``n_unroll`` steps.

    Args:
        val_dataloader: If ``checkpoint_every > 0``, used to compute ``val_loss`` when saving.
        checkpoint_every: Save a checkpoint every this many completed epochs (0 = disabled).
        checkpoint_dir: Directory for periodic checkpoints (required if ``checkpoint_every > 0``).
        checkpoint_meta: Extra keys merged into each saved dict (e.g. ``dim``, ``architecture``);
            should include everything :func:`l2o.checkpoint.load_learned_optimizer_checkpoint` needs.
        val_max_batches: Optional cap on validation batches per metric (faster estimates).
    """
    device = torch.device(device) if isinstance(device, str) else device
    if checkpoint_every > 0:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when checkpoint_every > 0")
        if val_dataloader is None:
            raise ValueError("val_dataloader is required when checkpoint_every > 0")
    checkpoint_path = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
    if checkpoint_every > 0:
        assert checkpoint_path is not None
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    n_steps = n_outer_chunks
    total_steps = n_epochs * max(1, len(dataloader)) * n_steps
    opt = torch.optim.Adam(optimizer.parameters(), lr=meta_lr, weight_decay=meta_weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=meta_lr, total_steps=max(1, total_steps))
    amp_enabled = bool(use_amp and device.type == "cuda")

    epoch_pbar = tqdm(range(n_epochs), desc="Epoch", position=0)
    for epoch in epoch_pbar:
        chunk_loss = torch.zeros((), device=device)
        batch_pbar = tqdm(
            dataloader,
            desc=f"Batches (epoch {epoch + 1}/{n_epochs})",
            position=1,
            leave=False,
        )
        for batch in batch_pbar:
            x_init, optimizee = batch
            x_init = x_init.to(device)
            optimizee = optimizee.to(device)

            if extra_init_noise_scale > 0:
                x = x_init.clone() + extra_init_noise_scale * torch.randn_like(x_init)
            else:
                x = x_init.clone()

            for _ in range(n_steps):
                opt.zero_grad()
                x = x.detach()
                chunk_loss = torch.zeros((), device=device)
                optimizer.reset(device=device)

                n_unroll = _inner_unroll_length(epoch, max_unroll, fixed_unroll)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    for i in range(n_unroll):
                        loss_tensor, grad = optimizee.loss_and_grad(x)
                        g = grad.detach()
                        if grad_scale_augment:
                            scale = float(torch.empty(1).uniform_(0.5, 2.0))
                            g = g * scale

                        step_feat = torch.tensor(float(i), device=device)
                        delta = optimizer(g, loss=loss_tensor, step=step_feat)
                        x = x + delta

                        loss_batch = optimizee(x)
                        chunk_loss = chunk_loss + loss_batch.mean() / n_unroll

                chunk_loss.backward()
                opt.step()
                scheduler.step()
                batch_pbar.set_postfix(chunk_loss=float(chunk_loss.detach().item()))
        batch_pbar.close()

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            assert checkpoint_path is not None and val_dataloader is not None
            val_loss = meta_validation_loss(
                optimizer,
                val_dataloader,
                device,
                epoch_for_schedule=epoch,
                max_unroll=max_unroll,
                fixed_unroll=fixed_unroll,
                max_batches=val_max_batches,
                extra_init_noise_scale=0.0,
            )
            stem = _safe_checkpoint_stem(epoch + 1, val_loss)
            payload = {
                **(checkpoint_meta or {}),
                "state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }
            out_file = checkpoint_path / f"{stem}.pt"
            torch.save(payload, out_file)
            tqdm.write(f"[checkpoint] saved {out_file} (val_loss={val_loss:.6g})")

    return optimizer

