"""Load learned L2O checkpoints (architecture metadata + weights)."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn as nn

from l2o.models import OPTIMIZER_REGISTRY


def _optimizer_ctor_kwargs(ckpt: dict, cls: type[nn.Module]) -> dict[str, object]:
    """Build constructor kwargs from checkpoint metadata for ``cls``.

    Supports both the legacy schema (dim/hidden_size/max_unroll_steps) and
    optional explicit ``init_kwargs`` in newer checkpoints.
    """
    kwargs: dict[str, object] = {"dim": int(ckpt["dim"])}
    if "hidden_size" in ckpt:
        kwargs["hidden_size"] = int(ckpt["hidden_size"])
    if "max_unroll_steps" in ckpt:
        kwargs["max_unroll_steps"] = float(ckpt["max_unroll_steps"])

    init_kwargs = ckpt.get("init_kwargs")
    if isinstance(init_kwargs, dict):
        kwargs.update(init_kwargs)

    sig = inspect.signature(cls.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    if has_var_kw:
        filtered = kwargs
    else:
        allowed = {p.name for p in params}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
    required = [
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and p.default is inspect.Parameter.empty
        and p.name not in filtered
    ]
    if required:
        raise ValueError(
            f"Checkpoint metadata missing required constructor args for {cls.__name__}: {required}"
        )
    return filtered


def build_learned_optimizer_from_meta(ckpt: dict) -> nn.Module:
    """Instantiate optimizer module described by checkpoint metadata."""
    arch = str(ckpt.get("architecture", "gradenc"))
    cls = OPTIMIZER_REGISTRY.get(arch)
    if cls is None:
        known = ", ".join(sorted(OPTIMIZER_REGISTRY))
        raise ValueError(f"Unknown architecture {arch!r} in checkpoint (known: {known})")
    return cls(**_optimizer_ctor_kwargs(ckpt, cls))


def load_learned_optimizer_checkpoint(
    path: str | Path,
    device: torch.device | str,
    *,
    eval_mode: bool = True,
) -> tuple[nn.Module, dict]:
    """Load checkpoint and return ``(optimizer_module, raw_checkpoint_dict)``.

    The loader instantiates architecture/dimension from metadata, loads
    parameters strictly, moves the module to ``device``, and optionally
    switches to eval mode.
    """
    path = Path(path)
    device_t = torch.device(device) if isinstance(device, str) else device
    try:
        ckpt = torch.load(path, map_location=device_t, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device_t)
    net = build_learned_optimizer_from_meta(ckpt)
    net.load_state_dict(dict(ckpt["state_dict"]), strict=True)
    net.to(device_t)
    if eval_mode:
        net.eval()
    return net, ckpt
