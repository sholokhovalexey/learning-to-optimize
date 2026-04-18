"""Repository layout helpers for the ``ilt`` package."""

from __future__ import annotations

from pathlib import Path

_ILT_PKG_ROOT = Path(__file__).resolve().parent


def repo_root() -> Path:
    """Directory containing the ``ilt`` package (project / repo root for this codebase)."""
    return _ILT_PKG_ROOT.parent
