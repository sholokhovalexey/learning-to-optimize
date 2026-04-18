"""
Parse ICCAD-style ``.glp`` layouts (RECT / PGON) and rasterize to a target image.

Layout files are fetched from the OpenILT repository (same ICCAD 2013 benchmark as OpenILT / LithoBench).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ParsedGlp:
    rects: list[tuple[int, int, int, int]]  # x, y, width, height
    pgons: list[list[tuple[int, int]]]


_LINE_RECT = re.compile(r"^\s*RECT\s+\S+\s+\S+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
_LINE_PGON = re.compile(r"^\s*PGON\s+\S+\s+\S+\s+(.+)$")


def parse_glp(text: str) -> ParsedGlp:
    rects: list[tuple[int, int, int, int]] = []
    pgons: list[list[tuple[int, int]]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("BEGIN") or line.startswith("ENDMSG"):
            continue
        if line.startswith("EQUIV") or line.startswith("CNAME") or line.startswith("LEVEL"):
            continue
        if line.startswith("CELL"):
            continue
        m = _LINE_RECT.match(line)
        if m:
            x, y, w, h = map(int, m.groups())
            rects.append((x, y, w, h))
            continue
        m = _LINE_PGON.match(line)
        if m:
            nums = list(map(int, m.group(1).split()))
            if len(nums) % 2 != 0:
                raise ValueError("Odd number of coordinates in PGON")
            poly = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
            pgons.append(poly)
            continue
    if not rects and not pgons:
        raise ValueError("No RECT/PGON primitives found in glp text")
    return ParsedGlp(rects=rects, pgons=pgons)


def _bbox(parsed: ParsedGlp) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for x, y, w, h in parsed.rects:
        xs.extend([x, x + w])
        ys.extend([y, y + h])
    for poly in parsed.pgons:
        for px, py in poly:
            xs.append(float(px))
            ys.append(float(py))
    return min(xs), max(xs), min(ys), max(ys)


def _rasterize_rect(
    target: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    H, W = target.shape
    eps = 1e-6
    for ii in range(H):
        for jj in range(W):
            u = xmin + (jj + 0.5) / W * (xmax - xmin + eps)
            v_top = ymax - (ii + 0.5) / H * (ymax - ymin + eps)
            if x <= u <= x + w and y <= v_top <= y + h:
                target[ii, jj] = 1.0


def _rasterize_pgon(
    target: np.ndarray,
    poly: list[tuple[int, int]],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    try:
        from matplotlib.path import Path as MplPath
    except ImportError as e:
        raise ImportError("Rasterizing PGON requires matplotlib (pip install matplotlib).") from e

    H, W = target.shape
    eps = 1e-6
    verts = []
    for px, py in poly:
        u = (px - xmin) / (xmax - xmin + eps) * (W - 1)
        v = (ymax - py) / (ymax - ymin + eps) * (H - 1)
        verts.append((u, v))
    path = MplPath(np.array(verts, dtype=np.float64))
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([jj.astype(np.float64) + 0.5, ii.astype(np.float64) + 0.5], axis=-1).reshape(-1, 2)
    inside = path.contains_points(pts).reshape(H, W)
    target[inside] = 1.0


def rasterize_parsed(parsed: ParsedGlp, height: int, width: int) -> np.ndarray:
    """Binary target image ``(H, W)`` in ``{0,1}`` (union of shapes)."""
    xmin, xmax, ymin, ymax = _bbox(parsed)
    pad = 0.02 * max(xmax - xmin, ymax - ymin, 1.0)
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    target = np.zeros((height, width), dtype=np.float32)
    for x, y, w, h in parsed.rects:
        _rasterize_rect(target, x, y, w, h, xmin, xmax, ymin, ymax)
    for poly in parsed.pgons:
        _rasterize_pgon(target, poly, xmin, xmax, ymin, ymax)
    return np.clip(target, 0.0, 1.0)


def glp_text_to_target_tensor(text: str, height: int, width: int) -> torch.Tensor:
    parsed = parse_glp(text)
    arr = rasterize_parsed(parsed, height, width)
    return torch.from_numpy(arr)


def load_glp_path(path: str, height: int, width: int) -> torch.Tensor:
    with open(path, encoding="utf-8", errors="replace") as f:
        return glp_text_to_target_tensor(f.read(), height, width)
