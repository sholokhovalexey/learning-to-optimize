#!/usr/bin/env python3
"""
Prepare training data for meta-training the neural optimizer on ILT-style tasks (``ICCADGlpDataset``).

Research context (neural / learning-based ILT)
---------------------------------------------
NeurIPS 2023 *LithoBench* (Zheng et al.) is the standard large-scale benchmark for deep
learning on computational lithography: **MetalSet** tiles are synthesized following ICCAD-2013
mask-optimization design rules; **ViaSet** covers via layers; evaluation often includes the
public **ICCAD 2013** 32nm M1 clips (same family as this repo's ``M1_test*.glp``). Works such as
*ILILT* (ICML 2024) and *Neural-ILT* train on LithoBench-style data and report ICCAD-13 / LithoBench
metrics. The full LithoBench archive is distributed via Google Drive (see ``--lithobench``).

This repository's L2O demo consumes **ICCAD-style ``.glp``** (RECT/PGON) rasterized by
``ilt.io.glp_raster``. That format matches OpenILT's ICCAD 2013 mirror, not LithoBench's numpy/png
tiles out of the box.

**ICCAD eval clips:** ``./get_benchmarks.sh`` or ``python -m ilt.datasets.download`` puts ``M1_test*.glp`` under
``benchmarks/iccad2013/`` only. **Training** defaults to synthetic ``.glp`` under ``data/synthetic_glp_train/``
(``synthetic`` subcommand); use ``scripts/train_ilt_l2o.py --metalset`` for LithoBench MetalSet PNGs under
``data/MetalSet/``. Optional ICCAD 2012 zip: ``python -m ilt.datasets.download --iccad2012-zip``.

"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# LithoBench README (https://github.com/shelljane/lithobench): dataset Google Drive file id
LITHOBENCH_DRIVE_FILE_ID = "1MzYiRRxi8Eu2L6WHCfZ1DtRnjVNOl4vu"


def _synthetic_glp_text(rng: random.Random) -> str:
    """Random ICCAD-like clip: RECT primitives on a ~1200 nm canvas (same order of magnitude as M1_test*)."""
    lines = [
        "BEGIN     /* synthetic Metal-1 clip for L2O meta-training */",
        "EQUIV  1  1000  MICRON  +X,+Y",
        "CNAME Synth_Top",
        "LEVEL M1",
        "",
        "CELL Synth_Top PRIME",
    ]
    n = rng.randint(4, 14)
    for _ in range(n):
        x = rng.randint(80, 780)
        y = rng.randint(80, 780)
        w = rng.randint(40, min(420, 1120 - x))
        h = rng.randint(40, min(420, 1120 - y))
        lines.append(f"   RECT N M1  {x}  {y}  {w}  {h}")
    # Occasional axis-aligned “L” or bar as a single PGON (rectangle outline)
    if rng.random() < 0.35:
        x0, y0 = rng.randint(100, 600), rng.randint(100, 600)
        dx, dy = rng.randint(80, 200), rng.randint(80, 200)
        x1, y1 = x0 + dx, y0 + dy
        lines.append(
            f"   PGON N M1  {x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1} {x0} {y0}"
        )
    lines.append("ENDMSG")
    return "\n".join(lines) + "\n"


def cmd_synthetic(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for i in range(args.n):
        path = out / f"synth_train_{i:05d}.glp"
        if path.exists() and not args.overwrite:
            continue
        path.write_text(_synthetic_glp_text(rng), encoding="utf-8")
    written = sorted(out.glob("synth_train_*.glp"))
    print(f"OK - {len(written)} synthetic .glp file(s) under {out}")
    print("Tip: train with (set --glp-train-dir if not using data/synthetic_glp_train/)")
    print(f"  python scripts/train_ilt_l2o.py --glp-train-dir {out}")
    return 0


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def cmd_lithobench(args: argparse.Namespace) -> int:
    print("LithoBench (NeurIPS 2023 Datasets & Benchmarks)")
    print("  Paper: https://openreview.net/forum?id=JqWtIIaS8n")
    print("  Code:  https://github.com/shelljane/lithobench")
    print()
    print("The published dataset is a large tarball on Google Drive (not plain HTTP).")
    print(f"  File id: {LITHOBENCH_DRIVE_FILE_ID}")
    print("  Typical manual steps: open the Drive link from the LithoBench README, download,")
    print("  then extract under a work/ directory per upstream instructions.")
    print()
    if not args.try_download:
        print("Re-run with --try-download to attempt automatic download via `gdown` (pip install gdown).")
        return 0

    dest = Path(args.dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown = _which("gdown")
    if gdown is None:
        print("ERROR: `gdown` not found. Install with:  python -m pip install gdown", file=sys.stderr)
        return 1
    url = f"https://drive.google.com/uc?id={LITHOBENCH_DRIVE_FILE_ID}"
    cmd = [gdown, url, "-O", str(dest)]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"gdown failed ({e}). Manual download may still work.", file=sys.stderr)
        return 1
    print(f"Saved: {dest}")
    print("Extract with:  tar xvfz <archive>   (see LithoBench README; paths may be lithodata*.tar.gz)")
    print("Note: LithoBench tiles are not ICCAD .glp; integration with this repo requires raster/target conversion.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Synthetic ICCAD-style .glp training clips and optional LithoBench fetch helper. "
            "ICCAD eval layouts: ./get_benchmarks.sh or python -m ilt.datasets.download -> benchmarks/iccad2013/."
        ),
        epilog="See module docstring.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_syn = sub.add_parser(
        "synthetic",
        help="Generate synthetic ICCAD-style .glp clips (default input for scripts/train_ilt_l2o.py).",
    )
    p_syn.add_argument("--out", type=str, default=str(ROOT / "data" / "synthetic_glp_train"), help="Output directory")
    p_syn.add_argument("--n", type=int, default=128, help="Number of clips to write")
    p_syn.add_argument("--seed", type=int, default=0)
    p_syn.add_argument("--overwrite", action="store_true", help="Overwrite existing synth_train_*.glp files")
    p_syn.set_defaults(func=cmd_synthetic)

    p_lb = sub.add_parser(
        "lithobench",
        help="(Optional) LithoBench info; download only with --try-download. Not used by the default ICCAD .glp demo.",
    )
    p_lb.add_argument(
        "--try-download",
        action="store_true",
        help="Run gdown to fetch the dataset archive (requires: pip install gdown)",
    )
    p_lb.add_argument(
        "--dest",
        type=str,
        default=str(ROOT / "data" / "lithobench_drive_download.bin"),
        help="Where to save the downloaded file (rename/extract per LithoBench README)",
    )
    p_lb.set_defaults(func=cmd_lithobench)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
