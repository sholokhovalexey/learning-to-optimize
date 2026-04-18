"""Download ICCAD 2013 ``M1_test*.glp`` clips and optional benchmark assets (OpenILT / CUHK mirrors)."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ilt.paths import repo_root as _repo_root

# All M1 ICCAD 2013 clips shipped with OpenILT (same benchmark family as LithoBench / neural-ILT).
DEFAULT_ICCAD_GLP_NAMES = tuple(f"M1_test{i}.glp" for i in range(1, 11))

OPENILT_ICCAD_BASE = (
    "https://raw.githubusercontent.com/OpenOPC/OpenILT/main/benchmark/ICCAD2013/"
)
OPENILT_BENCHMARK_BASE = "https://raw.githubusercontent.com/OpenOPC/OpenILT/main/benchmark"
ICCAD2012_GDS_ZIP_URL = "https://www.cse.cuhk.edu.hk/~byu/files/benchmarks/gdsiccad.zip"
GCD_45NM_GDS_NAME = "gcd_45nm.gds"


__all__ = [
    "DEFAULT_ICCAD_GLP_NAMES",
    "GCD_45NM_GDS_NAME",
    "ICCAD2012_GDS_ZIP_URL",
    "OPENILT_BENCHMARK_BASE",
    "OPENILT_ICCAD_BASE",
    "default_benchmark_iccad_dir",
    "default_iccad2012_dir",
    "default_metalset_data_dir",
    "download_gcd_45nm_gds",
    "download_iccad2012_zip",
    "download_iccad_glp_files",
    "download_url_to_file",
    "iccad2012_zip_path",
    "main",
]


def download_url_to_file(url: str, path: Path, *, overwrite: bool = False) -> Path:
    """Download ``url`` to ``path``; skip if ``path`` exists and ``overwrite`` is False."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    urllib.request.urlretrieve(url, path)
    return path


def download_iccad_glp_files(
    dest_dir: str | Path,
    names: tuple[str, ...] = DEFAULT_ICCAD_GLP_NAMES,
    overwrite: bool = False,
) -> list[Path]:
    """Fetch ``.glp`` files into ``dest_dir``; return local paths in sorted order."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for name in names:
        path = dest / name
        if path.exists() and not overwrite:
            out.append(path)
            continue
        url = OPENILT_ICCAD_BASE + name
        urllib.request.urlretrieve(url, path)
        out.append(path)
    return sorted(out)


def default_metalset_data_dir(repo_root: str | Path | None = None) -> Path:
    """Suggested unpack location for LithoBench MetalSet: ``<repo>/data/MetalSet`` (with ``target/*.png``)."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    return root / "data" / "MetalSet"


def default_benchmark_iccad_dir(repo_root: str | Path | None = None) -> Path:
    """``<repo>/benchmarks/iccad2013/`` (standard eval path for ``M1_test*.glp``)."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    return root / "benchmarks" / "iccad2013"


def default_iccad2012_dir(repo_root: str | Path | None = None) -> Path:
    """``<repo>/benchmarks/iccad2012/``."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    return root / "benchmarks" / "iccad2012"


def iccad2012_zip_path(repo_root: str | Path | None = None) -> Path:
    """Path where the ICCAD 2012 archive is stored after download (before unzip)."""
    return default_iccad2012_dir(repo_root) / "gdsiccad.zip"


def download_iccad2012_zip(repo_root: str | Path | None = None, *, overwrite: bool = False) -> Path:
    """Download ICCAD 2012 hotspot GDS benchmark zip (does not unzip)."""
    path = iccad2012_zip_path(repo_root)
    return download_url_to_file(ICCAD2012_GDS_ZIP_URL, path, overwrite=overwrite)


def download_gcd_45nm_gds(repo_root: str | Path | None = None, *, overwrite: bool = False) -> Path:
    """Download OpenILT ``gcd_45nm.gds`` into ``benchmarks/iccad2013/``."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    dest = default_benchmark_iccad_dir(root) / GCD_45NM_GDS_NAME
    url = f"{OPENILT_BENCHMARK_BASE}/{GCD_45NM_GDS_NAME}"
    return download_url_to_file(url, dest, overwrite=overwrite)


def _fetch_glps(dest: Path, *, overwrite: bool, quiet: bool, label: str) -> int:
    if not quiet:
        print(f"{label}: {dest.resolve()}")
        print(f"Source:      {OPENILT_ICCAD_BASE}")
    try:
        paths = download_iccad_glp_files(dest, overwrite=overwrite)
    except (urllib.error.URLError, OSError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    if not quiet:
        print(f"OK - {len(paths)} .glp files under {dest.resolve()}")
        for p in paths:
            print(f"  {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m ilt.datasets.download`` (repo root may also ship ``get_benchmarks.sh``)."""
    ap = argparse.ArgumentParser(
        description=(
            "Download ICCAD 2013 M1_test*.glp into benchmarks/iccad2013/ (eval only; no duplicate under data/). "
            "Large ILT L2O training uses LithoBench MetalSet PNGs - unpack under data/MetalSet/ "
            "or set LITHOBENCH_ROOT (see README). Optional: ICCAD 2012 zip, gcd GDS."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                          # ICCAD M1_test*.glp -> benchmarks/iccad2013/ (default)\n"
            "  %(prog)s --dest DIR # custom directory for those .glp files\n"
            "  %(prog)s --no-iccad               # skip ICCAD .glp (only if using --iccad2012-zip / --gcd)\n"
            "  %(prog)s --iccad2012-zip --gcd    # optional extras (download only)\n"
        ),
    )
    ap.add_argument(
        "--repo-root",
        type=str,
        default=None,
        metavar="DIR",
        help="Repository root (default: parent of the l2o package).",
    )
    ap.add_argument(
        "--dest",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for M1_test*.glp (default: <repo>/benchmarks/iccad2013/).",
    )
    ap.add_argument(
        "--benchmarks",
        action="store_true",
        help="Download M1_test*.glp (same as default; kept for explicit scripts).",
    )
    ap.add_argument(
        "--no-iccad",
        action="store_true",
        help="Do not fetch ICCAD .glp (use with --iccad2012-zip / --gcd only).",
    )
    ap.add_argument(
        "--all-required",
        action="store_true",
        help="Alias for default: ICCAD benchmark .glp only (historical flag).",
    )
    ap.add_argument(
        "--iccad2012-zip",
        action="store_true",
        help="Download ICCAD 2012 gdsiccad.zip to benchmarks/iccad2012/ (not used by the ILT demo; unzip separately).",
    )
    ap.add_argument(
        "--gcd",
        action="store_true",
        help="Download gcd_45nm.gds to benchmarks/iccad2013/ (optional; not used by this codebase).",
    )
    ap.add_argument(
        "--all-extras",
        action="store_true",
        help="Shorthand for --iccad2012-zip --gcd.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if files already exist.",
    )
    ap.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print errors.",
    )
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()

    want_iccad12 = bool(args.iccad2012_zip or args.all_extras)
    want_gcd = bool(args.gcd or args.all_extras)

    any_choice = bool(
        args.benchmarks
        or args.all_required
        or args.no_iccad
        or want_iccad12
        or want_gcd
        or args.dest
    )
    want_iccad_glps = True
    if args.no_iccad:
        want_iccad_glps = False
    elif not any_choice:
        want_iccad_glps = True

    bench_dest = Path(args.dest).resolve() if args.dest else default_benchmark_iccad_dir(root)

    if want_iccad_glps:
        rc = _fetch_glps(bench_dest, overwrite=args.overwrite, quiet=args.quiet, label="Destination (ICCAD benchmark .glp)")
        if rc != 0:
            return rc

    if want_iccad12:
        zpath = iccad2012_zip_path(root)
        if not args.quiet:
            print(f"Destination (ICCAD 2012 zip): {zpath.resolve()}")
            print(f"Source:      {ICCAD2012_GDS_ZIP_URL}")
        try:
            download_iccad2012_zip(root, overwrite=args.overwrite)
        except (urllib.error.URLError, OSError) as e:
            print(f"Download failed: {e}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"OK - {zpath.resolve()} (run scripts/prepare_downloaded_data.py --iccad2012 to unzip)")

    if want_gcd:
        gpath = default_benchmark_iccad_dir(root) / GCD_45NM_GDS_NAME
        if not args.quiet:
            print(f"Destination (gcd GDS): {gpath.resolve()}")
            print(f"Source:      {OPENILT_BENCHMARK_BASE}/{GCD_45NM_GDS_NAME}")
        try:
            download_gcd_45nm_gds(root, overwrite=args.overwrite)
        except (urllib.error.URLError, OSError) as e:
            print(f"Download failed: {e}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"OK - {gpath.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
