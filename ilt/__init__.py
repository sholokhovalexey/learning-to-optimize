"""Inverse lithography (ILT) helpers: datasets, layout I/O, litho proxy, evaluation, visualization.

Prefer subpackages: :mod:`ilt.datasets`, :mod:`ilt.io`, :mod:`ilt.sim`, :mod:`ilt.eval`, :mod:`ilt.viz`.
Learned-optimizer checkpoints live in :mod:`l2o.checkpoint`.
"""

from ilt.datasets.dataset import ICCADGlpDataset, benchmark_iccad_glp_paths, collate_ilt_batch
from ilt.datasets.download import (
    DEFAULT_ICCAD_GLP_NAMES,
    default_benchmark_iccad_dir,
    default_iccad2012_dir,
    default_metalset_data_dir,
    download_gcd_45nm_gds,
    download_iccad2012_zip,
    download_iccad_glp_files,
    download_url_to_file,
    iccad2012_zip_path,
)
from ilt.io.glp_raster import glp_text_to_target_tensor, load_glp_path, parse_glp, rasterize_parsed
from ilt.paths import repo_root
from ilt.sim.simple_litho import SimplifiedLitho

__all__ = [
    "DEFAULT_ICCAD_GLP_NAMES",
    "ICCADGlpDataset",
    "SimplifiedLitho",
    "benchmark_iccad_glp_paths",
    "collate_ilt_batch",
    "default_benchmark_iccad_dir",
    "default_iccad2012_dir",
    "default_metalset_data_dir",
    "download_gcd_45nm_gds",
    "download_iccad2012_zip",
    "download_iccad_glp_files",
    "download_url_to_file",
    "glp_text_to_target_tensor",
    "iccad2012_zip_path",
    "load_glp_path",
    "parse_glp",
    "repo_root",
    "rasterize_parsed",
]
