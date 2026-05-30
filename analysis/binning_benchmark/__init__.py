"""Federated binning quality benchmark on real benchmark AnnData."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = "output/binning_benchmark"

__all__ = ["REPO_ROOT", "OUTPUT_DIR"]
