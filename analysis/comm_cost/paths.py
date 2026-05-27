"""Shared paths for the communication-cost analysis package."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
COMM_COST_OUTPUT_DIR = "output/comm_cost"
COMM_COST_OUTPUT_PATH = REPO_ROOT / COMM_COST_OUTPUT_DIR
TEX_TEMPLATE = PACKAGE_DIR / "communication_cost.tex"
