"""Communication-cost benchmark for federated Clifti-GPT workflows."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMM_COST_OUTPUT_DIR = "output/comm_cost"
COMM_COST_OUTPUT_PATH = REPO_ROOT / COMM_COST_OUTPUT_DIR

__all__ = [
    "COMM_COST_OUTPUT_DIR",
    "COMM_COST_OUTPUT_PATH",
    "REPO_ROOT",
]
