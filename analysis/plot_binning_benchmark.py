#!/usr/bin/env python
"""Plot batch-effect binning benchmark results from analysis/binning_benchmark.py."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

STRATEGIES = ("fed-weight-avg", "fed-weight-avg-smpc", "fed-hist")
STRATEGY_LABELS = {
    "fed-weight-avg": "Weighted avg",
    "fed-weight-avg-smpc": "Weighted SMPC",
    "fed-hist": "Histogram",
}
STRATEGY_COLORS = {
    "fed-weight-avg": "#4477AA",
    "fed-weight-avg-smpc": "#EE6677",
    "fed-hist": "#228833",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_csv",
        type=str,
        default="output/binning_benchmark/results.csv",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="output/binning_benchmark/figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=("pdf", "png", "svg"),
        default="pdf",
    )
    return parser.parse_args()


def _grouped_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    subtitle: str,
    out_path: Path,
    fmt: str,
    reference_line: Optional[float] = None,
) -> None:
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        print(f"Skip {out_path.name}: no rows for metric={metric}")
        return

    datasets = list(sub["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(datasets)), 4.5))
    for i, strategy in enumerate(STRATEGIES):
        vals = []
        for ds in datasets:
            row = sub[(sub["dataset"] == ds) & (sub["strategy"] == strategy)]
            vals.append(float(row["value"].iloc[0]) if len(row) else np.nan)
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=STRATEGY_LABELS[strategy],
            color=STRATEGY_COLORS[strategy],
        )

    if reference_line is not None:
        ax.axhline(reference_line, color="black", linewidth=0.8, linestyle="--", label="no inflation")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_csv)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results CSV: {results_path}")

    df = pd.read_csv(results_path)
    fmt = args.format

    _grouped_bar(
        df,
        metric="cramers_v",
        ylabel="Cramér's V",
        title="Client–bin association after binning",
        subtitle="Lower is better (weaker batch effect in binned space)",
        out_path=figures_dir / "client_bin_association_cramers_v",
        fmt=fmt,
    )
    _grouped_bar(
        df,
        metric="js_amplification",
        ylabel="JS_binned / JS_raw",
        title="Heterogeneity amplification by binning",
        subtitle="Lower is better (binning inflates client separation less)",
        out_path=figures_dir / "heterogeneity_js_amplification",
        fmt=fmt,
        reference_line=1.0,
    )

    print(f"Figures written to {figures_dir} (format={fmt})")


if __name__ == "__main__":
    main()
