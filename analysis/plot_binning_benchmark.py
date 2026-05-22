#!/usr/bin/env python
"""Plot batch-effect binning benchmark results from analysis/binning_benchmark.py."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

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


def _grouped_bar_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    subtitle: str,
    reference_line: Optional[float] = None,
) -> None:
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        ax.set_visible(False)
        return

    datasets = list(sub["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.25

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
        ax.axhline(
            reference_line,
            color="black",
            linewidth=0.8,
            linestyle="--",
            label="no inflation",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)


def _plot_combined_figure(
    df: pd.DataFrame,
    out_paths: List[Path],
    fmt: str,
) -> None:
    cramers_sub = df[df["metric"] == "cramers_v"]
    amp_sub = df[df["metric"] == "js_amplification"]
    if cramers_sub.empty and amp_sub.empty:
        print(f"Skip {out_paths[0].name}: no rows for cramers_v or js_amplification")
        return

    n_datasets = max(
        cramers_sub["dataset"].nunique() if not cramers_sub.empty else 0,
        amp_sub["dataset"].nunique() if not amp_sub.empty else 0,
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(10, 3.2 * n_datasets), 4.5),
        sharey=False,
    )

    _grouped_bar_on_ax(
        axes[0],
        df,
        metric="cramers_v",
        ylabel="Cramér's V",
        title="Client–bin association after binning",
        subtitle="Lower is better (weaker batch effect in binned space)",
    )
    _grouped_bar_on_ax(
        axes[1],
        df,
        metric="js_amplification",
        ylabel="JS_binned / JS_raw",
        title="Heterogeneity amplification by binning",
        subtitle="Lower is better (binning inflates client separation less)",
        reference_line=1.0,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    amp_handles, amp_labels = axes[1].get_legend_handles_labels()
    for handle, label in zip(amp_handles, amp_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
    )
    fig.tight_layout()
    for out_path in out_paths:
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

    out_paths = [
        figures_dir / "client_bin_association_cramers_v",
        figures_dir / "heterogeneity_js_amplification",
    ]
    _plot_combined_figure(df, out_paths, fmt)

    print(f"Figures written to {figures_dir} (format={fmt})")


if __name__ == "__main__":
    main()
