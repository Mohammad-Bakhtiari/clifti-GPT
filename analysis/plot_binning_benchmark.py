#!/usr/bin/env python
"""Plot federated binning benchmark results from analysis/binning_benchmark.py."""

import argparse
import sys
from pathlib import Path

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
        "--results_wide_csv",
        type=str,
        default="output/binning_benchmark/results_wide.csv",
        help="Optional wide CSV for heterogeneity scatter (falls back if missing).",
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
    out_path: Path,
    fmt: str,
    logy: bool = False,
) -> None:
    sub = df[
        (df["reference"] == "centralized") & (df["metric"] == metric)
    ].copy()
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

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def _hist_improvement(wide: pd.DataFrame, out_path: Path, fmt: str) -> None:
    if wide.empty:
        print(f"Skip {out_path.name}: empty wide results")
        return

    datasets = wide["dataset"].tolist()
    linf_gain = wide["Linf_weighted_vs_centralized"] - wide["Linf_hist_vs_centralized"]
    agree_gain = wide["agreement_hist_vs_centralized"] - wide["agreement_weighted_vs_centralized"]

    x = np.arange(len(datasets))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2 * len(datasets)), 4.5))

    axes[0].bar(x, linf_gain, color=STRATEGY_COLORS["fed-hist"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=20, ha="right")
    axes[0].set_ylabel("L_inf gain (weighted - hist)")
    axes[0].set_title("Edge error reduction vs centralized")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, agree_gain, color=STRATEGY_COLORS["fed-hist"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=20, ha="right")
    axes[1].set_ylabel("Agreement gain (hist - weighted)")
    axes[1].set_title("Assignment agreement gain vs centralized")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("fed-hist improvement over fed-weight-avg", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def _smpc_fidelity(wide: pd.DataFrame, out_path: Path, fmt: str) -> None:
    if wide.empty:
        print(f"Skip {out_path.name}: empty wide results")
        return

    datasets = wide["dataset"].tolist()
    vals = wide["Linf_weighted_smpc_vs_weighted"].tolist()
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(datasets)), 4))
    ax.bar(x, vals, color=STRATEGY_COLORS["fed-weight-avg-smpc"])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("L_inf (SMPC vs plaintext weighted)")
    ax.set_title("SMPC fidelity: fed-weight-avg-smpc vs fed-weight-avg")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def _heterogeneity_scatter(wide: pd.DataFrame, out_path: Path, fmt: str) -> None:
    if wide.empty or "heterogeneity_js" not in wide.columns:
        print(f"Skip {out_path.name}: no heterogeneity column")
        return

    x = wide["heterogeneity_js"].values
    linf_gain = (
        wide["Linf_weighted_vs_centralized"] - wide["Linf_hist_vs_centralized"]
    ).values
    datasets = wide["dataset"].tolist()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x, linf_gain, s=80, color=STRATEGY_COLORS["fed-hist"], zorder=3)
    for xi, yi, label in zip(x, linf_gain, datasets):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Client heterogeneity (mean pairwise JS)")
    ax.set_ylabel("L_inf gain (weighted - hist vs central)")
    ax.set_title("Histogram benefit vs client heterogeneity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_csv)
    wide_path = Path(args.results_wide_csv)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results CSV: {results_path}")

    df = pd.read_csv(results_path)
    wide = pd.read_csv(wide_path) if wide_path.is_file() else pd.DataFrame()

    fmt = args.format
    _grouped_bar(
        df,
        metric="Linf_vs_central",
        ylabel="L_inf vs centralized",
        title="Global bin edge error vs centralized ground truth",
        out_path=figures_dir / "edge_linf_vs_centralized",
        fmt=fmt,
        logy=True,
    )
    _grouped_bar(
        df,
        metric="agreement_vs_central",
        ylabel="Fraction matching centralized bin",
        title="Bin assignment agreement vs centralized",
        out_path=figures_dir / "assignment_agreement_vs_centralized",
        fmt=fmt,
    )
    if not wide.empty:
        _hist_improvement(wide, figures_dir / "hist_minus_weighted_improvement", fmt)
        _smpc_fidelity(wide, figures_dir / "smpc_fidelity", fmt)
        _heterogeneity_scatter(wide, figures_dir / "heterogeneity_vs_hist_gain", fmt)

    print(f"Figures written to {figures_dir} (format={fmt})")


if __name__ == "__main__":
    main()
