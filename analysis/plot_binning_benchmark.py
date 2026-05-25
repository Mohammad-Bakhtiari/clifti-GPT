#!/usr/bin/env python
"""Plot batch-effect binning benchmark results from analysis/binning_benchmark.py.

Produces a single combined figure with up to three panels:

1. Cramer's V (client x bin association after binning; lower is better)
2. JS amplification (JS_binned / JS_raw; lower is better)
3. Peak downstream training accuracy per (dataset, prep_mode), read from
   ``results_summary.csv`` (higher is better). Bars are annotated with the
   round at which the peak was achieved. The accuracy panel is skipped if
   the CSV is missing or contains no matching rows.
"""

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

BINNING_STRATEGIES = ("fed-weight-avg", "fed-weight-avg-smpc", "fed-hist", "fed-hist-smpc")
ACCURACY_STRATEGIES = ("centralized",) + BINNING_STRATEGIES
STRATEGY_LABELS = {
    "centralized": "Centralized",
    "fed-weight-avg": "Weighted avg",
    "fed-weight-avg-smpc": "Weighted SMPC",
    "fed-hist": "Histogram",
    "fed-hist-smpc": "Histogram SMPC",
}
STRATEGY_COLORS = {
    "centralized": "#666666",
    "fed-weight-avg": "#4477AA",
    "fed-weight-avg-smpc": "#EE6677",
    "fed-hist": "#228833",
    "fed-hist-smpc": "#CCBB44",
}

DATASET_SLUG_TO_DISPLAY = {
    "ms": "MS",
    "cl": "CellLine",
    "lung": "LUNG",
    "myeloid-top5": "MYELOID-top5",
    "hp5": "HP5",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_csv",
        type=str,
        default="output/binning_benchmark/results.csv",
    )
    parser.add_argument(
        "--results_summary_csv",
        type=str,
        default=str(
            Path.home()
            / "Documents/Reaserch/clifti-GPT/results/results_summary.csv"
        ),
        help=(
            "Optional path to results_summary.csv with downstream training "
            "metrics. If missing or empty for the four binning prep_modes, "
            "the accuracy panel is skipped."
        ),
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


def _load_best_accuracy(csv_path: Path) -> pd.DataFrame:
    """Load per-(dataset, prep_mode) peak Accuracy from results_summary.csv.

    Deduplicates same-key rows by keeping the last occurrence in file order
    (matches the workflow of appending re-runs to the CSV), then picks the
    row with the maximum ``Value`` per ``(Dataset, prep_mode)``.

    Returns a DataFrame with columns ``dataset, prep_mode, best_accuracy,
    best_round``. ``dataset`` uses the display names from
    ``DATASET_SLUG_TO_DISPLAY``. Returns an empty DataFrame if the file is
    missing or yields no matching rows.
    """
    if not csv_path.is_file():
        return pd.DataFrame(
            columns=["dataset", "prep_mode", "best_accuracy", "best_round"]
        )

    df = pd.read_csv(csv_path)
    required_cols = {"Dataset", "Round", "Metric", "Value", "prep_mode"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"results_summary.csv missing required columns: {sorted(missing)}"
        )

    df = df[df["Metric"] == "Accuracy"].copy()
    df["prep_mode"] = (
        df["prep_mode"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "<NA>": pd.NA})
        .fillna("centralized")
    )
    df = df[df["prep_mode"].isin(ACCURACY_STRATEGIES)]
    if df.empty:
        return pd.DataFrame(
            columns=["dataset", "prep_mode", "best_accuracy", "best_round"]
        )

    dedup_keys = ["Dataset", "Round", "prep_mode"]
    for opt in ("n_epochs", "mu", "Aggregation"):
        if opt in df.columns:
            dedup_keys.append(opt)
    df = df.drop_duplicates(subset=dedup_keys, keep="last")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    if df.empty:
        return pd.DataFrame(
            columns=["dataset", "prep_mode", "best_accuracy", "best_round"]
        )

    idx = df.groupby(["Dataset", "prep_mode"])["Value"].idxmax()
    best = df.loc[idx, ["Dataset", "prep_mode", "Value", "Round"]].copy()
    best = best.rename(
        columns={"Value": "best_accuracy", "Round": "best_round"}
    )
    best["dataset"] = best["Dataset"].map(DATASET_SLUG_TO_DISPLAY).fillna(best["Dataset"])
    best = best.drop(columns=["Dataset"])
    return best[["dataset", "prep_mode", "best_accuracy", "best_round"]].reset_index(drop=True)


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
    width = 0.18

    for i, strategy in enumerate(BINNING_STRATEGIES):
        vals = []
        for ds in datasets:
            row = sub[(sub["dataset"] == ds) & (sub["strategy"] == strategy)]
            vals.append(float(row["value"].iloc[0]) if len(row) else np.nan)
        offset = (i - 1.5) * width
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


def _accuracy_bar_on_ax(
    ax: plt.Axes,
    acc_df: pd.DataFrame,
    metric_df: pd.DataFrame,
) -> None:
    """Grouped bars of peak Accuracy per (dataset, prep_mode), annotated with round.

    The x-axis dataset order is taken from ``metric_df`` (the batch-effect
    long CSV) so that the three panels share the same dataset ordering.
    Datasets present in ``acc_df`` but absent from ``metric_df`` are
    appended at the end.
    """
    if acc_df.empty:
        ax.set_visible(False)
        return

    metric_datasets = (
        list(metric_df["dataset"].unique()) if not metric_df.empty else []
    )
    extra = [d for d in acc_df["dataset"].unique() if d not in metric_datasets]
    datasets = metric_datasets + extra
    if not datasets:
        ax.set_visible(False)
        return

    x = np.arange(len(datasets))
    n_strategies = len(ACCURACY_STRATEGIES)
    width = 0.14
    all_finite: List[float] = []

    for i, strategy in enumerate(ACCURACY_STRATEGIES):
        vals: List[float] = []
        rounds: List[Optional[int]] = []
        for ds in datasets:
            row = acc_df[
                (acc_df["dataset"] == ds) & (acc_df["prep_mode"] == strategy)
            ]
            if len(row):
                vals.append(float(row["best_accuracy"].iloc[0]))
                rounds.append(int(row["best_round"].iloc[0]))
            else:
                vals.append(np.nan)
                rounds.append(None)
        offset = (i - (n_strategies - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=STRATEGY_LABELS[strategy],
            color=STRATEGY_COLORS[strategy],
        )
        all_finite.extend(v for v in vals if np.isfinite(v))
        for bar, r in zip(bars, rounds):
            if r is None or not np.isfinite(bar.get_height()):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"r{r}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Peak Accuracy")
    ax.set_title(
        "Downstream training accuracy (peak across rounds)\n"
        "Higher is better; label shows round of peak",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    finite = all_finite
    if finite:
        lo = max(0.0, min(finite) - 0.05)
        hi = min(1.0, max(finite) + 0.05)
        if hi > lo:
            ax.set_ylim(lo, hi)


def _plot_combined_figure(
    df: pd.DataFrame,
    acc_df: pd.DataFrame,
    out_path: Path,
    fmt: str,
) -> None:
    cramers_sub = df[df["metric"] == "cramers_v"]
    amp_sub = df[df["metric"] == "js_amplification"]
    if cramers_sub.empty and amp_sub.empty and acc_df.empty:
        print(f"Skip {out_path.name}: no rows for any panel")
        return

    n_datasets = max(
        cramers_sub["dataset"].nunique() if not cramers_sub.empty else 0,
        amp_sub["dataset"].nunique() if not amp_sub.empty else 0,
        acc_df["dataset"].nunique() if not acc_df.empty else 0,
    )
    include_acc = not acc_df.empty
    n_panels = 3 if include_acc else 2

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(4.0 * n_panels + 2.0, 3.2 * n_datasets), 4.5),
        sharey=False,
    )
    if n_panels == 1:
        axes = [axes]

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
    if include_acc:
        _accuracy_bar_on_ax(axes[2], acc_df, df)

    handles, labels = [], []
    for ax in axes:
        if not ax.get_visible():
            continue
        ah, al = ax.get_legend_handles_labels()
        for h, l in zip(ah, al):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_csv)
    summary_path = Path(args.results_summary_csv)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results CSV: {results_path}")

    df = pd.read_csv(results_path)
    fmt = args.format

    acc_df = _load_best_accuracy(summary_path)
    if acc_df.empty:
        print(
            f"Warning: no Accuracy rows for binning prep_modes in {summary_path}; "
            f"skipping the training-accuracy panel."
        )

    out_path = figures_dir / "binning_benchmark_panels"
    _plot_combined_figure(df, acc_df, out_path, fmt)

    print(f"Figure written to {out_path.with_suffix(f'.{fmt}')}")


if __name__ == "__main__":
    main()
