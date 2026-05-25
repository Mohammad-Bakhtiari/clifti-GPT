#!/usr/bin/env python
"""Plot batch-effect binning benchmark results from analysis/binning_benchmark.py.

Produces separate PNG figures for each panel plus a shared horizontal legend:

1. Cramer's V (client x bin association after binning; lower is better)
2. JS amplification (JS_binned / JS_raw; lower is better)
3. Peak downstream training accuracy per (dataset, prep_mode), read from
   ``results_summary.csv`` (higher is better). Bars are annotated with the
   round at which the peak was achieved. The accuracy panel is skipped if
   the CSV is missing or contains no matching rows.
4. Horizontal legend (strategies and optional reference line)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FONT_SIZE = 10

PLOT_STRATEGIES = (
    "centralized",
    "fed-weight-avg",
    "fed-weight-avg-smpc",
    "fed-hist",
    "fed-hist-smpc",
)
STRATEGY_LABELS = {
    "centralized": "Centralized",
    "fed-weight-avg": "Fed-weight-avg",
    "fed-weight-avg-smpc": "Fed-weight-avg-SMPC",
    "fed-hist": "Fed-hist-binning",
    "fed-hist-smpc": "Fed-hist-binning-SMPC",
}
STRATEGY_COLORS = {
    "centralized": "#666666",
    "fed-weight-avg": "#4477AA",
    "fed-weight-avg-smpc": "#EE6677",
    "fed-hist": "#228833",
    "fed-hist-smpc": "#AA8800",
}
STRATEGY_HATCHES = {
    "centralized": "",
    "fed-weight-avg": "",
    "fed-weight-avg-smpc": "///",
    "fed-hist": "",
    "fed-hist-smpc": "xxx",
}
STRATEGY_ALIASES = {
    "federated": "fed-weight-avg",
    "federated_smpc": "fed-weight-avg-smpc",
    "histogram": "fed-hist",
    "histogram_smpc": "fed-hist-smpc",
    "fed_hist": "fed-hist",
    "fed_hist_smpc": "fed-hist-smpc",
}
WIDE_METRIC_PREFIX = {
    "centralized": "centralized",
    "weighted": "fed-weight-avg",
    "weighted_smpc": "fed-weight-avg-smpc",
    "hist": "fed-hist",
    "hist_smpc": "fed-hist-smpc",
}
BATCH_EFFECT_METRICS = ("cramers_v", "js_binned", "js_amplification")
BAR_WIDTH = 0.11
BAR_EDGE_COLOR = "0.15"
BAR_EDGE_WIDTH = 0.6

DATASET_SLUG_TO_DISPLAY = {
    "ms": "MS",
    "cl": "CellLine",
    "lung": "LUNG",
    "myeloid-top5": "MYELOID-top5",
    "hp5": "HP5",
}


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )


def _figure_size(n_datasets: int) -> Tuple[float, float]:
    return (max(3.2 * n_datasets, 6.0), 4.5)


def _strategy_bar_kwargs(strategy: str) -> dict:
    return {
        "color": STRATEGY_COLORS[strategy],
        "hatch": STRATEGY_HATCHES[strategy],
        "edgecolor": BAR_EDGE_COLOR,
        "linewidth": BAR_EDGE_WIDTH,
        "zorder": 3 if strategy.endswith("-smpc") else 2,
    }


def _normalize_strategy_names(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace(STRATEGY_ALIASES)
    )


def _wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        c for c in ("dataset", "n_clients", "n_cells", "n_nonzero", "js_raw")
        if c in wide.columns
    ]
    rows: List[dict] = []
    for _, row in wide.iterrows():
        base = {c: row[c] for c in base_cols}
        for prefix, strategy in WIDE_METRIC_PREFIX.items():
            for metric in BATCH_EFFECT_METRICS:
                col = f"{metric}_{prefix}"
                if col not in wide.columns:
                    continue
                value = pd.to_numeric(row[col], errors="coerce")
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        **base,
                        "strategy": strategy,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _load_batch_effect_results(results_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(results_csv)
    if {"strategy", "metric", "value"}.issubset(raw.columns):
        long_df = raw.copy()
        long_df["strategy"] = _normalize_strategy_names(long_df["strategy"])
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    elif any(c.startswith("cramers_v_") for c in raw.columns):
        long_df = _wide_to_long(raw)
    else:
        raise ValueError(
            f"Unrecognized results format in {results_csv}. "
            "Expected long columns (strategy, metric, value) or wide cramers_v_* columns."
        )

    missing = sorted(set(PLOT_STRATEGIES) - set(long_df["strategy"].unique()))
    wide_path = results_csv.parent / "results_wide.csv"
    if missing and wide_path.is_file():
        wide_long = _wide_to_long(pd.read_csv(wide_path))
        supplement = wide_long[wide_long["strategy"].isin(missing)]
        if not supplement.empty:
            long_df = pd.concat([long_df, supplement], ignore_index=True)
            print(
                f"Supplemented missing strategies from {wide_path}: "
                f"{sorted(supplement['strategy'].unique())}"
            )

    still_missing = sorted(set(PLOT_STRATEGIES) - set(long_df["strategy"].unique()))
    if still_missing:
        print(
            "Warning: results are missing batch-effect rows for strategies "
            f"{still_missing}. Re-run analysis/binning_benchmark.py to populate them."
        )

    for metric in ("cramers_v", "js_amplification"):
        present = set(
            long_df.loc[long_df["metric"] == metric, "strategy"].unique()
        )
        metric_missing = sorted(set(PLOT_STRATEGIES) - present)
        if metric_missing:
            print(
                f"Warning: metric={metric} has no rows for strategies "
                f"{metric_missing}."
            )

    long_df = long_df.drop_duplicates(
        subset=["dataset", "strategy", "metric"], keep="last"
    )
    return long_df


def _set_grouped_bar_xlim(ax: plt.Axes, n_datasets: int, n_strategies: int) -> None:
    half_span = (n_strategies - 1) / 2.0 * BAR_WIDTH + BAR_WIDTH / 2.0
    ax.set_xlim(-0.5 - half_span - 0.08, n_datasets - 0.5 + half_span + 0.08)


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
        default="png",
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
    df = df[df["prep_mode"].isin(PLOT_STRATEGIES)]
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
) -> int:
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        return 0

    datasets = list(sub["dataset"].unique())
    x = np.arange(len(datasets))
    n_strategies = len(PLOT_STRATEGIES)

    for i, strategy in enumerate(PLOT_STRATEGIES):
        vals = []
        for ds in datasets:
            row = sub[(sub["dataset"] == ds) & (sub["strategy"] == strategy)]
            vals.append(float(row["value"].iloc[0]) if len(row) else np.nan)
        offset = (i - (n_strategies - 1) / 2.0) * BAR_WIDTH
        ax.bar(
            x + offset,
            vals,
            BAR_WIDTH,
            **_strategy_bar_kwargs(strategy),
        )

    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="black",
            linewidth=0.8,
            linestyle="--",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{subtitle}")
    _set_grouped_bar_xlim(ax, len(datasets), n_strategies)
    ax.grid(axis="y", alpha=0.3)
    return len(datasets)


def _accuracy_bar_on_ax(
    ax: plt.Axes,
    acc_df: pd.DataFrame,
    metric_df: pd.DataFrame,
) -> int:
    """Grouped bars of peak Accuracy per (dataset, prep_mode), annotated with round.

    The x-axis dataset order is taken from ``metric_df`` (the batch-effect
    long CSV) so that all panels share the same dataset ordering. Datasets
    present in ``acc_df`` but absent from ``metric_df`` are appended at the end.
    """
    if acc_df.empty:
        return 0

    metric_datasets = (
        list(metric_df["dataset"].unique()) if not metric_df.empty else []
    )
    extra = [d for d in acc_df["dataset"].unique() if d not in metric_datasets]
    datasets = metric_datasets + extra
    if not datasets:
        return 0

    x = np.arange(len(datasets))
    n_strategies = len(PLOT_STRATEGIES)
    all_finite: List[float] = []

    for i, strategy in enumerate(PLOT_STRATEGIES):
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
        offset = (i - (n_strategies - 1) / 2.0) * BAR_WIDTH
        bars = ax.bar(
            x + offset,
            vals,
            BAR_WIDTH,
            **_strategy_bar_kwargs(strategy),
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
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Peak Accuracy")
    ax.set_title(
        "Downstream training accuracy (peak across rounds)\n"
        "Higher is better; label shows round of peak"
    )
    _set_grouped_bar_xlim(ax, len(datasets), n_strategies)
    ax.grid(axis="y", alpha=0.3)
    finite = all_finite
    if finite:
        lo = max(0.0, min(finite) - 0.05)
        hi = min(1.0, max(finite) + 0.05)
        if hi > lo:
            ax.set_ylim(lo, hi)
    return len(datasets)


def _save_figure(fig: plt.Figure, out_path: Path, fmt: str) -> Path:
    path = out_path.with_suffix(f".{fmt}")
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.2,
        dpi=300 if fmt == "png" else None,
    )
    plt.close(fig)
    return path


def _save_grouped_bar_figure(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    subtitle: str,
    out_path: Path,
    fmt: str,
    reference_line: Optional[float] = None,
) -> Optional[Path]:
    sub = df[df["metric"] == metric]
    if sub.empty:
        print(f"Skip {out_path.name}: no rows for metric={metric}")
        return None

    n_datasets = sub["dataset"].nunique()
    fig, ax = plt.subplots(figsize=_figure_size(n_datasets))
    _grouped_bar_on_ax(
        ax,
        df,
        metric=metric,
        ylabel=ylabel,
        title=title,
        subtitle=subtitle,
        reference_line=reference_line,
    )
    return _save_figure(fig, out_path, fmt)


def _save_accuracy_figure(
    acc_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    out_path: Path,
    fmt: str,
) -> Optional[Path]:
    if acc_df.empty:
        print(f"Skip {out_path.name}: no accuracy rows")
        return None

    metric_datasets = (
        list(metric_df["dataset"].unique()) if not metric_df.empty else []
    )
    extra = [d for d in acc_df["dataset"].unique() if d not in metric_datasets]
    n_datasets = len(metric_datasets + extra)
    if n_datasets == 0:
        print(f"Skip {out_path.name}: no datasets")
        return None

    fig, ax = plt.subplots(figsize=_figure_size(n_datasets))
    _accuracy_bar_on_ax(ax, acc_df, metric_df)
    return _save_figure(fig, out_path, fmt)


def _save_legend_figure(out_path: Path, fmt: str) -> Path:
    handles = [
        Patch(
            facecolor=STRATEGY_COLORS[s],
            edgecolor=BAR_EDGE_COLOR,
            hatch=STRATEGY_HATCHES[s],
            linewidth=BAR_EDGE_WIDTH,
            label=STRATEGY_LABELS[s],
        )
        for s in PLOT_STRATEGIES
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="no inflation",
        )
    )
    n_cols = len(handles)
    fig, ax = plt.subplots(figsize=(max(1.4 * n_cols, 8.0), 0.8))
    ax.axis("off")
    fig.legend(
        handles=handles,
        loc="center",
        ncol=n_cols,
        frameon=False,
    )
    return _save_figure(fig, out_path, fmt)


def _plot_all_figures(
    df: pd.DataFrame,
    acc_df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
) -> List[Path]:
    written: List[Path] = []

    path = _save_grouped_bar_figure(
        df,
        metric="cramers_v",
        ylabel="Cramér's V",
        title="Client–bin association after binning",
        subtitle="Lower is better (weaker batch effect in binned space)",
        out_path=figures_dir / "binning_benchmark_cramers_v",
        fmt=fmt,
    )
    if path is not None:
        written.append(path)

    path = _save_grouped_bar_figure(
        df,
        metric="js_amplification",
        ylabel="JS_binned / JS_raw",
        title="Heterogeneity amplification by binning",
        subtitle="Lower is better (binning inflates client separation less)",
        out_path=figures_dir / "binning_benchmark_js_amplification",
        fmt=fmt,
        reference_line=1.0,
    )
    if path is not None:
        written.append(path)

    path = _save_accuracy_figure(
        acc_df,
        df,
        out_path=figures_dir / "binning_benchmark_accuracy",
        fmt=fmt,
    )
    if path is not None:
        written.append(path)

    written.append(_save_legend_figure(figures_dir / "binning_benchmark_legend", fmt))
    return written


def main() -> None:
    args = parse_args()
    _apply_plot_style()

    results_path = Path(args.results_csv)
    summary_path = Path(args.results_summary_csv)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results CSV: {results_path}")

    df = _load_batch_effect_results(results_path)
    fmt = args.format

    acc_df = _load_best_accuracy(summary_path)
    if acc_df.empty:
        print(
            f"Warning: no Accuracy rows for binning prep_modes in {summary_path}; "
            f"skipping the training-accuracy panel."
        )

    written = _plot_all_figures(df, acc_df, figures_dir, fmt)
    for path in written:
        print(f"Figure written to {path}")


if __name__ == "__main__":
    main()
