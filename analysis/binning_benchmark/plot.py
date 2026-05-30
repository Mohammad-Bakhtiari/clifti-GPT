#!/usr/bin/env python
"""Plot binning benchmark results from benchmark.py."""

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.binning_benchmark import OUTPUT_DIR
from analysis.binning_benchmark.config import (
    BATCH_METRICS,
    BINNING_STRATEGIES,
    DATASET_SLUG_TO_DISPLAY,
    WIDE_METRIC_PREFIX,
)

FONT_SIZE = 11
PANEL_DPI = 200
PANEL_FIGSIZE = (6.5, 3.5)
BAR_EDGE_COLOR = "0.15"
BAR_EDGE_WIDTH = 0.6
ACCURACY_YLIM = (0.0, 1.5)
SMPC_HATCH = "///"

# Same colorblind-friendly palette as analysis/utils.py (inlined to avoid that import).
_STRATEGY_BASE_COLORS = {
    "centralized": "#949494",
    "fed-weight-avg": "#0173b2",
    "fed-hist": "#029e73",
}

STRATEGY_LABELS = {
    "centralized": "Centralized",
    "fed-weight-avg": "Fed-weight-avg",
    "fed-weight-avg-smpc": "Fed-weight-avg-SMPC",
    "fed-hist": "Fed-hist-binning",
    "fed-hist-smpc": "Fed-hist-binning-SMPC",
}
STRATEGY_COLORS = {
    strategy: _STRATEGY_BASE_COLORS[strategy.replace("-smpc", "")]
    for strategy in BINNING_STRATEGIES
}
STRATEGY_HATCHES = {
    strategy: SMPC_HATCH if strategy.endswith("-smpc") else ""
    for strategy in BINNING_STRATEGIES
}
STRATEGY_ALIASES = {
    "federated": "fed-weight-avg",
    "federated_smpc": "fed-weight-avg-smpc",
    "histogram": "fed-hist",
    "histogram_smpc": "fed-hist-smpc",
    "fed_hist": "fed-hist",
    "fed_hist_smpc": "fed-hist-smpc",
}

FIGURE_STEMS = {
    "cramers_v": "binning_benchmark_cramers_v",
    "js_amplification": "binning_benchmark_js_amplification",
    "accuracy": "binning_benchmark_accuracy",
    "legend": "binning_benchmark_legend",
}

METRIC_PANELS = (
    ("cramers_v", "Cramér's V", None),
    ("js_amplification", "JS_binned / JS_raw", 1.0),
)

_EMPTY_ACC_COLUMNS = ["dataset", "strategy", "best_accuracy", "best_round"]


def _apply_plot_style() -> None:
    sns.set_theme(style="whitegrid")
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


def _normalize_strategy_names(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace(STRATEGY_ALIASES)


def _wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        c
        for c in ("dataset", "n_clients", "n_cells", "n_nonzero", "js_raw")
        if c in wide.columns
    ]
    rows: List[dict] = []
    for _, row in wide.iterrows():
        base = {c: row[c] for c in base_cols}
        for prefix, strategy in WIDE_METRIC_PREFIX.items():
            for metric in BATCH_METRICS:
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

    missing = sorted(set(BINNING_STRATEGIES) - set(long_df["strategy"].unique()))
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

    return long_df.drop_duplicates(
        subset=["dataset", "strategy", "metric"], keep="last"
    )


def _ordered_datasets(primary: pd.DataFrame, secondary: pd.DataFrame) -> List[str]:
    primary_names = list(primary["dataset"].unique()) if not primary.empty else []
    extra = [
        name
        for name in secondary["dataset"].unique()
        if name not in primary_names
    ]
    return primary_names + extra


def _prepare_barplot_df(
    table: pd.DataFrame,
    datasets: List[str],
    value_col: str,
) -> pd.DataFrame:
    plot_df = table.rename(columns={value_col: "value"}).copy()
    plot_df["dataset"] = pd.Categorical(
        plot_df["dataset"], categories=datasets, ordered=True
    )
    plot_df["strategy"] = pd.Categorical(
        plot_df["strategy"], categories=list(BINNING_STRATEGIES), ordered=True
    )
    return plot_df.sort_values(["dataset", "strategy"])


def _style_strategy_bars(ax: Axes) -> None:
    for strategy, container in zip(BINNING_STRATEGIES, ax.containers):
        hatch = STRATEGY_HATCHES[strategy]
        for bar in container:
            bar.set_hatch(hatch)
            bar.set_edgecolor(BAR_EDGE_COLOR)
            bar.set_linewidth(BAR_EDGE_WIDTH)


def _draw_grouped_bars(
    ax: Axes,
    datasets: List[str],
    table: pd.DataFrame,
    value_col: str,
    ylabel: str,
    reference_line: Optional[float] = None,
    ylim: Optional[tuple] = None,
    annotate: Optional[Callable[[Axes, object, str, str], None]] = None,
) -> None:
    plot_df = _prepare_barplot_df(table, datasets, value_col)
    sns.barplot(
        data=plot_df,
        x="dataset",
        y="value",
        hue="strategy",
        order=datasets,
        hue_order=list(BINNING_STRATEGIES),
        palette=STRATEGY_COLORS,
        ax=ax,
        width=0.8,
        dodge=True,
    )
    _style_strategy_bars(ax)
    if ax.legend_ is not None:
        ax.legend_.remove()

    if annotate is not None:
        for strategy, container in zip(BINNING_STRATEGIES, ax.containers):
            for bar, dataset in zip(container, datasets):
                annotate(ax, bar, dataset, strategy)

    if reference_line is not None:
        ax.axhline(reference_line, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_csv",
        type=str,
        default=f"{OUTPUT_DIR}/results.csv",
    )
    parser.add_argument(
        "--results_summary_csv",
        type=str,
        default=str(_REPO_ROOT / "output/annotation/results_summary.csv"),
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default=f"{OUTPUT_DIR}/figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=("pdf", "png", "svg"),
        default="png",
    )
    return parser.parse_args()


def _empty_accuracy_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_ACC_COLUMNS)


def _load_best_accuracy(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        return _empty_accuracy_df()

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
    df = df[df["prep_mode"].isin(BINNING_STRATEGIES)]
    if df.empty:
        return _empty_accuracy_df()

    dedup_keys = ["Dataset", "Round", "prep_mode"]
    for opt in ("n_epochs", "mu", "Aggregation"):
        if opt in df.columns:
            dedup_keys.append(opt)
    df = df.drop_duplicates(subset=dedup_keys, keep="last")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    if df.empty:
        return _empty_accuracy_df()

    idx = df.groupby(["Dataset", "prep_mode"])["Value"].idxmax()
    best = df.loc[idx, ["Dataset", "prep_mode", "Value", "Round"]].copy()
    best = best.rename(columns={"Value": "best_accuracy", "Round": "best_round"})
    best["dataset"] = best["Dataset"].map(DATASET_SLUG_TO_DISPLAY).fillna(best["Dataset"])
    best = best.rename(columns={"prep_mode": "strategy"})
    return best[["dataset", "strategy", "best_accuracy", "best_round"]].reset_index(
        drop=True
    )


def _annotate_round(ax: Axes, bar, dataset: str, strategy: str, acc_df: pd.DataFrame) -> None:
    row = acc_df[(acc_df["dataset"] == dataset) & (acc_df["strategy"] == strategy)]
    if row.empty or not np.isfinite(bar.get_height()):
        return
    round_no = int(row["best_round"].iloc[0])
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height(),
        f"{round_no}",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
    )


def _metric_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df[df["metric"] == metric][["dataset", "strategy", "value"]].copy()


def _save_figure(fig: Figure, out_path: Path, fmt: str) -> Path:
    path = out_path.with_suffix(f".{fmt}")
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=PANEL_DPI,
    )
    plt.close(fig)
    return path


def _save_panel(
    draw: Callable[[Axes], None],
    out_path: Path,
    fmt: str,
) -> Path:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    draw(ax)
    fig.tight_layout()
    return _save_figure(fig, out_path, fmt)


def _save_metric_panel(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
    fmt: str,
    reference_line: Optional[float] = None,
) -> Optional[Path]:
    table = _metric_table(df, metric)
    if table.empty:
        print(f"Skip {out_path.name}: no rows for metric={metric}")
        return None

    def draw(ax: Axes) -> None:
        datasets = list(table["dataset"].unique())
        _draw_grouped_bars(
            ax,
            datasets,
            table,
            value_col="value",
            ylabel=ylabel,
            reference_line=reference_line,
        )

    return _save_panel(draw, out_path, fmt)


def _save_accuracy_panel(
    acc_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    out_path: Path,
    fmt: str,
) -> Optional[Path]:
    if acc_df.empty:
        print(f"Skip {out_path.name}: no accuracy rows")
        return None

    datasets = _ordered_datasets(metric_df, acc_df)

    def draw(ax: Axes) -> None:
        annotate = lambda ax_, bar, ds, strategy: _annotate_round(
            ax_, bar, ds, strategy, acc_df
        )
        _draw_grouped_bars(
            ax,
            datasets,
            acc_df,
            value_col="best_accuracy",
            ylabel="Peak Accuracy",
            ylim=ACCURACY_YLIM,
            annotate=annotate,
        )

    return _save_panel(draw, out_path, fmt)


def _save_legend_figure(out_path: Path, fmt: str) -> Path:
    handles = [
        Patch(
            facecolor=STRATEGY_COLORS[s],
            edgecolor=BAR_EDGE_COLOR,
            hatch=STRATEGY_HATCHES[s],
            linewidth=BAR_EDGE_WIDTH,
            label=STRATEGY_LABELS[s],
        )
        for s in BINNING_STRATEGIES
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

    fig, ax = plt.subplots(figsize=(max(1.4 * len(handles), 8.0), 0.8))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=FONT_SIZE,
    )
    return _save_figure(fig, out_path, fmt)


def _plot_all_figures(
    df: pd.DataFrame,
    acc_df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
) -> List[Path]:
    written: List[Path] = []

    for metric, ylabel, reference_line in METRIC_PANELS:
        path = _save_metric_panel(
            df,
            metric=metric,
            ylabel=ylabel,
            out_path=figures_dir / FIGURE_STEMS[metric],
            fmt=fmt,
            reference_line=reference_line,
        )
        if path is not None:
            written.append(path)

    path = _save_accuracy_panel(
        acc_df,
        df,
        out_path=figures_dir / FIGURE_STEMS["accuracy"],
        fmt=fmt,
    )
    if path is not None:
        written.append(path)

    written.append(_save_legend_figure(figures_dir / FIGURE_STEMS["legend"], fmt))
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
    acc_df = _load_best_accuracy(summary_path)
    if acc_df.empty:
        print(
            f"Warning: no Accuracy rows for binning prep_modes in {summary_path}; "
            f"skipping the training-accuracy panel."
        )

    for path in _plot_all_figures(df, acc_df, figures_dir, args.format):
        print(f"Figure written to {path}")


if __name__ == "__main__":
    main()
