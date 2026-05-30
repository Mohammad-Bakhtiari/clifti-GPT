#!/usr/bin/env python
"""Multi-dataset federated binning benchmark (no model training)."""

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from analysis.binning_benchmark import OUTPUT_DIR
from analysis.binning_benchmark.config import (
    BATCH_METRICS,
    BINNING_STRATEGIES,
    METRIC_PREFIX,
    PRIMARY_DATASETS,
)
from analysis.binning_benchmark.core import (
    compute_all_strategy_edges,
    edges_npz_payload,
    load_partition_from_path,
    per_strategy_client_bins,
)
from analysis.binning_benchmark.metrics import heterogeneity_index, strategy_batch_metrics
from cliftiGPT.preprocessor.aggregation import (
    aggregate_bin_edge_contributions_smpc,
    aggregate_bin_edges,
    aggregate_histogram_bin_edges_plain,
    aggregate_secure_histogram_bin_edges,
)
from cliftiGPT.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/scgpt/benchmark",
        help="Root directory containing per-dataset benchmark folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory for results.csv, results_wide.csv, summary.md, per_dataset/.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(PRIMARY_DATASETS.keys()),
        help=f"Comma-separated dataset names (default: all primary). Choices: {', '.join(PRIMARY_DATASETS)}",
    )
    parser.add_argument("--n_bins", type=int, default=51)
    parser.add_argument("--grid_resolution", type=int, default=4096)
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="adata.layers[layer] to bin; default uses adata.X.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--time",
        action="store_true",
        default=False,
        help=(
            "Also record wall-clock timings for plain vs SMPC aggregation. "
            "Writes timings.csv into --output_dir."
        ),
    )
    parser.add_argument(
        "--time_n_reps",
        type=int,
        default=3,
        help="Repetitions per timed aggregation step (median is reported).",
    )
    return parser.parse_args()


def _median_seconds(fn, n_reps: int) -> float:
    fn()
    samples: List[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _time_aggregations(result, time_n_reps: int) -> Dict[str, float]:
    t_weighted_plain = _median_seconds(
        lambda: aggregate_bin_edges(result.local_edges_pairs),
        time_n_reps,
    )
    t_weighted_smpc = _median_seconds(
        lambda: aggregate_bin_edge_contributions_smpc(result.contrib_shares),
        time_n_reps,
    )
    t_hist_plain = _median_seconds(
        lambda: aggregate_histogram_bin_edges_plain(
            result.client_histograms,
            result.client_n_list,
            result.value_grid,
            len(result.strategy_edges["centralized"]) + 1,
        ),
        time_n_reps,
    )
    t_hist_smpc = _median_seconds(
        lambda: aggregate_secure_histogram_bin_edges(
            result.hist_shares,
            result.n_shares_hist,
            result.value_grid_smpc,
            len(result.strategy_edges["centralized"]) + 1,
        ),
        time_n_reps,
    )
    return {
        "fed-weight-avg": t_weighted_plain,
        "fed-weight-avg-smpc": t_weighted_smpc,
        "fed-hist": t_hist_plain,
        "fed-hist-smpc": t_hist_smpc,
    }


def evaluate_dataset(
    adata_path: Path,
    batch_key: str,
    n_bins: int,
    grid_resolution: int,
    layer: Optional[str],
    record_time: bool = False,
    time_n_reps: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Optional[Dict[str, float]]]:
    partition = load_partition_from_path(adata_path, batch_key, n_bins, layer)
    result = compute_all_strategy_edges(partition, n_bins, grid_resolution)
    js_raw = heterogeneity_index(partition.per_client_nonzero)
    per_strategy_bins = per_strategy_client_bins(partition, result.strategy_edges)

    metrics: Dict[str, Any] = {
        "n_clients": len(partition.unique_batches),
        "n_cells": partition.n_cells,
        "n_nonzero": int(partition.pooled_nonzero.size),
        "js_raw": js_raw,
        "n_bins": n_bins,
        "grid_resolution": grid_resolution,
        "max_expr": result.max_expr,
        "max_expr_smpc": result.max_expr_smpc,
    }

    for strategy in BINNING_STRATEGIES:
        batch = strategy_batch_metrics(
            per_strategy_bins[strategy],
            partition.unique_batches,
            js_raw,
            n_bins,
        )
        prefix = METRIC_PREFIX[strategy]
        metrics[f"cramers_v_{prefix}"] = batch["cramers_v"]
        metrics[f"js_binned_{prefix}"] = batch["js_binned"]
        metrics[f"js_amplification_{prefix}"] = batch["js_amplification"]

    timings = _time_aggregations(result, time_n_reps) if record_time else None
    return metrics, edges_npz_payload(result), timings


def _long_rows(dataset: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = {
        "dataset": dataset,
        "n_clients": metrics["n_clients"],
        "n_cells": metrics["n_cells"],
        "n_nonzero": metrics["n_nonzero"],
        "js_raw": metrics["js_raw"],
    }
    for strategy in BINNING_STRATEGIES:
        prefix = METRIC_PREFIX[strategy]
        for metric_name in BATCH_METRICS:
            rows.append(
                {
                    **base,
                    "strategy": strategy,
                    "metric": metric_name,
                    "value": metrics[f"{metric_name}_{prefix}"],
                }
            )
    return rows


def _write_summary_md(path: Path, wide_rows: List[Dict[str, Any]]) -> None:
    if not wide_rows:
        path.write_text("# Binning benchmark\n\nNo datasets evaluated.\n")
        return

    def _mean(col: str) -> float:
        return float(np.mean([r[col] for r in wide_rows]))

    lines = [
        "# Binning benchmark summary\n",
        "Batch-effect metrics (lower is better). JS_raw is pre-binning client heterogeneity.\n\n",
        "| Metric | centralized | fed-weight-avg | fed-weight-avg-smpc | fed-hist | fed-hist-smpc |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for label, metric_key in [
        ("Cramér's V (client x bin)", "cramers_v"),
        ("JS amplification (binned/raw)", "js_amplification"),
    ]:
        vals = [_mean(f"{metric_key}_{METRIC_PREFIX[s]}") for s in BINNING_STRATEGIES]
        lines.append(
            f"| {label} | {vals[0]:.6g} | {vals[1]:.6g} | {vals[2]:.6g} | {vals[3]:.6g} | {vals[4]:.6g} |\n"
        )

    lines.append("\n## Per dataset\n\n")
    lines.append(
        "| Dataset | JS_raw | Cramér V weighted | Cramér V hist | Cramér V hist SMPC | JS amp weighted | JS amp hist | JS amp hist SMPC |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for row in wide_rows:
        lines.append(
            f"| {row['dataset']} | {row['js_raw']:.4g} "
            f"| {row['cramers_v_weighted']:.6g} "
            f"| {row['cramers_v_hist']:.6g} "
            f"| {row['cramers_v_hist_smpc']:.6g} "
            f"| {row['js_amplification_weighted']:.6g} "
            f"| {row['js_amplification_hist']:.6g} "
            f"| {row['js_amplification_hist_smpc']:.6g} |\n"
        )
    path.write_text("".join(lines))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_dataset_dir = output_dir / "per_dataset"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in requested if d not in PRIMARY_DATASETS]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {unknown}. "
            f"Available: {list(PRIMARY_DATASETS.keys())}"
        )

    long_rows: List[Dict[str, Any]] = []
    wide_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    timing_rows: List[Dict[str, Any]] = []

    for name in requested:
        cfg = PRIMARY_DATASETS[name]
        adata_path = data_root / cfg["slug"] / cfg["reference_file"]
        print(f"\n=== {name} ===")
        if not adata_path.is_file():
            msg = f"Missing file: {adata_path}"
            print(f"SKIP: {msg}")
            skipped.append({"dataset": name, "reason": msg})
            continue
        try:
            metrics, edges, timings = evaluate_dataset(
                adata_path=adata_path,
                batch_key=cfg["batch_key"],
                n_bins=args.n_bins,
                grid_resolution=args.grid_resolution,
                layer=args.layer,
                record_time=args.time,
                time_n_reps=args.time_n_reps,
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"SKIP: {msg}")
            skipped.append({"dataset": name, "reason": msg})
            traceback.print_exc()
            continue

        wide_row = {"dataset": name, **metrics}
        wide_rows.append(wide_row)
        long_rows.extend(_long_rows(name, metrics))

        if timings is not None:
            plain_t_weighted = timings["fed-weight-avg"]
            plain_t_hist = timings["fed-hist"]
            for strategy, t_seconds in timings.items():
                if strategy.endswith("-smpc"):
                    base = strategy.replace("-smpc", "")
                    base_t = (
                        plain_t_weighted if base == "fed-weight-avg" else plain_t_hist
                    )
                    overhead = (t_seconds / base_t) if base_t > 0 else float("nan")
                else:
                    overhead = 1.0
                timing_rows.append(
                    {
                        "dataset": name,
                        "n_clients": metrics["n_clients"],
                        "n_cells": metrics["n_cells"],
                        "n_nonzero": metrics["n_nonzero"],
                        "strategy": strategy,
                        "t_seconds": t_seconds,
                        "crypto_overhead": overhead,
                    }
                )
            print(
                f"TIMING: weighted plain={plain_t_weighted*1e3:7.2f}ms "
                f"smpc={timings['fed-weight-avg-smpc']*1e3:7.2f}ms | "
                f"hist plain={plain_t_hist*1e3:7.2f}ms "
                f"smpc={timings['fed-hist-smpc']*1e3:7.2f}ms"
            )

        ds_out = per_dataset_dir / name
        ds_out.mkdir(parents=True, exist_ok=True)
        np.savez(ds_out / "edges.npz", **edges)
        print(
            f"OK: {metrics['n_clients']} clients, js_raw={metrics['js_raw']:.4g}, "
            f"Cramér V weighted={metrics['cramers_v_weighted']:.6g}, "
            f"weighted_smpc={metrics['cramers_v_weighted_smpc']:.6g}, "
            f"hist={metrics['cramers_v_hist']:.6g}, "
            f"hist_smpc={metrics['cramers_v_hist_smpc']:.6g}, "
            f"JS amp weighted={metrics['js_amplification_weighted']:.6g}, "
            f"weighted_smpc={metrics['js_amplification_weighted_smpc']:.6g}, "
            f"hist={metrics['js_amplification_hist']:.6g}, "
            f"hist_smpc={metrics['js_amplification_hist_smpc']:.6g}"
        )

    long_fields = [
        "dataset", "n_clients", "n_cells", "n_nonzero", "js_raw",
        "strategy", "metric", "value",
    ]
    wide_fields = [
        "dataset", "n_clients", "n_cells", "n_nonzero", "js_raw",
        "n_bins", "grid_resolution", "max_expr",
    ]
    for strategy in BINNING_STRATEGIES:
        prefix = METRIC_PREFIX[strategy]
        for metric_name in BATCH_METRICS:
            wide_fields.append(f"{metric_name}_{prefix}")

    results_csv = output_dir / "results.csv"
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=long_fields)
        writer.writeheader()
        writer.writerows(long_rows)

    wide_csv = output_dir / "results_wide.csv"
    with wide_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=wide_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(wide_rows)

    skipped_csv = output_dir / "skipped.csv"
    with skipped_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "reason"])
        writer.writeheader()
        writer.writerows(skipped)

    _write_summary_md(output_dir / "summary.md", wide_rows)

    if args.time:
        timings_csv = output_dir / "timings.csv"
        timing_fields = [
            "dataset", "n_clients", "n_cells", "n_nonzero",
            "strategy", "t_seconds", "crypto_overhead",
        ]
        with timings_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=timing_fields)
            writer.writeheader()
            writer.writerows(timing_rows)
        print(f"Wrote: {timings_csv} ({len(timing_rows)} rows)")

    print(f"\nEvaluated {len(wide_rows)} / {len(requested)} datasets.")
    print(f"Wrote: {results_csv}")
    print(f"Wrote: {wide_csv}")
    print(f"Wrote: {output_dir / 'summary.md'}")
    if skipped:
        print(f"Wrote: {skipped_csv} ({len(skipped)} skipped)")


if __name__ == "__main__":
    main()
