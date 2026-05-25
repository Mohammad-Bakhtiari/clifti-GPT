#!/usr/bin/env python
"""Multi-dataset federated binning benchmark (no model training).

Evaluates batch-effect and heterogeneity impact of four federated binning
strategies on the five primary annotation benchmark datasets:

- Cramér's V (client x bin association; higher = stronger batch effect)
- JS amplification (JS_binned / JS_raw; higher = binning inflates client separation)

See analysis/plot_binning_benchmark.py for figures.
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import anndata
import crypten
import numpy as np
import torch

from cliftiGPT.preprocessor.aggregation import (
    aggregate_bin_edges,
    aggregate_bin_edge_contributions_smpc,
    aggregate_global_max_expr,
    aggregate_histogram_bin_edges_plain,
    aggregate_secure_histogram_bin_edges,
    local_bin_edge_contribution,
    reveal_nonzero_total,
    secure_reveal_envelope_max,
)
from cliftiGPT.utils import set_seed

PRIMARY_DATASETS: Dict[str, Dict[str, str]] = {
    "MS": {
        "slug": "ms",
        "reference_file": "reference_annot.h5ad",
        "batch_key": "split_label",
    },
    "CellLine": {
        "slug": "cl",
        "reference_file": "reference.h5ad",
        "batch_key": "batch",
    },
    "LUNG": {
        "slug": "lung",
        "reference_file": "reference_annot.h5ad",
        "batch_key": "sample",
    },
    "MYELOID-top5": {
        "slug": "myeloid-top5",
        "reference_file": "reference.h5ad",
        "batch_key": "combined_batch",
    },
    "HP5": {
        "slug": "hp5",
        "reference_file": "reference.h5ad",
        "batch_key": "batch_name",
    },
}

FEDERATED_STRATEGIES = (
    "fed-weight-avg",
    "fed-weight-avg-smpc",
    "fed-hist",
    "fed-hist-smpc",
)

METRIC_PREFIX = {
    "fed-weight-avg": "weighted",
    "fed-weight-avg-smpc": "weighted_smpc",
    "fed-hist": "hist",
    "fed-hist-smpc": "hist_smpc",
}

BATCH_METRICS = ("cramers_v", "js_binned", "js_amplification")


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
        default="output/binning_benchmark",
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
    return parser.parse_args()


def _to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _quantile_probs(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins - 1)


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=True)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def mean_pairwise_js(hist_list: List[np.ndarray]) -> float:
    """Mean pairwise JS divergence over a list of histogram/count vectors."""
    if len(hist_list) < 2:
        return 0.0
    pairs = []
    for i in range(len(hist_list)):
        for j in range(i + 1, len(hist_list)):
            pairs.append(_js_divergence(hist_list[i], hist_list[j]))
    return float(np.mean(pairs)) if pairs else 0.0


def heterogeneity_index(
    per_client_nonzero: List[np.ndarray],
    n_grid: int = 256,
) -> float:
    """Mean pairwise JS divergence of client value histograms on continuous data."""
    if len(per_client_nonzero) < 2:
        return 0.0
    all_vals = np.concatenate([nz for nz in per_client_nonzero if nz.size > 0])
    if all_vals.size == 0:
        return 0.0
    grid = np.linspace(0.0, float(all_vals.max()), n_grid + 1, dtype=np.float64)
    hists = []
    for nz in per_client_nonzero:
        if nz.size == 0:
            hists.append(np.zeros(n_grid, dtype=np.float64))
        else:
            counts, _ = np.histogram(nz, bins=grid)
            hists.append(counts.astype(np.float64))
    return mean_pairwise_js(hists)


def client_bin_histograms(
    per_client_bins: List[np.ndarray],
    n_bins: int,
) -> List[np.ndarray]:
    """Count vectors over bin indices 1..n_bins-1 for each client."""
    n_categories = n_bins - 1
    hists = []
    for bins in per_client_bins:
        if bins.size == 0:
            hists.append(np.zeros(n_categories, dtype=np.float64))
        else:
            clipped = np.clip(bins, 1, n_categories)
            counts = np.bincount(clipped, minlength=n_categories + 1)[1:]
            hists.append(counts.astype(np.float64))
    return hists


def cramers_v(client_ids: np.ndarray, bin_ids: np.ndarray) -> float:
    """Cramér's V for client x bin association (0 = independent, 1 = perfect association)."""
    client_ids = np.asarray(client_ids)
    bin_ids = np.asarray(bin_ids)
    if client_ids.size == 0:
        return 0.0

    client_codes, client_labels = np.unique(client_ids, return_inverse=True)
    bin_codes, bin_labels = np.unique(bin_ids, return_inverse=True)
    n_clients = len(client_codes)
    n_bins_obs = len(bin_codes)
    if n_clients < 2 or n_bins_obs < 2:
        return 0.0

    contingency = np.zeros((n_clients, n_bins_obs), dtype=np.float64)
    np.add.at(contingency, (client_labels, bin_labels), 1.0)

    n = contingency.sum()
    if n <= 0:
        return 0.0

    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    mask = expected > 0
    chi2 = np.sum(((contingency - expected) ** 2 / expected)[mask])

    k = min(n_clients - 1, n_bins_obs - 1)
    if k <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def strategy_batch_metrics(
    per_client_bins: List[np.ndarray],
    client_names: List[str],
    js_raw: float,
    n_bins: int,
) -> Dict[str, float]:
    """Cramér's V, JS_binned, and JS amplification for one binning strategy."""
    hists = client_bin_histograms(per_client_bins, n_bins)
    js_binned = mean_pairwise_js(hists)

    client_id_repeated = []
    bin_id_repeated = []
    for client_name, bins in zip(client_names, per_client_bins):
        if bins.size == 0:
            continue
        client_id_repeated.append(np.full(bins.size, client_name))
        bin_id_repeated.append(bins)
    if client_id_repeated:
        all_clients = np.concatenate(client_id_repeated)
        all_bins = np.concatenate(bin_id_repeated)
        v = cramers_v(all_clients, all_bins)
    else:
        v = 0.0

    js_amplification = js_binned / max(js_raw, 1e-12)
    return {
        "cramers_v": v,
        "js_binned": js_binned,
        "js_amplification": js_amplification,
    }


def evaluate_dataset(
    adata_path: Path,
    batch_key: str,
    n_bins: int,
    grid_resolution: int,
    layer: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    adata = anndata.read_h5ad(adata_path)

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(
                f"Layer '{layer}' not in adata.layers (available: {list(adata.layers.keys())})."
            )
        layer_values = _to_dense(adata.layers[layer])
    else:
        layer_values = _to_dense(adata.X)

    if layer_values.min() < 0:
        raise ValueError(
            f"Expected non-negative data for binning, got min={layer_values.min()}."
        )

    if batch_key not in adata.obs.columns:
        raise KeyError(
            f"batch_key '{batch_key}' not in adata.obs (available: {list(adata.obs.columns)})."
        )

    batches = adata.obs[batch_key].astype(str).values
    unique_batches = sorted(set(batches))
    if len(unique_batches) < 2:
        raise ValueError(
            f"Need >= 2 partitions, got {len(unique_batches)} from batch_key '{batch_key}'."
        )

    probs = _quantile_probs(n_bins)
    pooled_nonzero = layer_values[layer_values > 0]
    if pooled_nonzero.size == 0:
        raise ValueError("All values are zero; cannot derive quantile bins.")

    local_edges_pairs = []
    per_client_nonzero = []
    client_max_list = []
    client_n_list = []
    for b in unique_batches:
        mask = batches == b
        sub = layer_values[mask]
        nz = sub[sub > 0]
        per_client_nonzero.append(nz)
        client_max_list.append(float(nz.max()) if nz.size > 0 else 0.0)
        client_n_list.append(int(nz.size))
        local_edges = (
            np.quantile(nz, probs).astype(np.float32)
            if nz.size > 0
            else np.zeros(len(probs), dtype=np.float32)
        )
        local_edges_pairs.append((local_edges, int(nz.size)))

    js_raw = heterogeneity_index(per_client_nonzero)

    edges_weighted = aggregate_bin_edges(local_edges_pairs).astype(np.float32)

    n_shares = [
        crypten.cryptensor(torch.tensor([float(n)], dtype=torch.float32))
        for _, n in local_edges_pairs
    ]
    total_n = reveal_nonzero_total(n_shares)
    contrib_shares = [
        crypten.cryptensor(
            torch.tensor(local_bin_edge_contribution(local_edges, n, total_n), dtype=torch.float32)
        )
        for local_edges, n in local_edges_pairs
    ]
    edges_weighted_smpc = aggregate_bin_edge_contributions_smpc(contrib_shares).astype(np.float32)

    max_expr = aggregate_global_max_expr(client_max_list)
    value_grid = np.linspace(0.0, max_expr, grid_resolution + 1, dtype=np.float32)
    client_histograms = [
        np.histogram(nz, bins=value_grid)[0].astype(np.float64)
        for nz in per_client_nonzero
    ]
    edges_hist = aggregate_histogram_bin_edges_plain(
        client_histograms, client_n_list, value_grid, n_bins
    ).astype(np.float32)

    max_shares = [
        crypten.cryptensor(torch.tensor([float(m)], dtype=torch.float32))
        for m in client_max_list
    ]
    n_shares_hist = [
        crypten.cryptensor(torch.tensor([float(n)], dtype=torch.float32))
        for n in client_n_list
    ]
    max_expr_smpc = secure_reveal_envelope_max(max_shares)
    value_grid_smpc = np.linspace(
        0.0, max_expr_smpc, grid_resolution + 1, dtype=np.float32
    )
    hist_shares = [
        crypten.cryptensor(
            torch.tensor(
                np.histogram(nz, bins=value_grid_smpc)[0].astype(np.float32)
            )
        )
        for nz in per_client_nonzero
    ]
    edges_hist_smpc = aggregate_secure_histogram_bin_edges(
        hist_shares, n_shares_hist, value_grid_smpc, n_bins
    ).astype(np.float32)

    strategy_edges = {
        "fed-weight-avg": edges_weighted,
        "fed-weight-avg-smpc": edges_weighted_smpc,
        "fed-hist": edges_hist,
        "fed-hist-smpc": edges_hist_smpc,
    }

    per_strategy_bins: Dict[str, List[np.ndarray]] = {}
    for strategy, edges in strategy_edges.items():
        per_strategy_bins[strategy] = [
            _bin_indices(nz, edges) for nz in per_client_nonzero
        ]

    metrics: Dict[str, Any] = {
        "n_clients": len(unique_batches),
        "n_cells": int(adata.n_obs),
        "n_nonzero": int(pooled_nonzero.size),
        "js_raw": js_raw,
        "n_bins": n_bins,
        "grid_resolution": grid_resolution,
        "max_expr": float(max_expr),
        "max_expr_smpc": float(max_expr_smpc),
    }

    for strategy in FEDERATED_STRATEGIES:
        batch = strategy_batch_metrics(
            per_strategy_bins[strategy],
            unique_batches,
            js_raw,
            n_bins,
        )
        prefix = METRIC_PREFIX[strategy]
        metrics[f"cramers_v_{prefix}"] = batch["cramers_v"]
        metrics[f"js_binned_{prefix}"] = batch["js_binned"]
        metrics[f"js_amplification_{prefix}"] = batch["js_amplification"]

    edges = {
        "fed_weight_avg": edges_weighted,
        "fed_weight_avg_smpc": edges_weighted_smpc,
        "fed_hist": edges_hist,
        "fed_hist_smpc": edges_hist_smpc,
        "value_grid": value_grid,
        "value_grid_smpc": value_grid_smpc,
    }
    return metrics, edges


def _long_rows(dataset: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = {
        "dataset": dataset,
        "n_clients": metrics["n_clients"],
        "n_cells": metrics["n_cells"],
        "n_nonzero": metrics["n_nonzero"],
        "js_raw": metrics["js_raw"],
    }

    for strategy in FEDERATED_STRATEGIES:
        prefix = METRIC_PREFIX[strategy]
        for metric_name in BATCH_METRICS:
            rows.append({
                **base,
                "strategy": strategy,
                "metric": metric_name,
                "value": metrics[f"{metric_name}_{prefix}"],
            })
    return rows


def _write_summary_md(
    path: Path,
    wide_rows: List[Dict[str, Any]],
) -> None:
    if not wide_rows:
        path.write_text("# Binning benchmark\n\nNo datasets evaluated.\n")
        return

    def _mean(col: str) -> float:
        return float(np.mean([r[col] for r in wide_rows]))

    lines = [
        "# Binning benchmark summary\n",
        "Batch-effect metrics (lower is better). JS_raw is pre-binning client heterogeneity.\n\n",
        "| Metric | fed-weight-avg | fed-weight-avg-smpc | fed-hist | fed-hist-smpc |\n",
        "|---|---|---|---|---|\n",
    ]
    for label, metric_key in [
        ("Cramér's V (client x bin)", "cramers_v"),
        ("JS amplification (binned/raw)", "js_amplification"),
    ]:
        vals = [_mean(f"{metric_key}_{METRIC_PREFIX[s]}") for s in FEDERATED_STRATEGIES]
        lines.append(
            f"| {label} | {vals[0]:.6g} | {vals[1]:.6g} | {vals[2]:.6g} | {vals[3]:.6g} |\n"
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
            metrics, edges = evaluate_dataset(
                adata_path=adata_path,
                batch_key=cfg["batch_key"],
                n_bins=args.n_bins,
                grid_resolution=args.grid_resolution,
                layer=args.layer,
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
    for strategy in FEDERATED_STRATEGIES:
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

    print(f"\nEvaluated {len(wide_rows)} / {len(requested)} datasets.")
    print(f"Wrote: {results_csv}")
    print(f"Wrote: {wide_csv}")
    print(f"Wrote: {output_dir / 'summary.md'}")
    if skipped:
        print(f"Wrote: {skipped_csv} ({len(skipped)} skipped)")


if __name__ == "__main__":
    main()
