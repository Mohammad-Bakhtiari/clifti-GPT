#!/usr/bin/env python
"""Multi-dataset federated binning benchmark (no model training).

Evaluates three implemented federated binning strategies against centralized
ground truth on the five primary annotation benchmark datasets. Writes tidy
and wide CSV summaries plus optional per-dataset edge arrays.

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
    local_bin_edge_contribution,
    reveal_nonzero_total,
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
)

METRIC_PREFIX = {
    "fed-weight-avg": "weighted",
    "fed-weight-avg-smpc": "weighted_smpc",
    "fed-hist": "hist",
}


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


def _edge_metrics(edges_a: np.ndarray, edges_b: np.ndarray) -> Dict[str, float]:
    diff = np.abs(edges_a - edges_b)
    return {"L1": float(diff.mean()), "Linf": float(diff.max())}


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


def heterogeneity_index(
    per_client_nonzero: List[np.ndarray],
    n_grid: int = 256,
) -> float:
    """Mean pairwise JS divergence of client value histograms (non-IID proxy)."""
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
    pairs = []
    for i in range(len(hists)):
        for j in range(i + 1, len(hists)):
            pairs.append(_js_divergence(hists[i], hists[j]))
    return float(np.mean(pairs)) if pairs else 0.0


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

    edges_centralized = np.quantile(pooled_nonzero, probs).astype(np.float32)

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
        local_edges = np.quantile(nz, probs).astype(np.float32) if nz.size > 0 else np.zeros(len(probs), dtype=np.float32)
        local_edges_pairs.append((local_edges, int(nz.size)))

    het_js = heterogeneity_index(per_client_nonzero)

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

    central_idx = _bin_indices(pooled_nonzero, edges_centralized)
    weighted_idx = _bin_indices(pooled_nonzero, edges_weighted)
    weighted_smpc_idx = _bin_indices(pooled_nonzero, edges_weighted_smpc)
    hist_idx = _bin_indices(pooled_nonzero, edges_hist)

    vs_central = {
        "fed-weight-avg": _edge_metrics(edges_weighted, edges_centralized),
        "fed-weight-avg-smpc": _edge_metrics(edges_weighted_smpc, edges_centralized),
        "fed-hist": _edge_metrics(edges_hist, edges_centralized),
    }
    vs_weighted = {
        "fed-weight-avg-smpc": _edge_metrics(edges_weighted_smpc, edges_weighted),
        "fed-hist": _edge_metrics(edges_hist, edges_weighted),
    }

    agreement = {
        "fed-weight-avg": float((weighted_idx == central_idx).mean()),
        "fed-weight-avg-smpc": float((weighted_smpc_idx == central_idx).mean()),
        "fed-hist": float((hist_idx == central_idx).mean()),
    }

    metrics = {
        "n_clients": len(unique_batches),
        "n_cells": int(adata.n_obs),
        "n_nonzero": int(pooled_nonzero.size),
        "heterogeneity_js": het_js,
        "n_bins": n_bins,
        "grid_resolution": grid_resolution,
        "max_expr": float(max_expr),
        "L1_weighted_vs_centralized": vs_central["fed-weight-avg"]["L1"],
        "Linf_weighted_vs_centralized": vs_central["fed-weight-avg"]["Linf"],
        "L1_weighted_smpc_vs_centralized": vs_central["fed-weight-avg-smpc"]["L1"],
        "Linf_weighted_smpc_vs_centralized": vs_central["fed-weight-avg-smpc"]["Linf"],
        "L1_hist_vs_centralized": vs_central["fed-hist"]["L1"],
        "Linf_hist_vs_centralized": vs_central["fed-hist"]["Linf"],
        "L1_weighted_smpc_vs_weighted": vs_weighted["fed-weight-avg-smpc"]["L1"],
        "Linf_weighted_smpc_vs_weighted": vs_weighted["fed-weight-avg-smpc"]["Linf"],
        "L1_hist_vs_weighted": vs_weighted["fed-hist"]["L1"],
        "Linf_hist_vs_weighted": vs_weighted["fed-hist"]["Linf"],
        "agreement_weighted_vs_centralized": agreement["fed-weight-avg"],
        "agreement_weighted_smpc_vs_centralized": agreement["fed-weight-avg-smpc"],
        "agreement_hist_vs_centralized": agreement["fed-hist"],
        "nonzero_differing_weighted_smpc_vs_weighted": int((weighted_idx != weighted_smpc_idx).sum()),
        "nonzero_differing_hist_vs_weighted": int((weighted_idx != hist_idx).sum()),
        "nonzero_differing_hist_vs_centralized": int((hist_idx != central_idx).sum()),
    }

    edges = {
        "centralized": edges_centralized,
        "fed_weight_avg": edges_weighted,
        "fed_weight_avg_smpc": edges_weighted_smpc,
        "fed_hist": edges_hist,
        "value_grid": value_grid,
    }
    return metrics, edges


def _long_rows(dataset: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = {
        "dataset": dataset,
        "n_clients": metrics["n_clients"],
        "n_cells": metrics["n_cells"],
        "n_nonzero": metrics["n_nonzero"],
        "heterogeneity_js": metrics["heterogeneity_js"],
    }

    for strategy in FEDERATED_STRATEGIES:
        prefix = METRIC_PREFIX[strategy]
        rows.append({
            **base,
            "strategy": strategy,
            "reference": "centralized",
            "metric": "L1_vs_central",
            "value": metrics[f"L1_{prefix}_vs_centralized"],
        })
        rows.append({
            **base,
            "strategy": strategy,
            "reference": "centralized",
            "metric": "Linf_vs_central",
            "value": metrics[f"Linf_{prefix}_vs_centralized"],
        })
        rows.append({
            **base,
            "strategy": strategy,
            "reference": "centralized",
            "metric": "agreement_vs_central",
            "value": metrics[f"agreement_{prefix}_vs_centralized"],
        })

    for strategy in ("fed-weight-avg-smpc", "fed-hist"):
        prefix = METRIC_PREFIX[strategy]
        rows.append({
            **base,
            "strategy": strategy,
            "reference": "fed-weight-avg",
            "metric": "L1_vs_weighted",
            "value": metrics[f"L1_{prefix}_vs_weighted"],
        })
        rows.append({
            **base,
            "strategy": strategy,
            "reference": "fed-weight-avg",
            "metric": "Linf_vs_weighted",
            "value": metrics[f"Linf_{prefix}_vs_weighted"],
        })

    rows.append({
        **base,
        "strategy": "fed-weight-avg-smpc",
        "reference": "fed-weight-avg",
        "metric": "nonzero_differing",
        "value": metrics["nonzero_differing_weighted_smpc_vs_weighted"],
    })
    rows.append({
        **base,
        "strategy": "fed-hist",
        "reference": "fed-weight-avg",
        "metric": "nonzero_differing",
        "value": metrics["nonzero_differing_hist_vs_weighted"],
    })
    rows.append({
        **base,
        "strategy": "fed-hist",
        "reference": "centralized",
        "metric": "nonzero_differing",
        "value": metrics["nonzero_differing_hist_vs_centralized"],
    })
    return rows


def _write_summary_md(
    path: Path,
    wide_rows: List[Dict[str, Any]],
) -> None:
    if not wide_rows:
        path.write_text("# Binning benchmark\n\nNo datasets evaluated.\n")
        return

    metric_cols = [
        ("Linf_weighted_vs_centralized", "fed-weight-avg"),
        ("Linf_weighted_smpc_vs_centralized", "fed-weight-avg-smpc"),
        ("Linf_hist_vs_centralized", "fed-hist"),
        ("agreement_weighted_vs_centralized", "fed-weight-avg"),
        ("agreement_weighted_smpc_vs_centralized", "fed-weight-avg-smpc"),
        ("agreement_hist_vs_centralized", "fed-hist"),
    ]

    lines = [
        "# Binning benchmark summary\n",
        "Mean metrics across evaluated datasets.\n\n",
        "| Metric | fed-weight-avg | fed-weight-avg-smpc | fed-hist |\n",
        "|---|---|---|---|\n",
    ]

    groups = {
        "fed-weight-avg": [],
        "fed-weight-avg-smpc": [],
        "fed-hist": [],
    }
    for col, strat in metric_cols:
        groups[strat].append(col)

    def _mean(col: str) -> float:
        return float(np.mean([r[col] for r in wide_rows]))

    for label, cols in [
        ("L_inf vs centralized", [c for c, _ in metric_cols[:3]]),
        ("Agreement vs centralized", [c for c, _ in metric_cols[3:]]),
    ]:
        vals = [_mean(c) for c in cols]
        lines.append(
            f"| {label} | {vals[0]:.6g} | {vals[1]:.6g} | {vals[2]:.6g} |\n"
        )

    lines.append("\n## Per dataset\n\n")
    lines.append("| Dataset | Het. JS | Linf weighted | Linf hist | Agree weighted | Agree hist |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for row in wide_rows:
        lines.append(
            f"| {row['dataset']} | {row['heterogeneity_js']:.4g} "
            f"| {row['Linf_weighted_vs_centralized']:.6g} "
            f"| {row['Linf_hist_vs_centralized']:.6g} "
            f"| {row['agreement_weighted_vs_centralized']:.6g} "
            f"| {row['agreement_hist_vs_centralized']:.6g} |\n"
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
            f"OK: {metrics['n_clients']} clients, "
            f"Linf hist vs central = {metrics['Linf_hist_vs_centralized']:.6g}"
        )

    long_fields = [
        "dataset", "n_clients", "n_cells", "n_nonzero", "heterogeneity_js",
        "strategy", "reference", "metric", "value",
    ]
    wide_fields = [
        "dataset", "n_clients", "n_cells", "n_nonzero", "heterogeneity_js",
        "n_bins", "grid_resolution", "max_expr",
        "L1_weighted_vs_centralized", "Linf_weighted_vs_centralized",
        "L1_weighted_smpc_vs_centralized", "Linf_weighted_smpc_vs_centralized",
        "L1_hist_vs_centralized", "Linf_hist_vs_centralized",
        "L1_weighted_smpc_vs_weighted", "Linf_weighted_smpc_vs_weighted",
        "L1_hist_vs_weighted", "Linf_hist_vs_weighted",
        "agreement_weighted_vs_centralized", "agreement_weighted_smpc_vs_centralized",
        "agreement_hist_vs_centralized",
        "nonzero_differing_weighted_smpc_vs_weighted",
        "nonzero_differing_hist_vs_weighted",
        "nonzero_differing_hist_vs_centralized",
    ]

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
