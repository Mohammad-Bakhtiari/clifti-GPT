#!/usr/bin/env python
"""Demonstrator for federated binning (R2.1 / R2.2).

Compares global bin edges from the same AnnData partitioned by ``--batch_key``.
Covers all four federated ``prep_mode`` strategies plus centralized ground truth:

1. **centralized** — pooled ``np.quantile`` on all non-zero values.
2. **fed-weight-avg** — ``aggregate_bin_edges`` (plaintext weighted average).
3. **fed-weight-avg-smpc** — two-phase SMPC sum of local contributions.
4. **fed-hist** — ``aggregate_global_max_expr`` + ``aggregate_histogram_bin_edges_plain``.
5. **fed-hist-smpc** — secret-shared histogram sum + ``secure_quantile_cuts``.

No scGPT fine-tuning — isolates the binning aggregation step only.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--adata", type=str, required=True,
        help="Path to a prepared .h5ad file (e.g. data/scgpt/benchmark/ms/reference.h5ad).",
    )
    parser.add_argument(
        "--batch_key", type=str, required=True,
        help="Column in adata.obs used to partition cells into federated clients.",
    )
    parser.add_argument(
        "--n_bins", type=int, default=51,
        help="Number of bins B requested by scGPT preprocessing (default: 51 to match annotation/config.yml).",
    )
    parser.add_argument(
        "--grid_resolution", type=int, default=4096,
        help="Histogram grid resolution M (default: 4096). The grid has M+1 edges over [0, max_expr].",
    )
    parser.add_argument(
        "--layer", type=str, default=None,
        help="adata.layers[layer] to bin; None uses adata.X (matches Preprocessor.use_key='X').",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="output/demo_secure_binning",
        help="Directory to write summary.csv, summary.md and edges.npz.",
    )
    return parser.parse_args()


def _to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _quantile_probs(n_bins: int) -> np.ndarray:
    """Match Preprocessor.compute_local_bin_edges and histogram aggregation."""
    return np.linspace(0.0, 1.0, n_bins - 1)


def _edge_metrics(edges_a: np.ndarray, edges_b: np.ndarray) -> dict:
    diff = np.abs(edges_a - edges_b)
    return {
        "L1": float(diff.mean()),
        "Linf": float(diff.max()),
    }


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print(f"Reading AnnData from {args.adata} ...")
    adata = anndata.read_h5ad(args.adata)

    if args.layer is not None:
        if args.layer not in adata.layers:
            raise KeyError(
                f"Layer '{args.layer}' not in adata.layers (available: {list(adata.layers.keys())})."
            )
        layer_values = _to_dense(adata.layers[args.layer])
    else:
        layer_values = _to_dense(adata.X)
    if layer_values.min() < 0:
        raise ValueError(
            f"Expected non-negative data for binning, got min={layer_values.min()}."
        )

    if args.batch_key not in adata.obs.columns:
        raise KeyError(
            f"batch_key '{args.batch_key}' not in adata.obs (available: {list(adata.obs.columns)})."
        )
    batches = adata.obs[args.batch_key].astype(str).values
    unique_batches = sorted(set(batches))
    if len(unique_batches) < 2:
        raise ValueError(
            f"Need >= 2 partitions, got {len(unique_batches)} from batch_key '{args.batch_key}'."
        )

    probs = _quantile_probs(args.n_bins)

    pooled_nonzero = layer_values[layer_values > 0]
    if pooled_nonzero.size == 0:
        raise ValueError("All values are zero; cannot derive quantile bins.")
    edges_centralized = np.quantile(pooled_nonzero, probs).astype(np.float32)

    print(
        f"Partitioning into {len(unique_batches)} clients via obs[{args.batch_key}]; "
        f"pooled non-zero entries: {pooled_nonzero.size:,}."
    )

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
        local_edges = np.quantile(nz, probs).astype(np.float32)
        local_edges_pairs.append((local_edges, int(nz.size)))
        print(f"  - client '{b}': {sub.shape[0]:,} cells, {nz.size:,} non-zero values")

    # fed-weight-avg
    edges_weighted = aggregate_bin_edges(local_edges_pairs).astype(np.float32)

    # fed-weight-avg-smpc
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

    # fed-hist
    max_expr = aggregate_global_max_expr(client_max_list)
    value_grid = np.linspace(0.0, max_expr, args.grid_resolution + 1, dtype=np.float32)
    client_histograms = [
        np.histogram(nz, bins=value_grid)[0].astype(np.float64)
        for nz in per_client_nonzero
    ]
    edges_hist = aggregate_histogram_bin_edges_plain(
        client_histograms, client_n_list, value_grid, args.n_bins
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
        0.0, max_expr_smpc, args.grid_resolution + 1, dtype=np.float32
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
        hist_shares, n_shares_hist, value_grid_smpc, args.n_bins
    ).astype(np.float32)

    central_idx = _bin_indices(pooled_nonzero, edges_centralized)
    weighted_idx = _bin_indices(pooled_nonzero, edges_weighted)
    weighted_smpc_idx = _bin_indices(pooled_nonzero, edges_weighted_smpc)
    hist_idx = _bin_indices(pooled_nonzero, edges_hist)
    hist_smpc_idx = _bin_indices(pooled_nonzero, edges_hist_smpc)

    vs_central = {
        "fed-weight-avg": _edge_metrics(edges_weighted, edges_centralized),
        "fed-weight-avg-smpc": _edge_metrics(edges_weighted_smpc, edges_centralized),
        "fed-hist": _edge_metrics(edges_hist, edges_centralized),
        "fed-hist-smpc": _edge_metrics(edges_hist_smpc, edges_centralized),
    }
    vs_weighted = {
        "fed-weight-avg-smpc": _edge_metrics(edges_weighted_smpc, edges_weighted),
        "fed-hist": _edge_metrics(edges_hist, edges_weighted),
        "fed-hist-smpc": _edge_metrics(edges_hist_smpc, edges_weighted),
    }
    vs_hist = {
        "fed-hist-smpc": _edge_metrics(edges_hist_smpc, edges_hist),
    }

    metrics = {
        "n_batches": len(unique_batches),
        "n_cells": int(adata.n_obs),
        "n_nonzero_pooled": int(pooled_nonzero.size),
        "n_bins": args.n_bins,
        "grid_resolution": args.grid_resolution,
        "max_expr": float(max_expr),
        "L1_weighted_vs_centralized": vs_central["fed-weight-avg"]["L1"],
        "Linf_weighted_vs_centralized": vs_central["fed-weight-avg"]["Linf"],
        "L1_weighted_smpc_vs_centralized": vs_central["fed-weight-avg-smpc"]["L1"],
        "Linf_weighted_smpc_vs_centralized": vs_central["fed-weight-avg-smpc"]["Linf"],
        "L1_hist_vs_centralized": vs_central["fed-hist"]["L1"],
        "Linf_hist_vs_centralized": vs_central["fed-hist"]["Linf"],
        "L1_hist_smpc_vs_centralized": vs_central["fed-hist-smpc"]["L1"],
        "Linf_hist_smpc_vs_centralized": vs_central["fed-hist-smpc"]["Linf"],
        "L1_weighted_smpc_vs_weighted": vs_weighted["fed-weight-avg-smpc"]["L1"],
        "Linf_weighted_smpc_vs_weighted": vs_weighted["fed-weight-avg-smpc"]["Linf"],
        "L1_hist_vs_weighted": vs_weighted["fed-hist"]["L1"],
        "Linf_hist_vs_weighted": vs_weighted["fed-hist"]["Linf"],
        "L1_hist_smpc_vs_weighted": vs_weighted["fed-hist-smpc"]["L1"],
        "Linf_hist_smpc_vs_weighted": vs_weighted["fed-hist-smpc"]["Linf"],
        "L1_hist_smpc_vs_hist": vs_hist["fed-hist-smpc"]["L1"],
        "Linf_hist_smpc_vs_hist": vs_hist["fed-hist-smpc"]["Linf"],
        "agreement_weighted_vs_centralized": float((weighted_idx == central_idx).mean()),
        "agreement_weighted_smpc_vs_centralized": float((weighted_smpc_idx == central_idx).mean()),
        "agreement_hist_vs_centralized": float((hist_idx == central_idx).mean()),
        "agreement_hist_smpc_vs_centralized": float((hist_smpc_idx == central_idx).mean()),
        "nonzero_differing_weighted_smpc_vs_weighted": int((weighted_idx != weighted_smpc_idx).sum()),
        "nonzero_differing_hist_vs_weighted": int((weighted_idx != hist_idx).sum()),
        "nonzero_differing_hist_smpc_vs_hist": int((hist_idx != hist_smpc_idx).sum()),
        "nonzero_differing_hist_vs_centralized": int((hist_idx != central_idx).sum()),
        "max_expr_smpc": float(max_expr_smpc),
    }

    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    npz_path = output_dir / "edges.npz"

    with csv_path.open("w") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")

    with md_path.open("w") as f:
        f.write("# Federated binning demo (four federated strategies)\n\n")
        f.write(f"- AnnData: `{args.adata}`\n")
        f.write(
            f"- Partition: `obs[{args.batch_key}]` -> {metrics['n_batches']} clients\n"
        )
        f.write(
            f"- Cells: {metrics['n_cells']:,} "
            f"(non-zero entries pooled: {metrics['n_nonzero_pooled']:,})\n"
        )
        f.write(
            f"- Bins requested (B): {metrics['n_bins']} -> "
            f"{metrics['n_bins'] - 1} cut points\n"
        )
        f.write(f"- Histogram grid resolution M: {metrics['grid_resolution']}\n")
        f.write(f"- Global max_expr (plain): {metrics['max_expr']:.6g}\n")
        f.write(f"- Global max_expr (SMPC envelope): {metrics['max_expr_smpc']:.6g}\n\n")

        f.write("## Edge distance vs centralized ground truth\n\n")
        f.write("| prep_mode | L1 | L_inf |\n")
        f.write("|---|---|---|\n")
        f.write(
            f"| `fed-weight-avg` | {metrics['L1_weighted_vs_centralized']:.6g} "
            f"| {metrics['Linf_weighted_vs_centralized']:.6g} |\n"
        )
        f.write(
            f"| `fed-weight-avg-smpc` | {metrics['L1_weighted_smpc_vs_centralized']:.6g} "
            f"| {metrics['Linf_weighted_smpc_vs_centralized']:.6g} |\n"
        )
        f.write(
            f"| `fed-hist` | {metrics['L1_hist_vs_centralized']:.6g} "
            f"| {metrics['Linf_hist_vs_centralized']:.6g} |\n"
        )
        f.write(
            f"| `fed-hist-smpc` | {metrics['L1_hist_smpc_vs_centralized']:.6g} "
            f"| {metrics['Linf_hist_smpc_vs_centralized']:.6g} |\n"
        )

        f.write("\n## Pairwise vs fed-weight-avg\n\n")
        f.write("| prep_mode | L1 | L_inf |\n")
        f.write("|---|---|---|\n")
        f.write(
            f"| `fed-weight-avg-smpc` | {metrics['L1_weighted_smpc_vs_weighted']:.6g} "
            f"| {metrics['Linf_weighted_smpc_vs_weighted']:.6g} |\n"
        )
        f.write(
            f"| `fed-hist` | {metrics['L1_hist_vs_weighted']:.6g} "
            f"| {metrics['Linf_hist_vs_weighted']:.6g} |\n"
        )
        f.write(
            f"| `fed-hist-smpc` | {metrics['L1_hist_smpc_vs_weighted']:.6g} "
            f"| {metrics['Linf_hist_smpc_vs_weighted']:.6g} |\n"
        )

        f.write("\n## SMPC fidelity vs plaintext counterpart\n\n")
        f.write("| pair | L1 | L_inf | differing assignments |\n")
        f.write("|---|---|---|---|\n")
        f.write(
            f"| weighted SMPC vs weighted | {metrics['L1_weighted_smpc_vs_weighted']:.6g} "
            f"| {metrics['Linf_weighted_smpc_vs_weighted']:.6g} "
            f"| {metrics['nonzero_differing_weighted_smpc_vs_weighted']:,} |\n"
        )
        f.write(
            f"| hist SMPC vs hist | {metrics['L1_hist_smpc_vs_hist']:.6g} "
            f"| {metrics['Linf_hist_smpc_vs_hist']:.6g} "
            f"| {metrics['nonzero_differing_hist_smpc_vs_hist']:,} |\n"
        )

        f.write("\n## Bin-assignment agreement vs centralized\n\n")
        f.write("| prep_mode | agreement |\n")
        f.write("|---|---|\n")
        f.write(
            f"| `fed-weight-avg` | {metrics['agreement_weighted_vs_centralized']:.6g} |\n"
        )
        f.write(
            f"| `fed-weight-avg-smpc` | {metrics['agreement_weighted_smpc_vs_centralized']:.6g} |\n"
        )
        f.write(
            f"| `fed-hist` | {metrics['agreement_hist_vs_centralized']:.6g} |\n"
        )
        f.write(
            f"| `fed-hist-smpc` | {metrics['agreement_hist_smpc_vs_centralized']:.6g} |\n"
        )

        f.write("\n## Non-zero entries with differing bin assignment\n\n")
        f.write(
            f"- `fed-weight-avg-smpc` vs `fed-weight-avg`: "
            f"**{metrics['nonzero_differing_weighted_smpc_vs_weighted']:,}**\n"
        )
        f.write(
            f"- `fed-hist-smpc` vs `fed-hist`: "
            f"**{metrics['nonzero_differing_hist_smpc_vs_hist']:,}**\n"
        )
        f.write(
            f"- `fed-hist` vs `fed-weight-avg`: "
            f"**{metrics['nonzero_differing_hist_vs_weighted']:,}**\n"
        )
        f.write(
            f"- `fed-hist` vs centralized: "
            f"**{metrics['nonzero_differing_hist_vs_centralized']:,}**\n"
        )

    np.savez(
        npz_path,
        centralized=edges_centralized,
        fed_weight_avg=edges_weighted,
        fed_weight_avg_smpc=edges_weighted_smpc,
        fed_hist=edges_hist,
        fed_hist_smpc=edges_hist_smpc,
        value_grid=value_grid,
        value_grid_smpc=value_grid_smpc,
    )

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote: {csv_path}\nWrote: {md_path}\nWrote: {npz_path}")


if __name__ == "__main__":
    main()
