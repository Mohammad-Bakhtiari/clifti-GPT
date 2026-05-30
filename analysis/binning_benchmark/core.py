"""Shared AnnData loading, client partition, and bin-edge aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata
import crypten
import numpy as np
import torch

from cliftiGPT.preprocessor.aggregation import (
    aggregate_bin_edge_contributions_smpc,
    aggregate_bin_edges,
    aggregate_global_max_expr,
    aggregate_histogram_bin_edges_plain,
    aggregate_secure_histogram_bin_edges,
    local_bin_edge_contribution,
    reveal_nonzero_total,
    secure_reveal_envelope_max,
)

from analysis.binning_benchmark.config import BINNING_STRATEGIES


def to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def quantile_probs(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins - 1)


def bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=True)


def load_layer_values(adata: anndata.AnnData, layer: Optional[str]) -> np.ndarray:
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(
                f"Layer '{layer}' not in adata.layers (available: {list(adata.layers.keys())})."
            )
        layer_values = to_dense(adata.layers[layer])
    else:
        layer_values = to_dense(adata.X)
    if layer_values.min() < 0:
        raise ValueError(
            f"Expected non-negative data for binning, got min={layer_values.min()}."
        )
    return layer_values


@dataclass
class ClientPartition:
    unique_batches: List[str]
    per_client_nonzero: List[np.ndarray]
    local_edges_pairs: List[Tuple[np.ndarray, int]]
    client_max_list: List[float]
    client_n_list: List[int]
    pooled_nonzero: np.ndarray
    n_cells: int


def partition_by_batch(
    adata: anndata.AnnData,
    batch_key: str,
    n_bins: int,
    layer: Optional[str] = None,
) -> ClientPartition:
    layer_values = load_layer_values(adata, layer)
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

    probs = quantile_probs(n_bins)
    pooled_nonzero = layer_values[layer_values > 0]
    if pooled_nonzero.size == 0:
        raise ValueError("All values are zero; cannot derive quantile bins.")

    local_edges_pairs: List[Tuple[np.ndarray, int]] = []
    per_client_nonzero: List[np.ndarray] = []
    client_max_list: List[float] = []
    client_n_list: List[int] = []
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

    return ClientPartition(
        unique_batches=unique_batches,
        per_client_nonzero=per_client_nonzero,
        local_edges_pairs=local_edges_pairs,
        client_max_list=client_max_list,
        client_n_list=client_n_list,
        pooled_nonzero=pooled_nonzero,
        n_cells=int(adata.n_obs),
    )


@dataclass
class BinningResult:
    strategy_edges: Dict[str, np.ndarray]
    value_grid: np.ndarray
    value_grid_smpc: np.ndarray
    max_expr: float
    max_expr_smpc: float
    local_edges_pairs: List[Tuple[np.ndarray, int]]
    client_histograms: List[np.ndarray]
    client_n_list: List[int]
    contrib_shares: List[Any]
    hist_shares: List[Any]
    n_shares_hist: List[Any]


def compute_all_strategy_edges(
    partition: ClientPartition,
    n_bins: int,
    grid_resolution: int,
) -> BinningResult:
    probs = quantile_probs(n_bins)
    local_edges_pairs = partition.local_edges_pairs

    edges_weighted = aggregate_bin_edges(local_edges_pairs).astype(np.float32)

    n_shares = [
        crypten.cryptensor(torch.tensor([float(n)], dtype=torch.float32))
        for _, n in local_edges_pairs
    ]
    total_n = reveal_nonzero_total(n_shares)
    contrib_shares = [
        crypten.cryptensor(
            torch.tensor(
                local_bin_edge_contribution(local_edges, n, total_n),
                dtype=torch.float32,
            )
        )
        for local_edges, n in local_edges_pairs
    ]
    edges_weighted_smpc = aggregate_bin_edge_contributions_smpc(contrib_shares).astype(
        np.float32
    )

    max_expr = aggregate_global_max_expr(partition.client_max_list)
    value_grid = np.linspace(0.0, max_expr, grid_resolution + 1, dtype=np.float32)
    client_histograms = [
        np.histogram(nz, bins=value_grid)[0].astype(np.float64)
        for nz in partition.per_client_nonzero
    ]
    edges_hist = aggregate_histogram_bin_edges_plain(
        client_histograms,
        partition.client_n_list,
        value_grid,
        n_bins,
    ).astype(np.float32)

    max_shares = [
        crypten.cryptensor(torch.tensor([float(m)], dtype=torch.float32))
        for m in partition.client_max_list
    ]
    n_shares_hist = [
        crypten.cryptensor(torch.tensor([float(n)], dtype=torch.float32))
        for n in partition.client_n_list
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
        for nz in partition.per_client_nonzero
    ]
    edges_hist_smpc = aggregate_secure_histogram_bin_edges(
        hist_shares, n_shares_hist, value_grid_smpc, n_bins
    ).astype(np.float32)

    edges_centralized = np.quantile(partition.pooled_nonzero, probs).astype(np.float32)

    return BinningResult(
        strategy_edges={
            "centralized": edges_centralized,
            "fed-weight-avg": edges_weighted,
            "fed-weight-avg-smpc": edges_weighted_smpc,
            "fed-hist": edges_hist,
            "fed-hist-smpc": edges_hist_smpc,
        },
        value_grid=value_grid,
        value_grid_smpc=value_grid_smpc,
        max_expr=float(max_expr),
        max_expr_smpc=float(max_expr_smpc),
        local_edges_pairs=local_edges_pairs,
        client_histograms=client_histograms,
        client_n_list=partition.client_n_list,
        contrib_shares=contrib_shares,
        hist_shares=hist_shares,
        n_shares_hist=n_shares_hist,
    )


def per_strategy_client_bins(
    partition: ClientPartition,
    strategy_edges: Dict[str, np.ndarray],
) -> Dict[str, List[np.ndarray]]:
    per_strategy_bins: Dict[str, List[np.ndarray]] = {}
    for strategy in BINNING_STRATEGIES:
        edges = strategy_edges[strategy]
        per_strategy_bins[strategy] = [
            bin_indices(nz, edges) for nz in partition.per_client_nonzero
        ]
    return per_strategy_bins


def edges_npz_payload(result: BinningResult) -> Dict[str, np.ndarray]:
    se = result.strategy_edges
    return {
        "centralized": se["centralized"],
        "fed_weight_avg": se["fed-weight-avg"],
        "fed_weight_avg_smpc": se["fed-weight-avg-smpc"],
        "fed_hist": se["fed-hist"],
        "fed_hist_smpc": se["fed-hist-smpc"],
        "value_grid": result.value_grid,
        "value_grid_smpc": result.value_grid_smpc,
    }


def load_partition_from_path(
    adata_path: Path,
    batch_key: str,
    n_bins: int,
    layer: Optional[str] = None,
) -> ClientPartition:
    adata = anndata.read_h5ad(adata_path)
    return partition_by_batch(adata, batch_key, n_bins, layer)
