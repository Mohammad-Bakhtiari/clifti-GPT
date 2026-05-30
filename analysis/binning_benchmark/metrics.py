"""Batch-effect metrics for federated binning strategies."""

from typing import Dict, List

import numpy as np


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
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
    if len(hist_list) < 2:
        return 0.0
    pairs = []
    for i in range(len(hist_list)):
        for j in range(i + 1, len(hist_list)):
            pairs.append(js_divergence(hist_list[i], hist_list[j]))
    return float(np.mean(pairs)) if pairs else 0.0


def heterogeneity_index(
    per_client_nonzero: List[np.ndarray],
    n_grid: int = 256,
) -> float:
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
