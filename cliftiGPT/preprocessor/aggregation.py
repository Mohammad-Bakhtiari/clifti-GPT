import numpy as np
import torch
import crypten
from typing import Dict, List, Tuple, Set, Optional
import scgpt as scg


def local_bin_edge_contribution(
    bin_edges: np.ndarray, n_nonzero: int, total_n: float
) -> np.ndarray:
    """Weighted contribution B_i * (n_i / N) computed on the client (plaintext)."""
    if total_n <= 0:
        raise ValueError("total_n must be positive.")
    return bin_edges * (float(n_nonzero) / total_n)


def reveal_nonzero_total(
    client_n_shares: List["crypten.CrypTensor"],
) -> float:
    """SMPC sum of non-zero counts; reveal global N = sum_i n_i only."""
    if len(client_n_shares) == 0:
        raise ValueError("client_n_shares must contain at least one entry.")
    shared_total = client_n_shares[0].clone().view(1)
    for n_share in client_n_shares[1:]:
        shared_total = shared_total + n_share.view(1)
    total = float(shared_total.get_plain_text().item())
    if total <= 0:
        raise ValueError("Aggregated non-zero count must be positive.")
    return total


def aggregate_bin_edge_contributions_smpc(
    client_contribution_shares: List["crypten.CrypTensor"],
) -> np.ndarray:
    """Sum secret-shared contribution vectors and finalize bin edges."""
    if len(client_contribution_shares) == 0:
        raise ValueError("client_contribution_shares must contain at least one entry.")
    n_bins = int(client_contribution_shares[0].size()[0])
    if any(int(c.size()[0]) != n_bins for c in client_contribution_shares):
        raise ValueError("All contribution shares must have the same length.")
    shared_sum = client_contribution_shares[0]
    for contrib in client_contribution_shares[1:]:
        shared_sum = shared_sum + contrib
    weighted_bin_edges = shared_sum.get_plain_text().cpu().numpy().astype(np.float32)
    return _finalize_bin_edges(weighted_bin_edges)


def aggregate_gene_counts(filter_gene_by_counts, local_gene_counts_list: List[Dict[str, int]],
                          logger: scg.logger = None) -> np.ndarray:
    all_gene_names = list(local_gene_counts_list[0].keys())
    combined_gene_counts = np.zeros(len(all_gene_names))

    for local_counts in local_gene_counts_list:
        for i, gene in enumerate(all_gene_names):
            combined_gene_counts[i] += local_counts[gene]

    global_gene_mask = combined_gene_counts >= filter_gene_by_counts
    s = np.sum(~global_gene_mask)
    if s > 0:
        msg = f"Filtered out {s} genes that are detected in less than {self.filter_gene_by_counts} counts"
        if logger:
            logger.info(msg)
        else:
            print(msg)

    return global_gene_mask


def aggregate_hvg_stats(local_stats_list: List[Dict]) -> Dict:
    all_means = np.stack([stats['means'] for stats in local_stats_list])
    all_variances = np.stack([stats['variances'] for stats in local_stats_list])
    all_variances_norm = np.stack([stats['variances_norm'] for stats in local_stats_list])

    global_means = np.mean(all_means, axis=0)
    global_variances = np.mean(all_variances, axis=0)
    global_variances_norm = np.mean(all_variances_norm, axis=0)

    return {
        'means': global_means,
        'variances': global_variances,
        'variances_norm': global_variances_norm
    }


def _finalize_bin_edges(weighted_bin_edges: np.ndarray) -> np.ndarray:
    """Re-quantile the weighted-average cut vector (shared by plain and SMPC paths)."""
    n_bins = len(weighted_bin_edges)
    return np.quantile(weighted_bin_edges, np.linspace(0, 1, n_bins))


def aggregate_bin_edges(local_bin_edges_list: List[Tuple[np.ndarray, int]]) -> np.ndarray:
    """Aggregate local quantile bin edges into global edges (prep_mode=fed-weight-avg).

       """
    total_samples = sum([samples for _, samples in local_bin_edges_list])
    n_bins = len(local_bin_edges_list[0][0])

    if any(len(bin_edges) != n_bins for bin_edges, _ in local_bin_edges_list):
        raise ValueError("All local bin edge lists must have the same number of bins.")

    weighted_bin_edges = np.zeros(n_bins)

    for bin_edges, num_samples in local_bin_edges_list:
        weighted_bin_edges += bin_edges * num_samples

    weighted_bin_edges /= total_samples

    return _finalize_bin_edges(weighted_bin_edges)


def aggregate_bin_edges_smpc(
    client_n_shares: List["crypten.CrypTensor"],
    client_contribution_shares: List["crypten.CrypTensor"],
) -> np.ndarray:
    """SMPC weighted-average of local bin edges (prep_mode=fed-weight-avg-smpc).

    Two-phase protocol (avoids secret B_i * n_i multiply in CrypTen):

    1. ``client_n_shares``: ``cryptensor(n_i)`` per client. The coordinator
       reveals ``N = sum_i n_i`` (global count only; per-client counts stay secret).
    2. Each client forms ``B_i * (n_i / N)`` locally, then sends
       ``client_contribution_shares``.
    3. This function sums contribution shares and applies ``_finalize_bin_edges``.
    """
    if len(client_n_shares) != len(client_contribution_shares):
        raise ValueError("client_n_shares and client_contribution_shares must have the same length.")
    total_n = reveal_nonzero_total(client_n_shares)
    return aggregate_bin_edge_contributions_smpc(client_contribution_shares)


def _quantile_cuts_plain(
    histogram: np.ndarray,
    total_n: float,
    grid_upper_edges: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Read quantile cuts from a pooled histogram CDF (plaintext fed-hist)."""
    H = np.asarray(histogram, dtype=np.float64)
    grid_upper_edges = np.asarray(grid_upper_edges, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float64)
    if H.ndim != 1 or H.size == 0:
        raise ValueError(f"histogram must be a non-empty 1D array, got shape {H.shape}.")
    if grid_upper_edges.ndim != 1 or grid_upper_edges.size != H.size:
        raise ValueError(
            f"grid_upper_edges must have length {H.size}, got {grid_upper_edges.shape}."
        )
    if total_n <= 0:
        raise ValueError("total_n must be positive.")
    cdf = np.cumsum(H)
    cuts = np.empty(probs.size, dtype=np.float32)
    for j, p in enumerate(probs):
        target = max(float(p) * total_n, 1.0)
        idx = int(np.searchsorted(cdf, target, side="left"))
        if idx >= H.size:
            idx = H.size - 1
        cuts[j] = grid_upper_edges[idx]
    return cuts


def aggregate_global_max_expr(
    client_max_list: List[float],
    min_max_expr: float = 1e-6,
) -> float:
    """Plaintext global max expression across clients (fed-hist Round 1)."""
    if len(client_max_list) == 0:
        raise ValueError("client_max_list must contain at least one entry.")
    max_val = float(max(client_max_list))
    return float(max(max_val, min_max_expr))


def aggregate_histogram_bin_edges_plain(
    client_histograms: List[np.ndarray],
    client_n_list: List[int],
    value_grid: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Plaintext histogram-based global bin edges
    """        
    value_grid = np.asarray(value_grid, dtype=np.float32)  
    M = value_grid.size - 1
    hist = np.zeros(M, dtype=np.float64)
    for client_hist in client_histograms:
        client_hist = np.asarray(client_hist, dtype=np.float64)
        if client_hist.shape != (M,):
            raise ValueError(
                f"Each client histogram must have shape ({M},), got {client_hist.shape}."
            )
        hist += client_hist

    total_n = float(sum(client_n_list))
    hist_total = float(hist.sum())
    if hist_total != total_n:
        raise ValueError(
            f"Histogram count sum ({hist_total}) must match aggregated n ({total_n})."
        )

    grid_upper_edges = value_grid[1:]
    probs = np.linspace(0.0, 1.0, n_bins - 1)
    return _quantile_cuts_plain(hist, total_n, grid_upper_edges, probs)


def secure_reveal_envelope_max(
    client_max_shares: List["crypten.CrypTensor"],
    envelope_padding: float = 1.0,
    min_envelope: float = 1e-6,
) -> float:
    """
    Securely compute and reveal the global maximum non-zero value across clients.

    The returned scalar is the only value disclosed at this stage of the
    secure-histogram binning protocol; it is needed to define a public
    envelope grid shared by all clients before secret-shared histograms
    can be aggregated. No per-client maxima are revealed.

    Parameters
    ----------
    client_max_shares : list of crypten.CrypTensor
        Per-client secret-shared local maxima (each of shape ``(1,)`` or
        0-dim), as produced by ``ClientAnnotator.get_local_envelope_stats``.
    envelope_padding : float, optional
        Multiplicative padding applied to the revealed maximum. Defaults to
        ``1.0`` (no padding) so the SMPC envelope grid matches the plaintext
        grid produced by ``aggregate_global_max_expr`` bit-for-bit. Values
        greater than ``1.0`` are still accepted for advanced use (e.g. to
        absorb fixed-point rounding margin in adversarial settings) but are
        no longer the default.
    min_envelope : float, optional
        Lower bound for the returned envelope max to keep the grid valid
        when all clients have only zeros. Defaults to ``1e-6``.

    Returns
    -------
    float
        ``max_global`` used as the upper end of the public envelope grid
        (optionally padded by ``envelope_padding``).
    """
    if len(client_max_shares) == 0:
        raise ValueError("client_max_shares must contain at least one entry.")
    stacked = crypten.cat([m.view(1) for m in client_max_shares], dim=0)
    max_val = float(stacked.max().get_plain_text().item())
    return float(max(max_val * envelope_padding, min_envelope))


def aggregate_secure_histogram_bin_edges(
    client_histogram_shares: List["crypten.CrypTensor"],
    client_n_shares: List["crypten.CrypTensor"],
    envelope_grid: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """
    Securely derive global bin edges from per-client secret-shared histograms.

    Phase 1 (SMPC): sum secret-shared per-client histograms and non-zero counts.
    Phase 2 (plaintext): read ``n_bins - 1`` quantile cuts from the aggregated
    histogram with ``_quantile_cuts_plain``, matching ``fed-hist``. Individual
    client histograms are not revealed; only the pooled histogram (equivalent
    to plaintext ``fed-hist`` aggregation) and the final bin edges are used
    downstream.

    A fully encrypted quantile-cut loop (``secure_quantile_cuts``) is not used
    here because CrypTen's mixed public/secret reductions are unreliable for
    this pattern in our environment.

    Parameters
    ----------
    client_histogram_shares : list of crypten.CrypTensor
        Per-client secret-shared histograms of shape ``(M,)`` over the same
        ``envelope_grid``, as produced by ``ClientAnnotator.get_local_histogram``.
    client_n_shares : list of crypten.CrypTensor
        Per-client secret-shared non-zero counts of shape ``(1,)`` or 0-dim,
        as produced by ``ClientAnnotator.get_local_envelope_stats``.
    envelope_grid : np.ndarray
        Public strictly increasing envelope grid of length ``M + 1`` shared
        across clients (typically ``np.linspace(0, max_global, M + 1)``).
    n_bins : int
        Number of bins requested by the downstream scGPT preprocessing
        (matching ``preprocess.n_bins`` in the YAML configs). The function
        returns ``n_bins - 1`` cut values, matching ``np.quantile(...,
        np.linspace(0, 1, n_bins - 1))``.

    Returns
    -------
    np.ndarray
        Plaintext global bin edges of length ``n_bins - 1`` ready to be
        passed to ``Preprocessor.apply_binning``.
    """
    if len(client_histogram_shares) != len(client_n_shares):
        raise ValueError(
            "client_histogram_shares and client_n_shares must have the same length."
        )
    if len(client_histogram_shares) == 0:
        raise ValueError("At least one client must contribute to aggregation.")

    shared_H = client_histogram_shares[0].clone()
    for h in client_histogram_shares[1:]:
        shared_H = shared_H + h
    shared_total = client_n_shares[0].clone().view(1)
    for n in client_n_shares[1:]:
        shared_total = shared_total + n.view(1)

    envelope_grid = np.asarray(envelope_grid, dtype=np.float32)
    if envelope_grid.ndim != 1 or envelope_grid.size < 2:
        raise ValueError(
            f"envelope_grid must be a 1D array of length >= 2, got shape {envelope_grid.shape}."
        )
    if not np.all(np.diff(envelope_grid) > 0):
        raise ValueError("envelope_grid must be strictly increasing.")

    H_agg = (
        shared_H.get_plain_text().detach().cpu().numpy().astype(np.float64)
    )
    N_agg = float(shared_total.get_plain_text().detach().cpu().item())
    hist_total = float(H_agg.sum())
    if hist_total <= 0:
        raise ValueError("Aggregated secure histogram is empty.")
    if abs(hist_total - N_agg) > max(1.0, 1e-6 * N_agg):
        raise ValueError(
            f"Aggregated histogram count sum ({hist_total}) "
            f"does not match aggregated N ({N_agg})."
        )

    probs = np.linspace(0.0, 1.0, n_bins - 1)
    edges = _quantile_cuts_plain(H_agg, N_agg, envelope_grid[1:], probs)
    if edges.size != n_bins - 1:
        raise ValueError(
            f"Expected {n_bins - 1} secure histogram bin edges, got {edges.size}."
        )
    if not np.all(np.isfinite(edges)):
        raise ValueError("Secure histogram bin edges contain non-finite values.")
    if not np.any(edges > 0):
        raise ValueError("Secure histogram bin edges are degenerate (all zero).")
    return edges


def aggregate_local_gene_sets(local_gene_sets: List[Set[str]]) -> Dict[int, str]:
    combined_gene_set = set()
    for gene_set in local_gene_sets:
        combined_gene_set.update(gene_set)
    return dict(enumerate(combined_gene_set))


def aggregate_local_celltype_sets(local_celltype_sets: List[Set[str]]) -> Dict[int, str]:
    combined_celltype_set = set()
    for celltype_set in local_celltype_sets:
        combined_celltype_set.update(celltype_set)
    return dict(enumerate(combined_celltype_set))


