#!/usr/bin/env python
"""Communication-cost benchmark for federated workflows.

Addresses reviewer R3 ("wall-clock time, bandwidth usage, cryptographic
overhead, runtime scaling") with two artifacts:

1. **Analytical bandwidth model**. Closed-form byte counts per workflow
   derived from tensor shapes and the CrypTen additive-sharing convention.
   No network instrumentation is required because the bytes are determined
   by the protocol, not by transport latency.

2. **Simulated wall-clock benchmark**. Runs ``FedAvg.aggregate_plain`` vs
   ``aggregate_smpc``, federated KNN (plaintext FAISS vs SMPC
   ``top_k_encrypted_distances``), and federated binning aggregation
   (plain vs SMPC) under CrypTen's single-process simulated MPC. Records
   median wall-clock time; the SMPC times exclude any real network I/O
   (none happens) but include the encryption/decryption and CrypTen
   protocol overhead that would dominate cryptographic cost in a real
   deployment.

Outputs:
    output/comm_cost/comm_cost_results.csv
    output/comm_cost/comm_cost_metadata.json
    output/comm_cost/comm_cost_*.png
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.comm_cost.paths import COMM_COST_OUTPUT_DIR, REPO_ROOT  # noqa: E402

import numpy as np

# torch / crypten are imported lazily inside the wall-clock benchmark
# functions so the analytical formulas remain usable without the heavy
# scientific stack.

FLOAT32_BYTES = 4
INT64_BYTES = 8
SHA256_HEX_BYTES = 64

# Env var read by ``resolve_n_parties()`` when ``--n_parties`` is omitted.
COMM_COST_N_PARTIES_ENV = "COMM_COST_N_PARTIES"
COMM_COST_GPU_IDS_ENV = "COMM_COST_GPU_IDS"
_DEFAULT_N_PARTIES = 3


def _log(msg: str) -> None:
    print(msg, flush=True)


def resolve_n_parties(cli_value: Optional[int] = None) -> int:
    """Resolve SMPC party count P: CLI ``--n_parties`` > env > default."""
    if cli_value is not None:
        return cli_value
    env_val = os.environ.get(COMM_COST_N_PARTIES_ENV)
    if env_val is not None and str(env_val).strip():
        return int(env_val)
    return _DEFAULT_N_PARTIES

# ---------------------------------------------------------------------------
# Analytical bandwidth model
# ---------------------------------------------------------------------------


@dataclass
class CommCost:
    """Single (workflow, mode, config) bandwidth record.

    ``n_clients`` (``C``) is the number of federated data-holding clients;
    ``n_parties`` (``P``) is the number of SMPC computational parties.
    The two are *independent*: plaintext federation costs scale with ``C``,
    while the additive-sharing share factor is ``(P-1)`` regardless of
    ``C``. Bytes counts are *per workflow*, summing all rounds; ``rounds``
    is the number of communication rounds.
    """

    workflow: str
    mode: str
    n_clients: int
    n_parties: int
    rounds: int
    payload_label: str
    payload_value: int
    bytes_up_per_client: int
    bytes_dn_per_client: int
    bytes_total_per_client: int
    bytes_total_federation: int
    notes: str = ""

    def per_client_per_round(self) -> float:
        return self.bytes_total_per_client / max(1, self.rounds)


def _check_pos(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def ft_weight_sharing_bytes(
    theta: int,
    n_clients: int,
    n_parties: int,
    n_rounds: int,
    smpc: bool,
) -> CommCost:
    """Bandwidth for one fine-tuning experiment.

    Plaintext FedAvg/FedProx (matches ``cliftiGPT/federated/aggregator.py``
    ``aggregate_plain``):

      - Per client per round: upload ``4 * theta`` bytes (state_dict);
        download ``4 * theta`` bytes (global state_dict broadcast).
      - The number of SMPC parties ``P`` does not enter the plaintext
        path; per-client bytes are independent of both ``C`` and ``P``.

    SMPC FedAvg (matches ``aggregate_smpc`` together with the
    ``client.get_local_updates`` SMPC branch in
    ``cliftiGPT/federated/client.py``):

      - Each client converts every parameter tensor via
        ``crypten.cryptensor(...)``. CrypTen's additive-sharing protocol
        with ``P`` parties sends one share of size ``4 * theta`` to every
        other party (a "secret-shared upload"). With ``P`` parties the
        per-client upload becomes ``(P - 1) * 4 * theta`` bytes,
        independent of the number of federated clients ``C``.
      - The aggregator reveals the encrypted global weights via
        ``get_plain_text``. The repo (``FedAnnotator.local_update``,
        ``annotator.py:139-147``) broadcasts the *plaintext* global
        weights back to clients, so each client receives the
        ``4 * theta`` plaintext download.
      - Federation totals still scale with ``C`` because each of the
        ``C`` clients emits its own share set.
    """
    _check_pos("theta", theta)
    _check_pos("n_clients", n_clients)
    _check_pos("n_parties", n_parties)
    _check_pos("n_rounds", n_rounds)
    share_factor = max(1, n_parties - 1)
    if smpc:
        bytes_up_per_round = share_factor * FLOAT32_BYTES * theta
    else:
        bytes_up_per_round = FLOAT32_BYTES * theta
    bytes_dn_per_round = FLOAT32_BYTES * theta
    bytes_up = bytes_up_per_round * n_rounds
    bytes_dn = bytes_dn_per_round * n_rounds
    bytes_total_per_client = bytes_up + bytes_dn
    bytes_total_federation = bytes_total_per_client * n_clients
    return CommCost(
        workflow="fine_tuning",
        mode="smpc" if smpc else "plain",
        n_clients=n_clients,
        n_parties=n_parties,
        rounds=n_rounds,
        payload_label="theta",
        payload_value=theta,
        bytes_up_per_client=bytes_up,
        bytes_dn_per_client=bytes_dn,
        bytes_total_per_client=bytes_total_per_client,
        bytes_total_federation=bytes_total_federation,
        notes=(
            f"SMPC upload uses CrypTen additive sharing with P={n_parties} "
            "parties (P-1 shares per tensor); global broadcast stays "
            "plaintext in this repo."
            if smpc
            else "Plaintext FedAvg/FedProx upload+broadcast."
        ),
    )


def knn_reference_mapping_bytes(
    n_query: int,
    n_ref_total: int,
    n_clients: int,
    n_parties: int,
    d_embed: int,
    k: int,
    n_classes: int,
    smpc: bool,
) -> CommCost:
    """Bandwidth for a single federated reference-mapping pass.

    Reference cells are assumed evenly distributed across clients
    (``n_ref_local = n_ref_total / n_clients``). ``n_parties`` is the
    SMPC computational party count and is independent of ``n_clients``;
    the additive-sharing share factor ``(P-1)`` multiplies every SMPC
    secret-shared tensor.

      - Plaintext (``federated_reference_map`` non-SMPC branch): each
        client sends ``(n_query, k)`` float32 distances plus
        ``(n_query, k)`` SHA-256 hex hashes
        (``embedder.compute_local_distances`` → ``hash_indices``).
        Coordinator broadcasts the global top-k hashes back per
        ``global_aggregate_distances``.

      - SMPC: query embeddings encrypted at the coordinator (one share
        per other party); each client encrypts its local reference
        matrix and global-index offset (``(P-1)`` shares each), runs
        ``top_k_encrypted_distances`` (k iterations of ``min`` +
        ``suppress_argmin`` over the full ``(n_query, n_ref_local)``
        matrix), and ships ``(n_query, k)`` encrypted distances +
        encrypted global indices to the coordinator. The
        Beaver-triple online traffic for the masked-min loop in
        ``top_k_ind_selection`` scales as
        ``(P-1) * 2 * 4 * n_query * n_ref_local * k`` bytes per client.

    Bandwidth is computed for a **single pass** (no rounds).
    """
    _check_pos("n_query", n_query)
    _check_pos("n_ref_total", n_ref_total)
    _check_pos("n_clients", n_clients)
    _check_pos("n_parties", n_parties)
    _check_pos("d_embed", d_embed)
    _check_pos("k", k)
    _check_pos("n_classes", n_classes)
    n_ref_local = max(1, n_ref_total // n_clients)
    if smpc:
        share_factor = max(1, n_parties - 1)
        bytes_query_share = share_factor * FLOAT32_BYTES * n_query * d_embed
        bytes_ref_share = share_factor * FLOAT32_BYTES * n_ref_local * d_embed
        bytes_offset_share = share_factor * INT64_BYTES * n_ref_local
        bytes_topk_loop = (
            share_factor * 2 * FLOAT32_BYTES * n_query * n_ref_local * k
        )
        bytes_client_topk_out = (
            FLOAT32_BYTES * n_query * k + INT64_BYTES * n_query * k
        )
        bytes_up_per_client = (
            bytes_ref_share
            + bytes_offset_share
            + bytes_topk_loop
            + bytes_client_topk_out
        )
        bytes_dn_per_client = (
            bytes_query_share
            + (FLOAT32_BYTES * n_query * k)
            + (INT64_BYTES * n_query * k)
            + (FLOAT32_BYTES * n_query * n_classes)
        )
        notes = (
            f"SMPC with P={n_parties} parties; Beaver-triple online cost "
            "(P-1)*2*4*n_q*n_r_local*k for top-k loop; query/ref/offset "
            "shares charged once per pass."
        )
    else:
        bytes_topk_out = n_query * k * (FLOAT32_BYTES + SHA256_HEX_BYTES)
        bytes_up_per_client = bytes_topk_out
        bytes_dn_per_client = (
            n_query * k * SHA256_HEX_BYTES
            + n_query * d_embed * FLOAT32_BYTES
        )
        notes = (
            "Plaintext: top-k float32 distances + 64-byte SHA-256 hex hashes; "
            "coordinator broadcasts global k-NN hashes back."
        )
    bytes_total_per_client = bytes_up_per_client + bytes_dn_per_client
    bytes_total_federation = bytes_total_per_client * n_clients
    return CommCost(
        workflow="reference_mapping",
        mode="smpc" if smpc else "plain",
        n_clients=n_clients,
        n_parties=n_parties,
        rounds=1,
        payload_label=f"n_q={n_query},n_r={n_ref_total},d={d_embed},k={k}",
        payload_value=n_query * n_ref_local * d_embed,
        bytes_up_per_client=bytes_up_per_client,
        bytes_dn_per_client=bytes_dn_per_client,
        bytes_total_per_client=bytes_total_per_client,
        bytes_total_federation=bytes_total_federation,
        notes=notes,
    )


def binning_bytes(
    strategy: str,
    n_clients: int,
    n_parties: int,
    n_bins: int = 51,
    hist_grid_resolution: int = 4096,
) -> CommCost:
    """Bandwidth for the one-shot federated binning aggregation.

    Plain modes (``fed-weight-avg`` / ``fed-hist``) send per-client
    summary statistics in clear. SMPC modes secret-share the same
    statistics; CrypTen sharing multiplies the per-client byte count by
    ``(P - 1)`` where ``P`` is the SMPC party count (independent of
    ``C``). Federation totals still scale with ``C``.

    See ``cliftiGPT/preprocessor/aggregation.py`` for the exact tensor
    shapes that the protocol consumes.
    """
    _check_pos("n_clients", n_clients)
    _check_pos("n_parties", n_parties)
    _check_pos("n_bins", n_bins)
    _check_pos("hist_grid_resolution", hist_grid_resolution)
    share_factor = max(1, n_parties - 1)
    if strategy == "fed-weight-avg":
        per_client = FLOAT32_BYTES * (n_bins - 1) + FLOAT32_BYTES  # B_i and n_i
        notes = "Plaintext B_i (n_bins-1 floats) + n_i (scalar)."
    elif strategy == "fed-weight-avg-smpc":
        per_client = share_factor * (
            FLOAT32_BYTES * (n_bins - 1) + FLOAT32_BYTES
        )
        notes = (
            f"Secret-shared B_i*(n_i/N) contribution + n_i scalar "
            f"with P={n_parties} parties."
        )
    elif strategy == "fed-hist":
        per_client = (
            FLOAT32_BYTES * hist_grid_resolution
            + FLOAT32_BYTES  # max
            + FLOAT32_BYTES  # n
        )
        notes = "Plaintext histogram (M=grid_resolution) + max + n."
    elif strategy == "fed-hist-smpc":
        per_client = share_factor * (
            FLOAT32_BYTES * hist_grid_resolution
            + FLOAT32_BYTES  # max
            + FLOAT32_BYTES  # n
        )
        notes = f"Secret-shared histogram + max + n with P={n_parties} parties."
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    bytes_total_per_client = per_client
    bytes_total_federation = per_client * n_clients
    return CommCost(
        workflow="binning",
        mode=strategy,
        n_clients=n_clients,
        n_parties=n_parties,
        rounds=1,
        payload_label=f"n_bins={n_bins},M={hist_grid_resolution}",
        payload_value=hist_grid_resolution if "hist" in strategy else n_bins,
        bytes_up_per_client=per_client,
        bytes_dn_per_client=FLOAT32_BYTES * (n_bins - 1),
        bytes_total_per_client=bytes_total_per_client
        + FLOAT32_BYTES * (n_bins - 1),
        bytes_total_federation=(per_client + FLOAT32_BYTES * (n_bins - 1))
        * n_clients,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Wall-clock benchmark utilities
# ---------------------------------------------------------------------------


def _median_seconds(fn, n_reps: int) -> float:
    """Return median wall-clock seconds of ``n_reps`` calls to ``fn``.

    A warm-up call is performed first and excluded so CrypTen lazy-init
    or PyTorch kernel-compile costs do not skew the timing.
    """
    fn()  # warm-up
    samples: List[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _median_smpc_seconds(
    n_parties: int,
    worker,
    *args,
    **kwargs,
) -> float:
    """Time an SMPC kernel under CrypTen with world size ``n_parties``.

    For ``n_parties >= 2`` this uses ``crypten.mpc.run_multiprocess`` so
    each timed call runs with the requested party count. The parent process
    must not call ``crypten.init()`` or create CUDA tensors beforehand.
    """
    if n_parties < 2:
        return float(worker(*args, **kwargs))

    device_name = str(args[-1]) if args else "cpu"
    _warn_cuda_multiprocess(_smpc_use_cuda(device_name))

    from crypten.mpc import run_multiprocess

    @run_multiprocess(n_parties)
    def _launch():
        return worker(*args, **kwargs)

    results = _launch()
    if results is None:
        raise RuntimeError(
            f"CrypTen multiprocess benchmark failed for P={n_parties}. "
            "For GPU SMPC, export CUDA_VISIBLE_DEVICES=0 (one GPU shared by "
            "all parties) or use --device cpu."
        )
    return float(results[0])


def _smpc_use_cuda(device_name: str) -> bool:
    return device_name == "cuda" or (
        device_name == "auto" and _cuda_available_lightweight()
    )


def _cuda_available_lightweight() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _party_gpu_ids(n_parties: int) -> List[int]:
    """Physical GPU indices for CrypTen parties (used in docs / logging only)."""
    raw = os.environ.get(COMM_COST_GPU_IDS_ENV, "").strip()
    if raw:
        ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if ids:
            return ids
    return list(range(n_parties))


def _warn_cuda_multiprocess(smpc_use_cuda: bool) -> None:
    """Log GPU env requirements; CrypTen uses fork, so parent must stay off CUDA."""
    if not smpc_use_cuda:
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        _log(
            "GPU SMPC: export CUDA_VISIBLE_DEVICES=0 before starting Python "
            "(one GPU shared by all parties). Do not set multiple GPUs."
        )


def _assert_crypten_world_size(n_parties: int) -> None:
    import crypten.communicator as comm

    actual = comm.get().get_world_size()
    if actual != n_parties:
        raise RuntimeError(
            f"CrypTen world_size={actual}, expected {n_parties}"
        )


def resolve_device(device_arg: Optional[str] = None) -> "torch.device":
    """Resolve wall-clock device: auto → cuda when available, else cpu."""
    import torch

    choice = (device_arg or "auto").lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {device_arg} requested but torch.cuda.is_available() is False"
        )
    return device


def _init_smpc_worker(
    device_name: str, seed: int, n_parties: int
) -> "torch.device":
    """Select party-local device, seed RNGs, and verify CrypTen world size."""
    import torch

    from cliftiGPT.utils import set_seed

    requested = torch.device(device_name)
    if requested.type == "cuda":
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    set_seed(seed)
    _assert_crypten_world_size(n_parties)
    return device


def _tensor_on(data: Any, device: "torch.device", dtype: Any = None) -> "torch.Tensor":
    import torch

    dtype = dtype or torch.float32
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    return torch.tensor(data, dtype=dtype, device=device)


def _make_state_dict(
    theta: int,
    device: Optional["torch.device"] = None,
    n_layers: int = 4,
) -> Dict[str, "torch.Tensor"]:
    """Build a plain state_dict whose total parameter count is ``theta``.

    Used only to feed ``FedAvg.aggregate_plain`` / ``aggregate_smpc`` with
    realistic shapes without instantiating the full scGPT model.
    """
    import torch

    if device is None:
        device = torch.device("cpu")
    per_layer = max(1, theta // n_layers)
    side = max(1, int(math.sqrt(per_layer)))
    state: Dict[str, torch.Tensor] = {}
    remaining = theta
    for li in range(n_layers):
        if li == n_layers - 1:
            shape = (max(1, remaining // side), side)
        else:
            shape = (side, side)
        t = torch.randn(*shape, dtype=torch.float32, device=device) * 0.01
        state[f"layer{li}.weight"] = t
        remaining -= int(np.prod(shape))
        if remaining <= 0:
            break
    if remaining > 0:
        state[f"bias"] = torch.randn(remaining, dtype=torch.float32, device=device) * 0.01
    return state


def _total_params(state: Dict[str, "torch.Tensor"]) -> int:
    return int(sum(t.numel() for t in state.values()))


def _time_ft_smpc_worker(
    theta: int,
    n_clients: int,
    n_parties: int,
    n_reps: int,
    seed: int,
    device_name: str,
) -> float:
    """Run fine-tuning SMPC aggregation inside a CrypTen process group."""
    import crypten

    from cliftiGPT.federated.aggregator import FedAvg

    device = _init_smpc_worker(device_name, seed, n_parties)

    state = _make_state_dict(theta, device)
    agg_smpc = FedAvg(
        weighted=False,
        n_rounds=1,
        smpc=True,
        debug=False,
    )
    agg_smpc.global_model_keys = list(state.keys())
    agg_smpc.global_weight_shapes = {k: v.shape for k, v in state.items()}
    encrypted_clients = [
        [crypten.cryptensor(v) for v in state.values()]
        for _ in range(n_clients)
    ]
    return _median_seconds(
        lambda: agg_smpc.aggregate_smpc(encrypted_clients),
        n_reps,
    )


# ---------------------------------------------------------------------------
# Fine-tuning wall-clock benchmark
# ---------------------------------------------------------------------------


def benchmark_fine_tuning(
    theta_values: List[int],
    n_clients_values: List[int],
    n_parties: int,
    n_rounds: int,
    n_reps: int,
    device_name: str,
) -> List[Dict[str, Any]]:
    """Time ``FedAvg.aggregate_plain`` and ``aggregate_smpc``.

    Plaintext aggregation runs in the parent process. SMPC aggregation is
    timed under CrypTen with ``n_parties`` processes via
    ``crypten.mpc.run_multiprocess``.
    """

    import torch

    from cliftiGPT.federated.aggregator import FedAvg

    _ = torch  # silence linter; torch is used via state-dict tensors below

    smpc_use_cuda = _smpc_use_cuda(device_name)
    plain_device = torch.device("cpu")

    rows: List[Dict[str, Any]] = []

    for theta in theta_values:
        state = _make_state_dict(theta, plain_device)
        actual_theta = _total_params(state)
        for n_clients in n_clients_values:
            local_states = [
                {k: v.clone() for k, v in state.items()} for _ in range(n_clients)
            ]
            n_samples = [1000 for _ in range(n_clients)]

            agg_plain = FedAvg(
                weighted=True,
                n_rounds=1,
                smpc=False,
                debug=False,
            )
            agg_plain.global_model_keys = list(state.keys())
            agg_plain.global_weight_shapes = {k: v.shape for k, v in state.items()}

            _log(
                f"[FT] theta={actual_theta} C={n_clients} P={n_parties}: "
                f"timing plaintext ({n_reps} reps)..."
            )
            t_plain = _median_seconds(
                lambda: agg_plain.aggregate_plain(local_states, n_samples),
                n_reps,
            )

            _log(
                f"[FT]   plaintext median={t_plain*1e3:.1f}ms — "
                f"starting SMPC on {device_name} (spawns {n_parties} CrypTen "
                f"processes)..."
            )
            if smpc_use_cuda and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                _log(
                    "  Hint: export CUDA_VISIBLE_DEVICES=0 before running GPU SMPC "
                    "(all parties share one GPU; avoids CrypTen init errors)."
                )
            t_smpc = _median_smpc_seconds(
                n_parties,
                _time_ft_smpc_worker,
                theta,
                n_clients,
                n_parties,
                n_reps,
                42,
                device_name,
            )

            cost_plain = ft_weight_sharing_bytes(
                actual_theta, n_clients, n_parties, n_rounds, smpc=False
            )
            cost_smpc = ft_weight_sharing_bytes(
                actual_theta, n_clients, n_parties, n_rounds, smpc=True
            )

            overhead = (t_smpc / t_plain) if t_plain > 0 else float("nan")

            rows.append(
                {
                    "workflow": "fine_tuning",
                    "mode": "plain",
                    "n_clients": n_clients,
                    "n_parties": n_parties,
                    "payload": f"theta={actual_theta}",
                    "rounds": n_rounds,
                    "t_seconds": t_plain,
                    "bytes_per_client_per_round": cost_plain.per_client_per_round(),
                    "bytes_per_client_total": cost_plain.bytes_total_per_client,
                    "bytes_federation_total": cost_plain.bytes_total_federation,
                    "crypto_overhead": 1.0,
                    "notes": cost_plain.notes,
                }
            )
            rows.append(
                {
                    "workflow": "fine_tuning",
                    "mode": "smpc",
                    "n_clients": n_clients,
                    "n_parties": n_parties,
                    "payload": f"theta={actual_theta}",
                    "rounds": n_rounds,
                    "t_seconds": t_smpc,
                    "bytes_per_client_per_round": cost_smpc.per_client_per_round(),
                    "bytes_per_client_total": cost_smpc.bytes_total_per_client,
                    "bytes_federation_total": cost_smpc.bytes_total_federation,
                    "crypto_overhead": overhead,
                    "notes": cost_smpc.notes,
                }
            )
            print(
                f"[FT] theta={actual_theta:>9d} C={n_clients} P={n_parties} "
                f"t_plain={t_plain*1e3:8.2f}ms t_smpc={t_smpc*1e3:8.2f}ms "
                f"overhead={overhead:5.1f}x",
                flush=True,
            )

    return rows


# ---------------------------------------------------------------------------
# Reference-mapping (KNN) wall-clock benchmark
# ---------------------------------------------------------------------------


def _plain_knn(
    query: np.ndarray, references: List[np.ndarray], k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the plaintext KNN path of ``ClientEmbedder``.

    Mirrors ``compute_local_distances`` (FAISS branch) + the coordinator
    concatenation in ``global_aggregate_distances`` non-SMPC branch, but
    without the hash bookkeeping; we only need the timing.
    """
    import faiss

    n_q = query.shape[0]
    all_dists: List[np.ndarray] = []
    all_inds: List[np.ndarray] = []
    for ri, ref in enumerate(references):
        index = faiss.IndexFlatL2(ref.shape[1])
        index.add(ref.astype(np.float32))
        D, I = index.search(query.astype(np.float32), k)
        all_dists.append(D)
        all_inds.append(I + ri * 10**8)  # cheap unique offset
    all_dists_arr = np.hstack(all_dists)
    all_inds_arr = np.hstack(all_inds)
    sorted_indices = np.argsort(all_dists_arr, axis=1)[:, :k]
    rows = np.arange(n_q)[:, None]
    return (
        all_dists_arr[rows, sorted_indices],
        all_inds_arr[rows, sorted_indices],
    )


def _smpc_knn(
    query_enc,
    references,
    k: int,
):
    """Reproduce the SMPC KNN path of ``ClientEmbedder.compute_squared_distances``
    plus the global ``secure_top_k_distance_agg`` step.

    Returns the encrypted global top-k distances; the caller can decide to
    reveal them outside the timed region.
    """
    import crypten

    from cliftiGPT.utils import (
        top_k_encrypted_distances,
        top_k_ind_selection,
    )

    client_topks = []
    for ref in references:
        ref_enc = crypten.cryptensor(ref)
        query_norm = query_enc.square().sum(dim=1).unsqueeze(1)
        ref_norm = ref_enc.square().sum(dim=1).unsqueeze(0)
        cross = query_enc @ ref_enc.transpose(0, 1)
        distances = query_norm + ref_norm - 2 * cross
        encrypted_topk, _ = top_k_encrypted_distances(distances, k)
        client_topks.append(encrypted_topk)

    cat = crypten.cat(client_topks, dim=1)
    one_hot = top_k_ind_selection(cat.clone(), k)
    return [(one_hot[i] * cat).sum(dim=1) for i in range(k)]


def _time_knn_smpc_worker(
    n_query: int,
    n_ref_local: int,
    n_clients: int,
    d_embed: int,
    k: int,
    n_parties: int,
    n_reps: int,
    seed: int,
    device_name: str,
) -> float:
    """Run federated KNN SMPC inside a CrypTen process group."""
    import crypten

    device = _init_smpc_worker(device_name, seed, n_parties)

    rng = np.random.default_rng(seed)
    query = rng.standard_normal((n_query, d_embed)).astype(np.float32)
    references_np = [
        rng.standard_normal((n_ref_local, d_embed)).astype(np.float32)
        for _ in range(n_clients)
    ]
    references_torch = [_tensor_on(r, device) for r in references_np]
    query_enc = crypten.cryptensor(_tensor_on(query, device))
    return _median_seconds(
        lambda: _smpc_knn(query_enc, references_torch, k),
        n_reps,
    )


def benchmark_reference_mapping(
    n_query_values: List[int],
    n_ref_values: List[int],
    n_clients_values: List[int],
    n_parties: int,
    d_embed: int,
    k_values: List[int],
    n_classes: int,
    n_reps: int,
    device_name: str,
) -> List[Dict[str, Any]]:
    """Time plaintext FAISS KNN vs the CrypTen SMPC distance + top-k pipeline.

    SMPC timings use ``crypten.mpc.run_multiprocess(n_parties)`` so wall-clock
    matches the analytical party count.
    """

    rng = np.random.default_rng(42)

    rows: List[Dict[str, Any]] = []

    for n_query in n_query_values:
        for n_ref_total in n_ref_values:
            for n_clients in n_clients_values:
                n_ref_local = max(1, n_ref_total // n_clients)
                for k in k_values:
                    query = rng.standard_normal((n_query, d_embed)).astype(
                        np.float32
                    )
                    references_np = [
                        rng.standard_normal((n_ref_local, d_embed)).astype(
                            np.float32
                        )
                        for _ in range(n_clients)
                    ]

                    t_plain = _median_seconds(
                        lambda: _plain_knn(query, references_np, k),
                        n_reps,
                    )

                    payload = (
                        f"n_q={n_query},n_r={n_ref_total},d={d_embed},k={k}"
                    )
                    _log(
                        f"[KNN] {payload} C={n_clients} P={n_parties}: "
                        f"plaintext={t_plain*1e3:.1f}ms — starting SMPC on "
                        f"{device_name}..."
                    )

                    t_smpc = _median_smpc_seconds(
                        n_parties,
                        _time_knn_smpc_worker,
                        n_query,
                        n_ref_local,
                        n_clients,
                        d_embed,
                        k,
                        n_parties,
                        n_reps,
                        42,
                        device_name,
                    )

                    cost_plain = knn_reference_mapping_bytes(
                        n_query,
                        n_ref_total,
                        n_clients,
                        n_parties,
                        d_embed,
                        k,
                        n_classes,
                        smpc=False,
                    )
                    cost_smpc = knn_reference_mapping_bytes(
                        n_query,
                        n_ref_total,
                        n_clients,
                        n_parties,
                        d_embed,
                        k,
                        n_classes,
                        smpc=True,
                    )
                    overhead = (t_smpc / t_plain) if t_plain > 0 else float("nan")
                    rows.append(
                        {
                            "workflow": "reference_mapping",
                            "mode": "plain",
                            "n_clients": n_clients,
                            "n_parties": n_parties,
                            "payload": payload,
                            "rounds": 1,
                            "t_seconds": t_plain,
                            "bytes_per_client_per_round": cost_plain.bytes_total_per_client,
                            "bytes_per_client_total": cost_plain.bytes_total_per_client,
                            "bytes_federation_total": cost_plain.bytes_total_federation,
                            "crypto_overhead": 1.0,
                            "notes": cost_plain.notes,
                        }
                    )
                    rows.append(
                        {
                            "workflow": "reference_mapping",
                            "mode": "smpc",
                            "n_clients": n_clients,
                            "n_parties": n_parties,
                            "payload": payload,
                            "rounds": 1,
                            "t_seconds": t_smpc,
                            "bytes_per_client_per_round": cost_smpc.bytes_total_per_client,
                            "bytes_per_client_total": cost_smpc.bytes_total_per_client,
                            "bytes_federation_total": cost_smpc.bytes_total_federation,
                            "crypto_overhead": overhead,
                            "notes": cost_smpc.notes,
                        }
                    )
                    print(
                        f"[KNN] {payload} C={n_clients} P={n_parties} "
                        f"t_plain={t_plain*1e3:8.2f}ms "
                        f"t_smpc={t_smpc*1e3:8.2f}ms "
                        f"overhead={overhead:7.1f}x",
                        flush=True,
                    )

    return rows


# ---------------------------------------------------------------------------
# Binning wall-clock benchmark
# ---------------------------------------------------------------------------


def _time_binning_weighted_smpc_worker(
    n_clients: int,
    n_samples_per_client: int,
    n_bins: int,
    n_parties: int,
    n_reps: int,
    seed: int,
    device_name: str,
) -> float:
    """Time fed-weight-avg SMPC aggregation inside a CrypTen process group."""
    import crypten

    from cliftiGPT.preprocessor.aggregation import (
        aggregate_bin_edge_contributions_smpc,
        local_bin_edge_contribution,
        reveal_nonzero_total,
    )

    device = _init_smpc_worker(device_name, seed, n_parties)

    rng = np.random.default_rng(seed)
    per_client_nonzero = [
        np.abs(rng.standard_normal(n_samples_per_client)).astype(np.float32)
        for _ in range(n_clients)
    ]
    probs = np.linspace(0.0, 1.0, n_bins - 1)
    local_edges_pairs = [
        (np.quantile(c, probs).astype(np.float32), int(c.size))
        for c in per_client_nonzero
    ]
    n_shares = [
        crypten.cryptensor(_tensor_on([float(n)], device))
        for _, n in local_edges_pairs
    ]
    total_n = reveal_nonzero_total(n_shares)
    contrib_shares = [
        crypten.cryptensor(
            _tensor_on(local_bin_edge_contribution(le, n, total_n), device)
        )
        for le, n in local_edges_pairs
    ]
    return _median_seconds(
        lambda: aggregate_bin_edge_contributions_smpc(contrib_shares),
        n_reps,
    )


def _time_binning_hist_smpc_worker(
    n_clients: int,
    n_samples_per_client: int,
    n_bins: int,
    hist_grid_resolution: int,
    n_parties: int,
    n_reps: int,
    seed: int,
    device_name: str,
) -> float:
    """Time fed-hist SMPC aggregation inside a CrypTen process group."""
    import crypten

    from cliftiGPT.preprocessor.aggregation import (
        aggregate_global_max_expr,
        aggregate_secure_histogram_bin_edges,
        secure_reveal_envelope_max,
    )

    device = _init_smpc_worker(device_name, seed, n_parties)

    rng = np.random.default_rng(seed)
    per_client_nonzero = [
        np.abs(rng.standard_normal(n_samples_per_client)).astype(np.float32)
        for _ in range(n_clients)
    ]
    client_max_list = [float(c.max()) for c in per_client_nonzero]
    client_n_list = [int(c.size) for c in per_client_nonzero]
    max_shares = [
        crypten.cryptensor(_tensor_on([float(m)], device))
        for m in client_max_list
    ]
    n_shares_hist = [
        crypten.cryptensor(_tensor_on([float(n)], device))
        for n in client_n_list
    ]
    max_expr_smpc = secure_reveal_envelope_max(max_shares)
    value_grid_smpc = np.linspace(
        0.0, max_expr_smpc, hist_grid_resolution + 1, dtype=np.float32
    )
    hist_shares = [
        crypten.cryptensor(
            _tensor_on(np.histogram(nz, bins=value_grid_smpc)[0].astype(np.float32), device)
        )
        for nz in per_client_nonzero
    ]
    return _median_seconds(
        lambda: aggregate_secure_histogram_bin_edges(
            hist_shares, n_shares_hist, value_grid_smpc, n_bins
        ),
        n_reps,
    )


def benchmark_binning(
    n_clients_values: List[int],
    n_parties: int,
    n_bins: int,
    hist_grid_resolution: int,
    n_samples_per_client: int,
    n_reps: int,
    device_name: str,
) -> List[Dict[str, Any]]:
    """Time fed-weight-avg / fed-hist plain and SMPC aggregation paths.

    SMPC timings use ``crypten.mpc.run_multiprocess(n_parties)``.
    """

    from cliftiGPT.preprocessor.aggregation import (
        aggregate_bin_edges,
        aggregate_global_max_expr,
        aggregate_histogram_bin_edges_plain,
    )

    rng = np.random.default_rng(42)

    rows: List[Dict[str, Any]] = []

    for n_clients in n_clients_values:
        per_client_nonzero = [
            np.abs(rng.standard_normal(n_samples_per_client)).astype(np.float32)
            for _ in range(n_clients)
        ]
        client_max_list = [float(c.max()) for c in per_client_nonzero]
        client_n_list = [int(c.size) for c in per_client_nonzero]
        probs = np.linspace(0.0, 1.0, n_bins - 1)
        local_edges_pairs = [
            (np.quantile(c, probs).astype(np.float32), int(c.size))
            for c in per_client_nonzero
        ]

        t_weighted_plain = _median_seconds(
            lambda: aggregate_bin_edges(local_edges_pairs),
            n_reps,
        )

        _log(
            f"[BIN] C={n_clients} P={n_parties}: weighted plaintext "
            f"={t_weighted_plain*1e3:.1f}ms — starting weighted SMPC..."
        )
        t_weighted_smpc = _median_smpc_seconds(
            n_parties,
            _time_binning_weighted_smpc_worker,
            n_clients,
            n_samples_per_client,
            n_bins,
            n_parties,
            n_reps,
            42,
            device_name,
        )

        max_expr = aggregate_global_max_expr(client_max_list)
        value_grid = np.linspace(
            0.0, max_expr, hist_grid_resolution + 1, dtype=np.float32
        )
        client_histograms = [
            np.histogram(nz, bins=value_grid)[0].astype(np.float64)
            for nz in per_client_nonzero
        ]
        t_hist_plain = _median_seconds(
            lambda: aggregate_histogram_bin_edges_plain(
                client_histograms, client_n_list, value_grid, n_bins
            ),
            n_reps,
        )

        _log(
            f"[BIN] C={n_clients} P={n_parties}: hist plaintext "
            f"={t_hist_plain*1e3:.1f}ms — starting hist SMPC..."
        )
        t_hist_smpc = _median_smpc_seconds(
            n_parties,
            _time_binning_hist_smpc_worker,
            n_clients,
            n_samples_per_client,
            n_bins,
            hist_grid_resolution,
            n_parties,
            n_reps,
            42,
            device_name,
        )

        for strategy, t_val in [
            ("fed-weight-avg", t_weighted_plain),
            ("fed-weight-avg-smpc", t_weighted_smpc),
            ("fed-hist", t_hist_plain),
            ("fed-hist-smpc", t_hist_smpc),
        ]:
            cost = binning_bytes(
                strategy, n_clients, n_parties, n_bins, hist_grid_resolution
            )
            rows.append(
                {
                    "workflow": "binning",
                    "mode": strategy,
                    "n_clients": n_clients,
                    "n_parties": n_parties,
                    "payload": f"n_bins={n_bins},M={hist_grid_resolution}",
                    "rounds": 1,
                    "t_seconds": t_val,
                    "bytes_per_client_per_round": cost.bytes_total_per_client,
                    "bytes_per_client_total": cost.bytes_total_per_client,
                    "bytes_federation_total": cost.bytes_total_federation,
                    "crypto_overhead": (
                        (t_weighted_smpc / t_weighted_plain)
                        if strategy == "fed-weight-avg-smpc" and t_weighted_plain > 0
                        else (
                            (t_hist_smpc / t_hist_plain)
                            if strategy == "fed-hist-smpc" and t_hist_plain > 0
                            else 1.0
                        )
                    ),
                    "notes": cost.notes,
                }
            )
        print(
            f"[BIN] C={n_clients} P={n_parties} "
            f"weighted plain={t_weighted_plain*1e3:7.2f}ms "
            f"smpc={t_weighted_smpc*1e3:7.2f}ms "
            f"hist plain={t_hist_plain*1e3:7.2f}ms "
            f"smpc={t_hist_smpc*1e3:7.2f}ms",
            flush=True,
        )

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output_dir", type=str, default=COMM_COST_OUTPUT_DIR,
        help="Output directory for results.csv and metadata.json.",
    )
    p.add_argument(
        "--workflows", type=str, default="fine_tuning,reference_mapping,binning",
        help="Comma-separated subset of {fine_tuning,reference_mapping,binning}.",
    )
    p.add_argument(
        "--n_reps", type=int, default=5,
        help="Number of timed repetitions per configuration (median reported).",
    )
    p.add_argument(
        "--n_parties",
        type=int,
        default=None,
        help=(
            "Number of SMPC computational parties P. Fixed across all "
            "workflows; share factor in the analytical model is (P-1). "
            "Independent of the number of federated clients C. "
            f"If omitted, reads ${COMM_COST_N_PARTIES_ENV} (default "
            f"{_DEFAULT_N_PARTIES})."
        ),
    )
    p.add_argument(
        "--ft_thetas", type=str, default="1_000_000,10_000_000,50_000_000",
        help="Comma-separated parameter counts for fine-tuning benchmark.",
    )
    p.add_argument(
        "--ft_clients", type=str, default="2,3,5,10",
        help="Comma-separated federated client counts C for fine-tuning.",
    )
    p.add_argument(
        "--ft_rounds", type=int, default=5,
        help="Number of rounds used in the analytical FT bandwidth formula.",
    )
    p.add_argument(
        "--knn_n_query", type=str, default="500,2000",
        help="Comma-separated query counts for the KNN benchmark.",
    )
    p.add_argument(
        "--knn_n_ref", type=str, default="1000,5000",
        help="Comma-separated total reference counts for the KNN benchmark.",
    )
    p.add_argument(
        "--knn_clients", type=str, default="2,5",
        help="Comma-separated federated client counts C for the KNN benchmark.",
    )
    p.add_argument("--knn_d_embed", type=int, default=128)
    p.add_argument(
        "--knn_k", type=str, default="5,20",
        help="Comma-separated k values for the KNN benchmark.",
    )
    p.add_argument("--knn_n_classes", type=int, default=10)
    p.add_argument(
        "--binning_clients", type=str, default="2,5,10",
        help="Comma-separated federated client counts C for the binning benchmark.",
    )
    p.add_argument("--binning_n_bins", type=int, default=51)
    p.add_argument("--binning_grid_resolution", type=int, default=4096)
    p.add_argument(
        "--binning_n_samples_per_client", type=int, default=100_000,
        help="Synthetic non-zero values per client for the binning benchmark.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help=(
            "PyTorch device for wall-clock benchmarks. auto=cuda when available. "
            "Under cuda, CrypTen party rank r uses cuda:r %% num_gpus."
        ),
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Minimal sweep for smoke testing: small |θ|, one client count "
            "per workflow, n_reps=1. Analytical bytes still use full formulas; "
            "wall-clock is only indicative."
        ),
    )
    return p.parse_args()


def _apply_quick_preset(args: argparse.Namespace) -> None:
    args.n_reps = 1
    args.ft_thetas = "10_000"
    args.ft_clients = "2"
    args.knn_n_query = "500"
    args.knn_n_ref = "1000"
    args.knn_clients = "2"
    args.knn_k = "5"
    args.binning_clients = "2"
    args.binning_n_samples_per_client = 1_000


def _parse_thetas(s: str) -> List[int]:
    return [int(x.replace("_", "")) for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    if args.quick:
        _apply_quick_preset(args)
    args.n_parties = resolve_n_parties(args.n_parties)
    device = resolve_device(args.device)
    device_name = str(device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    workflows = {w.strip() for w in args.workflows.split(",") if w.strip()}
    valid = {"fine_tuning", "reference_mapping", "binning"}
    unknown = workflows - valid
    if unknown:
        raise ValueError(f"Unknown workflow(s): {unknown}. Choose from {valid}.")

    if args.n_parties < 2:
        raise ValueError(
            f"--n_parties must be >= 2 for additive sharing; got {args.n_parties}"
        )

    _log(
        f"Communication-cost benchmark — P={args.n_parties}, "
        f"device={device_name}, workflows={sorted(workflows)}, "
        f"n_reps={args.n_reps}, output={output_dir.resolve()}"
    )
    if device.type == "cuda":
        gpu_ids = _party_gpu_ids(args.n_parties)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
        _log(
            f"CUDA SMPC: recommended export CUDA_VISIBLE_DEVICES=0 "
            f"(parties share one GPU). Optional {COMM_COST_GPU_IDS_ENV}="
            f"{','.join(str(i) for i in gpu_ids)}. "
            f"Current CUDA_VISIBLE_DEVICES={visible}"
        )
    if args.quick:
        _log("(--quick preset: small configs; full sweep omit --quick)")

    all_rows: List[Dict[str, Any]] = []

    if "fine_tuning" in workflows:
        all_rows.extend(
            benchmark_fine_tuning(
                theta_values=_parse_thetas(args.ft_thetas),
                n_clients_values=_parse_int_list(args.ft_clients),
                n_parties=args.n_parties,
                n_rounds=args.ft_rounds,
                n_reps=args.n_reps,
                device_name=device_name,
            )
        )

    if "reference_mapping" in workflows:
        all_rows.extend(
            benchmark_reference_mapping(
                n_query_values=_parse_int_list(args.knn_n_query),
                n_ref_values=_parse_int_list(args.knn_n_ref),
                n_clients_values=_parse_int_list(args.knn_clients),
                n_parties=args.n_parties,
                d_embed=args.knn_d_embed,
                k_values=_parse_int_list(args.knn_k),
                n_classes=args.knn_n_classes,
                n_reps=args.n_reps,
                device_name=device_name,
            )
        )

    if "binning" in workflows:
        all_rows.extend(
            benchmark_binning(
                n_clients_values=_parse_int_list(args.binning_clients),
                n_parties=args.n_parties,
                n_bins=args.binning_n_bins,
                hist_grid_resolution=args.binning_grid_resolution,
                n_samples_per_client=args.binning_n_samples_per_client,
                n_reps=args.n_reps,
                device_name=device_name,
            )
        )

    results_csv = output_dir / "comm_cost_results.csv"
    fieldnames = [
        "workflow", "mode", "n_clients", "n_parties", "payload", "rounds",
        "t_seconds", "bytes_per_client_per_round",
        "bytes_per_client_total", "bytes_federation_total",
        "crypto_overhead", "notes",
    ]
    with results_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    metadata = {
        "args": vars(args),
        "n_parties": args.n_parties,
        "caveats": [
            f"Analytical bandwidth uses a fixed SMPC party count "
            f"P={args.n_parties}; share factor is (P-1). The number of "
            "federated clients C is independent and sweeps per workflow.",
            f"SMPC wall-clock timings are measured under CrypTen "
            f"run_multiprocess(P={args.n_parties}); each timed kernel "
            "spawns P synchronized processes on a single host.",
            "CrypTen wall-clock includes encryption/decryption and "
            "protocol cost but excludes real inter-site network latency.",
            "Bandwidth is derived analytically from CrypTen's additive-"
            "sharing convention (one share per other party).",
            "Beaver-triple precomputation is assumed offline (CrypTen "
            "default) and not included in the SMPC wall-clock.",
        ],
    }
    (output_dir / "comm_cost_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print(f"\nWrote {results_csv} ({len(all_rows)} rows)", flush=True)
    print(f"Wrote {output_dir / 'comm_cost_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
