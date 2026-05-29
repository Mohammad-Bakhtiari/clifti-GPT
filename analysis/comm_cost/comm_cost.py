#!/usr/bin/env python
"""Communication-cost benchmark: GPU wall-clock for plain and SMPC (+ bytes in CSV).

Plaintext and SMPC timings both run on GPU (plain on ``cuda:0``, SMPC on
``cuda:0..P-1``). Default FT sweep: |θ| from ``models/init/hp5.pth`` (when
present) plus 1M and 10M. Default KNN sweep: n_q in {500,2000},
n_r in {1000,5000,10000}, C in {2,5}, k in {5,10}. See analysis/comm_cost/README.md.
"""

import argparse
import csv
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.comm_cost import COMM_COST_OUTPUT_DIR, REPO_ROOT  # noqa: E402

import numpy as np

FLOAT32_BYTES = 4
INT64_BYTES = 8
SHA256_HEX_BYTES = 64
SMPC_DEVICE = "cuda"
PLAIN_BENCH_DEVICE = "cuda:0"

COMM_COST_N_PARTIES_ENV = "COMM_COST_N_PARTIES"
COMM_COST_SMPC_ONE_GPU_PER_PARTY_ENV = "COMM_COST_SMPC_ONE_GPU_PER_PARTY"
COMM_COST_INIT_WEIGHTS_ENV = "COMM_COST_INIT_WEIGHTS"
_DEFAULT_N_PARTIES = 3
_DEFAULT_INIT_WEIGHTS = _REPO_ROOT / "models/init/hp5.pth"
_DEFAULT_FT_THETAS_SYNTHETIC = (1_000_000, 10_000_000)

RESULT_CSV_FIELDNAMES = [
    "workflow",
    "mode",
    "n_clients",
    "n_parties",
    "payload",
    "rounds",
    "t_seconds",
    "bytes_per_client_per_round",
    "bytes_per_client_total",
    "bytes_federation_total",
    "crypto_overhead",
    "notes",
]


class IncrementalResultsWriter:
    """Append benchmark rows to CSV after each configuration completes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=RESULT_CSV_FIELDNAMES).writeheader()
        self.row_count = 0

    def append(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_CSV_FIELDNAMES)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        self.row_count += len(rows)


def _commit_config_rows(
    rows: List[Dict[str, Any]],
    config_rows: List[Dict[str, Any]],
    writer: Optional[IncrementalResultsWriter],
) -> None:
    rows.extend(config_rows)
    if writer is not None:
        writer.append(config_rows)


def _log(msg: str) -> None:
    print(msg, flush=True)


def resolve_n_parties(cli_value: Optional[int] = None) -> int:
    if cli_value is not None:
        return cli_value
    env_val = os.environ.get(COMM_COST_N_PARTIES_ENV)
    if env_val is not None and str(env_val).strip():
        return int(env_val)
    return _DEFAULT_N_PARTIES


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SMPC wall-clock benchmarks.")


@dataclass
class CommCost:
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


def _plain_bench_device() -> "torch.device":
    import torch

    device = torch.device(PLAIN_BENCH_DEVICE)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _gpu_sync(device: "torch.device") -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_seconds(fn, n_reps: int) -> float:
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
    if n_parties < 2:
        return float(worker(*args, **kwargs))
    return _run_smpc_isolated_subprocess(n_parties, worker, args)


def _smpc_one_gpu_per_party_enabled() -> bool:
    return os.environ.get(COMM_COST_SMPC_ONE_GPU_PER_PARTY_ENV) == "1"


def _smpc_subprocess_cuda_visible_devices(
    n_parties: int,
    one_gpu_per_party: bool,
    env_cuda: str,
) -> str:
    if one_gpu_per_party:
        if env_cuda.strip():
            return env_cuda.strip()
        return ",".join(str(i) for i in range(n_parties))
    if env_cuda.strip():
        return env_cuda.strip()
    return "0"


def _run_smpc_isolated_subprocess(
    n_parties: int,
    worker,
    args: tuple,
) -> float:
    env = os.environ.copy()
    env["COMM_COST_ISOLATED_SMPC"] = "1"
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    one_gpu_per_party = _smpc_one_gpu_per_party_enabled()
    env["CUDA_VISIBLE_DEVICES"] = _smpc_subprocess_cuda_visible_devices(
        n_parties,
        one_gpu_per_party,
        env.get("CUDA_VISIBLE_DEVICES", ""),
    )
    if one_gpu_per_party:
        env[COMM_COST_SMPC_ONE_GPU_PER_PARTY_ENV] = "1"

    fd, payload_path = tempfile.mkstemp(suffix=".pkl", prefix="comm_cost_smpc_")
    os.close(fd)
    try:
        with open(payload_path, "wb") as f:
            pickle.dump(
                {"n_parties": n_parties, "worker": worker, "args": args}, f
            )
        env["COMM_COST_SMPC_PAYLOAD"] = payload_path
        script = str(Path(__file__).resolve())
        proc = subprocess.run(
            [sys.executable, script],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                "Isolated GPU SMPC subprocess failed "
                f"(exit {proc.returncode}). {err[-2000:]}"
            )
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError("Isolated GPU SMPC subprocess produced no output")
        return float(lines[-1])
    finally:
        try:
            os.unlink(payload_path)
        except OSError:
            pass


def _smpc_isolated_main() -> None:
    payload_path = os.environ.get("COMM_COST_SMPC_PAYLOAD")
    if not payload_path:
        sys.exit("COMM_COST_SMPC_PAYLOAD missing")
    with open(payload_path, "rb") as f:
        payload = pickle.load(f)

    n_parties = int(payload["n_parties"])
    worker = payload["worker"]
    args = tuple(payload["args"])

    from crypten.mpc import run_multiprocess

    @run_multiprocess(n_parties)
    def _launch():
        return worker(*args)

    results = _launch()
    if results is None:
        sys.exit(1)
    print(float(results[0]), flush=True)
    sys.exit(0)


def _benchmark_smpc_seed(seed: int) -> None:
    import random

    import crypten
    import torch
    from crypten.config import cfg

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg.encoder.precision_bits = 32
    cfg.debug.debug_mode = True
    crypten.manual_seed(seed, seed, seed)


def _smpc_device_for_party(n_parties: int) -> "torch.device":
    import torch
    import crypten.communicator as comm

    if _smpc_one_gpu_per_party_enabled() and torch.cuda.is_available():
        rank = comm.get().get_rank()
        n_visible = torch.cuda.device_count()
        if rank >= n_visible:
            raise RuntimeError(
                f"CrypTen party rank {rank} but only {n_visible} visible GPU(s). "
                f"For P={n_parties} with one GPU per party, set "
                f"CUDA_VISIBLE_DEVICES to {n_parties} devices "
                f"(e.g. 0,1,2) or pass --no-smpc-one-gpu-per-party."
            )
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def _init_smpc_worker(seed: int, n_parties: int) -> "torch.device":
    device = _smpc_device_for_party(n_parties)
    _benchmark_smpc_seed(seed)
    _assert_crypten_world_size(n_parties)
    return device


def _assert_crypten_world_size(n_parties: int) -> None:
    import crypten.communicator as comm

    actual = comm.get().get_world_size()
    if actual != n_parties:
        raise RuntimeError(
            f"CrypTen world_size={actual}, expected {n_parties}"
        )


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
) -> float:
    import crypten

    from cliftiGPT.federated.aggregator import FedAvg

    device = _init_smpc_worker(seed, n_parties)

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


def benchmark_fine_tuning(
    theta_values: List[int],
    n_clients_values: List[int],
    n_parties: int,
    n_rounds: int,
    n_reps: int,
    writer: Optional[IncrementalResultsWriter] = None,
) -> List[Dict[str, Any]]:
    import torch

    from cliftiGPT.federated.aggregator import FedAvg

    _ = torch
    plain_device = _plain_bench_device()

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

            def _run_ft_plain() -> None:
                agg_plain.aggregate_plain(local_states, n_samples)
                _gpu_sync(plain_device)

            _log(
                f"[FT] theta={actual_theta} C={n_clients} P={n_parties}: "
                f"timing GPU plaintext on {plain_device} ({n_reps} reps)..."
            )
            t_plain = _median_seconds(_run_ft_plain, n_reps)

            _log(
                f"[FT]   GPU plaintext median={t_plain*1e3:.1f}ms — "
                f"starting GPU SMPC ({n_parties} parties)..."
            )
            t_smpc = _median_smpc_seconds(
                n_parties,
                _time_ft_smpc_worker,
                theta,
                n_clients,
                n_parties,
                n_reps,
                42,
            )

            cost_plain = ft_weight_sharing_bytes(
                actual_theta, n_clients, n_parties, n_rounds, smpc=False
            )
            cost_smpc = ft_weight_sharing_bytes(
                actual_theta, n_clients, n_parties, n_rounds, smpc=True
            )

            overhead = (t_smpc / t_plain) if t_plain > 0 else float("nan")

            config_rows = [
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
                },
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
                },
            ]
            _commit_config_rows(rows, config_rows, writer)
            print(
                f"[FT] theta={actual_theta:>9d} C={n_clients} P={n_parties} "
                f"t_plain={t_plain*1e3:8.2f}ms t_smpc={t_smpc*1e3:8.2f}ms "
                f"overhead={overhead:5.1f}x",
                flush=True,
            )

    return rows


def _plain_knn_gpu(
    query: "torch.Tensor",
    references: List["torch.Tensor"],
    k: int,
    device: "torch.device",
) -> "torch.Tensor":
    """GPU plaintext KNN using the same distance/top-k structure as SMPC."""
    import torch

    client_topks: List[torch.Tensor] = []
    for ref in references:
        query_norm = query.square().sum(dim=1, keepdim=True)
        ref_norm = ref.square().sum(dim=1).unsqueeze(0)
        cross = query @ ref.transpose(0, 1)
        distances = query_norm + ref_norm - 2 * cross
        client_topks.append(torch.topk(distances, k, dim=1, largest=False).values)
    cat = torch.cat(client_topks, dim=1)
    topk_vals = torch.topk(cat, k, dim=1, largest=False).values
    _gpu_sync(device)
    return topk_vals


def _smpc_knn(
    query_enc,
    references,
    k: int,
):
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
) -> float:
    import crypten

    device = _init_smpc_worker(seed, n_parties)

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
    writer: Optional[IncrementalResultsWriter] = None,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(42)
    plain_device = _plain_bench_device()

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

                    query_gpu = _tensor_on(query, plain_device)
                    references_gpu = [
                        _tensor_on(ref, plain_device) for ref in references_np
                    ]
                    t_plain = _median_seconds(
                        lambda: _plain_knn_gpu(
                            query_gpu, references_gpu, k, plain_device
                        ),
                        n_reps,
                    )

                    payload = (
                        f"n_q={n_query},n_r={n_ref_total},d={d_embed},k={k}"
                    )
                    _log(
                        f"[KNN] {payload} C={n_clients} P={n_parties}: "
                        f"GPU plaintext on {plain_device}={t_plain*1e3:.1f}ms — "
                        f"starting GPU SMPC..."
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
                    config_rows = [
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
                        },
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
                        },
                    ]
                    _commit_config_rows(rows, config_rows, writer)
                    print(
                        f"[KNN] {payload} C={n_clients} P={n_parties} "
                        f"t_plain={t_plain*1e3:8.2f}ms "
                        f"t_smpc={t_smpc*1e3:8.2f}ms "
                        f"overhead={overhead:7.1f}x",
                        flush=True,
                    )

    return rows


def _time_binning_weighted_smpc_worker(
    n_clients: int,
    n_samples_per_client: int,
    n_bins: int,
    n_parties: int,
    n_reps: int,
    seed: int,
) -> float:
    import crypten

    from cliftiGPT.preprocessor.aggregation import (
        aggregate_bin_edge_contributions_smpc,
        local_bin_edge_contribution,
        reveal_nonzero_total,
    )

    device = _init_smpc_worker(seed, n_parties)

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
) -> float:
    import crypten

    from cliftiGPT.preprocessor.aggregation import (
        aggregate_global_max_expr,
        aggregate_secure_histogram_bin_edges,
        secure_reveal_envelope_max,
    )

    device = _init_smpc_worker(seed, n_parties)

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


def _aggregate_bin_edges_gpu(
    local_bin_edges_list: List[Tuple[np.ndarray, int]],
    device: "torch.device",
) -> np.ndarray:
    import torch

    from cliftiGPT.preprocessor.aggregation import _finalize_bin_edges

    total_samples = sum(samples for _, samples in local_bin_edges_list)
    n_bins = len(local_bin_edges_list[0][0])
    weighted = torch.zeros(n_bins, device=device, dtype=torch.float32)
    for bin_edges, num_samples in local_bin_edges_list:
        weighted += _tensor_on(bin_edges, device) * float(num_samples)
    weighted /= float(total_samples)
    result = _finalize_bin_edges(weighted.detach().cpu().numpy())
    _gpu_sync(device)
    return result


def _aggregate_histogram_bin_edges_gpu(
    client_histograms: List[np.ndarray],
    client_n_list: List[int],
    value_grid: np.ndarray,
    n_bins: int,
    device: "torch.device",
) -> np.ndarray:
    import torch

    from cliftiGPT.preprocessor.aggregation import _quantile_cuts_plain

    value_grid = np.asarray(value_grid, dtype=np.float32)
    m_bins = value_grid.size - 1
    hist = torch.zeros(m_bins, device=device, dtype=torch.float64)
    for client_hist in client_histograms:
        hist += _tensor_on(client_hist, device, dtype=torch.float64)
    total_n = float(sum(client_n_list))
    grid_upper_edges = value_grid[1:]
    probs = np.linspace(0.0, 1.0, n_bins - 1)
    result = _quantile_cuts_plain(
        hist.detach().cpu().numpy(), total_n, grid_upper_edges, probs
    )
    _gpu_sync(device)
    return result


def benchmark_binning(
    n_clients_values: List[int],
    n_parties: int,
    n_bins: int,
    hist_grid_resolution: int,
    n_samples_per_client: int,
    n_reps: int,
    writer: Optional[IncrementalResultsWriter] = None,
) -> List[Dict[str, Any]]:
    from cliftiGPT.preprocessor.aggregation import aggregate_global_max_expr

    rng = np.random.default_rng(42)
    plain_device = _plain_bench_device()

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
            lambda: _aggregate_bin_edges_gpu(local_edges_pairs, plain_device),
            n_reps,
        )

        _log(
            f"[BIN] C={n_clients} P={n_parties}: weighted GPU plaintext "
            f"on {plain_device}={t_weighted_plain*1e3:.1f}ms — "
            f"starting weighted SMPC..."
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
            lambda: _aggregate_histogram_bin_edges_gpu(
                client_histograms,
                client_n_list,
                value_grid,
                n_bins,
                plain_device,
            ),
            n_reps,
        )

        _log(
            f"[BIN] C={n_clients} P={n_parties}: hist GPU plaintext "
            f"on {plain_device}={t_hist_plain*1e3:.1f}ms — starting hist SMPC..."
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
        )

        config_rows: List[Dict[str, Any]] = []
        for strategy, t_val in [
            ("fed-weight-avg", t_weighted_plain),
            ("fed-weight-avg-smpc", t_weighted_smpc),
            ("fed-hist", t_hist_plain),
            ("fed-hist-smpc", t_hist_smpc),
        ]:
            cost = binning_bytes(
                strategy, n_clients, n_parties, n_bins, hist_grid_resolution
            )
            config_rows.append(
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
        _commit_config_rows(rows, config_rows, writer)
        print(
            f"[BIN] C={n_clients} P={n_parties} "
            f"weighted plain={t_weighted_plain*1e3:7.2f}ms "
            f"smpc={t_weighted_smpc*1e3:7.2f}ms "
            f"hist plain={t_hist_plain*1e3:7.2f}ms "
            f"smpc={t_hist_smpc*1e3:7.2f}ms",
            flush=True,
        )

    return rows


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
        "--ft_thetas", type=str, default=None,
        help=(
            "Comma-separated parameter counts for fine-tuning. "
            "If omitted, uses |θ| from models/init/hp5.pth when present, "
            "plus 1M and 10M synthetic sizes."
        ),
    )
    p.add_argument(
        "--ft_clients", type=str, default="2,3,5",
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
        "--knn_n_ref", type=str, default="1000,5000,10000",
        help=(
            "Comma-separated federation-wide reference counts n_r for KNN. "
            "Default stops at 10k (50k can OOM at n_q=2000 on ~15 GiB GPUs)."
        ),
    )
    p.add_argument(
        "--knn_clients", type=str, default="2,5",
        help="Comma-separated federated client counts C for the KNN benchmark.",
    )
    p.add_argument("--knn_d_embed", type=int, default=128)
    p.add_argument(
        "--knn_k", type=str, default="5,10",
        help=(
            "Comma-separated k values for KNN (matches tasks/args default k=10). "
            "k=20 is omitted from defaults (much slower; higher OOM risk)."
        ),
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
        "--quick",
        action="store_true",
        help="Minimal smoke-test sweep (small configs, n_reps=1).",
    )
    p.add_argument(
        "--smpc-one-gpu-per-party",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Map CrypTen party rank r to cuda:r using P visible GPUs (default: on). "
            "When CUDA_VISIBLE_DEVICES is unset, the isolated SMPC child uses "
            "0,1,...,P-1."
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


def _theta_from_checkpoint(path: Path) -> int:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    return int(sum(t.numel() for t in state.values()))


def resolve_ft_theta_values(cli_thetas: Optional[str]) -> List[int]:
    """Default FT sweep: checkpoint |θ| (if found) plus 1M and 10M."""
    if cli_thetas is not None:
        return _parse_thetas(cli_thetas)

    values: List[int] = []
    weights_path = Path(os.environ.get(COMM_COST_INIT_WEIGHTS_ENV, _DEFAULT_INIT_WEIGHTS))
    if not weights_path.is_absolute():
        weights_path = REPO_ROOT / weights_path
    if weights_path.is_file():
        theta = _theta_from_checkpoint(weights_path)
        values.append(theta)
        _log(f"FT sweep includes |θ|={theta:,} from {weights_path}")
    else:
        _log(
            f"FT sweep: init weights not found at {weights_path}; "
            f"using synthetic |θ| only ({', '.join(str(t) for t in _DEFAULT_FT_THETAS_SYNTHETIC)})"
        )
    for theta in _DEFAULT_FT_THETAS_SYNTHETIC:
        if theta not in values:
            values.append(theta)
    return values


def main() -> None:
    args = parse_args()
    if args.quick:
        _apply_quick_preset(args)
    args.n_parties = resolve_n_parties(args.n_parties)
    ft_theta_values = resolve_ft_theta_values(args.ft_thetas)
    args.ft_thetas = ",".join(str(theta) for theta in ft_theta_values)
    _require_cuda()
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

    if args.smpc_one_gpu_per_party:
        os.environ[COMM_COST_SMPC_ONE_GPU_PER_PARTY_ENV] = "1"
    else:
        os.environ.pop(COMM_COST_SMPC_ONE_GPU_PER_PARTY_ENV, None)

    parent_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if args.smpc_one_gpu_per_party:
        smpc_cuda = _smpc_subprocess_cuda_visible_devices(
            args.n_parties, True, parent_cuda
        )
        n_visible = len([x for x in smpc_cuda.split(",") if x.strip()])
        if n_visible < args.n_parties:
            _log(
                f"WARNING: --smpc-one-gpu-per-party needs >= {args.n_parties} "
                f"visible GPUs for P={args.n_parties}; SMPC child will use "
                f"CUDA_VISIBLE_DEVICES={smpc_cuda} ({n_visible} device(s))."
            )
    else:
        smpc_cuda = parent_cuda or "0"

    _log(
        f"Communication-cost benchmark — P={args.n_parties}, GPU, "
        f"workflows={sorted(workflows)}, n_reps={args.n_reps}, "
        f"smpc_one_gpu_per_party={args.smpc_one_gpu_per_party}, "
        f"SMPC CUDA_VISIBLE_DEVICES={smpc_cuda}, output={output_dir.resolve()}"
    )
    if args.quick:
        _log("(--quick preset)")

    results_csv = output_dir / "comm_cost_results.csv"
    results_writer = IncrementalResultsWriter(results_csv)

    if "fine_tuning" in workflows:
        benchmark_fine_tuning(
            theta_values=ft_theta_values,
            n_clients_values=_parse_int_list(args.ft_clients),
            n_parties=args.n_parties,
            n_rounds=args.ft_rounds,
            n_reps=args.n_reps,
            writer=results_writer,
        )

    if "reference_mapping" in workflows:
        benchmark_reference_mapping(
            n_query_values=_parse_int_list(args.knn_n_query),
            n_ref_values=_parse_int_list(args.knn_n_ref),
            n_clients_values=_parse_int_list(args.knn_clients),
            n_parties=args.n_parties,
            d_embed=args.knn_d_embed,
            k_values=_parse_int_list(args.knn_k),
            n_classes=args.knn_n_classes,
            n_reps=args.n_reps,
            writer=results_writer,
        )

    if "binning" in workflows:
        benchmark_binning(
            n_clients_values=_parse_int_list(args.binning_clients),
            n_parties=args.n_parties,
            n_bins=args.binning_n_bins,
            hist_grid_resolution=args.binning_grid_resolution,
            n_samples_per_client=args.binning_n_samples_per_client,
            n_reps=args.n_reps,
            writer=results_writer,
        )

    metadata = {
        "args": vars(args),
        "n_parties": args.n_parties,
        "plain_device": PLAIN_BENCH_DEVICE,
        "smpc_device": SMPC_DEVICE,
        "smpc_one_gpu_per_party": args.smpc_one_gpu_per_party,
        "smpc_cuda_visible_devices": smpc_cuda,
        "rows_written": results_writer.row_count,
    }
    (output_dir / "comm_cost_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print(f"\nWrote {results_csv} ({results_writer.row_count} rows)", flush=True)
    print(f"Wrote {output_dir / 'comm_cost_metadata.json'}", flush=True)


if __name__ == "__main__":
    if os.environ.get("COMM_COST_ISOLATED_SMPC") == "1":
        _smpc_isolated_main()
    else:
        main()
