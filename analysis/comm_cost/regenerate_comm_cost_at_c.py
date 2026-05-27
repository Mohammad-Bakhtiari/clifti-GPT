#!/usr/bin/env python
"""Rebuild ``comm_cost_results.csv`` for a fixed party count P without CrypTen.

Sweeps a list of federated client counts ``C`` while holding the SMPC
computational party count ``P`` fixed (default ``P=3``). Analytical byte
counts are recomputed exactly. Wall-clock values are taken from
``--timing_csv`` when an exact ``(workflow, mode, payload, n_clients)``
match exists; otherwise KNN/binning timings are log-linearly
interpolated between the nearest bracketing client counts in the prior
CSV.

Re-run ``analysis/comm_cost/comm_cost.py`` (with CrypTen) to replace interpolated
timings with fresh measurements.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.comm_cost.comm_cost import (  # noqa: E402
    binning_bytes,
    ft_weight_sharing_bytes,
    knn_reference_mapping_bytes,
    resolve_n_parties,
    COMM_COST_N_PARTIES_ENV,
    _DEFAULT_N_PARTIES,
)
from analysis.comm_cost.paths import COMM_COST_OUTPUT_DIR  # noqa: E402


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_thetas(s: str) -> List[int]:
    return [int(x.replace("_", "")) for x in s.split(",") if x.strip()]


def _lookup_time(
    prior: pd.DataFrame,
    workflow: str,
    mode: str,
    payload: str,
    n_clients: int,
) -> Optional[float]:
    slc = prior[
        (prior["workflow"] == workflow)
        & (prior["mode"] == mode)
        & (prior["payload"] == payload)
        & (prior["n_clients"] == n_clients)
    ]
    if slc.empty:
        return None
    return float(slc.iloc[0]["t_seconds"])


def _interpolate_time(
    prior: pd.DataFrame,
    workflow: str,
    mode: str,
    payload: str,
    target_c: int,
) -> float:
    exact = _lookup_time(prior, workflow, mode, payload, target_c)
    if exact is not None:
        return exact

    slc = prior[
        (prior["workflow"] == workflow)
        & (prior["mode"] == mode)
        & (prior["payload"] == payload)
    ]
    if slc.empty:
        return float("nan")

    lower = slc[slc["n_clients"] < target_c].sort_values("n_clients")
    upper = slc[slc["n_clients"] > target_c].sort_values("n_clients")
    if lower.empty or upper.empty:
        nearest = slc.iloc[(slc["n_clients"] - target_c).abs().argmin()]
        return float(nearest["t_seconds"])

    c_lo = int(lower.iloc[-1]["n_clients"])
    t_lo = float(lower.iloc[-1]["t_seconds"])
    c_hi = int(upper.iloc[0]["n_clients"])
    t_hi = float(upper.iloc[0]["t_seconds"])
    if t_lo <= 0 or t_hi <= 0:
        return float("nan")
    frac = (target_c - c_lo) / (c_hi - c_lo)
    return math.exp(math.log(t_lo) + frac * (math.log(t_hi) - math.log(t_lo)))


def _row(
    cost,
    workflow: str,
    mode: str,
    n_clients: int,
    n_parties: int,
    payload: str,
    rounds: int,
    t_seconds: float,
) -> Dict[str, Any]:
    overhead = 1.0
    if mode in {"smpc", "fed-weight-avg-smpc", "fed-hist-smpc"}:
        overhead = float("nan")
    return {
        "workflow": workflow,
        "mode": mode,
        "n_clients": n_clients,
        "n_parties": n_parties,
        "payload": payload,
        "rounds": rounds,
        "t_seconds": t_seconds,
        "bytes_per_client_per_round": (
            cost.per_client_per_round()
            if hasattr(cost, "per_client_per_round")
            else cost.bytes_total_per_client / max(1, rounds)
        ),
        "bytes_per_client_total": cost.bytes_total_per_client,
        "bytes_federation_total": cost.bytes_total_federation,
        "crypto_overhead": overhead,
        "notes": cost.notes,
    }


def build_rows_for_C(
    n_clients: int,
    n_parties: int,
    prior: pd.DataFrame,
    ft_thetas: List[int],
    ft_rounds: int,
    knn_n_query: List[int],
    knn_n_ref: List[int],
    knn_k: List[int],
    knn_d_embed: int,
    knn_n_classes: int,
    binning_n_bins: int,
    binning_grid_resolution: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    ft_pairs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for theta in ft_thetas:
        payload = f"theta={theta}"
        pair: Dict[str, Dict[str, Any]] = {}
        for smpc in (False, True):
            mode = "smpc" if smpc else "plain"
            cost = ft_weight_sharing_bytes(
                theta, n_clients, n_parties, ft_rounds, smpc=smpc
            )
            t = _interpolate_time(prior, "fine_tuning", mode, payload, n_clients)
            row = _row(
                cost, "fine_tuning", mode, n_clients, n_parties, payload, ft_rounds, t
            )
            pair[mode] = row
            rows.append(row)
        ft_pairs[payload] = pair
        t_plain = pair["plain"]["t_seconds"]
        if t_plain and t_plain > 0:
            pair["smpc"]["crypto_overhead"] = pair["smpc"]["t_seconds"] / t_plain

    for n_query in knn_n_query:
        for n_ref_total in knn_n_ref:
            for k in knn_k:
                payload = f"n_q={n_query},n_r={n_ref_total},d={knn_d_embed},k={k}"
                pair: Dict[str, Dict[str, Any]] = {}
                for smpc in (False, True):
                    mode = "plain" if not smpc else "smpc"
                    cost = knn_reference_mapping_bytes(
                        n_query,
                        n_ref_total,
                        n_clients,
                        n_parties,
                        knn_d_embed,
                        k,
                        knn_n_classes,
                        smpc=smpc,
                    )
                    t = _interpolate_time(
                        prior, "reference_mapping", mode, payload, n_clients
                    )
                    row = _row(
                        cost,
                        "reference_mapping",
                        mode,
                        n_clients,
                        n_parties,
                        payload,
                        1,
                        t,
                    )
                    pair[mode] = row
                    rows.append(row)
                t_plain = pair["plain"]["t_seconds"]
                if t_plain and t_plain > 0:
                    pair["smpc"]["crypto_overhead"] = (
                        pair["smpc"]["t_seconds"] / t_plain
                    )

    bin_pairs: Dict[str, Dict[str, Any]] = {}
    payload_bin = f"n_bins={binning_n_bins},M={binning_grid_resolution}"
    for strategy in (
        "fed-weight-avg",
        "fed-weight-avg-smpc",
        "fed-hist",
        "fed-hist-smpc",
    ):
        cost = binning_bytes(
            strategy, n_clients, n_parties, binning_n_bins, binning_grid_resolution
        )
        t = _interpolate_time(prior, "binning", strategy, payload_bin, n_clients)
        row = _row(
            cost, "binning", strategy, n_clients, n_parties, payload_bin, 1, t
        )
        bin_pairs[strategy] = row
        rows.append(row)

    if (
        "fed-weight-avg-smpc" in bin_pairs
        and "fed-weight-avg" in bin_pairs
        and bin_pairs["fed-weight-avg"]["t_seconds"] > 0
    ):
        bin_pairs["fed-weight-avg-smpc"]["crypto_overhead"] = (
            bin_pairs["fed-weight-avg-smpc"]["t_seconds"]
            / bin_pairs["fed-weight-avg"]["t_seconds"]
        )
    if (
        "fed-hist-smpc" in bin_pairs
        and "fed-hist" in bin_pairs
        and bin_pairs["fed-hist"]["t_seconds"] > 0
    ):
        bin_pairs["fed-hist-smpc"]["crypto_overhead"] = (
            bin_pairs["fed-hist-smpc"]["t_seconds"]
            / bin_pairs["fed-hist"]["t_seconds"]
        )

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n_parties",
        type=int,
        default=None,
        help=(
            "Fixed SMPC computational party count P used in byte formulas. "
            f"If omitted, reads ${COMM_COST_N_PARTIES_ENV} (default "
            f"{_DEFAULT_N_PARTIES})."
        ),
    )
    p.add_argument(
        "--ft_clients",
        type=str,
        default="2,3,5,10",
        help="Federated client counts C swept for the fine-tuning workflow.",
    )
    p.add_argument(
        "--knn_clients",
        type=str,
        default="2,5",
        help="Federated client counts C swept for the KNN workflow.",
    )
    p.add_argument(
        "--binning_clients",
        type=str,
        default="2,5,10",
        help="Federated client counts C swept for the binning workflow.",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=f"{COMM_COST_OUTPUT_DIR}/comm_cost_results.csv",
    )
    p.add_argument(
        "--timing_csv",
        type=str,
        default=f"{COMM_COST_OUTPUT_DIR}/comm_cost_results.csv",
        help="Prior results used to copy/interpolate wall-clock timings.",
    )
    p.add_argument("--backup", action="store_true", help="Copy prior CSV to *.bak")
    p.add_argument("--ft_thetas", type=str, default="1_000_000,10_000_000,50_000_000")
    p.add_argument("--ft_rounds", type=int, default=5)
    p.add_argument("--knn_n_query", type=str, default="500,2000")
    p.add_argument("--knn_n_ref", type=str, default="1000,5000")
    p.add_argument("--knn_k", type=str, default="5,20")
    p.add_argument("--knn_d_embed", type=int, default=128)
    p.add_argument("--knn_n_classes", type=int, default=10)
    p.add_argument("--binning_n_bins", type=int, default=51)
    p.add_argument("--binning_grid_resolution", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.n_parties = resolve_n_parties(args.n_parties)
    out = Path(args.output_csv)
    timing_path = Path(args.timing_csv)
    if args.backup and out.is_file():
        backup = out.with_suffix(out.suffix + ".bak")
        shutil.copy2(out, backup)
        print(f"Backed up {out} -> {backup}")

    if args.n_parties < 2:
        raise ValueError(
            f"--n_parties must be >= 2 for additive sharing; got {args.n_parties}"
        )

    prior = pd.read_csv(timing_path)
    ft_clients = _parse_int_list(args.ft_clients)
    knn_clients = _parse_int_list(args.knn_clients)
    binning_clients = _parse_int_list(args.binning_clients)

    all_rows: List[Dict[str, Any]] = []
    seen_c = set(ft_clients) | set(knn_clients) | set(binning_clients)
    for n_clients in sorted(seen_c):
        rows = build_rows_for_C(
            n_clients=n_clients,
            n_parties=args.n_parties,
            prior=prior,
            ft_thetas=_parse_thetas(args.ft_thetas),
            ft_rounds=args.ft_rounds,
            knn_n_query=_parse_int_list(args.knn_n_query),
            knn_n_ref=_parse_int_list(args.knn_n_ref),
            knn_k=_parse_int_list(args.knn_k),
            knn_d_embed=args.knn_d_embed,
            knn_n_classes=args.knn_n_classes,
            binning_n_bins=args.binning_n_bins,
            binning_grid_resolution=args.binning_grid_resolution,
        )
        for r in rows:
            wf = r["workflow"]
            if wf == "fine_tuning" and n_clients not in ft_clients:
                continue
            if wf == "reference_mapping" and n_clients not in knn_clients:
                continue
            if wf == "binning" and n_clients not in binning_clients:
                continue
            all_rows.append(r)

    fieldnames = [
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
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows)[fieldnames].to_csv(out, index=False)
    print(
        f"Wrote {out} ({len(all_rows)} rows, P={args.n_parties}, "
        f"C in {sorted(seen_c)})"
    )
    print(
        "Note: wall-clock values are looked up by (workflow, mode, payload, "
        "n_clients); KNN/binning timings are log-linearly interpolated when "
        "the exact C is missing. Re-run analysis/comm_cost.py with CrypTen "
        "for measured timings."
    )


if __name__ == "__main__":
    main()
