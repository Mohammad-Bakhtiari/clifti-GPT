#!/usr/bin/env python
"""Render communication-cost LaTeX table fragments and macros from CSV.

Reads ``comm_cost_results.csv`` and writes auto-generated fragments into
``output/comm_cost/``. Only **measured** wall-clock fields are exported
(``t_seconds``, ``crypto_overhead``); analytical byte counts in the CSV are
ignored.

- ``communication_cost_table.tex`` — wrapper that ``\\input``s the per-workflow tables
- ``communication_cost_table_*.tex`` — longtables (one per workflow)
- ``communication_cost_macros.tex`` — headline ``\\newcommand`` values for prose

Run whenever the CSV changes::

    python analysis/comm_cost/render_comm_cost_tex.py
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.comm_cost import COMM_COST_OUTPUT_DIR  # noqa: E402

_GENERATOR = "analysis/comm_cost/render_comm_cost_tex.py"

CONFIG_COL_WIDTH = "2.8cm"
COL_MODE = "2.2cm"
COL_C = "0.65cm"
COL_TIME = "2.4cm"
COL_OVERHEAD = "2.5cm"
NUM_COLUMNS = 5

_MODE_SORT_ORDER = {
    "plain": 0,
    "smpc": 1,
    "fed-weight-avg": 0,
    "fed-weight-avg-smpc": 1,
    "fed-hist": 2,
    "fed-hist-smpc": 3,
}

WORKFLOW_TABLES = (
    (
        "fine_tuning",
        "Fine-tuning weight sharing: measured wall-clock.",
        "tab:comm_cost_ft",
        "communication_cost_table_fine_tuning.tex",
    ),
    (
        "reference_mapping",
        "Federated reference mapping (KNN): measured wall-clock.",
        "tab:comm_cost_knn",
        "communication_cost_table_reference_mapping.tex",
    ),
    (
        "binning",
        "Federated binning: measured wall-clock.",
        "tab:comm_cost_binning",
        "communication_cost_table_binning.tex",
    ),
)


def _fmt_time(t: float) -> str:
    """Format runtime for LaTeX tables (non-breaking unit suffixes)."""
    if not np.isfinite(t):
        return "--"
    if t >= 1.0:
        return f"{t:.2f}~s"
    if t >= 1e-3:
        return f"{t * 1e3:.2f}~ms"
    return f"{t * 1e6:.0f}~$\\mu$s"



def _latex_value_inline(n: int) -> str:
    """Single-line math value (no spaces around \\times)."""
    if n >= 10000:
        exp = int(math.floor(math.log10(n)))
        mant = n / (10**exp)
        if abs(mant - round(mant)) < 1e-9:
            mant_i = int(round(mant))
            if mant_i == 1:
                return f"10^{{{exp}}}"
            return f"{mant_i}\\times10^{{{exp}}}"
    return str(n)


def _param_line(name: str, value: int | str) -> str:
    """One configuration item on exactly one line."""
    if isinstance(value, int):
        value = _latex_value_inline(value)
    else:
        value = str(value).replace(" \\times ", "\\times")
    return f"${name}={value}$"


def _join_config_lines(parts: List[str]) -> str:
    return r" \\ ".join(parts)


def _fmt_theta_line(theta: int) -> str:
    return f"$|\\theta|={_latex_value_inline(theta)}$"


def _fmt_payload_latex(payload: str) -> str:
    """Render one configuration item per line (never split mid-parameter)."""
    payload = str(payload).strip()
    if payload.startswith("theta="):
        theta = int(payload.split("=", 1)[1])
        return _fmt_theta_line(theta)

    knn_match = re.fullmatch(
        r"n_q=(\d+),n_r=(\d+),d=(\d+),k=(\d+)", payload
    )
    if knn_match:
        n_q, n_r, d, k = knn_match.groups()
        parts = [
            _param_line("n_q", int(n_q)),
            _param_line("n_r", int(n_r)),
            _param_line("d", int(d)),
            _param_line("k", int(k)),
        ]
        return _join_config_lines(parts)

    bin_match = re.fullmatch(r"n_bins=(\d+),M=(\d+)", payload)
    if bin_match:
        n_bins, grid = bin_match.groups()
        parts = [
            _param_line("B", int(n_bins)),
            _param_line("M", int(grid)),
        ]
        return _join_config_lines(parts)

    return payload.replace("_", "\\_")


def _wrap_cell(width: str, content: str, halign: str = "left") -> str:
    """Constrain cell content to a fixed width so columns cannot bleed."""
    align_cmd = {
        "left": r"\raggedright",
        "center": r"\centering",
        "right": r"\raggedleft",
    }[halign]
    return (
        rf"\begin{{minipage}}[c]{{{width}}}{align_cmd} {content}\end{{minipage}}"
    )


def _config_cell(config: str, n_rows: int, first_in_group: bool) -> str:
    """Merged configuration cell, vertically centred across ``n_rows``."""
    if not first_in_group:
        return ""
    body = (
        rf"\begin{{minipage}}[c]{{{CONFIG_COL_WIDTH}}}"
        rf"\setlength{{\parskip}}{{0pt}}"
        rf"\setlength{{\baselineskip}}{{11pt}}"
        rf"\centering {config}\end{{minipage}}"
    )
    return rf"\multirow{{{n_rows}}}{{=}}{{{body}}}"


def _group_rule() -> str:
    """Full-width separator between configuration blocks (all columns)."""
    return rf" \\ \noalign{{\vskip 6pt}}\cline{{{1}-{NUM_COLUMNS}}}\noalign{{\vskip 6pt}}"


def _fmt_mode(mode: str) -> str:
    labels = {
        "plain": "Plaintext",
        "smpc": "SMPC",
        "fed-weight-avg": "W-avg",
        "fed-weight-avg-smpc": "W-avg SMPC",
        "fed-hist": "Hist",
        "fed-hist-smpc": "Hist SMPC",
    }
    return labels.get(mode, mode.replace("_", "\\_"))


def _sort_workflow_table(sub: pd.DataFrame) -> pd.DataFrame:
    """Group rows by configuration, then C, then mode."""
    out = sub.copy()
    out["_mode_order"] = out["mode"].map(
        lambda m: _MODE_SORT_ORDER.get(str(m), 99)
    )
    return (
        out.sort_values(["payload", "n_clients", "_mode_order"], kind="stable")
        .drop(columns="_mode_order")
        .reset_index(drop=True)
    )


def _workflow_table_colspec() -> str:
    return (
        r">{\raggedright\arraybackslash}p{" + COL_MODE + "} "
        r">{\centering\arraybackslash}p{" + COL_C + "} "
        r">{\centering\arraybackslash}p{" + CONFIG_COL_WIDTH + "} "
        r">{\raggedleft\arraybackslash}p{" + COL_TIME + "} "
        r">{\raggedleft\arraybackslash}p{" + COL_OVERHEAD + "}"
    )


def _write_workflow_table(
    sub: pd.DataFrame,
    caption: str,
    label: str,
    out_path: Path,
) -> None:
    """Write one longtable for a single workflow (no Workflow column)."""
    header = rf"% Auto-generated by {_GENERATOR}. Do not edit by hand."
    parties_vals = (
        sorted(int(v) for v in sub["n_parties"].dropna().unique())
        if "n_parties" in sub.columns
        else []
    )
    if len(parties_vals) == 1:
        caption = (
            caption.rstrip(".")
            + f". Fixed $P={parties_vals[0]}$ SMPC parties."
        )
    elif parties_vals:
        caption = (
            caption.rstrip(".")
            + f". SMPC parties $P\\in\\{{{','.join(str(p) for p in parties_vals)}\\}}$."
        )
    lines: List[str] = [header]
    lines.append(r"{\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\setlength{\extrarowheight}{2pt}")
    lines.append(r"\renewcommand{\multirowsetup}{\centering}")
    lines.append(r"\begin{longtable}{" + _workflow_table_colspec() + "}")
    lines.append(rf"\caption{{{caption}}}\label{{{label}}} \\")
    lines.append(r"\toprule")
    lines.append(
        r"Mode & $C$ & Configuration & $t$ (median) & Overhead \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(
        r"Mode & $C$ & Configuration & $t$ (median) & Overhead \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    sub = _sort_workflow_table(sub)
    groups = list(sub.groupby("payload", sort=False))
    for group_idx, (payload, group) in enumerate(groups):
        config = _fmt_payload_latex(str(payload))
        is_last_group = group_idx == len(groups) - 1
        n_rows = len(group)
        for row_idx, (_, row) in enumerate(group.iterrows()):
            mode = _fmt_mode(str(row["mode"]))
            n_clients = int(row["n_clients"])
            time_str = _fmt_time(float(row["t_seconds"]))
            overhead = float(row["crypto_overhead"])
            overhead_str = f"{overhead:.1f}$\\times$"
            config_cell = _config_cell(config, n_rows, row_idx == 0)
            if row_idx == n_rows - 1 and not is_last_group:
                row_end = _group_rule()
            else:
                row_end = r" \\"
            lines.append(
                f"{_wrap_cell(COL_MODE, mode, 'left')} & "
                f"{_wrap_cell(COL_C, str(n_clients), 'center')} & "
                f"{config_cell} & "
                f"{_wrap_cell(COL_TIME, time_str, 'right')} & "
                f"{_wrap_cell(COL_OVERHEAD, overhead_str, 'right')}"
                f"{row_end}"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"}")
    out_path.write_text("\n".join(lines))


def write_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    """Render one longtable per workflow plus a wrapper ``\\input`` file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_lines: List[str] = [rf"% Auto-generated by {_GENERATOR}. Do not edit by hand."]

    for wf_key, caption, label, filename in WORKFLOW_TABLES:
        sub = df[df["workflow"] == wf_key]
        if sub.empty:
            continue
        part_path = out_path.parent / filename
        _write_workflow_table(sub, caption, label, part_path)
        wrapper_lines.append(rf"\input{{{filename}}}")
        wrapper_lines.append("")
        print(f"LaTeX table fragment written to {part_path}")

    wrapper_lines.append(
        r"% Legacy label: cite workflow-specific tables "
        r"(tab:comm_cost_ft, tab:comm_cost_knn, tab:comm_cost_binning)."
    )
    out_path.write_text("\n".join(wrapper_lines))
    print(f"LaTeX table wrapper written to {out_path}")


def write_headline_macros(df: pd.DataFrame, out_path: Path) -> None:
    """Write headline ``\\newcommand`` values for results prose."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _safe(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}"

    ft = df[df["workflow"] == "fine_tuning"]
    knn = df[df["workflow"] == "reference_mapping"]
    binning = df[df["workflow"] == "binning"]

    macros: List[str] = [rf"% Auto-generated by {_GENERATOR}. Do not edit by hand."]

    if "n_parties" in df.columns and not df["n_parties"].dropna().empty:
        parties = sorted(int(v) for v in df["n_parties"].dropna().unique())
        if len(parties) == 1:
            macros.append(_safe("smpcParties", f"{parties[0]}"))
        else:
            macros.append(
                _safe("smpcParties", "/".join(str(p) for p in parties))
            )

    if not ft.empty:
        ft_smpc = ft[ft["mode"] == "smpc"]
        if not ft_smpc.empty:
            macros.append(
                _safe(
                    "ftSmpcOverheadMax",
                    f"{float(ft_smpc['crypto_overhead'].max()):.1f}",
                )
            )
            macros.append(
                _safe(
                    "ftSmpcTimeMax",
                    _fmt_time(float(ft_smpc["t_seconds"].max())),
                )
            )

    if not knn.empty:
        knn_plain = knn[knn["mode"] == "plain"]
        knn_smpc = knn[knn["mode"] == "smpc"]
        if not knn_smpc.empty:
            ov_median = float(knn_smpc["crypto_overhead"].median())
            ov_max = float(knn_smpc["crypto_overhead"].max())
            macros.append(_safe("knnSmpcOverheadMedian", f"{ov_median:.1f}"))
            macros.append(_safe("knnSmpcOverheadMax", f"{ov_max:.1f}"))
            if not knn_plain.empty:
                t_plain = float(knn_plain["t_seconds"].median())
                t_smpc = float(knn_smpc["t_seconds"].median())
                if t_plain > 0:
                    fig_ov = t_smpc / t_plain
                    macros.append(_safe("knnSmpcOverheadFig", f"{fig_ov:.1f}"))

    if not binning.empty:
        for family, macro_prefix in [
            ("fed-weight-avg", "binWeighted"),
            ("fed-hist", "binHist"),
        ]:
            plain = binning[binning["mode"] == family]
            smpc = binning[binning["mode"] == f"{family}-smpc"]
            if plain.empty or smpc.empty:
                continue
            t_plain = float(plain["t_seconds"].median())
            t_smpc = float(smpc["t_seconds"].median())
            if t_plain > 0:
                macros.append(
                    _safe(f"{macro_prefix}SmpcOverhead", f"{t_smpc / t_plain:.1f}")
                )

    out_path.write_text("\n".join(macros))
    print(f"LaTeX macros written to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_csv",
        type=str,
        default=f"{COMM_COST_OUTPUT_DIR}/comm_cost_results.csv",
    )
    p.add_argument(
        "--latex_table",
        type=str,
        default=f"{COMM_COST_OUTPUT_DIR}/communication_cost_table.tex",
        help="Output path for the auto-generated longtable wrapper.",
    )
    p.add_argument(
        "--latex_macros",
        type=str,
        default=f"{COMM_COST_OUTPUT_DIR}/communication_cost_macros.tex",
        help="Output path for auto-generated headline-number macros.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.results_csv)
    df = pd.read_csv(csv_path)
    table_wrapper = Path(args.latex_table)
    macros_path = Path(args.latex_macros)
    write_latex_table(df, table_wrapper)
    write_headline_macros(df, macros_path)


if __name__ == "__main__":
    main()
