#!/usr/bin/env python
"""Plot communication-cost benchmark results from analysis/comm_cost.py.

Produces PNG figures and a supplementary LaTeX table that summarise the
analytical bandwidth + simulated wall-clock benchmark covering:

- Fine-tuning weight sharing (FedAvg plaintext vs SMPC)
- Federated reference mapping (plaintext FAISS KNN vs SMPC top-k)
- Federated binning (plaintext vs SMPC aggregation)

Figures:
    comm_cost_scaling.png   — bytes vs C for FT and KNN (plain vs SMPC)
    comm_cost_wallclock.png — measured median t (s) per workflow, grouped
                              by plain vs SMPC

LaTeX:
    docs/methods/communication_cost.tex (rendered with article + booktabs)
"""

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FONT_SIZE = 10
PLAIN_COLOR = "#4477AA"
SMPC_COLOR = "#EE6677"
BAR_EDGE_COLOR = "0.15"
BAR_EDGE_WIDTH = 0.6


def _apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {path}")


def plot_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    """Two-panel bandwidth scaling: FT vs C and KNN vs n_ref."""
    import matplotlib.pyplot as plt

    _apply_style()
    fig, (ax_ft, ax_knn) = plt.subplots(1, 2, figsize=(9, 3.4))

    ft = df[df["workflow"] == "fine_tuning"].copy()
    if not ft.empty:
        ft["theta"] = ft["payload"].str.extract(r"theta=(\d+)").astype(int)
        theta_target = int(ft["theta"].mode().iloc[0])
        ft_t = ft[ft["theta"] == theta_target].copy()
        ft_t.sort_values(["mode", "n_clients"], inplace=True)
        for mode, color in [("plain", PLAIN_COLOR), ("smpc", SMPC_COLOR)]:
            sub = ft_t[ft_t["mode"] == mode]
            if not sub.empty:
                ax_ft.plot(
                    sub["n_clients"],
                    sub["bytes_per_client_total"] / 1e6,
                    "o-",
                    color=color,
                    label=mode.upper() if mode == "smpc" else "Plaintext",
                    markeredgecolor=BAR_EDGE_COLOR,
                    markeredgewidth=BAR_EDGE_WIDTH,
                )
        ax_ft.set_xlabel("Number of clients C")
        ax_ft.set_ylabel("Bytes / client (MB)")
        ax_ft.set_title(f"Fine-tuning, |θ|={theta_target:,}")
        ax_ft.set_xticks(sorted(ft_t["n_clients"].unique()))
        ax_ft.legend(loc="upper left", frameon=False)
        ax_ft.grid(alpha=0.3)

    knn = df[df["workflow"] == "reference_mapping"].copy()
    if not knn.empty:
        knn["n_ref"] = knn["payload"].str.extract(r"n_r=(\d+)").astype(int)
        knn["n_q"] = knn["payload"].str.extract(r"n_q=(\d+)").astype(int)
        # Pick one (n_q, C) slice for cleaner curves
        ref_nq = int(knn["n_q"].mode().iloc[0])
        ref_C = int(knn["n_clients"].mode().iloc[0])
        slc = knn[(knn["n_q"] == ref_nq) & (knn["n_clients"] == ref_C)].copy()
        slc.sort_values(["mode", "n_ref"], inplace=True)
        for mode, color in [("plain", PLAIN_COLOR), ("smpc", SMPC_COLOR)]:
            sub = slc[slc["mode"] == mode]
            if not sub.empty:
                ax_knn.plot(
                    sub["n_ref"],
                    sub["bytes_per_client_total"] / 1e6,
                    "s-",
                    color=color,
                    label="SMPC" if mode == "smpc" else "Plaintext",
                    markeredgecolor=BAR_EDGE_COLOR,
                    markeredgewidth=BAR_EDGE_WIDTH,
                )
        ax_knn.set_xlabel("Total reference cells")
        ax_knn.set_ylabel("Bytes / client (MB)")
        ax_knn.set_title(f"KNN, n_q={ref_nq}, C={ref_C}")
        ax_knn.set_xscale("log")
        ax_knn.set_yscale("log")
        ax_knn.legend(loc="upper left", frameon=False)
        ax_knn.grid(alpha=0.3, which="both")

    fig.tight_layout()
    _save(fig, out_dir / "comm_cost_scaling.png")


def plot_wallclock(df: pd.DataFrame, out_dir: Path) -> None:
    """Median wall-clock per workflow with overhead annotation."""
    import matplotlib.pyplot as plt

    _apply_style()

    workflows = ["fine_tuning", "reference_mapping", "binning"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))

    bars: List[Dict] = []
    x_labels: List[str] = []
    overheads: List[float] = []

    for wf in workflows:
        wf_df = df[df["workflow"] == wf]
        if wf_df.empty:
            continue
        if wf == "binning":
            # Show two binning families: weight-avg and hist
            for family, label in [
                ("fed-weight-avg", "Binning W-avg"),
                ("fed-hist", "Binning Hist"),
            ]:
                plain = wf_df[wf_df["mode"] == family]
                smpc = wf_df[wf_df["mode"] == f"{family}-smpc"]
                if plain.empty or smpc.empty:
                    continue
                t_plain = float(plain["t_seconds"].median())
                t_smpc = float(smpc["t_seconds"].median())
                x_labels.append(label)
                bars.append({"label": label, "t_plain": t_plain, "t_smpc": t_smpc})
                overheads.append(t_smpc / t_plain if t_plain > 0 else float("nan"))
        else:
            plain = wf_df[wf_df["mode"] == "plain"]
            smpc = wf_df[wf_df["mode"] == "smpc"]
            if plain.empty or smpc.empty:
                continue
            label_map = {
                "fine_tuning": "Fine-tuning",
                "reference_mapping": "Reference mapping",
            }
            t_plain = float(plain["t_seconds"].median())
            t_smpc = float(smpc["t_seconds"].median())
            x_labels.append(label_map[wf])
            bars.append(
                {"label": label_map[wf], "t_plain": t_plain, "t_smpc": t_smpc}
            )
            overheads.append(t_smpc / t_plain if t_plain > 0 else float("nan"))

    if not bars:
        print("No wall-clock rows in CSV; skipping wall-clock figure.")
        plt.close(fig)
        return

    x = np.arange(len(bars))
    width = 0.38
    plain_vals = [b["t_plain"] for b in bars]
    smpc_vals = [b["t_smpc"] for b in bars]
    ax.bar(
        x - width / 2,
        plain_vals,
        width,
        color=PLAIN_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        label="Plaintext",
    )
    ax.bar(
        x + width / 2,
        smpc_vals,
        width,
        color=SMPC_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        label="SMPC",
    )
    for i, ovh in enumerate(overheads):
        y_top = max(plain_vals[i], smpc_vals[i])
        ax.text(
            x[i],
            y_top * 1.05,
            f"{ovh:.1f}x",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 1,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Wall-clock time (s, median)")
    ax.set_yscale("log")
    ax.set_title("Simulated MPC wall-clock per workflow")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    _save(fig, out_dir / "comm_cost_wallclock.png")


def _fmt_bytes(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}~GB"
    if n >= 1e6:
        return f"{n/1e6:.2f}~MB"
    if n >= 1e3:
        return f"{n/1e3:.2f}~KB"
    return f"{int(n)}~B"


def _fmt_time(t: float) -> str:
    """Format runtime for LaTeX tables (non-breaking unit suffixes)."""
    if not np.isfinite(t):
        return "--"
    if t >= 1.0:
        return f"{t:.2f}~s"
    if t >= 1e-3:
        return f"{t * 1e3:.2f}~ms"
    return f"{t * 1e6:.0f}~$\\mu$s"


def _latex_int(n: int) -> str:
    """Compact integer for math mode (scientific notation when large)."""
    if n >= 10000:
        exp = int(math.floor(math.log10(n)))
        mant = n / (10**exp)
        if abs(mant - round(mant)) < 1e-9:
            mant_i = int(round(mant))
            if mant_i == 1:
                return f"10^{{{exp}}}"
            return f"{mant_i} \\times 10^{{{exp}}}"
    return str(n)


CONFIG_MAX_CHARS = 10
CONFIG_COL_WIDTH = "1.35cm"


def _join_config_lines(parts: List[str]) -> str:
    return r" \\ ".join(parts)


def _param_lines(name: str, value: str) -> List[str]:
    """One table parameter; each line has at most CONFIG_MAX_CHARS characters."""
    one_line = f"${name}={value}$"
    if len(one_line) <= CONFIG_MAX_CHARS:
        return [one_line]

    lines = [f"${name}=$"]
    if " \\times " in value:
        mant, exp = value.split(" \\times ", 1)
        lines.extend([f"${mant}$", r"$\times$", f"${exp}$"])
    else:
        chunk = f"${value}$"
        while chunk:
            if len(chunk) <= CONFIG_MAX_CHARS:
                lines.append(chunk)
                break
            lines.append(chunk[:CONFIG_MAX_CHARS])
            chunk = chunk[CONFIG_MAX_CHARS:]
    return lines


def _fmt_theta_lines(theta: int) -> List[str]:
    lines = [r"$|\theta|$"]
    value = _latex_int(theta)
    eq_line = f"$={value}$"
    if len(eq_line) <= CONFIG_MAX_CHARS:
        lines.append(eq_line)
        return lines

    if " \\times " in value:
        mant, exp = value.split(" \\times ", 1)
        lines.extend([f"$={mant}$", r"$\times$", f"${exp}$"])
    else:
        lines.append(eq_line)
    return lines


def _fmt_payload_latex(payload: str) -> str:
    """Render configuration strings on multiple narrow lines."""
    payload = str(payload).strip()
    if payload.startswith("theta="):
        theta = int(payload.split("=", 1)[1])
        return _join_config_lines(_fmt_theta_lines(theta))

    knn_match = re.fullmatch(
        r"n_q=(\d+),n_r=(\d+),d=(\d+),k=(\d+)", payload
    )
    if knn_match:
        n_q, n_r, d, k = knn_match.groups()
        parts: List[str] = []
        for name, val in (
            ("n_q", n_q),
            ("n_r", _latex_int(int(n_r))),
            ("d", d),
            ("k", k),
        ):
            parts.extend(_param_lines(name, val))
        return _join_config_lines(parts)

    bin_match = re.fullmatch(r"n_bins=(\d+),M=(\d+)", payload)
    if bin_match:
        n_bins, grid = bin_match.groups()
        parts: List[str] = []
        parts.extend(_param_lines("B", n_bins))
        parts.extend(_param_lines("M", grid))
        return _join_config_lines(parts)

    chunk = payload.replace("_", "\\_")
    if len(chunk) <= CONFIG_MAX_CHARS:
        return chunk
    lines = [
        chunk[i : i + CONFIG_MAX_CHARS]
        for i in range(0, len(chunk), CONFIG_MAX_CHARS)
    ]
    return _join_config_lines(lines)


def _config_multirow_cell(config: str, n_rows: int) -> str:
    return (
        rf"\multirow{{{n_rows}}}{{*}}{{"
        rf"\begin{{minipage}}[t]{{{CONFIG_COL_WIDTH}}}\raggedright "
        rf"{config}\end{{minipage}}}}"
    )


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


_MODE_SORT_ORDER = {
    "plain": 0,
    "smpc": 1,
    "fed-weight-avg": 0,
    "fed-weight-avg-smpc": 1,
    "fed-hist": 2,
    "fed-hist-smpc": 3,
}


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
        r"@{} "
        r">{\raggedright\arraybackslash}p{1.7cm} "
        r"c "
        r">{\raggedright\arraybackslash}p{" + CONFIG_COL_WIDTH + "} "
        r">{\raggedleft\arraybackslash}p{1.6cm} "
        r">{\raggedleft\arraybackslash}p{1.5cm} "
        r">{\raggedleft\arraybackslash}p{1.3cm} "
        r"@{}"
    )


def _write_workflow_table(
    sub: pd.DataFrame,
    caption: str,
    label: str,
    out_path: Path,
) -> None:
    """Write one longtable for a single workflow (no Workflow column)."""
    lines: List[str] = []
    lines.append(r"% Auto-generated by analysis/plot_comm_cost.py. Do not edit by hand.")
    lines.append(r"\begin{longtable}{" + _workflow_table_colspec() + "}")
    lines.append(rf"\caption{{{caption}}}\label{{{label}}} \\")
    lines.append(r"\toprule")
    lines.append(
        r"Mode & $C$ & Configuration & Bytes/client & "
        r"$t$ (median) & Overhead \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(
        r"Mode & $C$ & Configuration & Bytes/client & "
        r"$t$ (median) & Overhead \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    sub = _sort_workflow_table(sub)
    groups = list(sub.groupby("payload", sort=False))
    for group_idx, (payload, group) in enumerate(groups):
        config = _fmt_payload_latex(str(payload))
        n_rows = len(group)
        is_last_group = group_idx == len(groups) - 1
        for row_idx, (_, row) in enumerate(group.iterrows()):
            mode = _fmt_mode(str(row["mode"]))
            n_clients = int(row["n_clients"])
            bytes_str = _fmt_bytes(float(row["bytes_per_client_total"]))
            time_str = _fmt_time(float(row["t_seconds"]))
            overhead = float(row["crypto_overhead"])
            if row_idx == 0:
                config_cell = _config_multirow_cell(config, n_rows)
            else:
                config_cell = ""
            if row_idx == n_rows - 1 and not is_last_group:
                # longtable + multirow: booktabs \midrule breaks the next row;
                # use \noalign{\hrule} immediately after the row break instead.
                row_end = r" \\ \noalign{\vskip 5pt\hrule height 0.9pt\vskip 5pt}"
            else:
                row_end = r" \\"
            lines.append(
                f"{mode} & {n_clients} & {config_cell} & "
                f"{bytes_str} & {time_str} & {overhead:.1f}$\\times${row_end}"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    out_path.write_text("\n".join(lines))


WORKFLOW_TABLES = (
    (
        "fine_tuning",
        "Fine-tuning weight sharing: bandwidth and wall-clock.",
        "tab:comm_cost_ft",
        "communication_cost_table_fine_tuning.tex",
    ),
    (
        "reference_mapping",
        "Federated reference mapping (KNN): bandwidth and wall-clock.",
        "tab:comm_cost_knn",
        "communication_cost_table_reference_mapping.tex",
    ),
    (
        "binning",
        "Federated binning: bandwidth and wall-clock.",
        "tab:comm_cost_binning",
        "communication_cost_table_binning.tex",
    ),
)


def write_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    """Render one longtable per workflow plus a wrapper ``\\input`` file.

    ``out_path`` is the legacy wrapper path (e.g.
    ``docs/methods/communication_cost_table.tex``). Individual tables are
    written alongside it as ``communication_cost_table_<workflow>.tex``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_lines: List[str] = [
        r"% Auto-generated by analysis/plot_comm_cost.py. Do not edit by hand.",
    ]

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
    """Write a small ``\\newcommand`` block with computed headline numbers.

    Lets the prose in ``docs/methods/communication_cost.tex`` cite numbers
    such as the SMPC fine-tuning bandwidth multiplier at the largest C or
    the crypto overhead measured for the KNN benchmark, without having to
    hardcode them in the tex source.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _safe(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}"

    ft = df[df["workflow"] == "fine_tuning"]
    knn = df[df["workflow"] == "reference_mapping"]

    macros: List[str] = [
        r"% Auto-generated by analysis/plot_comm_cost.py. Do not edit by hand.",
    ]

    if not ft.empty:
        ft_smpc = ft[ft["mode"] == "smpc"]
        if not ft_smpc.empty:
            max_c = int(ft_smpc["n_clients"].max())
            row_smpc = ft_smpc[ft_smpc["n_clients"] == max_c].iloc[0]
            plain_match = ft[(ft["mode"] == "plain") & (ft["n_clients"] == max_c)]
            if not plain_match.empty:
                plain_bytes = float(plain_match["bytes_per_client_total"].iloc[0])
                ratio = float(row_smpc["bytes_per_client_total"]) / max(1.0, plain_bytes)
                macros.append(_safe("ftBytesMaxC", f"{max_c}"))
                macros.append(_safe("ftBytesRatioMax", f"{ratio:.1f}"))
                macros.append(
                    _safe(
                        "ftSmpcOverheadMax",
                        f"{float(ft_smpc['crypto_overhead'].max()):.1f}",
                    )
                )

    if not knn.empty:
        knn_plain = knn[knn["mode"] == "plain"]
        knn_smpc = knn[knn["mode"] == "smpc"]
        if not knn_smpc.empty:
            ov_median = float(knn_smpc["crypto_overhead"].median())
            ov_max = float(knn_smpc["crypto_overhead"].max())
            byt = float(knn_smpc["bytes_per_client_total"].median())
            macros.append(_safe("knnSmpcOverheadMedian", f"{ov_median:.1f}"))
            macros.append(_safe("knnSmpcOverheadMax", f"{ov_max:.1f}"))
            macros.append(_safe("knnSmpcBytesMedian", _fmt_bytes(byt)))
            if not knn_plain.empty:
                t_plain = float(knn_plain["t_seconds"].median())
                t_smpc = float(knn_smpc["t_seconds"].median())
                if t_plain > 0:
                    fig_ov = t_smpc / t_plain
                    macros.append(_safe("knnSmpcOverheadFig", f"{fig_ov:.1f}"))

    out_path.write_text("\n".join(macros))
    print(f"LaTeX macros written to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_csv",
        type=str,
        default="output/comm_cost/comm_cost_results.csv",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="output/comm_cost/figures",
    )
    p.add_argument(
        "--latex_table",
        type=str,
        default="docs/methods/communication_cost_table.tex",
        help="Output path for the auto-generated longtable fragment.",
    )
    p.add_argument(
        "--latex_macros",
        type=str,
        default="docs/methods/communication_cost_macros.tex",
        help="Output path for auto-generated headline-number macros.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    out_dir = Path(args.out_dir)
    plot_scaling(df, out_dir)
    plot_wallclock(df, out_dir)
    write_latex_table(df, Path(args.latex_table))
    write_headline_macros(df, Path(args.latex_macros))


if __name__ == "__main__":
    main()
