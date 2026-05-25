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
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {path}")


def plot_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    """Two-panel bandwidth scaling: FT vs C and KNN vs n_ref."""
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
        return f"{n/1e9:.2f}\\,GB"
    if n >= 1e6:
        return f"{n/1e6:.2f}\\,MB"
    if n >= 1e3:
        return f"{n/1e3:.2f}\\,KB"
    return f"{int(n)}\\,B"


def _fmt_time(t: float) -> str:
    if not np.isfinite(t):
        return "--"
    if t >= 1.0:
        return f"{t:.2f}\\,s"
    if t >= 1e-3:
        return f"{t*1e3:.2f}\\,ms"
    return f"{t*1e6:.2f}\\,\\textmu s"


def write_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    """Render a LaTeX ``longtable`` fragment that can be ``\\input``'d.

    The fragment is just the table (no document preamble) so a host
    supplementary document can include it via ``\\input{...}``. The host
    document is responsible for ``\\usepackage{longtable,booktabs}``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(r"% Auto-generated by analysis/plot_comm_cost.py. Do not edit by hand.")
    lines.append(r"\begin{longtable}{l l r l r r r}")
    lines.append(r"\caption{Communication-cost benchmark: analytical bandwidth (CrypTen additive-sharing) and median wall-clock per workflow.}\label{tab:comm_cost} \\")
    lines.append(r"\toprule")
    lines.append(
        r"Workflow & Mode & $C$ & Payload & Bytes/client & "
        r"$t$ (median) & Overhead \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(
        r"Workflow & Mode & $C$ & Payload & Bytes/client & "
        r"$t$ (median) & Overhead \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    workflow_labels = {
        "fine_tuning": "Fine-tuning",
        "reference_mapping": "Reference mapping",
        "binning": "Binning",
    }
    backslash_underscore = "\\_"
    for wf in ("fine_tuning", "reference_mapping", "binning"):
        sub = df[df["workflow"] == wf]
        for _, row in sub.iterrows():
            wf_label = workflow_labels.get(wf, wf)
            mode = str(row["mode"]).replace("_", backslash_underscore)
            payload = str(row["payload"]).replace("_", backslash_underscore)
            n_clients = int(row["n_clients"])
            bytes_str = _fmt_bytes(float(row["bytes_per_client_total"]))
            time_str = _fmt_time(float(row["t_seconds"]))
            overhead = float(row["crypto_overhead"])
            line = (
                f"{wf_label} & {mode} & {n_clients} & {payload} & "
                f"{bytes_str} & {time_str} & {overhead:.2f}x \\\\"
            )
            lines.append(line)
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    out_path.write_text("\n".join(lines))
    print(f"LaTeX table fragment written to {out_path}")


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
        knn_smpc = knn[knn["mode"] == "smpc"]
        if not knn_smpc.empty:
            ov = float(knn_smpc["crypto_overhead"].median())
            byt = float(knn_smpc["bytes_per_client_total"].median())
            macros.append(_safe("knnSmpcOverheadMedian", f"{ov:.1f}"))
            macros.append(_safe("knnSmpcBytesMedian", _fmt_bytes(byt)))

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
