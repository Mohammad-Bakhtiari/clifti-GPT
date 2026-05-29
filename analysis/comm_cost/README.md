# Communication-cost analysis

Benchmarks communication cost for federated fine-tuning, reference mapping (KNN),
and binning. Produces GPU SMPC wall-clock timings in `output/comm_cost/`.
Analytical byte formulas are computed internally for reference but are **not**
included in generated figures or LaTeX tables.

## Parameters

Two independent counts:

- **C (`n_clients`)** — federated data-holding clients. Swept per workflow; scales
  federation totals.
- **P (`n_parties`)** — SMPC parties. Fixed per run (default **3**). Share factor
  in the byte model is **(P − 1)**; sets CrypTen `WORLD_SIZE` for wall-clock.

**P** is set via `--n_parties`, env `COMM_COST_N_PARTIES`, or default `3`.

### Default sweeps

| Workflow | Swept parameters |
|---|---|
| Fine-tuning | `\|θ\|` ∈ {1M, 10M}, C ∈ {2, 3, 5}, R = 5 rounds (bytes in CSV only) |
| Reference mapping | n_q ∈ {500, 2000}, n_r ∈ {1000, 5000, 10000}, C ∈ {2, 5}, k ∈ {5, 10}, d = 128 |
| Binning | C ∈ {2, 5, 10}, n_bins = 51, grid M = 4096 |

Wall-clock uses **5** timed repetitions per config (median reported) unless `--quick`.

Each completed configuration is appended to `comm_cost_results.csv` immediately (with `fsync`), so a crash preserves all finished rows. **Re-running truncates the CSV** and starts fresh in that output directory.

## Run

From the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export COMM_COST_N_PARTIES=3

python analysis/comm_cost/comm_cost.py
python analysis/comm_cost/plot_comm_cost.py
python analysis/comm_cost/render_comm_cost_tex.py
```

SMPC wall-clock runs in an isolated subprocess per config. By default
(`--smpc-one-gpu-per-party`, on), CrypTen party rank `r` uses `cuda:r`, so set
`CUDA_VISIBLE_DEVICES` to **P** GPUs (e.g. `0,1,2` for P=3). If
`CUDA_VISIBLE_DEVICES` is unset, the child subprocess uses `0,1,...,P-1`
automatically.

To force all parties onto one GPU (legacy behavior):

```bash
export CUDA_VISIBLE_DEVICES=0
python analysis/comm_cost/comm_cost.py --no-smpc-one-gpu-per-party
```

### `--quick`

Smoke test: one small config per workflow, `n_reps=1` (e.g. |θ| = 10k, C = 2).
Use to verify the pipeline before the full sweep.

```bash
python analysis/comm_cost/comm_cost.py --quick
```

### Resume / add a workflow without rerunning others

`comm_cost.py` always **truncates** the CSV in `--output_dir` at startup. To add
binning to an existing CSV:

```bash
# backup
cp output/comm_cost/comm_cost_results.csv output/comm_cost/comm_cost_results.bak.csv

# run only the missing workflow in a separate directory
python analysis/comm_cost/comm_cost.py \
  --workflows binning \
  --output_dir output/comm_cost/binning_only

# append new rows (skip the duplicate header)
tail -n +2 output/comm_cost/binning_only/comm_cost_results.csv \
  >> output/comm_cost/comm_cost_results.csv
```

Then regenerate plots and tables from the merged CSV.

### Plotting note (KNN scaling panel)

`plot_comm_cost.py` plots wall-clock vs n_r at the modal n_q and C. If the CSV
contains multiple k values, filter or use a single k for a clean line plot (e.g.
k = 5); the full table lists all k.

## Files

| File | Role |
|---|---|
| `comm_cost.py` | Main benchmark → CSV + metadata |
| `plot_comm_cost.py` | Figures from CSV (wall-clock only) |
| `render_comm_cost_tex.py` | LaTeX table fragments and macros from CSV |
| `regenerate_comm_cost_at_c.py` | Recompute analytical bytes for a new P |

## Outputs

`output/comm_cost/comm_cost_results.csv` (appended after each configuration),
`comm_cost_metadata.json`, `comm_cost_*.png` (wall-clock only),
`communication_cost_table.tex`, `communication_cost_table_*.tex`, `communication_cost_macros.tex`
