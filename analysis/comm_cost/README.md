# Communication-cost analysis

Benchmarks communication cost for federated fine-tuning, reference mapping (KNN),
and binning. Produces analytical bandwidth (bytes) and GPU SMPC wall-clock timings
in `output/comm_cost/`.

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
| Fine-tuning | `\|θ\|` ∈ {1M, 10M}, C ∈ {2, 3, 5}, R = 5 rounds (bytes only) |
| Reference mapping | n_q ∈ {500, 2000}, n_r ∈ {1000, 5000}, C ∈ {2, 5}, k ∈ {5, 20}, d = 128 |
| Binning | C ∈ {2, 5, 10}, n_bins = 51, grid M = 4096 |

Wall-clock uses **5** timed repetitions per config (median reported) unless `--quick`.

## Run

From the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export COMM_COST_N_PARTIES=3

python analysis/comm_cost/comm_cost.py
python analysis/comm_cost/plot_comm_cost.py
python analysis/comm_cost/render_comm_cost_tex.py
```

SMPC wall-clock runs in an isolated subprocess per config. Use one visible GPU
(`CUDA_VISIBLE_DEVICES=0`); all P parties share it.

### `--quick`

Smoke test: one small config per workflow, `n_reps=1` (e.g. |θ| = 10k, C = 2).
Use to verify the pipeline before the full sweep.

```bash
python analysis/comm_cost/comm_cost.py --quick
```

## Files

| File | Role |
|---|---|
| `comm_cost.py` | Main benchmark → CSV + metadata |
| `plot_comm_cost.py` | Figures from CSV |
| `render_comm_cost_tex.py` | LaTeX table fragments and macros from CSV |
| `regenerate_comm_cost_at_c.py` | Recompute analytical bytes for a new P |

## Outputs

`output/comm_cost/comm_cost_results.csv`, `comm_cost_metadata.json`, `comm_cost_*.png`,
`communication_cost_table.tex`, `communication_cost_table_*.tex`, `communication_cost_macros.tex`
