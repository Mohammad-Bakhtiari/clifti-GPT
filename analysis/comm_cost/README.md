# Communication-cost analysis

Benchmarks communication cost for federated fine-tuning, reference mapping (KNN),
and binning. Produces **GPU wall-clock** timings for both plaintext and SMPC in
`output/comm_cost/`. Analytical byte formulas are computed internally for reference
but are **not** included in generated figures or LaTeX tables.

## Parameters

Two independent counts:

- **C (`n_clients`)** — federated data-holding clients. Swept per workflow; scales
  federation totals.
- **P (`n_parties`)** — SMPC parties. Fixed per run (default **3**). Sets CrypTen
  `WORLD_SIZE` for wall-clock.

**P** is set via `--n_parties`, env `COMM_COST_N_PARTIES`, or default `3`.

### Timing methodology

- **Plaintext** baselines run on **`cuda:0`** (Torch GPU kernels).
- **SMPC** runs on GPU with **P** parties (`cuda:0..P-1` when
  `--smpc-one-gpu-per-party` is on).
- KNN plaintext uses the same GPU distance / top-\(k\) structure as the SMPC path
  (not CPU FAISS).
- Fine-tuning SMPC applies sample ratios client-side before encryption, matching
  `Client.get_local_updates` and the weighted plaintext FedAvg path.
- Overhead ratios \(t_{\mathrm{SMPC}} / t_{\mathrm{plain}}\) therefore reflect
  cryptographic cost on comparable hardware, not CPU vs GPU.
- Timings exclude real inter-site network latency.

### Default sweeps

Running `comm_cost.py` with no workflow-specific flags executes all three workflows
below. Fine-tuning automatically includes the real checkpoint size from
`models/init/hp5.pth` when that file is present (override path with env
`COMM_COST_INIT_WEIGHTS`).

| Workflow | Swept parameters |
|---|---|
| Fine-tuning | `\|θ\|` ∈ {hp5 checkpoint, 1M, 10M}, C ∈ {2, 3, 5}, R = 5 (bytes in CSV only) |
| Reference mapping | n_q ∈ {500, 2000}, n_r ∈ {1000, 5000, 10000}, C ∈ {2, 5}, k ∈ {5, 10}, d = 128 |
| Binning | C ∈ {2, 5, 10}, n_bins = 51, grid M = 4096 |

Wall-clock uses **5** timed repetitions per config (median reported) unless `--quick`.

Each completed configuration is appended to `comm_cost_results.csv` immediately
(with `fsync`). Re-running truncates the CSV in `--output_dir` and starts fresh.

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
`CUDA_VISIBLE_DEVICES` to **P** GPUs (e.g. `0,1,2` for P=3). Plaintext uses
`cuda:0` (the first visible GPU).

### `--quick`

Smoke test: one small config per workflow, `n_reps=1`.

```bash
python analysis/comm_cost/comm_cost.py --quick
```

### Subset of workflows

```bash
python analysis/comm_cost/comm_cost.py --workflows binning
python analysis/comm_cost/comm_cost.py --workflows reference_mapping
```

## Files

| File | Role |
|---|---|
| `comm_cost.py` | Main benchmark → CSV + metadata |
| `plot_comm_cost.py` | Figures from CSV (wall-clock only) |
| `render_comm_cost_tex.py` | LaTeX table fragments and macros from CSV |
| `regenerate_comm_cost_at_c.py` | Recompute analytical bytes for a new P |

## Outputs

`output/comm_cost/comm_cost_results.csv`, `comm_cost_metadata.json`,
`comm_cost_*.png`, `communication_cost_table.tex`,
`communication_cost_table_*.tex`, `communication_cost_macros.tex`
