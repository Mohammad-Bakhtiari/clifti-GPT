# Communication-cost analysis

This package contains the scripts used to quantify the communication
cost of the federated Clifti-GPT workflows (fine-tuning, federated KNN
reference mapping, federated binning) under plaintext and CrypTen-based
SMPC modes. All generated artifacts — CSV, figures, LaTeX fragments, and
the supplement PDF — land in `output/comm_cost/`.

## Scope

For each workflow and mode the analysis reports:

- Analytical per-client and per-federation bandwidth (bytes).
- Median wall-clock for a representative kernel under CrypTen running
  as a true multi-process simulated MPC.
- The ratio of SMPC to plaintext wall-clock (crypto overhead).

## Clients vs. parties

The analysis treats two parameters as independent:

- `C = n_clients`: number of federated data-holding clients. Drives
  federation totals and is swept per workflow.
- `P = n_parties`: number of SMPC computational parties. Fixed across
  a run. Drives the additive-sharing share factor `(P - 1)` and
  determines CrypTen's world size at runtime.

Plaintext federation costs scale with `C` only. SMPC per-client costs
scale with `(P - 1)` and are independent of `C`; federation totals
still scale with `C`.

## Files

| File | Purpose |
|---|---|
| `comm_cost.py` | Main benchmark. Writes CSV and metadata under `output/comm_cost/`. |
| `regenerate_comm_cost_at_c.py` | Recomputes analytical bytes for a fixed `P` without CrypTen. Interpolates wall-clock from a prior CSV. |
| `plot_comm_cost.py` | Renders PNG figures into `output/comm_cost/`. |
| `render_comm_cost_tex.py` | Reads CSV; writes LaTeX tables, macros, and syncs `communication_cost.tex` into `output/comm_cost/`. |
| `communication_cost.tex` | Supplement source (edited here; synced to output before `pdflatex`). |
| `Makefile` | `make pdf` runs the full LaTeX build in `output/comm_cost/`. |

## Setting the number of parties

`P` is resolved in this order:

1. `--n_parties` on the CLI.
2. Environment variable `COMM_COST_N_PARTIES`.
3. Default `3`.

Examples:

```bash
python analysis/comm_cost/comm_cost.py --n_parties 3
export COMM_COST_N_PARTIES=3 && python analysis/comm_cost/comm_cost.py
```

SMPC wall-clock is timed under `crypten.mpc.run_multiprocess(P)`, which
spawns `P` synchronized worker processes on a single host and sets
CrypTen's `WORLD_SIZE`, `RANK`, and rendezvous variables inside each
child. Each worker asserts `comm.get().get_world_size() == P`.

## Running the full pipeline

CrypTen and a CUDA-capable PyTorch are required for the wall-clock
benchmark. From the repository root:

```bash
export COMM_COST_N_PARTIES=3

python analysis/comm_cost/comm_cost.py
python analysis/comm_cost/plot_comm_cost.py
python analysis/comm_cost/render_comm_cost_tex.py
```

**Runtime:** The default sweep times SMPC on up to 50M parameters with
`n_reps=5` and `P=3` CrypTen processes. On CPU this can take many hours
before the first `[FT]` line appears, because progress is only printed
after each SMPC block finishes. Use `--quick` to verify the pipeline in
minutes:

```bash
python analysis/comm_cost/comm_cost.py --quick
```

Analytical byte counts in the CSV do not depend on wall-clock; you can
also recompute bytes without CrypTen via `regenerate_comm_cost_at_c.py`
and keep timings from an earlier full run.

Or build the supplement PDF in one step:

```bash
make -C analysis/comm_cost pdf
```

To refresh bytes for a different `P` without re-running CrypTen:

```bash
python analysis/comm_cost/regenerate_comm_cost_at_c.py --n_parties 3 --backup
```

## Outputs

**`output/comm_cost/`** (all generated artifacts):

- `comm_cost_results.csv` — one row per `(workflow, mode, n_clients, n_parties, payload)`.
- `comm_cost_metadata.json` — CLI arguments, resolved `n_parties`, caveats.
- `comm_cost_*.png` — bandwidth scaling and wall-clock panels.
- `communication_cost_*.tex` — workflow tables, wrapper, headline macros
  (`\smpcParties`, `\ftBytesRatioMax`, ...), copy-paste bundle.
- `communication_cost.pdf` — supplementary PDF (after `make pdf`).

## Caveats

- Wall-clock is measured under CrypTen's single-host multi-process
  simulator. It captures encryption, fixed-point, and protocol cost,
  but excludes real inter-site network latency.
- Beaver-triple precomputation is assumed offline (CrypTen default)
  and is not included in the SMPC wall-clock.
- Analytical formulas assume CrypTen additive sharing with `float32`
  payloads. A different sharing scheme would require updating the
  formulas and the supplement prose.
- `regenerate_comm_cost_at_c.py` is bytes-exact for the chosen `P` but
  inherits wall-clock values from the source CSV. Re-run `comm_cost.py`
  for measured timings.
