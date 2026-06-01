# Reproducing Clifti-GPT experiments

This folder contains the shell entry points used to rerun the training and evaluation workflows behind the Clifti-GPT supplementary figures (annotation, scalability, reference mapping, hyperparameter search, and federated binning comparisons).

Benchmark datasets, file paths, and observation keys are registered in `configs.sh`. The higher-level scripts read that registry and launch the Python tasks under `tasks/`.

## Quick start: run everything

From the repository root, use a GPU machine with benchmark data and the pretrained scGPT checkpoint in place (`data/scgpt/benchmark/`, `models/pretrained_models/scGPT_human/`, client init weights under `models/init/`).

```bash
cd experiments
chmod +x run-all.sh
./run-all.sh
```

`run-all.sh` is the full reproduction driver. It runs the experiment blocks below in sequence and writes results under `output/annotation/` and `output/embedding/`. Annotation runs also append to `output/annotation/results_summary.csv`, which feeds the analysis scripts in `analysis/`.

The full pipeline is long and GPU-intensive. Edit `GPU=` at the top of `run-all.sh` if needed, or run individual scripts instead of the full bundle.

## Shell scripts

| Script | Role |
|--------|------|
| **`run-all.sh`** | End-to-end reproduction: parameter tuning, scalability runs, main annotation benchmarks, reference-mapping embeddings, and binning `prep_mode` comparisons. |
| **`run_param_tuning.sh`** | Grid search over local epochs (and FedProx `mu` when applicable) for federated annotation; populates `output/annotation/results_summary.csv` used to pick best hyperparameters. |
| **`run_annotation.sh`** | Batch driver over one or more datasets from `configs.sh` for a given annotation mode (centralized, federated, local clients, etc.). |
| **`annotation.sh`** | Runs a single annotation job via `tasks/annotation.py` for one dataset configuration. Called by `run_annotation.sh` and `run_param_tuning.sh`. |
| **`run_embedding.sh`** | Batch driver for reference-mapping / zero-shot embedding experiments across datasets. |
| **`embedding.sh`** | Runs a single embedding job via `tasks/embedding.py`. Called by `run_embedding.sh`. |
| **`run_prep_mode_comparison.sh`** | Replays the best tuned federated settings on the five scGPT benchmark datasets under each binning `prep_mode` (centralized, federated weighted-average, SMPC variants, histogram binning). Used for the federated binning benchmark accuracy panel. |
| **`configs.sh`** | Shared dataset registry and helper to resolve dataset names; sourced by the batch scripts above. |

## What `run-all.sh` covers

In order, it launches:

1. **Hyperparameter tuning** — FedAvg and FedProx federated annotation with and without SMPC on selected datasets.
2. **Scalability (Myeloid)** — Centralized baseline, per-client local training, and FedProx-SMPC across Top5–Top30 client partitions.
3. **Main annotation benchmark** — Centralized scGPT, local client models, and federated Clifti-GPT (FedAvg / FedProx, plain and SMPC) on the standard and COVID-related datasets.
4. **Reference mapping** — Centralized, federated zero-shot, federated SMPC, and per-client local embedding modes.
5. **Binning prep-mode comparison** — Best-config federated runs under all five `prep_mode` strategies: `centralized`, `fed-weight-avg`, `fed-weight-avg-smpc`, `fed-hist`, `fed-hist-smpc`.

Together, these runs produce the metrics and outputs that the paper figures and tables are built from. Post-processing (plots, communication-cost tables, binning benchmark figures) lives in `analysis/` and is run separately after the training jobs finish.

## Outputs

- **Annotation:** `output/annotation/<dataset>/<mode>/` per run; aggregated metrics in `output/annotation/results_summary.csv`.
- **Embedding / reference mapping:** `output/embedding/<dataset>/<mode>/`.
- **Binning benchmark (analysis step):** after prep-mode runs complete, use `analysis/binning_benchmark/` (see that folder’s README).
