# Binning benchmark

Reproducing the binning benchmark requires **two separate steps in two different places**:

1. **`experiments/`** — train Clifti-GPT under each binning strategy and record annotation accuracy.
2. **`analysis/binning_benchmark/`** (this folder) — compute binning comparison metrics on the benchmark h5ad files and generate the figures.

Neither step replaces the other. Training produces downstream accuracy; the scripts here produce binning metrics and combine both into the final plots.

Both steps use the same five benchmark datasets and five `prep_mode` strategies (see below).

---

## 1. Training (`experiments/`)

From the `experiments/` directory, run once per strategy:

```bash
cd experiments
./run_prep_mode_comparison.sh centralized
./run_prep_mode_comparison.sh fed-weight-avg
./run_prep_mode_comparison.sh fed-weight-avg-smpc
./run_prep_mode_comparison.sh fed-hist
./run_prep_mode_comparison.sh fed-hist-smpc
```

Each command fine-tunes Clifti-GPT on all five benchmark datasets with that binning strategy: federated binning during preprocessing, then federated annotation fine-tuning. Dataset-specific hyperparameters (epochs, rounds, aggregation) are fixed inside the script.

**Output:** `output/annotation/results_summary.csv` — per-run metrics including cell-type Accuracy, keyed by dataset and `prep_mode`.

---

## 2. Analysis (`analysis/binning_benchmark/`)

From the repository root:

```bash
python analysis/binning_benchmark/benchmark.py
python analysis/binning_benchmark/plot.py
```

`benchmark.py` loads each benchmark h5ad, partitions cells into federated clients, computes global bin edges under all five strategies, and writes comparison metrics (Cramér's V, Jensen–Shannon divergence among client bin histograms, JS amplification).

`plot.py` reads those metrics and the training accuracies from `results_summary.csv` to produce the benchmark figures. Run `benchmark.py` before `plot.py`.

**Outputs:**

| Path | Contents |
|---|---|
| `output/binning_benchmark/results_wide.csv` | Binning metrics, one row per dataset |
| `output/binning_benchmark/results.csv` | Same data in long format |
| `output/binning_benchmark/summary.md` | Readable summary |
| `output/binning_benchmark/figures/binning_benchmark_cramers_v.png` | Cramér's V by dataset and strategy |
| `output/binning_benchmark/figures/binning_benchmark_js_amplification.png` | JS amplification by dataset and strategy |
| `output/binning_benchmark/figures/binning_benchmark_accuracy.png` | Peak annotation accuracy by dataset and strategy |
| `output/binning_benchmark/figures/binning_benchmark_legend.png` | Shared legend |

---

## Datasets

| Dataset | Path under `data/scgpt/benchmark/` | Client partition (`obs` column) |
|---|---|---|
| MS | `ms/reference_annot.h5ad` | `split_label` |
| CellLine | `cl/reference.h5ad` | `batch` |
| LUNG | `lung/reference_annot.h5ad` | `sample` |
| MYELOID-top5 | `myeloid-top5/reference.h5ad` | `combined_batch` |
| HP5 | `hp5/reference.h5ad` | `batch_name` |

## Strategies

| `prep_mode` | Description |
|---|---|
| `centralized` | Pooled quantiles on all non-zero expression values |
| `fed-weight-avg` | Plaintext weighted average of local quantile edges |
| `fed-weight-avg-smpc` | SMPC aggregation of local edge contributions |
| `fed-hist` | Plaintext histogram-based aggregation |
| `fed-hist-smpc` | SMPC histogram aggregation |

