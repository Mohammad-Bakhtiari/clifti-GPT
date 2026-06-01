# Data and model assets

Clifti-GPT experiments require three types of files that are **not** shipped in this repository:

| Asset | Location | Documentation |
|-------|----------|---------------|
| Benchmark AnnData (reference + query h5ad) | `data/scgpt/benchmark/` | [10.5281/zenodo.20491148](https://doi.org/10.5281/zenodo.20491148) · [`scgpt/benchmark/README.md`](scgpt/benchmark/README.md) |
| Per-dataset init weights | `models/init/` | [10.5281/zenodo.20489646](https://doi.org/10.5281/zenodo.20489646) · [`../models/init/README.md`](../models/init/README.md) |
| scGPT whole-human checkpoint | `models/pretrained_models/scGPT_human/` | [scGPT GitHub](https://github.com/bowang-lab/scGPT) · [Nature Methods (2024)](https://www.nature.com/articles/s41592-024-02201-0) |

Download benchmark h5ad from [10.5281/zenodo.20491148](https://doi.org/10.5281/zenodo.20491148) and init weights from [10.5281/zenodo.20489646](https://doi.org/10.5281/zenodo.20489646), then extract into the paths above.

## After Zenodo download

The Zenodo bundle ships six cohort archives only. Two experiment families need an extra local step before you can run them.

### Myeloid scalability (required for `MYELOID-top5` … `MYELOID-top30`)

Base `myeloid/` is included; Top5–Top30 splits are not. **Run** `myeloid_prep.py` before myeloid scalability experiments:

```bash
python data/myeloid_prep.py
```

### COVID corrected (required for `COVID-corrected`)

Uncorrected `covid/` is included; the centrally corrected split is not. **Run** batch correction before `COVID-corrected` experiments. This needs upstream `Covid.h5ad` under `covid/` (not in the Zenodo bundle) and [fedscGen](https://github.com/HelmholtzAI/fedscGen):

```bash
bash data/correct_batch_effect.sh data/scgpt/benchmark
```

This writes `covid-corrected/`. It does **not** rebuild from `reference-raw.h5ad` / `query-raw.h5ad` alone. Helper: `prep_batch_effect_correction.py` (called by the script above).

## Prep scripts (build from raw sources)

Use these only if you rebuild benchmarks from upstream public cohorts instead of the Zenodo bundle:

| Script | Purpose |
|--------|---------|
| `prep.sh` | Reference/query splits for CellLine, LUNG, COVID |
| `ms-prep.py` | MS annotation labels and ref/query split |
| `myeloid_prep.py` | Myeloid Top5–Top30 from base `myeloid/` ref/query |
| `correct_batch_effect.sh` | COVID central batch correction |
| `prep_batch_effect_correction.py` | COVID corrected split helper (used by `correct_batch_effect.sh`) |

