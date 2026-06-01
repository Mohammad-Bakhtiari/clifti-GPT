# Per-dataset initial model weights

This folder holds **initial PyTorch checkpoints** (`.pth` files) used to reproduce Clifti-GPT experiments. Each file is a `state_dict` saved **after loading the scGPT whole-human foundation model and before any dataset-specific fine-tuning**.

They are **not** fine-tuned Clifti-GPT models and **not** the original scGPT pretrained weights.

## Download

Download the init-weight archive from Zenodo:

- **DOI:** [10.5281/zenodo.20489646](https://doi.org/10.5281/zenodo.20489646)
- **Record:** https://zenodo.org/records/20489646

Extract every `.pth` file directly into this directory:

```
models/init/
├── ms.pth
├── hp5.pth
├── lung.pth
├── cl.pth
├── covid.pth
├── covid-corrected.pth
├── myeloid-top5.pth
└── …
```

Filenames must match the table below — experiment scripts resolve paths as `models/init/<dataset_slug>.pth`.

## Original scGPT weights (required separately)

These init files are **derived from** the scGPT whole-human checkpoint. Download the upstream model from the official repository:

- **Code & checkpoint links:** [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)
- **Publication:** Cui, H. et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods* **21**, 1470–1480 (2024). [https://www.nature.com/articles/s41592-024-02201-0](https://www.nature.com/articles/s41592-024-02201-0)

Place the scGPT files in:

```
models/pretrained_models/scGPT_human/
├── best_model.pt
├── vocab.json
└── args.json
```

If an init weight is missing locally, Clifti-GPT can create it on first run by loading the scGPT checkpoint and saving to this folder — using the Zenodo bundle avoids that extra step and fixes the starting weights across machines.

## Files

| File | Cohort | Used by experiment key |
|------|--------|-------------------------|
| `ms.pth` | Multiple Sclerosis | `MS` |
| `hp5.pth` | Human Pancreas | `HP5` |
| `lung.pth` | Lung-Kim | `LUNG` |
| `cl.pth` | Cell line | `CellLine` |
| `covid.pth` | COVID-19 (uncorrected) | `COVID` |
| `covid-corrected.pth` | COVID-19 (centrally corrected) | `COVID-corrected` |
| `myeloid-top5.pth` | Myeloid Top5 clients | `MYELOID-top5` |
| `myeloid-top10.pth` | Myeloid Top10 clients | `MYELOID-top10` |
| `myeloid-top20.pth` | Myeloid Top20 clients | `MYELOID-top20` |
| `myeloid-top30.pth` | Myeloid Top30 clients | `MYELOID-top30` |

## Usage in experiments

Scripts pass `--init_weights_dir models/init/<slug>.pth` via `experiments/annotation.sh` and `experiments/embedding.sh`. The slug matches the benchmark folder name under `data/scgpt/benchmark/` (e.g. `ms`, `hp5`, `lung`).

See [`experiments/README.md`](../../experiments/README.md) for how to run training after both benchmark data and weights are in place.
