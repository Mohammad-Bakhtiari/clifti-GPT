# Conda environments

| File | Conda env name | Used for |
|------|----------------|----------|
| [`annotation.yml`](annotation.yml) | `annotation` | Fine-tuning and annotation (`experiments/annotation.sh`, `tasks/annotation.py`) |
| [`embedding.yml`](embedding.yml) | `embedding` | Reference mapping / embedding (`experiments/embedding.sh`, `tasks/embedding.py`) |
| [`batch_correction.yml`](batch_correction.yml) | `batch_correction` | Optional COVID batch-effect correction (`data/correct_batch_effect.sh`) |

## Create environments

```bash
conda env create -f environment/annotation.yml
conda env create -f environment/embedding.yml
conda env create -f environment/batch_correction.yml   # optional
```

Update after pulling changes:

```bash
conda env update -n annotation -f environment/annotation.yml --prune
conda env update -n embedding -f environment/embedding.yml --prune
```

## Hardware and CUDA (tested setup)

Reproduction experiments for **annotation** and **embedding** were run on Linux with **4× NVIDIA Tesla T4** GPUs (driver **560.35.05**), using PyTorch **2.1.0+cu121** (CUDA toolkit **12.1**). Details are in the root [`README.md`](../README.md#installation).
