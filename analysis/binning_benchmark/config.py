"""Dataset registry and binning strategy names."""

from typing import Dict, Tuple

PRIMARY_DATASETS: Dict[str, Dict[str, str]] = {
    "MS": {
        "slug": "ms",
        "reference_file": "reference_annot.h5ad",
        "batch_key": "split_label",
    },
    "CellLine": {
        "slug": "cl",
        "reference_file": "reference.h5ad",
        "batch_key": "batch",
    },
    "LUNG": {
        "slug": "lung",
        "reference_file": "reference_annot.h5ad",
        "batch_key": "sample",
    },
    "MYELOID-top5": {
        "slug": "myeloid-top5",
        "reference_file": "reference.h5ad",
        "batch_key": "combined_batch",
    },
    "HP5": {
        "slug": "hp5",
        "reference_file": "reference.h5ad",
        "batch_key": "batch_name",
    },
}

FEDERATED_STRATEGIES: Tuple[str, ...] = (
    "fed-weight-avg",
    "fed-weight-avg-smpc",
    "fed-hist",
    "fed-hist-smpc",
)
BINNING_STRATEGIES: Tuple[str, ...] = ("centralized",) + FEDERATED_STRATEGIES

METRIC_PREFIX = {
    "centralized": "centralized",
    "fed-weight-avg": "weighted",
    "fed-weight-avg-smpc": "weighted_smpc",
    "fed-hist": "hist",
    "fed-hist-smpc": "hist_smpc",
}

BATCH_METRICS: Tuple[str, ...] = ("cramers_v", "js_binned", "js_amplification")

DATASET_SLUG_TO_DISPLAY = {
    "ms": "MS",
    "cl": "CellLine",
    "lung": "LUNG",
    "myeloid-top5": "MYELOID-top5",
    "hp5": "HP5",
}

WIDE_METRIC_PREFIX = {
    "centralized": "centralized",
    "weighted": "fed-weight-avg",
    "weighted_smpc": "fed-weight-avg-smpc",
    "hist": "fed-hist",
    "hist_smpc": "fed-hist-smpc",
}
