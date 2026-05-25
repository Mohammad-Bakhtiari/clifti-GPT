import random
import numpy as np
import torch
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import crypten
from crypten.config import cfg
from collections import OrderedDict
import hashlib
import gc
import fcntl

SEED = 42
def set_seed(seed=None):
    """Sets the seed for reproducibility. Checks for an environment variable first, then defaults to SEED."""
    global SEED
    if seed is None:
        seed = SEED
    else:
        SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    cfg.encoder.precision_bits = 32
    crypten.init()
    cfg.debug.debug_mode = True
    crypten.manual_seed(seed, seed, seed)
    print(f"✅ Seed set to {seed} (from `set_seed()`)")

import os
import anndata
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import scgpt as scg
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
import dataclasses
import yaml
import scanpy as sc
import pickle
import matplotlib.pyplot as plt
import logging
import sys




def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed + SEED)
    random.seed(worker_seed + SEED)
    torch.manual_seed(worker_seed + SEED)

BASE_CLIENT_LEVEL_NUM = 35


@dataclasses.dataclass
class ADVTrainConfig:
    E_delay_epochs: int
    D_delay_epochs: int
    lr: float


@dataclasses.dataclass
class TrainConfig:
    dab_weight: float
    lr: float
    batch_size: int
    eval_batch_size: int
    epochs: int
    schedule_ratio: float
    schedule_interval: int
    amp: bool
    save_eval_interval: int
    MLM: bool
    CLS: bool
    ADV: bool or ADVTrainConfig
    CCE: bool
    MVC: bool
    ecs_thres: float
    DAB: bool
    INPUT_BATCH_LABELS: bool
    mvc_decoder_style: str
    freeze: bool
    do_sample_in_train: bool
    DSBN: bool
    ECS: bool = dataclasses.field(init=False)

    def __post_init__(self):
        self.ECS = self.ecs_thres > 0


@dataclasses.dataclass
class ModelConfig:
    embsize: int
    nhead: int
    d_hid: int
    nlayers: int
    nlayers_cls: int
    n_cls: int
    dropout: float
    do_mvc: bool
    do_dab: bool
    use_batch_labels: bool
    domain_spec_batchnorm: bool
    input_emb_style: str
    n_input_bins: int
    cell_emb_style: str
    mvc_decoder_style: str
    ecs_threshold: float
    explicit_zero_prob: bool
    use_fast_transformer: bool
    fast_transformer_backend: str
    pre_norm: bool


@dataclasses.dataclass
class PreprocessConfig:
    n_bins: int
    pre_norm: bool
    include_zero_gene: bool
    input_style: str
    output_style: str
    input_emb_style: str
    mask_ratio: float
    cell_emb_style: str
    pad_token: str
    special_tokens: list
    mask_value: str or int
    max_seq_len: int
    per_seq_batch_sample: bool
    pad_value: int = None


@dataclasses.dataclass
class DatasetConfig:
    raw_data_key: str = "X"
    data_is_raw: bool = False
    filter_gene_by_counts: bool = False
    filter_cell_by_counts: bool = False
    normalize_total: int = 1e4  # 3. whether to normalize the raw data and to what sum
    result_normed_key: str = "X_normed"  # the key in adata.layers to store the normalized data
    log1p: bool = False  # 4. whether to log1p the normalized data
    result_log1p_key: str = "X_log1p"
    subset_hvg: bool = False  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor: str = "cell_ranger"
    result_binned_key: str = "X_binned"

    def __post_init__(self):
        if isinstance(self.normalize_total, str):
            self.normalize_total = float(self.normalize_total)


@dataclasses.dataclass
class PreprocessConfig:
    n_bins: int
    pre_norm: bool
    include_zero_gene: bool
    input_style: str
    output_style: str
    input_emb_style: str
    mask_ratio: float
    cell_emb_style: str
    pad_token: str
    special_tokens: list
    mask_value: str or int
    max_seq_len: int
    per_seq_batch_sample: bool
    pad_value: int = None


@dataclasses.dataclass
class LogConfig:
    log_interval: int
    save_eval_interval: int
    do_eval_scib_metrics: bool
    retain_best_model: bool


@dataclasses.dataclass
class Config:
    preprocess: PreprocessConfig
    train: TrainConfig
    model: ModelConfig
    dataset: DatasetConfig
    log: LogConfig


@dataclasses.dataclass
class FedPreprocessConfig:
    filter_gene_by_counts: bool
    filter_cell_by_counts: bool
    normalize_total: bool
    log1p: bool
    subset_hvg: bool
    binning: bool


@dataclasses.dataclass
class FedConfig:
    n_rounds: int
    aggregation_type: str
    condition_key: str
    preprocess: FedPreprocessConfig


def load_config(file_path: str, task, verbose) -> Config:
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    if verbose:
        print_config(config_dict)
    preprocess_config = PreprocessConfig(**config_dict[task]['preprocess'])
    train_config = TrainConfig(**config_dict[task]['train'])
    model_config = ModelConfig(**config_dict[task]['model'])
    log_config = LogConfig(**config_dict[task]['log'])
    dataset_config = DatasetConfig(**config_dict[task]['dataset'])
    return Config(preprocess=preprocess_config,
                  train=train_config,
                  model=model_config,
                  dataset=dataset_config,
                  log=log_config
                  )


def load_fed_config(file_path: str, task: str) -> FedConfig:
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    print("Federated Config")
    print_config(config_dict)
    preprocess_config = FedPreprocessConfig(**config_dict[task]['preprocess'])
    config_dict[task].pop('preprocess')
    return FedConfig(**config_dict[task], preprocess=preprocess_config)


def print_config(config: dict or tuple, level=0):
    for k, v in config.items():
        if isinstance(v, dict):
            print("  " * level + str(k) + ":")
            print_config(v, level + 1)
        else:
            print("  " * level + str(k) + ":", v)


def get_cuda_device(device_index: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda")
    else:
        return torch.device("cpu")


def read_h5ad(data_dir, adata):
    if os.path.isabs(adata):
        print(f"Reading data from {adata} ...")
        adata = anndata.read_h5ad(adata)
    else:
        print(f"Reading data from {data_dir}/{adata} ...")
        adata = anndata.read_h5ad(f"{data_dir}/{adata}")
    return adata


def add_federated_logging(logger):
    FEDERATED_LEVEL_NUM = 25
    logging.addLevelName(FEDERATED_LEVEL_NUM, "FEDERATED")

    def federated(self, message, *args, **kws):
        if self.isEnabledFor(FEDERATED_LEVEL_NUM):
            self._log(FEDERATED_LEVEL_NUM, message, args, **kws)

    setattr(logging.Logger, 'federated', federated)
    logger.setLevel(min(logger.level, FEDERATED_LEVEL_NUM))  # Ensure logger level includes the new custom level


def add_inference_logging(logger):
    INFERENCE_LEVEL_NUM = 26  # Using a different level number to avoid conflict
    logging.addLevelName(INFERENCE_LEVEL_NUM, "INFERENCE")

    def inference(self, message, *args, **kws):
        if self.isEnabledFor(INFERENCE_LEVEL_NUM):
            self._log(INFERENCE_LEVEL_NUM, message, args, **kws)

    setattr(logging.Logger, 'inference', inference)
    logger.setLevel(min(logger.level, INFERENCE_LEVEL_NUM))  # Ensure logger level includes the new custom level


def add_client_logging(logger, client_id, level_num):
    level_name = f"CLIENT_{client_id}"
    logging.addLevelName(level_num, level_name)

    def log_for_client(self, message, *args, **kws):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kws)

    setattr(logging.Logger, level_name, log_for_client)
    logger.setLevel(min(logger.level, level_num))  # Ensure logger level includes the new custom level

def get_logger(output_dir, logger_title="scGPT", client_ids=None, debug=False):
    assert logger_title in ["scGPT", "cliftiGPT"], f"Invalid logger title: {logger_title}"
    if client_ids is None:
        client_ids = []

    logger = logging.getLogger(logger_title)
    if not logger.hasHandlers() or len(logger.handlers) == 0:
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        h = logging.FileHandler(f"{output_dir}/run.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        h.setFormatter(formatter)
        h.setLevel(logger.level)
        logger.addHandler(h)

    # Add federated logging level if logger_title is "cliftiGPT"
    if logger_title == "cliftiGPT":
        add_federated_logging(logger)
        for idx, client_id in enumerate(client_ids):
            add_client_logging(logger, client_id, BASE_CLIENT_LEVEL_NUM + idx)

    add_inference_logging(logger)
    if debug:
        print_available_log_levels(logger)
    return logger


def print_available_log_levels(logger):
    """Print all log levels available for the given logger."""
    print(f"Logger: {logger.name}, Effective Level: {logging.getLevelName(logger.level)} ({logger.level})")
    print("Available Log Levels:")
    # Sort levels by numeric value
    levels = sorted(logging._levelToName.items(), key=lambda x: x[0])
    for level_num, level_name in levels:
        # Only include levels that are >= the logger's effective level
        if level_num >= logger.level:
            print(f"  {level_name} ({level_num})")

def per_epoch_data_prep(tokenized_train, tokenized_valid, train_celltype_labels, valid_celltype_labels,
                        train_batch_labels,
                        valid_batch_labels, mask_value, pad_value, mask_ratio, epoch, sort_seq_batch=False):
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader


def get_mixin_config(config):
    return {"lr": config["train"]["lr"],
            "schedule_interval": config["train"]["schedule_interval"],
            "schedule_ratio": config["train"]["schedule_ratio"],
            "amp": config["train"]["amp"],
            }


def model_config_adj(config):
    if config["preprocess"]["input_emb_style"] == "category":
        config['model']['mask_value'] = config["preprocess"]["n_bins"] + 1
        config['model']['pad_value'] = config["preprocess"]["n_bins"]  # for padding gene expr values
        config['model']['n_input_bins'] = config["preprocess"]["n_bins"] + 2
    else:
        config['model']['mask_value'] = -1
        config['model']['pad_value'] = -2
        config['model']['n_input_bins'] = config["preprocess"]["n_bins"]

def generate_palette(unique_celltypes):
    """
    Build a large palette:
     - First 20 colors from 'tab20'
     - If > 20, next up to 20 from 'tab20b'
     - If > 40, sample the remainder from 'gist_ncar'
    Args:
        unique_celltypes: list of category names
    Returns:
        dict mapping each category to an (r, g, b, a) tuple
    """
    n_cats = len(unique_celltypes)
    palette_ = {}

    if n_cats <= 20:
        base = plt.get_cmap("tab20", 20)
        for i, cat in enumerate(unique_celltypes):
            palette_[cat] = base(i)
    elif n_cats <= 40:
        base1 = plt.get_cmap("tab20", 20)
        base2 = plt.get_cmap("tab20b", 20)
        for i, cat in enumerate(unique_celltypes):
            if i < 20:
                palette_[cat] = base1(i)
            else:
                palette_[cat] = base2(i - 20)
    else:
        base1 = plt.get_cmap("tab20", 20)
        base2 = plt.get_cmap("tab20b", 20)
        # For the remainder, sample evenly from gist_ncar
        n_extra = n_cats - 40
        base3 = plt.get_cmap("gist_ncar", n_extra)
        for i, cat in enumerate(unique_celltypes):
            if i < 20:
                palette_[cat] = base1(i)
            elif i < 40:
                palette_[cat] = base2(i - 20)
            else:
                palette_[cat] = base3(i - 40)
    return palette_

def plot_umap(adata, cell_type_key, unique_celltypes, file_name, legend='no_legend'):
    palette_ = generate_palette(unique_celltypes)
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 10)  # Create a grid spec with 1 row and 10 columns

    # Use columns 0-4 for the first plot and 4-8 for the second plot
    ax1 = fig.add_subplot(gs[0, 0:5])  # First plot in columns 0-5
    ax2 = fig.add_subplot(gs[0, 5:10])  # Second plot in columns 5-10
    # Plot the UMAP for "celltype" and "predictions"
    for ax, color in zip([ax1, ax2], [cell_type_key, "predictions"]):
        if legend == 'no_legend':
            sc.pl.umap(adata, color=color, palette=palette_, ax=ax, show=False, legend_loc=None)
        elif legend == 'legend_only':
            sc.pl.umap(adata, color=color, palette=palette_, ax=ax, show=False)
        else:
            raise ValueError(f"Invalid value for legend: {legend}")

    if legend == 'no_legend':
        plt.tight_layout()
        plt.savefig(file_name, dpi=300)
    else:
        handles, labels = ax1.get_legend_handles_labels()

        # Plot the legend separately
        fig_legend, ax_legend = plt.subplots(figsize=(9, 3))  # Separate figure for the legend
        ax_legend.legend(handles, labels, loc='center', fontsize='small', frameon=False, ncol=2)
        ax_legend.axis('off')  # Hide the axis for the legend area

        # Save the legend separately
        plt.savefig(file_name, dpi=300)

    plt.close()


def plot(adata, celltype: list, celltype_key: str, save_dir: str):
    if "X_umap" not in adata.obsm.keys() or "X_pca" not in adata.obsm.keys():
        print("UMAP or PCA coordinates are not computed for the dataset. Calculating X_umap for 30 neighbors.")
        sc.pp.neighbors(adata, n_neighbors=30)
    plot_umap(adata, celltype_key, celltype, f"{save_dir}/umap_plots.png", legend='no_legend')
    plot_umap(adata, celltype_key, celltype, f"{save_dir}/legend.png", legend='legend_only')

def dump_results(predictions, labels, results, id2type, save_dir, epoch=None, n_rounds=None):
    save_dict = {
        "predictions": predictions,
        "labels": labels,
        "results": results,
        "id_maps": id2type,
    }
    if epoch is None or n_rounds is None:
        with open(f"{save_dir}/results.pkl", "wb") as f:
            pickle.dump(save_dict, f)
    else:
        save_path = f"{save_dir}/results.pkl"
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                all_results = pickle.load(f)
        else:
            all_results = {}


        # Ensure the dictionary structure exists
        if epoch not in all_results:
            all_results[epoch] = {}

        if n_rounds not in all_results[epoch]:
            all_results[epoch][n_rounds] = {}

        # Store the current result dictionary under the epoch and round
        all_results[epoch][n_rounds] = save_dict

        # Save the updated results back to the file
        with open(save_path, "wb") as f:
            pickle.dump(all_results, f)

        print(f"Results saved for epoch {epoch}, round {n_rounds} in {save_path}")


def confusion_matrix_evaluation(celltypes: list, predictions, labels, id2type, save_dir):
    for i in set([id2type[p] for p in predictions]):
        if i not in celltypes:
            celltypes.remove(i)
    cm = confusion_matrix(labels, predictions)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
    nan_rows = [ind for i, ind in enumerate(cm.index) if all(cm.iloc[i].isna())]
    cm.drop(index=nan_rows, inplace=True)
    cm.to_csv(f"{save_dir}/confusion_matrix.csv")
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300)


def weighted_average(state_dicts, n_samples):
    sample_ratios = [n / sum(n_samples) for n in n_samples]
    global_weights = {}
    for param in state_dicts[0].keys():
        global_weights[param] = torch.stack(
            [state_dicts[i][param] * sample_ratios[i] for i in range(len(state_dicts))]).sum(0)
    return global_weights


def average_weights(state_dicts):
    global_weights = {}
    for param in state_dicts[0].keys():
        global_weights[param] = torch.stack(
            [state_dicts[i][param] for i in range(len(state_dicts))]).mean(0)
    return global_weights


def validate_fed_code(centralized_adata, clients_adata):
    for adata in clients_adata:
        print(adata.shape)
    # Concatenate client data
    adata_fed = anndata.concat(clients_adata, join='outer', label='batch')

    # Ensure the data matrices have the same shape for comparison
    if centralized_adata.shape != adata_fed.shape:
        raise ValueError(
            f"Centralized and federated data shapes do not match. Centralized shape: {centralized_adata.shape}, Federated shape: {adata_fed.shape}")

    # Calculate the absolute difference between the centralized and federated results
    diff = np.abs(centralized_adata.X - adata_fed.X)

    # Calculate the mean, sum, min, and max differences
    mean_diff = np.mean(diff)
    sum_diff = np.sum(diff)
    min_diff = np.min(diff)
    max_diff = np.max(diff)

    # Print the differences
    print("Mean difference between centralized and federated results:", mean_diff)
    print("Sum of differences between centralized and federated results:", sum_diff)
    print("Minimum difference between centralized and federated results:", min_diff)
    print("Maximum difference between centralized and federated results:", max_diff)


def split_data_by_batch(adata, batch_key, keep_vars):
    original_categories = {k: adata.obs[k].cat.categories for k in adata.obs.keys() if adata.obs[k].dtype == "category"}
    unique_batch_ids = sorted(adata.obs[batch_key].unique())
    batches = {}
    for client, batch_id in enumerate(unique_batch_ids):
        batch_adata = adata[adata.obs[batch_key] == batch_id].copy()
        for k, v in original_categories.items():
            batch_adata.obs[k] = pd.Categorical(batch_adata.obs[k], categories=v)
        if keep_vars:
            batch_adata.var = adata.var.copy()
        batches[batch_id] = batch_adata
    return batches

def save_data_batches(batches: dict, data_dir: list or str, filename: str, keep_vars: bool = False):
    if type(data_dir) == str:
        data_dir = [f"{data_dir}/client_{i}"for i in batches.keys()]
    for client, batch_adata in enumerate(batches.values()):
        if not os.path.exists(data_dir[client]):
            print(f"{data_dir[client]} does not exist!")
            os.makedirs(data_dir[client], exist_ok=True)
        if "gene_name" in batch_adata.var.keys() and not keep_vars:
            batch_adata.var.drop(columns=["gene_name"], inplace=True)
        batch_adata.write_h5ad(f"{data_dir[client]}/{filename}")
    return data_dir


def compare_models(model1, model2):
    if isinstance(model1, dict) and isinstance(model2, dict):
        state_dict1, state_dict2 = model1, model2
    else:
        state_dict1, state_dict2 = model1.state_dict(), model2.state_dict()

    if len(state_dict1) != len(state_dict2):
        return False, "The models have different numbers of parameters."

    shape_discrepancies = []
    weight_discrepancies = []

    for (name1, param1), (name2, param2) in zip(state_dict1.items(), state_dict2.items()):
        if name1 != name2:
            return False, f"Layer names do not match: {name1} vs {name2}"
        if param1.shape != param2.shape:
            shape_discrepancies.append((name1, param1.shape, param2.shape))
        elif not torch.equal(param1, param2):
            weight_discrepancies.append(name1)

    if shape_discrepancies:
        return False, f"Shape discrepancies found: {shape_discrepancies}"

    if weight_discrepancies:
        if len(weight_discrepancies) > 20:  # Arbitrary threshold for too many discrepancies
            return False, f"Weight discrepancies found in {len(weight_discrepancies)} layers."
        else:
            return False, f"Weight discrepancies found in layers: {weight_discrepancies}"

    return True, "The models have the same weights and shapes."


def compare_batchnorm_stats(model1, model2):
    def get_modules_and_state_dict(model):
        if isinstance(model, dict):
            return [(name, None) for name in model.keys()], model
        else:
            return list(model.named_modules()), model.state_dict()

    modules1, state_dict1 = get_modules_and_state_dict(model1)
    modules2, state_dict2 = get_modules_and_state_dict(model2)

    for (name1, _), (name2, _) in zip(modules1, modules2):
        if name1 in state_dict1 and "running_mean" in name1 and "running_var" in name1:
            running_mean1 = state_dict1[name1.replace(".running_mean", "") + ".running_mean"]
            running_var1 = state_dict1[name1.replace(".running_var", "") + ".running_var"]
            running_mean2 = state_dict2[name2.replace(".running_mean", "") + ".running_mean"]
            running_var2 = state_dict2[name2.replace(".running_var", "") + ".running_var"]
            if not torch.equal(running_mean1, running_mean2) or not torch.equal(running_var1, running_var2):
                print(f"Discrepancy in BatchNorm layer {name1}")
                print(f"Model 1 - {name1} running_mean: {running_mean1}")
                print(f"Model 1 - {name1} running_var: {running_var1}")
                print(f"Model 2 - {name2} running_mean: {running_mean2}")
                print(f"Model 2 - {name2} running_var: {running_var2}")


def clients_have_same_wights(clients):
    return models_have_same_weights([c.model for c in clients])


def models_have_same_weights(models):
    first_model = models[0]
    for model in models[1:]:
        is_same, msg = compare_models(first_model, model)
        compare_batchnorm_stats(first_model, model)
        if not is_same:
            print(msg)
            return False, msg
    print("The models have the same weights and shapes.")
    return True, "The models have the same weights and shapes."


def compare_vocabulary(vocab1, vocab2):
    """
    Compare two GeneVocab vocabularies to ensure they are identical.

    Parameters:
    - vocab1, vocab2: GeneVocab objects representing vocabularies.

    Returns:
    - bool: True if vocabularies are identical, False otherwise.
    - str: Message describing any discrepancies.
    """
    def extract_vocab_dict(gene_vocab):
        # Extract the torchtext Vocab object
        vocab_obj = gene_vocab.vocab
        # Convert to a dictionary {word: index}
        vocab_dict = {word: vocab_obj[word] for word in vocab_obj.get_itos()}
        return vocab_dict

    vocab1_dict = extract_vocab_dict(vocab1)
    vocab2_dict = extract_vocab_dict(vocab2)

    if len(vocab1_dict) != len(vocab2_dict):
        return False, "Vocabularies have different lengths."

    for gene, index1 in vocab1_dict.items():
        index2 = vocab2_dict.get(gene)
        if index2 is None:
            return False, f"Gene '{gene}' is missing in the second vocabulary."
        if index1 != index2:
            return False, f"Different indices for gene '{gene}': {index1} vs {index2}"

    return True, "Vocabularies are consistent."


def verify_tokenization_consistency(client_tokenized_data_list):
    """
    Verifies the consistency of tokenization across different clients.

    Parameters:
    - client_tokenized_data_list: A list of dictionaries containing tokenized data from different clients.

    Returns:
    - bool: True if all tokenizations are consistent, False otherwise.
    - str: Message describing any discrepancies.
    """
    if not client_tokenized_data_list:
        return False, "No tokenized data provided."

    # Take the first client's tokenized data as the reference
    reference_data = client_tokenized_data_list[0]
    ref_feature_length = reference_data['genes'].shape[1]

    for idx, tokenized_data in enumerate(client_tokenized_data_list[1:], start=1):
        feature_length = tokenized_data['genes'].shape[1]

        if feature_length != ref_feature_length:
            return False, f"Inconsistent feature length for client {idx}: {feature_length} vs {ref_feature_length}"

        # Optional: Check if the tokenized content is the same
        if not np.array_equal(reference_data['genes'], tokenized_data['genes']):
            return False, f"Tokenized content differs for client {idx}"

    return True, "Tokenization is consistent across all clients."


class ResultsRecorder:
    def __init__(self, dataset, file_name='param_tuning', logger=None, verbose=False, agg_method="FedAvg", mu=None, prep_mode="fed-weight-avg"):
        self.results_file = file_name + '.csv'
        self.pickle_file = file_name + '.pkl'
        self.columns = ['Dataset', 'Round', 'Metric', 'Value', 'n_epochs', 'mu', 'Aggregation', 'prep_mode']
        self.dataset = dataset
        self.results_df = pd.DataFrame(columns=self.columns)
        self.all_results = {}
        self.logger = logger if logger else print
        self.verbose = verbose
        self.agg_method = agg_method
        self.mu = mu
        self.prep_mode = prep_mode
        if dataset not in self.all_results:
            self.all_results[dataset] = {}
        if self.agg_method not in self.all_results[dataset]:
            self.all_results[dataset][self.agg_method] = {}


    def load_or_create_dataframe(self):
        """Load the DataFrame from a CSV file or create a new one if the file doesn't exist."""
        if os.path.exists(self.results_file):
            return pd.read_csv(self.results_file)
        else:
            return pd.DataFrame(columns=self.columns)

    def load_or_create_pickle(self):
        """Load the results dictionary from a pickle file or create a new one if the file doesn't exist."""
        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def update_dataframe(self, accuracy, precision, recall, macro_f1, round_number=None, n_epochs=None, dataset=None, mu=None):
        """Update the DataFrame with new round results."""
        if dataset is None:
            dataset = self.dataset
        if mu is None:
            mu = self.mu
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Macro_F1': macro_f1
        }
        self.logger(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Macro F1: {macro_f1:.3f}")
        new_rows = pd.DataFrame([{
            'Round': round_number,
            'Metric': metric,
            'Value': value,
            'n_epochs': n_epochs,
            'Dataset': dataset,
            'mu': mu,
            'Aggregation': self.agg_method,
            'prep_mode': self.prep_mode,
        } for metric, value in metrics.items()])
        self.results_df = pd.concat([self.results_df, new_rows], ignore_index=True)



    def save_dataframe(self):
        """Save the DataFrame to the CSV file with file locking (Unix)."""
        if os.path.exists(self.results_file):
            mode = 'r+'
        else:
            mode = 'w+'

        with open(self.results_file, mode) as f:
            try:
                # Acquire an exclusive lock (blocking)
                fcntl.flock(f, fcntl.LOCK_EX)

                # Move to the beginning to read (if file exists)
                f.seek(0)
                if os.path.getsize(self.results_file) > 0:
                    f.seek(0)
                    df = pd.read_csv(f)
                    self.results_df = pd.concat([df, self.results_df], ignore_index=True)

                # Truncate and write updated content
                f.seek(0)
                f.truncate()
                self.results_df.to_csv(f, index=False)

                if self.verbose:
                    self.logger(f"Data successfully saved to {self.results_file}")
            finally:
                # Always release the lock
                fcntl.flock(f, fcntl.LOCK_UN)

    def update_pickle(self, predictions, labels, id_maps, epoch, round_number, all_results=None, dataset=None, mu=None, agg_method=None):
        """Update the pickle file with detailed results for each epoch and round."""
        if dataset is None:
            dataset = self.dataset
        if agg_method is None:
            agg_method = self.agg_method
        if all_results is None:
            all_results = self.all_results
        if mu is None:
            mu = self.mu

        all_results.setdefault(dataset, {}) \
            .setdefault(agg_method, {}) \
            .setdefault(epoch, {}) \
            .setdefault(round_number, {})[mu] = {'predictions': predictions, 'labels': labels}

        if 'id_maps' not in all_results[dataset]:
            all_results[dataset]['id_maps'] = id_maps
        else:
            assert all_results[dataset]['id_maps'] == id_maps, f"ID Maps mismatch for dataset {dataset}"

    def save_pickle(self):
        """Merge existing pickle file with current results and save, using file lock (Unix)."""
        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, 'rb+') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)  # lock the file for exclusive access
                    try:
                        old_results = pickle.load(f)
                    except EOFError:
                        old_results = {}

                    # Merge logic
                    for ds_name, ds_dict in self.all_results.items():
                        for agg_method, agg_dict in ds_dict.items():
                            if agg_method == "id_maps":
                                continue
                            for epoch, epoch_dict in agg_dict.items():
                                for round_number, round_dict in epoch_dict.items():
                                    for mu, preds in round_dict.items():
                                        self.update_pickle(preds['predictions'],
                                                           preds['labels'],
                                                           ds_dict['id_maps'],
                                                           epoch,
                                                           round_number,
                                                           all_results=old_results,
                                                           dataset=ds_name,
                                                           mu=mu,
                                                           agg_method=agg_method
                                                           )

                    # Write back merged result
                    f.seek(0)
                    f.truncate()
                    pickle.dump(old_results, f)
                    if self.verbose:
                        self.logger(f"Merged and saved results to {self.pickle_file}")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            # First-time save (file doesn't exist yet)
            with open(self.pickle_file, 'wb') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    pickle.dump(self.all_results, f)
                    if self.verbose:
                        self.logger(f"Saved new results to {self.pickle_file}")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def record_metrics(self, round_number, accuracy, precision, recall, macro_f1, n_epochs):
        """Record and save metrics in the DataFrame."""
        self.update_dataframe(accuracy, precision, recall, macro_f1, round_number, n_epochs)
        self.save_dataframe()

    def record_detailed_results(self, epoch, round_number, predictions, labels, id_maps):
        """Record and save detailed results in the pickle file."""
        self.update_pickle(predictions, labels, id_maps, epoch, round_number)
        self.save_pickle()

    def update(self, accuracy, precision, recall, macro_f1, predictions, labels, id_maps, round_number, n_epochs):
        self.update_dataframe(accuracy, precision, recall, macro_f1, round_number, n_epochs)
        self.update_pickle(predictions, labels, id_maps, n_epochs, round_number)


    def save(self):
        self.save_dataframe()
        self.save_pickle()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab, count_matrix, gene_ids, emb_style='<cls>', pad_value='<pad>', batch_ids=None):
        self.vocab = vocab
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids
        self.emb_style = emb_style
        self.pad_value = pad_value

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        # append <cls> token at the beginning
        genes = np.insert(genes, 0, self.vocab[self.emb_style])
        values = np.insert(values, 0, self.pad_value)
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output


def plot_embedding(adata, query, cell_type_key, output_dir):
    def concat_reference_query(ref, query):
        n_ref_samples, n_query_samples = len(ref), len(query)
        adata_concat = query.concatenate(ref, batch_key="dataset")
        # mark the reference vs. query dataset
        adata_concat.obs["is_ref"] = ["Query"] * n_query_samples + ["Reference"] * n_ref_samples
        adata_concat.obs["is_ref"] = adata_concat.obs["is_ref"].astype("category")
        # mask the query dataset cell types
        adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].astype("category")
        adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].cat.add_categories(["To be predicted"])
        adata_concat.obs.iloc[:n_query_samples, adata_concat.obs.columns.get_loc(cell_type_key)] = "To be predicted"
        return adata_concat



    concat_adata = concat_reference_query(adata, query)

    # Compute neighbors and UMAP
    sc.pp.neighbors(concat_adata, use_rep="X_scGPT")
    sc.tl.umap(concat_adata)

    def custom_plot_umap(adata, color_by, file_name, legend='no_legend'):
        """
        Custom function to plot UMAP embeddings with or without legends.

        Args:
            adata (AnnData): The AnnData object containing the UMAP coordinates.
            color_by (list): List of column names in `adata.obs` to color the plots by.
            file_name (str): Name of the file to save the plot.
            legend (str): Either 'no_legend' or 'legend_only'.
        """
        # Create the UMAP plot
        fig, axes = plt.subplots(1, len(color_by), figsize=(len(color_by) * 5, 5))

        if len(color_by) == 1:
            axes = [axes]

        for ax, color in zip(axes, color_by):
            sc.pl.umap(adata, color=color, ax=ax, show=False, frameon=False)
            if legend == 'no_legend':
                ax.get_legend().remove()  # Remove the legend
            elif legend == 'legend_only':
                # Clear the data but keep the legend
                for coll in ax.collections:
                    coll.remove()
                ax.set_title('')
                ax.set_xlabel('')
                ax.set_ylabel('')
            else:
                raise ValueError(f"Invalid value for legend: {legend}")

        if legend == 'no_legend':
            plt.tight_layout()
        elif legend == 'legend_only':
            # Adjust the subplot parameters to include more space for the legend
            plt.subplots_adjust(right=0.5)
            for ax in axes:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])  # resize the plot

        plt.savefig(file_name, dpi=300)
        plt.close()

    sc.pp.neighbors(concat_adata, use_rep="X_scGPT")
    sc.tl.umap(concat_adata)

    # Plot UMAP with and without legends
    custom_plot_umap(concat_adata, color_by=["is_ref", cell_type_key], file_name=f"{output_dir}/embedding_umap_plot.png",
                     legend='no_legend')
    custom_plot_umap(concat_adata, color_by=["is_ref", cell_type_key], file_name=f"{output_dir}/embedding_umap_legend.png",
                     legend='legend_only')



def l2_sim(a, b):
    sims = -np.linalg.norm(a - b, axis=1)
    return sims

def get_similar_vectors(vector, ref, top_k=10):
    # sims = cos_sim(vector, ref)
    sims = l2_sim(vector, ref)

    top_k_idx = np.argsort(sims)[::-1][:top_k]
    return top_k_idx, sims[top_k_idx]


def eval_reference_mapping(gt, preds, output_dir="output", logger=None):
    if logger is None:
        logger = print
    # Calculate evaluation metrics
    res_dict = {
        "accuracy": accuracy_score(gt, preds),
        "precision": precision_score(gt, preds, average="macro"),
        "recall": recall_score(gt, preds, average="macro"),
        "macro_f1": f1_score(gt, preds, average="macro"),
    }

    # Print the evaluation metrics
    logger("Evaluation Metrics:")
    for key, value in res_dict.items():
        logger(f"{key.capitalize()}: {value:.4f}")

    # Save the evaluation metrics to a CSV file
    metrics_df = pd.DataFrame([res_dict])
    metrics_df.to_csv(f"{output_dir}/evaluation_metrics.csv", index=False)

    # Prepare confusion matrix
    y_true = gt
    y_pred = preds
    cell_type_list = np.unique(y_true)
    matrix = confusion_matrix(y_true, y_pred, labels=cell_type_list)
    matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    # Create a DataFrame for the confusion matrix
    df = pd.DataFrame(matrix, index=cell_type_list[:matrix.shape[0]], columns=cell_type_list[:matrix.shape[1]])

    # Create and save the clustermap
    ax = sns.clustermap(df,
                        cmap='Purples',
                        annot=True, fmt=".2f",
                        annot_kws={'size': 8},
                        vmin=0,
                        vmax=1,
                        row_cluster=False,
                        col_cluster=False,
                        figsize=(14, 14))

    clustermap_path = f"{output_dir}/confusion_matrix_clustermap.png"
    plt.savefig(clustermap_path)
    plt.close()

def check_weights_nan(weights, when, debug):
    if debug:
        # Convert odict_values to a list
        if isinstance(weights, (dict, OrderedDict)):  # Ensure it's a dictionary-like object
            weights = weights if isinstance(weights, dict) else dict(weights)  # Convert OrderedDict to dict

            for name, param in weights.items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"⚠️ NaN or Inf found in {name} {when}!")

        elif isinstance(weights, list):  # Handle list of tensors
            for param in weights:
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"⚠️ NaN or Inf found {when}!")

        elif isinstance(weights, torch.Tensor):  # Handle single tensor case
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                print(f"⚠️ NaN or Inf found {when}!")

        else:
            print(f"⚠️ Unexpected type {type(weights)} in check_weights_nan {when}!")


def concat_encrypted_distances(distances):
    """
    Concatenate a list of encrypted distance tensors along the feature dimension.

    Parameters
    ----------
    distances : list of crypten.CrypTensor
        List of encrypted distance tensors, each of shape (n_query, k_i).

    Returns
    -------
    crypten.CrypTensor
        Concatenated encrypted distances with shape (n_query, sum(k_i)).
    """
    return crypten.cat(distances, dim=1)


def suppress_argmin(dist_matrix, argmin_onehot, batch_size=128, large_val=1e9):
    """
    Securely suppress the minimum value in each row of an encrypted distance matrix by masking.

    This replaces the current minimum in each query row with a large value to exclude it
    from subsequent minimum searches, using one-hot encoded masks in batches for efficiency.

    Parameters
    ----------
    dist_matrix : crypten.CrypTensor
        Encrypted distance matrix, shape (n_query, n_ref).
    argmin_onehot : crypten.CrypTensor or torch.Tensor
        One-hot mask tensor indicating positions of the current minimum values.
        Shape (n_query, n_ref).
    batch_size : int, optional
        Number of rows to process per batch. Default is 128.
    large_val : float, optional
        Large value to add at masked positions. Default is 1e9.

    Returns
    -------
    crypten.CrypTensor
        Encrypted distance matrix with minimum values suppressed, shape (n_query, n_ref).
    """
    n_query, n_ref = dist_matrix.size()
    large_val_enc = crypten.cryptensor(
        torch.tensor(large_val, device=dist_matrix.device)
    )
    updated_batches = []
    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        dist_batch = dist_matrix[start:end]  # (bs, n_ref)
        mask_batch = argmin_onehot[start:end]  # (bs, n_ref)
        updated = dist_batch + mask_batch * large_val_enc
        updated_batches.append(updated)

    return crypten.cat(updated_batches, dim=0)


def top_k_encrypted_distances(encrypted_dist_matrix, k):
    """
    Extract the top-k smallest encrypted distances from each query to references.

    This uses one-hot index selection to mask out other distances.

    Parameters
    ----------
    encrypted_dist_matrix : crypten.CrypTensor
        Encrypted squared distance matrix, shape (n_query, n_ref).
    k : int
        Number of smallest distances to extract per query.

    Returns
    -------
    encrypted_topk : crypten.CrypTensor
        Encrypted tensor of top-k distances per query, shape (n_query, k).
    topk_indices : list of crypten.CrypTensor
        List of one-hot index masks for each of the k smallest distances,
        each mask of shape (n_query, n_ref).
    """
    topk_indices = top_k_ind_selection(encrypted_dist_matrix.clone(), k)
    encrypted_topk = (encrypted_dist_matrix * topk_indices[0]).sum(dim=1, keepdim=True)
    for i in range(1, k):
        next_k = (encrypted_dist_matrix * topk_indices[i]).sum(dim=1, keepdim=True)
        encrypted_topk = crypten.cat([encrypted_topk, next_k], dim=1)
    return encrypted_topk, topk_indices


def top_k_ind_selection(dist_matrix, k):
    """
    Securely select one-hot masks for the k smallest entries in each row.

    Iteratively finds the minimum entry, masks it, and repeats to build k masks.

    Parameters
    ----------
    dist_matrix : crypten.CrypTensor
        Encrypted distance matrix, shape (n_query, n_ref).
    k : int
        Number of minima to select per query.

    Returns
    -------
    list of crypten.CrypTensor
        List of k one-hot mask tensors, each of shape (n_query, n_ref),
        where True (1) indicates the position of the i-th smallest value.
    """
    topk_indices = []
    for _ in range(k):
        _, argmin = dist_matrix.min(dim=1)
        topk_indices.append(argmin)
        dist_matrix = suppress_argmin(dist_matrix, argmin)
    return topk_indices


def secure_quantile_cuts(shared_histogram, shared_total, envelope_grid_tail, probs):
    """
    Securely derive quantile cut values from a secret-shared histogram.

    Given an aggregated secret-shared histogram ``H`` of length ``M`` over a
    public, strictly increasing envelope grid ``g`` of length ``M + 1``, and a
    vector of public probabilities ``p`` in ``[0, 1]^{n_cuts}``, return secret-
    shared cut values where the ``j``-th cut is the upper edge ``g[m + 1]`` of
    the smallest bin ``m`` whose cumulative count ``S[m] = sum_{i<=m} H[i]``
    reaches ``p[j] * N`` (with ``N`` the secret-shared total). The
    construction mirrors the iterative one-hot selection style of
    ``top_k_ind_selection``: each cut is a one-hot dot product against the
    public envelope, and only the resulting cut values are intended to be
    revealed by the caller (the legitimate output of the protocol).
    Probabilities ``p[j] = 0`` are clamped so the corresponding cut falls
    inside the first non-empty bin rather than at the global lower bound.
    The cumulative sum is computed as a public-matrix / secret-vector product
    using a lower-triangular ones matrix, avoiding any dependency on a
    CrypTen-specific ``cumsum`` implementation.

    Parameters
    ----------
    shared_histogram : crypten.CrypTensor
        Secret-shared non-negative count vector of shape ``(M,)``.
    shared_total : crypten.CrypTensor
        Secret-shared scalar total count ``N`` (shape ``(1,)`` or 0-dim).
    envelope_grid_tail : torch.Tensor or np.ndarray
        Public envelope upper edges of shape ``(M,)`` (i.e. ``g[1:]``).
    probs : torch.Tensor or np.ndarray
        Public probabilities of shape ``(n_cuts,)`` in ``[0, 1]``.

    Returns
    -------
    crypten.CrypTensor
        Secret-shared cut values of shape ``(n_cuts,)``. The caller decides
        when to reveal these via ``get_plain_text()``.
    """
    if not isinstance(envelope_grid_tail, torch.Tensor):
        envelope_grid_tail = torch.as_tensor(envelope_grid_tail, dtype=torch.float32)
    else:
        envelope_grid_tail = envelope_grid_tail.to(dtype=torch.float32)

    if not isinstance(probs, torch.Tensor):
        probs = torch.as_tensor(probs, dtype=torch.float32)
    else:
        probs = probs.to(dtype=torch.float32)

    M = envelope_grid_tail.numel()
    device = getattr(shared_histogram, "device", torch.device("cpu"))
    envelope_grid_tail = envelope_grid_tail.to(device=device)

    tri = torch.tril(torch.ones(M, M, dtype=torch.float32, device=device))
    hist_col = shared_histogram.view(-1, 1)
    shared_S = crypten.cryptensor(tri).matmul(hist_col).squeeze(-1)

    public_envelope = crypten.cryptensor(
        envelope_grid_tail.detach().clone().to(device=device, dtype=torch.float32)
    )

    one_const = crypten.cryptensor(
        torch.ones(1, dtype=torch.float32, device=device)
    )
    zero_pad = crypten.cryptensor(
        torch.zeros(1, dtype=torch.float32, device=device)
    )

    cuts = []
    for p in probs.tolist():
        target = shared_total * float(p)
        target = (target - one_const).relu() + one_const
        ge_mask = shared_S.ge(target)
        shifted = crypten.cat([zero_pad, ge_mask[:-1]], dim=0)
        one_hot = ge_mask * (1 - shifted)
        cut = one_hot.mul(public_envelope).sum()
        cuts.append(cut.view(1))

    return crypten.cat(cuts, dim=0)


def get_plain_indices(topk_indices):
    """
    Decrypt and collect top-k index selections into a NumPy array.

    Parameters
    ----------
    topk_indices : list of crypten.CrypTensor
        List of encrypted index masks, each of shape (n_query,) indicating positions.

    Returns
    -------
    np.ndarray
        Decrypted indices array of shape (n_query, k).
    """
    topk_tensor = crypten.stack(topk_indices, dim=1)
    return topk_tensor.get_plain_text().long().cpu().numpy()


def encrypted_present_hashes(hash_to_index, labels):
    """
    Encode presence of string labels as an encrypted binary vector using SHA-256 hashes.

    Maps each provided label to its hashed index in the hash_to_index map,
    setting corresponding positions to 1, others remain 0.

    Parameters
    ----------
    hash_to_index : dict
        Mapping from SHA-256 hex hash strings to integer indices.
    labels : list of str
        Sequence of label strings to encode.

    Returns
    -------
    crypten.CrypTensor
        Encrypted binary presence vector of length len(hash_to_index).
    """
    hashed = [hashlib.sha256(l.encode()).hexdigest() for l in labels]
    presence = torch.zeros(len(hash_to_index), dtype=torch.float32)
    for h in hashed:
        idx = hash_to_index[h]
        presence[idx] = 1
    return crypten.cryptensor(presence)

def dump_predictions(preds, path):
    """
    Save predictions to a CSV file.

    Parameters
    ----------
    preds : list of str
        List of predicted labels.
    path : str
        Path to save the CSV file.
    """
    df = pd.DataFrame(preds, columns=["predictions"])
    df.to_csv(f"{path}/preds.csv", index=False)


class EfficientGPUContext:
    def __init__(self, outer_self, model=None, reset=False, debug=False):
        self.obj = outer_self
        self.debug = debug
        self.model = model
        self.reset = lambda: None
        if reset and hasattr(self.obj, 'reset') and callable(self.obj.reset):
            self.reset = self.obj.reset


    def __enter__(self):
        if self.debug:
            self.obj.log("🚀 Entering EfficientGPUContext: moving model to GPU.")
        self.obj.move_to_gpu(self.model)

        if getattr(self.obj, "use_fedprox", False) and hasattr(self.obj, "global_model") and self.obj.global_model:
            if self.debug:
                self.obj.log("📦 Moving global_model to GPU for FedProx.")
            self.obj.global_model = {
                k: v.to(self.obj.device) for k, v in self.obj.global_model.items()
            }

    def __exit__(self, exc_type, exc_value, traceback):
        if self.debug:
            self.obj.log("🔄 Exiting EfficientGPUContext: moving model to CPU.")
        self.reset()
        self.obj.move_to_cpu(self.model)
        if hasattr(self.obj, 'best_model') and self.obj.best_model is not None:
            self.obj.move_to_cpu(self.obj.best_model)
        if hasattr(self.model, "cur_gene_token_embs"):
            if self.debug:
                self.obj.log("🧹 Clearing model.cur_gene_token_embs from GPU")
            self.model.cur_gene_token_embs = self.model.cur_gene_token_embs.cpu()
            del self.model.cur_gene_token_embs

        if hasattr(self.obj, "global_model"):
            if self.debug:
                self.obj.log("🗑️ Deleting global_model to release GPU memory.")
            del self.obj.global_model
            self.obj.global_model = None

        gc.collect()
        torch.cuda.empty_cache()
        if self.debug:
            self.obj.log("🧹 Clearing CUDA memory cache (safe for FlashAttention).")
            self.log_gpu_objects()

    def log_gpu_objects(self):
        gpu_tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    device = obj.device if torch.is_tensor(obj) else obj.data.device
                    if device.type == 'cuda':
                        obj_type = type(obj).__name__
                        shape = tuple(obj.shape) if hasattr(obj, 'shape') else "N/A"
                        gpu_tensors.append((obj_type, device, shape))
            except Exception:
                pass  # skip broken refs

        if gpu_tensors:
            self.obj.log(f"📊 Found {len(gpu_tensors)} objects on GPU:")
            for obj_type, device, shape in gpu_tensors:
                self.obj.log(f"   🔍 {obj_type} on {device}, shape: {shape}")
        else:
            self.obj.log("✅ No residual objects on GPU detected.")


def list_gpu_objects(print_summary=True):
    """
    Lists all Python objects (tensors and models) currently allocated on CUDA devices.
    Optionally prints a summary.
    Returns a list of dicts with object info.
    """
    gpu_objects = []
    total_bytes = 0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                tensor = obj.data if hasattr(obj, "data") else obj
                if tensor.device.type == "cuda":
                    nbytes = tensor.element_size() * tensor.nelement()
                    total_bytes += nbytes
                    gpu_objects.append({
                        "type": type(obj).__name__,
                        "device": tensor.device,
                        "shape": tuple(tensor.shape),
                        "dtype": tensor.dtype,
                        "size_MB": round(nbytes / (1024**2), 2)
                    })
        except Exception:
            pass  # Ignore objects that raise errors on inspection

    if print_summary:
        print(f"\n📦 Found {len(gpu_objects)} tensors on GPU, total {round(total_bytes / (1024**2), 2)} MB.")
        for obj in gpu_objects:
            print(f"   🔍 {obj['type']} on {obj['device']}, shape: {obj['shape']}, "
                  f"dtype: {obj['dtype']}, size: {obj['size_MB']} MB")


