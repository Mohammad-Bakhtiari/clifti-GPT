from torch.utils.data import DataLoader
import torch
from typing import Dict, List
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shutil
import json
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from cliftiGPT.utils import SeqDataset, dump_results, plot, ResultsRecorder
from cliftiGPT.centralized.models import ScGPT
from cliftiGPT.utils import read_h5ad, seed_worker, EfficientGPUContext
import copy
from functools import partial

class Base(ScGPT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.dataset.data_is_raw = False
        self.config.dataset.filter_gene_by_counts = False
        self.preprocessor = None

    def get_raw_testset(self, query_adata):
        if query_adata is None:
            return None
        return read_h5ad(self.data_dir, query_adata)

    def harmonize(self, adata, adata_test=None):
        self.manege_id2type(adata)
        self.set_obs_and_vars(adata)
        if adata_test is not None:
            self.set_obs_and_vars(adata_test)
            self.check_category(adata, adata_test, self.celltype_key)
        else:
            self.log("No query dataset is provided.")

    def manege_id2type(self, adata):
        self.unique_cell_types = adata.obs[self.celltype_key].cat.categories.tolist()
        self.config.model.n_cls = len(self.unique_cell_types)
        self.cell_id2type = dict(enumerate(self.unique_cell_types))

    def set_obs_and_vars(self, adata):
        adata.obs["batch_id"] = adata.obs[self.batch_key].astype("category").cat.codes.values
        adata.obs["celltype_id"] = adata.obs[self.celltype_key].astype("category").cat.codes.values
        adata.var["gene_name"] = adata.var.index.tolist()

    def harmonize_query(self, adata_test):
        self.manege_id2type(adata_test)
        self.set_obs_and_vars(adata_test)

    def check_category(self, reference, query, obs_key):
        if reference.obs[obs_key].dtype != "category":
            self.log(f"{obs_key} is not a category in the reference dataset.")
        if query.obs[obs_key].dtype != "category":
            self.log(f"{obs_key} is not a category in the query dataset.")
        if all(reference.obs[obs_key].cat.categories != query.obs[obs_key].cat.categories):
            self.log(f"Categories of {obs_key} are not the same in the reference and query datasets.")
            reference_unique_obs = reference.obs[obs_key].unique().tolist()
            query_unique_cell_obs = query.obs[obs_key].unique().tolist()
            self.log(f"Unique {obs_key} in the reference dataset: {reference_unique_obs}")
            reference.obs[obs_key] = reference.obs[obs_key].cat.set_categories(query_unique_cell_obs)
            query.obs[obs_key] = query.obs[obs_key].cat.set_categories(reference_unique_obs)





    def filter(self, adata, adata_test=None):
        self.log("Filtering genes in the reference dataset that are not in the vocabulary.")
        adata = self.filter_id_in_vocab(adata)
        if adata_test is not None:
            self.log("Filtering genes in the query dataset that are not in the vocabulary.")
            adata_test = self.filter_id_in_vocab(adata_test)
            return adata, adata_test
        return adata

    def instantiate_preprocessor(self):
        self.preprocessor = Preprocessor(
            use_key=self.config.dataset.raw_data_key,
            filter_gene_by_counts=self.config.dataset.filter_gene_by_counts,
            filter_cell_by_counts=self.config.dataset.filter_cell_by_counts,
            normalize_total=self.config.dataset.normalize_total,
            result_normed_key=self.config.dataset.result_normed_key,
            log1p=self.config.dataset.log1p,
            result_log1p_key=self.config.dataset.result_log1p_key,
            subset_hvg=self.config.dataset.subset_hvg,
            hvg_flavor=self.config.dataset.hvg_flavor,
            binning=self.config.preprocess.n_bins,
            result_binned_key=self.config.dataset.result_binned_key,
        )

    def set_layer_key(self):
        self.input_layer_key = {
            "normed_raw": "X_normed",
            "log1p": "X_normed",
            "binned": "X_binned",
        }[self.config.preprocess.input_style]

    def get_all_counts(self, adata):
        try:
            return adata.layers[self.input_layer_key].A if issparse(adata.layers[self.input_layer_key]) else adata.layers[
                self.input_layer_key]
        except:

            if len(adata.layers.keys()) == 0:
                msg = "adata.layers is empty. Make sure the data is preprocessed!"
            else:
                msg = f"{self.input_layer_key} is not in adata.layers.keys()!"
            raise ValueError(msg)

class Training(Base):
    def __init__(self, reference_adata, **kwargs):
        super().__init__(**kwargs)
        self.read_reference(reference_adata)
        self.train_data = None
        self.train_celltype_labels = None
        self.train_batch_labels = None
        self.valid_data = None
        self.valid_celltype_labels = None
        self.valid_batch_labels = None

    def preprocess_reference(self):
        self.preprocessor(self.adata, batch_key=None)

    def post_prep(self, test_size=0.1):
        self.set_layer_key()
        all_counts = self.get_all_counts(self.adata)
        celltypes_labels = self.adata.obs["celltype_id"].tolist()
        celltypes_labels = np.array(celltypes_labels)
        batch_ids = self.adata.obs["batch_id"].tolist()
        if test_size > 0:
            batch_ids = np.array(batch_ids)
            (
                self.train_data,
                self.valid_data,
                self.train_celltype_labels,
                self.valid_celltype_labels,
                self.train_batch_labels,
                self.valid_batch_labels,
            ) = train_test_split(all_counts, celltypes_labels, batch_ids, test_size=test_size, shuffle=True)
        else:
            self.train_data = np.array(all_counts)
            self.train_celltype_labels = np.array(celltypes_labels)
            self.train_batch_labels = np.array(batch_ids)


class Inference(Base):
    def __init__(self, query_adata, dataset_name, param_tuning_res, load_model=True, model_name="model.pt", param_tuning=False, agg_method=None, mu=None, **kwargs):
        super().__init__(**kwargs)
        self.celltypes_labels = None
        self.read_query(query_adata)
        self.manege_id2type(self.adata_test)
        self.set_obs_and_vars(self.adata_test)
        self.load_pretrained_config()
        self.adata_test = self.filter_id_in_vocab(self.adata_test)
        self.instantiate_preprocessor()
        self.preprocess_query()
        self.gene_ids = np.array(self.vocab(self.adata_test.var["gene_name"].tolist()), dtype=int)
        self.instantiate_transformer_model()
        if load_model:
            self.load_pretrained_model(model_name)
        self.setup_losses()
        self.best_model = copy.deepcopy(self.model)
        self.best_model.eval()
        self.plot_dir = f"{self.output_dir}/plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir, exist_ok=True)
        self.test_loader = None
        self.param_tuning = param_tuning
        if agg_method == "federated":
            agg_method = "FedProx" if self.use_fedprox else "FedAvg"
            agg_method = f"weighted-{agg_method}" if kwargs["weighted"] else agg_method
            agg_method = f"SMPC-{agg_method}" if kwargs['smpc'] else agg_method
        self.result_recorder = ResultsRecorder(dataset=dataset_name,
                                               file_name=param_tuning_res,
                                               logger=self.log,
                                               agg_method=agg_method,
                                               mu=mu,
                                               prep_mode=kwargs.get("prep_mode", "fed-weight-avg"),
                                               )

    def read_query(self, query_adata):
        self.adata_test_raw = read_h5ad(self.data_dir, query_adata)
        self.adata_test = self.adata_test_raw.copy()

    def preprocess_query(self):
        self.preprocessor(self.adata_test, batch_key=None)
        self.set_layer_key()

    def test(self, round_num, n_epochs, mu=None) -> (np.ndarray, np.ndarray, Dict[str, float]):
        if self.test_loader is None or self.celltypes_labels is None:
            self.load_test_loader()
        with EfficientGPUContext(self, model=self.best_model):
            predictions = self.evaluate(self.best_model, loader=self.test_loader, return_raw=True)
        accuracy = accuracy_score(self.celltypes_labels, predictions)
        precision = precision_score(self.celltypes_labels, predictions, average="macro")
        recall = recall_score(self.celltypes_labels, predictions, average="macro")
        macro_f1 = f1_score(self.celltypes_labels, predictions, average="macro")
        self.result_recorder.update(accuracy,
                                    precision,
                                    recall,
                                    macro_f1,
                                    predictions,
                                    self.celltypes_labels,
                                    self.cell_id2type,
                                    round_num,
                                    n_epochs,
                                    )
        results = {
            "test/accuracy": accuracy,
            "test/precision": precision,
            "test/recall": recall,
            "test/macro_f1": macro_f1,
        }
        return predictions, self.celltypes_labels, results

    def load_test_loader(self):
        all_counts = self.get_all_counts(self.adata_test)
        self.celltypes_labels = np.array(self.adata_test.obs["celltype_id"].tolist())
        batch_ids = self.adata_test.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        tokenized_test = self.tokenize_and_pad_batch(all_counts)
        input_values_test = self.random_mask_value(tokenized_test["values"])
        test_data_pt = {
            "gene_ids": tokenized_test["genes"],
            "values": input_values_test,
            "target_values": tokenized_test["values"],
            "batch_labels": torch.from_numpy(batch_ids).long(),
            "celltype_labels": torch.from_numpy(self.celltypes_labels).long(),
        }
        self.test_loader = DataLoader(
            dataset=SeqDataset(test_data_pt),
            batch_size=self.config.train.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), self.config.train.eval_batch_size // 2),
            pin_memory=True,
            worker_init_fn=seed_worker
        )

    def inference(self, plot_results=False, round_num=None, n_epochs=None, mu=None):
        predictions, labels, results = self.test(round_num, n_epochs, mu)
        self.adata_test_raw.obs["predictions"] = [self.cell_id2type[p] for p in predictions]
        if plot_results:
            plot(self.adata_test_raw, self.unique_cell_types, self.celltype_key, self.plot_dir)
        return predictions, labels

    def save_results(self, labels, predictions, results):
        dump_results(predictions, labels, results, self.cell_id2type, self.output_dir)

    def save_records(self):
        self.result_recorder.save()

    def update_records(self, **kwargs):
        self.result_recorder.update(labels=self.celltypes_labels, id_maps=self.cell_id2type, **kwargs)

class CellTypeAnnotator(Training, Inference):
    def __init__(self, reference_adata, query_adata=None, **kwargs):
        Base.__init__(self, **kwargs)
        self.read_reference(reference_adata)
        if query_adata is not None:
            self.read_query(query_adata)
