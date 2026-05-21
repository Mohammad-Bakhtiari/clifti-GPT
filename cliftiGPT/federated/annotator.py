import copy
import os.path
import torch
import crypten
import numpy as np
from typing import Dict
from cliftiGPT.base import FedBase
from cliftiGPT.centralized.annotator import Training, Inference
from cliftiGPT.utils import read_h5ad
from cliftiGPT.preprocessor.local import Preprocessor
from cliftiGPT.preprocessor.aggregation import aggregate_gene_counts, aggregate_bin_edges, aggregate_hvg_stats, \
    aggregate_local_gene_sets, aggregate_local_celltype_sets, scale_binning_nonzero_count, \
    reveal_scaled_nonzero_total, aggregate_bin_edge_contributions_smpc, local_bin_edge_contribution
from cliftiGPT.federated.aggregator import FedAvg
from cliftiGPT.federated.client import Client
from cliftiGPT.centralized.annotator import Training

class ClientAnnotator(Client, Training):
    """
    cell_id2type: Here is calculated locally. No global ID!
    """

    def __init__(self, **kwargs):
        Training.__init__(self, **kwargs)
        self.n_samples = len(self.adata)
        Client.__init__(self, **kwargs)
        self.preprocessor = Preprocessor(
            log=self.log,
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

    def get_local_gene_set(self):
        return self.preprocessor.get_local_gene_set(self.adata)

    def get_local_celltype_set(self):
        return self.preprocessor.get_local_celltype_set(self.adata, self.celltype_key)

    def check_local_gene_set(self, global_gene_dict: Dict[int, str]):
        assert set(self.adata.var.index.tolist()) == set(
            global_gene_dict.values()), "Local gene set is not consistent with global gene set."

    def fed_harmonize(self, global_cellytpe_dict: Dict[int, str]):
        """
        In centralized all unique cell types can be read from the dataset.
        Returns
        -------

        """
        self.adata.obs["batch_id"] = self.adata.obs[self.batch_key].astype("category").cat.codes.values
        cellytpe_dict = {v: k for k, v in global_cellytpe_dict.items()}
        self.adata.obs["celltype_id"] = [cellytpe_dict[i] for i in self.adata.obs[self.celltype_key]]
        self.config.model.n_cls = len(global_cellytpe_dict) if self.config.train.CLS else 1
        self.cell_id2type = global_cellytpe_dict
        self.adata.var["gene_name"] = self.adata.var.index.tolist()
        self.unique_cell_types = list(global_cellytpe_dict.values())
        self.load_pretrained_config()
        self.filter(self.adata)

    def local_harmonize(self):
        super().harmonize(self.adata)

    def get_local_gene_counts(self):
        return self.preprocessor.local_gene_counts(self.adata)

    def apply_gene_mask(self, gene_mask):
        self.adata = self.preprocessor.apply_global_gene_counts(self.adata, gene_mask)

    def filter_cells(self):
        self.adata = self.preprocessor.filter_cells(self.adata)

    def total_normalization(self):
        self.adata = self.preprocessor.total_normalization(self.adata)

    def log1p(self):
        self.adata = self.preprocessor.log1p(self.adata)

    def get_local_hvg_stats(self):
        return self.preprocessor.compute_local_hvg_stats(self.adata)

    def apply_hvg_stats(self, global_hvg_stats):
        self.adata = self.preprocessor.subset_hvgs(self.adata, global_hvg_stats, n_top_genes=None)

    def get_local_bin_edges(self):
        return self.preprocessor.compute_local_bin_edges(self.adata)

    def get_local_binning_n_share(self):
        """Secret-shared scaled non-zero count for fed-weight-avg-smpc (phase 1)."""
        _, n = self.preprocessor.compute_local_bin_edges(self.adata)
        n_scaled = torch.tensor(
            scale_binning_nonzero_count(n), dtype=torch.float32, device=self.device
        )
        return crypten.cryptensor(n_scaled.view(1))

    def get_local_binning_contribution_share(self, total_n_scaled: float):
        """Secret-shared weighted edge contribution B_i * (n_i / N) (phase 2)."""
        bin_edges, n = self.preprocessor.compute_local_bin_edges(self.adata)
        contrib = local_bin_edge_contribution(bin_edges, n, total_n_scaled)
        contrib_t = torch.tensor(contrib, dtype=torch.float32, device=self.device)
        return crypten.cryptensor(contrib_t)

    def get_local_envelope_stats(self):
        """Secret-shared local max and non-zero count for secure-histogram binning.

        Returns a pair of ``crypten.CrypTensor`` shares (each of shape ``(1,)``)
        suitable for being consumed by ``secure_reveal_envelope_max`` and the
        secret-shared ``N`` channel of ``aggregate_secure_histogram_bin_edges``,
        mirroring ``ClientEmbedder.report_n_local_samples``.
        """
        local_max, local_n = self.preprocessor.compute_local_envelope_stats(self.adata)
        local_max = local_max.view(1).to(self.device)
        local_n = local_n.view(1).to(self.device)
        return crypten.cryptensor(local_max), crypten.cryptensor(local_n)

    def get_local_histogram(self, envelope_grid: np.ndarray):
        """Secret-shared local histogram of non-zero values over the public envelope.

        ``envelope_grid`` is the public ``(M + 1,)`` grid derived from the
        previously revealed ``max_global``; the returned share has shape
        ``(M,)`` and is to be summed across clients inside
        ``aggregate_secure_histogram_bin_edges``.
        """
        hist = self.preprocessor.compute_local_histogram(self.adata, envelope_grid)
        hist = hist.to(self.device)
        return crypten.cryptensor(hist)

    def binning(self, global_bin_edges):
        self.adata = self.preprocessor.apply_binning(self.adata, global_bin_edges)

    def local_update(self, global_weights, round_num):
        if self.use_fedprox:
            self.global_model = copy.deepcopy(global_weights)
        if round_num > 1:
            self.set_weights(global_weights)
        else:
            self.model.load_state_dict(global_weights)
        self.train()
        return self.get_local_updates()

    def centralized_training(self, init_weights=None):
        if init_weights is None:
            init_weights = self.model.state_dict()
        else:
            self.model.load_state_dict(init_weights)
        self.train()
        trained_weights = self.model.state_dict()
        self.model.load_state_dict(init_weights)
        return trained_weights


class FedAnnotator(FedBase, FedAvg):
    def __init__(self, reference_adata, data_dir, output_dir, n_rounds, **kwargs):
        FedBase.__init__(self, data_dir=data_dir, output_dir=output_dir, n_rounds=n_rounds, **kwargs)
        FedAvg.__init__(self, n_rounds=self.fed_config.n_rounds, **kwargs)
        self.prep_mode = kwargs.get("prep_mode", "fed-weight-avg")
        adata = read_h5ad(data_dir, reference_adata)
        n_total_samples = len(adata)
        self.distribute_adata_by_batch(adata, kwargs['batch_key'])
        for c in range(self.n_clients):
            client = ClientAnnotator(reference_adata='adata.h5ad',
                                     data_dir=self.clients_data_dir[c],
                                     output_dir=self.clients_output_dir[c],
                                     log_id=f"CLIENT_{self.client_ids[c]}",
                                     logger=self.logger,
                                     n_total_samples=n_total_samples,
                                     **kwargs)
            self.clients.append(client)
        self.retain_best_model(False)
        # TODO: Support post-finetune federated zeroshot

    def aggregate_gene_sets(self):
        local_gene_sets = [client.get_local_gene_set() for client in self.clients]
        global_gene_dict = aggregate_local_gene_sets(local_gene_sets)
        for client in self.clients:
            client.check_local_gene_set(global_gene_dict)

    def aggregate_celltype_sets(self):
        local_celltype_sets = [client.get_local_celltype_set() for client in self.clients]
        global_celltype_dict = aggregate_local_celltype_sets(local_celltype_sets)
        for client in self.clients:
            client.fed_harmonize(global_celltype_dict)

    def load_pretrained_config(self):
        for client in self.clients:
            client.load_pretrained_config()
    def filter_genes(self):
        for client in self.clients:
            client.adata = client.filter(client.adata)

    def preprocess_data(self):
        if self.prep_mode == "centralized":
            self._preprocess_data_centralized()
        elif self.prep_mode in ("fed-weight-avg", "fed-weight-avg-smpc", "fed-hist", "fed-hist-smpc"):
            self._preprocess_data_federated()
        else:
            raise ValueError(f"Unknown prep_mode: {self.prep_mode!r}")

    def _preprocess_data_federated(self):
        if self.fed_config.preprocess.filter_gene_by_counts:
            self.logger.federated("Federated filtering genes by counts ...")
            local_gene_counts_list = [client.get_local_gene_counts() for client in self.clients]
            global_gene_mask = aggregate_gene_counts(self.fed_config.preprocess.filter_gene_by_counts,
                                                     local_gene_counts_list)
            for client in self.clients:
                client.apply_global_gene_counts(global_gene_mask)
        if self.fed_config.preprocess.filter_cell_by_counts:
            self.logger.federated("Local filtering cells by counts ...")
            for client in self.clients:
                client.filter_cells()
        if self.fed_config.preprocess.normalize_total:
            self.logger.federated("Local normalization of total counts ...")
            for client in self.clients:
                client.total_normalization()
        if self.fed_config.preprocess.log1p:
            self.logger.federated("Local log1p transformation ...")
            for client in self.clients:
                client.log1p()
        if self.fed_config.preprocess.subset_hvg:
            self.logger.federated("Federated subset HVGs ...")
            # TODO: Validate the code
            local_hvg_stats = [client.get_local_hvg_stats() for client in self.clients]
            global_hvg_stats = aggregate_hvg_stats(local_hvg_stats)
            for client in self.clients:
                client.apply_hvg_stats(global_hvg_stats)
        self._federated_binning()

    def _federated_binning(self):
        if not self.fed_config.preprocess.binning:
            return
        if self.prep_mode == "fed-weight-avg":
            self._binning_weighted_avg_plain()
        elif self.prep_mode == "fed-weight-avg-smpc":
            self._binning_weighted_avg_smpc()
        elif self.prep_mode == "fed-hist":
            raise NotImplementedError(
                "prep_mode='fed-hist': plaintext histogram binning — "
                "see §3 in docs/methods/federated_binning.tex"
            )
        elif self.prep_mode == "fed-hist-smpc":
            raise NotImplementedError(
                "prep_mode='fed-hist-smpc': secure histogram binning — "
                "see §4 in docs/methods/federated_binning.tex"
            )
        else:
            raise ValueError(f"Unknown prep_mode for federated binning: {self.prep_mode!r}")

    def _binning_weighted_avg_plain(self):
        self.logger.federated("Federated binning (weighted-average, plaintext) ...")
        local_bin_edges_list = [client.get_local_bin_edges() for client in self.clients]
        global_bin_edges = aggregate_bin_edges(local_bin_edges_list)
        for client in self.clients:
            client.binning(global_bin_edges)

    def _binning_weighted_avg_smpc(self):
        self.logger.federated("Federated binning (weighted-average, SMPC) ...")
        n_shares = [client.get_local_binning_n_share() for client in self.clients]
        total_n_scaled = reveal_scaled_nonzero_total(n_shares)
        contrib_shares = [
            client.get_local_binning_contribution_share(total_n_scaled)
            for client in self.clients
        ]
        global_bin_edges = aggregate_bin_edge_contributions_smpc(contrib_shares)
        for client in self.clients:
            client.binning(global_bin_edges)

    def _preprocess_data_centralized(self):
        """Per-client scGPT centralized preprocessing.

        fed_config.preprocess.* boolean flags continue to gate which steps run;
        the numeric settings (target_sum, n_bins, hvg_flavor, ...) come from
        config.yml via each client's preprocessor instantiation.
        """
        fc = self.fed_config.preprocess
        for client in self.clients:
            pp = client.preprocessor
            pp.filter_gene_by_counts = pp.filter_gene_by_counts if fc.filter_gene_by_counts else False
            pp.filter_cell_by_counts = pp.filter_cell_by_counts if fc.filter_cell_by_counts else False
            pp.normalize_total       = pp.normalize_total       if fc.normalize_total       else False
            pp.log1p                 = pp.log1p                 if fc.log1p                 else False
            pp.subset_hvg            = pp.subset_hvg            if fc.subset_hvg            else False
            pp.binning               = pp.binning               if fc.binning               else None
            self.logger.federated(f"scGPT preprocessing for {client.log_id} ...")
            pp(client.adata, batch_key=None)

    def post_prep_setup(self):
        for client in self.clients:
            self.logger.federated(f"Setting up client {client.log_id} ...")
            client.post_prep(test_size=0)
            client.tokenize()
            client.instantiate_transformer_model()
            client.load_pretrained_model()
            client.setup_losses()