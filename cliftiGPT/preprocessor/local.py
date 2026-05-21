import numpy as np
import torch
from scipy.sparse import issparse
import scanpy as sc
from anndata import AnnData
from scgpt.preprocess import Preprocessor as CentralPreprocessor
from typing import Dict, List, Tuple, Set
from scanpy.get import _get_obs_rep, _set_obs_rep
from scanpy.preprocessing._utils import _get_mean_var
from scanpy.preprocessing._highly_variable_genes import _mad
import pandas as pd


class Preprocessor(CentralPreprocessor):
    """
    A federated preprocessor that aggregates gene counts, normalizes total counts, and filters highly variable genes
    Gene count filtering: Filter genes that have cumulative counts less than the specified threshold across all clients

    """

    def __init__(self, log, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = log
        self.key_to_process = self.use_key
        # preliminary checks, will use later
        if self.key_to_process == "X":
            self.key_to_process = None  # the following scanpy apis use arg None to use X
        self.local_total_count = None

    @staticmethod
    def get_local_gene_set(adata: AnnData) -> Set[str]:
        return set(adata.var_names)

    @staticmethod
    def get_local_celltype_set(adata: AnnData, celltype_key: str) -> Set[str]:
        if adata.obs[celltype_key].dtype == "category":
            return set(adata.obs[celltype_key].cat.categories)
        print(f"adata.obs[{celltype_key}] is not categorical")
        return set(adata.obs[celltype_key].unique())

    @staticmethod
    def local_gene_counts(adata: AnnData) -> Dict[str, int]:
        gene_counts = np.asarray((adata.X > 0).sum(axis=0)).ravel()
        gene_names = adata.var_names
        gene_count_dict = dict(zip(gene_names, gene_counts))
        return gene_count_dict

    def apply_global_gene_counts(self, adata: AnnData, global_gene_mask: np.ndarray) -> AnnData:
        self.log("Filtering genes by counts ...")
        adata = adata[:, global_gene_mask]
        return adata

    def filter_cells(self, adata: AnnData) -> AnnData:
        if isinstance(self.filter_cell_by_counts, int) and self.filter_cell_by_counts > 0:
            self.log("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )
        return adata

    def total_normalization(self, adata: AnnData) -> AnnData:
        self.log("Normalizing total counts ...")
        normed = sc.pp.normalize_total(
            adata,
            target_sum=self.normalize_total,
            inplace=False
        )["X"]
        adata.layers[self.result_normed_key] = normed
        return adata

    def log1p(self, adata: AnnData) -> AnnData:
        if self.log1p:
            is_logged = self.check_logged(adata, obs_key=self.key_to_process)
            self.log("Log1p transforming ...")
            if is_logged:
                self.log(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=self.key_to_process),
                    layer=self.result_log1p_key,
                )
                self.key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=self.key_to_process)

    def compute_local_hvg_stats(self, adata: AnnData, layer: str = None) -> Dict:
        data = _get_obs_rep(adata, layer=layer)
        if self.hvg_flavor == "seurat_v3":
            means, variances = sc.pp._utils._get_mean_var(data)
            gene_variances_norm = variances / means
        elif self.hvg_flavor == "seurat":
            X = data.copy()
            if "log1p" in adata.uns_keys() and adata.uns["log1p"].get("base") is not None:
                X *= np.log(adata.uns["log1p"]["base"])
            if isinstance(X, np.ndarray):
                np.expm1(X, out=X)
            else:
                X = np.expm1(X)
            means, variances = _get_mean_var(X)
            dispersion = variances / means
            mean_bin = pd.cut(means, bins=20)
            disp_grouped = pd.DataFrame({'means': means, 'dispersions': dispersion}).groupby(mean_bin)['dispersions']
            disp_bin_stats = disp_grouped.agg(avg='mean', dev='std')
            norm_gene_var = (dispersion - disp_bin_stats.loc[mean_bin].avg) / disp_bin_stats.loc[mean_bin].dev
            gene_variances_norm = norm_gene_var
        elif self.hvg_flavor == "cell_ranger":
            means, variances = _get_mean_var(data)
            dispersion = variances / means
            mean_bin = pd.cut(means, bins=np.r_[-np.inf, np.percentile(means, np.arange(10, 105, 5)), np.inf])
            disp_grouped = pd.DataFrame({'means': means, 'dispersions': dispersion}).groupby(mean_bin)['dispersions']
            disp_bin_stats = disp_grouped.agg(avg='median', dev=_mad)
            norm_gene_var = (dispersion - disp_bin_stats.loc[mean_bin].avg) / disp_bin_stats.loc[mean_bin].dev
            gene_variances_norm = norm_gene_var
        else:
            raise ValueError(f"Unsupported flavor: {self.hvg_flavor}")

        return {
            'means': means,
            'variances': variances,
            'variances_norm': gene_variances_norm
        }

    def subset_hvgs(self, adata: AnnData, global_stats: Dict, n_top_genes: int) -> AnnData:
        df = pd.DataFrame({
            'means': global_stats['means'],
            'variances': global_stats['variances'],
            'variances_norm': global_stats['variances_norm']
        }, index=adata.var_names)

        df['highly_variable'] = False
        top_genes = df['variances_norm'].sort_values(ascending=False).head(n_top_genes).index
        df.loc[top_genes, 'highly_variable'] = True

        adata.var['highly_variable'] = df['highly_variable'].values
        adata.var['means'] = df['means'].values
        adata.var['variances'] = df['variances'].values
        adata.var['variances_norm'] = df['variances_norm'].values

        if self.subset_hvg:
            adata._inplace_subset_var(df['highly_variable'])

        return adata

    def compute_local_bin_edges(self, adata: AnnData) -> Tuple[np.ndarray, int]:
        self.log("Computing local bin edges ...")
        self.layer_data = _get_obs_rep(adata, layer=self.key_to_process)
        self.layer_data = self.layer_data.A if issparse(self.layer_data) else self.layer_data
        if self.layer_data.min() < 0:
            raise ValueError(f"Assuming non-negative data, but got min value {self.layer_data.min()}.")
        all_non_zero_values = self.layer_data[self.layer_data > 0]
        bin_edges = np.quantile(all_non_zero_values, np.linspace(0, 1, self.binning - 1))
        return bin_edges, len(all_non_zero_values)

    def _materialize_layer_data(self, adata: AnnData) -> np.ndarray:
        """Cache and return a dense non-negative view of the binning input layer."""
        self.layer_data = _get_obs_rep(adata, layer=self.key_to_process)
        self.layer_data = self.layer_data.A if issparse(self.layer_data) else self.layer_data
        if self.layer_data.min() < 0:
            raise ValueError(f"Assuming non-negative data, but got min value {self.layer_data.min()}.")
        return self.layer_data

    def compute_local_envelope_stats(self, adata: AnnData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Local envelope statistics for the secure-histogram binning path.

        Returns a torch float32 tensor pair ``(local_max, local_n_nonzero)`` ready
        to be wrapped in ``crypten.cryptensor`` by the client, matching the style
        of ``ClientEmbedder.report_n_local_samples``.
        """
        self.log("Computing local envelope stats ...")
        layer_data = self._materialize_layer_data(adata)
        nonzero = layer_data[layer_data > 0]
        if nonzero.size == 0:
            local_max = torch.tensor(0.0, dtype=torch.float32)
        else:
            local_max = torch.tensor(float(nonzero.max()), dtype=torch.float32)
        local_n_nonzero = torch.tensor(float(nonzero.size), dtype=torch.float32)
        return local_max, local_n_nonzero

    def compute_local_histogram(
        self, adata: AnnData, envelope_grid: np.ndarray
    ) -> torch.Tensor:
        """Local non-zero-value histogram over a shared envelope grid.

        Parameters
        ----------
        adata : AnnData
            Local AnnData to histogram.
        envelope_grid : np.ndarray
            Monotonically increasing grid of length ``M + 1`` shared across
            clients, starting at 0 and ending at the globally revealed
            ``max_global``. The returned histogram has length ``M`` and
            represents counts of non-zero values falling into each
            ``(envelope_grid[m], envelope_grid[m + 1]]`` interval, with the
            last interval inclusive on both ends to absorb the global maximum.

        Returns
        -------
        torch.Tensor
            Length-``M`` float32 count vector ready to be wrapped in
            ``crypten.cryptensor`` by the client.
        """
        self.log("Computing local histogram ...")
        if getattr(self, "layer_data", None) is None:
            self._materialize_layer_data(adata)
        layer_data = self.layer_data
        nonzero = layer_data[layer_data > 0]
        if envelope_grid.ndim != 1 or envelope_grid.size < 2:
            raise ValueError(
                f"envelope_grid must be a 1D array of length >= 2, "
                f"got shape {envelope_grid.shape}."
            )
        if not np.all(np.diff(envelope_grid) > 0):
            raise ValueError("envelope_grid must be strictly increasing.")
        counts, _ = np.histogram(nonzero, bins=envelope_grid)
        return torch.tensor(counts, dtype=torch.float32)

    def apply_binning(self, adata: AnnData, global_bin_edges: np.ndarray) -> AnnData:
        self.log("Applying binning ...")
        binned_data = np.zeros_like(self.layer_data, dtype=np.int64)

        for row_idx, row in enumerate(self.layer_data):
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            non_zero_digits = np.digitize(non_zero_row, global_bin_edges, right=True)
            non_zero_digits = np.clip(non_zero_digits, 1, self.binning - 1)
            binned_data[row_idx, non_zero_ids] = non_zero_digits

        adata.layers[self.result_binned_key] = binned_data
        adata.uns["bin_edges"] = global_bin_edges
        return adata
