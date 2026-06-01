from pathlib import Path

import anndata
import pandas as pd
from prep_batch_effect_correction import calc_umap


def assign_topN_clients(ref_adata, N, batch_map, rest=False):
    mapped_ids = ref_adata.obs['batch'].map(batch_map)
    topN_ids = [str(i + 1) for i in range(N - int(rest))]
    ref_adata.obs['combined_batch'] = mapped_ids.apply(lambda x: x if x in topN_ids else 'rest')
    return ref_adata


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent / "scgpt" / "benchmark"
    celltype_key = 'cell_type'
    ref = anndata.read_h5ad(root_dir / "myeloid" / "reference_adata.h5ad")
    query = anndata.read_h5ad(root_dir / "myeloid" / "query_adata.h5ad")

    # Combine ref and query temporarily to compute global top 30 batches
    adata = ref.concatenate(query, batch_key=None)
    calc_umap(adata)
    adata.var = ref.var
    # Identify top 30 batches based on frequency
    top30_batches = adata.obs['batch'].value_counts().head(30).index.tolist()

    # Assign numerical IDs to top 30 batches: '1', '2', ..., '30'
    batch_map = {b: str(i + 1) for i, b in enumerate(top30_batches)}

    # Build reference and query from original data (not the combined one)
    reference = adata[adata.obs['batch'].isin(top30_batches)].copy()
    reference.var = adata.var
    query = adata[~adata.obs['batch'].isin(top30_batches)].copy()
    query.obs['combined_batch'] = query.obs['batch']  # keep original
    query.var = adata.var
    unique_cts = adata.obs[celltype_key].cat.categories.tolist()
    query.obs[celltype_key] = pd.Categorical(query.obs[celltype_key], categories=unique_cts)
    reference.obs[celltype_key] = pd.Categorical(reference.obs[celltype_key], categories=unique_cts)

    for n in [5, 10, 20, 30]:
        assigned_ref = assign_topN_clients(reference, n, batch_map, rest=n!=30)
        out_dir = root_dir / f"myeloid-top{n}"
        out_dir.mkdir(parents=True, exist_ok=True)

        assigned_ref.write_h5ad(out_dir / "reference.h5ad")
        query.write_h5ad(out_dir / "query.h5ad")
        print(f"Wrote myeloid-top{n}: {out_dir / 'reference.h5ad'}, {out_dir / 'query.h5ad'}")
