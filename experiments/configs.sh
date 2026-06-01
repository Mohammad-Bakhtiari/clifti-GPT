#!/bin/bash

declare -A datasets
datasets["MS"]="ms|reference_annot.h5ad|query_annot.h5ad|Factor Value[inferred cell type - authors labels]|split_label|index"
datasets["HP5"]="hp5|reference.h5ad|query.h5ad|Celltype|batch_name|index"
datasets["LUNG"]="lung|reference_annot.h5ad|query_annot.h5ad|cell_type|sample|gene_name"
datasets["CellLine"]="cl|reference.h5ad|query.h5ad|cell_type|batch|index"
datasets["COVID"]="covid|reference-raw.h5ad|query-raw.h5ad|celltype|batch_group|gene_name"
datasets["COVID-corrected"]="covid-corrected|reference_corrected.h5ad|query_corrected.h5ad|celltype|batch_group|gene_name"


# Scalability experiments
datasets["MYELOID-top5"]="myeloid-top5|reference.h5ad|query.h5ad|cell_type|combined_batch|index"
datasets["MYELOID-top10"]="myeloid-top10|reference.h5ad|query.h5ad|cell_type|combined_batch|index"
datasets["MYELOID-top20"]="myeloid-top20|reference.h5ad|query.h5ad|cell_type|combined_batch|index"
datasets["MYELOID-top30"]="myeloid-top30|reference.h5ad|query.h5ad|cell_type|combined_batch|index"



# ----------------------
# Dataset selection logic
# Usage:
#   resolve_dataset_keys "dataset1,dataset2"
#   resolve_dataset_keys "all"
# Result:
#   Sets `keys` array variable with valid dataset names.
# ----------------------
resolve_dataset_keys() {
    local datasetnames="$1"
    IFS=',' read -ra input_keys <<< "$datasetnames"

    if [[ "$datasetnames" != "all" ]]; then
        for key in "${input_keys[@]}"; do
            if [[ -z "${datasets[$key]}" ]]; then
                echo "❌ Dataset \"$key\" not found."
                echo "✅ Available datasets: ${!datasets[@]}"
                exit 1
            fi
        done
        keys=("${input_keys[@]}")
        echo "✅ Selected datasets: ${keys[*]}"
    else
        keys=("${!datasets[@]}")
    fi
}
