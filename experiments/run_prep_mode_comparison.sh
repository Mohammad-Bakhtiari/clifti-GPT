#!/bin/bash

source ./configs.sh

prep_mode="${1-federated}"
GPU=0
datasetname="${2-all}"



if [[ "$prep_mode" != "federated" && "$prep_mode" != "centralized" && "$prep_mode" != "smpc" ]]; then
    echo "Invalid prep_mode '$prep_mode'. Use 'federated', 'centralized', or 'smpc'."
    exit 1
fi

chmod +x run_annotation.sh

echo -e "\e[34m================================================\e[0m"
echo -e "\e[34m Best-config replay  | prep_mode=${prep_mode}\e[0m"
echo -e "\e[34m================================================\e[0m"

# rows: dataset|n_epochs|n_rounds|smpc|agg|mu
rows=(
    # MS
    "MS|3|3|true|fedavg|0"
    # CellLine (CL)
    "CellLine|1|7|true|fedavg|0"
    # Lung
    "LUNG|1|13|true|fedavg|0"
    # Myeloid
    "MYELOID-top5|1|4|true|fedavg|0"
    # HP
    "HP5|1|52|true|fedprox|0.01"
)

run_best_config_row() {
    local row="$1"
    local ds nE nR smpc agg mu
    IFS='|' read -r ds nE nR smpc agg mu <<< "$row"
    echo -e "\e[34m------------------------------------------------\e[0m"
    echo -e "\e[34m ${ds} | ${agg} | smpc=${smpc} | E=${nE} R=${nR} mu=${mu} | prep=${prep_mode}\e[0m"
    echo -e "\e[34m------------------------------------------------\e[0m"
    ./run_annotation.sh "$ds" federated_finetune "$nE" "$nR" "$smpc" "$GPU" "$agg" true "$mu" "$prep_mode"
}

# check if the dataset name is valid by comparing to rows or is "all"
if [[ "$datasetname" != "all" ]]; then
    found=false
    for row in "${rows[@]}"; do
        if [[ "$row" == "${datasetname}|"* ]]; then
            found=true
            break
        fi
    done
    if [[ "$found" != true ]]; then
        echo "Invalid dataset name '$datasetname'. Use 'MS', 'CellLine', 'LUNG', 'MYELOID-top5', 'HP5', or 'all'."
        exit 1
    fi
fi

if [[ "$datasetname" == "all" ]]; then
    for row in "${rows[@]}"; do
        run_best_config_row "$row"
    done
else
    for row in "${rows[@]}"; do
        if [[ "$row" == "${datasetname}|"* ]]; then
            run_best_config_row "$row"
            break
        fi
    done
fi