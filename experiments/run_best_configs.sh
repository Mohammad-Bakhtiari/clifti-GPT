#!/bin/bash

source ./configs.sh

prep_mode="${1-federated}"
GPU=0

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
    "MS|1|20|false|fedavg|0"
    "MS|3|3|true|fedavg|0"
    "MS|1|9|false|fedprox|0.01"
    "MS|3|6|true|fedprox|0.05"
    # CellLine (CL)
    "CellLine|1|7|false|fedavg|0"
    "CellLine|1|7|true|fedavg|0"
    "CellLine|1|7|false|fedprox|0.01"
    "CellLine|1|7|true|fedprox|0.01"
    # Lung
    "LUNG|1|11|false|fedavg|0"
    "LUNG|1|13|true|fedavg|0"
    "LUNG|1|6|false|fedprox|0.01"
    "LUNG|1|6|true|fedprox|0.01"
    # Myeloid
    "MYELOID-top5|1|3|false|fedavg|0"
    "MYELOID-top5|1|4|true|fedavg|0"
    "MYELOID-top5|1|4|false|fedprox|0.01"
    "MYELOID-top5|1|4|true|fedprox|0.01"
    # HP
    "HP5|3|17|false|fedavg|0"
    "HP5|1|156|true|fedavg|0"
    "HP5|1|17|false|fedprox|0.01"
    "HP5|1|52|true|fedprox|0.01"
)

for row in "${rows[@]}"; do
    IFS='|' read -r ds nE nR smpc agg mu <<< "$row"
    echo -e "\e[34m------------------------------------------------\e[0m"
    echo -e "\e[34m ${ds} | ${agg} | smpc=${smpc} | E=${nE} R=${nR} mu=${mu} | prep=${prep_mode}\e[0m"
    echo -e "\e[34m------------------------------------------------\e[0m"
    ./run_annotation.sh "$ds" federated_finetune "$nE" "$nR" "$smpc" "$GPU" "$agg" true "$mu" "$prep_mode"
done
