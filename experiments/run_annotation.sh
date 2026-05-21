#!/bin/bash

source ./configs.sh

# Assign command-line arguments to variables
datasetnames="${1}"
mode="$2"
n_epochs="$3"
n_rounds="$4"
smpc="${5-false}"
GPU="${6-0}"
agg_method="${7-fedavg}"
weighted="${8-false}"
mu="${9-0}"
prep_mode="${10-fed-weight-avg}"

resolve_dataset_keys "$datasetnames"

chmod +x annotation.sh

echo "Running annotation for ${mode}"


for key in "${keys[@]}"; do
    echo -e "\e[32m******************************************\e[0m"
    echo -e "\e[32mRunning annotation for $key dataset [SMPC is ${smpc}]\e[0m"
    IFS='|' read -r -a args <<< "${datasets[$key]}"
    echo -e "\e[32mArguments: ${args[0]} ${args[1]} ${args[2]} ${args[3]} ${args[4]} ${args[5]}\e[0m"
    echo -e "\e[32m******************************************\e[0m"
    ./annotation.sh "${mode}" "${args[0]}" "${args[1]}" "${args[2]}" "${args[3]}" "${args[4]}" "${GPU}" "${n_epochs}" "${n_rounds}" "${agg_method}" "${weighted}" "${smpc}" "${mu}" "${prep_mode}"

    if [ $? -ne 0 ]; then
        echo -e "\e[31Error processing dataset $key. Please check the configuration.\e[0m"
        continue
    fi
done
