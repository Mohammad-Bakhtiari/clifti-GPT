#!/bin/bash

# Assign command-line arguments to variables
mode="$1"
dataset="$2"
reference_file="$3"
query_file="$4"
celltype_key="$5"
batch_key="$6"
gpu="$7"
n_epochs="${8-0}"
n_rounds="${9-0}"
agg_method="${10-fedavg}"
weighted="${11-true}"
smpc="${12-flase}"
mu="${13-0.01}"
prep_mode="${14-fed-weight-avg}"

if [[ "$agg_method" != "fedavg" && "$agg_method" != "fedprox" ]]; then
    echo "Invalid aggregation method. Use 'fedavg' or 'fedprox'."
    exit 1
fi

# Get the root directory, which is the parent directory of the current working directory
root_dir="$(dirname "$PWD")"

# Determine the general mode based on the mode provided
general_mode="centralized"
smpc_subdir=""
if [[ "$mode" == "cent_prep_fed_finetune" ]]; then
  general_mode="federated/cent-prep"
elif [[ "$mode" == *"federated"* ]]; then
  if [[ "$agg_method" == "fedprox" ]]; then
    general_mode="fedprox"
  elif [[ "$agg_method" == "fedavg" ]]; then
    general_mode="fedavg"
  else
    echo "Invalid aggregation method. Use 'fedavg' or 'fedprox'."
    exit 1
  fi
  if [ "$smpc" == "true" ]; then
    general_mode="${general_mode}-smpc"
  fi
  if [[ "$weighted" == "true" ]]; then
    general_mode="${general_mode}-weighted"
  fi
fi

if [ "$prep_mode" != "fed-weight-avg" ]; then
    general_mode="${general_mode}-prep-${prep_mode}"
fi

# Set up directory paths
data_dir="${root_dir}/data/scgpt/benchmark/${dataset}"
reference="${data_dir}/${reference_file}"
query="${data_dir}/${query_file}"
output="${root_dir}/output/annotation/${dataset}/${general_mode}"
INTI_WEIGHTS_DIR="${root_dir}/models/init"

cmd="python ${root_dir}/tasks/annotation.py \
 --dataset_name $dataset \
 --data-dir $data_dir \
 --reference_adata $reference \
 --query_adata $query \
 --output-dir $output \
 --celltype_key \"$celltype_key\" \
 --batch_key \"$batch_key\" \
 --mode $mode \
 --gpu $gpu \
 --pretrained_model_dir ${root_dir}/models/pretrained_models/scGPT_human \
 --config_file ${root_dir}/experiments/configs/annotation/config.yml \
 --fed_config_file ${root_dir}/experiments/configs/annotation/fed_config.yml \
 --init_weights_dir ${INTI_WEIGHTS_DIR}/${dataset}.pth"


# Add optional arguments if they are set
if [ "${#n_epochs}" != 0 ]; then
    cmd="$cmd --n_epochs $n_epochs"
fi

# Check and add --n_rounds if n_rounds is not empty and not "None"
if [ "${#n_rounds}" != 0 ]; then
    cmd="$cmd --n_rounds $n_rounds"
fi

if [ "$smpc" == "true" ]; then
    cmd="$cmd --smpc"
fi

if [ "$weighted" == "true" ]; then
    cmd+=" --weighted"
fi

if [ "$agg_method" == "fedprox" ]; then
    cmd="$cmd --use_fedprox --mu $mu"
fi

cmd="$cmd --prep_mode $prep_mode"

# Execute the command
CUDA_VISIBLE_DEVICES=$gpu \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
eval $cmd
