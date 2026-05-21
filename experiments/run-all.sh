#!/bin/bash
GPU=1

chmod +x run_param_tuning.sh
# Arguments: datasetnames, aggregation method, weighted, smpc, N_ROUNDS, GPU, epochs_values
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running parameter tuning for FedAvg with SMPC\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_param_tuning.sh "HP,MYELOID-top4+rest,MS" fedavg true true 20 $GPU 1-5

echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running parameter tuning for FedAvg without SMPC\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_param_tuning.sh "HP,MYELOID-top4+rest,MS" fedavg true false 20 $GPU 1-5


echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running parameter tuning for FedProx with SMPC for MS dataset\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_param_tuning.sh MS fedprox true true 20 0 1-5




# scalability experiments
./run_annotation.sh MYELOID-top5 centralized_finetune_inference 20 0 false 0


./run_annotation.sh "MYELOID-top5" centralized_clients 20 0 false 0
./run_annotation.sh "MYELOID-top10" centralized_clients 20 0 false 1
./run_annotation.sh "MYELOID-top20" centralized_clients 20 0 false 2
./run_annotation.sh "MYELOID-top30" centralized_clients 20 0 false 3





./run_annotation.sh "MYELOID-top5" federated_finetune 1 20 true 0 fedprox true 0.01
./run_annotation.sh MYELOID-top10 federated_finetune 1 20 true 1 fedprox true 0.01
./run_annotation.sh "MYELOID-top20" federated_finetune 1 20 true 2 fedprox true 0.01
./run_annotation.sh "MYELOID-top30" federated_finetune 1 20 true 3 fedprox true 0.01



chmod +x run_annotation.sh
# Arguments datasetnames, mode, n_epochs, n_rounds, smpc, GPU, agg_method, weighted, mu
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running centralized annotation with scGPT\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_annotation.sh all centralized_finetune_inference 20 0 false $GPU

echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running local clients annotation with scGPT\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_annotation.sh all centralized_clients 20 0 false $GPU

echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running Federated annotation with cliftiGPT using weighted FedAvg\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_annotation.sh "LUNG,CellLine,COVID,COVID-corrected,COVID-fed-corrected" federated_finetune 1 20 false $GPU fedavg true


echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34mRunning Federated annotation with cliftiGPT using weighted FedAvg and SMPC\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_annotation.sh "LUNG,CellLine,COVID,COVID-corrected,COVID-fed-corrected" federated_finetune 1 20 true $GPU fedavg true

echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running Federated annotation with cliftiGPT using weighted FedProx aggregation \e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_annotation.sh "LUNG,CellLine,COVID,COVID-corrected,COVID-fed-corrected" federated_finetune 1 20 false $GPU fedprox true 0.01

echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running Federated annotation with cliftiGPT using weighted FedProx aggregation and SMPC \e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_annotation.sh "LUNG,CellLine,COVID,COVID-corrected,COVID-fed-corrected" federated_finetune 1 20 true $GPU fedprox true 0.01


chmod +x run_embedding.sh
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running embedding for all modes\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34mRunning embedding for centralized\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_embedding.sh all centralized false $GPU
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34mRunning embedding for federated without SMPC\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_embedding.sh all federated_zeroshot false $GPU
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34mRunning embedding for federated with SMPC\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_embedding.sh all federated_zeroshot true $GPU
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34mRunning embedding for clients local training\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_embedding.sh all centralized_clients false $GPU

chmod +x run_prep_mode_comparison.sh
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running best-config annotation comparison across prep_mode\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_prep_mode_comparison.sh fed-weight-avg
./run_prep_mode_comparison.sh centralized
./run_prep_mode_comparison.sh fed-weight-avg-smpc
# fed-hist, fed-hist-smpc: placeholders in FedAnnotator._federated_binning