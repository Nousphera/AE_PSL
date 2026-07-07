#!/bin/bash

PROJECT_DIR="."

DATA_DIR="$PROJECT_DIR/data/torch"
MODEL_WEIGHTS="$PROJECT_DIR/data/models"
AE_WEIGHTS="$PROJECT_DIR/data/ae_checkpoints"
EXP_DIR="$PROJECT_DIR/data/experiments"

# Runs experiments for the different design choices. See paper for more details on GA, CSA and FZ
# Randomly initialized AE
python3 src/orchestrator.py \
--ae_general_alignment false \
--client_specific_alignment false \
--ae_freeze_encoder_during_finetuning false \
--ae_freeze_decoder_during_finetuning false \
--model vit_b_32 \
--dataset cifar100 food101 sun397 femnist \
--batch_size 125 \
--nr_of_epochs 10 \
--split_layer 5 \
--ae_latent_dim 36 \
--gpu_id 0 \
--nr_of_clients 5 25 \
--dataset_split_type noniid \
--experiment_name ABLATION__AE \
--compression_method ae \
--random_seed 42 43 44 \
--torch_data_dir $DATA_DIR \
--model_weights_dir $MODEL_WEIGHTS \
--ae_weights_dir $AE_WEIGHTS \
--experiments_dir $EXP_DIR

# AE trained using General Alignment
python3 src/orchestrator.py \
--ae_general_alignment true \
--client_specific_alignment false \
--ae_freeze_encoder_during_finetuning false \
--ae_freeze_decoder_during_finetuning false \
--model vit_b_32 \
--dataset cifar100 food101 sun397 femnist \
--batch_size 125 \
--nr_of_epochs 10 \
--split_layer 5 \
--ae_latent_dim 36 \
--gpu_id 0 \
--nr_of_clients 5 25 \
--dataset_split_type noniid \
--experiment_name ABLATION__AE_GA \
--compression_method ae \
--random_seed 42 43 44 \
--torch_data_dir $DATA_DIR \
--model_weights_dir $MODEL_WEIGHTS \
--ae_weights_dir $AE_WEIGHTS \
--experiments_dir $EXP_DIR

# AE trained using General Alignment and freezing during DFT
python3 src/orchestrator.py \
--ae_general_alignment true \
--client_specific_alignment false \
--ae_freeze_encoder_during_finetuning true \
--ae_freeze_decoder_during_finetuning true \
--model vit_b_32 \
--dataset cifar100 food101 sun397 femnist \
--batch_size 125 \
--nr_of_epochs 10 \
--split_layer 5 \
--ae_latent_dim 36 \
--gpu_id 0 \
--nr_of_clients 5 25 \
--dataset_split_type noniid \
--experiment_name ABLATION__AE_GA_FZ \
--compression_method ae \
--random_seed 42 43 44 \
--torch_data_dir $DATA_DIR \
--model_weights_dir $MODEL_WEIGHTS \
--ae_weights_dir $AE_WEIGHTS \
--experiments_dir $EXP_DIR

# AE trained using General Alignment, Client Specific Alignment and freezing during DFT (AE-PSL)
python3 src/orchestrator.py \
--ae_general_alignment true \
--client_specific_alignment true \
--ae_freeze_encoder_during_finetuning true \
--ae_freeze_decoder_during_finetuning true \
--model vit_b_32 \
--dataset cifar100 food101 sun397 femnist \
--batch_size 125 \
--nr_of_epochs 10 \
--split_layer 5 \
--ae_latent_dim 36 \
--gpu_id 0 \
--nr_of_clients 5 25 \
--dataset_split_type noniid \
--experiment_name ABLATION__AE_GA_CSA_FZ \
--compression_method ae \
--random_seed 42 43 44 \
--torch_data_dir $DATA_DIR \
--model_weights_dir $MODEL_WEIGHTS \
--ae_weights_dir $AE_WEIGHTS \
--experiments_dir $EXP_DIR

