#!/bin/bash

PROJECT_DIR="."

DATA_DIR="$PROJECT_DIR/data/torch"
MODEL_WEIGHTS="$PROJECT_DIR/data/models"
AE_WEIGHTS="$PROJECT_DIR/data/ae_checkpoints"
EXP_DIR="$PROJECT_DIR/data/experiments"

# Runs a sweep over different AE architectures (Sec 5.3.3)
python3 src/orchestrator.py \
--model vit_b_32 \
--dataset food101 \
--batch_size 125 \
--nr_of_epochs 10 \
--split_layer 5 \
--ae_latent_dim 192 96 48 32 24 16 \
--ae_force_retrain true \
--ae_type "1_layer_CONV 1_layer_MLP_(d_d_z) 2_layer_MLP_(d_d_4_d_z) 2_layer_MLP_(d_d_d_z) 3_layer_MLP_(d_d_2_d_4_d_z)" \
--gpu_id 0 \
--nr_of_clients 1 \
--dataset_split_type noniid \
--experiment_name AE_ARCHITECTURE \
--compression_method ae \
--random_seed 42 43 44 \
--torch_data_dir $DATA_DIR \
--model_weights_dir $MODEL_WEIGHTS \
--ae_weights_dir $AE_WEIGHTS \
--experiments_dir $EXP_DIR