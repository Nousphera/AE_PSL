#!/bin/bash

PROJECT_DIR="."

DATA_DIR="$PROJECT_DIR/data/torch"
MODEL_WEIGHTS="$PROJECT_DIR/data/models"
AE_WEIGHTS="$PROJECT_DIR/data/ae_checkpoints"
EXP_DIR="$PROJECT_DIR/data/experiments"

# Runs a sweep over different AE latent dimensions (corresponding to R=5,10,20), all datasets and number of clients. (Sec 5.3.1)
python3 src/orchestrator.py \
--model vit_b_32 \
--dataset cifar100 food101 sun397 femnist \
--batch_size 125 \
--nr_of_epochs 10 \
--split_layer 5 \
--ae_latent_dim 140 64 24 \
--gpu_id 0 \
--nr_of_clients 5 25 \
--dataset_split_type noniid \
--experiment_name MAIN_RESULTS \
--compression_method ae \
--random_seed 42 43 44 \
--torch_data_dir $DATA_DIR \
--model_weights_dir $MODEL_WEIGHTS \
--ae_weights_dir $AE_WEIGHTS \
--experiments_dir $EXP_DIR