#!/bin/bash

PROJECT_DIR="."

DATA_DIR="$PROJECT_DIR/data/torch"
MODEL_WEIGHTS="$PROJECT_DIR/data/models"
AE_WEIGHTS="$PROJECT_DIR/data/ae_checkpoints"
EXP_DIR="$PROJECT_DIR/data/experiments"

# Runs a sweep over different AE latent dimensions (corresponding to R=5,10,20), all datasets and number of clients. (Sec 5.3.1)

# Rand Top K
python3 src/orchestrator.py \
--model vit_b_32 \
--dataset cifar100 food101 sun397 femnist \
--batch_size 125 \
--nr_of_epochs 10 \
--split_layer 5 \
--gpu_id 0 \
--nr_of_clients 5 25 \
--dataset_split_type noniid \
--experiment_name RAND_TOP_K \
--compression_method rand_top_k \
--sparsity 0.15 0.065 0.025 \
--random_seed 42 43 44 \
--torch_data_dir $DATA_DIR \
--model_weights_dir $MODEL_WEIGHTS \
--ae_weights_dir $AE_WEIGHTS \
--experiments_dir $EXP_DIR

# C3-SL
# compression ratio also accounts for additional communication induced by leaving the CLS token uncompressed. Thus resulting in roughly R=5,10,20 for all methods.
python3 src/orchestrator.py \
--model vit_b_32 \
--dataset cifar100 food101 sun397 femnist \
--batch_size 125 \
--nr_of_epochs 10 \
--split_layer 5 \
--gpu_id 0 \
--nr_of_clients 5 25 \
--dataset_split_type noniid \
--experiment_name C3_SL \
--compression_method c3_sl \
--c3_sl_compression_ratio 5 12 34 \
--random_seed 42 43 44 \
--torch_data_dir $DATA_DIR \
--model_weights_dir $MODEL_WEIGHTS \
--ae_weights_dir $AE_WEIGHTS \
--experiments_dir $EXP_DIR

# ADC
# In the code, the sqrt of adc_compression_ratio is taken, resulting in the batch and token compression ratio (which is inverted for ADC).
# CLS token is compressed as this led to increased performance.
python3 src/orchestrator.py \
--model vit_b_32 \
--dataset cifar100 food101 sun397 femnist \
--batch_size 125 \
--nr_of_epochs 10 \
--split_layer 5 \
--gpu_id 0 \
--nr_of_clients 5 25 \
--dataset_split_type noniid \
--experiment_name ADC \
--compression_method adc \
--adc_compression_ratio 0.2 0.01 0.05 \
--random_seed 42 43 44 \
--torch_data_dir $DATA_DIR \
--model_weights_dir $MODEL_WEIGHTS \
--ae_weights_dir $AE_WEIGHTS \
--experiments_dir $EXP_DIR