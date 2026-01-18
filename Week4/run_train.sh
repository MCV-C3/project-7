#!/bin/bash
#SBATCH -J run_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.err
#SBATCH -t 0-02:00

# Load conda environment
module load conda
conda activate c3

# Set wandb environment variables
export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/Week4/wandb

# Change to Week4 directory
cd /home/mcvstudent29/Week4

# ============================================================================
# BASELINE CONFIGURATION - First Training Run
# ============================================================================
# Dataset: MIT Scenes with 50 images/class (400 total training images)
# Strategy: Conservative approach to avoid overfitting on small dataset
# 
# Baseline settings rationale:
# - batch_size=16: Good balance for 400 images (25 batches/epoch)
# - epochs=50: Enough to see full learning curve and convergence
# - learning_rate=1e-3: Standard starting point for AdamW
# - weight_decay=1e-4: Light L2 regularization to prevent overfitting
# - dropout=0.3: Moderate dropout in FC layers for regularization
# - NO scheduler: Keep constant LR for baseline simplicity
# ============================================================================

python main.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output \
    --experiment_name narrow_baseline \
    --wandb_project C3_Week4 \
    --batch_size 16 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --dropout 0.3 \
    --kernel_size 3 \
    --model_type flexible \
    --channels "16,32,64,128" \
    --fc_hidden 512 \
    --seed 42 \
    --num_workers 8
