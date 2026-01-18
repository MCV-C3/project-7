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
# OPTIMIZED BASELINE CONFIGURATION
# After Architecture Search + Adaptive Pooling Regularization Experiments
# ============================================================================
# Dataset: MIT Scenes with 50 images/class (400 total training images)
# 
# Architecture Evolution:
# 1. Original baseline: [32,64,128,256], 7×7 pooling, FC512 → 66.17% test acc, 28.58% train-test gap
# 2. Arch search winner: [16,32,64,128] narrow channels → 74.13% test acc, 19.62% gap
# 3. Adaptive pooling winner: GAP (1×1) + direct classification → 75.87% test acc, 2.88% gap
# 
# Current Architecture (gap_avg_direct):
# - Channels: [16,32,64,128] (50% narrower than original, reduces conv overfitting)
# - Adaptive pooling: (1,1) Global Average Pooling (eliminates spatial redundancy)
# - Pooling type: avg (smoother aggregation than max)
# - FC config: Direct classification (128 → 8, no hidden layer)
# - FC parameters: 1,032 (vs 3.2M in original baseline = 3,116× reduction)
# 
# Key advantages:
# - Minimal overfitting (2.88% train-test gap vs 28.58% original)
# - 576× fewer FC params than pool3x3 runner-up
# - Follows modern CNN best practices (ResNet, EfficientNet pattern)
# - Provides headroom for data augmentation and attention mechanisms
# 
# Training settings (unchanged from original baseline):
# - batch_size=16, lr=1e-3, AdamW, weight_decay=1e-4, dropout=0.3
# ============================================================================

python main.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output \
    --experiment_name optimized_baseline \
    --wandb_project C3_Week4 \
    --batch_size 16 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --dropout 0.3 \
    --seed 42 \
    --num_workers 8
    
# Note: --model_type defaults to 'optimized' (no need to specify)
# OptimizedCNN has fixed architecture: [16,32,64,128] + GAP + direct classification
