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
# OPTIMIZED BASELINE WITH CBAM ATTENTION + DATA AUGMENTATION
# Fixed Configuration for Hyperparameter Optimization Baseline
# ============================================================================
# Dataset: MIT Scenes with 50 images/class (400 total training images)
# 
# Architecture: CBAMOptimizedCNN with fixed attention parameters
# - Base channels: [16,32,64,128] (optimized from architecture search)
# - CBAM attention: Fixed optimal configuration
#   * Reduction ratio: 4 (channel attention efficiency)
#   * Spatial kernel: 5 (spatial attention coverage)
#   * Dilation: 1 (standard convolution)
#   * Num blocks: 1 (minimal attention overhead)
# - Adaptive pooling: (1,1) Global Average Pooling
# - FC config: Direct classification (128 → 8)
# 
# Data Augmentation: Fixed optimal configuration (1.5x augmentation ratio)
# - Horizontal flip: Enabled (geometric invariance)
# - Color jitter: Enabled (lighting robustness)
# - Geometric transforms: Enabled (rotation/scaling robustness)
# - Translation: Enabled (positional invariance)
# - Aug ratio: 1.5 (600 total training samples: 400 base + 200 augmented)
# 
# Training settings (standard hyperparameters):
# - batch_size=16, lr=1e-3, AdamW, weight_decay=1e-4, dropout=0.3, epochs=20
# 
# This configuration matches the fixed parameters used in hyperparameter optimization
# for consistent baseline evaluation with attention mechanisms and data augmentation.
# ============================================================================

python main.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output \
    --experiment_name cbam_baseline_with_augmentation \
    --wandb_project C3_Week4 \
    --model_type cbam_optimized \
    --batch_size 16 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --dropout 0.3 \
    --seed 42 \
    --num_workers 8 \
    --cbam_reduction 4 \
    --cbam_spatial_kernel 5 \
    --cbam_dilation 1 \
    --cbam_num_blocks 1 \
    --use_flip \
    --use_color \
    --use_geometric \
    --use_translation \
    --aug_ratio 1.5
    
# Configuration Summary:
# - Model: CBAMOptimizedCNN with fixed optimal CBAM parameters
# - Data Augmentation: All enabled with 1.5x ratio (600 total training samples)
# - Training: Standard hyperparameters from previous experiments
# - This matches the fixed configuration used in hyperparameter optimization
