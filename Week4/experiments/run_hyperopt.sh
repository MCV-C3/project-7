#!/bin/bash
#SBATCH --job-name=hyperopt_week4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH -o /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.err
#SBATCH -t 1-00:00

# Hyperparameter Optimization with Optuna for Week 4 CNN
# This script runs hyperparameter optimization using cross-validation on train set
# and evaluates final performance on test set
# 
# Multi-objective optimization:
#   - Maximize test accuracy
#   - Minimize overfitting (train-test gap)
#
# Optimized parameters: batch_size, epochs, optimizer, learning_rate, momentum, weight_decay, dropout

echo "Starting Week 4 hyperparameter optimization..."
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Current directory: $(pwd)"

# ------------------ ENV SETUP ------------------
module load conda
conda activate c3

export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/Week4/wandb/hyperopt

cd /home/mcvstudent29/Week4/experiments

# Clear Python cache to ensure latest code is used
find .. -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find .. -type f -name '*.pyc' -delete 2>/dev/null || true

# Run hyperparameter optimization
python hyperparameter_optimization.py \
    --dataset_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output/hyperopt/ \
    --n_trials 60 \
    --study_name week4_hyperopt_$(date +%Y%m%d_%H%M%S) \
    --wandb_project C3_Week4_HyperOpt

echo "Hyperparameter optimization complete!"
echo "Date: $(date)"
