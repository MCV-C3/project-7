#!/bin/bash
#SBATCH --job-name=hyperopt_week3
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH -o /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.err
#SBATCH -t 1-00:00

# Hyperparameter Optimization with Optuna
# This script runs hyperparameter optimization using 3-fold cross-validation
# Fixed parameters: --unfreeze_blocks 7, use_batchnorm_blocks=True
# Optimized parameters: batch_size, epochs, optimizer, learning_rate, momentum, weight_decay

echo "Starting hyperparameter optimization..."
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Current directory: $(pwd)"

# ------------------ ENV SETUP ------------------
module load conda

export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/wandb/hyperopt

cd /home/mcvstudent29/Week3

# Clear Python cache to ensure latest code is used
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true

# Run hyperparameter optimization
python hyperparameter_optimization.py \
    --dataset_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/output/hyperopt/ \
    --n_trials 50 \
    --study_name week3_hyperopt_$(date +%Y%m%d_%H%M%S) \
    --wandb_project C3_Week3_HyperOpt

echo "Hyperparameter optimization complete!"
echo "Date: $(date)"
