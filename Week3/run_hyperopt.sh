#!/bin/bash
#SBATCH --job-name=hyperopt_week3
#SBATCH --ntasks=4
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=hyperopt_%j.out
#SBATCH --error=hyperopt_%j.err

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
conda activate mydl

# Activate virtual environment (if using one)
# source /path/to/your/venv/bin/activate

# Run hyperparameter optimization
python hyperparameter_optimization.py \
    --dataset_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/output/hyperopt/ \
    --n_trials 50 \
    --study_name week3_hyperopt_$(date +%Y%m%d_%H%M%S) \
    --wandb_project C3_Week3_HyperOpt

echo "Hyperparameter optimization complete!"
echo "Date: $(date)"
