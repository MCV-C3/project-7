#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.err
#SBATCH -t 0-06:00
#SBATCH --job-name=best_hyperparams

# Run the best hyperparameter configurations from Optuna optimization
# Based on recommended_configs.json results

# ------------------ ENV SETUP ------------------
module load conda
conda activate mydl

export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/wandb/best_hyperparams

cd /home/mcvstudent29/Week3

# Clear Python cache to ensure latest code is used
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true

# ------------------ CONFIGURATION 1: BEST ACCURACY ------------------
# Trial 38 from hyperparameter optimization
# Val Accuracy: 0.8725, Overfitting: 0.1075
echo "=========================================="
echo "Running BEST ACCURACY configuration (Trial 38)"
echo "Val Acc: 0.8725, Overfitting: 0.1075"
echo "=========================================="

python main.py \
    --experiment_name "trial_38_best_accuracy" \
    --unfreeze_blocks 7 \
    --batch_size 8 \
    --epochs 20 \
    --optimizer AdamW \
    --learning_rate 0.0002847453902902796 \
    --weight_decay 8.870445015199785e-05 \
    --reg_type none \
    --reg_lambda 0.0 \
    --l1_ratio 0.5 \
    --aug_type none \
    --aug_ratio 0.0

echo "Finished BEST ACCURACY configuration"
echo ""

# ------------------ CONFIGURATION 2: RECOMMENDED BALANCED ------------------
# Trial 34 from hyperparameter optimization
# Val Accuracy: 0.7723, Overfitting: -0.0395 (slight underfitting)
echo "=========================================="
echo "Running RECOMMENDED BALANCED configuration (Trial 34)"
echo "Val Acc: 0.7723, Overfitting: -0.0395"
echo "=========================================="

python main.py \
    --experiment_name "trial_34_recommended_balanced" \
    --unfreeze_blocks 7 \
    --batch_size 8 \
    --epochs 20 \
    --optimizer AdamW \
    --learning_rate 0.0010053707605371044 \
    --weight_decay 0.0002193566409866228 \
    --reg_type none \
    --reg_lambda 0.0 \
    --l1_ratio 0.5 \
    --aug_type none \
    --aug_ratio 0.0

echo "Finished RECOMMENDED BALANCED configuration"
echo ""

echo "=========================================="
echo "ALL BEST HYPERPARAMETER EXPERIMENTS FINISHED"
echo "=========================================="
