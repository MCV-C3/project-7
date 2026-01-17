#!/bin/bash
#SBATCH -J run_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.err
#SBATCH -t 0-04:00

# Load conda environment
module load conda
conda activate c3

# Set wandb environment variables
export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/Week4/wandb

# Change to Week4 directory
cd /home/mcvstudent29/Week4

# Run training with baseline configuration
# Output structure: output/first_run/first_run_20260117_180544/
python main.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output \
    --experiment_name first_run \
    --wandb_project C3_Week4 \
    --batch_size 16 \
    --epochs 20 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --dropout 0.25 \
    --use_scheduler \
    --seed 42 \
    --num_workers 8
