#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.err
#SBATCH -t 0-02:00

# Load conda environment
module load conda
conda activate c3

export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/wandb

cd /home/mcvstudent29/Week3

python main.py \
        --unfreeze_blocks 7 \