#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent27/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent27/logs/%x_%u_%j.err
#SBATCH -t 0-02:00

module load conda
conda activate c3

export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent27/wandb

cd /home/mcvstudent27/Week3

# Baseline run (no residual removed)
python main.py

# Ablation runs
NUM_RESIDUALS=16

for ((i=0; i<NUM_RESIDUALS; i++)); do
    echo "Running with residual $i disabled"
    python main.py --disable_residual $i
done
