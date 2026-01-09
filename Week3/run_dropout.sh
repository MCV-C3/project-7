#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 0-12:00
#SBATCH -o /data/uabmcv2526/mcvstudent27/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent27/logs/%x_%u_%j.err
#SBATCH --job-name=mnasnet_progressive_dropout

# ------------------ ENV SETUP ------------------
module load conda
conda activate c3

export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent27/wandb

cd /home/mcvstudent27/Week3

# ------------------ RUN LOOP ------------------
NUM_BLOCKS=17   # backbone.layers length (0..16)

for DROPOUT in $(seq 0 $NUM_BLOCKS); do
    echo "=========================================="
    echo "Running experiment with DROPOUT_BLOCKS=${DROPOUT}"
    echo "=========================================="

    python main.py \
        --unfreeze_blocks 7 \
        --dropout_blocks ${DROPOUT} \
        --dropout_value 0.2
    echo "Finished DROPOUT_BLOCKS=${DROPOUT}"
done

echo "ALL EXPERIMENTS FINISHED"
