#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 0-12:00
#SBATCH -o /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.err
#SBATCH --job-name=mnasnet_progressive_ft

# ------------------ ENV SETUP ------------------
module load conda
conda activate c3

export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/wandb

cd /home/mcvstudent29/Week3

# ------------------ RUN LOOP ------------------
NUM_BLOCKS=17   # backbone.layers length (0..16)

for UNFREEZE in $(seq 0 $NUM_BLOCKS); do
    echo "=========================================="
    echo "Running experiment with UNFREEZE_BLOCKS=${UNFREEZE}"
    echo "=========================================="

    python main.py \
        --unfreeze_blocks ${UNFREEZE}

    echo "Finished UNFREEZE_BLOCKS=${UNFREEZE}"
done

echo "ALL EXPERIMENTS FINISHED"
