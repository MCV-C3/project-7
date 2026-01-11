#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 0-12:00
#SBATCH -o %x_%u_%j.out
#SBATCH -e %x_%u_%j.err
#SBATCH --job-name=mnasnet_progressive_ft

# ------------------ ENV SETUP ------------------
module load conda
conda activate c3

export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/wandb

cd /home/mcvstudent29/Week3

# ------------------ PARAMETERS ------------------
NUM_BLOCKS=17             # backbone.layers length (0..16)
DROPOUT_BLOCKS=0          # si quieres probar dropout en bloques
DROPOUT_VALUE=0.5         # valor de dropout
REG_TYPE="none"           # none, l1, l2, elastic
REG_LAMBDA=0.0
L1_RATIO=0.5

# ------------------ RUN LOOP ------------------
for UNFREEZE in $(seq 0 $NUM_BLOCKS); do
    echo "=========================================="
    echo "Running experiment with UNFREEZE_BLOCKS=${UNFREEZE}"
    echo "=========================================="

    python main.py \
        --unfreeze_blocks ${UNFREEZE} \
        --dropout_blocks ${DROPOUT_BLOCKS} \
        --dropout_value ${DROPOUT_VALUE} \
        --reg_type ${REG_TYPE} \
        --reg_lambda ${REG_LAMBDA} \
        --l1_ratio ${L1_RATIO}

    echo "Finished UNFREEZE_BLOCKS=${UNFREEZE}"
done

echo "ALL EXPERIMENTS FINISHED"
