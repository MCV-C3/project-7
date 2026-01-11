#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.err
#SBATCH -t 0-12:00
#SBATCH --job-name=mnasnet_regularizers

# ------------------ ENV SETUP ------------------
module load conda
conda activate mydl

export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/wandb

cd /home/mcvstudent29/Week3

# ------------------ EXPERIMENT SWEEP ------------------
REG_TYPES=(none l1 l2 elastic)
# Lambdas to try (0 included as baseline)
LAMBDAS=(0 1e-5 1e-4 1e-3 1e-2)
# L1 ratios for elastic net experiments
L1_RATIOS=(0.1 0.5 0.9)

for REG in "${REG_TYPES[@]}"; do
    if [ "${REG}" = "none" ]; then
        echo "=========================================="
        echo "Running experiment: REG_TYPE=none, REG_LAMBDA=0, UNFREEZE_BLOCKS=7"
        echo "=========================================="

        python main.py \
            --unfreeze_blocks 7 \
            --reg_type none \
            --reg_lambda 0 \
            --l1_ratio 0.5

        echo "Finished REG_TYPE=none, REG_LAMBDA=0"
        echo ""
    else
        for L in "${LAMBDAS[@]}"; do
            if [ "$L" = "0" ]; then
                continue
            fi
            if [ "${REG}" = "elastic" ]; then
                for A in "${L1_RATIOS[@]}"; do
                    echo "=========================================="
                    echo "Running experiment: REG_TYPE=${REG}, REG_LAMBDA=${L}, L1_RATIO=${A}, UNFREEZE_BLOCKS=7"
                    echo "=========================================="

                    python main.py \
                        --unfreeze_blocks 7 \
                        --reg_type ${REG} \
                        --reg_lambda ${L} \
                        --l1_ratio ${A}

                    echo "Finished REG_TYPE=${REG}, REG_LAMBDA=${L}, L1_RATIO=${A}"
                    echo ""
                done
            else
                echo "=========================================="
                echo "Running experiment: REG_TYPE=${REG}, REG_LAMBDA=${L}, UNFREEZE_BLOCKS=7"
                echo "=========================================="

                python main.py \
                    --unfreeze_blocks 7 \
                    --reg_type ${REG} \
                    --reg_lambda ${L} \
                    --l1_ratio 0.5

                echo "Finished REG_TYPE=${REG}, REG_LAMBDA=${L}"
                echo ""
            fi
        done
    fi
done

echo "ALL REGULARIZER EXPERIMENTS FINISHED"