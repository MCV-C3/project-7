#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/logs/%x_%u_%j.err
#SBATCH -t 0-12:00
#SBATCH --job-name=mnasnet_data_augmentation

# ------------------ ENV SETUP ------------------
module load conda
conda activate mydl


export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/wandb/data_augmentation

cd /home/mcvstudent29/Week3

# ------------------ EXPERIMENT SWEEP ------------------
AUG_TYPES=(none flip color geometric translation)
# Different augmentation ratios to test (0.0 = no augmentation, 1.0 = all data augmented)
AUG_RATIOS=(0.0 0.25 0.5 0.75)

for AUG in "${AUG_TYPES[@]}"; do
    if [ "${AUG}" = "none" ]; then
        echo "=========================================="
        echo "Running experiment: AUG_TYPE=none, AUG_RATIO=0.0, UNFREEZE_BLOCKS=7"
        echo "=========================================="

        python main.py \
            --unfreeze_blocks 7 \
            --reg_type none \
            --reg_lambda 0 \
            --l1_ratio 0.5 \
            --aug_type none \
            --aug_ratio 0.0

        echo "Finished AUG_TYPE=none, AUG_RATIO=0.0"
        echo ""
    else
        for RATIO in "${AUG_RATIOS[@]}"; do
            # Skip 0.0 ratio for augmentation types (already covered by "none")
            if [ "${RATIO}" != "0.0" ]; then
                echo "=========================================="
                echo "Running experiment: AUG_TYPE=${AUG}, AUG_RATIO=${RATIO}, UNFREEZE_BLOCKS=7"
                echo "=========================================="

                python main.py \
                    --unfreeze_blocks 7 \
                    --reg_type none \
                    --reg_lambda 0 \
                    --l1_ratio 0.5 \
                    --aug_type ${AUG} \
                    --aug_ratio ${RATIO}

                echo "Finished AUG_TYPE=${AUG}, AUG_RATIO=${RATIO}"
                echo ""
            fi
        done
    fi
done

echo "ALL DATA AUGMENTATION EXPERIMENTS FINISHED"