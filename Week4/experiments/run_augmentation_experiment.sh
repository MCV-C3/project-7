#!/bin/bash
#SBATCH -J aug_ablation
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 1-00:00
#SBATCH -o /data/uabmcv2526/mcvstudent27/Week4/logs/%x_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent27/Week4/logs/%x_%j.err

module load conda
conda activate c3

export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent27/Week4/wandb/augmentation
export PYTHONUNBUFFERED=1

cd /home/mcvstudent27/Week4

python experiments/run_augmentation_experiment.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent27/Week4/output/augmentation \
    --wandb_project C3_Week4_AUG \
    --model_type cbam_optimized \
    --batch_size 16 \
    --epochs 20 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.3 \
    --seed 42 \
    --num_workers 8 \
    --cbam_reduction 4 \
    --cbam_spatial_kernel 5 \
    --cbam_dilation 1 \
    --cbam_num_blocks 1 \