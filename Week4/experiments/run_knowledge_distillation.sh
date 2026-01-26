#!/bin/bash
#SBATCH -J cbam_search
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent27/Week4/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent27/Week4/logs/%x_%u_%j.err
#SBATCH -t 1-00:00

echo "========================================================================"
echo "KNOWLEDGE DISTILLATION EXPERIMENT"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"
echo ""

# Load conda environment
module load conda
conda activate c3

# Environment variables
export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent27/Week4/wandb/cbam_search
export PYTHONUNBUFFERED=1

# Change to project root
cd /home/mcvstudent27/Week4

# Create required directories
mkdir -p /data/uabmcv2526/mcvstudent27/Week4/logs
mkdir -p /data/uabmcv2526/mcvstudent27/Week4/wandb/knowledge_distillation

echo "Starting Knowledge Distillation..."
echo "Testing 18 configurations:"
echo "  - Temperature: 2, 4, 8"
echo "  - Alpha: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0"
echo ""

python experiments/run_knowledge_distillation.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent27/Week4/output/knowl_dist \
    --wandb_project C3_Week4_Knowledge_Distillation \
    --seed 42 \
    --num_workers 8 \
    --model_type student \
    --batch_size 16 \
    --epochs 20 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --momentum 0.9 \
    --optimizer AdamW \
    --dropout 0.3 \
    --cbam_reduction 4 \
    --cbam_spatial_kernel 5 \
    --cbam_dilation 1 \
    --cbam_num_blocks 1 \
    --use_flip \
    --use_color \
    --use_geometric \
    --use_translation \
    --aug_ratio 1.5 \
    --use_distillation \
    --teacher_model_type cbam_optimized \
    --teacher_checkpoint /data/uabmcv2526/mcvstudent27/Week4/output/augmentation/aug_flip_color_geometric_translation/aug_flip_color_geometric_translation_20260124_011348/best_model.pt


echo ""
echo "========================================================================"
echo "Knowledge distillation search completed!"
echo "End time: $(date)"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - /data/uabmcv2526/mcvstudent27/Week4/wandb/knowledge_distillation"
echo "  - progress.json (intermediate results)"
echo ""
echo "========================================================================"