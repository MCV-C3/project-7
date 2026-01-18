#!/bin/bash
#SBATCH -J arch_search
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.err
#SBATCH -t 0-12:00

# Script: Architecture Search - Depth, Width & FC Layer Grid Search
# Description: Tests 18 combinations of depth (3/4/5), width (narrow/baseline/wide), and FC size (256/512)
# Expected runtime: ~12 hours for 18 trials at 20 epochs each

echo "========================================================================"
echo "ARCHITECTURE SEARCH: DEPTH, WIDTH & FC LAYER GRID SEARCH"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"
echo ""

# Load conda environment
module load conda
conda activate c3

# Set environment variables
export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/Week4/wandb/arch_search
export PYTHONUNBUFFERED=1

# Change to working directory
cd /home/mcvstudent29/Week4

# Create logs directory if it doesn't exist
mkdir -p /data/uabmcv2526/mcvstudent29/Week4/logs

# Run architecture search
echo "Starting architecture search..."
echo ""

python experiments/run_architecture_search.py \
    --batch_size 16 \
    --epochs 20 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.3 \
    --optimizer AdamW \
    --seed 42 \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output/arch_search

echo ""
echo "========================================================================"
echo "Architecture search completed!"
echo "End time: $(date)"
echo "========================================================================"
