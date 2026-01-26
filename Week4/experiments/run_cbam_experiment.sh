#!/bin/bash
#SBATCH -J cbam_search
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent27/Week4/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent27/Week4/logs/%x_%u_%j.err
#SBATCH -t 1-00:00

# ============================================================================
# CBAM HYPERPARAMETER SEARCH
# ============================================================================
#
# EXPERIMENT GOAL:
#   Systematic evaluation of CBAM (Convolutional Block Attention Module)
#   hyperparameters on the optimized CNN architecture.
#
# SEARCH SPACE:
#
#   Channel Attention (reduction ratio):
#     - 16
#     - 8
#     - 4
#
#   Spatial Attention:
#     - kernel=7, dilation=1
#     - kernel=5, dilation=1
#     - kernel=3, dilation=1
#     - kernel=5, dilation=2
#     - kernel=3, dilation=3
#
#   Number of CBAM Blocks:
#     - 1
#     - 2
#     - 3
#     - 4
#
# TOTAL CONFIGURATIONS:
#   3 (channel reductions) × 5 (spatial configs) × 4 (num blocks) = 60 experiments
#
# NOTES:
#   - Architecture is fixed (CBAMOptimizedCNN)
#   - CBAM applied before pooling in each conv block
#   - Global Average Pooling + direct classification
#
# HYPOTHESIS:
#   - Lower reduction ratios (r=4,8) improve fine-grained channel modeling
#   - Dilated spatial attention captures broader context without extra pooling
#   - Excessive dilation may degrade localization
#
# ============================================================================

echo "========================================================================"
echo "CBAM HYPERPARAMETER SEARCH"
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
mkdir -p /data/uabmcv2526/mcvstudent27/Week4/wandb/cbam_search

# --------------------------------------------------------------------------
# RUN CBAM SEARCH
# --------------------------------------------------------------------------

echo "Starting CBAM hyperparameter search..."
echo "Testing 15 configurations:"
echo "  - Channel reduction: 16, 8, 4"
echo "  - Spatial attention:"
echo "      * kernel=7,5,3 (dilation=1)"
echo "      * kernel=5 (dilation=2)"
echo "      * kernel=3 (dilation=3)"
echo "  - Number of CBAM blocks: 1, 2, 3, 4"
echo ""

python experiments/run_cbam_experiment.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent27/Week4/output/cbam_search \
    --wandb_project C3_Week4_CBAM \
    --batch_size 16 \
    --epochs 20 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.3 \
    --seed 42 \
    --num_workers 8

echo ""
echo "========================================================================"
echo "CBAM hyperparameter search completed!"
echo "End time: $(date)"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - /data/uabmcv2526/mcvstudent27/Week4/output/cbam_search/cbam_results.json"
echo "  - progress.json (intermediate results)"
echo ""
echo "Check logs for:"
echo "  - Per-configuration accuracy"
echo "  - Reduction vs dilation trade-offs"
echo "========================================================================"
