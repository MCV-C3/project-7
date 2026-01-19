#!/bin/bash
#SBATCH -J arch_search_v2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.err
#SBATCH -t 1-12:00

# ============================================================================
# ARCHITECTURE SEARCH V2: COMPREHENSIVE GRID SEARCH
# ============================================================================
#
# IMPROVEMENTS FROM V1:
#   - Removed FC_HIDDEN_CONFIGS variation (256 vs 512 was negligible)
#   - Tests BOTH max and avg pooling to compare systematically
#   - Tests narrower architectures (extra_narrow [8,16,32,64])
#   - Tests 2, 3, 4 blocks instead of 3, 4, 5 (removed 5-block as it showed overfitting)
#   - Combines depth, width, and adaptive pooling in single experiment
#   - Fixed parsing error for train accuracy at best epoch
#
# EXPERIMENT DESIGN:
#   Total configurations: 72
#   
#   Depth (3 options):
#     - shallow: 2 blocks
#     - baseline: 3 blocks
#     - deep: 4 blocks
#
#   Width (3 options):
#     - extra_narrow: [8, 16, 32, 64]
#     - narrow: [16, 32, 64, 128]
#     - baseline: [32, 64, 128, 256]
#
#   Adaptive Pooling (8 configs, both max and avg):
#     - (5,5) max/avg with FC hidden (512)  → ~1.6M FC params (narrow)
#     - (3,3) max/avg with FC hidden (512)  → ~590k FC params (narrow)
#     - (1,1) max/avg with FC hidden (512)  → ~70k FC params (narrow)
#     - (1,1) max/avg direct classification → ~1k FC params (narrow)
#
# BREAKDOWN:
#   - For pool (5,5) and (3,3): 3 depths × 3 widths × 2 sizes × 2 types × 1 FC = 36 configs
#   - For pool (1,1): 3 depths × 3 widths × 1 size × 2 types × 2 FC = 36 configs
#   - Total: 72 configurations
#
# HYPOTHESIS:
#   - Narrower architectures generalize better on small datasets
#   - Aggressive adaptive pooling (1×1 GAP) reduces overfitting
#   - Direct classification from GAP provides strong regularization
#
#
# ============================================================================

echo "========================================================================"
echo "ARCHITECTURE SEARCH V2: COMPREHENSIVE GRID SEARCH"
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
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/Week4/wandb/arch_search_v2
export PYTHONUNBUFFERED=1

# Change to working directory
cd /home/mcvstudent29/Week4

# Create necessary directories
mkdir -p /data/uabmcv2526/mcvstudent29/Week4/logs
mkdir -p /data/uabmcv2526/mcvstudent29/Week4/wandb/arch_search_v2

# Run architecture search V2
echo "Starting architecture search V2..."
echo "Testing 72 configurations:"
echo "  - 3 depths (2, 3, 4 blocks)"
echo "  - 3 widths (extra_narrow, narrow, baseline)"
echo "  - 8 pooling configs (both max and avg)"
echo ""

python experiments/run_architecture_search_v2.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output/arch_search_v2 \
    --wandb_project C3_Week4 \
    --batch_size 16 \
    --epochs 20 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.3 \
    --seed 42 \
    --num_workers 8

echo ""
echo "========================================================================"
echo "Architecture search V2 completed!"
echo "End time: $(date)"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - /data/uabmcv2526/mcvstudent29/Week4/output/arch_search_v2/arch_search_v2_results.json"
echo "  - /data/uabmcv2526/mcvstudent29/Week4/output/arch_search_v2/arch_search_v2_summary.txt"
echo ""
echo "Check the summary file for:"
echo "  - Top 10 architectures by test accuracy"
echo "  - Analysis by depth, width, and pooling strategy"
echo "  - Overfitting analysis (train-test gap)"
echo "========================================================================"
