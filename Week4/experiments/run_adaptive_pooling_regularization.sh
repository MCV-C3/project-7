#!/bin/bash
#SBATCH -J adaptive_pool_reg
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.err
#SBATCH -t 0-03:00

# Load conda environment
module load conda
conda activate c3

# Set wandb environment variables
export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/Week4/wandb/adaptive_pooling_regularization

# Change to Week4 directory
cd /home/mcvstudent29/Week4

# ============================================================================
# ADAPTIVE POOLING REGULARIZATION SEARCH EXPERIMENT
# ============================================================================
#
# PROBLEM IDENTIFIED:
#   Previous architecture search showed that narrow architectures (fewer 
#   channels) performed better than wide one, suggesting overfitting is 
#   the primary bottleneck on this small dataset  (400 training images, 8 classes).
#
#   The narrow_baseline architecture uses AdaptiveAvgPool2d((7, 7)) which creates
#   6,272 features (128 channels × 7 × 7) before FC layers. This results in
#   ~3.2M parameters in just the FC layers (6,272 → 512 → 8), which is 
#   excessive for 400 training images and likely causes overfitting.
#
# HYPOTHESIS:
#   By reducing the spatial dimensions before FC layers (through more 
#   aggressive adaptive pooling AFTER all conv blocks), we can drastically 
#   reduce the FC parameter count, which will reduce overfitting and improve 
#   generalization on the test set.
#
#   The key insight is that spatial information after deep conv layers may be
#   redundant, and preserving it (7×7) only adds parameters without adding
#   discriminative power. Global pooling (1×1) forces the network to learn
#   more robust, spatially-invariant features.
#
# NOTE: This experiment varies ADAPTIVE POOLING (after all conv blocks),
#       NOT the MaxPool2d inside each conv block (which stays constant).
#
# EXPERIMENT DESIGN:
#   We test 8 configurations that systematically explore:
#
#   1. POOLING OUTPUT SIZES: How much spatial info to preserve
#      - (5,5): Light pooling → 3,200 features → ~1.6M FC params
#      - (3,3): Moderate pooling → 1,152 features → ~590k FC params
#      - (1,1): Global Average/Max Pooling → 128 features → ~70k FC params
#
#   2. POOLING TYPE: How to aggregate spatial information
#      - Max: Takes strongest activation (may be more discriminative)
#      - Avg: Averages all activations (smoother, less prone to outliers)
#
#   3. FC CONFIGURATION: How to map features to classes
#      - FC512: Hidden layer with 512 units (more capacity)
#      - Direct: No hidden layer, direct classification (aggressive reg.)
#        * Note: Only tested for (1,1) - direct classification with 3,200
#          features doesn't make architectural sense
#
# TESTED CONFIGURATIONS:
#
#   For (5,5) and (3,3): Only FC512 hidden layer (4 configs)
#     1. pool5x5_max_fc512: 5×5 max pooling → 3,200 features → 512 → 8
#     2. pool5x5_avg_fc512: 5×5 avg pooling → 3,200 features → 512 → 8
#     3. pool3x3_max_fc512: 3×3 max pooling → 1,152 features → 512 → 8
#     4. pool3x3_avg_fc512: 3×3 avg pooling → 1,152 features → 512 → 8
#
#   For (1,1): Both FC configs (4 configs)
#     5. gap_max_fc512: GAP max → 128 features → 512 → 8 (~70k FC params)
#     6. gap_avg_fc512: GAP avg → 128 features → 512 → 8 (~70k FC params)
#     7. gap_max_direct: GAP max → 128 features → 8 (~1k FC params)
#     8. gap_avg_direct: GAP avg → 128 features → 8 (~1k FC params)
#
# WHAT STAYS CONSTANT (Controlled Variables):
#   - Conv architecture: baseline_narrow [16, 32, 64, 128] (best from arch search)
#   - Learning rate: 1e-3 (AdamW)
#   - Batch size: 16
#   - Epochs: 20
#   - Weight decay: 1e-4
#   - Dropout: 0.3
#   - Seed: 42

#
# ============================================================================

echo ""
echo "======================================================================="
echo "ADAPTIVE POOLING REGULARIZATION SEARCH"
echo "======================================================================="
echo "Experiment: Systematic search of adaptive pooling strategies"
echo "Hypothesis: Reducing FC parameters via adaptive pooling reduces overfitting"
echo "Configurations: 8 (pooling size × type × FC config)"
echo "Architecture: baseline_narrow [16, 32, 64, 128]"
echo "======================================================================="
echo ""

python experiments/run_adaptive_pooling_regularization_search.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output/adaptive_pooling_regularization \
    --wandb_project C3_Week4 \
    --batch_size 16 \
    --epochs 20 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --seed 42 \
    --num_workers 8

echo ""
echo "======================================================================="
echo "EXPERIMENT COMPLETED"
echo "======================================================================="
echo "Results saved to: /data/uabmcv2526/mcvstudent29/Week4/output/adaptive_pooling_regularization/"
echo ""
echo "Check these files:"
echo "  - adaptive_pooling_regularization_results.json (detailed JSON results)"
echo "  - adaptive_pooling_regularization_summary.txt (human-readable summary)"
echo ""
echo "Compare against baseline: 74.13% val acc with 7×7 pooling"
echo "======================================================================="
echo ""
