#!/bin/bash
#SBATCH -J attention
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.out
#SBATCH -e /data/uabmcv2526/mcvstudent29/Week4/logs/%x_%u_%j.err
#SBATCH -t 0-02:00

# Load conda environment
module load conda
conda activate c3

# Set wandb environment variables
export WANDB_MODE=offline
export WANDB_DIR=/data/uabmcv2526/mcvstudent29/Week4/wandb/attention

# Change to Week4 directory
cd /home/mcvstudent29/Week4

# ============================================================================
# CBAM (CONVOLUTIONAL BLOCK ATTENTION MODULE) EXPERIMENT
# ============================================================================
#
# BASELINE CONTEXT:
#   OptimizedCNN achieved:
#   - Test accuracy: 75.87%
#   - Train-test gap: 2.88% (minimal overfitting)
#   - Parameters: 98,952 total
#   - Architecture: [16,32,64,128] channels + GAP + direct classification
#
# MOTIVATION:
#   With minimal overfitting (2.88% gap), we have headroom to add model capacity
#   through attention mechanisms. CBAM provides dual attention (channel + spatial)
#   which is particularly effective for scene recognition tasks.
#
# CBAM MECHANISM:
#   CBAM sequentially applies two attention modules BEFORE pooling:
#   1. Channel Attention:
#      - Uses both avg & max pooling (richer than SE's avg-only)
#      - Learns 'what' is meaningful
#   2. Spatial Attention:
#      - Aggregates channel info via avg & max along channel axis
#      - 7×7 conv learns 'where' is meaningful
#      - Requires spatial resolution → applied BEFORE pooling
#
#   Benefits over SE:
#   - Dual attention: both 'what' (channel) and 'where' (spatial)
#   - Better spatial awareness for scene recognition
#   - Richer feature aggregation (avg + max pooling)
#
# ARCHITECTURE: CBAMOptimizedCNN
#   - Base: Same as OptimizedCNN ([16,32,64,128] + GAP + direct classification)
#   - Addition: CBAM blocks BEFORE pooling in each of 4 conv blocks
#   - Channel reduction ratio: 4 (standard for small networks)
#   - Spatial kernel size: 7 (standard CBAM configuration)
#   - Order: Conv → BN → ReLU → CBAM → Pool
#
# PARAMETER OVERHEAD:
#   Channel Attention (per block): Uses Conv2d instead of Linear
#   Spatial Attention (per block): 2-channel conv with 7×7 kernel
#   
#   Total CBAM overhead: ~11,000-12,000 params (11-12% increase)
#   Total model: ~110,000 params (vs 98,952 baseline)
#
# PLACEMENT RATIONALE:
#   - CBAM BEFORE pooling (not after like SE)
#   - Spatial attention needs spatial resolution to be effective
#   - Pooling discards spatial info that spatial attention needs
#   - Follows CBAM paper's recommended practice (ECCV 2018)
#
# TRAINING CONFIGURATION:
#   - Same as OptimizedCNN baseline for fair comparison
#   - batch_size=16, lr=1e-3, AdamW, weight_decay=1e-4, dropout=0.3
#   - 20 epochs (baseline converged by epoch 15-17)
# ============================================================================

echo "=================================================="
echo "CBAM Attention Experiment"
echo "Model: CBAMOptimizedCNN"
echo "Architecture: [16,32,64,128] + CBAM (r=4, k=7) BEFORE pooling + GAP + direct classification"
echo "Expected params: ~110,000 (11-12% increase over baseline)"
echo "=================================================="

python main.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output/attention \
    --experiment_name cbam_optimized_r4_k7 \
    --wandb_project C3_Week4_Attention \
    --model_type cbam_optimized \
    --cbam_reduction 16 \
    --cbam_spatial_kernel 3 \
    --batch_size 16 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --dropout 0.3 \
    --seed 42 \
    --num_workers 8

echo "=================================================="
echo "CBAM Experiment completed!"
echo "Results saved to: /data/uabmcv2526/mcvstudent29/Week4/output/attention"
echo "W&B logs: /data/uabmcv2526/mcvstudent29/Week4/wandb/attention"
echo "=================================================="
