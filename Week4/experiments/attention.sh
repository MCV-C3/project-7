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
# SQUEEZE-AND-EXCITATION ATTENTION EXPERIMENT
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
#   through attention mechanisms. Channel attention (SE blocks) can improve
#   feature discrimination without significantly increasing overfitting risk.
#
# SQUEEZE-AND-EXCITATION (SE) MECHANISM:
#   SE blocks add channel attention after each convolutional block:
#   1. Squeeze: Global average pooling to aggregate spatial information
#   2. Excitation: Two FC layers to learn channel interdependencies
#   3. Scale: Multiply features by learned channel-wise weights
#
#   This allows the network to:
#   - Emphasize important feature channels
#   - Suppress irrelevant channels
#   - Adapt attention dynamically per image
#
# ARCHITECTURE: SEOptimizedCNN
#   - Base: Same as OptimizedCNN ([16,32,64,128] + GAP + direct classification)
#   - Addition: SE blocks after each of 4 conv blocks
#   - SE reduction ratio: 4 (standard for small networks)
#
# PARAMETER OVERHEAD:
#   SE Block 1 (16 ch):  16×4 + 4×16 = 128 params
#   SE Block 2 (32 ch):  32×8 + 8×32 = 512 params
#   SE Block 3 (64 ch):  64×16 + 16×64 = 2,048 params
#   SE Block 4 (128 ch): 128×32 + 32×128 = 8,192 params
#   Total SE overhead: ~10,880 params (11% increase)
#   Total model: ~109,832 params (vs 98,952 baseline)
#
# SYNERGY WITH BASELINE:
#   - GAP already uses global pooling (spatial → channel aggregation)
#   - SE blocks use same principle but throughout the network
#   - Philosophically aligned → natural integration
#
# TRAINING CONFIGURATION:
#   - Same as OptimizedCNN baseline for fair comparison
#   - batch_size=16, lr=1e-3, AdamW, weight_decay=1e-4, dropout=0.3
#   - 20 epochs (baseline converged by epoch 15-17)
# ============================================================================

echo "=================================================="
echo "SE Attention Experiment"
echo "Model: SEOptimizedCNN"
echo "Architecture: [16,32,64,128] + SE blocks (reduction=4) + GAP + direct classification"
echo "Expected params: ~109,832 (11% increase over baseline)"
echo "=================================================="

python main.py \
    --data_root /data/uabmcv2526/shared/dataset/2425/MIT_small_train_1 \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output/attention \
    --experiment_name se_optimized_r4 \
    --wandb_project C3_Week4_Attention \
    --model_type se_optimized \
    --se_reduction 4 \
    --batch_size 16 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --dropout 0.3 \
    --seed 42 \
    --num_workers 8

echo "=================================================="
echo "Experiment completed!"
echo "Results saved to: /data/uabmcv2526/mcvstudent29/Week4/output/attention"
echo "W&B logs: /data/uabmcv2526/mcvstudent29/Week4/wandb/attention"
echo "=================================================="
