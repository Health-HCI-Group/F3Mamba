#!/bin/bash
# ============================================================
# Dual-View Fusion Experiments: EarlyFusion + LateFusion
# 3-Fold CV, lr=0.0005, MSE loss, 20 epochs, wandb logging
# 
# Usage: nohup bash run_fusion_experiments.sh > fusion_exp.log 2>&1 &
# ============================================================
source /root/miniconda3/etc/profile.d/conda.sh
conda activate f3mamba
cd /root/zt/F3Mamba

LR=0.0005
EPOCHS=20
BATCH=32
DEVICE=cuda:0
N_FOLDS=3

echo "=== Dual-View Fusion Experiment $(date) ==="
echo "  LR=$LR, EPOCHS=$EPOCHS, BATCH=$BATCH, DEVICE=$DEVICE"
echo ""

# ============================================================
# 1. Early Fusion (channel concatenation)
# ============================================================
echo "========== EarlyFusionInfinitePulse =========="
python train_3fold.py \
    --video_backbone EarlyFusionInfinitePulse \
    --loss_type mse \
    --lr $LR \
    --batch_size $BATCH \
    --epochs $EPOCHS \
    --device $DEVICE \
    --n_folds $N_FOLDS \
    --use_wandb \
    --wandb_project F3Mamba_Fusion \
    --output_dir ./checkpoints_earlyfusion

# ============================================================
# 2. Late Fusion (feature-level concatenation)
# ============================================================
echo ""
echo "========== LateFusionInfinitePulse =========="
python train_3fold.py \
    --video_backbone LateFusionInfinitePulse \
    --loss_type mse \
    --lr $LR \
    --batch_size $BATCH \
    --epochs $EPOCHS \
    --device $DEVICE \
    --n_folds $N_FOLDS \
    --use_wandb \
    --wandb_project F3Mamba_Fusion \
    --output_dir ./checkpoints_latefusion

echo ""
echo "=== Fusion Experiments Done $(date) ==="
