#!/bin/bash
# ============================================================
# Dual-View Fusion Experiments — Multi-GPU DDP Edition
# Runs EarlyFusion + LateFusion on 4 GPUs with max batch_size
#
# Effective batch_size = PER_GPU_BATCH × N_GPUS × GRAD_ACCUM
# ============================================================
source /root/miniconda3/etc/profile.d/conda.sh
conda activate f3mamba
cd /root/zt/F3Mamba
export PYTHONUNBUFFERED=1

# ── Configurable ──
N_GPUS=${N_GPUS:-4}
PER_GPU_BATCH=${BATCH:-32}
GRAD_ACCUM=${ACCUM:-1}
LR=0.0005
EPOCHS=20
WANDB_PROJECT=F3Mamba_Fusion

EFFECTIVE_BATCH=$((PER_GPU_BATCH * N_GPUS * GRAD_ACCUM))

echo "============================================"
echo "  Dual-View Fusion DDP Experiments"
echo "  GPUs:        $N_GPUS"
echo "  Batch/GPU:   $PER_GPU_BATCH"
echo "  Grad Accum:  $GRAD_ACCUM"
echo "  Eff. Batch:  $EFFECTIVE_BATCH"
echo "  LR:          $LR"
echo "  Epochs:      $EPOCHS"
echo "  Start:       $(date)"
echo "============================================"

# ── 1. Early Fusion ──
echo ""
echo "========== EarlyFusionInfinitePulse (DDP) =========="
torchrun --nproc_per_node=$N_GPUS train_3fold_ddp.py \
    --video_backbone EarlyFusionInfinitePulse \
    --loss_type mse \
    --lr $LR \
    --batch_size $PER_GPU_BATCH \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --epochs $EPOCHS \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --output_dir ./checkpoints_earlyfusion_ddp \
    --use_amp

# ── 2. Late Fusion ──
echo ""
echo "========== LateFusionInfinitePulse (DDP) =========="
torchrun --nproc_per_node=$N_GPUS train_3fold_ddp.py \
    --video_backbone LateFusionInfinitePulse \
    --loss_type mse \
    --lr $LR \
    --batch_size $PER_GPU_BATCH \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --epochs $EPOCHS \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --output_dir ./checkpoints_latefusion_ddp \
    --use_amp

echo ""
echo "============================================"
echo "  Fusion DDP Experiments Done: $(date)"
echo "============================================"
