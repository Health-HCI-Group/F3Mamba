#!/bin/bash
# ============================================================================
# Training: EarlyFusionInfinitePulse (4-GPU DDP)
# Model params: 580,429 | Batch: 16/GPU (effective 64)
# ============================================================================
source /root/miniconda3/etc/profile.d/conda.sh
conda activate f3mamba
cd /root/zt/F3Mamba
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "Model: EarlyFusionInfinitePulse"
echo "GPUs: 4 | Batch/gpu: 16 | Effective: 64"
echo "Start time: $(date)"
echo "============================================"

torchrun --nproc_per_node=4 train_fusion_ddp.py \
    --video_backbone EarlyFusionInfinitePulse \
    --batch_size 16 \
    --lr 0.0001 \
    --epochs 30 \
    --loss_type neg_pearson \
    --n_folds 3 \
    --eval_period 5 \
    --output_dir ./checkpoints \
    --seed 2024
