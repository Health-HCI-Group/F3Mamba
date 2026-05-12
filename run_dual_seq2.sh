#!/bin/bash
BASE_ARGS=(
  --batch_size 2
  --lr 6.25e-5
  --epochs 10
  --gradient_accumulation_steps 2
  --use_wandb
  --use_scheduler
  --modal_used front back
  --data_suffix ""
  --seq_mult 2
  --use_amp
  --loss_weight_spo2 0.0
  --loss_weight_bp_sys 1.0
  --loss_weight_bp_dia 1.0
  --weight_decay 1e-5
)
export NCCL_TIMEOUT=600
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=1,2,3,4
cd /root/zt/F3Mamba
setsid /root/miniconda3/envs/f3mamba/bin/torchrun --nproc_per_node=4 train_3fold_ddp_multilabel.py --video_backbone LateFusionInfinitePulseEnhancedBP --fusion_mode sums "${BASE_ARGS[@]}" > train_dual_seq2.log 2>&1 &
echo "PID=$!"