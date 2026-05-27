#!/bin/bash
# Quick LR sweep with wandb online: fold 0 of 3-fold CV, 10 epochs each
# Run: nohup bash run_lr_sweep_wandb.sh &
source /root/miniconda3/etc/profile.d/conda.sh
conda activate f3mamba
cd /root/zt/F3Mamba

LOGFILE=lr_sweep_wandb_$(date +%m%d_%H%M).log
echo "=== LR Sweep (wandb online) $(date) ===" | tee $LOGFILE
echo "LRs: 1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3" | tee -a $LOGFILE
echo "Fold 0 of 3-fold CV, 10 epochs each" | tee -a $LOGFILE

for lr in 1e-5 3e-5 1e-4 3e-4 5e-4 1e-3; do
    echo "" | tee -a $LOGFILE
    echo ">>> LR=$lr <<< $(date +%H:%M:%S)" | tee -a $LOGFILE
    python train_v2.py \
        --epochs 10 \
        --lr $lr \
        --device cuda:0 \
        --n_folds 3 \
        --fold_start 0 \
        --fold_end 1 \
        --use_wandb \
        --wandb_project F3Mamba_Train \
        --output_dir ./checkpoints_lr_sweep \
        2>&1 | tee -a $LOGFILE
    echo "<<< LR=$lr done $(date +%H:%M:%S)" | tee -a $LOGFILE
done

echo "" | tee -a $LOGFILE
echo "=== LR Sweep done $(date) ===" | tee -a $LOGFILE
