#!/bin/bash
# 3-Fold full experiment with wandb online, Neg_Pearson, per-subject HR eval
# Run: nohup bash run_experiment.sh &
source /root/miniconda3/etc/profile.d/conda.sh
conda activate f3mamba
cd /root/zt/F3Mamba

LOGFILE=experiment_$(date +%m%d_%H%M).log
echo "=== Full Experiment (wandb online) $(date) ===" | tee $LOGFILE

for lr in 3e-4 5e-4 1e-3; do
    echo "" | tee -a $LOGFILE
    echo "========== LR=$lr ==========" | tee -a $LOGFILE
    
    python train_v2.py --epochs 20 --lr $lr --device cuda:0 --n_folds 3 \
        --use_wandb --wandb_project F3Mamba_Train \
        --output_dir ./checkpoints_lr${lr} \
        2>&1 | tee -a $LOGFILE
done

echo "" | tee -a $LOGFILE
echo "=== Experiment done $(date) ===" | tee -a $LOGFILE
