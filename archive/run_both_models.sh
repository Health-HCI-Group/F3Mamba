#!/bin/bash
# ============================================================================
# Master script: Train BOTH dual-view fusion models sequentially
# Model 1: EarlyFusionInfinitePulse
# Model 2: LateFusionInfinitePulse
# ============================================================================
LOG_DIR="/root/zt/F3Mamba/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Starting BOTH models training (sequential)"
echo "Start time: $(date)"
echo "============================================"

# ─── Model 1: EarlyFusionInfinitePulse ───
echo ">>>>>> [1/2] EarlyFusionInfinitePulse <<<<<<"
bash /root/zt/F3Mamba/run_fusion_4gpu_early.sh
echo ">>>>>> EarlyFusionInfinitePulse DONE at $(date) <<<<<<"

# ─── Model 2: LateFusionInfinitePulse ───
echo ">>>>>> [2/2] LateFusionInfinitePulse <<<<<<"
bash /root/zt/F3Mamba/run_fusion_4gpu_late.sh
echo ">>>>>> LateFusionInfinitePulse DONE at $(date) <<<<<<"

echo "============================================"
echo "ALL training complete! $(date)"
echo "Checkpoints: /root/zt/F3Mamba/checkpoints/"
echo "============================================"
