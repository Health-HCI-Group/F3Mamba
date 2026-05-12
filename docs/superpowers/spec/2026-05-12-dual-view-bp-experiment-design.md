# Dual-View vs Single-View BP Estimation Experiment Design

Date: 2026-05-12

## Problem Statement

Current experiments show dual-view (front+back) performance equals single-view (back-only) for BP estimation. The goal is to verify if dual-view truly outperforms single-view under controlled conditions.

**Key Finding**: 
- Single-view (back): bp_sys MAE = 11.88 mmHg
- Dual-view (front+back): bp_sys MAE = 11.90 mmHg
- Difference: 0.02 mmHg (statistically insignificant)

## Hypothesis

Dual-view fusion should provide complementary information that single-view lacks, leading to better BP estimation accuracy.

## Experiment Design

### Fixed Variables
- **Data**: Non-overlap (no sliding window stride=40)
- **Model Architecture**: BP head + rPPG head (single layer)
- **Backbone**: 
  - Single-view: `InfinitePulseEnhancedBP`
  - Dual-view: `LateFusionInfinitePulseEnhancedBP`
- **Fusion Mode**: sums (current approach)
- **Hyperparameters**: batch_size=2, lr=6.25e-5, epochs=10, gradient_accumulation=2, AMP=on

### Experimental Variables
- **seq_mult**: 2, 4, 6, 8 (chunk lengths: 320, 640, 960, 1280 frames)
- **View Configuration**:
  - `back-only`: Single-view back camera
  - `front-only`: Single-view front camera
  - `front+back`: Dual-view fusion

### Experiment Matrix (12 experiments)

| seq_mult | Frames | back-only | front-only | front+back |
|----------|--------|-----------|-----------|------------|
| 2        | 320    | Run       | Run       | Run        |
| 4        | 640    | Run       | Run       | Run        |
| 6        | 960    | Run       | Run       | Run        |
| 8        | 1280   | Run       | Run       | Run        |

### Execution Order
1. seq_mult=2 (320 frames) - 3 experiments
2. seq_mult=4 (640 frames) - 3 experiments
3. seq_mult=6 (960 frames) - 3 experiments
4. seq_mult=8 (1280 frames) - 3 experiments (if OOM, reduce batch_size)

### Success Criteria
- **Primary**: dual-view bp_sys MAE < single-view bp_sys MAE by >= 0.5 mmHg
- **Secondary**: dual-view bp_dia MAE and BVP RMSE also improve
- **Statistical**: 3-fold CV variance analysis to confirm significance

### Implementation Notes
1. Use `train_3fold_ddp_multilabel.py` with appropriate `--modal_used` argument
2. Single-view experiments need model modification to accept single camera input
3. Monitor GPU memory for longer sequences

### Expected Output
- 12 experiment logs in `/root/zt/F3Mamba/train_*.log`
- Updated EXPERIMENTS.md with all results
- Statistical analysis comparing single vs dual-view performance