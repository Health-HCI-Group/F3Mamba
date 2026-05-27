# xMAE V2 Design with V3 Phase-Prior Roadmap

Date: 2026-05-16
Owner: OpenCode + User
Scope: xMAE model/training pipeline evolution for dual-view face+finger BP estimation

## Goal

Build a new model version `V2` that improves BP performance while restoring HR evaluation via BVP (without explicit HR loss), and define a compatible path for a `V3` phase-delay prior based on face-finger waveform timing.

Primary metric goals:
- HR MAE < 2 (evaluated from BVP via FFT)
- BP MAE (sys/dia) < 10
- BP Pearson > 0.3

## Confirmed Decisions

1. V2 task outputs prioritize BP, with rPPG/BVP as auxiliary signal.
2. Do not add explicit HR loss in V2.
3. HR is computed during evaluation from predicted BVP (FFT-based), following the prior pipeline style.
4. Model large-scale structure remains intact (no full redesign).
5. V3 can introduce phase/delay prior between face and finger VP waveforms.

## Existing Baseline (Current State)

Current `xMAEBPModel` has:
- Dual-view encoders (front/back)
- xMAE-style continuous masking + directional cross-attention reconstruction
- Fusion + Mamba temporal modeling
- Heads: `rPPG` and `bp_sys/bp_dia`

Current sweep script only optimizes BP losses and logs BP metrics; HR metric is currently absent in this path although the multi-task training script already contains FFT-based HR evaluation utilities on BVP.

## V2 Architecture Design

### 1) Model Definition

Create a new model class and file:
- `Models/xMAE_Model_V2.py`
- class name: `xMAEBPModelV2`

Keep core backbone unchanged:
- front/back feature extraction
- curriculum continuous mask
- directional cross-attention reconstruction
- fusion + Mamba temporal encoder

Refine prediction heads:
- Keep `rPPG` head (sequence output)
- Split BP head into two explicit branches for systolic/diastolic
  - Shared pooled trunk + branch-specific linear projections is acceptable
  - Goal: reduce systolic/diastolic coupling noise

Output dictionary for training/inference:
- `rPPG`
- `bp_sys`
- `bp_dia`
- `recon_loss`

No explicit `pred_hr` head in V2.

### 2) Loss Design (BP-First)

V2 total loss:

`L = w_sys * L_bp_sys + w_dia * L_bp_dia + w_bvp * L_bvp + w_recon * L_recon + w_phys * L_phys`

Where:
- `L_bp_sys`, `L_bp_dia`: MSE on normalized BP labels
- `L_bvp`: NegPearson between predicted and target BVP
- `L_recon`: xMAE reconstruction loss (auxiliary)
- `L_phys`: optional physiological consistency (PP/MAP), lightweight regularizer

Recommended default weighting (starting point):
- `w_sys=1.0`, `w_dia=1.0`
- `w_bvp=0.4` to `0.8` (auxiliary but meaningful)
- `w_recon=0.05` to `0.1`
- `w_phys=0.1` to `0.3` when enabled

### 3) HR Handling in V2

Training:
- no HR loss term

Evaluation:
- compute HR from predicted BVP and label BVP via FFT windows
- report/log:
  - `test/hr_MAE`
  - `test/hr_RMSE`
  - `test/hr_Pearson`

This restores HR monitoring while avoiding an extra objective that may conflict with BP optimization.

## Training Pipeline Changes for V2

1. Add V2 model option in training entry script(s) and sweep script.
2. Ensure batch processing includes BVP path when available (`pred_bvp`, `label_bvp`).
3. Enable BP-first weighted multi-task loss with optional physiology term.
4. Re-enable/log FFT-based HR metrics in V2 evaluation path.
5. Keep all current observability telemetry (`train/*`, `eval/*`, `runtime/*`, `error/*`).

## V3 Concept: Face-Finger Phase/Delay Prior

### Hypothesis

Face and finger VP waveforms exhibit a physiologically constrained time delay (related to pulse transit), which can serve as a useful prior for cross-view alignment and BP estimation.

### Key Constraint

Delay is not perfectly constant; it is slowly varying. Therefore use a soft prior, not a hard fixed offset.

### V3-Compatible Module (planned)

Add `DelayAlign` module before fusion:
- Input: front/back temporal features
- Predict/estimate delay `tau` in a bounded range
- Apply differentiable temporal shift/interpolation to align one stream
- Output aligned features for downstream fusion

Regularization:
- range prior: `tau` within plausible physiological window
- smoothness prior: temporal/batch continuity of `tau`

### Compatibility with V2

V2 should reserve a clean insertion point:
- `encoder outputs -> (optional align module) -> fusion`

No need to activate in V2, but keep interfaces clean so V3 can be added with minimal churn.

## Risks and Mitigation

Risk 1: BP optimization regresses when BVP weight is too high.
- Mitigation: BP-first weighting, tune `w_bvp` conservatively.

Risk 2: Reconstruction objective competes with BP objective.
- Mitigation: keep `w_recon` low and sweep in narrow range.

Risk 3: HR metric appears unstable across windows.
- Mitigation: fixed FFT window policy + robust aggregation.

Risk 4: V3 delay prior over-constrains dynamic physiology.
- Mitigation: soft bounded delay + smoothness penalty only.

## Validation Plan

V2 acceptance checks:
1. Model forward returns expected keys (`rPPG`, `bp_sys`, `bp_dia`, `recon_loss`).
2. Training runs without NaN/shape regressions.
3. Eval logs HR metrics from BVP FFT again.
4. BP metrics improve relative to current sweep baseline.

V3 readiness checks (design-level):
1. V2 code has explicit alignment insertion boundary.
2. Delay prior implementation can be toggled by config without breaking V2 behavior.

## Out of Scope (This Spec)

- Full V3 implementation in this iteration
- Major backbone replacement
- Multi-node/distributed redesign
- Non-video modality expansion beyond face+finger
