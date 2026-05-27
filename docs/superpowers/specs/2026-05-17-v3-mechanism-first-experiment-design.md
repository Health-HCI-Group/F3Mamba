# V3 Mechanism-First Experiment Design

Date: 2026-05-17
Owner: OpenCode + User
Project: `F3Mamba`

## Goal

Prioritize mechanism-level proof that V3 delay prior is beneficial before pursuing broad task optimization.

Primary stage goal (must-pass first):
- `eval/align_corr_gain >= 0`
- `eval/align_mech_pass = 1`
- `0.15 <= eval/align_apply_frac <= 0.45`

Secondary goal (after mechanism pass):
- Improve BP Pearson (especially `pearson_sys`) relative to V2 reference without MAE regression.

## Context Snapshot

Current status shows V3 can run stably after dtype/OOM fixes, but mechanism metrics are still negative in most runs:
- `align_corr_gain` frequently `< 0`
- `align_mech_pass` frequently `0`

This indicates the alignment module is active but not yet producing reliable positive signal transformation.

## Design Principles

1. Mechanism-first gating: do not scale task optimization until mechanism criteria pass.
2. Small, controlled rounds: 4-run batches for fast stop/go decisions.
3. Conservative memory and stability defaults: single GPU (`CUDA_VISIBLE_DEVICES=3`), small batch.
4. Explicit rollback hierarchy: reduce alignment influence before changing broader model knobs.

## Experimental Phases

### Phase M1: Mechanism Stop-Loss (4 runs)

Purpose:
- Minimize harmful alignment while testing whether gain can become non-negative.

Fixed:
- `batch_size=2`
- `bp_hidden_dim=64`
- `delay_max_shift=2`
- `avoid_edge_shift=True`

Sweep:
- `align_blend in [0.01, 0.03]`
- `min_corr_improve in [0.01, 0.02]`

Pass condition (round-level):
- At least 2/4 runs satisfy mechanism gate.

Fast-stop:
- If first 2 runs both have `align_corr_gain < -0.02` and `align_apply_frac > 0.5`, stop M1 and reduce `align_blend` again.

### Phase M2: Mechanism Strengthen (4 runs, only if M1 passes)

Purpose:
- Keep mechanism gate passing while pushing positive gain stability.

Fixed:
- M1 best non-alignment settings.

Sweep:
- `delay_max_shift in [2, 3]`
- `align_blend in [0.03, 0.05]`
- `min_corr_improve in [0.02, 0.03]`

Pass condition:
- At least 2/4 runs with `align_mech_pass=1` and `align_corr_gain > 0.005`.

### Phase T1: Task Relay (6 runs, only if M2 passes)

Purpose:
- Transfer mechanism improvements to BP/HR metrics.

Fixed:
- Best mechanism parameters from M2.

Sweep:
- `lr in [6.25e-5, 1e-4]`
- `bp_dropout in [0.05, 0.1]`
- `initial_mask_ratio in [0.8, 0.9]`

Success target:
- Reproducible `pearson_sys` improvement vs V2 reference with no material MAE degradation.

## Monitoring Layout

### Panel A: Mechanism
- `eval/align_corr_gain`
- `eval/align_mech_pass`
- `eval/align_apply_frac`
- `eval/align_edge_frac`
- `eval/align_delay_ms_abs_mean`

### Panel B: Task
- `eval/pearson_sys`, `eval/pearson_dia`
- `eval/mae_sys`, `eval/mae_dia`
- `eval/hr_MAE`

### Panel C: Runtime Safety
- `train/loss`, `train/loss_isfinite`
- `error/*`
- `runtime/global_step`

## Operational Protocol

Before each round:
1. Kill residual `sweep_xmae.py` processes.
2. Ensure no unintended competing run on GPU 3.
3. Launch a new sweep for the round (do not reuse a failed sweep).

Rollback order:
1. Decrease `align_blend`
2. Increase `min_corr_improve`
3. Narrow `delay_max_shift`

## Decision Rules

1. If mechanism gate is not passing, do not broaden task hyperparameter search.
2. If mechanism gate passes but task metrics do not improve, keep mechanism fixed and tune task-only knobs.
3. If both mechanism and task regress, stop and re-baseline against best V2 config before any further V3 changes.
