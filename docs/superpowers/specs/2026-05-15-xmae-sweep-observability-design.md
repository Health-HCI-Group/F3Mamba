# xMAE Sweep Observability Design

Date: 2026-05-15
Owner: OpenCode + User
Scope: `/root/zt/F3Mamba/sweep_xmae.py` only

## Goal

Improve observability for the current W&B sweep run (`legendary-sweep-1`) so training progress, evaluation metrics, and failure points are visible and diagnosable in near real time.

Success means:
- W&B shows metrics beyond `_runtime` and beyond only `loss/lr`
- stage transitions are visible (`init`, fold start/end, train epoch, eval, done)
- failures still produce structured telemetry (`error/*`) with fold/epoch context
- no behavior change to model architecture or dataset semantics

## Constraints and Non-Goals

Constraints:
- Keep changes minimal and localized to `sweep_xmae.py`
- Preserve existing training/evaluation logic and sweep parameter behavior
- Avoid refactoring model or data pipeline files

Non-goals:
- No model performance tuning in this change
- No architecture refactor into multiple modules
- No dashboard redesign outside metric naming/structure

## Approach Options Considered

1) Minimal instrumentation (recommended)
- Add structured metric namespaces, global step, stage heartbeats, and exception telemetry.
- Pros: fast, low risk, immediate visibility.
- Cons: training loop remains monolithic.

2) Helper-based structured logging
- Add lightweight wrappers (`safe_log`, `mark_stage`) and optional fold tables.
- Pros: cleaner and reusable pattern.
- Cons: moderate extra code surface.

3) Full loop refactor
- Split into `train_one_epoch/eval_one_fold` and standardize logs across functions.
- Pros: best maintainability.
- Cons: high risk and too heavy for urgent run recovery.

Decision: option 1 with one light helper (`safe_log`) from option 2.

## Detailed Design

### 1) Metric Taxonomy and Naming

Use stable namespaces so panels can filter by prefix:

- `train/*`
  - `train/loss`
  - `train/lr`
  - `train/recon_loss` (only if present)
  - `train/loss_isfinite` (0/1)
- `eval/*`
  - `eval/mae_sys`
  - `eval/mae_dia`
  - `eval/pearson_sys`
  - `eval/pearson_dia`
- `fold/*`
  - `fold/index`
  - `fold/train_subjects`
  - `fold/test_subjects`
- `final/*`
  - `final/avg_mae_sys`
  - `final/avg_pearson_sys`
- `runtime/*`
  - `runtime/stage`
  - `runtime/epoch`
  - `runtime/global_step`
  - `runtime/elapsed_sec`
- `error/*`
  - `error/type`
  - `error/message`
  - `error/fold`
  - `error/epoch`

### 2) Step and Stage Semantics

- Introduce `global_step` counter that increments per training batch.
- Use `wandb.define_metric("*", step_metric="runtime/global_step")` or equivalent stable step behavior.
- Every `wandb.log` attaches `runtime/global_step` to avoid disjoint charts.
- Stage heartbeat events:
  - `init`
  - `fold_start`
  - `train_epoch`
  - `eval_start`
  - `fold_end`
  - `done`

### 3) Reliability and Error Telemetry

- Wrap core training body in `try/except/finally`.
- On exception:
  - log `error/*` with fold/epoch/global_step context
  - re-raise after logging so failure is explicit
- In `finally`, always call `wandb.finish()`.
- Fail-fast validation checks with clear error messages:
  - no subject IDs discovered
  - empty train/test split
  - empty dataloader
- During training, assert `loss` finite; if not finite, log and abort with context.

### 4) Compatibility and Behavioral Safety

- Keep optimizer, scheduler, forward pass, and metric math unchanged.
- Keep sweep config schema unchanged unless separately requested.
- Keep existing local JSON output behavior, while ensuring failures are visible in W&B first.

## Data Flow

1. `wandb.init` -> emit `runtime/stage=init`
2. fold setup -> emit `fold/*` + `runtime/stage=fold_start`
3. epoch loop -> emit `train/*` + `runtime/stage=train_epoch`
4. eval -> emit `eval/*` + `runtime/stage=eval_start`
5. fold finish -> emit `runtime/stage=fold_end`
6. final aggregate -> emit `final/*` + `runtime/stage=done`
7. any exception -> emit `error/*` before process exit

## Verification Plan

Runtime verification after patch:
- Start script and confirm early heartbeat appears in run history (`runtime/stage=init`)
- Confirm fold metadata appears (`fold/index`, subject counts)
- Confirm at least one epoch writes `train/loss` and `train/lr` with increasing step
- Confirm eval metrics appear after fold (`eval/*`)
- Confirm final metrics appear on normal completion (`final/*`)

Failure-path verification:
- If a run fails, confirm `error/type` and `error/message` are present with fold/epoch context.

## Risks and Mitigations

Risk: excessive log volume.
- Mitigation: log training aggregates at epoch granularity (not every micro-event), keep batch-step only as scalar context.

Risk: inconsistent step behavior in mixed logs.
- Mitigation: ensure all logs include the same `runtime/global_step` key.

Risk: masked crashes due to swallowed exceptions.
- Mitigation: log error then re-raise.

## Out of Scope Follow-ups

- throughput optimization (loader workers, AMP tuning, fold reduction)
- model quality optimization (loss weights, schedule, regularization)
- refactor into reusable trainer modules
