# xMAE Sweep Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sweep_xmae.py` emit complete and reliable W&B telemetry so fold/epoch progress and failures are visible while preserving training behavior.

**Architecture:** Keep training logic intact and add an observability wrapper layer in-place: standardized metric namespaces, monotonic global step, stage heartbeats, fail-fast checks, and exception telemetry with guaranteed `wandb.finish()`. Use small helper functions in the same file for consistent logging.

**Tech Stack:** Python, PyTorch, Weights & Biases (wandb), existing xMAE training script

---

### Task 1: Add logging helpers and runtime context

**Files:**
- Modify: `sweep_xmae.py`
- Test: runtime log output from W&B run page

- [ ] **Step 1: Write the failing check (missing telemetry baseline)**

Run:

```bash
python sweep_xmae.py
```

Expected failure condition (current behavior): W&B run may show only partial metrics and no clear stage heartbeat (`runtime/stage`).

- [ ] **Step 2: Add observability helper code near `train_xmae()`**

Insert helper utilities in `sweep_xmae.py`:

```python
import time
import traceback

def safe_log(payload, global_step=None):
    if global_step is not None:
        payload['runtime/global_step'] = int(global_step)
    wandb.log(payload)

def mark_stage(stage, fold=None, epoch=None, global_step=None, start_time=None):
    p = {'runtime/stage': stage}
    if fold is not None:
        p['fold/index'] = int(fold)
    if epoch is not None:
        p['runtime/epoch'] = int(epoch)
    if start_time is not None:
        p['runtime/elapsed_sec'] = float(time.time() - start_time)
    safe_log(p, global_step=global_step)
```

- [ ] **Step 3: Initialize run-level step metric and context immediately after `wandb.init()`**

Add in `train_xmae()` right after init:

```python
start_time = time.time()
global_step = 0
current_fold = -1
current_epoch = -1
wandb.define_metric('runtime/global_step')
wandb.define_metric('*', step_metric='runtime/global_step')
mark_stage('init', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
```

- [ ] **Step 4: Run script and verify early stage appears**

Run:

```bash
python sweep_xmae.py
```

Expected: run history contains `runtime/stage=init` and `runtime/global_step=0`.

- [ ] **Step 5: Commit Task 1**

```bash
git add sweep_xmae.py
git commit -m "chore: add wandb runtime stage and step scaffolding"
```

### Task 2: Add fold/epoch training telemetry with stable namespaces

**Files:**
- Modify: `sweep_xmae.py`
- Test: W&B charts for `train/*`, `fold/*`, `runtime/*`

- [ ] **Step 1: Write failing check for missing namespaced metrics**

Run one sweep trial and inspect run page.

Expected failure condition (current behavior): lack of stable namespaced metrics (`train/loss`, `fold/train_subjects`, `runtime/epoch`).

- [ ] **Step 2: Add fold-level stage and metadata logs**

Before training each fold, log split metadata:

```python
current_fold = fold_idx
mark_stage('fold_start', fold=current_fold, epoch=-1, global_step=global_step, start_time=start_time)
safe_log({
    'fold/index': current_fold,
    'fold/train_subjects': len(train_ids),
    'fold/test_subjects': len(test_ids),
}, global_step=global_step)
```

- [ ] **Step 3: Log epoch-level train metrics using namespace keys**

At epoch end, replace old per-fold keys with:

```python
current_epoch = epoch
mark_stage('train_epoch', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
safe_log({
    'train/loss': total_loss / len(train_loader),
    'train/lr': optimizer.param_groups[0]['lr'],
}, global_step=global_step)
```

If recon loss is available in-loop, also aggregate and log:

```python
safe_log({'train/recon_loss': recon_epoch_avg}, global_step=global_step)
```

- [ ] **Step 4: Increment `global_step` per batch**

Inside each train batch iteration, after optimizer/scheduler step:

```python
global_step += 1
```

- [ ] **Step 5: Run and verify training telemetry**

Run:

```bash
python sweep_xmae.py
```

Expected: W&B run shows `fold/*` at fold start and `train/loss`, `train/lr` with increasing `runtime/global_step`.

- [ ] **Step 6: Commit Task 2**

```bash
git add sweep_xmae.py
git commit -m "feat: add namespaced fold and training metrics for sweep"
```

### Task 3: Add eval/final metrics and failure telemetry

**Files:**
- Modify: `sweep_xmae.py`
- Test: W&B history for `eval/*`, `final/*`, and `error/*` when failure occurs

- [ ] **Step 1: Write failing check for missing structured eval/final/error fields**

Run and inspect run page.

Expected failure condition (current behavior): eval/final metrics may be inconsistent with training step series; failures may end without `error/*` fields.

- [ ] **Step 2: Add eval stage and namespaced eval logs**

Before eval per fold:

```python
mark_stage('eval_start', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
```

After fold eval:

```python
safe_log({
    'eval/mae_sys': mae[0].item(),
    'eval/mae_dia': mae[1].item(),
    'eval/pearson_sys': r_sys,
    'eval/pearson_dia': r_dia,
}, global_step=global_step)
mark_stage('fold_end', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
```

- [ ] **Step 3: Add final aggregate namespace logs**

At run end:

```python
safe_log({
    'final/avg_mae_sys': avg_mae,
    'final/avg_pearson_sys': avg_p,
}, global_step=global_step)
mark_stage('done', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
```

- [ ] **Step 4: Add fail-fast validations and finite-loss checks**

Implement checks:

```python
if len(all_ids) == 0:
    raise RuntimeError('No subject IDs found in ProcessedDataset path')
if len(train_ids) == 0 or len(test_ids) == 0:
    raise RuntimeError(f'Empty split at fold {fold_idx}: train={len(train_ids)}, test={len(test_ids)}')
if len(train_loader) == 0:
    raise RuntimeError(f'Empty train loader at fold {fold_idx}')
if not torch.isfinite(loss):
    safe_log({'train/loss_isfinite': 0}, global_step=global_step)
    raise RuntimeError(f'Non-finite loss at fold={current_fold}, epoch={current_epoch}, step={global_step}')
else:
    safe_log({'train/loss_isfinite': 1}, global_step=global_step)
```

- [ ] **Step 5: Wrap `train_xmae()` body with try/except/finally and structured error logs**

Add:

```python
try:
    ...
except Exception as e:
    safe_log({
        'error/type': type(e).__name__,
        'error/message': str(e),
        'error/fold': current_fold,
        'error/epoch': current_epoch,
        'error/traceback': traceback.format_exc(),
    }, global_step=global_step)
    raise
finally:
    wandb.finish()
```

- [ ] **Step 6: Run and verify both success and failure visibility**

Run:

```bash
python sweep_xmae.py
```

Expected:
- Success path: `eval/*` and `final/*` visible with aligned step axis.
- Failure path (if any): `error/type` and `error/message` visible before run ends.

- [ ] **Step 7: Commit Task 3**

```bash
git add sweep_xmae.py
git commit -m "fix: add robust eval and error telemetry for xmae sweep"
```

### Task 4: Verification checklist and run handoff

**Files:**
- Modify: none (verification only)
- Test: runtime and W&B sweep panel

- [ ] **Step 1: Execute one live run under the active sweep and observe logs**

Run:

```bash
python sweep_xmae.py
```

Expected: run enters `init -> fold_start -> train_epoch -> eval_start -> fold_end -> done`.

- [ ] **Step 2: Verify metric families exist in W&B UI**

Checklist:
- `train/*` present
- `eval/*` present
- `fold/*` present
- `final/*` present
- `runtime/*` present

- [ ] **Step 3: Verify global step continuity**

Expected: `runtime/global_step` monotonic increasing and chart x-axis consistent across metric groups.

- [ ] **Step 4: Confirm no training behavior regression from logs**

Expected: same fold split cardinalities and normal epoch progression compared with previous script behavior.

- [ ] **Step 5: Commit verification notes if you keep a runbook (optional)**

```bash
git status
```

Expected: clean working tree or only intended observability changes.
