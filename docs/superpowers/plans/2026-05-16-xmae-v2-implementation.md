# xMAE V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `xMAEBPModelV2` (BP-first with BVP auxiliary), restore HR evaluation via FFT-from-BVP (no explicit HR loss), and keep a clean insertion point for V3 delay prior.

**Architecture:** Add a new V2 model file with split BP heads and an alignment hook stub, then integrate V2 into training/sweep entry paths. Keep backbone behavior consistent with existing xMAE while adding HR metric logging in sweep/eval path from BVP outputs.

**Tech Stack:** Python, PyTorch, wandb, existing F3Mamba/xMAE codebase

---

### Task 1: Create V2 model module

**Files:**
- Create: `Models/xMAE_Model_V2.py`
- Modify: `Models/__init__` (if used; optional)
- Test: import + forward smoke check in Python shell

- [ ] **Step 1: Write failing import check**

Run:

```bash
/root/miniconda3/envs/f3mamba/bin/python -c "from Models.xMAE_Model_V2 import xMAEBPModelV2"
```

Expected: FAIL (module not found before creation).

- [ ] **Step 2: Implement `xMAEBPModelV2`**

Implement in `Models/xMAE_Model_V2.py`:
- keep dual encoders, curriculum mask, directional cross-attention, fusion, mamba
- keep `rPPG` head
- replace single BP head with split heads: `bp_sys_head` and `bp_dia_head`
- add `align_hook` insertion point as identity-compatible callable
- output dict keys: `rPPG`, `bp_sys`, `bp_dia`, `recon_loss`

- [ ] **Step 3: Run import + minimal forward shape check**

Run:

```bash
/root/miniconda3/envs/f3mamba/bin/python -c "import torch; from Models.xMAE_Model_V2 import xMAEBPModelV2; m=xMAEBPModelV2(chunk_size=160); x={'front':torch.randn(2,160,36,36,3),'back':torch.randn(2,160,36,36,3)}; y=m(x); print(sorted(y.keys()), y['rPPG'].shape, y['bp_sys'].shape, y['bp_dia'].shape)"
```

Expected: PASS, keys and shapes returned.

### Task 2: Integrate V2 into multi-task training pipeline

**Files:**
- Modify: `train_3fold_ddp_multilabel.py`
- Test: argument parse + model build smoke check

- [ ] **Step 1: Write failing option check**

Run:

```bash
/root/miniconda3/envs/f3mamba/bin/python train_3fold_ddp_multilabel.py --video_backbone xMAEBPModelV2 --help
```

Expected: FAIL before option is added.

- [ ] **Step 2: Wire V2 model class into imports/choices/factory**

Modify `train_3fold_ddp_multilabel.py`:
- import `xMAEBPModelV2`
- add `'xMAEBPModelV2'` to argparse choices
- add `'xMAEBPModelV2'` to `FUSION_BACKBONES`
- add factory branch in `build_model(args)` with xMAE params

- [ ] **Step 3: Ensure BP-first + BVP auxiliary remains active**

Verify existing `process_batch_multitask` and `MultiTaskLoss` path handles:
- `pred_bvp`/`label_bvp` from `rPPG` + `bvp` labels
- BP losses unchanged
- no explicit HR training loss added

- [ ] **Step 4: Run model build smoke test**

Run:

```bash
/root/miniconda3/envs/f3mamba/bin/python -c "from train_3fold_ddp_multilabel import get_args, build_model; import sys; sys.argv=['x','--video_backbone','xMAEBPModelV2']; args=get_args(); m=build_model(args); print(type(m).__name__)"
```

Expected: PASS, prints `xMAEBPModelV2`.

### Task 3: Restore HR metrics in sweep path (FFT from BVP)

**Files:**
- Modify: `sweep_xmae.py`
- Test: run log / wandb summary contains `eval/hr_*`

- [ ] **Step 1: Write failing metric expectation**

Run one sweep trial and inspect metrics.

Expected: no `eval/hr_MAE` before implementation.

- [ ] **Step 2: Add BVP auxiliary loss support in sweep train loop**

In `sweep_xmae.py`:
- add config weight `bvp_loss_weight`
- if `bvp` exists in labels, compute normalized BVP loss (`NegPearson` style or equivalent)
- add weighted BVP term to total loss
- log `train/bvp_loss`

- [ ] **Step 3: Add HR eval from BVP (no HR loss)**

In sweep eval:
- collect predicted `rPPG` and label `bvp`
- compute HR series using `calculate_metric_simple_inference`
- aggregate and log:
  - `eval/hr_MAE`
  - `eval/hr_RMSE`
  - `eval/hr_Pearson`

- [ ] **Step 4: Verify JSON serialization and run stability**

Ensure local result saving remains JSON-safe and no `float32` dump crash.

### Task 4: Sweep config update for V2 and validation run

**Files:**
- Modify: `sweep_xmae.py`
- Test: one resumed/new sweep run

- [ ] **Step 1: Add model selector and V2 default**

In config/runtime:
- support `model_version` in sweep or runtime config (`xmae_v1`/`xmae_v2`)
- set V2 as active for new experiments

- [ ] **Step 2: Run syntax and smoke checks**

Run:

```bash
/root/miniconda3/envs/f3mamba/bin/python -m py_compile sweep_xmae.py train_3fold_ddp_multilabel.py Models/xMAE_Model_V2.py
```

Expected: PASS

- [ ] **Step 3: Launch one sweep run and verify telemetry keys**

Expected in wandb:
- `train/loss`, `train/bvp_loss`
- `eval/mae_sys`, `eval/mae_dia`, `eval/pearson_sys`, `eval/pearson_dia`
- `eval/hr_MAE`, `eval/hr_RMSE`, `eval/hr_Pearson`

- [ ] **Step 4: Commit**

```bash
git add Models/xMAE_Model_V2.py train_3fold_ddp_multilabel.py sweep_xmae.py docs/superpowers/specs/2026-05-16-xmae-v2-with-v3-phase-prior-design.md docs/superpowers/plans/2026-05-16-xmae-v2-implementation.md
git commit -m "feat: add xmae v2 bp-first architecture and restore hr evaluation from bvp"
```
