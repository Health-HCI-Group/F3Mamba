# V3 Mechanism-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a mechanism-first V3 experiment program that can prove positive alignment benefit before broad task tuning.

**Architecture:** Reuse the existing `sweep_xmae.py` V3 path and run controlled sweep rounds (M1 -> M2 -> T1) with strict gate checks between rounds. Each round runs on clean process state and dedicated sweep IDs, with round summaries persisted to docs for decision traceability.

**Tech Stack:** Python, wandb API/agent, PyTorch training script (`sweep_xmae.py`), bash process control

---

### Task 1: Stabilize runtime baseline and verify clean environment

**Files:**
- Modify: `sweep_xmae.py`
- Test: runtime smoke run logs (`run_sweep_xmae_v3_*`)

- [ ] **Step 1: Write failing dtype reproduction check**

Run:

```bash
/root/miniconda3/envs/f3mamba/bin/python - <<'PY'
import torch
from sweep_xmae import XCorrSoftAlignHook
hook=XCorrSoftAlignHook(2,0.1,0.0,True)
f=torch.randn(2,160,32,dtype=torch.float16)
o=hook(f, f, f)
print(f.dtype, o.dtype)
assert o.dtype == f.dtype
PY
```

Expected before fix: FAIL with dtype mismatch.

- [ ] **Step 2: Implement minimal dtype-preserving blend fix**

In `sweep_xmae.py`, ensure `blend` tensor is cast to `fused_feat.dtype` and same device before returning aligned output.

- [ ] **Step 3: Re-run dtype check and compile check**

Run:

```bash
/root/miniconda3/envs/f3mamba/bin/python -m py_compile sweep_xmae.py
/root/miniconda3/envs/f3mamba/bin/python - <<'PY'
import torch
from sweep_xmae import XCorrSoftAlignHook
hook=XCorrSoftAlignHook(2,0.1,0.0,True)
f=torch.randn(2,160,32,dtype=torch.float16)
o=hook(f, f, f)
assert o.dtype == f.dtype
print('PASS')
PY
```

Expected: PASS.

- [ ] **Step 4: Clear stale processes and verify GPU headroom**

Run:

```bash
pkill -f "sweep_xmae.py"; sleep 1; pgrep -af "sweep_xmae.py"
nvidia-smi
```

Expected: no `sweep_xmae.py` process; no heavy stale allocation on target GPU.

### Task 2: Run Round M1 (mechanism stop-loss) and gate decision

**Files:**
- Modify: `sweep_xmae.py` (M1 settings)
- Create: `docs/superpowers/specs/2026-05-17-v3-round-m1-results.md`

- [ ] **Step 1: Set M1 sweep settings**

Set V3 profile values in `sweep_xmae.py`:
- `batch_size=2`
- `bp_hidden_dim=64`
- `delay_max_shift=[2]`
- `align_blend=[0.01, 0.03]`
- `min_corr_improve=[0.01, 0.02]`
- `avoid_edge_shift=[True]`

- [ ] **Step 2: Launch M1 sweep with 4 runs**

Run:

```bash
nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 XMAE_SWEEP_PROFILE=v3_pilot XMAE_AGENT_COUNT=4 /root/miniconda3/envs/f3mamba/bin/python "sweep_xmae.py"' > "run_sweep_xmae_v3_m1.log" 2>&1 &
```

Expected: new sweep created, 4 runs scheduled.

- [ ] **Step 3: Wait for M1 completion and evaluate gate**

Gate per run:
- `eval/align_corr_gain >= 0`
- `eval/align_mech_pass == 1`
- `0.15 <= eval/align_apply_frac <= 0.45`

Round pass:
- at least 2/4 runs pass.

- [ ] **Step 4: Write M1 report**

Create `docs/superpowers/specs/2026-05-17-v3-round-m1-results.md` with:
- sweep id
- per-run metrics table
- gate counts
- pass/fail decision

### Task 3: Conditional Round M2 (mechanism strengthen)

**Files:**
- Modify: `sweep_xmae.py` (M2 settings)
- Create: `docs/superpowers/specs/2026-05-17-v3-round-m2-results.md`

- [ ] **Step 1: If M1 fails, stop and rollback**

Rollback order:
1. lower `align_blend`
2. raise `min_corr_improve`
3. narrow `delay_max_shift`

Expected: no M2 launch until M1 pass.

- [ ] **Step 2: If M1 passes, set M2 ranges**

Use best M1 base config and sweep:
- `delay_max_shift=[2,3]`
- `align_blend=[0.03,0.05]`
- `min_corr_improve=[0.02,0.03]`

- [ ] **Step 3: Launch M2 (4 runs) and evaluate gate**

Run with `XMAE_AGENT_COUNT=4` and record metrics.

M2 pass condition:
- >=2/4 runs with `eval/align_mech_pass=1`
- and `eval/align_corr_gain > 0.005`

- [ ] **Step 4: Write M2 report**

Create `docs/superpowers/specs/2026-05-17-v3-round-m2-results.md` with decision.

### Task 4: Conditional Round T1 (task relay)

**Files:**
- Modify: `sweep_xmae.py` (T1 settings)
- Create: `docs/superpowers/specs/2026-05-17-v3-round-t1-results.md`

- [ ] **Step 1: Only proceed when M2 passes**

If M2 fails, stop at mechanism stage and do not run T1.

- [ ] **Step 2: Configure T1 search (6 runs)**

Fix mechanism params from best M2 run(s), then sweep:
- `lr=[6.25e-5,1e-4]`
- `bp_dropout=[0.05,0.1]`
- `initial_mask_ratio=[0.8,0.9]`

- [ ] **Step 3: Launch T1 and collect final task metrics**

Expected report fields:
- `test/mae_sys`, `test/mae_dia`
- `test/pearson_sys`, `test/pearson_dia`
- `test/hr_MAE`
- comparison against V2 reference best cluster.

- [ ] **Step 4: Write T1 report and final recommendation**

Create `docs/superpowers/specs/2026-05-17-v3-round-t1-results.md` with:
- whether V3 has proven mechanism benefit
- whether task benefit is statistically/practically meaningful
- recommended next action.

### Task 5: Monitoring and safety automation checks

**Files:**
- Modify: `sweep_xmae.py` (if needed for telemetry consistency)
- Test: wandb summary keys

- [ ] **Step 1: Verify all required panels are populated**

Required keys:
- Mechanism: `eval/align_corr_gain`, `eval/align_mech_pass`, `eval/align_apply_frac`, `eval/align_edge_frac`, `eval/align_delay_ms_abs_mean`
- Task: `eval/pearson_sys`, `eval/pearson_dia`, `eval/mae_sys`, `eval/mae_dia`, `eval/hr_MAE`
- Runtime safety: `train/loss`, `train/loss_isfinite`, `error/*`, `runtime/global_step`

- [ ] **Step 2: Verify fast-stop criteria are manually enforceable per round**

Fast-stop rule:
- first 2 runs both have `align_corr_gain < -0.02` and `align_apply_frac > 0.5` -> stop round.

- [ ] **Step 3: Commit round artifacts (only when user requests commit)**

If user asks:

```bash
git add sweep_xmae.py docs/superpowers/specs/2026-05-17-v3-*.md docs/superpowers/plans/2026-05-17-v3-mechanism-first-implementation-plan.md
git commit -m "chore: execute v3 mechanism-first rounds with gated reporting"
```
