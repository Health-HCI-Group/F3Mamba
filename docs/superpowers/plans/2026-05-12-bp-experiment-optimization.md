# BP Experiment Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve BP MAE < 10 mmHg and Pearson correlation > 0.4 through optimized training and fusion strategies

**Architecture:** 
- 3-Fold Cross-Validation with DDP (4 GPUs)
- Single-view: InfinitePulseEnhancedBP (front/back)
- Dual-view: LateFusionInfinitePulseEnhancedBP with different fusion modes

**Tech Stack:** PyTorch, wandb, DDP, Mamba2

---

## Current Status

| Modal | seq_mult | Frames | Epochs | bp_sys MAE | Status |
|-------|----------|--------|--------|------------|--------|
| back | 2 | 320 | 10 | 11.88 | Done |
| front | 2 | 320 | 10 | 11.85 | Done |
| dual (sums) | 2 | 320 | 10 | 11.91 | Done |
| back | 4 | 640 | 10 | 11.92 | Done |
| front | 4 | 640 | 10 | 11.90 | Done |
| dual (sums) | 4 | 640 | 10 | 11.96 | Done |
| back | 6 | 960 | 10 | 11.97 | Done |
| front | 6 | 960 | 10 | 11.95 | Done |
| dual (sums) | 6 | 960 | 10 | OOM | Failed |

**Problem:** Dual-view not outperforming single-view. Need to investigate fusion methods and longer training.

---

## Phase 1: Baseline with 20 Epochs (seq_mult=1, 2)

**Objective:** Establish baseline with longer training for 160 and 320 frame configurations

### Task 1.1: Update sweep config for 20 epochs

**Files:**
- Create: `/root/zt/F3Mamba/sweep_phase1.yaml`

- [ ] **Step 1: Create sweep config for 20-epoch experiments**

```yaml
program: run_sweep_entry_20ep.py
method: grid
name: BP-Phase1-20ep

metric:
  name: bp_sys_MAE
  goal: minimize

parameters:
  modal:
    values: ["back", "front", "dual"]
  seq_mult:
    values: [1, 2]
```

### Task 1.2: Create 20-epoch sweep entry script

**Files:**
- Create: `/root/zt/F3Mamba/run_sweep_entry_20ep.py`

- [ ] **Step 1: Create entry script with 20 epochs**

```python
#!/root/miniconda3/envs/f3mamba/bin/python
"""
Sweep entrypoint for 20-epoch experiments.
"""
import subprocess, os, sys, re, json, time

OUTPUT_DIR = '/root/zt/F3Mamba'
TRAIN_SCRIPT = os.path.join(OUTPUT_DIR, 'train_3fold_ddp_multilabel.py')

MODAL_CFG = {
    'back':  ('InfinitePulseEnhancedBP',           '',   '--modal_used back'),
    'front': ('InfinitePulseEnhancedBP',           '',   '--modal_used front'),
    'dual':  ('LateFusionInfinitePulseEnhancedBP', '--fusion_mode sums', '--modal_used front back'),
}

def main():
    raw = ' '.join(sys.argv)
    m_m = re.search(r'--modal[= ](\w+)', raw)
    m_s = re.search(r'--seq_mult[= ](\d+)', raw)
    modal = m_m.group(1) if m_m else 'back'
    seq_mult = int(m_s.group(1)) if m_s else 1

    backbone, extra, modal_args = MODAL_CFG[modal]
    log_path = os.path.join(OUTPUT_DIR, f'train_{modal}_seq{seq_mult}_20ep.log')
    if os.path.exists(log_path):
        os.remove(log_path)

    import subprocess as _sp
    for _attempt in range(30):
        _out = _sp.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        _used = [int(x) for x in _out.stdout.strip().split('\n') if x.strip()]
        _gpu_used = [_used[i] for i in range(len(_used)) if i in (1,2,3,4) and i < len(_used)]
        if all(u < 100 for u in _gpu_used):
            break
        print(f'Waiting for GPU memory: {_gpu_used} MiB', flush=True)
        time.sleep(10)

    batch = 4 if seq_mult == 1 else 2
    grad_accum = 2

    cmd = (
        f'cd {OUTPUT_DIR} && CUDA_VISIBLE_DEVICES=1,2,3,4 '
        f'export MASTER_PORT=$((29500 + (RANDOM % 1000))) ; '
        f'torchrun --nproc_per_node=4 --master_port $MASTER_PORT {TRAIN_SCRIPT} '
        f'--video_backbone {backbone} {extra} '
        f'--batch_size {batch} --lr 6.25e-5 --epochs 20 '
        f'--gradient_accumulation_steps {grad_accum} '
        f'--use_wandb --use_scheduler '
        f'{modal_args} --data_suffix "" --seq_mult {seq_mult} '
        f'--use_amp '
        f'--loss_weight_spo2 0.0 --loss_weight_bp_sys 1.0 --loss_weight_bp_dia 1.0 '
        f'--weight_decay 1e-5 '
        f'> {log_path} 2>&1'
    )

    env = os.environ.copy()
    for k in list(env.keys()):
        if k.startswith('WANDB_') and k not in ('WANDB_API_KEY',):
            del env[k]

    print(f'modal={modal} seq_mult={seq_mult} epochs=20', flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)
    proc.wait()

    with open(log_path) as f:
        content = f.read()

    completed = '3-Fold CV Complete!' in content
    oom = 'CUDA out of memory' in content

    sys_vals, dia_vals = [], []
    for line in content.split('\n'):
        m = re.search(r'Fold (\d) summary: bp_sys_MAE=([\d.]+), bp_dia_MAE=([\d.]+)', line)
        if m:
            sys_vals.append(float(m.group(2)))
            dia_vals.append(float(m.group(3)))

    result = {
        'modal': modal, 'seq_mult': seq_mult, 'epochs': 20,
        'completed': completed, 'oom': oom,
        'returncode': proc.returncode,
        'bp_sys_MAE': round(sum(sys_vals)/len(sys_vals), 2) if sys_vals else None,
        'bp_dia_MAE': round(sum(dia_vals)/len(dia_vals), 2) if dia_vals else None,
    }

    result_path = log_path.replace('.log', '_result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'Result: {json.dumps(result)}', flush=True)
    sys.exit(0 if completed else (1 if oom else 2))

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Make script executable**

Run: `chmod +x /root/zt/F3Mamba/run_sweep_entry_20ep.py`

### Task 1.3: Initialize and run Phase 1 sweep

- [ ] **Step 1: Initialize wandb sweep**

Run: `cd /root/zt/F3Mamba && wandb sweep sweep_phase1.yaml`

- [ ] **Step 2: Start sweep agent**

Run the wandb agent command returned from step 1, e.g.:
`wandb agent 2796321819-beihang-university/F3Mamba/<SWEEP_ID> &> sweep_phase1.log &`

- [ ] **Step 3: Monitor sweep progress**

Run: `tail -f /root/zt/F3Mamba/sweep_phase1.log`

Expected: 6 experiments (back/front/dual × seq_mult=1/2), ~2 hours total

---

## Phase 2: Fusion Mode Experiments

**Objective:** Test alternative fusion modes to improve dual-view performance

**Available fusion modes:**
- `sums` (current): Simple average of features
- `concat`: Concatenate features + linear projection (preserves more information)
- `conv`: Concatenate + 1D convolution (learnable fusion)

### Task 2.1: Create fusion mode sweep config

**Files:**
- Create: `/root/zt/F3Mamba/sweep_phase2_fusion.yaml`

- [ ] **Step 1: Create sweep config for fusion experiments**

```yaml
program: run_sweep_entry_fusion.py
method: grid
name: BP-Phase2-Fusion

metric:
  name: bp_sys_MAE
  goal: minimize

parameters:
  fusion_mode:
    values: ["concat", "conv"]
  seq_mult:
    values: [1, 2]
```

### Task 2.2: Create fusion sweep entry script

**Files:**
- Create: `/root/zt/F3Mamba/run_sweep_entry_fusion.py`

- [ ] **Step 1: Create fusion entry script**

```python
#!/root/miniconda3/envs/f3mamba/bin/python
"""
Sweep entrypoint for fusion mode experiments.
"""
import subprocess, os, sys, re, json, time

OUTPUT_DIR = '/root/zt/F3Mamba'
TRAIN_SCRIPT = os.path.join(OUTPUT_DIR, 'train_3fold_ddp_multilabel.py')

def main():
    raw = ' '.join(sys.argv)
    m_f = re.search(r'--fusion_mode[= ](\w+)', raw)
    m_s = re.search(r'--seq_mult[= ](\d+)', raw)
    fusion_mode = m_f.group(1) if m_f else 'concat'
    seq_mult = int(m_s.group(1)) if m_s else 1

    log_path = os.path.join(OUTPUT_DIR, f'train_dual_{fusion_mode}_seq{seq_mult}_20ep.log')
    if os.path.exists(log_path):
        os.remove(log_path)

    import subprocess as _sp
    for _attempt in range(30):
        _out = _sp.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        _used = [int(x) for x in _out.stdout.strip().split('\n') if x.strip()]
        _gpu_used = [_used[i] for i in range(len(_used)) if i in (1,2,3,4) and i < len(_used)]
        if all(u < 100 for u in _gpu_used):
            break
        print(f'Waiting for GPU memory: {_gpu_used} MiB', flush=True)
        time.sleep(10)

    batch = 4 if seq_mult == 1 else 2
    grad_accum = 2

    cmd = (
        f'cd {OUTPUT_DIR} && CUDA_VISIBLE_DEVICES=1,2,3,4 '
        f'export MASTER_PORT=$((29500 + (RANDOM % 1000))) ; '
        f'torchrun --nproc_per_node=4 --master_port $MASTER_PORT {TRAIN_SCRIPT} '
        f'--video_backbone LateFusionInfinitePulseEnhancedBP '
        f'--fusion_mode {fusion_mode} '
        f'--batch_size {batch} --lr 6.25e-5 --epochs 20 '
        f'--gradient_accumulation_steps {grad_accum} '
        f'--use_wandb --use_scheduler '
        f'--modal_used front back --data_suffix "" --seq_mult {seq_mult} '
        f'--use_amp '
        f'--loss_weight_spo2 0.0 --loss_weight_bp_sys 1.0 --loss_weight_bp_dia 1.0 '
        f'--weight_decay 1e-5 '
        f'> {log_path} 2>&1'
    )

    env = os.environ.copy()
    for k in list(env.keys()):
        if k.startswith('WANDB_') and k not in ('WANDB_API_KEY',):
            del env[k]

    print(f'fusion_mode={fusion_mode} seq_mult={seq_mult} epochs=20', flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)
    proc.wait()

    with open(log_path) as f:
        content = f.read()

    completed = '3-Fold CV Complete!' in content
    oom = 'CUDA out of memory' in content

    sys_vals, dia_vals = [], []
    for line in content.split('\n'):
        m = re.search(r'Fold (\d) summary: bp_sys_MAE=([\d.]+), bp_dia_MAE=([\d.]+)', line)
        if m:
            sys_vals.append(float(m.group(2)))
            dia_vals.append(float(m.group(3)))

    result = {
        'fusion_mode': fusion_mode, 'seq_mult': seq_mult, 'epochs': 20,
        'completed': completed, 'oom': oom,
        'returncode': proc.returncode,
        'bp_sys_MAE': round(sum(sys_vals)/len(sys_vals), 2) if sys_vals else None,
        'bp_dia_MAE': round(sum(dia_vals)/len(dia_vals)) if dia_vals else None,
    }

    result_path = log_path.replace('.log', '_result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'Result: {json.dumps(result)}', flush=True)
    sys.exit(0 if completed else (1 if oom else 2))

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Make script executable**

Run: `chmod +x /root/zt/F3Mamba/run_sweep_entry_fusion.py`

### Task 2.3: Run Phase 2 sweep

- [ ] **Step 1: Initialize wandb sweep**

Run: `cd /root/zt/F3Mamba && wandb sweep sweep_phase2_fusion.yaml`

- [ ] **Step 2: Start sweep agent**

Run the returned command, e.g.:
`wandb agent 2796321819-beihang-university/F3Mamba/<SWEEP_ID> &> sweep_phase2.log &`

- [ ] **Step 3: Monitor sweep**

Expected: 4 experiments (concat/conv × seq_mult=1/2), ~1.5 hours total

---

## Phase 3: Extended Training Analysis

**Objective:** Analyze results and determine next steps

### Task 3.1: Collect and analyze results

- [ ] **Step 1: Parse all experiment logs**

Run after Phase 1 and Phase 2 complete:
```bash
cd /root/zt/F3Mamba
for log in train_*_20ep.log; do
  echo "=== $log ==="
  grep -E "(Fold \d summary|MAE)" "$log" | tail -20
done
```

- [ ] **Step 2: Update EXPERIMENTS.md with results**

Update the markdown table with new results.

- [ ] **Step 3: Analyze Pearson correlation**

Check if experiments meet the >0.4 requirement:
```bash
grep -E "pearson|Pearson" /root/zt/F3Mamba/train_*_20ep.log
```

---

## Success Criteria

| Metric | Target | Current Best |
|--------|--------|--------------|
| bp_sys MAE | < 10 mmHg | 11.85 mmHg |
| bp_dia MAE | < 10 mmHg | 8.06 mmHg |
| Pearson correlation | > 0.4 | Unknown |

---

## Rollback Plan

If Phase 1 and 2 fail to achieve targets:
1. Try separate backbones for front/back (shared_backbone=False)
2. Add contrastive loss between front/back features
3. Increase model capacity (more layers/larger hidden dim)
4. Try different learning rates or schedulers

---

## File Summary

| File | Purpose |
|------|---------|
| `sweep_phase1.yaml` | Phase 1 sweep config |
| `run_sweep_entry_20ep.py` | Phase 1 entry script |
| `sweep_phase2_fusion.yaml` | Phase 2 sweep config |
| `run_sweep_entry_fusion.py` | Phase 2 entry script |
| `EXPERIMENTS.md` | Results tracking |

---

## Execution Order

1. **Phase 1** (seq_mult=1,2 with 20 epochs) → ~2 hours
2. **Phase 2** (fusion mode experiments) → ~1.5 hours  
3. **Phase 3** (analysis) → immediate

Total estimated time: ~4 hours