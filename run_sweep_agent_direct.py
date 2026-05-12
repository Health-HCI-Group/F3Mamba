#!/root/miniconda3/envs/f3mamba/bin/python
"""
Wandb Sweep Agent for Dual-View BP experiments (seq_mult=6,8).
Must be invoked via `python run_sweep_agent_direct.py` (not /usr/bin/env python).
"""
import subprocess
import os
import re
import time
import sys
import wandb


MODAL_CFG = {
    'back': {
        'backbone': 'InfinitePulseEnhancedBP',
        'extra': '',
        'modal_args': '--modal_used back',
    },
    'front': {
        'backbone': 'InfinitePulseEnhancedBP',
        'extra': '',
        'modal_args': '--modal_used front',
    },
    'dual': {
        'backbone': 'LateFusionInfinitePulseEnhancedBP',
        'extra': '--fusion_mode sums',
        'modal_args': '--modal_used front back',
    },
}


def main():
    run = wandb.init()
    config = wandb.config

    modal = config['modal']
    seq_mult = config['seq_mult']
    cfg = MODAL_CFG[modal]

    log_name = f"train_{modal}_seq{seq_mult}.log"
    log_path = os.path.join('/root/zt/F3Mamba', log_name)

    cmd = (
        f'cd /root/zt/F3Mamba && '
        f'CUDA_VISIBLE_DEVICES=1,2,3,4 '
        f'torchrun --nproc_per_node=4 train_3fold_ddp_multilabel.py '
        f'--video_backbone {cfg["backbone"]} {cfg["extra"]} '
        f'--batch_size 1 --lr 6.25e-5 --epochs 10 '
        f'--gradient_accumulation_steps 4 --use_wandb --use_scheduler '
        f'{cfg["modal_args"]} --data_suffix "" --seq_mult {seq_mult} '
        f'--use_amp '
        f'--loss_weight_spo2 0.0 --loss_weight_bp_sys 1.0 --loss_weight_bp_dia 1.0 '
        f'--weight_decay 1e-5 '
        f'> {log_path} 2>&1'
    )

    print(f'\n{"="*60}')
    print(f'Dual-View BP Sweep: modal={modal}, seq_mult={seq_mult}')
    print(f'Backbone: {cfg["backbone"]}')
    print(f'Log: {log_path}')
    print(f'{"="*60}')

    # Delete old log
    if os.path.exists(log_path):
        os.remove(log_path)

    # Launch training
    proc = subprocess.Popen(cmd, shell=True)

    # Poll for completion
    timeout = 7200
    start = time.time()
    completed = False
    oom = False

    while time.time() - start < timeout:
        if os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
                if '3-Fold CV Complete!' in content:
                    completed = True
                    proc.wait()
                    break
                elif 'CUDA out of memory' in content:
                    oom = True
                    proc.kill()
                    break
        rc = proc.poll()
        if rc is not None:
            if not completed and os.path.exists(log_path):
                with open(log_path) as f:
                    if '3-Fold CV Complete!' in f.read():
                        completed = True
            break
        time.sleep(60)

    if not completed:
        reason = 'OOM' if oom else 'timeout'
        print(f'FAILED ({reason})')
        wandb.log({'status': reason})
        wandb.finish(exit_code=1)
        return

    # Parse results
    results = {'modal': modal, 'seq_mult': seq_mult, 'status': 'completed'}
    sys_vals, dia_vals = [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Fold (\d) summary: bp_sys_MAE=([\d.]+), bp_dia_MAE=([\d.]+)', line)
            if m:
                f_idx = int(m.group(1))
                s = float(m.group(2))
                d = float(m.group(3))
                results[f'fold{f_idx}_bp_sys_MAE'] = s
                results[f'fold{f_idx}_bp_dia_MAE'] = d
                sys_vals.append(s)
                dia_vals.append(d)
            m2 = re.search(r'Fold \d reg2cls_Acc=([\d.]+)', line)
            if m2:
                results['reg2cls_Acc'] = float(m2.group(1))

    if sys_vals:
        results['bp_sys_MAE'] = sum(sys_vals) / len(sys_vals)
        results['bp_dia_MAE'] = sum(dia_vals) / len(dia_vals)

    print(f'\nResults: {results}')
    wandb.log(results)
    wandb.finish()


if __name__ == '__main__':
    main()
