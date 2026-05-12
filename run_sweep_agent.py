"""
Wandb Sweep Agent for Dual-View BP experiments.

The training script handles its own wandb logging (per-epoch, per-fold).
This agent manages experiment orchestration: parameter selection, launching,
monitoring, and final result collection.

Usage:
    # Create sweep (one time)
    wandb sweep sweep_config.yaml

    # Launch agent (repeat for each experiment)
    wandb agent <sweep_id>
"""
import subprocess
import os
import re
import time
import sys

# batch_size depends on both seq_mult and modal (dual backbone uses more memory)
def get_batch_size(modal, seq_mult):
    if modal == 'dual' and seq_mult >= 4:
        return 1
    return {2: 2, 4: 2, 6: 1, 8: 1}[seq_mult]

def get_grad_accum(modal, seq_mult):
    if modal == 'dual' and seq_mult >= 4:
        return 4
    return 2

PARAM_MAP = {
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
    import wandb
    run = wandb.init()
    config = wandb.config

    modal = config['modal']
    seq_mult = config['seq_mult']
    batch_size = get_batch_size(modal, seq_mult)
    grad_accum = get_grad_accum(modal, seq_mult)
    params = PARAM_MAP[modal]

    log_name = f"train_{modal}_seq{seq_mult}.log"
    log_path = os.path.join('/root/zt/F3Mamba', log_name)

    base_args = ' '.join([
        f'--batch_size {batch_size}',
        f'--lr 6.25e-5',
        f'--epochs 10',
        f'--gradient_accumulation_steps {grad_accum}',
        '--use_wandb',
        '--use_scheduler',
        params['modal_args'],
        '--data_suffix ""',
        f'--seq_mult {seq_mult}',
        '--use_amp',
        '--loss_weight_spo2 0.0',
        '--loss_weight_bp_sys 1.0',
        '--loss_weight_bp_dia 1.0',
        '--weight_decay 1e-5',
    ])

    # Delete old log if exists
    if os.path.exists(log_path):
        os.remove(log_path)

    cmd = (
        f'cd /root/zt/F3Mamba && '
        f'CUDA_VISIBLE_DEVICES=1,2,3,4 '
        f'torchrun --nproc_per_node=4 train_3fold_ddp_multilabel.py '
        f'--video_backbone {params["backbone"]} {params["extra"]} '
        f'{base_args} > {log_path} 2>&1'
    )

    print(f'\n{"="*60}')
    print(f'DIspatching: modal={modal}, seq_mult={seq_mult}, batch={batch_size}')
    print(f'Backbone: {params["backbone"]}')
    print(f'Cmd: {cmd[:120]}...')
    print(f'Log: {log_path}')
    print(f'{"="*60}')

    # Launch DDP training (non-blocking)
    proc = subprocess.Popen(cmd, shell=True)

    # Poll for completion (with timeout)
    timeout = 7200
    start = time.time()
    completed = False
    oom = False

    while time.time() - start < timeout:
        rc = proc.poll()
        if rc is not None:
            # Process exited — check log for completion marker
            if os.path.exists(log_path):
                with open(log_path) as f:
                    content = f.read()
                    if '3-Fold CV Complete!' in content:
                        completed = True
                    elif 'CUDA out of memory' in content:
                        oom = True
            break
        # Check log periodically
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
        time.sleep(60)

    if not completed:
        reason = 'OOM' if oom else 'timeout'
        print(f'FAILED ({reason})')
        wandb.log({'status': reason})
        wandb.finish(exit_code=1)
        return

    # Parse final results from log
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

    print(f'\n{"="*60}')
    print(f'Results: {results}')
    print(f'{"="*60}')

    wandb.log(results)
    wandb.finish()


if __name__ == '__main__':
    main()
