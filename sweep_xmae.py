"""
xMAE Sweep — 最简单的写法：wandb.agent(sweep_id, train_fn)，train_fn 内 wandb.init()
"""
import sys, os, json, random, re, glob, time, traceback
sys.path.insert(0, '/root/zt/F3Mamba')

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, ConcatDataset, Subset
import wandb
import numpy as np
from scipy.stats import pearsonr

from Models.xMAE_Model import xMAEBPModel
from Models.xMAE_Model_V2 import xMAEBPModelV2
from Models.xMAE_Model_V3 import xMAEBPTaskDecoupled
from Process.PostProcess import calculate_metric_simple_inference

# Default to the known-good GPU unless user overrides it.
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# ---------------------------
# Hyperparameter organization
# ---------------------------
RUNTIME_DEFAULTS = {
    'seed': 2024,
    'sweep_project': 'F3Mamba',
    'sweep_name': 'xMAE-CrossModal-BP',
    'agent_count': 40,
    'init_timeout_sec': 240,
}

WANDB_DEFAULT_ENTITY = '2796321819-beihang-university'

DATA_DEFAULTS = {
    'dataset_glob': './ProcessedDataset/AZDualCamera/AZDualCamera_facephys_label_facephys_dataset_*.pth',
    'trim_left': 3,
    'trim_right': 1,
    'num_workers': 0,
    'pin_memory': True,
    'folds': 3,
    'split_seed': 2024,
}

MODEL_DEFAULTS = {
    'model_version': 'v2',
    'initial_mask_ratio': 0.8,
    'shared_encoder': True,
    'bp_hidden_dim': 64,
    'bp_dropout': 0.1,
    'recon_loss_weight': 0.1,
    'bvp_loss_weight': 0.6,
    'align_mode': 'none',
    'delay_max_shift': 0,
    'align_blend': 0.0,
    'min_corr_improve': 0.0,
    'avoid_edge_shift': True,
    'bp_loss_type': 'mse',
    'bp_huber_delta': 1.0,
    'quality_fusion': False,
    'quality_temp': 1.0,
}

TRAIN_DEFAULTS = {
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 4,
    'epochs': 20,
    'eval_every_epochs': 5,
}

LABEL_STATS = {
    'bp_sys_mean': 124.58,
    'bp_sys_std': 15.09,
    'bp_dia_mean': 75.33,
    'bp_dia_std': 9.84,
}

FPS = 30.0
MS_PER_FRAME = 1000.0 / FPS

SWEEP_PARAMETERS = {
    'model_version': {'values': ['v2']},
    'initial_mask_ratio': {'values': [0.8, 0.9]},
    'shared_encoder': {'values': [True, False]},
    'bp_hidden_dim': {'values': [64, 128]},
    'bp_dropout': {'values': [0.1, 0.2]},
    'lr': {'values': [8.0e-5, 1.0e-4, 1.2e-4]},
    'weight_decay': {'values': [1e-5, 5e-5]},
    'recon_loss_weight': {'values': [0.05, 0.1]},
    'bvp_loss_weight': {'values': [0.4, 0.6, 0.8]},
    'batch_size': {'values': [2]},
}


V3_PILOT_SWEEP_PARAMETERS = {
    'model_version': {'values': ['v3']},
    'initial_mask_ratio': {'values': [0.8, 0.9]},
    'shared_encoder': {'values': [True]},
    'bp_hidden_dim': {'values': [64]},
    'bp_dropout': {'values': [0.05, 0.1]},
    'lr': {'values': [6.25e-5, 1.0e-4]},
    'weight_decay': {'values': [1e-5]},
    'recon_loss_weight': {'values': [0.1]},
    'bvp_loss_weight': {'values': [0.6]},
    'delay_max_shift': {'values': [6, 12]},
    'align_blend': {'values': [0.05, 0.1]},
    'min_corr_improve': {'values': [-0.01, 0.0]},
    'avoid_edge_shift': {'values': [False]},
    'align_mode': {'values': ['xcorr_soft']},
    'batch_size': {'values': [2]},
}


V3_GATE_STRICT_SWEEP_PARAMETERS = {
    'model_version': {'values': ['v3']},
    'initial_mask_ratio': {'values': [0.9]},
    'shared_encoder': {'values': [True]},
    'bp_hidden_dim': {'values': [64]},
    'bp_dropout': {'values': [0.1]},
    'lr': {'values': [1.0e-4]},
    'weight_decay': {'values': [1e-5]},
    'recon_loss_weight': {'values': [0.1]},
    'bvp_loss_weight': {'values': [0.6]},
    'delay_max_shift': {'values': [12]},
    'align_blend': {'values': [0.05]},
    'min_corr_improve': {'values': [0.01, 0.02, 0.03]},
    'avoid_edge_shift': {'values': [True]},
    'align_mode': {'values': ['xcorr_soft']},
    'batch_size': {'values': [2]},
}


V3_MECH_AB_SWEEP_PARAMETERS = {
    'model_version': {'values': ['v3']},
    'initial_mask_ratio': {'values': [0.9]},
    'shared_encoder': {'values': [True]},
    'bp_hidden_dim': {'values': [64]},
    'bp_dropout': {'values': [0.1]},
    'lr': {'values': [1.0e-4]},
    'weight_decay': {'values': [1e-5]},
    'recon_loss_weight': {'values': [0.1]},
    'bvp_loss_weight': {'values': [0.6]},
    'delay_max_shift': {'values': [12]},
    'align_blend': {'values': [0.05]},
    'min_corr_improve': {'values': [0.03]},
    'avoid_edge_shift': {'values': [True]},
    'align_mode': {'values': ['none', 'xcorr_soft']},
    'batch_size': {'values': [2]},
}


V3_BP_MAINLINE_SWEEP_PARAMETERS = {
    'model_version': {'values': ['v3']},
    'initial_mask_ratio': {'values': [0.9]},
    'shared_encoder': {'values': [True]},
    'bp_hidden_dim': {'values': [64]},
    'bp_dropout': {'values': [0.05, 0.1]},
    'lr': {'values': [6.25e-5, 1.0e-4]},
    'weight_decay': {'values': [1e-5]},
    'recon_loss_weight': {'values': [0.1]},
    'bvp_loss_weight': {'values': [0.6]},
    'delay_max_shift': {'values': [12]},
    'align_blend': {'values': [0.05]},
    'min_corr_improve': {'values': [0.03]},
    'avoid_edge_shift': {'values': [True]},
    'align_mode': {'values': ['none']},
    'batch_size': {'values': [2]},
}


V3_BP_REWEIGHT_SWEEP_PARAMETERS = {
    'model_version': {'values': ['v3']},
    'initial_mask_ratio': {'values': [0.9]},
    'shared_encoder': {'values': [True]},
    'bp_hidden_dim': {'values': [64]},
    'bp_dropout': {'values': [0.05]},
    'lr': {'values': [6.25e-5]},
    'weight_decay': {'values': [1e-5]},
    'recon_loss_weight': {'values': [0.0, 0.05, 0.1]},
    'bvp_loss_weight': {'values': [0.2, 0.4, 0.6]},
    'delay_max_shift': {'values': [12]},
    'align_blend': {'values': [0.05]},
    'min_corr_improve': {'values': [0.03]},
    'avoid_edge_shift': {'values': [True]},
    'align_mode': {'values': ['none']},
    'batch_size': {'values': [2]},
}


V3_BP_HUBER_SWEEP_PARAMETERS = {
    'model_version': {'values': ['v3']},
    'initial_mask_ratio': {'values': [0.9]},
    'shared_encoder': {'values': [True]},
    'bp_hidden_dim': {'values': [64]},
    'bp_dropout': {'values': [0.05]},
    'lr': {'values': [6.25e-5]},
    'weight_decay': {'values': [1e-5]},
    'recon_loss_weight': {'values': [0.1]},
    'bvp_loss_weight': {'values': [0.6]},
    'delay_max_shift': {'values': [12]},
    'align_blend': {'values': [0.05]},
    'min_corr_improve': {'values': [0.03]},
    'avoid_edge_shift': {'values': [True]},
    'align_mode': {'values': ['none']},
    'batch_size': {'values': [2]},
    'bp_loss_type': {'values': ['mse', 'huber', 'smooth_l1']},
    'bp_huber_delta': {'values': [0.5, 1.0, 2.0]},
}


V3_BP_ROBUST_ONLY_SWEEP_PARAMETERS = {
    'model_version': {'values': ['v3']},
    'initial_mask_ratio': {'values': [0.9]},
    'shared_encoder': {'values': [True]},
    'bp_hidden_dim': {'values': [64]},
    'bp_dropout': {'values': [0.05]},
    'lr': {'values': [6.25e-5]},
    'weight_decay': {'values': [1e-5]},
    'recon_loss_weight': {'values': [0.1]},
    'bvp_loss_weight': {'values': [0.6]},
    'delay_max_shift': {'values': [12]},
    'align_blend': {'values': [0.05]},
    'min_corr_improve': {'values': [0.03]},
    'avoid_edge_shift': {'values': [True]},
    'align_mode': {'values': ['none']},
    'batch_size': {'values': [2]},
    'bp_loss_type': {'values': ['huber', 'smooth_l1']},
    'bp_huber_delta': {'values': [0.5, 1.0, 2.0]},
}


V3_QUALITY_FUSION_SWEEP_PARAMETERS = {
    'model_version': {'values': ['v3']},
    'initial_mask_ratio': {'values': [0.9]},
    'shared_encoder': {'values': [True]},
    'bp_hidden_dim': {'values': [64]},
    'bp_dropout': {'values': [0.05]},
    'lr': {'values': [6.25e-5]},
    'weight_decay': {'values': [1e-5]},
    'recon_loss_weight': {'values': [0.1]},
    'bvp_loss_weight': {'values': [0.6]},
    'delay_max_shift': {'values': [12]},
    'align_blend': {'values': [0.05]},
    'min_corr_improve': {'values': [0.03]},
    'avoid_edge_shift': {'values': [True]},
    'align_mode': {'values': ['none']},
    'batch_size': {'values': [2]},
    'quality_fusion': {'values': [False, True]},
}


V4_TASK_DECOUPLED_PARAMETERS = {
    'model_version': {'values': ['v4']},
    'initial_mask_ratio': {'values': [0.9]},
    'shared_encoder': {'values': [True]},
    'bp_hidden_dim': {'values': [64]},
    'bp_dropout': {'values': [0.05]},
    'lr': {'values': [6.25e-5]},
    'weight_decay': {'values': [1e-5]},
    'recon_loss_weight': {'values': [0.1]},
    'bvp_loss_weight': {'values': [0.6]},
    'delay_max_shift': {'values': [12]},
    'align_blend': {'values': [0.05]},
    'min_corr_improve': {'values': [0.03]},
    'avoid_edge_shift': {'values': [True]},
    'align_mode': {'values': ['xcorr_soft']},
    'batch_size': {'values': [2]},
}


def get_sweep_parameters():
    profile = os.environ.get('XMAE_SWEEP_PROFILE', '').strip().lower()
    if profile == 'v3_pilot':
        return V3_PILOT_SWEEP_PARAMETERS
    if profile == 'v3_gate_strict':
        return V3_GATE_STRICT_SWEEP_PARAMETERS
    if profile == 'v3_mech_ab':
        return V3_MECH_AB_SWEEP_PARAMETERS
    if profile == 'v3_bp_mainline':
        return V3_BP_MAINLINE_SWEEP_PARAMETERS
    if profile == 'v3_bp_reweight':
        return V3_BP_REWEIGHT_SWEEP_PARAMETERS
    if profile == 'v3_bp_huber':
        return V3_BP_HUBER_SWEEP_PARAMETERS
    if profile == 'v3_bp_robust_only':
        return V3_BP_ROBUST_ONLY_SWEEP_PARAMETERS
    if profile == 'v3_quality_fusion':
        return V3_QUALITY_FUSION_SWEEP_PARAMETERS
    if profile == 'v4_task_decoupled':
        return V4_TASK_DECOUPLED_PARAMETERS
    return SWEEP_PARAMETERS


def get_sweep_value(config, key, default):
    value = config.get(key, default)
    if key == 'shared_encoder':
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ('true', '1', 'yes', 'y'):
                return True
            if v in ('false', '0', 'no', 'n'):
                return False
            raise ValueError(f'Invalid shared_encoder value: {value}')
        return bool(value)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def build_run_config(raw_config):
    return {
        'model_version': get_sweep_value(raw_config, 'model_version', MODEL_DEFAULTS['model_version']),
        'lr': get_sweep_value(raw_config, 'lr', TRAIN_DEFAULTS['lr']),
        'batch_size': get_sweep_value(raw_config, 'batch_size', TRAIN_DEFAULTS['batch_size']),
        'epochs': TRAIN_DEFAULTS['epochs'],
        'eval_every_epochs': TRAIN_DEFAULTS['eval_every_epochs'],
        'weight_decay': get_sweep_value(raw_config, 'weight_decay', TRAIN_DEFAULTS['weight_decay']),
        'initial_mask_ratio': get_sweep_value(raw_config, 'initial_mask_ratio', MODEL_DEFAULTS['initial_mask_ratio']),
        'shared_encoder': get_sweep_value(raw_config, 'shared_encoder', MODEL_DEFAULTS['shared_encoder']),
        'bp_hidden_dim': get_sweep_value(raw_config, 'bp_hidden_dim', MODEL_DEFAULTS['bp_hidden_dim']),
        'bp_dropout': get_sweep_value(raw_config, 'bp_dropout', MODEL_DEFAULTS['bp_dropout']),
        'recon_loss_weight': get_sweep_value(raw_config, 'recon_loss_weight', MODEL_DEFAULTS['recon_loss_weight']),
        'bvp_loss_weight': get_sweep_value(raw_config, 'bvp_loss_weight', MODEL_DEFAULTS['bvp_loss_weight']),
        'align_mode': get_sweep_value(raw_config, 'align_mode', MODEL_DEFAULTS['align_mode']),
        'delay_max_shift': get_sweep_value(raw_config, 'delay_max_shift', MODEL_DEFAULTS['delay_max_shift']),
        'align_blend': get_sweep_value(raw_config, 'align_blend', MODEL_DEFAULTS['align_blend']),
        'min_corr_improve': get_sweep_value(raw_config, 'min_corr_improve', MODEL_DEFAULTS['min_corr_improve']),
        'avoid_edge_shift': get_sweep_value(raw_config, 'avoid_edge_shift', MODEL_DEFAULTS['avoid_edge_shift']),
        'bp_loss_type': str(get_sweep_value(raw_config, 'bp_loss_type', MODEL_DEFAULTS['bp_loss_type'])),
        'bp_huber_delta': float(get_sweep_value(raw_config, 'bp_huber_delta', MODEL_DEFAULTS['bp_huber_delta'])),
        'quality_fusion': get_sweep_value(raw_config, 'quality_fusion', MODEL_DEFAULTS['quality_fusion']),
        'quality_temp': float(get_sweep_value(raw_config, 'quality_temp', MODEL_DEFAULTS['quality_temp'])),
    }


def print_config_summary(raw_config, cfg):
    print(f'xMAE sweep raw config: {raw_config}')
    print('xMAE resolved config:')
    print(f"  train: lr={cfg['lr']}, batch_size={cfg['batch_size']}, epochs={cfg['epochs']}, eval_every={cfg['eval_every_epochs']}, wd={cfg['weight_decay']}")
    print(f"  model: version={cfg['model_version']}, shared_encoder={cfg['shared_encoder']}, bp_hidden_dim={cfg['bp_hidden_dim']}, bp_dropout={cfg['bp_dropout']}, mask_ratio={cfg['initial_mask_ratio']}, recon_w={cfg['recon_loss_weight']}, bvp_w={cfg['bvp_loss_weight']}")
    if str(cfg['model_version']).lower() == 'v3':
        print(f"  align: mode={cfg['align_mode']}, max_shift={cfg['delay_max_shift']}, blend={cfg['align_blend']}, min_improve={cfg['min_corr_improve']}, avoid_edge={cfg['avoid_edge_shift']}")
    print(f"  loss: bp_loss_type={cfg['bp_loss_type']}, bp_huber_delta={cfg['bp_huber_delta']}")
    if 'quality_fusion' in cfg:
        print(f"  fusion: quality_fusion={cfg['quality_fusion']}, quality_temp={cfg.get('quality_temp', 1.0)}")
    print(f"  data: folds={DATA_DEFAULTS['folds']}, split_seed={DATA_DEFAULTS['split_seed']}, trim=[{DATA_DEFAULTS['trim_left']}, -{DATA_DEFAULTS['trim_right']}], workers={DATA_DEFAULTS['num_workers']}")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return obj


def zscore_normalize(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)


def neg_pearson_loss(pred, target):
    if pred.dim() == 3 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if target.dim() == 3 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    if pred.shape != target.shape:
        raise ValueError(f'BVP shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}')
    pred = zscore_normalize(pred)
    target = zscore_normalize(target)
    corr = (pred * target).mean(dim=-1)
    return 1.0 - corr.mean()


class XCorrSoftAlignHook:
    """Cross-correlation based soft delay alignment with lightweight stats."""

    def __init__(self, max_shift, blend, min_corr_improve=0.0, avoid_edge_shift=True):
        self.max_shift = int(max(0, max_shift))
        self.blend = float(min(1.0, max(0.0, blend)))
        self.min_corr_improve = float(min_corr_improve)
        self.avoid_edge_shift = bool(avoid_edge_shift)
        self._latest_stats = None

    def _zscore(self, x):
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

    def _set_stats(self, best_shift, corr_before, corr_after, apply_mask):
        edge = torch.eq(best_shift.abs(), self.max_shift).float()
        self._latest_stats = {
            'shift_mean': float(best_shift.float().mean().item()),
            'shift_abs_mean': float(best_shift.float().abs().mean().item()),
            'shift_std': float(best_shift.float().std(unbiased=False).item()),
            'edge_frac': float(edge.mean().item()),
            'corr_before': float(corr_before.mean().item()),
            'corr_after': float(corr_after.mean().item()),
            'corr_gain': float((corr_after - corr_before).mean().item()),
            'apply_frac': float(apply_mask.float().mean().item()),
        }

    def consume_stats(self):
        stats = self._latest_stats
        self._latest_stats = None
        return stats

    def __call__(self, fused_feat, front_feat, back_feat, mask=None):
        if self.max_shift <= 0 or self.blend <= 0.0:
            self._latest_stats = None
            return fused_feat

        front = self._zscore(front_feat.mean(dim=-1))
        back = self._zscore(back_feat.mean(dim=-1))

        shifts = list(range(-self.max_shift, self.max_shift + 1))
        corr_candidates = []
        for shift in shifts:
            if shift > 0:
                f_seg = front[:, :-shift]
                b_seg = back[:, shift:]
            elif shift < 0:
                f_seg = front[:, -shift:]
                b_seg = back[:, :shift]
            else:
                f_seg = front
                b_seg = back
            corr_candidates.append((f_seg * b_seg).mean(dim=1))

        corr_all = torch.stack(corr_candidates, dim=1)
        best_idx = torch.argmax(corr_all, dim=1)
        shift_values = torch.tensor(shifts, device=best_idx.device, dtype=torch.long)
        best_shift = shift_values[best_idx]
        best_corr = corr_all.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        zero_idx = self.max_shift
        base_corr = corr_all[:, zero_idx]

        apply_mask = best_corr >= (base_corr + self.min_corr_improve)
        if self.avoid_edge_shift:
            apply_mask = apply_mask & (best_shift.abs() < self.max_shift)

        best_shift = torch.where(apply_mask, best_shift, torch.zeros_like(best_shift))

        aligned_back_feat = back_feat.clone()
        aligned_back_sig = back.clone()
        for b in range(back_feat.shape[0]):
            shift = int(best_shift[b].item())
            aligned_back_feat[b] = torch.roll(back_feat[b], shifts=shift, dims=0)
            aligned_back_sig[b] = torch.roll(back[b], shifts=shift, dims=0)

        corr_before = (front * back).mean(dim=1)
        corr_after = (front * aligned_back_sig).mean(dim=1)
        self._set_stats(best_shift, corr_before, corr_after, apply_mask)

        aligned_fused = 0.5 * (front_feat + aligned_back_feat)
        blend_scale = apply_mask.float().view(-1, 1, 1)
        blend = (self.blend * blend_scale).to(dtype=fused_feat.dtype, device=fused_feat.device)
        return (1.0 - blend) * fused_feat + blend * aligned_fused


def _extract_align_hook(model):
    module = model.module if hasattr(model, 'module') else model
    return getattr(module, 'align_hook', None)


def pop_align_stats(model):
    hook = _extract_align_hook(model)
    if hook is None or not hasattr(hook, 'consume_stats'):
        return None
    return hook.consume_stats()


def aggregate_align_stats(stats_list, metric_prefix):
    if len(stats_list) == 0:
        return {}

    key_map = {
        'shift_mean': 'align_shift_mean',
        'shift_abs_mean': 'align_shift_abs_mean',
        'shift_std': 'align_shift_std',
        'edge_frac': 'align_edge_frac',
        'corr_before': 'align_corr_before',
        'corr_after': 'align_corr_after',
        'corr_gain': 'align_corr_gain',
        'apply_frac': 'align_apply_frac',
    }
    metrics = {}
    for src_key, out_key in key_map.items():
        vals = [float(s[src_key]) for s in stats_list if src_key in s]
        if len(vals) > 0:
            metrics[f'{metric_prefix}/{out_key}'] = float(np.mean(vals))

    if f'{metric_prefix}/align_shift_mean' in metrics:
        metrics[f'{metric_prefix}/align_delay_ms_mean'] = float(metrics[f'{metric_prefix}/align_shift_mean'] * MS_PER_FRAME)
    if f'{metric_prefix}/align_shift_abs_mean' in metrics:
        metrics[f'{metric_prefix}/align_delay_ms_abs_mean'] = float(metrics[f'{metric_prefix}/align_shift_abs_mean'] * MS_PER_FRAME)
    if f'{metric_prefix}/align_shift_std' in metrics:
        metrics[f'{metric_prefix}/align_delay_ms_std'] = float(metrics[f'{metric_prefix}/align_shift_std'] * MS_PER_FRAME)

    return metrics


def add_align_quality_flags(metrics, metric_prefix):
    out = {}
    shift_abs = metrics.get(f'{metric_prefix}/align_shift_abs_mean')
    edge_frac = metrics.get(f'{metric_prefix}/align_edge_frac')
    corr_gain = metrics.get(f'{metric_prefix}/align_corr_gain')

    if shift_abs is not None:
        out[f'{metric_prefix}/align_mech_shift_active'] = int(shift_abs > 0.3)
    if edge_frac is not None:
        out[f'{metric_prefix}/align_mech_edge_ok'] = int(edge_frac < 0.4)
    if corr_gain is not None:
        out[f'{metric_prefix}/align_mech_corr_gain_ok'] = int(corr_gain > 0.005)

    if len(out) == 3:
        out[f'{metric_prefix}/align_mech_pass'] = int(all(v == 1 for v in out.values()))

    return out


def to_global_align_namespace(stage_metrics, stage_prefix):
    alias_map = {
        f'{stage_prefix}/align_shift_mean': 'align/shift_mean',
        f'{stage_prefix}/align_shift_abs_mean': 'align/shift_abs_mean',
        f'{stage_prefix}/align_shift_std': 'align/shift_std',
        f'{stage_prefix}/align_edge_frac': 'align/edge_frac',
        f'{stage_prefix}/align_corr_before': 'align/corr_before',
        f'{stage_prefix}/align_corr_after': 'align/corr_after',
        f'{stage_prefix}/align_corr_gain': 'align/corr_gain',
        f'{stage_prefix}/align_apply_frac': 'align/apply_frac',
        f'{stage_prefix}/align_delay_ms_mean': 'align/delay_ms_mean',
        f'{stage_prefix}/align_delay_ms_abs_mean': 'align/delay_ms_abs_mean',
        f'{stage_prefix}/align_delay_ms_std': 'align/delay_ms_std',
        f'{stage_prefix}/align_mech_shift_active': 'align/mech_shift_active',
        f'{stage_prefix}/align_mech_edge_ok': 'align/mech_edge_ok',
        f'{stage_prefix}/align_mech_corr_gain_ok': 'align/mech_corr_gain_ok',
        f'{stage_prefix}/align_mech_pass': 'align/mech_pass',
    }
    payload = {}
    for src, dst in alias_map.items():
        if src in stage_metrics:
            payload[dst] = float(stage_metrics[src])
    return payload


def build_model_from_config(cfg):
    version = str(cfg['model_version']).lower()
    if version == 'v1':
        model_cls = xMAEBPModel
        align_hook = None
    elif version == 'v2':
        model_cls = xMAEBPModelV2
        align_hook = None
    elif version == 'v3':
        model_cls = xMAEBPModelV2
        align_mode = str(cfg.get('align_mode', 'none')).lower()
        if align_mode == 'xcorr_soft':
            align_hook = XCorrSoftAlignHook(
                cfg['delay_max_shift'],
                cfg['align_blend'],
                min_corr_improve=cfg.get('min_corr_improve', 0.0),
                avoid_edge_shift=cfg.get('avoid_edge_shift', True),
            )
        elif align_mode == 'none':
            align_hook = None
        else:
            raise ValueError(f"Unsupported align_mode for v3: {cfg['align_mode']}")
    elif version == 'v4':
        model_cls = xMAEBPTaskDecoupled
        align_mode = str(cfg.get('align_mode', 'none')).lower()
        if align_mode == 'xcorr_soft':
            align_hook = XCorrSoftAlignHook(
                cfg['delay_max_shift'],
                cfg['align_blend'],
                min_corr_improve=cfg.get('min_corr_improve', 0.0),
                avoid_edge_shift=cfg.get('avoid_edge_shift', True),
            )
        elif align_mode == 'none':
            align_hook = None
        else:
            raise ValueError(f"Unsupported align_mode for v4: {cfg['align_mode']}")
    else:
        raise ValueError(f"Unsupported model_version: {cfg['model_version']}")

    if model_cls is xMAEBPModelV2:
        return model_cls(
            initial_mask_ratio=cfg['initial_mask_ratio'],
            shared_encoder=cfg['shared_encoder'],
            bp_hidden_dim=cfg['bp_hidden_dim'],
            bp_dropout=cfg['bp_dropout'],
            align_hook=align_hook,
            quality_fusion=cfg['quality_fusion'],
            quality_temp=cfg['quality_temp'],
        )
    if model_cls is xMAEBPTaskDecoupled:
        return model_cls(
            initial_mask_ratio=cfg['initial_mask_ratio'],
            shared_encoder=cfg['shared_encoder'],
            bp_hidden_dim=cfg['bp_hidden_dim'],
            bp_dropout=cfg['bp_dropout'],
            align_hook=align_hook,
            quality_temp=cfg['quality_temp'],
        )

    return model_cls(
        initial_mask_ratio=cfg['initial_mask_ratio'],
        shared_encoder=cfg['shared_encoder'],
        bp_hidden_dim=cfg['bp_hidden_dim'],
        bp_dropout=cfg['bp_dropout'],
    )


def safe_pearson(x, y):
    if len(x) < 2 or len(y) < 2:
        return 0.0, 1
    if np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return 0.0, 1
    return float(pearsonr(x, y)[0]), 0


def compute_hr_metrics_from_bvp(pred_bvp_seq, label_bvp_seq, metric_prefix='eval', original_fps=30, window_len=640):
    if len(pred_bvp_seq) == 0 or len(label_bvp_seq) == 0:
        return None

    pred_flat_parts = []
    label_flat_parts = []

    for pred_batch, label_batch in zip(pred_bvp_seq, label_bvp_seq):
        if pred_batch.dim() == 3 and pred_batch.shape[-1] == 1:
            pred_batch = pred_batch.squeeze(-1)
        if label_batch.dim() == 3 and label_batch.shape[-1] == 1:
            label_batch = label_batch.squeeze(-1)
        if pred_batch.dim() != 2 or label_batch.dim() != 2:
            continue
        if pred_batch.shape != label_batch.shape:
            continue

        pred_flat_parts.append(pred_batch.numpy().flatten())
        label_flat_parts.append(label_batch.numpy().flatten())

    if len(pred_flat_parts) == 0 or len(label_flat_parts) == 0:
        return None

    pred_flat = np.concatenate(pred_flat_parts)
    label_flat = np.concatenate(label_flat_parts)
    total_samples = min(len(pred_flat), len(label_flat))
    if total_samples < window_len:
        return None

    pred_flat = pred_flat[:total_samples]
    label_flat = label_flat[:total_samples]

    pred_hr_list, gt_hr_list = [], []
    idx = 0
    while (idx + 1) * window_len <= total_samples:
        start = idx * window_len
        end = (idx + 1) * window_len
        pred_hr, gt_hr = calculate_metric_simple_inference(
            pred_flat[start:end], label_flat[start:end], original_fps=original_fps
        )
        pred_hr_list.append(float(pred_hr))
        gt_hr_list.append(float(gt_hr))
        idx += 1

    remaining = total_samples - idx * window_len
    if remaining > 0:
        start = idx * window_len
        pred_hr, gt_hr = calculate_metric_simple_inference(
            pred_flat[start:], label_flat[start:], original_fps=original_fps
        )
        pred_hr_list.append(float(pred_hr))
        gt_hr_list.append(float(gt_hr))

    if len(pred_hr_list) == 0:
        return None

    pred_vals = np.array(pred_hr_list)
    gt_vals = np.array(gt_hr_list)
    mae = float(np.mean(np.abs(pred_vals - gt_vals)))
    rmse = float(np.sqrt(np.mean((pred_vals - gt_vals) ** 2)))

    pearson, _ = safe_pearson(pred_vals, gt_vals)

    return {
        f'{metric_prefix}/hr_MAE': mae,
        f'{metric_prefix}/hr_RMSE': rmse,
        f'{metric_prefix}/hr_Pearson': pearson,
        f'{metric_prefix}/hr_num_windows': int(len(pred_vals)),
    }


def evaluate_fold_metrics(model, test_loader, device, bp_sys_mean, bp_sys_std, bp_dia_mean, bp_dia_std,
                          metric_prefix='eval'):
    model.eval()
    preds, labels = [], []
    pred_bvp_seq, label_bvp_seq = [], []
    align_stats = []

    with torch.no_grad():
        for batch in test_loader:
            output = model({
                'front': batch['modals']['video_front'].to(device, non_blocking=True),
                'back': batch['modals']['video_back'].to(device, non_blocking=True),
            })
            stats = pop_align_stats(model)
            if stats is not None:
                align_stats.append(stats)
            ps = output['bp_sys'].detach().cpu() * bp_sys_std + bp_sys_mean
            pd = output['bp_dia'].detach().cpu() * bp_dia_std + bp_dia_mean
            ls = batch['labels']['bp_sys'].cpu().float().mean(dim=-1)
            ld = batch['labels']['bp_dia'].cpu().float().mean(dim=-1)
            preds.append(torch.stack([ps, pd], dim=1))
            labels.append(torch.stack([ls, ld], dim=1))
            if 'bvp' in batch['labels'] and 'rPPG' in output:
                pred_bvp_seq.append(output['rPPG'].detach().cpu())
                label_bvp_seq.append(batch['labels']['bvp'].cpu().float())

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    mae = (preds - labels).abs().mean(dim=0)
    r_sys, sys_nan = safe_pearson(preds[:, 0].numpy(), labels[:, 0].numpy())
    r_dia, dia_nan = safe_pearson(preds[:, 1].numpy(), labels[:, 1].numpy())

    metrics = {
        f'{metric_prefix}/mae_sys': mae[0].item(),
        f'{metric_prefix}/mae_dia': mae[1].item(),
        f'{metric_prefix}/pearson_sys': r_sys,
        f'{metric_prefix}/pearson_dia': r_dia,
        f'{metric_prefix}/num_samples': int(preds.shape[0]),
        f'{metric_prefix}/pearson_sys_nan': sys_nan,
        f'{metric_prefix}/pearson_dia_nan': dia_nan,
    }

    hr_metrics = compute_hr_metrics_from_bvp(pred_bvp_seq, label_bvp_seq, metric_prefix=metric_prefix, original_fps=30)
    if hr_metrics is not None:
        metrics.update(hr_metrics)

    metrics.update(aggregate_align_stats(align_stats, metric_prefix))
    metrics.update(add_align_quality_flags(metrics, metric_prefix))

    fold_summary = {
        'mae_sys': round(mae[0].item(), 2),
        'mae_dia': round(mae[1].item(), 2),
        'pearson_sys': round(r_sys, 4),
        'pearson_dia': round(r_dia, 4),
    }
    if hr_metrics is not None:
        fold_summary.update({
            'hr_mae': float(hr_metrics[f'{metric_prefix}/hr_MAE']),
            'hr_rmse': float(hr_metrics[f'{metric_prefix}/hr_RMSE']),
            'hr_pearson': float(hr_metrics[f'{metric_prefix}/hr_Pearson']),
        })

    return metrics, fold_summary

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_all_ids():
    """获取所有 subject IDs（复用训练脚本逻辑）"""
    files = glob.glob(DATA_DEFAULTS['dataset_glob'])
    ids = []
    for f in files:
        basename = os.path.basename(f).replace('.pth', '')
        parts = basename.split('_dataset_')
        if len(parts) == 2:
            uid = parts[1]
            if '_overlap_' in uid:
                continue
            ids.append(uid)
    return sorted(ids)

def group_key(uid):
    return re.sub(r'_V\d+$', '', uid)

def build_loader(subject_ids, batch_size=4, shuffle=True):
    """构建 DataLoader（无 DDP）并过滤长度过短样本"""
    datasets = []
    skipped_short = 0
    kept_subjects = 0
    for uid in subject_ids:
        path = f'./ProcessedDataset/AZDualCamera/AZDualCamera_facephys_label_facephys_dataset_{uid}.pth'
        d = torch.load(path, weights_only=False)
        min_len_needed = DATA_DEFAULTS['trim_left'] + DATA_DEFAULTS['trim_right'] + 1
        if len(d) < min_len_needed:
            skipped_short += 1
            continue
        d = Subset(d, range(DATA_DEFAULTS['trim_left'], len(d) - DATA_DEFAULTS['trim_right']))
        if len(d) == 0:
            skipped_short += 1
            continue
        datasets.append(d)
        kept_subjects += 1
    if len(datasets) == 0:
        raise RuntimeError('No usable subjects after filtering short sequences in build_loader')
    combined = ConcatDataset(datasets)
    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=DATA_DEFAULTS['num_workers'],
        pin_memory=DATA_DEFAULTS['pin_memory'],
    )
    info = {
        'requested_subjects': len(subject_ids),
        'kept_subjects': kept_subjects,
        'skipped_short_subjects': skipped_short,
        'samples': len(combined),
    }
    return loader, info


def safe_log(payload, global_step=None):
    if global_step is not None:
        payload['runtime/global_step'] = int(global_step)
    wandb.log(payload)


def mark_stage(stage, fold=None, epoch=None, global_step=None, start_time=None):
    payload = {'runtime/stage': stage}
    if fold is not None:
        payload['fold/index'] = int(fold)
    if epoch is not None:
        payload['runtime/epoch'] = int(epoch)
    if start_time is not None:
        payload['runtime/elapsed_sec'] = float(time.time() - start_time)
    safe_log(payload, global_step=global_step)

def train_xmae():
    """由 wandb.agent 直接调用"""
    wandb.init(settings=wandb.Settings(init_timeout=RUNTIME_DEFAULTS['init_timeout_sec']))

    start_time = time.time()
    global_step = 0
    current_fold = -1
    current_epoch = -1

    wandb.define_metric('runtime/global_step')
    wandb.define_metric('*', step_metric='runtime/global_step')

    try:
        # 从 sweep config 读取超参
        raw_config = {}
        for key in wandb.config.keys():
            val = wandb.config[key]
            if hasattr(val, 'value'):
                val = val.value
            raw_config[key] = val
        cfg = build_run_config(raw_config)
        print_config_summary(raw_config, cfg)

        # device + multi-GPU
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this script, but no GPU is available.')
        num_gpus = torch.cuda.device_count()
        device = torch.device('cuda:0')

        mark_stage('init', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
        safe_log({'runtime/num_gpus': num_gpus}, global_step=global_step)
        print(f'Using {num_gpus} GPU(s).')

        # Label stats (和训练脚本一致)
        bp_sys_mean = LABEL_STATS['bp_sys_mean']
        bp_sys_std = LABEL_STATS['bp_sys_std']
        bp_dia_mean = LABEL_STATS['bp_dia_mean']
        bp_dia_std = LABEL_STATS['bp_dia_std']

        # Fold split
        all_ids = get_all_ids()
        if len(all_ids) == 0:
            raise RuntimeError('No subject IDs found under ProcessedDataset/AZDualCamera')

        groups = {}
        for uid in all_ids:
            groups.setdefault(group_key(uid), []).append(uid)
        group_keys = sorted(groups.keys())

        rng = np.random.RandomState(DATA_DEFAULTS['split_seed'])
        shuffled = rng.permutation(group_keys).tolist()
        n = len(shuffled)
        fold_size = n // DATA_DEFAULTS['folds']

        fold_results = []

        for fold_idx in range(DATA_DEFAULTS['folds']):
            current_fold = fold_idx
            start = fold_idx * fold_size
            test_groups = shuffled[start:] if fold_idx == DATA_DEFAULTS['folds'] - 1 else shuffled[start:start + fold_size]

            test_ids = [uid for gk in test_groups for uid in groups[gk]]
            train_ids = [uid for gk in shuffled if gk not in test_groups for uid in groups[gk]]
            if len(train_ids) == 0 or len(test_ids) == 0:
                raise RuntimeError(f'Empty split at fold {fold_idx}: train={len(train_ids)}, test={len(test_ids)}')

            train_loader, train_info = build_loader(train_ids, cfg['batch_size'])
            test_loader, test_info = build_loader(test_ids, cfg['batch_size'], shuffle=False)
            if len(train_loader) == 0:
                raise RuntimeError(f'Empty train loader at fold {fold_idx}')
            if len(test_loader) == 0:
                raise RuntimeError(f'Empty test loader at fold {fold_idx}')

            print(f'Fold {fold_idx+1}: train={len(train_ids)}, test={len(test_ids)}')
            mark_stage('fold_start', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
            safe_log({
                'fold/index': current_fold,
                'fold/train_subjects': len(train_ids),
                'fold/test_subjects': len(test_ids),
                'fold/train_kept_subjects': train_info['kept_subjects'],
                'fold/test_kept_subjects': test_info['kept_subjects'],
                'fold/train_skipped_short_subjects': train_info['skipped_short_subjects'],
                'fold/test_skipped_short_subjects': test_info['skipped_short_subjects'],
                'fold/train_samples': train_info['samples'],
                'fold/test_samples': test_info['samples'],
            }, global_step=global_step)

            base_model = build_model_from_config(cfg).to(device)
            model = torch.nn.DataParallel(base_model) if num_gpus > 1 else base_model

            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg['lr'], steps_per_epoch=len(train_loader), epochs=cfg['epochs']
            )
            scaler = torch.cuda.amp.GradScaler()
            last_eval_summary = None

            for epoch in range(cfg['epochs']):
                current_epoch = epoch
                model.train()
                total_loss = 0.0
                total_recon = 0.0
                recon_count = 0
                total_bvp = 0.0
                bvp_count = 0
                align_stats_train = []
                for batch in train_loader:
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        video = {
                            'front': batch['modals']['video_front'].to(device, non_blocking=True),
                            'back': batch['modals']['video_back'].to(device, non_blocking=True),
                        }
                        output = model(video)
                        stats = pop_align_stats(model)
                        if stats is not None:
                            align_stats_train.append(stats)

                        ps = output['bp_sys']
                        pd = output['bp_dia']
                        ls = batch['labels']['bp_sys'].to(device, non_blocking=True).float().mean(dim=-1)
                        ld = batch['labels']['bp_dia'].to(device, non_blocking=True).float().mean(dim=-1)

                        ls_norm = (ls - bp_sys_mean) / bp_sys_std
                        ld_norm = (ld - bp_dia_mean) / bp_dia_std

                        bp_loss_type = cfg.get('bp_loss_type', 'mse')
                        if bp_loss_type == 'huber':
                            delta = cfg.get('bp_huber_delta', 1.0)
                            loss = torch.nn.functional.huber_loss(ps, ls_norm, delta=delta) + torch.nn.functional.huber_loss(pd, ld_norm, delta=delta)
                        elif bp_loss_type == 'smooth_l1':
                            loss = torch.nn.functional.smooth_l1_loss(ps, ls_norm) + torch.nn.functional.smooth_l1_loss(pd, ld_norm)
                        else:
                            loss = torch.nn.functional.mse_loss(ps, ls_norm) + torch.nn.functional.mse_loss(pd, ld_norm)
                        if 'bvp' in batch['labels'] and 'rPPG' in output:
                            pred_bvp = output['rPPG']
                            label_bvp = batch['labels']['bvp'].to(device, non_blocking=True).float()
                            bvp_loss = neg_pearson_loss(pred_bvp, label_bvp)
                            loss += cfg['bvp_loss_weight'] * bvp_loss
                            total_bvp += float(bvp_loss.detach().item())
                            bvp_count += 1
                        if 'recon_loss' in output:
                            recon_loss = output['recon_loss']
                            loss += recon_loss * cfg['recon_loss_weight']
                            total_recon += float(recon_loss.detach().item())
                            recon_count += 1

                    if not torch.isfinite(loss).item():
                        safe_log({'train/loss_isfinite': 0}, global_step=global_step)
                        raise RuntimeError(f'Non-finite loss at fold={current_fold}, epoch={current_epoch}, step={global_step}')

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    total_loss += loss.item()
                    global_step += 1

                mark_stage('train_epoch', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
                log_payload = {
                    'train/loss': total_loss / len(train_loader),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/loss_isfinite': 1,
                }
                if recon_count > 0:
                    log_payload['train/recon_loss'] = total_recon / recon_count
                if bvp_count > 0:
                    log_payload['train/bvp_loss'] = total_bvp / bvp_count
                log_payload.update(aggregate_align_stats(align_stats_train, 'train'))
                log_payload.update(add_align_quality_flags(log_payload, 'train'))
                safe_log(log_payload, global_step=global_step)

                if (epoch + 1) % cfg['eval_every_epochs'] == 0:
                    mark_stage('eval_start', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
                    eval_metrics, last_eval_summary = evaluate_fold_metrics(
                        model, test_loader, device,
                        bp_sys_mean, bp_sys_std, bp_dia_mean, bp_dia_std,
                        metric_prefix='eval'
                    )
                    eval_alias = to_global_align_namespace(eval_metrics, 'eval')
                    if len(eval_alias) > 0:
                        eval_metrics.update(eval_alias)
                    safe_log(eval_metrics, global_step=global_step)

            # fold final test
            mark_stage('test_start', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
            test_metrics, test_summary = evaluate_fold_metrics(
                model, test_loader, device,
                bp_sys_mean, bp_sys_std, bp_dia_mean, bp_dia_std,
                metric_prefix='test'
            )
            test_alias = to_global_align_namespace(test_metrics, 'test')
            if len(test_alias) > 0:
                test_metrics.update(test_alias)
            safe_log(test_metrics, global_step=global_step)

            print(
                f"Fold {fold_idx+1}: "
                f"TEST MAE_sys={test_summary['mae_sys']:.2f}, MAE_dia={test_summary['mae_dia']:.2f}, "
                f"Pearson_sys={test_summary['pearson_sys']:.4f}, Pearson_dia={test_summary['pearson_dia']:.4f}"
            )

            mark_stage('fold_end', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
            fold_results.append(test_summary)

        avg_mae = sum(r['mae_sys'] for r in fold_results) / DATA_DEFAULTS['folds']
        avg_p = sum(r['pearson_sys'] for r in fold_results) / DATA_DEFAULTS['folds']
        safe_log({'final/avg_mae_sys': avg_mae, 'final/avg_pearson_sys': avg_p}, global_step=global_step)
        mark_stage('done', fold=current_fold, epoch=current_epoch, global_step=global_step, start_time=start_time)
        print(f'Average MAE: {avg_mae:.2f}, Pearson: {avg_p:.4f}')

        # 保存本地结果
        exp_name = f"xmae_se{cfg['shared_encoder']}_h{cfg['bp_hidden_dim']}_d{cfg['bp_dropout']}"
        with open(f'/root/zt/F3Mamba/train_{exp_name}_result.json', 'w') as f:
            result_payload = {
                'config': raw_config,
                'resolved_config': cfg,
                'fold_results': fold_results,
                'avg_mae_sys': float(avg_mae),
                'avg_pearson_sys': float(avg_p),
                'completed': True,
            }
            json.dump(to_jsonable(result_payload), f, indent=2)
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


if __name__ == '__main__':
    seed_everything(RUNTIME_DEFAULTS['seed'])
    
    sweep_config = {
        'method': 'random',
        'name': RUNTIME_DEFAULTS['sweep_name'],
        'parameters': get_sweep_parameters(),
    }

    profile = os.environ.get('XMAE_SWEEP_PROFILE', '').strip().lower()
    if profile == 'v3_pilot':
        sweep_config['name'] = f"{RUNTIME_DEFAULTS['sweep_name']}-V3-Pilot"
    if profile == 'v3_gate_strict':
        sweep_config['name'] = f"{RUNTIME_DEFAULTS['sweep_name']}-V3-Gate-Strict"
    if profile == 'v3_mech_ab':
        sweep_config['name'] = f"{RUNTIME_DEFAULTS['sweep_name']}-V3-Mech-AB"
    if profile == 'v3_bp_mainline':
        sweep_config['name'] = f"{RUNTIME_DEFAULTS['sweep_name']}-V3-BP-Mainline"
    if profile == 'v3_bp_reweight':
        sweep_config['name'] = f"{RUNTIME_DEFAULTS['sweep_name']}-V3-BP-Reweight"
    if profile == 'v3_bp_huber':
        sweep_config['name'] = f"{RUNTIME_DEFAULTS['sweep_name']}-V3-BP-Huber"
    if profile == 'v3_bp_robust_only':
        sweep_config['name'] = f"{RUNTIME_DEFAULTS['sweep_name']}-V3-BP-Robust-Only"
    if profile == 'v3_quality_fusion':
        sweep_config['name'] = f"{RUNTIME_DEFAULTS['sweep_name']}-V3-Quality-Fusion"

    agent_count = int(os.environ.get('XMAE_AGENT_COUNT', RUNTIME_DEFAULTS['agent_count']))
    
    resume_sweep_id = os.environ.get('RESUME_SWEEP_ID', '').strip()
    if resume_sweep_id:
        sweep_id = resume_sweep_id
        if '/' not in sweep_id:
            entity = os.environ.get('WANDB_ENTITY', '').strip() or WANDB_DEFAULT_ENTITY
            sweep_id = f'{entity}/{RUNTIME_DEFAULTS["sweep_project"]}/{sweep_id}'
        print(f'Resuming sweep ID: {sweep_id}')
    else:
        sweep_id = wandb.sweep(sweep_config, project=RUNTIME_DEFAULTS['sweep_project'])
        print(f'Create sweep with ID: {sweep_id.split("/")[-1]}')
    sweep_parts = sweep_id.split('/')
    if len(sweep_parts) >= 3:
        sweep_entity, sweep_project, sweep_short_id = sweep_parts[-3], sweep_parts[-2], sweep_parts[-1]
    else:
        sweep_entity = os.environ.get('WANDB_ENTITY', '').strip() or WANDB_DEFAULT_ENTITY
        sweep_project = RUNTIME_DEFAULTS['sweep_project']
        sweep_short_id = sweep_parts[-1]
    print(f'Sweep URL: https://wandb.ai/{sweep_entity}/{sweep_project}/sweeps/{sweep_short_id}')
    
    wandb.agent(sweep_id, train_xmae, count=agent_count)
