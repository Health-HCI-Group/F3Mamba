"""
3-Fold Cross-Validation DDP (multi-GPU) Training Script — Multi-Label Edition.

Predicts rPPG + SpO2 + BP systolic + BP diastolic simultaneously.
Reports 4 metrics (MAE, RMSE, MAPE, Pearson) for each target.

Launch:
    torchrun --nproc_per_node=4 train_3fold_ddp_multilabel.py \
        --video_backbone InfinitePulseMultitask \
        --batch_size 16 \
        --lr 5e-4 \
        --use_wandb \
        --gradient_accumulation_steps 2 \
        --modal_used front
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.amp import autocast, GradScaler
import random
import re
import argparse
import os
import glob
import gc
import numpy as np
import sys
import time
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Process.data_process import MultimodalDataset

try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False

from Process.Trainer import BaseProcessor, LossCalculator

try:
    from Models.InfinitePulse import InfinitePulse, InfinitePulseMultitask, InfinitePulseEnhancedBP, NegPearson
except ImportError:
    InfinitePulse = None
    InfinitePulseMultitask = None
    NegPearson = None
try:
    from Models.PhysMamba import PhysMamba
except ImportError:
    PhysMamba = None
try:
    from Models.RhythmFormer import RhythmFormer
except ImportError:
    RhythmFormer = None
try:
    from Models.FusionInfinitePulse import (EarlyFusionInfinitePulse, LateFusionInfinitePulse,
                                              EarlyFusionInfinitePulseMultitask,
                                              EarlyFusionInfinitePulseEnhancedBP,
                                              LateFusionInfinitePulseMultitask,
                                              LateFusionInfinitePulseEnhancedBP,
                                              LateFusionInfinitePulseEnhancedBPDemo,
                                              LateFusionInfinitePulseClassifier,
                                              FreqAlignLateFusionMultitask)
except ImportError:
    EarlyFusionInfinitePulse = None
    LateFusionInfinitePulse = None

from Process.PostProcess import calculate_metric_simple_inference
from Process.metrics import calculate_metrics_simple

try:
    from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                                  r2_score, precision_recall_curve, average_precision_score)
    from sklearn.calibration import calibration_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

FUSION_BACKBONES = {'EarlyFusionInfinitePulse', 'LateFusionInfinitePulse',
                     'EarlyFusionInfinitePulseMultitask', 'EarlyFusionInfinitePulseEnhancedBP',
                     'LateFusionInfinitePulseMultitask', 'LateFusionInfinitePulseEnhancedBP',
                     'LateFusionInfinitePulseEnhancedBPDemo',
                     'LateFusionInfinitePulseClassifier',
                     'FreqAlignLateFusionMultitask'}
MULTITASK_BACKBONES = {'InfinitePulseMultitask', 'InfinitePulseEnhancedBP'}
TARGET_LABELS = ['bvp', 'spo2', 'bp_sys', 'bp_dia']

# Global label normalisation statistics (computed from training data)
LABEL_STATS = {}

# Demographic data (loaded lazily): subject_visit → (age_norm, gender_female, gender_male)
_DEMO_MAP = None


def _load_demo_data():
    """Load demographic data from CSV files. Returns dict: 'subject_visit' → (age, is_female, is_male)."""
    global _DEMO_MAP
    if _DEMO_MAP is not None:
        return _DEMO_MAP
    import csv
    demo_map = {}

    # Read session_mapping.csv
    sess_to_subj = {}
    map_path = '/root/shared/az_dual_camera/az_hspt_data/data/organized/session_mapping.csv'
    if os.path.exists(map_path):
        with open(map_path) as f:
            for row in csv.DictReader(f):
                subj_visit = f"{row['subject_id']}_{row['visit']}"
                sess_to_subj[row['session_id']] = subj_visit

    # Read dataset_index.csv for age/gender
    index_path = '/root/shared/az_dual_camera/az_hspt_data/data/processed/dataset_index.csv'
    ages, genders = [], []
    demo_raw = {}  # subj_visit → (age, gender)
    if os.path.exists(index_path):
        with open(index_path) as f:
            for row in csv.DictReader(f):
                sess_id = row.get('session_id', '')
                age_str = row.get('subject_age', '').strip()
                gender_str = row.get('subject_gender', '').strip()
                if sess_id in sess_to_subj and age_str:
                    try:
                        age = float(age_str)
                        subj_visit = sess_to_subj[sess_id]
                        demo_raw[subj_visit] = (age, gender_str)
                        ages.append(age)
                        genders.append(gender_str)
                    except ValueError:
                        pass

    if ages:
        mean_age = sum(ages) / len(ages)
        std_age = (sum((a - mean_age) ** 2 for a in ages) / len(ages)) ** 0.5 or 1.0
    else:
        mean_age, std_age = 40.0, 15.0

    for subj_visit, (age, gender) in demo_raw.items():
        age_norm = (age - mean_age) / std_age
        is_female = 1.0 if gender == '女' else 0.0
        is_male = 1.0 if gender == '男' else 0.0
        if is_female == 0.0 and is_male == 0.0:
            is_female = 1.0  # default
        demo_map[subj_visit] = (age_norm, is_female, is_male)

    _DEMO_MAP = demo_map
    return demo_map


class SubjectIDDataset(Dataset):
    """Wrap a single-subject dataset to inject subject_id into labels."""
    def __init__(self, dataset, subject_id):
        self.ds = dataset
        self.sid = subject_id

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        sample['labels']['__subject_id__'] = self.sid
        return sample


def compute_label_stats(args, subject_ids, suffix=''):
    """Compute global mean/std for sequence-level labels (SpO2, BP) from the dataset.

    Each subject contributes the per-sequence mean.  These are aggregated across
    all subjects to obtain dataset-wide normalisation constants.
    """
    global LABEL_STATS
    all_means = {'spo2': [], 'bp_sys': [], 'bp_dia': []}

    for uid in subject_ids:
        path = (f'./ProcessedDataset/{args.dataset_name}/'
                f'{args.dataset_name}_{args.front_standard_type}_label_{args.label_standard_type}'
                f'_dataset_{uid}{suffix}.pth')
        if not os.path.exists(path):
            continue
        d = torch.load(path, weights_only=False)  # returns MultimodalDataset
        for key in all_means:
            if key in d.data["labels"]:
                vals = d.data["labels"][key]  # (N, L) where N = #sequences, L = chunk_size
                vals = vals.mean(dim=-1)       # per-sequence mean → (N,)
                all_means[key].extend(vals.tolist())

    for key in all_means:
        if all_means[key]:
            arr = np.array(all_means[key])
            LABEL_STATS[key] = {'mean': float(arr.mean()), 'std': float(arr.std()) + 1e-6}
            print(f"  Label stats [{key}]: mean={arr.mean():.4f}, std={arr.std():.4f} (n={len(arr)})")


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_rank(rank):
    return rank == 0


def barrier():
    dist.barrier()


def all_gather_concat(tensor, world_size):
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def reduce_sum(tensor, world_size):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_backbone', type=str, default='InfinitePulseMultitask',
                        choices=['InfinitePulse', 'InfinitePulseMultitask', 'InfinitePulseEnhancedBP',
                                 'PhysMamba', 'RhythmFormer',
                                 'EarlyFusionInfinitePulse', 'LateFusionInfinitePulse',
                                 'EarlyFusionInfinitePulseMultitask',
                                 'EarlyFusionInfinitePulseEnhancedBP',
                                 'LateFusionInfinitePulseMultitask',
                                 'LateFusionInfinitePulseEnhancedBP',
                                  'LateFusionInfinitePulseClassifier',
                                  'LateFusionInfinitePulseEnhancedBPDemo',
                                  'FreqAlignLateFusionMultitask'])
    parser.add_argument('--dataset_name', type=str, default='AZDualCamera')
    parser.add_argument('--front_standard_type', type=str, default='facephys')
    parser.add_argument('--label_standard_type', type=str, default='facephys')
    parser.add_argument('--seq_len', type=int, default=160)
    parser.add_argument('--seq_mult', type=int, default=1,
                        help='Concat N adjacent seqs along time (seq_len=N*160)')
    parser.add_argument('--resize_height', type=int, default=36)
    parser.add_argument('--resize_width', type=int, default=36)
    parser.add_argument('--target_hz', type=int, default=30)
    parser.add_argument('--modal_used', type=str, nargs='+', default=['front'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for Adam optimizer (default 0)')
    parser.add_argument('--backbone_lr_factor', type=float, default=1.0,
                        help='LR multiplier for backbone vs BP head (default 1.0 = same)')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'neg_pearson', 'multitask', 'uncertainty'])
    parser.add_argument('--use_scheduler', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='F3Mamba_Train')
    parser.add_argument('--n_folds', type=int, default=3)
    parser.add_argument('--data_suffix', type=str, default='_overlap_stride40')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--label_noise', type=float, default=0.0,
                        help='Gaussian noise std (mmHg) added to BP labels during training')
    parser.add_argument('--classification', action='store_true', default=False,
                        help='BP binary classification (HTN vs Normal) instead of regression')
    parser.add_argument('--cls_pos_weight', type=float, default=1.0,
                        help='Positive class weight for CE loss (classification)')
    parser.add_argument('--test_every', type=int, default=5)
    parser.add_argument('--loss_weight_bvp', type=float, default=1.0,
                        help='Loss weight for rPPG/BVP (NegPearson)')
    parser.add_argument('--loss_weight_spo2', type=float, default=0.5,
                        help='Loss weight for SpO2 (MSE)')
    parser.add_argument('--loss_weight_bp_sys', type=float, default=0.5,
                        help='Loss weight for BP systolic (MSE)')
    parser.add_argument('--loss_weight_bp_dia', type=float, default=0.5,
                        help='Loss weight for BP diastolic (MSE)')
    parser.add_argument('--fusion_mode', type=str, default='concat',
                        choices=['concat', 'sums', 'conv'],
                        help='Late fusion mode (LateFusionInfinitePulseMultitask)')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _group_key(subject_id):
    return re.sub(r'_V\d+$', '', subject_id)


def get_all_subject_ids(args, suffix=''):
    pattern = f'./ProcessedDataset/{args.dataset_name}/{args.dataset_name}_{args.front_standard_type}_label_{args.label_standard_type}_dataset_*{suffix}.pth'
    files = glob.glob(pattern)
    ids = []
    for f in files:
        basename = os.path.basename(f).replace('.pth', '')
        if suffix and basename.endswith(suffix):
            basename = basename[:-len(suffix)]
        parts = basename.split('_dataset_')
        if len(parts) == 2:
            uid = parts[1]
            # When suffix is empty, skip overlap files (which contain _overlap_stride40 in the ID)
            if not suffix and '_overlap_stride40' in uid:
                continue
            ids.append(uid)
    return sorted(ids)


def build_dataloader(args, subject_ids, shuffle, suffix, use_ddp, rank, world_size):
    datasets = []
    for uid in subject_ids:
        path = (f'./ProcessedDataset/{args.dataset_name}/'
                f'{args.dataset_name}_{args.front_standard_type}_label_{args.label_standard_type}'
                f'_dataset_{uid}{suffix}.pth')
        if not os.path.exists(path):
            if is_main_rank(rank):
                print(f"WARNING: {path} not found, skipping")
            continue
        d = torch.load(path, weights_only=False)
        d = Subset(d, range(1, len(d) - 1))
        if args.video_backbone == 'LateFusionInfinitePulseEnhancedBPDemo':
            d = SubjectIDDataset(d, uid)
        if args.seq_mult > 1:
            d = MultiSeqDataset(d, seq_mult=args.seq_mult)
        datasets.append(d)
        if is_main_rank(rank):
            uid_short = os.path.basename(path).split('_dataset_')[-1].replace('.pth','')
            print(f"  Loaded {uid_short}: {len(d)} groups (seq_mult={args.seq_mult})")

    if not datasets:
        raise RuntimeError("No datasets found!")

    combined = ConcatDataset(datasets)
    if is_main_rank(rank):
        print(f"  Total: {len(combined)} groups (pre-loaded)")

    num_workers = 0 if use_ddp else args.num_workers
    if use_ddp:
        sampler = DistributedSampler(combined, num_replicas=world_size, rank=rank, shuffle=shuffle)
        loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
        return loader, sampler
    else:
        loader = DataLoader(combined, batch_size=args.batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
        return loader, None


class MultiSeqDataset(Dataset):
    """Wrap a single-subject dataset to return concatenated adjacent sequences.

    Each original item has modals={'video_*': (160, H, W, 3)}, labels={...}.
    MultiSeqDataset concatenates seq_mult items along the time dim (dim=0),
    yielding sequences of length seq_mult × 160 frames.

    For stride=40 data, adjacent sequences overlap by 120 frames (75%).
    Actual unique frames per group = 160 + (seq_mult-1)*40.

    e.g. seq_mult=2 → 320 frames total, 200 unique (62% efficiency)
         seq_mult=4 → 640 frames total, 280 unique (44% efficiency)
         seq_mult=8 → 1280 frames total, 440 unique (34% efficiency)
    """

    def __init__(self, single_dataset: Subset, seq_mult: int = 2):
        self.ds = single_dataset
        self.seq_mult = seq_mult
        self.n = max(0, len(single_dataset) - seq_mult + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        items = [self.ds[idx + i] for i in range(self.seq_mult)]
        # items are dicts with 'modals' and 'labels'
        modals = {}
        labels = {}
        for key in items[0]['modals']:
            tensors = [item['modals'][key] for item in items]
            modals[key] = torch.cat(tensors, dim=0)
        for key in items[0]['labels']:
            val = items[0]['labels'][key]
            if isinstance(val, str):
                labels[key] = val  # pass through string (e.g. subject_id)
            else:
                tensors = [item['labels'][key] for item in items]
                labels[key] = torch.cat(tensors, dim=0)
        return {'modals': modals, 'labels': labels}


def create_model(args):
    if args.video_backbone == 'InfinitePulse':
        model = InfinitePulse(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len)
    elif args.video_backbone == 'InfinitePulseMultitask':
        model = InfinitePulseMultitask(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len)
    elif args.video_backbone == 'InfinitePulseEnhancedBP':
        model = InfinitePulseEnhancedBP(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len)
    elif args.video_backbone == 'PhysMamba':
        model = PhysMamba(theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=args.seq_len)
    elif args.video_backbone == 'RhythmFormer':
        model = RhythmFormer()
    elif args.video_backbone == 'EarlyFusionInfinitePulse':
        model = EarlyFusionInfinitePulse(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len)
    elif args.video_backbone == 'LateFusionInfinitePulse':
        model = LateFusionInfinitePulse(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len, shared_backbone=True)
    elif args.video_backbone == 'EarlyFusionInfinitePulseMultitask':
        model = EarlyFusionInfinitePulseMultitask(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len)
    elif args.video_backbone == 'EarlyFusionInfinitePulseEnhancedBP':
        model = EarlyFusionInfinitePulseEnhancedBP(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len)
    elif args.video_backbone == 'LateFusionInfinitePulseMultitask':
        model = LateFusionInfinitePulseMultitask(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len, shared_backbone=True, fusion_mode=args.fusion_mode)
    elif args.video_backbone == 'LateFusionInfinitePulseEnhancedBP':
        model = LateFusionInfinitePulseEnhancedBP(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len, shared_backbone=True, fusion_mode=args.fusion_mode)
    elif args.video_backbone == 'LateFusionInfinitePulseClassifier':
        model = LateFusionInfinitePulseClassifier(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len, shared_backbone=True, fusion_mode=args.fusion_mode)
    elif args.video_backbone == 'LateFusionInfinitePulseEnhancedBPDemo':
        model = LateFusionInfinitePulseEnhancedBPDemo(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len, shared_backbone=True, fusion_mode=args.fusion_mode)
    elif args.video_backbone == 'FreqAlignLateFusionMultitask':
        model = FreqAlignLateFusionMultitask(n_layers=[2, 2, 2, 2], filters=[32, 32, 32, 32], chunk_size=args.seq_len, shared_backbone=True)
    else:
        raise ValueError(f"Unknown backbone: {args.video_backbone}")
    return model


def zscore_normalize(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)


def process_batch_multitask(model, batch, device, args):
    """Forward for multi-task model — returns dict of all (pred, label) pairs.

    Shape conventions:
      - bvp:     (B, L)  per-timestep, z-score normalized
      - spo2:    (B,)    per-sequence, z-score normalized (for loss)
                         plus ``_denorm`` key for original-scale test metrics
      - bp_sys:  (B,)    same as spo2
      - bp_dia:  (B,)    same as spo2
    """
    raw_model = model.module if hasattr(model, 'module') else model
    modal = args.modal_used[0]

    if args.video_backbone in FUSION_BACKBONES:
        video_input = {
            "front": batch["modals"]["video_front"].to(device),
            "back": batch["modals"]["video_back"].to(device),
        }
    else:
        video_input = batch["modals"][f"video_{modal}"].to(device)

    # Demographic features for EnhancedBPDemo
    if args.video_backbone == 'LateFusionInfinitePulseEnhancedBPDemo':
        sid_key = '__subject_id__'
        demo_map = _load_demo_data()
        B = batch['labels'].get('bvp', batch['labels'].get('bp_sys')).shape[0]
        demo_feat = torch.zeros(B, 3)
        if sid_key in batch['labels']:
            sids = batch['labels'][sid_key]
            for b in range(min(B, len(sids))):
                sid = sids[b]
                if isinstance(sid, torch.Tensor):
                    sid = sid.item()
                sid = str(sid)
                if sid in demo_map:
                    demo_feat[b] = torch.tensor(demo_map[sid])
        video_input['demo'] = demo_feat.to(device)

    output = raw_model(video_input)

    results = {}
    if "bvp" in batch["labels"]:
        pred_bvp = zscore_normalize(output["rPPG"])
        label_bvp = zscore_normalize(batch["labels"]["bvp"].to(device))
        results["pred_bvp"] = pred_bvp
        results["label_bvp"] = label_bvp

    # spo2, bp_sys, bp_dia: globally-normalised labels, model outputs directly
    for key in ['spo2', 'bp_sys', 'bp_dia']:
        if key in batch["labels"] and key in output:
            pred = output[key]                     # (B,) — model raw output, learns norm scale
            label = batch["labels"][key].to(device).mean(dim=-1)  # (B,) raw labels

            stats = LABEL_STATS.get(key)
            if stats is not None:
                label_norm = (label - stats['mean']) / stats['std']
            else:
                label_norm = label

            if model.training and args.label_noise > 0 and key.startswith('bp_'):
                noise_std = args.label_noise / (stats['std'] if stats else 1.0)
                label_norm = label_norm + torch.randn_like(label_norm) * noise_std

            results[f"pred_{key}"] = pred          # model outputs on normalised scale
            results[f"label_{key}"] = label_norm

    # BP classification: convert continuous BP to binary label
    if 'bp_cls' in output:
        sys_label = batch["labels"]["bp_sys"].to(device).mean(dim=-1)
        dia_label = batch["labels"]["bp_dia"].to(device).mean(dim=-1)
        label_cls = ((sys_label >= 130) | (dia_label >= 80)).long()
        results["pred_bp_cls"] = output["bp_cls"]
        results["label_bp_cls"] = label_cls

    return results


def process_batch_single_modal(model, batch, device, args):
    """For single-modal single-task backbones — returns (pred_ppg, label_ppg)."""
    raw_model = model.module if hasattr(model, 'module') else model
    modal = args.modal_used[0]
    video_input = batch["modals"][f"video_{modal}"].to(device)
    label_ppg = batch["labels"]["bvp"].to(device)

    backbone = args.video_backbone
    if backbone == 'InfinitePulse':
        pred_ppg = raw_model(video_input)
    elif backbone == 'PhysMamba':
        video_input = video_input.permute(0, 4, 1, 3, 2)
        pred_ppg = raw_model(video_input)
    elif backbone == 'RhythmFormer':
        video_input = video_input.permute(0, 1, 4, 2, 3)
        pred_ppg = raw_model(video_input)
    else:
        pred_ppg = raw_model(video_input)

    pred_ppg = zscore_normalize(pred_ppg)
    label_ppg = zscore_normalize(label_ppg)
    return pred_ppg, label_ppg


# ── Multi-task loss ──────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weights = {
            'bvp': args.loss_weight_bvp,
            'spo2': args.loss_weight_spo2,
            'bp_sys': args.loss_weight_bp_sys,
            'bp_dia': args.loss_weight_bp_dia,
        }
        self.neg_pearson = NegPearson()
        self.mse = nn.MSELoss()
        cls_weight = args.cls_pos_weight if args.cls_pos_weight != 1.0 else 0.0
        self.cls_weight = cls_weight if cls_weight > 0 else 0.0
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.args = args

    def forward(self, results, device):
        total = 0.0
        losses = {}

        if 'pred_bvp' in results and 'label_bvp' in results:
            l = self.neg_pearson(results['pred_bvp'], results['label_bvp'])
            losses['bvp'] = l
            total += self.weights['bvp'] * l

        if 'pred_bp_cls' in results and 'label_bp_cls' in results:
            cls_pred = results['pred_bp_cls'].to(device)
            cls_label = results['label_bp_cls'].to(device)
            l = self.ce(cls_pred, cls_label)
            if self.cls_weight > 0:
                w = torch.where(cls_label == 1, self.cls_weight, 1.0)
                l = (l * w).mean()
            else:
                l = l.mean()
            losses['bp_cls'] = l
            total += self.weights['bp_sys'] * l  # reuse bp_sys weight for classification

        for key in ['spo2', 'bp_sys', 'bp_dia']:
            pk, lk = f'pred_{key}', f'label_{key}'
            if pk in results and lk in results:
                l = self.mse(results[pk], results[lk])
                losses[key] = l
                total += self.weights[key] * l

        return total, losses


class BPPriorLoss(MultiTaskLoss):
    """Extends MultiTaskLoss with BP physiological priors.

    Adds consistency losses for:
      - Pulse Pressure (PP = bp_sys - bp_dia)
      - Mean Arterial Pressure (MAP ≈ bp_dia + (bp_sys - bp_dia) / 3)
    These encode the known physiological coupling between systolic and diastolic BP.
    """
    def __init__(self, args):
        super().__init__(args)
        self.weight_pp = 0.3
        self.weight_map = 0.2

    def forward(self, results, device):
        total, losses = super().forward(results, device)

        if all(k in results for k in ['pred_bp_sys', 'pred_bp_dia', 'label_bp_sys', 'label_bp_dia']):
            pred_sys, pred_dia = results['pred_bp_sys'], results['pred_bp_dia']
            label_sys, label_dia = results['label_bp_sys'], results['label_bp_dia']

            # Pulse Pressure consistency
            loss_pp = self.mse(pred_sys - pred_dia, label_sys - label_dia)
            losses['pp'] = loss_pp
            total += self.weight_pp * loss_pp

            # Mean Arterial Pressure consistency
            pred_map = pred_dia + (pred_sys - pred_dia) / 3.0
            label_map = label_dia + (label_sys - label_dia) / 3.0
            loss_map = self.mse(pred_map, label_map)
            losses['map'] = loss_map
            total += self.weight_map * loss_map

        return total, losses


class UncertaintyWeightedLoss(MultiTaskLoss):
    """Multi-task loss with learnable task uncertainty (Kendall et al., 2018).

    L_total = Σ( exp(-log_var_t) * L_t + 0.5 * log_var_t )

    where log_var_t = log(σ_t²) is a learnable parameter per task.
    The model automatically balances tasks: high-uncertainty (noisy) tasks
    are down-weighted, and the log_var regulariser prevents σ → ∞.
    """
    def __init__(self, args):
        super().__init__(args)
        # Learnable log-variance per task (initialised near zero → σ ≈ 1)
        self.log_vars = nn.ParameterDict({
            'bvp': nn.Parameter(torch.zeros(1)),
            'bp_sys': nn.Parameter(torch.zeros(1)),
            'bp_dia': nn.Parameter(torch.zeros(1)),
        })

    def forward(self, results, device):
        total = 0.0
        losses = {}

        if 'pred_bvp' in results and 'label_bvp' in results:
            l = self.neg_pearson(results['pred_bvp'], results['label_bvp'])
            lv = self.log_vars['bvp'].to(device)
            precision = torch.exp(-lv)
            losses['bvp'] = l
            total += precision * l + 0.5 * lv

        for key in ['bp_sys', 'bp_dia']:
            pk, lk = f'pred_{key}', f'label_{key}'
            if pk in results and lk in results:
                l = self.mse(results[pk], results[lk])
                lv = self.log_vars[key].to(device)
                precision = torch.exp(-lv)
                losses[key] = l
                total += precision * l + 0.5 * lv

        # Also log the learned uncertainty values for monitoring
        losses['unc_bvp'] = self.log_vars['bvp'].detach()
        losses['unc_bp_sys'] = self.log_vars['bp_sys'].detach()
        losses['unc_bp_dia'] = self.log_vars['bp_dia'].detach()

        return total, losses


# ── Global fold results accumulator ──────────────────────────────
FOLD_RESULTS = []


def plot_confusion_matrix(cm, class_names=('Normal', 'HTN')):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def plot_pr_curve(recall, precision, ap):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2, label=f'AP = {ap:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    fig.tight_layout()
    return fig


def plot_bland_altman(pred, gt, label='BP', unit='mmHg'):
    diff = pred - gt
    mean = (pred + gt) / 2
    md = np.mean(diff)
    sd = np.std(diff)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(mean, diff, alpha=0.5, s=10)
    ax.axhline(md, color='r', linestyle='-', lw=1, label=f'Mean diff = {md:.2f}')
    ax.axhline(md + 1.96 * sd, color='gray', linestyle='--', lw=1,
               label=f'+1.96 SD = {md + 1.96*sd:.2f}')
    ax.axhline(md - 1.96 * sd, color='gray', linestyle='--', lw=1,
               label=f'-1.96 SD = {md - 1.96*sd:.2f}')
    ax.set_xlabel(f'Mean of Pred and GT ({unit})')
    ax.set_ylabel(f'Pred - GT ({unit})')
    ax.set_title(f'{label}: Bland-Altman')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_scatter(pred, gt, label='BP', unit='mmHg'):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(gt, pred, alpha=0.5, s=10)
    lims = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
    ax.plot(lims, lims, 'r--', lw=1, label='y=x')
    ax.set_xlim(lims[0] - 5, lims[1] + 5)
    ax.set_ylim(lims[0] - 5, lims[1] + 5)
    ax.set_xlabel(f'Ground Truth ({unit})')
    ax.set_ylabel(f'Prediction ({unit})')
    ax.set_title(f'{label}: Prediction vs Ground Truth')
    ax.legend()
    fig.tight_layout()
    return fig


class DDPTrainerMultitask:
    """DDP trainer for multi-task (rPPG + SpO2 + BP) with AMP & grad accum."""

    def __init__(self, args, model, train_loader_data, eval_loader_data, test_loader_data, rank, world_size):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        self.use_amp = args.use_amp
        self.grad_accum = args.gradient_accumulation_steps
        self.max_grad_norm = args.max_grad_norm

        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)

        self.train_loader, self.train_sampler = train_loader_data
        self.eval_loader, self.eval_sampler = eval_loader_data
        self.test_loader, self.test_sampler = test_loader_data

        self.is_multitask = args.video_backbone in MULTITASK_BACKBONES or args.video_backbone in FUSION_BACKBONES
        if args.loss_type == 'uncertainty':
            self.loss_fn = UncertaintyWeightedLoss(args)
        elif args.video_backbone in ('InfinitePulseEnhancedBP', 'EarlyFusionInfinitePulseEnhancedBP',
                                       'LateFusionInfinitePulseEnhancedBP', 'LateFusionInfinitePulseEnhancedBPDemo'):
            self.loss_fn = BPPriorLoss(args)
        else:
            self.loss_fn = MultiTaskLoss(args)

        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        if args.backbone_lr_factor != 1.0 and hasattr(raw_model, 'bp_parameters'):
            bp_params = raw_model.bp_parameters()
            backbone_params = raw_model.backbone_params()
            param_groups = [
                {"params": backbone_params, "lr": args.lr * args.backbone_lr_factor, "weight_decay": args.weight_decay},
                {"params": bp_params, "lr": args.lr, "weight_decay": args.weight_decay},
            ]
        else:
            param_groups = [{"params": self.model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}]
        if hasattr(self.loss_fn, 'log_vars') and self.loss_fn.log_vars:
            param_groups.append({
                "params": list(self.loss_fn.log_vars.values()),
                "lr": args.lr * 10,
                "weight_decay": 0.0,  # no decay on learned uncertainty
            })
        self.optimizer = torch.optim.Adam(param_groups)
        self.num_epoch = args.epochs
        self.num_train_batches = len(self.train_loader)
        self.step = 0

        if args.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=args.lr, epochs=self.num_epoch,
                steps_per_epoch=self.num_train_batches)
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0)

        self.scaler = GradScaler(device='cuda') if self.use_amp else None

        if args.dataset_name in ("Lab_multimodal", "AZDualCamera"):
            self.hr_calculate_batch_step = 4
            self.video_Hz = 30
        else:
            raise ValueError("No available test configuration, please check!")
        self.hr_calculate_step = self.hr_calculate_batch_step * args.seq_len

    def _compute_loss(self, batch):
        results = process_batch_multitask(self.model, batch, self.device, self.args)
        loss, sub_losses = self.loss_fn(results, self.device)
        return loss, sub_losses, results

    def train_loop(self, save_path=None):
        for epoch in range(self.num_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            loss_epoch, sub_losses_epoch, time_cost = self._train_one_epoch()

            if is_main_rank(self.rank):
                log_msg = (f"[Rank 0] Epoch {epoch+1}/{self.num_epoch} | "
                           f"Train/Loss: {loss_epoch:.4f} | Time: {time_cost:.2f}s | "
                           f"LR: {self.scheduler.get_last_lr()[0]:.2e}")
                for k, v in sub_losses_epoch.items():
                    log_msg += f" | {k}: {v:.4f}"
                print(log_msg)
                if USE_WANDB:
                    log_dict = {"train/loss_epoch": loss_epoch, "epoch": epoch,
                                "lr": self.scheduler.get_last_lr()[0]}
                    for k, v in sub_losses_epoch.items():
                        log_dict[f"train/loss_{k}"] = v
                    wandb.log(log_dict)

            self.eval_loop()

            if (epoch + 1) % self.args.test_every == 0:
                self.test_loop(epoch)

            barrier()

        if is_main_rank(self.rank) and save_path is not None:
            model_path = os.path.join(save_path, f"{self.args.video_backbone}_{self.args.dataset_name}_model_weights.pth")
            torch.save(self.model.module.state_dict(), model_path)
            print(f"[Rank 0] Model saved to {model_path}")

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()

        pbar = tqdm(self.train_loader, desc="Train", disable=not is_main_rank(self.rank),
                     ncols=100, file=sys.stdout)

        loss_sum = 0.0
        sub_loss_sums = {k: 0.0 for k in TARGET_LABELS}
        sub_loss_counts = {k: 0 for k in TARGET_LABELS}
        self.optimizer.zero_grad()

        for idx, batch in enumerate(pbar):
            if self.use_amp:
                with autocast('cuda'):
                    loss, sub_losses, _ = self._compute_loss(batch)
                    loss = loss / self.grad_accum
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                self.scaler.scale(loss).backward()
            else:
                loss, sub_losses, _ = self._compute_loss(batch)
                loss = loss / self.grad_accum
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()

            loss_sum += loss.item() * self.grad_accum
            for k, v in sub_losses.items():
                if k not in sub_loss_sums:
                    sub_loss_sums[k] = 0.0
                    sub_loss_counts[k] = 0
                sub_loss_sums[k] += v.item()
                sub_loss_counts[k] += 1

            if (idx + 1) % self.grad_accum == 0 or (idx + 1) == len(self.train_loader):
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            if is_main_rank(self.rank) and USE_WANDB:
                wandb.log({"train/loss_step": loss.item() * self.grad_accum}, step=self.step)
            self.step += 1

        avg_sub = {}
        for k in sub_loss_sums:
            if sub_loss_counts[k] > 0:
                avg_sub[k] = sub_loss_sums[k] / sub_loss_counts[k]

        return loss_sum / max(len(self.train_loader), 1), avg_sub, time.perf_counter() - t0

    def eval_loop(self):
        self.model.eval()
        pbar = tqdm(self.eval_loader, desc="Eval", disable=not is_main_rank(self.rank),
                     ncols=100, file=sys.stdout)

        loss_sum = 0.0
        count = 0
        with torch.no_grad():
            for batch in pbar:
                if self.use_amp:
                    with autocast('cuda'):
                        loss, sub_losses, _ = self._compute_loss(batch)
                else:
                    loss, sub_losses, _ = self._compute_loss(batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss_sum += loss.item()
                count += 1

        loss_tensor = torch.tensor([loss_sum, count], device=self.device, dtype=torch.float32)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        eval_loss = loss_tensor[0].item() / max(loss_tensor[1].item(), 1)

        if is_main_rank(self.rank):
            print(f"Valid/Loss: {eval_loss:.4f}")
            if USE_WANDB:
                wandb.log({"valid/Loss": eval_loss}, step=self.step)

    def test_loop(self, epoch):
        if is_main_rank(self.rank):
            print("Testing!")

        self.model.eval()

        all_preds = {key: [] for key in TARGET_LABELS}
        all_labels = {key: [] for key in TARGET_LABELS}
        cls_preds, cls_labels = [], []
        bp_denorm = {'pred_sys': [], 'label_sys': [], 'pred_dia': [], 'label_dia': []}

        pbar = tqdm(self.test_loader, desc="Test", disable=not is_main_rank(self.rank),
                     ncols=100, file=sys.stdout)

        with torch.no_grad():
            for batch in pbar:
                results = process_batch_multitask(self.model, batch, self.device, self.args)
                for key in TARGET_LABELS:
                    pk, lk = f"pred_{key}", f"label_{key}"
                    if pk in results and lk in results:
                        all_preds[key].append(results[pk].cpu())
                        all_labels[key].append(results[lk].cpu())
                if 'pred_bp_cls' in results and 'label_bp_cls' in results:
                    cls_preds.append(results['pred_bp_cls'].cpu())
                    cls_labels.append(results['label_bp_cls'].cpu())

        # Gather across GPUs — only rank 0 computes metrics
        for key in TARGET_LABELS:
            if not all_preds[key]:
                continue

            gather_pred = torch.cat(all_preds[key], dim=0)
            gather_label = torch.cat(all_labels[key], dim=0)

            sizes_pred = torch.tensor([gather_pred.shape[0]], device=self.device, dtype=torch.int)
            sizes_label = torch.tensor([gather_label.shape[0]], device=self.device, dtype=torch.int)
            size_list_pred = [torch.zeros(1, dtype=torch.int, device=self.device) for _ in range(self.world_size)]
            size_list_label = [torch.zeros(1, dtype=torch.int, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(size_list_pred, sizes_pred)
            dist.all_gather(size_list_label, sizes_label)
            max_size_pred = max(int(s.item()) for s in size_list_pred)
            max_size_label = max(int(s.item()) for s in size_list_label)

            if max_size_pred > 0:
                pad_pred = torch.zeros(max_size_pred, *gather_pred.shape[1:], device=self.device)
                pad_pred[:gather_pred.shape[0]] = gather_pred.to(self.device)
                gathered = [torch.zeros_like(pad_pred) for _ in range(self.world_size)]
                dist.all_gather(gathered, pad_pred)
                gathered_preds = [gathered[i][:int(size_list_pred[i].item())].cpu() for i in range(self.world_size)]

            if max_size_label > 0:
                pad_label = torch.zeros(max_size_label, *gather_label.shape[1:], device=self.device)
                pad_label[:gather_label.shape[0]] = gather_label.to(self.device)
                gathered_l = [torch.zeros_like(pad_label) for _ in range(self.world_size)]
                dist.all_gather(gathered_l, pad_label)
                gathered_labels = [gathered_l[i][:int(size_list_label[i].item())].cpu() for i in range(self.world_size)]

            if not is_main_rank(self.rank):
                continue

            all_pred_flat = torch.cat(gathered_preds, dim=0).flatten().numpy()
            all_label_flat = torch.cat(gathered_labels, dim=0).flatten().numpy()

            if key == 'bvp':
                # -- BVP → HR via FFT (same as single-task) --
                total_samples = len(all_pred_flat)
                pred_hr_list, gt_hr_list = [], []
                idx = 0
                while (idx + 1) * self.hr_calculate_step <= total_samples:
                    start = idx * self.hr_calculate_step
                    end = (idx + 1) * self.hr_calculate_step
                    pred_hr, gt_hr = calculate_metric_simple_inference(
                        all_pred_flat[start:end], all_label_flat[start:end],
                        original_fps=self.video_Hz)
                    pred_hr_list.append(pred_hr)
                    gt_hr_list.append(gt_hr)
                    idx += 1
                remaining = total_samples - idx * self.hr_calculate_step
                if remaining > 0:
                    pred_hr, gt_hr = calculate_metric_simple_inference(
                        all_pred_flat[idx * self.hr_calculate_step:],
                        all_label_flat[idx * self.hr_calculate_step:],
                        original_fps=self.video_Hz)
                    pred_hr_list.append(pred_hr)
                    gt_hr_list.append(gt_hr)

                pred_vals = np.array(pred_hr_list)
                gt_vals = np.array(gt_hr_list)
                display_key = 'hr'
                unit = 'BPM'
            else:
                # -- SpO2 / BP: denormalise using global label stats --
                stats = LABEL_STATS.get(key)
                if stats is not None:
                    pred_vals = all_pred_flat * stats['std'] + stats['mean']
                    gt_vals = all_label_flat * stats['std'] + stats['mean']
                else:
                    pred_vals = all_pred_flat
                    gt_vals = all_label_flat
                display_key = key
                unit = '%' if key == 'spo2' else 'mmHg'
                if key == 'bp_sys':
                    bp_denorm['pred_sys'] = pred_vals.copy()
                    bp_denorm['label_sys'] = gt_vals.copy()
                elif key == 'bp_dia':
                    bp_denorm['pred_dia'] = pred_vals.copy()
                    bp_denorm['label_dia'] = gt_vals.copy()

            metrics = calculate_metrics_simple(pred_vals, gt_vals)

            n_show = min(10, len(pred_vals))
            rng = np.random.RandomState(42)
            show_idx = rng.choice(len(pred_vals), n_show, replace=False)
            print(f"  [{display_key}] Sample pred/gt ({unit}):")
            for i, idx in enumerate(show_idx):
                print(f"    {i}: pred={pred_vals[idx]:.2f}, gt={gt_vals[idx]:.2f}")

            metrics_print = {}
            for mkey in ["FFT_MAE", "FFT_RMSE", "FFT_MAPE", "FFT_Pearson"]:
                val = metrics.get(mkey)
                if val is not None:
                    metrics_print[mkey] = val

            # Extra regression metrics (BP targets)
            if display_key in ('bp_sys', 'bp_dia') and HAS_SKLEARN:
                r2 = r2_score(gt_vals, pred_vals)
                metrics_print['R2'] = r2
                diff_abs = np.abs(pred_vals - gt_vals)
                p5 = np.mean(diff_abs <= 5) * 100
                p10 = np.mean(diff_abs <= 10) * 100
                p15 = np.mean(diff_abs <= 15) * 100
                metrics_print['P5(%)'] = p5
                metrics_print['P10(%)'] = p10
                metrics_print['P15(%)'] = p15
                print(f"  [{display_key}] R²={r2:.4f}  P5={p5:.1f}%  P10={p10:.1f}%  P15={p15:.1f}%")

            print(f"  [{display_key}] {metrics_print}")
            log_dict = {
                f"test/{display_key}_MAE": metrics.get("FFT_MAE"),
                f"test/{display_key}_RMSE": metrics.get("FFT_RMSE"),
                f"test/{display_key}_MAPE": metrics.get("FFT_MAPE"),
                f"test/{display_key}_Pearson": metrics.get("FFT_Pearson"),
            }
            if display_key in ('bp_sys', 'bp_dia') and HAS_SKLEARN:
                log_dict.update({
                    f"test/{display_key}_R2": r2,
                    f"test/{display_key}_P5": p5,
                    f"test/{display_key}_P10": p10,
                    f"test/{display_key}_P15": p15,
                })
            if USE_WANDB:
                wandb.log(log_dict)

            # Visualizations for BP targets
            if display_key in ('bp_sys', 'bp_dia') and HAS_MPL:
                try:
                    fig_ba = plot_bland_altman(pred_vals, gt_vals, label=display_key, unit=unit)
                    fig_sc = plot_scatter(pred_vals, gt_vals, label=display_key, unit=unit)
                    if USE_WANDB:
                        wandb.log({
                            f"viz/{display_key}_BlandAltman": wandb.Image(fig_ba),
                            f"viz/{display_key}_Scatter": wandb.Image(fig_sc),
                        })
                    plt.close(fig_ba)
                    plt.close(fig_sc)
                except Exception as e:
                    print(f"  [viz] Skipped plots for {display_key}: {e}")

            # Save full prediction set for BP analysis
            if display_key in ('bp_sys', 'bp_dia'):
                save_dir = f"./predictions/{self.args.fold_name}"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{display_key}_ep{epoch}.npz")
                np.savez_compressed(save_path,
                                    pred=pred_vals, gt=gt_vals,
                                    fold=self.args.fold_name, epoch=epoch)
                if is_main_rank(self.rank):
                    print(f"  [save] Predictions saved to {save_path} ({len(pred_vals)} samples)")

        # Classification metrics (if bp_cls exists)
        if cls_preds:
            cls_preds = torch.cat(cls_preds, dim=0).to(self.device)
            cls_labels = torch.cat(cls_labels, dim=0).to(self.device).long()
            # Gather across GPUs
            gathered_preds = [torch.zeros_like(cls_preds) for _ in range(self.world_size)]
            gathered_labels = [torch.zeros_like(cls_labels) for _ in range(self.world_size)]
            dist.all_gather(gathered_preds, cls_preds)
            dist.all_gather(gathered_labels, cls_labels)
            if is_main_rank(self.rank):
                cls_preds = torch.cat(gathered_preds, dim=0)
                cls_labels = torch.cat(gathered_labels, dim=0)
            probs = torch.softmax(cls_preds, dim=1)[:, 1].cpu().numpy()
            cls_labels_np = cls_labels.cpu().numpy()
            _, predicted = torch.max(cls_preds, 1)
            predicted_np = predicted.cpu().numpy()

            acc = (predicted == cls_labels).float().mean().item()
            tp = ((predicted == 1) & (cls_labels == 1)).sum().item()
            fp = ((predicted == 1) & (cls_labels == 0)).sum().item()
            fn = ((predicted == 0) & (cls_labels == 1)).sum().item()
            tn = ((predicted == 0) & (cls_labels == 0)).sum().item()
            ppv = tp / (tp + fp + 1e-8)
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            f1 = 2 * ppv * sens / (ppv + sens + 1e-8)
            cm = np.array([[tn, fp], [fn, tp]])

            print(f"  [bp_cls] Acc={acc:.4f} PPV={ppv:.4f} Sens={sens:.4f} Spec={spec:.4f} F1={f1:.4f}")
            print(f"  [bp_cls] Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

            log_dict = {
                "test/bp_cls_Acc": acc,
                "test/bp_cls_PPV": ppv,
                "test/bp_cls_Sens": sens,
                "test/bp_cls_Spec": spec,
                "test/bp_cls_F1": f1,
            }

            # AUC-ROC + visualizations
            if HAS_SKLEARN and len(set(cls_labels_np)) > 1:
                try:
                    auc = roc_auc_score(cls_labels_np, probs)
                    fpr, tpr, _ = roc_curve(cls_labels_np, probs)
                    precision, recall, _ = precision_recall_curve(cls_labels_np, probs)
                    ap = average_precision_score(cls_labels_np, probs)
                    log_dict["test/bp_cls_AUC"] = auc
                    log_dict["test/bp_cls_AP"] = ap
                    print(f"  [bp_cls] AUC={auc:.4f} AP={ap:.4f}")

                    if HAS_MPL:
                        fig_cm = plot_confusion_matrix(cm)
                        fig_roc = plot_roc_curve(fpr, tpr, auc)
                        fig_pr = plot_pr_curve(recall, precision, ap)
                        log_dict["viz/bp_cls_ConfusionMatrix"] = wandb.Image(fig_cm)
                        log_dict["viz/bp_cls_ROC"] = wandb.Image(fig_roc)
                        log_dict["viz/bp_cls_PRCurve"] = wandb.Image(fig_pr)
                        plt.close(fig_cm); plt.close(fig_roc); plt.close(fig_pr)
                except Exception as e:
                    print(f"  [bp_cls] Skipped AUC/plots: {e}")

            if USE_WANDB:
                wandb.log(log_dict)

        # HTN classification from BP regression predictions (if available)
        if isinstance(bp_denorm['pred_sys'], np.ndarray) and isinstance(bp_denorm['pred_dia'], np.ndarray):
            pred_sys = np.array(bp_denorm['pred_sys'])
            pred_dia = np.array(bp_denorm['pred_dia'])
            label_sys = np.array(bp_denorm['label_sys'])
            label_dia = np.array(bp_denorm['label_dia'])
            pred_cls = ((pred_sys >= 130) | (pred_dia >= 80)).astype(int)
            true_cls = ((label_sys >= 130) | (label_dia >= 80)).astype(int)
            acc_reg = (pred_cls == true_cls).mean()
            tp_r = ((pred_cls == 1) & (true_cls == 1)).sum()
            fp_r = ((pred_cls == 1) & (true_cls == 0)).sum()
            fn_r = ((pred_cls == 0) & (true_cls == 1)).sum()
            tn_r = ((pred_cls == 0) & (true_cls == 0)).sum()
            ppv_r = tp_r / (tp_r + fp_r + 1e-8)
            sens_r = tp_r / (tp_r + fn_r + 1e-8)
            spec_r = tn_r / (tn_r + fp_r + 1e-8)
            f1_r = 2 * ppv_r * sens_r / (ppv_r + sens_r + 1e-8)
            print(f"  [reg->cls] HTN from BP regression:")
            print(f"    Acc={acc_reg:.4f} PPV={ppv_r:.4f} Sens={sens_r:.4f} Spec={spec_r:.4f} F1={f1_r:.4f}")
            print(f"    Confusion: TN={tn_r} FP={fp_r} FN={fn_r} TP={tp_r}")
            if USE_WANDB:
                wandb.log({
                    "test/reg2cls_Acc": acc_reg, "test/reg2cls_PPV": ppv_r,
                    "test/reg2cls_Sens": sens_r, "test/reg2cls_Spec": spec_r,
                    "test/reg2cls_F1": f1_r,
                })


def run_fold(args, fold_idx, train_ids, test_ids, fold_name, rank, world_size):
    args.fold_name = fold_name  # for prediction saving in test_loop
    if args.video_backbone in FUSION_BACKBONES:
        args.modal_used = ['front', 'back']

    effective_batch = args.batch_size * world_size * args.gradient_accumulation_steps

    if is_main_rank(rank):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{args.n_folds}: {fold_name}")
        print(f"  Train: {len(train_ids)} subjects, Test: {len(test_ids)} subjects")
        print(f"  Config: backbone={args.video_backbone}, lr={args.lr}, "
              f"per_gpu_batch={args.batch_size}, "
              f"effective_batch={effective_batch}, "
              f"epochs={args.epochs}, AMP={'on' if args.use_amp else 'off'}, "
              f"modals={args.modal_used}, targets={TARGET_LABELS}")
        print(f"  Loss weights: bvp={args.loss_weight_bvp}, spo2={args.loss_weight_spo2}, "
              f"bp_sys={args.loss_weight_bp_sys}, bp_dia={args.loss_weight_bp_dia}")
        print(f"{'='*60}")

    if USE_WANDB:
        wandb_mode = "online" if (args.use_wandb and is_main_rank(rank)) else "disabled"
        wandb.init(
            project=args.wandb_project,
            name=f"{args.video_backbone}_ddp{world_size}_fold{fold_idx}_b{args.batch_size}",
            config=vars(args),
            settings=wandb.Settings(init_timeout=240),
            reinit=True,
            mode=wandb_mode,
        )

    train_loader_data = build_dataloader(
        args, train_ids, shuffle=True, suffix=args.data_suffix,
        use_ddp=True, rank=rank, world_size=world_size
    )
    eval_loader_data = build_dataloader(
        args, test_ids, shuffle=False, suffix=args.data_suffix,
        use_ddp=True, rank=rank, world_size=world_size
    )
    test_loader_data = build_dataloader(
        args, test_ids, shuffle=False, suffix=args.data_suffix,
        use_ddp=True, rank=rank, world_size=world_size
    )

    model = create_model(args)
    trainer = DDPTrainerMultitask(
        args, model, train_loader_data, eval_loader_data, test_loader_data, rank, world_size
    )

    save_path = os.path.join(args.output_dir, f"{fold_name}/")
    if is_main_rank(rank):
        os.makedirs(save_path, exist_ok=True)

    barrier()
    trainer.train_loop(save_path=save_path)
    barrier()

    if USE_WANDB and is_main_rank(rank):
        summary = dict(wandb.run.summary)
        sys_mae = summary.get('test/bp_sys_MAE')
        dia_mae = summary.get('test/bp_dia_MAE')
        sys_str = f"{sys_mae:.2f}" if isinstance(sys_mae, (int, float)) else "N/A"
        dia_str = f"{dia_mae:.2f}" if isinstance(dia_mae, (int, float)) else "N/A"
        print(f"  >> Fold {fold_idx+1} summary: bp_sys_MAE={sys_str}, "
              f"bp_dia_MAE={dia_str}, "
              f"bp_cls_Acc={summary.get('test/bp_cls_Acc', 'N/A')}")
        if 'test/bp_cls_AUC' in summary:
            print(f"  >> Fold {fold_idx+1} bp_cls_AUC={summary['test/bp_cls_AUC']:.4f}")
        if 'test/reg2cls_Acc' in summary:
            print(f"  >> Fold {fold_idx+1} reg2cls_Acc={summary['test/reg2cls_Acc']:.4f}")
        wandb.finish()


def main():
    local_rank, rank, world_size = setup_ddp()
    print(f"[Rank {rank}/{world_size}] DDP initialized on GPU {local_rank}")

    args = get_args()
    if args.seq_mult > 1:
        args.seq_len = args.seq_len * args.seq_mult
    seed_everything(args.seed)

    if is_main_rank(rank):
        os.makedirs(args.output_dir, exist_ok=True)

    all_ids = get_all_subject_ids(args, suffix=args.data_suffix)

    if is_main_rank(rank):
        print(f"Found {len(all_ids)} subjects (suffix: {args.data_suffix})")
        print(f"GPUs: {world_size}, Per-GPU batch: {args.batch_size}, "
              f"Grad accum: {args.gradient_accumulation_steps}, "
              f"Effective batch: {args.batch_size * world_size * args.gradient_accumulation_steps}, "
              f"AMP: {'on' if args.use_amp else 'off'}")

    # Group multi-video subjects (e.g. _V001, _V002) so they stay in the same fold
    groups = {}
    for uid in all_ids:
        groups.setdefault(_group_key(uid), []).append(uid)
    group_keys = list(groups.keys())

    if len(group_keys) < args.n_folds:
        raise ValueError(f"Need at least {args.n_folds} subject groups, got {len(group_keys)}")

    # Compute global label normalisation statistics (SpO2, BP) from all subjects
    if is_main_rank(rank):
        print("Computing label statistics ...")
    compute_label_stats(args, all_ids, suffix=args.data_suffix)
    if is_main_rank(rank):
        print(f"  LABEL_STATS: {LABEL_STATS}")

    rng = np.random.RandomState(args.seed)
    shuffled_groups = rng.permutation(group_keys).tolist()
    n_groups = len(shuffled_groups)
    fold_size = n_groups // args.n_folds

    for fold_idx in range(args.n_folds):
        start = fold_idx * fold_size
        if fold_idx == args.n_folds - 1:
            test_group_keys = shuffled_groups[start:]
        else:
            test_group_keys = shuffled_groups[start:start + fold_size]

        test_ids = []
        for gk in test_group_keys:
            test_ids.extend(groups[gk])

        train_ids = []
        for gk in shuffled_groups:
            if gk not in test_group_keys:
                train_ids.extend(groups[gk])

        fold_name = f"fold{fold_idx}_{args.video_backbone}_{args.dataset_name}"
        run_fold(args, fold_idx, train_ids, test_ids, fold_name, rank, world_size)

        # Cleanup between folds to prevent CUDA fragmentation
        gc.collect()
        torch.cuda.empty_cache()

    barrier()
    if is_main_rank(rank):
        print(f"\n{'='*60}")
        print(f"3-Fold CV Complete! Checkpoints: {os.path.abspath(args.output_dir)}")
        print(f"{'='*60}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
