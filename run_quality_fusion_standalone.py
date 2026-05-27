import sys, os, time, gc, random, argparse
sys.path.insert(0, '/root/zt/F3Mamba')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

from Process.PostProcess import calculate_metric_simple_inference
from Models.InfinitePulse import Mamba2
from Models.FusionInfinitePulse import InfinitePulseFeatureExtractor
from sweep_xmae import (
    get_all_ids, seed_everything, neg_pearson_loss, safe_pearson,
    build_model_from_config, DATA_DEFAULTS, LABEL_STATS, build_loader, group_key,
    compute_hr_metrics_from_bvp
)


class BackPrimaryDualModel(nn.Module):
    """Back-primary dual-view: BP uses back_weight*back + (1-back_weight)*front."""
    def __init__(self, bp_hidden_dim=64, bp_dropout=0.05, back_weight=0.8,
                 calibrate=False, uncertainty_weight=False):
        super().__init__()
        feat_dim = 32
        self.back_weight = back_weight
        self.calibrate = calibrate
        self.uncertainty_weight = uncertainty_weight
        self.encoder_front = InfinitePulseFeatureExtractor([2,2,2,2], [32,32,32,32], 160, in_channels=3)
        self.encoder_back = InfinitePulseFeatureExtractor([2,2,2,2], [32,32,32,32], 160, in_channels=3)
        self.bp_mamba = nn.Sequential(
            Mamba2(d_model=feat_dim, expand=2, d_state=128, d_conv=4, chunk_size=160, headdim=32),
            Mamba2(d_model=feat_dim, expand=2, d_state=128, d_conv=4, chunk_size=160, headdim=32),
        )
        self.bp_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(feat_dim, bp_hidden_dim), nn.ReLU(inplace=True), nn.Dropout(bp_dropout),
            nn.Linear(bp_hidden_dim, bp_hidden_dim), nn.ReLU(inplace=True), nn.Dropout(bp_dropout),
        )
        self.bp_sys_head = nn.Linear(bp_hidden_dim, 1)
        self.bp_dia_head = nn.Linear(bp_hidden_dim, 1)
        if calibrate:
            self.calib_head = nn.Sequential(
                nn.Linear(bp_hidden_dim, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 4),
            )
        if uncertainty_weight:
            self.log_var_sys = nn.Linear(bp_hidden_dim, 1)
            self.log_var_dia = nn.Linear(bp_hidden_dim, 1)
            nn.init.constant_(self.log_var_sys.bias, -2.3)
            nn.init.constant_(self.log_var_dia.bias, -2.3)
        self.hr_fusion = nn.Linear(feat_dim * 2, feat_dim)
        self.hr_mamba = nn.Sequential(
            Mamba2(d_model=feat_dim, expand=2, d_state=128, d_conv=4, chunk_size=160, headdim=32),
            Mamba2(d_model=feat_dim, expand=2, d_state=128, d_conv=4, chunk_size=160, headdim=32),
        )
        self.rppg_head = nn.Sequential(nn.Linear(feat_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))

    def forward(self, data_input):
        front = data_input['front'].permute(0, 4, 1, 2, 3).contiguous()
        back = data_input['back'].permute(0, 4, 1, 2, 3).contiguous()
        f_feat = self.encoder_front(front)
        b_feat = self.encoder_back(back)
        bp_feat = self.back_weight * b_feat + (1 - self.back_weight) * f_feat
        bp_feat = self.bp_mamba(bp_feat)
        bp_shared = self.bp_pool(bp_feat.transpose(1, 2))
        bp_sys = self.bp_sys_head(bp_shared).squeeze(-1)
        bp_dia = self.bp_dia_head(bp_shared).squeeze(-1)
        if self.calibrate:
            calib = self.calib_head(bp_shared)
            alpha_sys = 1.0 + torch.tanh(calib[:, 0]) * 0.2
            beta_sys = calib[:, 1] * 0.1
            alpha_dia = 1.0 + torch.tanh(calib[:, 2]) * 0.2
            beta_dia = calib[:, 3] * 0.1
            bp_sys = bp_sys * alpha_sys + beta_sys
            bp_dia = bp_dia * alpha_dia + beta_dia
        hr_feat = self.hr_fusion(torch.cat([f_feat, b_feat], dim=-1))
        hr_feat = self.hr_mamba(hr_feat)
        rPPG = self.rppg_head(hr_feat).squeeze(-1)
        output = {'rPPG': rPPG, 'bp_sys': bp_sys, 'bp_dia': bp_dia, 'recon_loss': torch.tensor(0.0, device=front.device)}
        if self.uncertainty_weight:
            output['log_var_sys'] = self.log_var_sys(bp_shared).squeeze(-1)
            output['log_var_dia'] = self.log_var_dia(bp_shared).squeeze(-1)
        return output

DATA_DEFAULTS['num_workers'] = 0


class SingleViewModel(nn.Module):
    """Single-view baseline: encoder → mamba → rPPG + BP heads."""
    def __init__(self, bp_hidden_dim=64, bp_dropout=0.05):
        super().__init__()
        feat_dim = 32
        self.encoder = InfinitePulseFeatureExtractor(
            [2, 2, 2, 2], [32, 32, 32, 32], 160, in_channels=3
        )
        self.mamba = nn.Sequential(
            Mamba2(d_model=feat_dim, expand=2, d_state=128, d_conv=4, chunk_size=160, headdim=32),
            Mamba2(d_model=feat_dim, expand=2, d_state=128, d_conv=4, chunk_size=160, headdim=32),
        )
        self.rppg_head = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1),
        )
        self.bp_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(feat_dim, bp_hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(bp_dropout),
            nn.Linear(bp_hidden_dim, bp_hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(bp_dropout),
        )
        self.bp_sys_head = nn.Linear(bp_hidden_dim, 1)
        self.bp_dia_head = nn.Linear(bp_hidden_dim, 1)

    def forward(self, data_input):
        key = 'front' if 'front' in data_input else 'back'
        x = data_input[key].permute(0, 4, 1, 2, 3).contiguous()
        x = self.encoder(x)
        x = self.mamba(x)
        rPPG = self.rppg_head(x).squeeze(-1)
        bp = self.bp_pool(x.transpose(1, 2))
        bp_sys = self.bp_sys_head(bp).squeeze(-1)
        bp_dia = self.bp_dia_head(bp).squeeze(-1)
        return {'rPPG': rPPG, 'bp_sys': bp_sys, 'bp_dia': bp_dia, 'recon_loss': torch.tensor(0.0, device=x.device)}


def evaluate_per_subject(model, subject_ids, view, device):
    """Evaluate model per-subject; returns list of dicts with per-subject metrics."""
    model.eval()
    rows = []
    for uid in subject_ids:
        path = f'./ProcessedDataset/AZDualCamera/AZDualCamera_facephys_label_facephys_dataset_{uid}.pth'
        d = torch.load(path, weights_only=False)
        min_len = DATA_DEFAULTS['trim_left'] + DATA_DEFAULTS['trim_right'] + 1
        if len(d) < min_len:
            continue
        d = Subset(d, range(DATA_DEFAULTS['trim_left'], len(d) - DATA_DEFAULTS['trim_right']))
        loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=0)

        all_ps, all_pd, all_ls, all_ld = [], [], [], []
        all_hr_p, all_hr_g = [], []
        with torch.no_grad():
            for batch in loader:
                if view == 'front':
                    video = {'front': batch['modals']['video_front'].to(device)}
                elif view == 'back':
                    video = {'back': batch['modals']['video_back'].to(device)}
                else:
                    video = {'front': batch['modals']['video_front'].to(device),
                             'back': batch['modals']['video_back'].to(device)}
                output = model(video)
                ps = output['bp_sys'].detach().cpu() * LABEL_STATS['bp_sys_std'] + LABEL_STATS['bp_sys_mean']
                pd = output['bp_dia'].detach().cpu() * LABEL_STATS['bp_dia_std'] + LABEL_STATS['bp_dia_mean']
                all_ps.extend(ps.numpy())
                all_pd.extend(pd.numpy())
                all_ls.extend(batch['labels']['bp_sys'].float().numpy().mean(axis=-1))
                all_ld.extend(batch['labels']['bp_dia'].float().numpy().mean(axis=-1))
                rppg_np = output['rPPG'].detach().cpu().numpy()
                bvp_np = batch['labels']['bvp'].float().numpy()
                for b in range(rppg_np.shape[0]):
                    ph, gh = calculate_metric_simple_inference(rppg_np[b], bvp_np[b])
                    all_hr_p.append(ph)
                    all_hr_g.append(gh)
        if len(all_ps) == 0:
            continue
        err_sys = np.array(all_ps) - np.array(all_ls)
        err_dia = np.array(all_pd) - np.array(all_ld)
        err_hr = np.array(all_hr_p) - np.array(all_hr_g)
        rows.append({
            'uid': uid, 'n_samples': len(all_ps),
            'mae_sys': float(np.mean(np.abs(err_sys))),
            'me_sys': float(np.mean(err_sys)),
            'std_sys': float(np.std(err_sys, ddof=1)) if len(err_sys) > 1 else 0.0,
            'mae_dia': float(np.mean(np.abs(err_dia))),
            'me_dia': float(np.mean(err_dia)),
            'std_dia': float(np.std(err_dia, ddof=1)) if len(err_dia) > 1 else 0.0,
            'mae_hr': float(np.mean(np.abs(err_hr))) if len(err_hr) > 0 else 0.0,
            'me_hr': float(np.mean(err_hr)) if len(err_hr) > 0 else 0.0,
            'std_hr': float(np.std(err_hr, ddof=1)) if len(err_hr) > 1 else 0.0,
        })
    return rows


def evaluate_calibration(model, subject_ids, view, device, calib_fracs):
    """Evaluate with per-subject calibration at multiple calibration fractions.
    Returns dict: frac -> {fold_me, fold_std, fold_mae, per_subject_rows}
    """
    model.eval()
    calib_results = {}

    all_subj_preds = {}  # uid -> {'ps': [...], 'pd': [...], 'ls': [...], 'ld': [...]}
    for uid in subject_ids:
        path = f'./ProcessedDataset/AZDualCamera/AZDualCamera_facephys_label_facephys_dataset_{uid}.pth'
        d = torch.load(path, weights_only=False)
        min_len = DATA_DEFAULTS['trim_left'] + DATA_DEFAULTS['trim_right'] + 1
        if len(d) < min_len:
            continue
        d = Subset(d, range(DATA_DEFAULTS['trim_left'], len(d) - DATA_DEFAULTS['trim_right']))
        loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=0)

        ps_list, pd_list, ls_list, ld_list = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                if view == 'front':
                    video = {'front': batch['modals']['video_front'].to(device)}
                elif view == 'back':
                    video = {'back': batch['modals']['video_back'].to(device)}
                else:
                    video = {'front': batch['modals']['video_front'].to(device),
                             'back': batch['modals']['video_back'].to(device)}
                output = model(video)
                ps = output['bp_sys'].detach().cpu() * LABEL_STATS['bp_sys_std'] + LABEL_STATS['bp_sys_mean']
                pd = output['bp_dia'].detach().cpu() * LABEL_STATS['bp_dia_std'] + LABEL_STATS['bp_dia_mean']
                ps_list.extend(ps.numpy())
                pd_list.extend(pd.numpy())
                ls_list.extend(batch['labels']['bp_sys'].float().numpy().mean(axis=-1))
                ld_list.extend(batch['labels']['bp_dia'].float().numpy().mean(axis=-1))
        if len(ps_list) == 0:
            continue
        all_subj_preds[uid] = {
            'ps': np.array(ps_list), 'pd': np.array(pd_list),
            'ls': np.array(ls_list), 'ld': np.array(ld_list),
        }

    for frac in calib_fracs:
        per_subj = []
        all_err_sys, all_err_dia = [], []
        for uid, data in all_subj_preds.items():
            n = len(data['ps'])
            n_calib = max(1, int(n * frac))
            if n_calib >= n:
                continue
            bias_sys = np.mean(data['ps'][:n_calib] - data['ls'][:n_calib])
            bias_dia = np.mean(data['pd'][:n_calib] - data['ld'][:n_calib])
            corr_ps = data['ps'][n_calib:] - bias_sys
            corr_pd = data['pd'][n_calib:] - bias_dia
            err_sys = corr_ps - data['ls'][n_calib:]
            err_dia = corr_pd - data['ld'][n_calib:]
            all_err_sys.extend(err_sys)
            all_err_dia.extend(err_dia)
            per_subj.append({
                'uid': uid, 'n_calib': n_calib, 'n_eval': len(err_sys),
                'calib_me_sys': float(np.mean(data['ps'][:n_calib] - data['ls'][:n_calib])),
                'me_sys': float(np.mean(err_sys)),
                'std_sys': float(np.std(err_sys, ddof=1)) if len(err_sys) > 1 else 0.0,
                'mae_sys': float(np.mean(np.abs(err_sys))),
                'me_dia': float(np.mean(err_dia)),
                'std_dia': float(np.std(err_dia, ddof=1)) if len(err_dia) > 1 else 0.0,
                'mae_dia': float(np.mean(np.abs(err_dia))),
            })
        if len(all_err_sys) == 0:
            continue
        all_err_sys = np.array(all_err_sys)
        all_err_dia = np.array(all_err_dia)
        calib_results[frac] = {
            'fold_me_sys': float(np.mean(all_err_sys)),
            'fold_std_sys': float(np.std(all_err_sys, ddof=1)),
            'fold_mae_sys': float(np.mean(np.abs(all_err_sys))),
            'fold_me_dia': float(np.mean(all_err_dia)),
            'fold_std_dia': float(np.std(all_err_dia, ddof=1)),
            'fold_mae_dia': float(np.mean(np.abs(all_err_dia))),
            'per_subject': per_subj,
        }
    return calib_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quality_fusion', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--quality_temp', type=float, default=1.0)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--model_version', type=str, default='v3', choices=['v3', 'v4'])
    parser.add_argument('--single_view', type=str, default='', choices=['', 'front', 'back'])
    parser.add_argument('--back_primary', type=str, default='false', choices=['true', 'false'])
    parser.add_argument('--fusion_name', type=str, default='BDF80')
    parser.add_argument('--bvp_w', type=float, default=0.6)
    parser.add_argument('--recon_w', type=float, default=0.1)
    parser.add_argument('--calibrate', type=str, default='false', choices=['true', 'false'],
                        help='calibration head: learn affine correction from features')
    parser.add_argument('--uncertainty_weight', type=str, default='false', choices=['true', 'false'],
                        help='heteroscedastic uncertainty-weighted BP loss')
    parser.add_argument('--calib_fracs', type=str, default='',
                        help='comma-separated calibration fractions, e.g. "0.05,0.1,0.2"')
    parser.add_argument('--bp_hidden_dim', type=int, default=64,
                        help='hidden dimension for BP regressor')
    parser.add_argument('--lr', type=float, default=6.25e-5,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs per fold')
    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    seed_everything(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, GPU mem: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    print(f"torch num_threads: {torch.get_num_threads()}")

    os.chdir('/root/zt/F3Mamba')

    is_single = args.single_view in ('front', 'back')
    is_back_primary = args.back_primary == 'true'
    
    if is_single:
        print(f"Single-view mode: {args.single_view} only")
    elif is_back_primary:
        extras = []
        if args.calibrate == 'true':
            extras.append('calibration_head')
        if args.uncertainty_weight == 'true':
            extras.append('uncertainty_weight')
        extra_str = f' [+{",".join(extras)}]' if extras else ''
        print(f"{args.fusion_name} dual-view mode: BP uses 80% back + 20% front{extra_str}")
    
    calib_fracs = [float(x) for x in args.calib_fracs.split(',') if x.strip()] if args.calib_fracs else []
    if calib_fracs:
        print(f"Calibration fractions: {calib_fracs}")

    cfg = {
        'model_version': args.model_version, 'initial_mask_ratio': 0.9,
        'shared_encoder': True, 'bp_hidden_dim': args.bp_hidden_dim, 'bp_dropout': 0.05,
        'lr': args.lr, 'weight_decay': 1e-5,
        'recon_loss_weight': args.recon_w, 'bvp_loss_weight': args.bvp_w,
        'delay_max_shift': 12, 'align_blend': 0.05,
        'min_corr_improve': 0.03, 'avoid_edge_shift': True,
        'align_mode': 'none', 'batch_size': 2, 'epochs': args.epochs, 'eval_every_epochs': 5,
        'quality_fusion': args.quality_fusion == 'true', 'quality_temp': args.quality_temp,
        'bp_loss_type': 'mse', 'bp_huber_delta': 1.0,
    }
    print(f"Loss weights: bvp_w={cfg['bvp_loss_weight']}, recon_w={cfg['recon_loss_weight']}")

    if is_single:
        print(f"Single-view mode: {args.single_view} only")

    # Fold split
    all_ids = get_all_ids()
    print(f"Total subjects: {len(all_ids)}")
    groups = {}
    for uid in all_ids:
        groups.setdefault(group_key(uid), []).append(uid)
    group_keys = sorted(groups.keys())
    rng = np.random.RandomState(2024)
    shuffled = rng.permutation(group_keys).tolist()
    n, fs = len(shuffled), len(shuffled) // 3
    folds = []
    for fold_idx in range(3):
        start = fold_idx * fs
        tg = shuffled[start:] if fold_idx == 2 else shuffled[start:start + fs]
        folds.append([uid for gk in tg for uid in groups[gk]])

    bpm, bps = LABEL_STATS['bp_sys_mean'], LABEL_STATS['bp_sys_std']
    bdm, bds = LABEL_STATS['bp_dia_mean'], LABEL_STATS['bp_dia_std']

    fold_results = []
    per_subject_rows = []

    for fold_idx, test_ids in enumerate(folds):
        train_ids = [uid for uid in all_ids if uid not in test_ids]
        print(f"\n=== Fold {fold_idx+1}: train={len(train_ids)}, test={len(test_ids)} ===")

        train_loader, _ = build_loader(train_ids, cfg['batch_size'])
        test_loader, _ = build_loader(test_ids, cfg['batch_size'], shuffle=False)
        print(f"  train batches: {len(train_loader)}, test batches: {len(test_loader)}")

        if is_single:
            model = SingleViewModel(bp_hidden_dim=cfg['bp_hidden_dim'], bp_dropout=cfg['bp_dropout']).to(device)
        elif is_back_primary:
            model = BackPrimaryDualModel(
                bp_hidden_dim=cfg['bp_hidden_dim'], bp_dropout=cfg['bp_dropout'], back_weight=0.8,
                calibrate=args.calibrate == 'true',
                uncertainty_weight=args.uncertainty_weight == 'true',
            ).to(device)
        else:
            model = build_model_from_config(cfg).to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg['lr'], steps_per_epoch=len(train_loader), epochs=cfg['epochs']
        )
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(cfg['epochs']):
            st = time.time()
            model.train()
            total_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    if is_single:
                        video = {args.single_view: batch['modals'][f'video_{args.single_view}'].to(device, non_blocking=True)}
                    else:
                        video = {
                            'front': batch['modals']['video_front'].to(device, non_blocking=True),
                            'back': batch['modals']['video_back'].to(device, non_blocking=True),
                        }
                    output = model(video)
                    ps, pd = output['bp_sys'], output['bp_dia']
                    ls = batch['labels']['bp_sys'].to(device, non_blocking=True).float().mean(dim=-1)
                    ld = batch['labels']['bp_dia'].to(device, non_blocking=True).float().mean(dim=-1)
                    ls_norm = (ls - bpm) / bps
                    ld_norm = (ld - bdm) / bds
                    if args.uncertainty_weight == 'true':
                        lv_s, lv_d = output['log_var_sys'], output['log_var_dia']
                        loss_sys = 0.5 * torch.exp(-lv_s) * F.mse_loss(ps, ls_norm, reduction='none') + 0.5 * lv_s
                        loss_dia = 0.5 * torch.exp(-lv_d) * F.mse_loss(pd, ld_norm, reduction='none') + 0.5 * lv_d
                        loss = loss_sys.mean() + loss_dia.mean()
                    else:
                        loss = F.mse_loss(ps, ls_norm) + F.mse_loss(pd, ld_norm)
                    if 'bvp' in batch['labels'] and 'rPPG' in output:
                        loss += cfg['bvp_loss_weight'] * neg_pearson_loss(
                            output['rPPG'], batch['labels']['bvp'].to(device, non_blocking=True).float()
                        )
                    if 'recon_loss' in output:
                        loss += output['recon_loss'] * cfg['recon_loss_weight']
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()
                if batch_idx % 20 == 0 and batch_idx > 0:
                    print(f"    batch {batch_idx}/{len(train_loader)}", flush=True)
            elapsed = time.time() - st
            if (epoch + 1) % cfg['eval_every_epochs'] == 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} ({elapsed:.0f}s)", flush=True)

        # Fold-level test
        print("  Testing...", flush=True)
        all_ps, all_pd, all_ls, all_ld = [], [], [], []
        rppg_seq, bvp_seq = [], []
        with torch.no_grad():
            for batch in test_loader:
                if is_single:
                    video = {args.single_view: batch['modals'][f'video_{args.single_view}'].to(device)}
                else:
                    video = {'front': batch['modals']['video_front'].to(device),
                             'back': batch['modals']['video_back'].to(device)}
                output = model(video)
                ps = output['bp_sys'].detach().cpu() * bps + bpm
                pd = output['bp_dia'].detach().cpu() * bds + bdm
                all_ps.extend(ps.numpy())
                all_pd.extend(pd.numpy())
                all_ls.extend(batch['labels']['bp_sys'].float().numpy().mean(axis=-1))
                all_ld.extend(batch['labels']['bp_dia'].float().numpy().mean(axis=-1))
                rppg_seq.append(output['rPPG'].detach().cpu())
                bvp_seq.append(batch['labels']['bvp'].float())

        err_sys = np.array(all_ps) - np.array(all_ls)
        err_dia = np.array(all_pd) - np.array(all_ld)
        mae_sys = float(np.mean(np.abs(err_sys)))
        mae_dia = float(np.mean(np.abs(err_dia)))
        me_sys = float(np.mean(err_sys))
        me_dia = float(np.mean(err_dia))
        std_sys = float(np.std(err_sys, ddof=1))
        std_dia = float(np.std(err_dia, ddof=1))
        corr_sys = safe_pearson(np.array(all_ps), np.array(all_ls))[0]
        corr_dia = safe_pearson(np.array(all_pd), np.array(all_ld))[0]

        # HR metrics using standard method (window_len=640)
        hr_metrics = compute_hr_metrics_from_bvp(rppg_seq, bvp_seq, metric_prefix='test', original_fps=30, window_len=640)
        if hr_metrics:
            mae_hr = hr_metrics['test/hr_MAE']
            me_hr = 'N/A'
            std_hr = 'N/A'
            corr_hr = hr_metrics['test/hr_Pearson']
            n_hr_win = hr_metrics['test/hr_num_windows']
        else:
            mae_hr = 0.0
            me_hr = 'N/A'
            std_hr = 'N/A'
            corr_hr = 0.0
            n_hr_win = 0

        hr_str = f"MAE_hr={mae_hr:.2f}, Pearson_hr={corr_hr:.4f}, n_windows={n_hr_win}"
        print(f"  TEST MAE_sys={mae_sys:.2f}, MAE_dia={mae_dia:.2f}, "
              f"ME±STD_sys={me_sys:.2f}±{std_sys:.2f}, ME±STD_dia={me_dia:.2f}±{std_dia:.2f}, "
              f"Pearson_sys={corr_sys:.4f}, Pearson_dia={corr_dia:.4f}, {hr_str}", flush=True)

        fold_results.append({
            'mae_sys': mae_sys, 'mae_dia': mae_dia,
            'me_sys': me_sys, 'me_dia': me_dia,
            'std_sys': std_sys, 'std_dia': std_dia,
            'pearson_sys': corr_sys, 'pearson_dia': corr_dia,
            'mae_hr': mae_hr, 'me_hr': me_hr, 'std_hr': std_hr, 'pearson_hr': corr_hr,
        })

        # Per-subject evaluation
        if is_single:
            view = args.single_view
        elif is_back_primary:
            view = args.fusion_name
        else:
            view = 'dual'
        subj = evaluate_per_subject(model, test_ids, view, device)
        for r in subj:
            r['fold'] = fold_idx + 1
            r['experiment'] = args.tag
            r['view'] = view
        per_subject_rows.extend(subj)
        print(f"  Per-subject: {len(subj)} subjects collected")

        if calib_fracs:
            print(f"  Calibration evaluation [fracs={calib_fracs}]...", flush=True)
            calib = evaluate_calibration(model, test_ids, view, device, calib_fracs)
            for frac, cr in calib.items():
                print(f"    calib@{frac:.0%}: MAE_sys={cr['fold_mae_sys']:.2f}, "
                      f"ME±STD_sys={cr['fold_me_sys']:.2f}±{cr['fold_std_sys']:.2f}")
                fold_results.append({
                    'mae_sys': cr['fold_mae_sys'], 'mae_dia': cr['fold_mae_dia'],
                    'me_sys': cr['fold_me_sys'], 'me_dia': cr['fold_me_dia'],
                    'std_sys': cr['fold_std_sys'], 'std_dia': cr['fold_std_dia'],
                    'pearson_sys': 0.0, 'pearson_dia': 0.0,
                    'mae_hr': 0.0, 'me_hr': 0.0, 'std_hr': 0.0, 'pearson_hr': 0.0,
                    'calib_frac': frac, 'fold': fold_idx + 1,
                })
                for sr in cr['per_subject']:
                    sr['fold'] = fold_idx + 1
                    sr['calib_frac'] = frac
                    sr['experiment'] = args.tag + '_calib'
                    sr['view'] = view
                    per_subject_rows.append(sr)

        del model, optimizer, scheduler, scaler, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Cleanup done. GPU: {torch.cuda.memory_allocated()/1024**2:.0f} MiB", flush=True)

    # Final summary (exclude calibration rows)
    main_results = [r for r in fold_results if 'calib_frac' not in r]
    calib_summaries = [r for r in fold_results if 'calib_frac' in r]

    if main_results:
        avg_mae = np.mean([r['mae_sys'] for r in main_results])
        avg_p = np.mean([r['pearson_sys'] for r in main_results])
        avg_me_sys = np.mean([r['me_sys'] for r in main_results])
        avg_me_dia = np.mean([r['me_dia'] for r in main_results])
        avg_std_sys = np.mean([r['std_sys'] for r in main_results])
        avg_std_dia = np.mean([r['std_dia'] for r in main_results])
        avg_mae_hr = np.mean([r['mae_hr'] for r in main_results])
        avg_p_hr = np.mean([r['pearson_hr'] for r in main_results])

        print(f"\n{'='*60}")
        print(f"  FINAL: avg_mae_sys={avg_mae:.2f}, avg_pearson_sys={avg_p:.4f}")
        print(f"  BP AAMI: SYS ME±STD={avg_me_sys:.2f}±{avg_std_sys:.2f} mmHg")
        print(f"           DIA ME±STD={avg_me_dia:.2f}±{avg_std_dia:.2f} mmHg")
        print(f"  HR: MAE={avg_mae_hr:.2f}, Pearson={avg_p_hr:.4f} BPM (window_len=640, standard method)")
        print(f"{'='*60}")

    if calib_summaries:
        print(f"\n  --- Calibration Results (averaged across folds) ---")
        fracs = sorted(set(r['calib_frac'] for r in calib_summaries))
        for frac in fracs:
            crs = [r for r in calib_summaries if r['calib_frac'] == frac]
            if crs:
                amae = np.mean([r['mae_sys'] for r in crs])
                ame = np.mean([r['me_sys'] for r in crs])
                astd = np.mean([r['std_sys'] for r in crs])
                print(f"    calib@{frac:.0%}: MAE_sys={amae:.2f}, ME±STD_sys={ame:.2f}±{astd:.2f}")

    print(f"\n  Metrics calculation:")
    print(f"    BP error = pred(un-normalized) - ground_truth")
    print(f"      ME = mean(error), STD = std(error, ddof=1)")
    print(f"      MAE = mean(|error|), Pearson = pearsonr(pred, truth)")
    print(f"    HR error = HR(rPPG via Welch PSD) - HR(BVP via Welch PSD)")
    print(f"      (same ME/STD/MAE/Pearson formulas)")
    print(f"{'='*60}")
    print("DONE")

    if args.tag:
        tag = args.tag
    elif is_single:
        tag = f"single_{args.single_view}"
    elif is_back_primary:
        tag = f"{args.fusion_name.lower()}_dual"
    else:
        tag = 'dual'
    with open(f"/root/zt/F3Mamba/result_{tag}.txt", "w") as f:
        for i, r in enumerate(main_results):
            f.write(f"fold{i+1}: mae_sys={r['mae_sys']:.2f} mae_dia={r['mae_dia']:.2f} "
                    f"me_sys={r['me_sys']:.2f} std_sys={r['std_sys']:.2f} "
                    f"me_dia={r['me_dia']:.2f} std_dia={r['std_dia']:.2f} "
                    f"mae_hr={r['mae_hr']:.2f} pearson_hr={r['pearson_hr']:.4f} "
                    f"pearson_sys={r['pearson_sys']:.4f} pearson_dia={r['pearson_dia']:.4f}\n")
        f.write(f"avg: mae_sys={avg_mae:.2f} pearson_sys={avg_p:.4f} "
                f"me_sys={avg_me_sys:.2f} std_sys={avg_std_sys:.2f} "
                f"me_dia={avg_me_dia:.2f} std_dia={avg_std_dia:.2f} "
                f"mae_hr={avg_mae_hr:.2f} pearson_hr={avg_p_hr:.4f}\n")
        for frac in fracs:
            crs = [r for r in calib_summaries if r['calib_frac'] == frac]
            if crs:
                for j, cr in enumerate(crs):
                    f.write(f"calib_fold{j+1}@{frac:.0%}: mae_sys={cr['mae_sys']:.2f} "
                            f"me_sys={cr['me_sys']:.2f} std_sys={cr['std_sys']:.2f}\n")
        f.write(f"calib_summary:\n")
        for frac in fracs:
            crs = [r for r in calib_summaries if r['calib_frac'] == frac]
            if crs:
                amae = np.mean([r['mae_sys'] for r in crs])
                ame = np.mean([r['me_sys'] for r in crs])
                astd = np.mean([r['std_sys'] for r in crs])
                f.write(f"  calib@{frac:.0%}: MAE_sys={amae:.2f}, ME±STD={ame:.2f}±{astd:.2f}\n")
    print(f"Results saved to result_{tag}.txt")

    csv_path = f"/root/zt/F3Mamba/per_subject_{tag}.csv"
    with open(csv_path, "w") as f:
        cols = ['experiment', 'fold', 'view', 'uid', 'n_samples',
                'mae_sys', 'me_sys', 'std_sys', 'mae_dia', 'me_dia', 'std_dia',
                'mae_hr', 'std_hr']
        calib_cols = ['calib_frac', 'n_calib', 'n_eval', 'calib_me_sys']
        extra_cols = cols + calib_cols
        f.write(','.join(extra_cols) + '\n')
        for r in per_subject_rows:
            f.write(','.join(str(r.get(c, '')) for c in extra_cols) + '\n')
    n_raw = len([r for r in per_subject_rows if 'calib_frac' not in r])
    n_calib = len([r for r in per_subject_rows if 'calib_frac' in r])
    print(f"Per-subject CSV saved to {csv_path} ({n_raw} raw + {n_calib} calibration rows)")


if __name__ == '__main__':
    main()
