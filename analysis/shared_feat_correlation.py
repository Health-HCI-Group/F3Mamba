#!/root/miniconda3/envs/f3mamba/bin/python
"""
P1: 分析 shared_feat 与 BP 的相关性

目标：验证假设 - shared_feat 是否包含 BP 相关信息
"""

import torch
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.insert(0, '/root/zt/F3Mamba')
from Process.data_process import MultimodalDataset
from Models.FusionInfinitePulse import LateFusionInfinitePulseEnhancedBP

def load_model():
    """加载训练好的模型"""
    ckpt_path = '/root/zt/F3Mamba/checkpoints/fold0_LateFusionInfinitePulseEnhancedBP_AZDualCamera/LateFusionInfinitePulseEnhancedBP_AZDualCamera_model_weights.pth'
    
    if not os.path.exists(ckpt_path):
        print("No checkpoint found!")
        return None, None
    
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 从 checkpoint key 推断 fusion_mode
    if 'fusion_proj.0.weight' in ckpt:
        fusion_mode = 'conv'
    elif 'fusion_proj.weight' in ckpt and ckpt['fusion_proj.weight'].shape[0] == 32:
        fusion_mode = 'sums'
    elif 'fusion_proj.weight' in ckpt and ckpt['fusion_proj.weight'].shape[0] == 64:
        fusion_mode = 'concat'
    else:
        fusion_mode = 'sums'  # default
    
    print(f"Detected fusion_mode: {fusion_mode}")
    
    model = LateFusionInfinitePulseEnhancedBP(
        n_layers=[2, 2, 2, 2],
        filters=[32, 32, 32, 32],
        chunk_size=160,
        shared_backbone=True,
        fusion_mode=fusion_mode
    )
    
    model.load_state_dict(ckpt)
    return model, ckpt_path

def load_test_subjects(data_root, subject_ids=None, max_subjects=10):
    """加载测试集 subject 数据"""
    all_files = glob.glob(f"{data_root}/*_dataset_*.pth")
    
    if subject_ids is None:
        # 使用前几个 subjects 作为测试
        test_files = all_files[:max_subjects]
    else:
        test_files = [f for f in all_files if any(sid in f for sid in subject_ids)]
    
    print(f"Loading {len(test_files)} test subjects...")
    
    datasets = []
    for path in tqdm(test_files, desc="Loading"):
        try:
            d = torch.load(path, weights_only=False)
            datasets.append(d)
        except Exception as e:
            print(f"Error: {e}")
    
    return datasets

def extract_features_and_labels(model, datasets, device='cuda', max_samples=300):
    """提取 shared_feat 和标签"""
    model = model.to(device)
    model.eval()
    
    # Hook 提取 shared_feat
    shared_feats = []
    bp_sys_labels = []
    bp_dia_labels = []
    hr_labels = []
    
    hook_buffer = []
    handle = model.head.register_forward_hook(lambda m, i, o: hook_buffer.append(o.detach().cpu()))
    
    sample_count = 0
    
    with torch.no_grad():
        for dataset in datasets:
            for idx in range(len(dataset)):
                if sample_count >= max_samples:
                    break
                
                hook_buffer.clear()
                sample = dataset[idx]
                
                # 构建输入
                front = sample["modals"].get("video_front", None)
                back = sample["modals"].get("video_back", None)
                
                if front is None or back is None:
                    continue
                
                # 输入格式: {"front": tensor, "back": tensor}
                # 需要调整 shape
                if front.dim() == 3:  # (L, H, W, 3)
                    front = front.unsqueeze(0)  # (1, L, H, W, 3)
                if back.dim() == 3:
                    back = back.unsqueeze(0)
                
                x = {"front": front.to(device), "back": back.to(device)}
                
                try:
                    outputs = model(x)
                    
                    if len(hook_buffer) > 0:
                        sf = hook_buffer[0].squeeze(0).numpy()  # (L, 32)
                        shared_feats.append(sf)
                        
                        # 标签 (取序列平均值)
                        labels = sample["labels"]
                        bp_sys = labels.get("bp_sys", None)
                        bp_dia = labels.get("bp_dia", None)
                        hr = labels.get("bvp", labels.get("hr", None))
                        
                        if bp_sys is not None:
                            bp_sys_labels.append(bp_sys.mean().item() if torch.is_tensor(bp_sys) else np.mean(bp_sys))
                        if bp_dia is not None:
                            bp_dia_labels.append(bp_dia.mean().item() if torch.is_tensor(bp_dia) else np.mean(bp_dia))
                        if hr is not None:
                            hr_labels.append(hr.mean().item() if torch.is_tensor(hr) else np.mean(hr))
                        
                        sample_count += 1
                        
                except Exception as e:
                    print(f"Error at sample {sample_count}: {e}")
                    continue
    
    handle.remove()
    
    return {
        'shared_feat': np.array(shared_feats),  # (N, L, 32)
        'bp_sys': np.array(bp_sys_labels),
        'bp_dia': np.array(bp_dia_labels),
        'hr': np.array(hr_labels)
    }

def analyze_correlation(data):
    """分析相关性"""
    sf = data['shared_feat']  # (N, L, 32)
    bp_sys = data['bp_sys']
    bp_dia = data['bp_dia']
    hr = data['hr']
    
    print(f"\nshared_feat shape: {sf.shape}")
    print(f"bp_sys: {len(bp_sys)}, bp_dia: {len(bp_dia)}, hr: {len(hr)}")
    
    # 池化: 取时间平均
    sf_mean = sf.mean(axis=1)  # (N, 32)
    
    n_channels = sf_mean.shape[1]
    
    print("\n=== Channel-wise Pearson Correlation ===")
    print("Channel\tbp_sys\tbp_dia\tHR")
    print("-" * 50)
    
    best_bp_sys, best_bp_dia, best_hr = 0, 0, 0
    sig_bp_sys, sig_bp_dia, sig_hr = 0, 0, 0
    
    for c in range(n_channels):
        feat_c = sf_mean[:, c]
        
        corr_sys, p_sys = pearsonr(feat_c, bp_sys) if len(bp_sys) == len(feat_c) else (0, 1)
        corr_dia, p_dia = pearsonr(feat_c, bp_dia) if len(bp_dia) == len(feat_c) else (0, 1)
        corr_hr, p_hr = pearsonr(feat_c, hr) if len(hr) == len(feat_c) else (0, 1)
        
        if abs(corr_sys) > best_bp_sys:
            best_bp_sys = abs(corr_sys)
        if abs(corr_dia) > best_bp_dia:
            best_bp_dia = abs(corr_dia)
        if abs(corr_hr) > best_hr:
            best_hr = abs(corr_hr)
        
        if abs(corr_sys) > 0.1 and p_sys < 0.05:
            sig_bp_sys += 1
        if abs(corr_dia) > 0.1 and p_dia < 0.05:
            sig_bp_dia += 1
        if abs(corr_hr) > 0.1 and p_hr < 0.05:
            sig_hr += 1
        
        if abs(corr_sys) > 0.15 or abs(corr_dia) > 0.15 or abs(corr_hr) > 0.15:
            print(f"{c}\t{corr_sys:.4f}\t{corr_dia:.4f}\t{corr_hr:.4f}")
    
    print("\n=== Summary ===")
    print(f"Best bp_sys correlation: {best_bp_sys:.4f}")
    print(f"Best bp_dia correlation: {best_bp_dia:.4f}")
    print(f"Best HR correlation: {best_hr:.4f}")
    
    print(f"\nSignificant channels (|r| > 0.1, p < 0.05):")
    print(f"bp_sys: {sig_bp_sys}/{n_channels}")
    print(f"bp_dia: {sig_bp_dia}/{n_channels}")
    print(f"HR: {sig_hr}/{n_channels}")
    
    return {
        'best_bp_sys': best_bp_sys,
        'best_bp_dia': best_bp_dia,
        'best_hr': best_hr,
        'sig_bp_sys': sig_bp_sys,
        'sig_bp_dia': sig_bp_dia,
        'sig_hr': sig_hr
    }

def main():
    print("=" * 60)
    print("P1: 分析 shared_feat 与 BP 的相关性")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载模型
    print("\n[1] Loading model...")
    model, ckpt_path = load_model()
    if model is None:
        return
    
    # 2. 加载测试数据
    print("\n[2] Loading test data...")
    data_root = '/root/zt/F3Mamba/ProcessedDataset/AZDualCamera'
    datasets = load_test_subjects(data_root, max_subjects=15)
    
    if len(datasets) == 0:
        print("No data loaded!")
        return
    
    # 3. 提取特征
    print("\n[3] Extracting shared_feat...")
    data = extract_features_and_labels(model, datasets, device=device, max_samples=200)
    
    # 4. 分析相关性
    print("\n[4] Analyzing correlation...")
    results = analyze_correlation(data)
    
    # 5. 结论
    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    
    if results['best_bp_sys'] < 0.15 and results['best_bp_dia'] < 0.15:
        print("⚠️ shared_feat 与 BP 相关性极低 (< 0.15)")
        print("   验证假设: backbone 提取的特征不含 BP 信息")
        print("   建议: 更换 backbone 或引入 BP 专用特征")
    else:
        print("✅ shared_feat 包含一定 BP 相关性")
        print("   问题可能在于 BP head 设计")
    
    if results['best_hr'] > 0.3:
        print(f"✅ shared_feat 与 HR 相关性较高 ({results['best_hr']:.4f})")
        print("   backbone 能提取心率相关信息")

if __name__ == '__main__':
    main()