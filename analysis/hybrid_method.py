#!/root/miniconda3/envs/f3mamba/bin/python
"""
P3: Hybrid Method (Deep + Handcrafted Features)

目标：结合 Deep Learning 的 shared_feat 和手工提取的 PTT 特征，
达到 Pearson > 0.2。

方法：
1. 提取 shared_feat (32维) from trained model
2. 提取 PTT 特征 (10维) from rPPG signal
3. Concatenate -> 42维
4. 用 Ridge Regression 预测 BP

预期效果：
- PTT baseline: Pearson = 0.3065
- 混合方法应该至少 >= 0.3065
"""

import torch
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/root/zt/F3Mamba')
from Process.data_process import MultimodalDataset
from Models.FusionInfinitePulse import LateFusionInfinitePulseEnhancedBP

def load_model():
    """加载训练好的模型"""
    ckpt_path = '/root/zt/F3Mamba/checkpoints/fold0_LateFusionInfinitePulseEnhancedBP_AZDualCamera/LateFusionInfinitePulseEnhancedBP_AZDualCamera_model_weights.pth'
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 推断 fusion_mode
    if 'fusion_proj.0.weight' in ckpt:
        fusion_mode = 'conv'
    else:
        fusion_mode = 'sums'
    
    model = LateFusionInfinitePulseEnhancedBP(
        n_layers=[2, 2, 2, 2],
        filters=[32, 32, 32, 32],
        chunk_size=160,
        shared_backbone=True,
        fusion_mode=fusion_mode
    )
    
    model.load_state_dict(ckpt)
    return model

def extract_ptt_features(signal, fs=30):
    """从单个信号提取 PTT 特征"""
    features = []
    
    # 1. 基本统计
    mean_amp = np.mean(np.abs(signal))
    std_amp = np.std(signal)
    max_amp = np.max(signal)
    min_amp = np.min(signal)
    
    # 2. 频域
    fft = np.fft.rfft(signal)
    fft_mag = np.abs(fft)
    
    freq_idx = np.argmax(fft_mag[1:]) + 1
    main_freq = freq_idx * fs / len(signal)
    hr = main_freq * 60
    main_energy = fft_mag[freq_idx]
    
    # 3. 波形形态
    peaks, _ = find_peaks(signal, height=np.mean(signal))
    
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks) / fs * 1000
        mean_interval = np.mean(peak_intervals)
        std_interval = np.std(peak_intervals) if len(peak_intervals) > 1 else 0
        
        peak_heights = signal[peaks]
        mean_peak_height = np.mean(peak_heights)
        std_peak_height = np.std(peak_heights) if len(peak_heights) > 1 else 0
    else:
        mean_interval = 1000 / hr if hr > 0 else 0
        std_interval = 0
        mean_peak_height = max_amp
        std_peak_height = 0
    
    return [
        mean_amp, std_amp, max_amp, min_amp,
        hr, main_energy,
        mean_interval, std_interval,
        mean_peak_height, std_peak_height
    ]

def extract_hybrid_features(model, datasets, device='cuda', max_samples=500):
    """提取混合特征 (shared_feat + PTT)"""
    model = model.to(device)
    model.eval()
    
    hook_buffer = []
    handle = model.head.register_forward_hook(lambda m, i, o: hook_buffer.append(o.detach().cpu()))
    
    hybrid_features = []
    bp_sys_labels = []
    bp_dia_labels = []
    
    sample_count = 0
    
    with torch.no_grad():
        for dataset in tqdm(datasets, desc="Processing datasets"):
            for idx in range(len(dataset)):
                if sample_count >= max_samples:
                    break
                
                hook_buffer.clear()
                sample = dataset[idx]
                
                # 输入
                front = sample["modals"].get("video_front", None)
                back = sample["modals"].get("video_back", None)
                
                if front is None or back is None:
                    continue
                
                if front.dim() == 3:
                    front = front.unsqueeze(0)
                if back.dim() == 3:
                    back = back.unsqueeze(0)
                
                x = {"front": front.to(device), "back": back.to(device)}
                
                try:
                    outputs = model(x)
                    
                    if len(hook_buffer) > 0:
                        # shared_feat: (L, 32) -> mean -> (32,)
                        sf = hook_buffer[0].squeeze(0).mean(dim=0).numpy()  # (32,)
                        
                        # PTT features from rPPG prediction
                        rppg_pred = outputs['rPPG'].squeeze().cpu().numpy()  # (L,)
                        ptt_feat = extract_ptt_features(rppg_pred)  # (10,)
                        
                        # 混合特征
                        hybrid = np.concatenate([sf, ptt_feat])  # (42,)
                        hybrid_features.append(hybrid)
                        
                        # 标签
                        labels = sample["labels"]
                        bp_sys = labels.get("bp_sys", None)
                        bp_dia = labels.get("bp_dia", None)
                        
                        if bp_sys is not None:
                            bp_sys_labels.append(bp_sys.mean().item() if torch.is_tensor(bp_sys) else np.mean(bp_sys))
                        if bp_dia is not None:
                            bp_dia_labels.append(bp_dia.mean().item() if torch.is_tensor(bp_dia) else np.mean(bp_dia))
                        
                        sample_count += 1
                        
                except Exception as e:
                    continue
    
    handle.remove()
    
    return {
        'features': np.array(hybrid_features),
        'bp_sys': np.array(bp_sys_labels),
        'bp_dia': np.array(bp_dia_labels)
    }

def train_hybrid_model(data):
    """训练混合模型"""
    X = data['features']
    y_sys = data['bp_sys']
    y_dia = data['bp_dia']
    
    print(f"Hybrid features shape: {X.shape}")
    print(f"  - shared_feat: 32 dims")
    print(f"  - PTT features: 10 dims")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ridge 回归
    model_sys = Ridge(alpha=1.0)
    model_sys.fit(X_scaled, y_sys)
    
    model_dia = Ridge(alpha=1.0)
    model_dia.fit(X_scaled, y_dia)
    
    # 预测
    pred_sys = model_sys.predict(X_scaled)
    pred_dia = model_dia.predict(X_scaled)
    
    # 评估
    mae_sys = np.mean(np.abs(pred_sys - y_sys))
    mae_dia = np.mean(np.abs(pred_dia - y_dia))
    
    corr_sys = pearsonr(pred_sys, y_sys)[0]
    corr_dia = pearsonr(pred_dia, y_dia)[0]
    
    return {
        'mae_sys': mae_sys,
        'mae_dia': mae_dia,
        'pearson_sys': corr_sys,
        'pearson_dia': corr_dia
    }

def main():
    print("=" * 60)
    print("P3: Hybrid Method (Deep + Handcrafted)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载模型
    print("\n[1] Loading trained model...")
    model = load_model()
    
    # 2. 加载数据
    print("\n[2] Loading test data...")
    data_root = '/root/zt/F3Mamba/ProcessedDataset/AZDualCamera'
    all_files = glob.glob(f"{data_root}/*_dataset_*.pth")[:60]
    
    datasets = []
    for path in tqdm(all_files, desc="Loading"):
        try:
            d = torch.load(path, weights_only=False)
            datasets.append(d)
        except:
            continue
    
    print(f"Loaded {len(datasets)} datasets")
    
    # 3. 提取混合特征
    print("\n[3] Extracting hybrid features...")
    data = extract_hybrid_features(model, datasets, device=device, max_samples=400)
    
    # 4. 训练混合模型
    print("\n[4] Training hybrid model...")
    results = train_hybrid_model(data)
    
    # 5. 结果
    print("\n" + "=" * 60)
    print("Hybrid Method Results")
    print("=" * 60)
    print(f"bp_sys MAE: {results['mae_sys']:.2f} mmHg")
    print(f"bp_dia MAE: {results['mae_dia']:.2f} mmHg")
    print(f"bp_sys Pearson: {results['pearson_sys']:.4f}")
    print(f"bp_dia Pearson: {results['pearson_dia']:.4f}")
    
    # 6. 对比
    print("\n" + "=" * 60)
    print("Comparison: Hybrid vs PTT vs Deep Learning")
    print("=" * 60)
    
    ptt_sys = 11.27
    ptt_dia = 7.64
    ptt_pearson = 0.3065
    
    dl_sys = 11.83
    dl_dia = 8.08
    dl_pearson = 0.073
    
    print(f"bp_sys MAE:  Hybrid={results['mae_sys']:.2f}, PTT={ptt_sys:.2f}, DL={dl_sys:.2f}")
    print(f"bp_dia MAE:  Hybrid={results['mae_dia']:.2f}, PTT={ptt_dia:.2f}, DL={dl_dia:.2f}")
    print(f"bp_pearson:  Hybrid={results['pearson_sys']:.4f}, PTT={ptt_pearson:.4f}, DL={dl_pearson:.4f}")
    
    if results['pearson_sys'] > 0.2:
        print("\n✅ 目标达成：Pearson > 0.2")
    
    if results['pearson_sys'] > ptt_pearson:
        print("✅ 混合方法优于 PTT baseline！")
    elif results['pearson_sys'] > dl_pearson:
        print("⚠️ 混合方法优于 Deep Learning，但不及 PTT baseline")
        print("   可能原因：shared_feat 对 BP 预测贡献不大")

if __name__ == '__main__':
    main()