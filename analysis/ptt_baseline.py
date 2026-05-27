#!/root/miniconda3/envs/f3mamba/bin/python
"""
P2: PTT-based Baseline for BP Prediction

PTT (Pulse Transit Time) 是经典的基于 rPPG 的血压预测方法。
核心思想：血压与脉搏波传输时间存在相关性。

方法：
1. 从 rPPG 信号提取 PTT 特征
2. 使用线性回归预测 BP

公式：BP = a * PTT + b (经典线性模型)
"""

import torch
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/root/zt/F3Mamba')

def load_all_subjects(data_root, max_subjects=50):
    """加载所有 subject 数据"""
    all_files = glob.glob(f"{data_root}/*_dataset_*.pth")
    all_files = all_files[:max_subjects]
    
    print(f"Loading {len(all_files)} subjects...")
    
    rppg_signals = []  # rPPG/bvp 信号
    bp_sys_labels = []
    bp_dia_labels = []
    hr_labels = []
    
    for path in tqdm(all_files):
        try:
            d = torch.load(path, weights_only=False)
            
            # 提取 rPPG 信号 (bvp)
            if "bvp" in d.data["labels"]:
                bvp = d.data["labels"]["bvp"]  # (N, L)
                rppg_signals.append(bvp.numpy())
            
            # 提取 BP 标签
            if "bp_sys" in d.data["labels"]:
                bp_sys = d.data["labels"]["bp_sys"]
                bp_sys_labels.append(bp_sys.mean(dim=-1).numpy())
            
            if "bp_dia" in d.data["labels"]:
                bp_dia = d.data["labels"]["bp_dia"]
                bp_dia_labels.append(bp_dia.mean(dim=-1).numpy())
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    # 合并
    rppg = np.concatenate(rppg_signals)  # (total_samples, L)
    bp_sys = np.concatenate(bp_sys_labels)  # (total_samples,)
    bp_dia = np.concatenate(bp_dia_labels)
    
    print(f"Total samples: {len(bp_sys)}")
    print(f"rPPG shape: {rppg.shape}")
    
    return rppg, bp_sys, bp_dia

def extract_ptt_features(rppg_signals, fs=30):
    """
    从 rPPG 信号提取 PTT 相关特征
    
    PTT 需要两个位置的信号差，但我们的数据只有单点 rPPG。
    因此我们提取脉搏波形态特征作为替代：
    
    Features:
    1. Pulse wave amplitude (脉搏波振幅)
    2. Pulse width (脉搏宽度)
    3. Heart rate (心率)
    4. Pulse wave variability (脉搏波变异度)
    """
    features = []
    
    for signal in tqdm(rppg_signals, desc="Extracting features"):
        # signal: (L,) 或 (chunk_size,)
        
        # 1. 基本统计特征
        mean_amp = np.mean(np.abs(signal))
        std_amp = np.std(signal)
        max_amp = np.max(signal)
        min_amp = np.min(signal)
        
        # 2. 频域特征 (FFT)
        fft = np.fft.rfft(signal)
        fft_mag = np.abs(fft)
        
        # 主频率 (心率相关)
        freq_idx = np.argmax(fft_mag[1:]) + 1  # 跳过 DC 分量
        main_freq = freq_idx * fs / len(signal)  # Hz
        hr = main_freq * 60  # BPM
        
        # 主频率能量
        main_energy = fft_mag[freq_idx]
        
        # 3. 波形形态特征
        # 寻找峰值
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(signal, height=np.mean(signal))
        
        if len(peaks) > 1:
            # 峰间隔 (心率)
            peak_intervals = np.diff(peaks) / fs * 1000  # ms
            mean_interval = np.mean(peak_intervals)
            std_interval = np.std(peak_intervals) if len(peak_intervals) > 1 else 0
            
            # 峰高度变异
            peak_heights = signal[peaks]
            mean_peak_height = np.mean(peak_heights)
            std_peak_height = np.std(peak_heights) if len(peak_heights) > 1 else 0
        else:
            mean_interval = 1000 / hr if hr > 0 else 0
            std_interval = 0
            mean_peak_height = max_amp
            std_peak_height = 0
        
        # 4. 组合特征
        feat = [
            mean_amp,
            std_amp,
            max_amp,
            min_amp,
            hr,
            main_energy,
            mean_interval,
            std_interval,
            mean_peak_height,
            std_peak_height,
        ]
        
        features.append(feat)
    
    return np.array(features)

def train_baseline(X, y_bp_sys, y_bp_dia):
    """训练线性回归 baseline"""
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练 bp_sys 模型
    model_sys = Ridge(alpha=1.0)
    model_sys.fit(X_scaled, y_bp_sys)
    
    # 训练 bp_dia 模型
    model_dia = Ridge(alpha=1.0)
    model_dia.fit(X_scaled, y_bp_dia)
    
    # 预测
    pred_sys = model_sys.predict(X_scaled)
    pred_dia = model_dia.predict(X_scaled)
    
    # 评估
    mae_sys = np.mean(np.abs(pred_sys - y_bp_sys))
    mae_dia = np.mean(np.abs(pred_dia - y_bp_dia))
    
    corr_sys = pearsonr(pred_sys, y_bp_sys)[0]
    corr_dia = pearsonr(pred_dia, y_bp_dia)[0]
    
    return {
        'mae_sys': mae_sys,
        'mae_dia': mae_dia,
        'pearson_sys': corr_sys,
        'pearson_dia': corr_dia,
        'model_sys': model_sys,
        'model_dia': model_dia,
        'scaler': scaler
    }

def main():
    print("=" * 60)
    print("P2: PTT-based Baseline for BP Prediction")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1] Loading data...")
    data_root = '/root/zt/F3Mamba/ProcessedDataset/AZDualCamera'
    rppg, bp_sys, bp_dia = load_all_subjects(data_root, max_subjects=80)
    
    # 2. 提取特征
    print("\n[2] Extracting PTT-related features...")
    X = extract_ptt_features(rppg)
    
    print(f"Features shape: {X.shape}")
    print(f"Feature names: amplitude, std, max, min, HR, energy, interval, interval_std, peak_height, peak_std")
    
    # 3. 训练 baseline
    print("\n[3] Training linear regression baseline...")
    results = train_baseline(X, bp_sys, bp_dia)
    
    # 4. 结果
    print("\n" + "=" * 60)
    print("PTT-based Baseline Results")
    print("=" * 60)
    print(f"bp_sys MAE: {results['mae_sys']:.2f} mmHg")
    print(f"bp_dia MAE: {results['mae_dia']:.2f} mmHg")
    print(f"bp_sys Pearson: {results['pearson_sys']:.4f}")
    print(f"bp_dia Pearson: {results['pearson_dia']:.4f}")
    
    # 5. 与 Deep Learning 对比
    print("\n" + "=" * 60)
    print("Comparison: PTT Baseline vs Deep Learning")
    print("=" * 60)
    
    dl_sys_mae = 11.83  # Deep Learning 最佳
    dl_dia_mae = 8.08
    dl_pearson = 0.073
    
    print(f"bp_sys MAE:  PTT={results['mae_sys']:.2f}, DL={dl_sys_mae:.2f}")
    print(f"bp_dia MAE:  PTT={results['mae_dia']:.2f}, DL={dl_dia_mae:.2f}")
    print(f"bp_pearson:  PTT={results['pearson_sys']:.4f}, DL={dl_pearson:.4f}")
    
    if results['pearson_sys'] > dl_pearson:
        print("\n✅ PTT Baseline Pearson 更高！")
        print("   说明：简单的形态特征比 Deep Learning 更能有效预测 BP")
    else:
        print("\n⚠️ PTT Baseline Pearson 仍然较低")
        print("   说明：数据本身可能不包含足够的 BP 相关信息")

if __name__ == '__main__':
    main()