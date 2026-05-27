#!/root/miniconda3/envs/f3mamba/bin/python
"""
BP Head with PTT Feature Extraction

目标: Pearson > 0.3, MAE < 10 mmHg

架构:
    Backbone → shared_feat (B, L, 32)
                   → rPPG output (B, L)
                           → PTT Extractor → 10 dims
                   → shared_feat pool → Time/Freq Branch → 16 dims
                           → Concat (26 dims) → MLP → BP (sys, dia)

PTT Features (10 dims):
    - Amplitude: mean, std, max, min
    - Frequency: HR, main_energy
    - Morphology: interval_mean, interval_std, peak_height_mean, peak_height_std
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks


class PTTExtractor(nn.Module):
    """从 rPPG 信号提取 PTT 相关特征"""
    
    def __init__(self, fs=30):
        super().__init__()
        self.fs = fs
    
    def forward(self, rppg_signal):
        """
        Args:
            rppg_signal: (B, L) rPPG waveform
        
        Returns:
                ptt_features: (B, 10)
        """
        # Normalize input
        if rppg_signal.dim() == 3:
            rppg_signal = rppg_signal.unsqueeze(0)
        
        batch_size, seq_len = rppg_signal.shape
        
        features_list = []
        
        for b in range(batch_size):
            signal = rppg_signal[b].detach().cpu().numpy() if torch.is_tensor(rppg_signal) else rppg_signal[b]
            
            # 1. Amplitude features
            mean_amp = np.mean(np.abs(signal))
            std_amp = np.std(signal)
            max_amp = np.max(signal)
            min_amp = np.min(signal)
            
            # 2. Frequency features (FFT)
            fft = np.fft.rfft(signal)
            fft_mag = np.abs(fft)
            
            # Main frequency (HR)
            freq_idx = np.argmax(fft_mag[1:]) + 1
            main_freq = freq_idx * self.fs / seq_len
            hr = main_freq * 60
            
            # Main frequency energy
            main_energy = fft_mag[freq_idx]
            
            # 3. Morphology features (find peaks)
            try:
                peaks, _ = find_peaks(signal, height=np.mean(signal))
                
                if len(peaks) > 1:
                    # Peak intervals (ms)
                    peak_intervals = np.diff(peaks) / self.fs * 1000
                    mean_interval = np.mean(peak_intervals)
                    std_interval = np.std(peak_intervals) if len(peak_intervals) > 1 else 0
                    
                    # Peak heights
                    peak_heights = signal[peaks]
                    mean_peak_height = np.mean(peak_heights)
                    std_peak_height = np.std(peak_heights) if len(peak_heights) > 1 else 0
                else:
                    # Default values if no peaks found
                    mean_interval = 1000 / hr if hr > 0 else 0
                    std_interval = 0
                    mean_peak_height = max_amp
                    std_peak_height = 0
            except Exception:
                # Fallback for any errors in peak finding
                mean_interval = 1000 / hr if hr > 0 else 0
                std_interval = 0
                mean_peak_height = max_amp
                std_peak_height = 0
            
            # Combine all features
            feat = [
                mean_amp, std_amp, max_amp, min_amp,
                hr, main_energy,
                mean_interval, std_interval,
                mean_peak_height, std_peak_height
            ]
            features_list.append(feat)
        
        # Stack features
        features = np.stack(features_list)  # (B, 10)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float().to(rppg_signal.device)
        
        return features_tensor


class BPHeadPTTOnly(nn.Module):
    """PTT-only BP Head - uses only PTT features, ignores shared_feat
    
    This is based on the finding that shared_feat from backbone has 
    no correlation with BP (Pearson < 0.07), while PTT features have
    Pearson ~0.31 (from PTT Baseline).
    
    Architecture:
        rPPG (B, L) → PTT features (B, 10) → MLP → bp_sys, bp_dia
    """
    
    def __init__(self, ptt_dim=10, hidden_dim=64, dropout=0.2):
        super().__init__()
        
        self.ptt_extractor = PTTExtractor(fs=30)
        
        self.mlp = nn.Sequential(
            nn.Linear(ptt_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(self, shared_feat, rppg_output):
        """
        Args:
            shared_feat: (B, L, 32) - IGNORED
            rppg_output: (B, L) rPPG waveform
        
        Returns:
            bp_sys: (B,)
            bp_dia: (B,)
        """
        ptt_feat = self.ptt_extractor(rppg_output)  # (B, 10)
        bp_out = self.mlp(ptt_feat)
        bp_sys, bp_dia = bp_out.unbind(1)
        return bp_sys, bp_dia


class BPHeadWithPTT(nn.Module):
    """BP Head with PTT feature integration
    
    Architecture:
        shared_feat (B, L, 32) + rPPG (B, L) 
        → shared_feat pool → 16 dims
        → PTT features → 10 dims
        → concat → 26 dims → MLP → 2 outputs (bp_sys, bp_dia)
    """
    
    def __init__(self, shared_feat_dim=32, ptt_dim=10, hidden_dim=64, 
                 fusion_mode='concat', ptt_weight=1.0, dropout=0.2):
        super().__init__()
        
        self.shared_feat_dim = shared_feat_dim
        self.ptt_dim = ptt_dim
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.ptt_weight = ptt_weight
        self.dropout = dropout
        
        # PTT extractor
        self.ptt_extractor = PTTExtractor(fs=30)
        
        # Shared feat processing (from existing BP head)
        # Time branch: dilated convs on shared_feat
        self.t_conv = nn.Sequential(
            nn.Conv1d(shared_feat_dim, 24, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(24, 24, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.t_pool = nn.AdaptiveAvgPool1d(1)
        
        # Frequency branch
        self.fconv1 = nn.Conv1d(shared_feat_dim, 24, kernel_size=3, padding=1)
        self.fconv2 = nn.Conv1d(24, 16, kernel_size=3, padding=1)
        self.f_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fusion layer
        shared_dim = 40  # 24 (time) + 16 (freq)
        if fusion_mode == 'concat':
            input_dim = shared_dim + ptt_dim  # 50 dims
            self.fusion = nn.Identity()
        elif fusion_mode == 'weighted_sum':
            # Project ptt_feat to match shared_dim for weighted sum
            self.ptt_proj = nn.Linear(ptt_dim, shared_dim)
            self.shared_weight = nn.Parameter(torch.ones(1) * 1.0)
            self.ptt_weight_param = nn.Parameter(torch.ones(1) * ptt_weight)
            input_dim = shared_dim  # 40 dims
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(self, shared_feat, rppg_output):
        """
        Args:
            shared_feat: (B, L, 32)
            rppg_output: (B, L) rPPG waveform
        
        Returns:
            bp_sys: (B,)
            bp_dia: (B,)
        """
        B, L, D = shared_feat.shape
        
        # 1. Extract PTT features from rPPG output
        ptt_feat = self.ptt_extractor(rppg_output)  # (B, 10)
        
        # 2. Process shared_feat (existing BP head logic)
        # Time branch
        z = shared_feat.transpose(1, 2)  # (B, 32, L)
        t_feat = self.t_pool(self.t_conv(z)).flatten(1)  # (B, 24)
        
        # Frequency branch
        f = torch.fft.rfft(shared_feat, n=256, dim=1)
        f = f.transpose(1, 2)
        f = torch.log(torch.abs(f) + 1e-8)
        f = F.relu(self.fconv1(f))
        f = F.relu(self.fconv2(f))
        f_feat = self.f_pool(f).flatten(1)  # (B, 16)
        
        # Combine time and freq features
        shared_processed = torch.cat([t_feat[:, :24], f_feat], dim=1)  # (B, 40) - take first 24 dims from time
        
        # 3. Fusion
        if self.fusion_mode == 'concat':
            combined = torch.cat([shared_processed, ptt_feat], dim=1)  # (B, 50)
        else:  # weighted_sum
            # Project ptt_feat to match shared_dim
            ptt_proj = self.ptt_proj(ptt_feat)  # (B, 40)
            # Normalize features
            shared_norm = shared_processed / (shared_processed.abs().max(dim=1, keepdim=True).values + 1e-8)
            ptt_norm = ptt_proj / (ptt_proj.abs().max(dim=1, keepdim=True).values + 1e-8)
            combined = self.shared_weight * shared_norm + self.ptt_weight_param * ptt_norm
        
        # 4. MLP prediction
        bp_out = self.mlp(combined)
        bp_sys, bp_dia = bp_out.unbind(1)
        
        return bp_sys, bp_dia


class LateFusionBPHeadPTT(nn.Module):
    """Late Fusion model with new BP Head
    
    Complete model that integrates new BP Head into existing backbone.
    """
    
    def __init__(self, n_layers=None, filters=None, chunk_size=160,
                 shared_backbone=True, fusion_mode='sums',
                 bp_fusion_mode='concat', ptt_weight=1.0, 
                 bp_hidden_dim=64, bp_dropout=0.2,
                 bp_head_type='with_shared'):  # 'with_shared' or 'ptt_only'
        super().__init__()
        
        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]
        
        self.chunk_size = chunk_size
        self.shared_backbone = shared_backbone
        self.fusion_mode = fusion_mode
        self.bp_head_type = bp_head_type
        feat_dim = filters[-1]
        
        # Backbone (from LateFusionInfinitePulseEnhancedBP)
        from Models.FusionInfinitePulse import InfinitePulseFeatureExtractor
        
        self.backbone_front = InfinitePulseFeatureExtractor(
            n_layers, filters, chunk_size, in_channels=3
        )
        if shared_backbone:
            self.backbone_back = self.backbone_front
        else:
            self.backbone_back = InfinitePulseFeatureExtractor(
                n_layers, filters, chunk_size, in_channels=3
            )
        
        # Fusion (from original)
        if fusion_mode == 'concat':
            self.fusion_proj = nn.Linear(feat_dim * 2, feat_dim, bias=False)
        elif fusion_mode == 'sums':
            self.fusion_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        elif fusion_mode == 'conv':
            self.fusion_proj = nn.Sequential(
                nn.Conv1d(feat_dim * 2, feat_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        
        self.out_proj = nn.Sequential(
            nn.Linear(feat_dim, 32, bias=False),
            nn.Dropout(0.5),
        )
        
        # Mamba2 layers (from original)
        from Models.InfinitePulse import Mamba2
        self.head = nn.Sequential(
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
        )
        
        # rPPG output (from original)
        self.head_out = nn.Linear(32, 1)
        
        # New BP Head - choose based on bp_head_type
        if bp_head_type == 'ptt_only':
            # PTT-only: ignores shared_feat, uses only PTT features
            self.bp_head = BPHeadPTTOnly(
                ptt_dim=10,
                hidden_dim=bp_hidden_dim,
                dropout=bp_dropout
            )
        else:
            # Default: fusion of shared_feat + PTT features
            self.bp_head = BPHeadWithPTT(
                shared_feat_dim=32,
                ptt_dim=10,
                hidden_dim=bp_hidden_dim,
                fusion_mode=bp_fusion_mode,
                ptt_weight=ptt_weight,
                dropout=bp_dropout
            )
    
    def forward(self, data_input):
        """
        Args:
            data_input: dict with 'front' and 'back' keys
                front: (B, L, H, W, 3) - front camera video
                back: (B, L, H, W, 3) - back camera video
        
        Returns:
            dict with:
                rPPG: (B, L) - predicted rPPG signal
                bp_sys: (B,) - predicted systolic BP
                bp_dia: (B,) - predicted diastolic BP
        """
        front = data_input['front']
        back = data_input['back']
        
        # NHWC → NCDHW (for backbone)
        front = front.permute(0, 4, 1, 2, 3).contiguous()  # (B, 3, L, H, W)
        back = back.permute(0, 4, 1, 2, 3).contiguous()
        
        # Backbone feature extraction
        f_feat = self.backbone_front(front)  # (B, L, feat_dim)
        b_feat = self.backbone_back(back)  # (B, L, feat_dim)
        
        # Fusion
        if self.fusion_mode == 'concat':
            x = torch.cat([f_feat, b_feat], dim=-1)
        elif self.fusion_mode == 'sums':
            x = f_feat + b_feat
        elif self.fusion_mode == 'conv':
            x = torch.cat([f_feat, b_feat], dim=-1)
            x = x.transpose(1, 2)
            x = self.fusion_proj(x)
            x = x.transpose(1, 2)
        
        del f_feat, b_feat
        
        # Out projection + Mamba2
        x = self.out_proj(x)
        shared_feat = self.head(x)  # (B, L, 32)
        
        # rPPG output
        rPPG = self.head_out(shared_feat).squeeze(-1)  # (B, L)
        
        # BP prediction with new BP Head
        bp_sys, bp_dia = self.bp_head(shared_feat, rPPG)
        
        return {
            'rPPG': rPPG,
            'bp_sys': bp_sys,
            'bp_dia': bp_dia
        }