"""
Enhanced BP Head v2 - Improved architecture for BP estimation.

Key improvements over original:
1. Multi-scale temporal features (kernel 3/7/15)
2. Temporal attention for key waveform segments
3. Deep MLP (4 layers) for stronger representation
4. Additional physiologically-inspired features

Author: Auto-generated
Date: 2026-05-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Optional

from Models.InfinitePulse import (
    InfinitePulse, Mamba2, RMSNorm
)
from Models.FusionInfinitePulse import InfinitePulseFeatureExtractor


def pearson_correlation(x: Tensor, y: Tensor) -> Tensor:
    """Compute Pearson correlation coefficient between two vectors.
    
    Args:
        x: (B,) predictions
        y: (B,) labels
    
    Returns:
        Pearson r: scalar tensor
    """
    x_mean = x.mean()
    y_mean = y.mean()
    x_std = x.std() + 1e-8
    y_std = y.std() + 1e-8
    corr = ((x - x_mean) * (y - y_mean)).mean() / (x_std * y_std)
    return corr


class NegPearsonLoss(nn.Module):
    """Negative Pearson loss for BP prediction.
    
    Returns -pearson to maximize correlation during training.
    Range: [-1, 0] where 0 = perfect correlation.
    """
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return -pearson_correlation(pred, target)


class TemporalAttention(nn.Module):
    """Attention mechanism for temporal features.
    
    Helps focus on key waveform segments (systolic peak, diastolic trough).
    """
    def __init__(self, d_model: int = 32, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, 32)
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out


class MultiScaleTemporalExtractor(nn.Module):
    """Multi-scale temporal feature extraction.
    
    Different kernel sizes capture:
    - k=3: Fast variations (dP/dt)
    - k=7: Waveform morphology
    - k=15: Overall pattern (diastolic decay)
    """
    def __init__(self, in_channels: int = 32, out_channels: int = 48):
        super().__init__()
        
        # Multi-scale convolutions
        self.conv_small = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_medium = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 3, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )
        self.conv_large = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 3, kernel_size=15, padding=7),
            nn.ReLU(inplace=True),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling options
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 32, L)
        small = self.conv_small(x)
        medium = self.conv_medium(x)
        large = self.conv_large(x)
        
        # Concatenate multi-scale features
        multi = torch.cat([small, medium, large], dim=1)
        fused = self.fusion(multi)
        
        # Dual pooling for richer representation
        avg = self.avg_pool(fused).flatten(1)
        max = self.max_pool(fused).flatten(1)
        
        return torch.cat([avg, max], dim=1)  # (B, 96)


class EnhancedBPHead(nn.Module):
    """Enhanced BP head with attention and deep MLP."""
    
    def __init__(self, shared_feat_dim: int = 32, chunk_size: int = 160):
        super().__init__()
        
        self.chunk_size = chunk_size
        
        # 1. Temporal attention for key segments
        self.temporal_attn = TemporalAttention(d_model=32, n_heads=4)
        
        # 2. Multi-scale temporal features
        self.temporal_extractor = MultiScaleTemporalExtractor(32, 48)
        
        # 3. Frequency branch (enhanced)
        self.freq_conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.freq_conv2 = nn.Conv1d(64, 48, kernel_size=3, padding=1)
        self.freq_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4. Waveform statistics (mean, std, skewness, kurtosis)
        self.stat_fc = nn.Linear(4, 16)
        
        # 5. Deep MLP (4 layers)
        # Total features: 96 (temporal) + 48 (freq) + 16 (stats) + 32 (attn) = 192
        self.bp_fc = nn.Sequential(
            nn.Linear(96 + 48 + 16 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )
        
    def forward(self, shared_feat: Tensor) -> Tensor:
        # shared_feat: (B, L, 32)
        B, L, D = shared_feat.shape
        
        # 1. Temporal attention
        attn_feat = self.temporal_attn(shared_feat)  # (B, L, 32)
        attn_pool = attn_feat.mean(dim=1)  # (B, 32)
        
        # 2. Multi-scale temporal features
        z = shared_feat.transpose(1, 2)  # (B, 32, L)
        temporal_feat = self.temporal_extractor(z)  # (B, 96)
        
        # 3. Frequency features
        f = torch.fft.rfft(shared_feat, n=256, dim=1)
        f = f.transpose(1, 2)  # (B, 32, 129)
        f = torch.log(torch.abs(f) + 1e-8)
        f = F.relu(self.freq_conv1(f))
        f = F.relu(self.freq_conv2(f))
        freq_feat = self.freq_pool(f).flatten(1)  # (B, 48)
        
        # 4. Waveform statistics
        rppg = shared_feat.mean(dim=-1)  # (B, L)
        mean = rppg.mean(dim=1)  # (B,)
        std = rppg.std(dim=1) + 1e-8  # (B,)
        # Skewness and kurtosis per sample
        rppg_centered = rppg - mean.unsqueeze(1)  # (B, L)
        skew = (rppg_centered ** 3).mean(dim=1) / (std ** 3)  # (B,)
        kurt = (rppg_centered ** 4).mean(dim=1) / (std ** 4)  # (B,)
        stats = torch.stack([mean, std, skew, kurt], dim=1)  # (B, 4)
        stat_feat = F.relu(self.stat_fc(stats))  # (B, 16)
        
        # 5. Concatenate all features
        all_feat = torch.cat([temporal_feat, freq_feat, stat_feat, attn_pool], dim=1)
        
        # 6. Deep MLP
        bp_out = self.bp_fc(all_feat)  # (B, 2)
        bp_sys, bp_dia = bp_out.unbind(1)
        
        return bp_sys, bp_dia


class LateFusionEnhancedBPv2(nn.Module):
    """Late fusion with enhanced BP head.
    
    Architecture:
        dual backbone → fusion → Mamba2×2 → shared_feat
                                            ├── head_out → rPPG
                                            └── enhanced_bp_head → BP
    
    Key improvements:
        1. Multi-scale temporal features
        2. Temporal attention
        3. Deep 4-layer MLP
        4. Waveform statistics
    """
    
    def __init__(self, n_layers=None, filters=None, chunk_size: int = 160,
                 shared_backbone: bool = True, fusion_mode: str = 'sums'):
        super().__init__()
        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]
        
        self.chunk_size = chunk_size
        self.shared_backbone = shared_backbone
        self.fusion_mode = fusion_mode
        feat_dim = filters[-1]
        
        # Backbone (same as original)
        self.backbone_front = InfinitePulseFeatureExtractor(
            n_layers, filters, chunk_size, in_channels=3
        )
        if shared_backbone:
            self.backbone_back = self.backbone_front
        else:
            self.backbone_back = InfinitePulseFeatureExtractor(
                n_layers, filters, chunk_size, in_channels=3
            )
        
        # Fusion (same as original)
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
        
        # Shared feature processing
        self.out_proj = nn.Sequential(
            nn.Linear(feat_dim, 32, bias=False),
            nn.Dropout(0.5),
        )
        self.head = nn.Sequential(
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
        )
        
        # rPPG output
        self.head_out = nn.Linear(32, 1)
        
        # Enhanced BP head
        self.bp_head = EnhancedBPHead(shared_feat_dim=32, chunk_size=chunk_size)
        
    def backbone_params(self):
        """Parameters NOT in BP head."""
        bp_set = set(self.bp_parameters())
        return [p for p in self.parameters() if p not in bp_set]
    
    def bp_parameters(self):
        """Parameters in BP head only."""
        return list(self.bp_head.parameters())
    
    def forward(self, data_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        front = data_input["front"]
        back = data_input["back"]
        
        front = front.permute(0, 4, 1, 2, 3).contiguous()
        back = back.permute(0, 4, 1, 2, 3).contiguous()
        
        f_feat = self.backbone_front(front)
        del front
        b_feat = self.backbone_back(back)
        del back
        
        if self.fusion_mode == 'concat':
            x = torch.cat([f_feat, b_feat], dim=-1)
            x = self.fusion_proj(x)
        elif self.fusion_mode == 'sums':
            x = (f_feat + b_feat) / 2
            x = self.fusion_proj(x)
        elif self.fusion_mode == 'conv':
            x = torch.cat([f_feat, b_feat], dim=-1)
            x = x.transpose(1, 2)
            x = self.fusion_proj(x)
            x = x.transpose(1, 2)
        del f_feat, b_feat
        
        x = self.out_proj(x)
        shared_feat = self.head(x)  # (B, L, 32)
        
        # rPPG output
        rPPG = self.head_out(shared_feat).squeeze(-1)
        
        # Enhanced BP prediction
        bp_sys, bp_dia = self.bp_head(shared_feat)
        
        return {"rPPG": rPPG, "bp_sys": bp_sys, "bp_dia": bp_dia}