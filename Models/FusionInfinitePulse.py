"""
Fusion models based on InfinitePulse for dual-camera (front + back) rPPG estimation.

Two strategies:
1. EarlyFusionInfinitePulse: Channel-level concatenation → single backbone
2. LateFusionInfinitePulse: Siamese backbones → feature-level concatenation + fusion head

Input: dict {"front": (B, L, H, W, 3), "back": (B, L, H, W, 3)}  [NHWC]
Output: dict {"rPPG": (B, L)}  BVP waveform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict

from Models.InfinitePulse import (
    InfinitePulse, SSCBlock, Mamba2, RMSNorm
)


class EarlyFusionInfinitePulse(nn.Module):
    """Early fusion: concatenate front/back along channel dim, feed through a single InfinitePulse.
    
    Input:  {"front": (B, L, H, W, 3), "back": (B, L, H, W, 3)}  NHWC
    Output: {"rPPG": (B, L)}
    """
    def __init__(self, n_layers: List[int] = None, filters: List[int] = None,
                 chunk_size: int = 160):
        super().__init__()
        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]

        self.chunk_size = chunk_size
        self.model = InfinitePulse(
            n_layers=n_layers,
            filters=filters,
            chunk_size=chunk_size,
            in_channels=6,          # 3 (front) + 3 (back)
            input_permute=True,     # NHWC → NCDHW internally
        )

    def forward(self, data_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        front = data_input["front"]   # (B, L, H, W, 3)
        back = data_input["back"]     # (B, L, H, W, 3)
        x = torch.cat([front, back], dim=-1)   # (B, L, H, W, 6)
        rPPG = self.model(x)                   # (B, L)
        return {"rPPG": rPPG}


class InfinitePulseFeatureExtractor(nn.Module):
    """SSCBlocks only — spatial feature extraction without output head.
    
    Input:  (B, C, L, H, W)  NCDHW
    Output: (B, L, filters[-1])  feature sequence
    """
    def __init__(self, n_layers: List[int], filters: List[int],
                 chunk_size: int = 160, in_channels: int = 3):
        super().__init__()
        self.chunk_size = chunk_size
        self.filters = filters

        blocks = []
        in_ch = in_channels
        for l, f in zip(n_layers, filters):
            blocks.append(SSCBlock(in_ch, l, f, chunk=chunk_size, downsample=True))
            in_ch = f
        self.network = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, L, H, W) → (B, L, filters[-1])"""
        for block in self.network:
            x = block(x)
        x = x.mean(dim=(-2, -1)).transpose(1, 2)   # (B, L, C)
        return x


class LateFusionInfinitePulse(nn.Module):
    """Late fusion: Siamese backbone extracts features from each view, then fuses at feature level.
    
    Input:  {"front": (B, L, H, W, 3), "back": (B, L, H, W, 3)}  NHWC
    Output: {"rPPG": (B, L)}
    """
    def __init__(self, n_layers: List[int] = None, filters: List[int] = None,
                 chunk_size: int = 160, shared_backbone: bool = True):
        super().__init__()
        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]

        self.chunk_size = chunk_size
        self.shared_backbone = shared_backbone

        # Siamese backbones
        self.backbone_front = InfinitePulseFeatureExtractor(
            n_layers, filters, chunk_size, in_channels=3
        )
        if shared_backbone:
            self.backbone_back = self.backbone_front
        else:
            self.backbone_back = InfinitePulseFeatureExtractor(
                n_layers, filters, chunk_size, in_channels=3
            )

        feat_dim = filters[-1]
        self.fusion_proj = nn.Linear(feat_dim * 2, feat_dim, bias=False)

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
        self.head_out = nn.Linear(32, 1)

    def forward(self, data_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        front = data_input["front"]   # (B, L, H, W, 3)  NHWC
        back = data_input["back"]     # (B, L, H, W, 3)  NHWC

        # NHWC → NCDHW
        front = front.permute(0, 4, 1, 2, 3).contiguous()   # (B, 3, L, H, W)
        back = back.permute(0, 4, 1, 2, 3).contiguous()

        # Feature extraction
        f_feat = self.backbone_front(front)   # (B, L, C)
        b_feat = self.backbone_back(back)     # (B, L, C)

        # Late fusion: concatenate → project
        x = torch.cat([f_feat, b_feat], dim=-1)   # (B, L, 2C)
        x = self.fusion_proj(x)                    # (B, L, C)
        x = self.out_proj(x)                       # (B, L, 32)
        x = self.head(x)                           # (B, L, 32)
        x = self.head_out(x)                       # (B, L, 1)
        rPPG = x.squeeze(-1)                       # (B, L)
        return {"rPPG": rPPG}


class LateFusionInfinitePulseMultitask(nn.Module):
    """Late fusion multi-task with configurable fusion modes.

    Fusion modes:
      - 'concat' (default): concat(f_feat, b_feat) → Linear(2C, C)
      - 'sums':   (f_feat + b_feat) / 2 → Linear(C, C)  [additive, from F3Mamba]
      - 'conv':   stack(f_feat, b_feat) → Conv1d(2,1,k=3) → squeeze  [learned fusion kernel]

    Input:  {"front": (B, L, H, W, 3), "back": (B, L, H, W, 3)}  NHWC
    Output: dict with rPPG, bp_sys, bp_dia
    """
    def __init__(self, n_layers=None, filters=None, chunk_size: int = 160,
                 shared_backbone: bool = True, fusion_mode: str = 'concat'):
        super().__init__()
        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]

        self.chunk_size = chunk_size
        self.shared_backbone = shared_backbone
        self.fusion_mode = fusion_mode
        feat_dim = filters[-1]

        self.backbone_front = InfinitePulseFeatureExtractor(
            n_layers, filters, chunk_size, in_channels=3
        )
        if shared_backbone:
            self.backbone_back = self.backbone_front
        else:
            self.backbone_back = InfinitePulseFeatureExtractor(
                n_layers, filters, chunk_size, in_channels=3
            )

        # Fusion projection (mode-dependent)
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
        self.head = nn.Sequential(
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
        )
        self.head_out = nn.Linear(32, 1)

        # Aux head: rPPG → BP
        self.aux_head = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(8, 2),
        )

    def forward(self, data_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        front = data_input["front"]
        back = data_input["back"]

        front = front.permute(0, 4, 1, 2, 3).contiguous()
        back = back.permute(0, 4, 1, 2, 3).contiguous()

        f_feat = self.backbone_front(front)   # (B, L, C)
        del front
        b_feat = self.backbone_back(back)     # (B, L, C)
        del back

        if self.fusion_mode == 'concat':
            x = torch.cat([f_feat, b_feat], dim=-1)        # (B, L, 2C)
            x = self.fusion_proj(x)                        # (B, L, C)
        elif self.fusion_mode == 'sums':
            x = (f_feat + b_feat) / 2                      # (B, L, C)
            x = self.fusion_proj(x)
        elif self.fusion_mode == 'conv':
            x = torch.cat([f_feat, b_feat], dim=-1)       # (B, L, 2C)
            x = x.transpose(1, 2)                          # (B, 2C, L)
            x = self.fusion_proj(x)                        # (B, C, L)
            x = x.transpose(1, 2)                          # (B, L, C)
        del f_feat, b_feat

        x = self.out_proj(x)
        x = self.head(x)
        rPPG = self.head_out(x).squeeze(-1)

        aux_in = rPPG.unsqueeze(1)
        aux = self.aux_head(aux_in)
        bp_sys, bp_dia = aux[:, 0], aux[:, 1]

        return {"rPPG": rPPG, "bp_sys": bp_sys, "bp_dia": bp_dia}


# ---------------------------------------------------------------------------
# LateFusionInfinitePulseEnhancedBP  —  BP head on shared_feat, same level as rPPG
# ---------------------------------------------------------------------------
class LateFusionInfinitePulseEnhancedBP(nn.Module):
    """Late fusion with BP head on shared features (peer to rPPG head).

    BP head takes the same 32-channel shared_feat as rPPG head, instead of
    the 1-channel rPPG waveform. This decouples BP from rPPG reconstruction
    quality and provides 32× richer temporal features.

    Architecture:
      dual backbone (F/B) → fusion → out_proj → Mamba2×2 → shared_feat
                                                              ├── head_out → rPPG
                                                              └── bp_head → BP

    Fusion modes: 'concat', 'sums', 'conv'  (same as Multitask)
    """
    def __init__(self, n_layers=None, filters=None, chunk_size: int = 160,
                 shared_backbone: bool = True, fusion_mode: str = 'concat'):
        super().__init__()
        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]

        self.chunk_size = chunk_size
        self.shared_backbone = shared_backbone
        self.fusion_mode = fusion_mode
        feat_dim = filters[-1]

        self.backbone_front = InfinitePulseFeatureExtractor(
            n_layers, filters, chunk_size, in_channels=3
        )
        if shared_backbone:
            self.backbone_back = self.backbone_front
        else:
            self.backbone_back = InfinitePulseFeatureExtractor(
                n_layers, filters, chunk_size, in_channels=3
            )

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
        self.head = nn.Sequential(
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
        )
        self.head_out = nn.Linear(32, 1)

        # Time branch: dilated convs on shared_feat
        self.t_conv = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(48, 48, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.t_pool = nn.AdaptiveAvgPool1d(1)

        # Frequency branch: log-magnitude spectrum
        self.fconv1 = nn.Conv1d(32, 48, kernel_size=3, padding=1)
        self.fconv2 = nn.Conv1d(48, 32, kernel_size=3, padding=1)
        self.f_pool = nn.AdaptiveAvgPool1d(1)

        self.bp_fc = nn.Sequential(
            nn.Linear(48 + 32, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(48, 2),
        )

    def backbone_params(self):
        """Parameters NOT in BP head (feature extractors + rPPG head)."""
        bp_set = set(self.bp_parameters())
        return [p for p in self.parameters() if p not in bp_set]

    def bp_parameters(self):
        """Parameters in BP head only."""
        return list(self.t_conv.parameters()) + list(self.t_pool.parameters()) + \
               list(self.fconv1.parameters()) + list(self.fconv2.parameters()) + \
               list(self.f_pool.parameters()) + list(self.bp_fc.parameters())

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
        shared_feat = self.head(x)                 # (B, L, 32)
        rPPG = self.head_out(shared_feat).squeeze(-1)

        # Time branch
        z = shared_feat.transpose(1, 2)            # (B, 32, L)
        t_feat = self.t_pool(self.t_conv(z)).flatten(1)  # (B, 48)

        # Frequency branch: log-magnitude of rFFT
        n_fft = 256
        f = torch.fft.rfft(shared_feat, n=n_fft, dim=1)  # (B, 129, 32)
        f = f.transpose(1, 2)                              # (B, 32, 129)
        f = torch.log(torch.abs(f) + 1e-8)
        f = F.relu(self.fconv1(f))
        f = F.relu(self.fconv2(f))
        f_feat = self.f_pool(f).flatten(1)                  # (B, 32)

        fused = torch.cat([t_feat, f_feat], dim=1)
        bp_sys, bp_dia = self.bp_fc(fused).unbind(1)

        return {"rPPG": rPPG, "bp_sys": bp_sys, "bp_dia": bp_dia}


# ---------------------------------------------------------------------------
# EarlyFusionInfinitePulseMultitask  —  fusion + rPPG + SpO2 + BP
# ---------------------------------------------------------------------------
class EarlyFusionInfinitePulseMultitask(nn.Module):
    """Early fusion multi-task: front+back → 8ch → InfinitePulseMultitask backbone.

    Input:  {"front": (B, L, H, W, 3), "back": (B, L, H, W, 3)}  NHWC
    Output: dict with rPPG, bp_sys, bp_dia (same signatures as InfinitePulseMultitask)
    """
    def __init__(self, n_layers=None, filters=None, chunk_size: int = 160):
        super().__init__()
        from Models.InfinitePulse import InfinitePulseMultitask
        self.model = InfinitePulseMultitask(
            n_layers=n_layers, filters=filters, chunk_size=chunk_size,
            in_channels=6,   # 3 (front) + 3 (back)
            input_permute=True,
        )

    def forward(self, data_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        front = data_input["front"]                     # (B, L, H, W, 3)
        back = data_input["back"]                       # (B, L, H, W, 3)
        x = torch.cat([front, back], dim=-1)            # (B, L, H, W, 6)
        return self.model(x)


# ---------------------------------------------------------------------------
# EarlyFusionInfinitePulseEnhancedBP  —  fusion + rPPG + BP (improved head)
# ---------------------------------------------------------------------------
class EarlyFusionInfinitePulseEnhancedBP(nn.Module):
    """Early fusion + EnhancedBP head.

    Input:  {"front": (B, L, H, W, 3), "back": (B, L, H, W, 3)}  NHWC
    Output: dict with rPPG, bp_sys, bp_dia
    """
    def __init__(self, n_layers=None, filters=None, chunk_size: int = 160):
        super().__init__()
        from Models.InfinitePulse import InfinitePulseEnhancedBP
        self.model = InfinitePulseEnhancedBP(
            n_layers=n_layers, filters=filters, chunk_size=chunk_size,
            in_channels=6,   # 3 (front) + 3 (back)
            input_permute=True,
        )

    def forward(self, data_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        front = data_input["front"]
        back = data_input["back"]
        x = torch.cat([front, back], dim=-1)
        return self.model(x)


# ---------------------------------------------------------------------------
# LateFusionInfinitePulseEnhancedBPDemo — dual-view + rPPG + BP + demographics
# ---------------------------------------------------------------------------
class LateFusionInfinitePulseEnhancedBPDemo(nn.Module):
    """Enhanced BP head with demographic feature fusion (age + gender).

    Same as EnhancedBP, but the BP head additionally fuses:
      - Age (scalar, normalised)
      - Gender (one-hot: female, male)
    These provide a prior that helps constrain BP prediction range,
    following works like Schrumpf et al. "Personalized BP Estimation".

    Input now expects data_input with optional 'demo' tensor (B, 3).
    If not provided, uses dataset defaults (age=40, gender=female).
    """

    def __init__(self, n_layers=None, filters=None, chunk_size: int = 160,
                 shared_backbone: bool = True, fusion_mode: str = 'sums',
                 demo_dim: int = 3):
        super().__init__()
        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]

        self.chunk_size = chunk_size
        self.shared_backbone = shared_backbone
        self.fusion_mode = fusion_mode
        self.demo_dim = demo_dim
        feat_dim = filters[-1]

        self.backbone_front = InfinitePulseFeatureExtractor(
            n_layers, filters, chunk_size, in_channels=3)
        if shared_backbone:
            self.backbone_back = self.backbone_front
        else:
            self.backbone_back = InfinitePulseFeatureExtractor(
                n_layers, filters, chunk_size, in_channels=3)

        if fusion_mode == 'concat':
            self.fusion_proj = nn.Linear(feat_dim * 2, feat_dim, bias=False)
        elif fusion_mode == 'sums':
            self.fusion_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        elif fusion_mode == 'conv':
            self.fusion_proj = nn.Sequential(
                nn.Conv1d(feat_dim * 2, feat_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True))

        self.out_proj = nn.Sequential(nn.Linear(feat_dim, 32, bias=False), nn.Dropout(0.5))
        self.head = nn.Sequential(
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32))
        self.head_out = nn.Linear(32, 1)

        # Time branch
        self.t_conv = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(48, 48, kernel_size=5, padding=4, dilation=2), nn.ReLU(inplace=True))
        self.t_pool = nn.AdaptiveAvgPool1d(1)

        # Frequency branch
        self.fconv1 = nn.Conv1d(32, 48, kernel_size=3, padding=1)
        self.fconv2 = nn.Conv1d(48, 32, kernel_size=3, padding=1)
        self.f_pool = nn.AdaptiveAvgPool1d(1)

        # BP head with demographic fusion: (48+32+3) → 48 → 2
        self.bp_fc = nn.Sequential(
            nn.Linear(48 + 32 + demo_dim, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(48, 2))

    def backbone_params(self):
        bp_set = set(self.bp_parameters())
        return [p for p in self.parameters() if p not in bp_set]

    def bp_parameters(self):
        return (list(self.t_conv.parameters()) + list(self.t_pool.parameters()) +
                list(self.fconv1.parameters()) + list(self.fconv2.parameters()) +
                list(self.f_pool.parameters()) + list(self.bp_fc.parameters()))

    def forward(self, data_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        front = data_input["front"]
        back = data_input["back"]
        demo = data_input.get("demo", None)  # (B, 3) or None

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
        shared_feat = self.head(x)
        rPPG = self.head_out(shared_feat).squeeze(-1)

        # Time branch
        z = shared_feat.transpose(1, 2)
        t_feat = self.t_pool(self.t_conv(z)).flatten(1)

        # Frequency branch
        n_fft = 256
        f = torch.fft.rfft(shared_feat, n=n_fft, dim=1)
        f = f.transpose(1, 2)
        f = torch.log(torch.abs(f) + 1e-8)
        f = F.relu(self.fconv1(f))
        f = F.relu(self.fconv2(f))
        f_feat = self.f_pool(f).flatten(1)

        fused = torch.cat([t_feat, f_feat], dim=1)  # (B, 48+32)

        # Fuse demographics
        if demo is None:
            B = fused.shape[0]
            demo = torch.full((B, self.demo_dim), 0.0, device=fused.device,
                              dtype=fused.dtype)
        else:
            demo = demo.to(fused.device, dtype=fused.dtype)
        fused = torch.cat([fused, demo], dim=1)  # (B, 48+32+3)

        bp_sys, bp_dia = self.bp_fc(fused).unbind(1)
        return {"rPPG": rPPG, "bp_sys": bp_sys, "bp_dia": bp_dia}


# ---------------------------------------------------------------------------
# FreqAlignLateFusionMultitask  —  dual-view + freq-domain alignment + BP
# ---------------------------------------------------------------------------
class FreqAlignLateFusionMultitask(nn.Module):
    """Late fusion with frequency-domain alignment.

    Pipeline:
      1. Siamese backbone extracts f_feat, b_feat
      2. FFT of both → average magnitudes (reinforce shared HR signal)
      3. IFFT back → freq-aligned features
      4. Concat[original_f, original_b, aligned] → Linear → head

    Input:  {"front": (B, L, H, W, 3), "back": (B, L, H, W, 3)}  NHWC
    Output: dict with rPPG, bp_sys, bp_dia
    """
    def __init__(self, n_layers=None, filters=None, chunk_size: int = 160,
                 shared_backbone: bool = True):
        super().__init__()
        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]

        self.chunk_size = chunk_size
        self.shared_backbone = shared_backbone
        feat_dim = filters[-1]

        self.backbone_front = InfinitePulseFeatureExtractor(
            n_layers, filters, chunk_size, in_channels=3
        )
        if shared_backbone:
            self.backbone_back = self.backbone_front
        else:
            self.backbone_back = InfinitePulseFeatureExtractor(
                n_layers, filters, chunk_size, in_channels=3
            )

        # Fusion: original(f) + original(b) + freq_aligned
        self.fusion_proj = nn.Linear(feat_dim * 3, feat_dim, bias=False)

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
        self.head_out = nn.Linear(32, 1)

        self.aux_head = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(8, 2),
        )

    def _freq_align(self, a: Tensor, b: Tensor) -> Tensor:
        """Frequency-domain alignment: average magnitudes, keep phase from a."""
        n_fft = 256  # power-of-2 for cuFFT half-precision
        A = torch.fft.rfft(a, n=n_fft, dim=1)   # (B, 129, C)
        B = torch.fft.rfft(b, n=n_fft, dim=1)

        mag_a, pha_a = torch.abs(A), torch.angle(A)
        mag_b = torch.abs(B)

        mag_aligned = (mag_a + mag_b) / 2         # reinforce shared frequency peaks
        A_aligned = mag_aligned * torch.exp(1j * pha_a)

        out = torch.fft.irfft(A_aligned, n=n_fft, dim=1)  # (B, 256, C)
        return out[:, :a.shape[1], :]               # trim to original length

    def forward(self, data_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        front = data_input["front"]
        back = data_input["back"]

        front = front.permute(0, 4, 1, 2, 3).contiguous()
        back = back.permute(0, 4, 1, 2, 3).contiguous()

        f_feat = self.backbone_front(front)   # (B, L, C)
        b_feat = self.backbone_back(back)

        aligned = self._freq_align(f_feat, b_feat)   # (B, L, C)

        x = torch.cat([f_feat, b_feat, aligned], dim=-1)  # (B, L, 3C)
        x = self.fusion_proj(x)
        x = self.out_proj(x)
        x = self.head(x)
        rPPG = self.head_out(x).squeeze(-1)

        aux_in = rPPG.unsqueeze(1)
        aux = self.aux_head(aux_in)
        bp_sys, bp_dia = aux[:, 0], aux[:, 1]

        return {"rPPG": rPPG, "bp_sys": bp_sys, "bp_dia": bp_dia}


# ---------------------------------------------------------------------------
# LateFusionInfinitePulseClassifier  —  BP binary classification
# ---------------------------------------------------------------------------
class LateFusionInfinitePulseClassifier(nn.Module):
    """Late fusion with BP binary classification head.

    Same architecture as EnhancedBP, but BP head outputs 2-class logits
    for HTN (sys>=130 or dia>=80) vs Normal classification.

    Output: {"rPPG": rPPG, "bp_cls": (B, 2) logits}
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

        self.backbone_front = InfinitePulseFeatureExtractor(
            n_layers, filters, chunk_size, in_channels=3
        )
        if shared_backbone:
            self.backbone_back = self.backbone_front
        else:
            self.backbone_back = InfinitePulseFeatureExtractor(
                n_layers, filters, chunk_size, in_channels=3
            )

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
        self.head = nn.Sequential(
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
        )
        self.head_out = nn.Linear(32, 1)

        # Time branch
        self.t_conv = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(48, 48, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.t_pool = nn.AdaptiveAvgPool1d(1)

        # Frequency branch
        self.fconv1 = nn.Conv1d(32, 48, kernel_size=3, padding=1)
        self.fconv2 = nn.Conv1d(48, 32, kernel_size=3, padding=1)
        self.f_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head (2-class: Normal / HTN)
        self.cls_head = nn.Sequential(
            nn.Linear(48 + 32, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(48, 2),
        )

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
        shared_feat = self.head(x)                 # (B, L, 32)
        rPPG = self.head_out(shared_feat).squeeze(-1)

        # Time branch
        z = shared_feat.transpose(1, 2)
        t_feat = self.t_pool(self.t_conv(z)).flatten(1)

        # Frequency branch
        n_fft = 256
        f = torch.fft.rfft(shared_feat, n=n_fft, dim=1)
        f = f.transpose(1, 2)
        f = torch.log(torch.abs(f) + 1e-8)
        f = F.relu(self.fconv1(f))
        f = F.relu(self.fconv2(f))
        f_feat = self.f_pool(f).flatten(1)

        fused = torch.cat([t_feat, f_feat], dim=1)
        bp_cls = self.cls_head(fused)               # (B, 2) logits

        return {"rPPG": rPPG, "bp_cls": bp_cls}
