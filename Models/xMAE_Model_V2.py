import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.InfinitePulse import Mamba2
from Models.FusionInfinitePulse import InfinitePulseFeatureExtractor


class CurriculumContinuousMask(nn.Module):
    """Continuous temporal masking with curriculum ratio updates."""
    def __init__(self, initial_ratio=0.8, max_ratio=0.9, step=0.05):
        super().__init__()
        self.initial_ratio = initial_ratio
        self.max_ratio = max_ratio
        self.step = step
        self.current_ratio = initial_ratio

    def step_curriculum(self, loss_improved=False):
        if loss_improved and self.current_ratio < self.max_ratio:
            self.current_ratio = min(self.max_ratio, self.current_ratio + self.step)
            return True
        return False

    def forward(self, x):
        """
        Args:
            x: (B, C, L, H, W)
        Returns:
            masked_x: (B, C, L, H, W)
            mask: (B, L) True where masked
        """
        B, _, L, _, _ = x.shape
        device = x.device

        mask_len = max(1, int(L * self.current_ratio))
        mask_len = min(mask_len, L)
        start = torch.randint(0, L - mask_len + 1, (B,), device=device)

        mask = torch.zeros(B, L, device=device, dtype=torch.bool)
        for b in range(B):
            mask[b, start[b]:start[b] + mask_len] = True

        mask_4d = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        masked_x = x.clone() * (~mask_4d).float()

        return masked_x, mask


class DirectionalCrossAttention(nn.Module):
    """Directional attention with Q=masked, K/V=visible."""
    def __init__(self, feat_dim=32, hidden_dim=128, num_heads=4):
        super().__init__()

        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            batch_first=True, dropout=0.1
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, feat_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, visible_feat, masked_feat):
        """
        Args:
            visible_feat: (B, L, D)
            masked_feat: (B, L, D)
        Returns:
            reconstructed: (B, L, D)
        """
        v = self.input_proj(visible_feat)
        m = self.input_proj(masked_feat)

        attn_out, _ = self.cross_attn(m, v, v)
        m = self.norm(m + attn_out)
        reconstructed = self.decoder(m)

        return reconstructed


class xMAEBPModelV2(nn.Module):
    """xMAE V2: BP-first model with BVP auxiliary and split BP heads."""
    def __init__(self, n_layers=None, filters=None, chunk_size=160,
                 initial_mask_ratio=0.8, max_mask_ratio=0.9,
                 shared_encoder=False,
                 bp_hidden_dim=64, bp_dropout=0.2,
                 align_hook=None,
                 quality_fusion=False, quality_temp=1.0):
        super().__init__()

        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]

        self.chunk_size = chunk_size
        feat_dim = filters[-1]

        self.encoder_front = InfinitePulseFeatureExtractor(
            n_layers, filters, chunk_size, in_channels=3
        )
        if shared_encoder:
            self.encoder_back = self.encoder_front
        else:
            self.encoder_back = InfinitePulseFeatureExtractor(
                n_layers, filters, chunk_size, in_channels=3
            )

        self.masker = CurriculumContinuousMask(
            initial_ratio=initial_mask_ratio,
            max_ratio=max_mask_ratio
        )

        self.decoder = DirectionalCrossAttention(feat_dim=feat_dim)
        self.recon_weight = 0.1

        self.quality_fusion = quality_fusion
        self.quality_temp = quality_temp

        if quality_fusion:
            self.quality_front = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(feat_dim, max(feat_dim // 4, 4)),
                nn.ReLU(inplace=True),
                nn.Linear(max(feat_dim // 4, 4), 1),
                nn.Sigmoid(),
            )
            self.quality_back = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(feat_dim, max(feat_dim // 4, 4)),
                nn.ReLU(inplace=True),
                nn.Linear(max(feat_dim // 4, 4), 1),
                nn.Sigmoid(),
            )
        else:
            self.quality_front = None
            self.quality_back = None

        self.fusion = nn.Linear(feat_dim * 2, feat_dim)
        self.mamba = nn.Sequential(
            Mamba2(d_model=feat_dim, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
            Mamba2(d_model=feat_dim, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
        )

        self.rppg_head = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1),
        )

        self.bp_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, bp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(bp_dropout),
            nn.Linear(bp_hidden_dim, bp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(bp_dropout),
        )
        self.bp_sys_head = nn.Linear(bp_hidden_dim, 1)
        self.bp_dia_head = nn.Linear(bp_hidden_dim, 1)

        if align_hook is None:
            self.align_hook = self._identity_align_hook
        else:
            if not callable(align_hook):
                raise TypeError('align_hook must be callable')
            self.align_hook = align_hook

    def _identity_align_hook(self, fused_feat, front_feat, back_feat, mask=None):
        return fused_feat

    def forward(self, data_input):
        """Forward with NHWC video input and dual-view reconstruction path."""
        front_raw = data_input['front']
        back_raw = data_input['back']

        front = front_raw.permute(0, 4, 1, 2, 3).contiguous()
        back = back_raw.permute(0, 4, 1, 2, 3).contiguous()

        f_feat = self.encoder_front(front)
        b_feat = self.encoder_back(back)

        recon_loss = torch.tensor(0.0, device=f_feat.device)
        mask = None

        if self.training:
            back_masked, mask = self.masker(back)
            b_feat_masked = self.encoder_back(back_masked)

            back_reconstructed = self.decoder(f_feat, b_feat_masked)

            if mask.any():
                recon_loss = F.mse_loss(
                    back_reconstructed[mask], b_feat[mask]
                )
            else:
                recon_loss = torch.tensor(0.0, device=f_feat.device, dtype=f_feat.dtype)

            b_feat_fused = b_feat_masked.clone()
            b_feat_fused[mask] = back_reconstructed[mask]
        else:
            b_feat_fused = b_feat

        if self.quality_fusion:
            q_front = self.quality_front(f_feat.transpose(1, 2))
            q_back = self.quality_back(b_feat_fused.transpose(1, 2))
            q = torch.cat([q_front, q_back], dim=-1)
            q = F.softmax(q / self.quality_temp, dim=-1)
            f_feat_w = f_feat * q[:, 0:1, None]
            b_feat_w = b_feat_fused * q[:, 1:2, None]
            x = torch.cat([f_feat_w, b_feat_w], dim=-1)
        else:
            x = torch.cat([f_feat, b_feat_fused], dim=-1)
        x = self.fusion(x)
        x = self.align_hook(x, f_feat, b_feat_fused, mask)
        if not torch.is_tensor(x):
            raise TypeError('align_hook must return a torch.Tensor')
        if x.shape != f_feat.shape:
            raise ValueError(f'align_hook must return shape {tuple(f_feat.shape)}, got {tuple(x.shape)}')
        if x.device != f_feat.device:
            raise ValueError('align_hook returned tensor on different device')
        if x.dtype != f_feat.dtype:
            raise TypeError('align_hook returned tensor with different dtype')
        x = self.mamba(x)

        rPPG = self.rppg_head(x).squeeze(-1)

        bp_shared = self.bp_pool(x.transpose(1, 2))
        bp_sys = self.bp_sys_head(bp_shared).squeeze(-1)
        bp_dia = self.bp_dia_head(bp_shared).squeeze(-1)

        return {
            'rPPG': rPPG,
            'bp_sys': bp_sys,
            'bp_dia': bp_dia,
            'recon_loss': recon_loss,
        }

    def step_curriculum(self, loss_improved=False):
        return self.masker.step_curriculum(loss_improved)
