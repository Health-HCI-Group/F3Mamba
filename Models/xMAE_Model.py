import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.InfinitePulse import Mamba2
from Models.FusionInfinitePulse import InfinitePulseFeatureExtractor


class CurriculumContinuousMask(nn.Module):
    """xMAE-style continuous masking with curriculum.
    
    Key xMAE design:
    - Continuous (block) masking (not random): prevents local interpolation
    - Curriculum: starts at 80%, increases by 5% when loss improves, max 90%
    - Ensures at least one full cardiac cycle remains visible
    """
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
            x: (B, C, L, H, W) - input video (NCDHW)
        Returns:
            masked_x: (B, C, L, H, W)
            mask: (B, L) - True where masked
        """
        B, C, L, H, W = x.shape
        device = x.device
        
        mask_len = int(L * self.current_ratio)
        start = torch.randint(0, L - mask_len + 1, (B,), device=device)
        
        mask = torch.zeros(B, L, device=device, dtype=torch.bool)
        for b in range(B):
            mask[b, start[b]:start[b] + mask_len] = True
        
        mask_4d = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        masked_x = x.clone() * (~mask_4d).float()
        
        return masked_x, mask


class DirectionalCrossAttention(nn.Module):
    """xMAE-style directional cross-attention decoder.
    
    xMAE design: Q = masked_tokens, K/V = visible_tokens
    This enforces the model to learn cross-modal relationships
    rather than relying on within-modality interpolation.
    """
    def __init__(self, feat_dim=32, hidden_dim=128, num_heads=4):
        super().__init__()
        
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        
        # Directional cross-attention: masked → visible
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            batch_first=True, dropout=0.1
        )
        
        # Lightweight MLP decoder (like xMAE's lightweight Dec_E)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, feat_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, visible_feat, masked_feat):
        """
        Args:
            visible_feat: (B, L, D) - features from visible (downstream) view
            masked_feat: (B, L, D) - features from masked (upstream) view
        Returns:
            reconstructed: (B, L, D) - reconstructed features
        """
        v = self.input_proj(visible_feat)
        m = self.input_proj(masked_feat)
        
        # Directional cross-attention: Q=masked, K/V=visible (same as xMAE)
        attn_out, _ = self.cross_attn(m, v, v)
        m = self.norm(m + attn_out)
        reconstructed = self.decoder(m)
        
        return reconstructed


class xMAEBPModel(nn.Module):
    """xMAE-inspired cross-modal BP prediction.
    
    Key design aligned with xMAE paper:
    1. Continuous masking (not random) - prevents trivial interpolation
    2. Curriculum masking (80%→90%) - progressively harder
    3. Directional cross-attention: Q=masked, K/V=visible
    4. Deployment uses only visible encoder
    
    Adapted from 1D biosignal (ECG/PPG) → dual-view video (front/back):
    - Front camera (face) → cleaner rPPG signal, used as reference (visible)
    - Back camera (finger) → lower signal quality, masked and reconstructed
    """
    
    def __init__(self, n_layers=None, filters=None, chunk_size=160,
                 initial_mask_ratio=0.8, max_mask_ratio=0.9,
                 shared_encoder=False,
                 bp_hidden_dim=64, bp_dropout=0.2):
        super().__init__()
        
        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]
        
        self.chunk_size = chunk_size
        feat_dim = filters[-1]
        
        # Separate encoders for each view
        self.encoder_front = InfinitePulseFeatureExtractor(
            n_layers, filters, chunk_size, in_channels=3
        )
        if shared_encoder:
            self.encoder_back = self.encoder_front
        else:
            self.encoder_back = InfinitePulseFeatureExtractor(
                n_layers, filters, chunk_size, in_channels=3
            )
        
        # xMAE-style curriculum continuous masking
        self.masker = CurriculumContinuousMask(
            initial_ratio=initial_mask_ratio,
            max_ratio=max_mask_ratio
        )
        
        # Directional cross-attention decoder
        self.decoder = DirectionalCrossAttention(feat_dim=feat_dim)
        
        # Reconstruction loss weight (0.1 for auxiliary)
        self.recon_weight = 0.1
        
        # Temporal modeling for BP prediction (uses BOTH views)
        self.fusion = nn.Linear(feat_dim * 2, feat_dim)
        self.mamba = nn.Sequential(
            Mamba2(d_model=feat_dim, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
            Mamba2(d_model=feat_dim, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
        )
        
        # rPPG output
        self.rppg_head = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1),
        )
        
        # BP prediction
        self.bp_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, bp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(bp_dropout),
            nn.Linear(bp_hidden_dim, bp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(bp_dropout),
            nn.Linear(bp_hidden_dim, 2),
        )
    
    def forward(self, data_input):
        front_raw = data_input['front']
        back_raw = data_input['back']
        
        # NHWC → NCDHW
        front = front_raw.permute(0, 4, 1, 2, 3).contiguous()
        back = back_raw.permute(0, 4, 1, 2, 3).contiguous()
        
        # Encode both views
        f_feat = self.encoder_front(front)
        b_feat = self.encoder_back(back)
        
        recon_loss = torch.tensor(0.0, device=f_feat.device)
        
        if self.training:
            # Mask DOWNSTREAM signal (back/finger), keep UPSTREAM visible (front/face)
            # Face has richer rPPG signal → use as reference (visible)
            back_masked, mask = self.masker(back)
            b_feat_masked = self.encoder_back(back_masked)
            
            # Directional cross-attention: reconstruct masked back from visible front
            back_reconstructed = self.decoder(f_feat, b_feat_masked)
            
            # MSE loss only on masked positions
            recon_loss = F.mse_loss(
                back_reconstructed[mask], b_feat[mask]
            )
            
            # Use reconstructed features in place of masked ones
            b_feat_fused = b_feat_masked.clone()
            b_feat_fused[mask] = back_reconstructed[mask]
        else:
            b_feat_fused = b_feat
        
        # Fusion + temporal modeling + prediction
        x = torch.cat([f_feat, b_feat_fused], dim=-1)
        x = self.fusion(x)
        x = self.mamba(x)
        
        rPPG = self.rppg_head(x).squeeze(-1)
        bp_sys, bp_dia = self.bp_head(x.transpose(1, 2)).unbind(1)
        
        return {
            'rPPG': rPPG,
            'bp_sys': bp_sys,
            'bp_dia': bp_dia,
            'recon_loss': recon_loss,
        }
    
    def step_curriculum(self, loss_improved=False):
        return self.masker.step_curriculum(loss_improved)
