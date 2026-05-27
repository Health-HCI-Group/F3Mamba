"""
InfinitePulse: Mamba2-based rPPG model (PyTorch port from FacePhys/Keras).

Architecture:
  Input: (B, L, H, W, 3)  [NHWC, 160 frames]
  → 4 × SSCBlock (Conv3D + TNM + Mamba2 SSD, spatial downsample ×2 each)
  → spatial_mean → Dense(32) → Dropout → 2×Mamba2 → Dense(1)
  → Output: (B, L)  BVP waveform
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# RMSNorm (gated)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
        x = x.float()
        if z is not None:
            x = x * F.silu(z.float())
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return (x * rms * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# segsum & SSD chunked scan  (Mamba-2 core)
# ---------------------------------------------------------------------------
def segsum(x: Tensor) -> Tensor:
    """Segment sum for lower-triangular cumulative log-decay matrix.
    Input:  (..., L)
    Output: (..., L, L)  lower-triangular, -inf above diagonal
    """
    T = x.shape[-1]
    x = x.unsqueeze(-1).expand(*x.shape, T)           # (..., L, L)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.float32), diagonal=-1)
    x = x * mask
    x_segsum = torch.cumsum(x, dim=-2)
    mask = 1 - torch.tril(torch.ones(T, T, device=x.device, dtype=torch.float32))
    x_segsum = x_segsum.masked_fill(mask.bool(), float('-inf'))
    return x_segsum


def ssd(x: Tensor, A: Tensor, B: Tensor, C: Tensor,
        chunk: int = 64,
        init_state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """State Space Duality (Mamba-2) chunked scan.

    Args:
        x: (B, L, H, P)   input projected by dt
        A: (B, L, H)      discretized log-A × dt  (negative values, will be exp'd)
        B: (B, L, N)      input-dependent B  (shared across heads)
        C: (B, L, N)      input-dependent C  (shared across heads)
        chunk: chunk size
        init_state: (B, 1, H, P, N) or None

    Returns:
        y: (B, L, H, P)   output
        final_state: (B, 1, H, P, N)
    """
    Bsize, L, H, P = x.shape
    _, _, N = C.shape

    # Chunk the sequence
    x = rearrange(x, 'b (c l) h p -> b c l h p', l=chunk)      # (B, C, Lc, H, P)
    A = rearrange(A, 'b (c l) h -> b c l h', l=chunk)            # (B, C, Lc, H)
    B = rearrange(B, 'b (c l) n -> b c l n', l=chunk)            # (B, C, Lc, N)
    C = rearrange(C, 'b (c l) n -> b c l n', l=chunk)            # (B, C, Lc, N)

    A = rearrange(A, 'b c l h -> b h c l')                       # (B, H, C, Lc)
    A_cum = torch.cumsum(A, dim=-1)                               # (B, H, C, Lc)

    # --- Intra-chunk (diagonal) ---
    # L_mat[b, h, c, i, j] = exp(sum_{k=j}^{i} A[k])
    L_mat = torch.exp(segsum(A))                                  # (B, H, C, Lc, Lc)

    # C_tiled: (B, C, Lc_q, Lc_s, N)   [repeat query C over source dim]
    # B_tiled: (B, C, Lc_q, Lc_s, N)   [repeat source B over query dim]
    C_tiled = C.unsqueeze(2).expand(-1, -1, L_mat.shape[-1], -1, -1)
    B_tiled = B.unsqueeze(2).expand(-1, -1, L_mat.shape[-1], -1, -1)

    Y_diag = torch.einsum('bclqn,bcsqn,bhcls,bcshp->bclhp',
                          C_tiled, B_tiled, L_mat, x)

    # --- Inter-chunk (off-diagonal) ---
    decay_states = torch.exp(A_cum[..., -1:] - A_cum)             # (B, H, C, Lc)
    B_tiled2 = B.unsqueeze(2).expand(-1, -1, L_mat.shape[-1], -1, -1)
    states_inner = torch.einsum('bclqn,bhcl,bclhp->bchpn',
                                B_tiled2, decay_states, x)        # (B, C, H, P, N)

    if init_state is None:
        init_state = torch.zeros_like(states_inner[:, :1])
    states_all = torch.cat([init_state, states_inner], dim=1)     # (B, C+1, H, P, N)

    # Decay between chunks
    A_end = A_cum[..., -1]                                         # (B, H, C)
    A_end_padded = F.pad(A_end, (1, 0))                            # (B, H, C+1)
    decay_chunk = torch.exp(segsum(A_end_padded))                   # (B, H, C+1, C+1)

    new_states = torch.einsum('bhzc,bchpn->bzhpn',
                              decay_chunk, states_all)             # (B, C+1, H, P, N)
    states_mid, final_state = new_states[:, :-1], new_states[:, -1:]

    state_decay_out = torch.exp(A_cum)                             # (B, H, C, Lc)
    C_tiled3 = C.unsqueeze(2).expand(-1, -1, L_mat.shape[-1], -1, -1)
    Y_off = torch.einsum('bclqn,bchpn,bhcl->bclhp',
                         C_tiled3, states_mid, state_decay_out)

    Y = Y_diag + Y_off
    Y = rearrange(Y, 'b c l h p -> b (c l) h p')
    return Y, final_state


# ---------------------------------------------------------------------------
# Mamba2 block
# ---------------------------------------------------------------------------
class Mamba2(nn.Module):
    """Mamba-2 block with chunked SSD scan and gated RMSNorm."""

    def __init__(self, d_model: int, expand: int = 2, d_state: int = 128,
                 d_conv: int = 4, chunk_size: int = 64, headdim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.chunk_size = chunk_size
        self.headdim = headdim

        self.d_inner = expand * d_model
        self.nheads = self.d_inner // headdim
        assert self.d_inner % headdim == 0, \
            f'd_inner ({self.d_inner}) must be divisible by headdim ({headdim})'

        # in_proj: z(d_inner) + xBC(d_inner+2*d_state) + dt(nheads)
        # xBC = x(d_inner) + B(d_state) + C(d_state)
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # Depthwise causal conv1d on [x; B; C]  (d_inner + 2*d_state channels)
        conv_dim = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv,
                                groups=conv_dim, padding=0)

        # SSM parameters (one scalar A per head, per Mamba-2 design)
        self.dt_bias = nn.Parameter(torch.randn(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads, dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        return self._step_chunk(x, None)[0]

    def _step_chunk(self, x: Tensor, state: Optional[List[Tensor]]) -> Tuple[Tensor, List[Tensor]]:
        B, L, _ = x.shape
        if state is None:
            conv_state = x.new_zeros(B, self.d_inner + 2 * self.d_state, self.d_conv)
            return self._step_chunk(x, [None, conv_state])

        # ---- in_proj and split ----
        zxbcdt = self.in_proj(x)                                 # (B, L, 2*d_inner+2*d_state+nheads)
        z   = zxbcdt[:, :, :self.d_inner]                        # d_inner
        xBC = zxbcdt[:, :, self.d_inner:2*self.d_inner+2*self.d_state]  # d_inner+2*d_state
        dt  = zxbcdt[:, :, 2*self.d_inner+2*self.d_state:]       # nheads

        dt = F.softplus(dt + self.dt_bias)                        # (B, L, nheads)
        A = -torch.exp(self.A_log)                                 # (nheads,)  per-head scalar decay
        # A_ssd: per-step log-decay for SSD scan
        A_ssd = A * dt                                             # (B, L, nheads)

        # ---- causal conv1d (valid padding) ----
        conv_state = state[1]                                     # (B, conv_dim, d_conv)
        xBC_t = xBC.transpose(1, 2)                               # (B, conv_dim, L)
        xBC_cat = torch.cat([conv_state, xBC_t], dim=-1)         # (B, conv_dim, d_conv+L)
        xBC_out = self.conv1d(xBC_cat)                             # (B, conv_dim, L+1)  [valid padding]
        xBC_out = F.silu(xBC_out)
        xBC_out = xBC_out[:, :, -L:]                               # (B, conv_dim, L)  causal trim
        xBC = xBC_out.transpose(1, 2)                              # (B, L, conv_dim)
        new_conv_state = xBC_cat[:, :, -self.d_conv:]              # (B, conv_dim, d_conv)

        # ---- split after conv ----
        x_s = xBC[:, :, :self.d_inner]
        B_s = xBC[:, :, self.d_inner:self.d_inner + self.d_state]
        C_s = xBC[:, :, self.d_inner + self.d_state:]

        # ---- reshape for SSD ----
        x_r = rearrange(x_s, 'b l (h p) -> b l h p', p=self.headdim)

        # ---- SSD scan ----
        y, new_ssd_state = ssd(
            x_r,
            A_ssd,             # (B, L, nheads)  log-decay per step
            B_s,                # (B, L, d_state)
            C_s,                # (B, L, d_state)
            chunk=self.chunk_size,
            init_state=state[0]
        )

        # ---- D skip ----
        y = y + x_r * self.D.view(1, 1, -1, 1)

        # ---- norm + out ----
        y = rearrange(y, 'b l h p -> b l (h p)')
        y = self.norm(y, z)
        y = self.out_proj(y)
        return y, [new_ssd_state, new_conv_state]


# ---------------------------------------------------------------------------
# TNM: Temporal Normalization Module
# ---------------------------------------------------------------------------
class TNM(nn.Module):
    """Per-segment temporal detrend + z-score normalization."""

    def __init__(self, frames: int = 160, eps: float = 2e-6):
        super().__init__()
        self.frames = frames
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, L, H, W)  [NCDHW]  — normalize along time dim=2"""
        B, C, L, H, W = x.shape
        frames = self.frames if self.frames else L
        n_groups = L // frames

        # Reshape: (B, C, G, F, H, W)
        x_r = x.reshape(B, C, n_groups, frames, H, W).float()

        # Linear trend fit
        t = torch.linspace(0, 1, frames, device=x.device, dtype=torch.float32)
        t = t.view(1, 1, 1, frames, 1, 1)
        xmean = x_r.mean(dim=3, keepdim=True)
        n = ((t - 0.5) * (x_r - xmean)).sum(dim=3, keepdim=True)
        d = ((t - 0.5) ** 2).sum()
        i = xmean - n / d * 0.5
        trend = n / d * t + i
        x_detrend = x_r - trend

        # Z-score (mean=0 after detrend)
        std = (x_detrend.pow(2).mean(dim=3, keepdim=True) + self.eps).sqrt()
        x_norm = x_detrend / std

        return x_norm.reshape(B, C, L, H, W).to(x.dtype)


# ---------------------------------------------------------------------------
# SSConv: Spatio-Spectral Convolution block
# ---------------------------------------------------------------------------
class SSConv(nn.Module):
    """Conv3D + Mamba2 + residual block."""

    def __init__(self, channels: int, kernel: Tuple[int, int] = (3, 3),
                 chunk: int = 160, d_state: int = 128):
        super().__init__()
        self.channels = channels
        self.chunk = chunk

        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, *kernel), padding=(0, kernel[0]//2, kernel[1]//2)),
            nn.Conv3d(channels, channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(channels * 2, channels, bias=False)
        self.mamba = Mamba2(d_model=channels * 2, expand=2, d_state=d_state,
                            d_conv=4, chunk_size=chunk, headdim=channels)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, *kernel), padding=(0, kernel[0]//2, kernel[1]//2)),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout(0.5)
        self.tn = TNM(frames=chunk)

    def forward(self, x0: Tensor) -> Tensor:
        """x0: (B, C, H, W, L) or internally tracked in a canonical format.

        We work in NCDHW internally: (B, C, L, H, W)
        """
        # x0: (B, C, L, H, W)  —  NCDHW
        x0_n = self.tn(x0)                                            # TNM
        x1 = self.conv1(x0_n)                                         # → (B, C*2, L, H, W)

        # spatial mean: (B, C*2, L, H, W) → (B, C*2, L) → (B, L, C*2)
        x2 = x1.mean(dim=(-2, -1)).transpose(1, 2)                    # (B, L, C*2)
        x2 = self.drop(x2)
        x2 = self.mamba(x2)                                            # (B, L, C*2)
        x2 = self.proj(x2)                                             # (B, L, C)

        # residual: broadcast back to spatial dims
        x0_n = x0_n + x2.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # (B, C, L, 1, 1)
        x1 = self.conv2(x0_n)                                          # (B, C, L, H, W)
        return x1


# ---------------------------------------------------------------------------
# SSCBlock: stack of SSConv + optional spatial downsample
# ---------------------------------------------------------------------------
class SSCBlock(nn.Module):
    def __init__(self, in_ch: int, num_layers: int, filters: int,
                 chunk: int = 160, kernel: Tuple[int, int] = (3, 3),
                 downsample: bool = True, d_state: int = 128):
        super().__init__()
        self.in_ch = in_ch
        self.filters = filters
        self.downsample = downsample

        self.in_proj = nn.Conv3d(in_ch, filters, kernel_size=1, bias=False)
        self.ssc = nn.ModuleList([
            SSConv(filters, kernel=kernel, chunk=chunk, d_state=d_state)
            for _ in range(num_layers)
        ])
        self.ds = nn.Conv3d(filters, filters, kernel_size=(1, 2, 2),
                            stride=(1, 2, 2), padding=(0, 0, 0)) if downsample else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C_in, L, H, W) → (B, filters, L, H', W')"""
        x1 = self.in_proj(x)
        x_out = x1
        for layer in self.ssc:
            x_out = layer(x_out)
        x_out = x_out + x1
        x_out = self.ds(x_out)
        return x_out


# ---------------------------------------------------------------------------
# InfinitePulse  —  full model
# ---------------------------------------------------------------------------
class InfinitePulse(nn.Module):
    """Mamba2-based rPPG estimation model.

    Input:  (B, L, H, W, 3)  [NHWC]  or  (B, 3, L, H, W)  [NCDHW]
    Output: (B, L)  BVP waveform
    """

    def __init__(self, n_layers: List[int] = None, filters: List[int] = None,
                 chunk_size: int = 160, in_channels: int = 3,
                 input_permute: bool = True):
        super().__init__()
        if n_layers is None:
            n_layers = [2, 2, 2, 2]
        if filters is None:
            filters = [32, 32, 32, 32]

        self.chunk_size = chunk_size
        self.input_permute = input_permute  # if input is NHWC → permute to NCDHW

        # Backbone: cascade of SSCBlocks
        blocks = []
        in_ch = in_channels
        for l, f in zip(n_layers, filters):
            blocks.append(SSCBlock(in_ch, l, f, chunk=chunk_size, downsample=True))
            in_ch = f
        self.network = nn.ModuleList(blocks)

        # Output projection
        head_dim = filters[-1]
        self.out_proj = nn.Sequential(
            nn.Linear(head_dim, 32, bias=False),
            nn.Dropout(0.5),
        )

        # Prediction head: 2×Mamba2 → Dense(1)
        self.head = nn.Sequential(
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
            Mamba2(d_model=32, expand=2, d_state=128, d_conv=4,
                   chunk_size=chunk_size, headdim=32),
        )
        self.head_out = nn.Linear(32, 1)

        self.n_layers = n_layers
        self.filters = filters

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, H, W, C) NHWC  or  (B, C, L, H, W) NCDHW  – depends on input_permute"""
        if self.input_permute:
            # NHWC → NCDHW
            x = x.permute(0, 4, 1, 2, 3).contiguous()   # (B, C, L, H, W)

        # Backbone
        for block in self.network:
            x = block(x)                                   # progressive spatial downsampling

        # Spatial mean pool → (B, filters[-1], L) → (B, L, filters[-1])
        x = x.mean(dim=(-2, -1)).transpose(1, 2)          # (B, L, C)

        # Output projection
        x = self.out_proj(x)                               # (B, L, 32)

        # Head
        x = self.head(x)                                   # (B, L, 32)
        x = self.head_out(x)                               # (B, L, 1)
        return x.squeeze(-1)                               # (B, L)


# ---------------------------------------------------------------------------
# InfinitePulseMultitask  —  multi-task version (rPPG + SpO2 + BP)
# ---------------------------------------------------------------------------
class InfinitePulseMultitask(InfinitePulse):
    """Multi-task extension: predicts rPPG, SpO2, BP systolic, BP diastolic.

    Architecture: backbone → rPPG waveform → aux_head → (SpO2, BP_sys, BP_dia).
    Auxiliary signals are predicted from the rPPG waveform features.

    Input:  (B, L, H, W, 3)  [NHWC]
    Output: dict with keys:
        - "rPPG":   (B, L)  BVP waveform
        - "spo2":   (B,)    oxygen saturation
        - "bp_sys": (B,)    systolic blood pressure
        - "bp_dia": (B,)    diastolic blood pressure
    """

    def __init__(self, n_layers: List[int] = None, filters: List[int] = None,
                 chunk_size: int = 160, in_channels: int = 3,
                 input_permute: bool = True):
        super().__init__(n_layers, filters, chunk_size, in_channels, input_permute)

        self.aux_head = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(8, 3),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if self.input_permute:
            x = x.permute(0, 4, 1, 2, 3).contiguous()

        for block in self.network:
            x = block(x)

        x = x.mean(dim=(-2, -1)).transpose(1, 2)
        x = self.out_proj(x)
        shared_feat = self.head(x)

        rPPG = self.head_out(shared_feat).squeeze(-1)

        aux_in = rPPG.unsqueeze(1)
        aux = self.aux_head(aux_in)
        spo2, bp_sys, bp_dia = aux[:, 0], aux[:, 1], aux[:, 2]

        return {
            "rPPG": rPPG,
            "spo2": spo2,
            "bp_sys": bp_sys,
            "bp_dia": bp_dia,
        }


# ---------------------------------------------------------------------------
# InfinitePulseEnhancedBP  —  BP-enhanced variant (rPPG + BP_sys + BP_dia)
# ---------------------------------------------------------------------------
class InfinitePulseEnhancedBP(InfinitePulse):
    """BP-enhanced variant: predicts rPPG + BP systolic + BP diastolic.

    Key improvements over InfinitePulseMultitask:
      1. BP head uses **shared_feat** (B, L, 32) instead of 1-channel rPPG,
         providing 32× richer temporal features.
      2. Multi-scale temporal convolutions (kernel 3, 7, 15) capture waveform
         morphology at different scales (related to PWV, augmentation index).
      3. Temporal attention focuses on informative cardiac cycles.
      4. No SpO₂ head — all capacity focused on BP.

    Physiological priors encoded in the architecture:
      - Multi-scale waveforms can capture dP/dt (systolic slope) and
        diastolic decay time constant, both correlated with BP.
      - Joint prediction of sys & dia from shared features models their
        natural correlation (r ≈ 0.6-0.8 in clinical data).

    Input:  (B, L, H, W, 3)  [NHWC]
    Output: dict with keys:
        - "rPPG":   (B, L)  BVP waveform
        - "bp_sys": (B,)    systolic blood pressure
        - "bp_dia": (B,)    diastolic blood pressure
    """

    def __init__(self, n_layers: List[int] = None, filters: List[int] = None,
                 chunk_size: int = 160, in_channels: int = 3,
                 input_permute: bool = True):
        super().__init__(n_layers, filters, chunk_size, in_channels, input_permute)

        # ── Time-domain branch: dilated convs on shared_feat ──
        self.t_conv = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(48, 48, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.t_pool = nn.AdaptiveAvgPool1d(1)   # → (B, 48)

        # ── Frequency-domain branch: log-magnitude spectrum ──
        #   STFT magnitude: (B, 32, n_fft//2+1) → conv → pool
        self.fconv1 = nn.Conv1d(32, 48, kernel_size=3, padding=1)
        self.fconv2 = nn.Conv1d(48, 32, kernel_size=3, padding=1)
        self.f_pool = nn.AdaptiveAvgPool1d(1)   # → (B, 32)

        # ── Fusion ──
        self.fc = nn.Sequential(
            nn.Linear(48 + 32, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(48, 2),  # [bp_sys, bp_dia]
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if self.input_permute:
            x = x.permute(0, 4, 1, 2, 3).contiguous()

        for block in self.network:
            x = block(x)

        x = x.mean(dim=(-2, -1)).transpose(1, 2)
        x = self.out_proj(x)
        shared_feat = self.head(x)  # (B, L, 32)

        rPPG = self.head_out(shared_feat).squeeze(-1)  # (B, L)

        # Time branch
        z = shared_feat.transpose(1, 2)          # (B, 32, L)
        t_feat = self.t_pool(self.t_conv(z)).flatten(1)  # (B, 48)

        # Frequency branch: log magnitude of FFT (pad to 256 for cuFFT half-precision)
        n_fft = 256
        f = torch.fft.rfft(shared_feat, n=n_fft, dim=1)  # (B, n_fft//2+1, 32)
        f = f.transpose(1, 2)                              # (B, 32, 129)
        f = torch.log(torch.abs(f) + 1e-8)                 # log-magnitude
        f = F.relu(self.fconv1(f))
        f = F.relu(self.fconv2(f))
        f_feat = self.f_pool(f).flatten(1)                  # (B, 32)

        # Fuse
        fused = torch.cat([t_feat, f_feat], dim=1)
        bp = self.fc(fused)

        return {
            "rPPG": rPPG,
            "bp_sys": bp[:, 0],
            "bp_dia": bp[:, 1],
        }


# ---------------------------------------------------------------------------
# Loss  (compatible with F3Mamba)
# ---------------------------------------------------------------------------
class NegPearson(nn.Module):
    """1 - Pearson correlation coefficient."""
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.flatten()
        target = target.flatten()
        pm = pred - pred.mean()
        tm = target - target.mean()
        num = (pm * tm).sum()
        den = (pm.pow(2).sum() * tm.pow(2).sum()).sqrt() + 1e-8
        return 1 - num / den
