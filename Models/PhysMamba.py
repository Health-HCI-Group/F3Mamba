import math
import torch
import torch.nn as nn
from torchsummary import summary
from timm.models.layers import trunc_normal_, DropPath
from torch.nn import functional as F
import sys
from mamba_ssm.modules.mamba_simple import Mamba,MambaWithCrossTemporalScanning

def conv_block(in_channels, out_channels, kernel_size, stride, padding=(0, 0, 0), bn=True, activation='relu'):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm3d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)

class SummitVitalNet(nn.Module):
    def __init__(self):
        super(SummitVitalNet, self).__init__()
        self.conv1 = nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  
        return x

class TNM(nn.Module):
    def __init__(self, enabled=True, frames=0, axis=2, eps=1e-6):
        super(TNM, self).__init__()
        self.enabled = enabled
        self.frames = frames
        self.axis = axis
        self.eps = eps

        def norm(x):
            if self.frames == 0:
                self.frames = x.shape[axis]
            dtype = x.dtype
            x_ = x.to(torch.float32)
            x_ = x_.reshape((*x.shape[:self.axis], -1, self.frames, *x.shape[self.axis+1:]))
            
            mean = x_.mean(dim=self.axis + 1, keepdim=True)
            tshape = [1] * len(x_.shape)
            tshape[self.axis + 1] = self.frames
            t = torch.linspace(0, 1, self.frames).reshape(tshape).to(x.device)
            
            n = ((t - 0.5) * (x_ - mean)).sum(dim=self.axis + 1, keepdim=True)
            d = ((t - 0.5) ** 2).sum(dim=self.axis + 1, keepdim=True)
            i = mean - n / d * 0.5
            trend = n / d * t + i
            x_ = x_ - trend
            
            std = ((x_ ** 2).mean(dim=self.axis + 1, keepdim=True) + self.eps).sqrt()
            x_ = x_ / std
            
            x_ = x_.reshape(x.shape)
            return x_.to(dtype)
        self.norm = norm

    def forward(self, x):
        if self.enabled:
            return self.norm(x)
        else:
            return x

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x*attention

class LateralConnection(nn.Module):
    def __init__(self, fast_channels=32, slow_channels=64):
        super(LateralConnection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, [3, 1, 1], stride=[2, 1, 1], padding=[1,0,0]),   
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        
    def forward(self, slow_path, fast_path):
        fast_path = self.conv(fast_path)
        return fast_path + slow_path

class CrossMambaBlock(nn.Module):
    def __init__(self, dim, d_state=8, expand=1, drop_rate=0.,norm_type="LN",bimamba=False):
        super().__init__()

        if norm_type == "LN":
            self.norm0 = nn.LayerNorm(dim)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        elif norm_type == "TNM":
            self.norm0 = TNM()
            self.norm1 = TNM()
            self.norm2 = TNM()

        self.crossblock = MambaWithCrossTemporalScanning(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                expand=expand,    # Block expansion factor
                mamba_type="fusion",
                bimamba = False,

        )
        self.block = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                expand=expand,    # Block expansion factor
                mamba_type="fusion", # Fusion type
                bimamba = bimamba
        )
        drop_path = drop_rate
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input0, input1):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        # output = self.crossblock(input0, extra_emb=input1)
        output = self.block(input0, extra_emb=input1)
        # output = self.norm2(output)
        x_out = skip + self.drop_path(output)
        return x_out

class FusionMambaLayer(nn.Module):
    def __init__(self, dim=64,drop_rate=0.,final=False,fft=False,bimamba=False):
        super(FusionMambaLayer, self).__init__()  # Initialize the parent class first
        self.final = final
        self.dim = dim
        self.use_fft = fft
        if self.use_fft:
            self.fft_block1 = FFT_Layer(dim, mlp_ratio=1)
            self.fft_block2 = FFT_Layer(dim, mlp_ratio=1)
            # self.front_cross_mamba_real = CrossMambaBlock(dim)
            # self.front_cross_mamba_imag = CrossMambaBlock(dim)
            # self.back_cross_mamba_real = CrossMambaBlock(dim)
            # self.back_cross_mamba_imag = CrossMambaBlock(dim)
            self.front_cross_mamba = CrossMambaBlock(dim,drop_rate=drop_rate,bimamba=bimamba)
            self.back_cross_mamba = CrossMambaBlock(dim,drop_rate=drop_rate,bimamba=bimamba)
        else:
            self.front_cross_mamba = CrossMambaBlock(dim,drop_rate=drop_rate,bimamba=bimamba)
            self.back_cross_mamba = CrossMambaBlock(dim,drop_rate=drop_rate,bimamba=bimamba)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, front, back):

        if self.use_fft:
            """
            front: (B, C, L) | back: (B, C, L) the last layer
            front: (B, C , L, H, W) | back: (B, C, L, H, W) the other layers
            """
            # B,C,L = front.shape
            # front_flat = front.permute(0, 2, 1) # (B,L,C)
            # back_flat = back.permute(0, 2, 1) # (B,L,C)

            B, d_model = front.shape[:2]
            assert d_model == self.dim
            n_tokens = front.shape[2:].numel()
            img_dims = front.shape[2:]
            front_flat = front.reshape(B, d_model, n_tokens).transpose(-1, -2) 
            back_flat = back.reshape(B, d_model, n_tokens).transpose(-1, -2)

            # 使用abs  
            front_fft_abs = torch.abs(self.fft_block1(front_flat))  # (B, L, d_model) 取模长
            back_fft_abs = torch.abs(self.fft_block2(back_flat))    # (B, L, d_model) 取模长
           
            front_fusion = self.front_cross_mamba(front_fft_abs, back_fft_abs)  # (B, L, d_model)
            back_fusion = self.back_cross_mamba(back_fft_abs, front_fft_abs)    # (B, L, d_model)

            # 使用real+imag 
            # front_real,front_imag = self.fft_block1(front_flat)
            # back_real,back_imag = self.fft_block2(back_flat)

            # front_fusion_real = self.front_cross_mamba_real(front_real,back_real) # (B,L,C)
            # back_fusion_real = self.back_cross_mamba_real(back_real,front_real)

            # front_fusion_imag = self.front_cross_mamba_imag(front_imag,back_imag) # (B,L,C)
            # back_fusion_imag= self.back_cross_mamba_imag(back_imag,front_imag)

            # front_fusion = torch.fft.ifft(torch.view_as_complex(torch.stack([front_fusion_real,front_fusion_imag],dim=-1)), dim=1, norm="ortho").to(torch.float32)
            # back_fusion = torch.fft.ifft(torch.view_as_complex(torch.stack([back_fusion_real,back_fusion_imag],dim=-1)), dim=1, norm="ortho").to(torch.float32)

            fusion = self.out_proj((front_fusion+back_fusion)/2) # (B,L,C)

            # for the last layer
            # fusion = fusion.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

            # for the other layers
            fusion = fusion.transpose(-1, -2).reshape(B, d_model, *img_dims)

        else:
            # front: (B,C,L,H,W) | back: (B,C,L,H,W)
            B, d_model = front.shape[:2]
            assert d_model == self.dim
            n_tokens = front.shape[2:].numel()
            img_dims = front.shape[2:]
            front_flat = front.reshape(B, d_model, n_tokens).transpose(-1, -2) 
            back_flat = back.reshape(B, d_model, n_tokens).transpose(-1, -2)
            front_fusion = self.front_cross_mamba(front_flat,back_flat) # (B,L,C)
            back_fusion = self.back_cross_mamba(back_flat,front_flat)
            fusion = self.out_proj((front_fusion+back_fusion)/2) # (B,L,C)
            fusion = fusion.transpose(-1, -2).reshape(B, d_model, *img_dims)

        if self.use_fft:
            if self.final:
                return fusion
            else:
                return (front+fusion)/2, (back+fusion)/2
        else:
            if self.final:
                return fusion
            else:
                return (front+fusion)/2, (back+fusion)/2
            
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal
    
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super(MambaLayer, self).__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        drop_path = 0
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                mamba_type="normal",
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_patch_token(self, x):
        B, C, nf, H, W = x.shape
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))
        out = x_out.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out 

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        out = self.forward_patch_token(x)
        return out

class PhysMamba(nn.Module):
    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128):
        super(PhysMamba, self).__init__()

        # Stem
        self.ConvBlock1 = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])  
        self.ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)

        # Downsampling
        self.ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)

        self.ConvBlock6 = conv_block(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0], activation='elu')

        # Temporal Difference Mamba Blocks
        # Slow Stream
        self.Block1 = self._build_block(64, theta)
        self.Block2 = self._build_block(64, theta)
        self.Block3 = self._build_block(64, theta)
        # Fast Stream
        self.Block4 = self._build_block(32, theta)
        self.Block5 = self._build_block(32, theta)
        self.Block6 = self._build_block(32, theta)

        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(96, 48, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(48),
            nn.ELU(),
        )

        self.ConvBlockLast = nn.Conv3d(48, 1, [1, 1, 1], stride=1, padding=0)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.fuse_1 = LateralConnection(fast_channels=32, slow_channels=64)
        self.fuse_2 = LateralConnection(fast_channels=32, slow_channels=64)

        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def _build_block(self, channels, theta):
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            MambaLayer(dim=channels),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )
    
    def forward(self, x): 
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x) 
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)  
        x = self.MaxpoolSpa(x) 
    
        # Process streams
        s_x = self.ConvBlock4(x) # Slow stream 
        f_x = self.ConvBlock5(x) # Fast stream 

        # First set of blocks and fusion
        s_x1 = self.Block1(s_x) 
        s_x1 = self.MaxpoolSpa(s_x1)
        s_x1 = self.drop_1(s_x1)

        f_x1 = self.Block4(f_x)
        f_x1 = self.MaxpoolSpa(f_x1)
        f_x1 = self.drop_2(f_x1)

        s_x1 = self.fuse_1(s_x1,f_x1) # LateralConnection [64]

        # Second set of blocks and fusion
        s_x2 = self.Block2(s_x1)
        s_x2 = self.MaxpoolSpa(s_x2)
        s_x2 = self.drop_3(s_x2)
        
        f_x2 = self.Block5(f_x1)
        f_x2 = self.MaxpoolSpa(f_x2)
        f_x2 = self.drop_4(f_x2)

        s_x2 = self.fuse_2(s_x2,f_x2) # LateralConnection
        
        # Third blocks and upsampling
        s_x3 = self.Block3(s_x2) 
        s_x3 = self.upsample1(s_x3) 
        s_x3 = self.drop_5(s_x3)

        f_x3 = self.Block6(f_x2)
        f_x3 = self.ConvBlock6(f_x3)
        f_x3 = self.drop_6(f_x3)

        # Final fusion and upsampling
        x_fusion = torch.cat((f_x3, s_x3), dim=1) 
        x_final = self.upsample2(x_fusion) 

        x_final = self.poolspa(x_final)
        x_final = self.ConvBlockLast(x_final)

        rPPG = x_final.view(-1, length)

        return rPPG
    
class PhysMambaEncoder(nn.Module):
    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128):
        super(PhysMambaEncoder, self).__init__()

        self.ConvBlock1 = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])  
        self.ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)
        self.ConvBlock6 = conv_block(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0], activation='elu')

        # Temporal Difference Mamba Blocks
        # Slow Stream
        self.Block1 = self._build_block(64, theta)
        self.Block2 = self._build_block(64, theta)
        self.Block3 = self._build_block(64, theta)
        # Fast Stream
        self.Block4 = self._build_block(32, theta)
        self.Block5 = self._build_block(32, theta)
        self.Block6 = self._build_block(32, theta)

        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(96, 48, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(48),
            nn.ELU(),
        )

        self.ConvBlockLast = nn.Conv3d(48, 1, [1, 1, 1], stride=1, padding=0)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.fuse_1 = LateralConnection(fast_channels=32, slow_channels=64)
        self.fuse_2 = LateralConnection(fast_channels=32, slow_channels=64)

        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def _build_block(self, channels, theta):
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            MambaLayer(dim=channels),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )
    
    def forward(self, x): 
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x) 
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)  
        x = self.MaxpoolSpa(x) 
    
        # Process streams
        s_x = self.ConvBlock4(x) # Slow stream 
        f_x = self.ConvBlock5(x) # Fast stream 

        # First set of blocks and fusion
        s_x1 = self.Block1(s_x)
        s_x1 = self.MaxpoolSpa(s_x1)
        s_x1 = self.drop_1(s_x1)

        f_x1 = self.Block4(f_x)
        f_x1 = self.MaxpoolSpa(f_x1)
        f_x1 = self.drop_2(f_x1)

        s_x1 = self.fuse_1(s_x1,f_x1) # LateralConnection

        # Second set of blocks and fusion
        s_x2 = self.Block2(s_x1)
        s_x2 = self.MaxpoolSpa(s_x2)
        s_x2 = self.drop_3(s_x2)
        
        f_x2 = self.Block5(f_x1)
        f_x2 = self.MaxpoolSpa(f_x2)
        f_x2 = self.drop_4(f_x2)

        s_x2 = self.fuse_2(s_x2,f_x2) # LateralConnection
        
        # Third blocks and upsampling
        s_x3 = self.Block3(s_x2) 
        s_x3 = self.upsample1(s_x3) 
        s_x3 = self.drop_5(s_x3)

        f_x3 = self.Block6(f_x2)
        f_x3 = self.ConvBlock6(f_x3)
        f_x3 = self.drop_6(f_x3)

        # Final fusion and upsampling
        x_fusion = torch.cat((f_x3, s_x3), dim=1) 
        x_final = self.upsample2(x_fusion) 

        x_final = self.poolspa(x_final)
        x_final = self.ConvBlockLast(x_final) # [B,L]

        return x_final

class FusionPhysMamba_V01(nn.Module):
    def __init__(self, args, theta=0.5, drop_rate1=0.25, drop_rate2=0.5):
        super(FusionPhysMamba_V01, self).__init__()

        self.args = args
        self.frames = self.args.seq_len

        self.ConvBlock1_front = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])  
        self.ConvBlock2_front = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3_front = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        
        self.ConvBlock1_back = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])  
        self.ConvBlock2_back = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3_back = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)


        self.ConvBlock4 = conv_block(64, 64, [2, 1, 1], stride=[2, 1, 1], padding=0)
        self.ConvBlock5 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.ConvBlock6 = conv_block(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0], activation='elu')


        # Temporal Difference Mamba Blocks
        # Slow Stream
        self.Block1 = self._build_block(64, theta=theta)
        self.Block2 = self._build_block(64, theta=theta)
        self.Block3 = self._build_block(64, theta=theta)
        # Fast Stream
        self.Block4 = self._build_block(64, theta=0.5)
        self.Block5 = self._build_block(64, theta=0.5)
        self.Block6 = self._build_block(64, theta=0.5)

        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(128, 64, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        # 64+32 / 2 = 48
        self.ConvBlockLast = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.fuse_1 = LateralConnection(fast_channels=64, slow_channels=64)
        self.fuse_2 = LateralConnection(fast_channels=64, slow_channels=64)

        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)

        self.poolspa = nn.AdaptiveAvgPool3d((self.frames, 1, 1))

    def _build_block(self, channels, theta):
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            MambaLayer(dim=channels),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )
    
    def forward(self, data_input): 
        outputs = {}
        front = data_input["front"]
        back = data_input["back"]

        front = self.ConvBlock1_front(front) # [4, 16, 160, 128, 128]
        front = self.MaxpoolSpa(front) # [4, 16, 160, 64, 64]
        front = self.ConvBlock2_front(front) # [4, 32, 160, 64, 64] 
        front = self.ConvBlock3_front(front) # [4, 64, 160, 64, 64]
        front = self.MaxpoolSpa(front) # [4, 64, 160, 32, 32]

        back = self.ConvBlock1_back(back) # [4, 16, 160, 128, 128]
        back = self.MaxpoolSpa(back) # [4, 16, 160, 64, 64]
        back = self.ConvBlock2_back(back) # [4, 32, 160, 64, 64] 
        back = self.ConvBlock3_back(back) # [4, 64, 160, 64, 64]
        back = self.MaxpoolSpa(back) # [4, 64, 160, 32, 32]

        # Process streams
        front = self.ConvBlock4(front) # [4, 64, 40, 32, 32] 
        back = self.ConvBlock5(back) # [4, 64, 40, 32, 32]

        # First set of blocks and fusion
        front1 = self.Block1(front) # [4, 64, 40, 32, 32]
        front1 = self.MaxpoolSpa(front1) # [4, 64, 40, 16, 16]
        front1 = self.drop_1(front1)

        back1 = self.Block4(back) # [4, 64, 40, 32, 32]
        back1 = self.MaxpoolSpa(back1) # [4, 64, 40, 16, 16]
        back1 = self.drop_2(back1) 

        back1 = self.fuse_1(back1,front1) # [4, 64, 40, 16, 16] 

        # Second set of blocks and fusion
        back2 = self.Block2(back1) # [4, 64, 40, 16, 16]
        back2 = self.MaxpoolSpa(back2) # [4, 64, 40, 8, 8]
        back2 = self.drop_3(back2)
        
        front2 = self.Block5(front1) # [4, 64, 40, 16, 16]
        front2 = self.MaxpoolSpa(front2) # [4, 64, 40, 8, 8]
        front2 = self.drop_4(front2)

        back2 = self.fuse_2(back2,front2) # [4, 64, 40, 8, 8]
        
        # Third blocks and upsampling
        back3 = self.Block3(back2) # [4, 64, 40, 8, 8]
        back3 = self.upsample1(back3) # [4, 64, 80, 8, 8]
        back3 = self.drop_5(back3)

        front3 = self.Block6(front2) # [4, 64, 80, 8, 8]
        front3 = self.ConvBlock6(front3) # [4, 64, 80, 8, 8]
        front3 = self.drop_6(front3)

        # Final fusion and upsampling
        x_fusion = torch.cat((back3, front3), dim=1) # [4, 128, 80, 8, 8]
        x_final = self.upsample2(x_fusion) # [4, 64, 160, 8, 8]

        x_final = self.poolspa(x_final) # [4, 64, 160, 1, 1]
        x_final = self.ConvBlockLast(x_final) # [4, 1, 160, 1, 1]

        rPPG = x_final.view(-1, self.frames)

        outputs['rPPG'] = rPPG

        return outputs
    
class FusionPhysMamba_V02(nn.Module):
    def __init__(self, args, theta=0.5, drop_rate=0.4, base_channel=32, fusion_stage=["stage1","stage2"]):
        super(FusionPhysMamba_V02, self).__init__()

        self.args = args
        self.frames = self.args.seq_len

        self.drop_rate = self.args.drop_rate
        self.theta = self.args.theta
        self.fusion_stage =self.args.fusion_stage
        self.base = base_channel 

        # 定义通道倍数
        self.channels = {
            'stem_out': int(self.base),  
            'down1_in': int(self.base),   
            'down1_out': int(self.base * 2),            
            'down2_in': int(self.base * 2),              
            'down2_out': int(self.base * 4),     
            # 'up1_in': int(self.base * 4),        
            # 'up1_out': int(self.base * 2),             
            # 'up2_in': int(self.base * 2),                
            # 'up2_out': self.base,     
            'fuse_out': int(self.base)*4,      
        }

        # --------------------- Stem ---------------------
        self.stem_front = self._create_stem()
        self.stem_back = self._create_stem()

        # --------------------- 下采样层 ---------------------
        self.down_conv1_front = conv_block(
            in_channels=self.channels['down1_in'],
            out_channels=self.channels['down1_out'],
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )
        self.down_conv1_back = conv_block(
            in_channels=self.channels['down1_in'],
            out_channels=self.channels['down1_out'],
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )

        self.down_conv2_front = conv_block(
            in_channels=self.channels['down2_in'],
            out_channels=self.channels['down2_out'],
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )
        self.down_conv2_back = conv_block(
            in_channels=self.channels['down2_in'],
            out_channels=self.channels['down2_out'],
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )

   # --------------------- 上采样层 ---------------------
        # self.up_conv1_front = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         in_channels=self.channels['up1_in'],
        #         out_channels=self.channels['up1_out'],
        #         kernel_size=(1, 2, 2),
        #         stride=(1, 2, 2)
        #     ),
        #     nn.BatchNorm3d(self.channels['up1_out']),
        #     nn.ELU()
        # )
        # self.up_conv1_back = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         in_channels=self.channels['up1_in'],
        #         out_channels=self.channels['up1_out'],
        #         kernel_size=(1, 2, 2),
        #         stride=(1, 2, 2)
        #     ),
        #     nn.BatchNorm3d(self.channels['up1_out']),
        #     nn.ELU()
        # )

        # self.up_conv2_front = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         in_channels=self.channels['up2_in'],
        #         out_channels=self.channels['up2_out'],
        #         kernel_size=(1, 2, 2),
        #         stride=(1, 2, 2)
        #     ),
        #     nn.BatchNorm3d(self.channels['up2_out']),
        #     nn.ELU()
        # )
        # self.up_conv2_back = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         in_channels=self.channels['up2_in'],
        #         out_channels=self.channels['up2_out'],
        #         kernel_size=(1, 2, 2),
        #         stride=(1, 2, 2)
        #     ),
        #     nn.BatchNorm3d(self.channels['up2_out']),
        #     nn.ELU()
        # )

        self.blocks_front = nn.ModuleList([
            self._build_block(self.channels['stem_out'], theta=self.theta),    
            self._build_block(self.channels['down1_out'], theta=self.theta),   
            self._build_block(self.channels['down2_out'], theta=self.theta),   
            # self._build_block(self.channels['up1_out'], theta=self.theta),     
            # self._build_block(self.channels['up2_out'], theta=self.theta),     
        ])
        self.blocks_back = nn.ModuleList([
            self._build_block(self.channels['stem_out'], theta=self.theta),
            self._build_block(self.channels['down1_out'], theta=self.theta),
            self._build_block(self.channels['down2_out'], theta=self.theta),
            # self._build_block(self.channels['up1_out'], theta=self.theta),
            # self._build_block(self.channels['up2_out'], theta=self.theta),
        ])

        # Upsampling
        # self.upsample1 = nn.Sequential(
        #     nn.ConvTranspose3d(64, 64, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
        #     nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1, 0, 0)),
        #     nn.BatchNorm3d(64),
        #     nn.ELU(),
        # )
        # self.upsample2 = nn.Sequential(
        #     nn.ConvTranspose3d(128, 64, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
        #     nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1, 0, 0)),
        #     nn.BatchNorm3d(64),
        #     nn.ELU(),
        # )

        # --------------------- skip connection 卷积块 ---------------------
        # self.ConvBlock1_front = conv_block(
        #     in_channels=self.channels['up1_out'],
        #     out_channels=self.channels['up1_out'],
        #     kernel_size=(3, 3, 3),
        #     stride=(1, 1, 1),
        #     padding=(1, 1, 1),
        #     activation='elu'
        # )
        # self.ConvBlock1_back = conv_block(
        #     in_channels=self.channels['up1_out'],
        #     out_channels=self.channels['up1_out'],
        #     kernel_size=(3, 3, 3),
        #     stride=(1, 1, 1),
        #     padding=(1, 1, 1),
        #     activation='elu'
        # )
        # self.ConvBlock2_front = conv_block(
        #     in_channels=self.channels['up2_out'],
        #     out_channels=self.channels['up2_out'],
        #     kernel_size=(3, 3, 3),
        #     stride=(1, 1, 1),
        #     padding=(1, 1, 1),
        #     activation='elu'
        # )
        # self.ConvBlock2_back = conv_block(
        #     in_channels=self.channels['up2_out'],
        #     out_channels=self.channels['up2_out'],
        #     kernel_size=(3, 3, 3),
        #     stride=(1, 1, 1),
        #     padding=(1, 1, 1),
        #     activation='elu'
        # )
        
        # --------------------- 融合层 ---------------------
        self.fuse_1 = FusionMambaLayer(dim=self.channels['stem_out'])       # 32
        self.fuse_2 = FusionMambaLayer(dim=self.channels['down1_out'])      # 64
        # self.fuse_3 = FusionMambaLayer(dim=self.channels['down2_out'])      # 128
        # self.fuse_4 = FusionMambaLayer(dim=self.channels['up1_out'])        # 64
        self.fuse_out = FusionMambaLayer(dim=self.channels['fuse_out'], final=True)  # 128

        self.end_fusion = SummitVitalNet()
        
        self.ConvBlockLast = nn.Conv3d(
            in_channels=self.channels['fuse_out'],
            out_channels=1,
            kernel_size=[1, 1, 1],
            stride=1,
            padding=0
        )

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((self.frames, 1, 1))

    def _build_block(self, channels, theta):
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            MambaLayer(dim=channels),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )
    
    def _create_stem(self):
        # tnm = TNM()
        conv_block1 = conv_block(3, 8, [1, 5, 5], stride=1, padding=[0, 2, 2])
        conv_block2 = conv_block(8, 16, [3, 3, 3], stride=1, padding=1)
        conv_block3 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        return nn.Sequential(conv_block1, nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
                            conv_block2,conv_block3, nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)))
    
    def forward(self, data_input): 
        outputs = {}
        front = data_input["front"]
        back = data_input["back"]
        
        # Stem 输出: (B, C, T, H, W)
        front = self.stem_front(front)
        back = self.stem_back(back) 

        # Block1：C → 2C
        front_enc1 = self.blocks_front[0](front) # (B, C, T, H, W)
        back_enc1 = self.blocks_back[0](back)

        if "stage1" in self.fusion_stage:
            front_fuse1_skip, back_fuse1_skip = self.fuse_1(front_enc1, back_enc1) # (B, C, T, H, W)
        else:
            front_fuse1_skip = front_enc1
            back_fuse1_skip = back_enc1

        front_fuse1 = self.down_conv1_front(front_fuse1_skip)  # (B, 2C, T, H/2, W/2)
        back_fuse1 = self.down_conv1_back(back_fuse1_skip)  # (B, 2C, T, H/2, W/2)

        # Block2：2C → 4C
        front_enc2 = self.blocks_front[1](front_fuse1)
        back_enc2 = self.blocks_back[1](back_fuse1)

        if "stage2" in self.fusion_stage:
            front_fuse2_skip, back_fuse2_skip = self.fuse_2(front_enc2, back_enc2) # (B, C, T, H, W)
        else:
            front_fuse2_skip = front_enc2
            back_fuse2_skip = back_enc2

        front_fuse2 = self.down_conv2_front(front_fuse2_skip)  # (B, 4C, T, H/4, W/4)
        back_fuse2 = self.down_conv2_back(back_fuse2_skip)  # (B, 4C, T, H/4, W/4)

        # Block3：4C → 2C
        front_enc3 = self.blocks_front[2](front_fuse2) # (B, 4C, T, H/4, W/4)
        back_enc3 = self.blocks_back[2](back_fuse2)

        # front_fuse3, back_fuse3 = self.fuse_3(front_enc3, back_enc3) 
        x_fusion = self.fuse_out(front_enc3, back_enc3)

        # front_dec1 = self.up_conv1_front(front_fuse3)  # (B, 2C, T, H/2, W/2)
        # back_dec1 = self.up_conv1_back(back_fuse3)  # (B, 64, T, H/2, W/2)

        # # Skip connection
        # front_dec1 = (front_dec1 + front_fuse2_skip)/2
        # back_dec1 = (back_dec1 + back_fuse2_skip)/2

        # # 3D conv
        # front_dec1 = self.ConvBlock1_front(front_dec1)
        # back_dec1 = self.ConvBlock1_back(back_dec1)

        # # Block4：64 → 32
        # front_dec2 = self.blocks_front[3](front_dec1)
        # back_dec2 = self.blocks_front[3](back_dec1)

        # front_fuse4, back_fuse4 = self.fuse_4(front_dec2, back_dec2)

        # front_dec2 = self.up_conv2_front(front_fuse4)  # (B, 32, T, H, W)
        # back_dec2 = self.up_conv2_back(back_fuse4)

        # # Skip connection
        # front_dec2 = (front_dec2 + front_fuse1_skip)/2
        # back_dec2 = (back_dec2 + back_fuse1_skip)/2

        # # 3D conv
        # front_dec2 = self.ConvBlock2_front(front_dec2)
        # back_dec2 = self.ConvBlock2_back(back_dec2)

        # # 最终融合
        # x_fusion = self.fuse_out(front_dec2, back_dec2)  # (B, 32, T, H, W)

        # 全局池化和输出
        x_final = self.poolspa(x_fusion)  # (B, 4C, T, 1, 1)
        # x_final = self.end_fusion(x_final)
        x_final = self.ConvBlockLast(x_final)  # (B, 1, T, 1, 1)
        rPPG = x_final.view(-1, self.frames)
        outputs['rPPG'] = rPPG

        return outputs

class FusionPhysMamba_V03(nn.Module):
    def __init__(self, args, theta=0.5, drop_rate=0.4, base_channel=64, fusion_stage=["stage1","stage2"]):
        super(FusionPhysMamba_V03, self).__init__()

        self.args = args
        self.frames = self.args.seq_len

        self.drop_rate = self.args.drop_rate
        self.theta = self.args.theta
        self.fusion_stage =self.args.fusion_stage
        self.base = base_channel 

        # 定义通道倍数
        self.channels = {
            'stem_out': int(self.base),  
            'down1_in': int(self.base),   
            'down1_out': int(self.base),            
            'down2_in': int(self.base),              
            'down2_out': int(self.base),     
            'fuse_out': int(self.base),      
        }

        # --------------------- Stem ---------------------
        self.stem_front = self._create_stem()
        self.stem_back = self._create_stem()

        # --------------------- 下采样层 ---------------------

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(4,1,1)),
            nn.Conv3d(self.base, self.base, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(self.base),
            nn.ELU(),
        )

        self.blocks_front = nn.ModuleList([
            self._build_block(self.channels['stem_out'], theta=self.theta),    
            self._build_block(self.channels['down1_out'], theta=self.theta),   
            self._build_block(self.channels['down2_out'], theta=self.theta),   

        ])
        self.blocks_back = nn.ModuleList([
            self._build_block(self.channels['stem_out'], theta=self.theta),
            self._build_block(self.channels['down1_out'], theta=self.theta),
            self._build_block(self.channels['down2_out'], theta=self.theta),
        ])

        # --------------------- 融合层 ---------------------
        self.fft_fusion = FusionMambaLayer(dim=self.channels['stem_out'],fft=True)
        self.fuse_1 = FusionMambaLayer(dim=self.channels['down1_out'],drop_rate=self.args.drop_rate_path,bimamba=False)  
        self.fuse_2 = FusionMambaLayer(dim=self.channels['down2_out'],drop_rate=self.args.drop_rate_path,bimamba=False)  
        self.fuse_out = FusionMambaLayer(dim=self.channels['fuse_out'],drop_rate=0.,final=True,bimamba=False)  

        self.ConvBlockLast = nn.Conv3d(
            in_channels=self.channels['fuse_out'],
            out_channels=1,
            kernel_size=[1, 1, 1],
            stride=1,
            padding=0
        )

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.poolspa = nn.AdaptiveAvgPool3d((self.frames, 1, 1))

    def _build_block(self, channels, theta):
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            MambaLayer(dim=channels),
            # ChannelAttention3D(in_channels=channels, reduction=2),
        )
    
    def _create_stem(self):
        conv_block1 = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])
        conv_block2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        conv_block3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        conv_block4 = conv_block(64, 64, [4, 1, 1], stride=[4,1,1], padding=0)
        return nn.Sequential(conv_block1,
                                nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
                                conv_block2,
                                conv_block3,
                                nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
                                conv_block4)
    
    def forward(self, data_input): 
        outputs = {}
        front = data_input["front"]
        back = data_input["back"]
        
        # Stem 输出: (B, C, T, H, W)
        front = self.stem_front(front)
        back = self.stem_back(back) 
        
        # Block1：
        front_enc1 = self.blocks_front[0](front) # (B, C, T, H, W)
        back_enc1 = self.blocks_back[0](back)

        if "stage1" in self.fusion_stage:
            front_fuse1_skip, back_fuse1_skip = self.fuse_1(front_enc1, back_enc1) # (B, C, T, H, W)
        else:
            front_fuse1_skip = front_enc1
            back_fuse1_skip = back_enc1

        front_fuse1 = self.MaxpoolSpa(front_fuse1_skip)  # (B, C, T, H/2, W/2)
        back_fuse1 = self.MaxpoolSpa(back_fuse1_skip)  # (B, C, T, H/2, W/2)

        # Block2：
        front_enc2 = self.blocks_front[1](front_fuse1)
        back_enc2 = self.blocks_back[1](back_fuse1)

        if "stage2" in self.fusion_stage:
            front_fuse2_skip, back_fuse2_skip = self.fuse_2(front_enc2, back_enc2) # (B, C, T, H, W)
        else:
            front_fuse2_skip = front_enc2
            back_fuse2_skip = back_enc2

        front_fuse2 = self.MaxpoolSpa(front_fuse2_skip)  # (B, C, T, H/4, W/4)
        back_fuse2 = self.MaxpoolSpa(back_fuse2_skip)  # (B, C, T, H/4, W/4)

        # Block3：
        front_enc3 = self.blocks_front[2](front_fuse2) # (B, C, T, H/4, W/4)
        back_enc3 = self.blocks_back[2](back_fuse2)

        x_fusion= self.fuse_out(front_enc3, back_enc3)

        # 全局池化和输出
        x_fusion = self.upsample(x_fusion)
        x_final = self.poolspa(x_fusion)  # (B, C, T, 1, 1)
        x_final = self.ConvBlockLast(x_final)  # (B, 1, T, 1, 1)
        rPPG = x_final.view(-1, self.frames)
        outputs['rPPG'] = rPPG

        return outputs
