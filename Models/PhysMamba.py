import math
import torch
import torch.nn as nn
from torchsummary import summary
from timm.models.layers import trunc_normal_, DropPath
from torch.nn import functional as F
import sys
from mamba_ssm.modules.mamba_simple import Mamba

def conv_block(in_channels, out_channels, kernel_size, stride, padding=(0, 0, 0), bn=True, activation='relu'):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm3d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)

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
    def __init__(self, dim, d_state=8, expand=1, drop_rate=0.):
        super().__init__()

        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.block = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                expand=expand,    # Block expansion factor
                mamba_type="v1", # Fusion type
        )
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, input0, input1):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, extra_emb=input1)
        x_out = skip + self.drop_path(output)
        return x_out

class FusionMambaLayer(nn.Module):
    def __init__(self, dim=64,drop_rate=0.,final=False):
        super(FusionMambaLayer, self).__init__()  # Initialize the parent class first
        self.final = final
        self.dim = dim
        self.front_cross_mamba = CrossMambaBlock(dim,drop_rate=drop_rate)
        self.back_cross_mamba = CrossMambaBlock(dim,drop_rate=drop_rate)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, front, back):
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
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
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
    