import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from torch.nn import functional as F
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

class CDT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):
        super(CDT, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape
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
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super(MambaLayer, self).__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        drop_path = 0
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
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

class FusionMambaLayer(nn.Module):
    def __init__(self, dim=64, drop_rate=0., final=False):
        super(FusionMambaLayer, self).__init__()
        self.final = final
        self.dim = dim
        self.front_cross_mamba = MambaLayer(dim)
        self.back_cross_mamba = MambaLayer(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, front, back):
        B, d_model = front.shape[:2]
        assert d_model == self.dim
        n_tokens = front.shape[2:].numel()
        img_dims = front.shape[2:]
        front_flat = front.reshape(B, d_model, n_tokens).transpose(-1, -2)
        back_flat = back.reshape(B, d_model, n_tokens).transpose(-1, -2)
        front_fusion = self.front_cross_mamba(front_flat)
        back_fusion = self.back_cross_mamba(back_flat)
        fusion = self.out_proj((front_fusion + back_fusion) / 2)
        fusion = fusion.transpose(-1, -2).reshape(B, d_model, *img_dims)
        if self.final:
            return fusion
        else:
            return (front + fusion) / 2, (back + fusion) / 2

class F3Mamba(nn.Module):
    def __init__(self, args, theta=0.5, drop_rate=0.4, base_channel=64):
        super(F3Mamba, self).__init__()
        self.args = args
        self.frames = self.args.seq_len
        self.drop_rate = self.args.drop_rate
        self.theta = self.args.theta
        self.base = base_channel
        self.channels = {
            'stem_out': int(self.base),
            'down1_in': int(self.base),
            'down1_out': int(self.base),
            'down2_in': int(self.base),
            'down2_out': int(self.base),
            'fuse_out': int(self.base),
        }
        self.stem_front = self._create_stem()
        self.stem_back = self._create_stem()
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
        self.fuse_1 = FusionMambaLayer(dim=self.channels['down1_out'], drop_rate=self.args.drop_rate_path)
        self.fuse_2 = FusionMambaLayer(dim=self.channels['down2_out'], drop_rate=self.args.drop_rate_path)
        self.fuse_out = FusionMambaLayer(dim=self.channels['fuse_out'], drop_rate=0., final=True)
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
            CDT(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            MambaLayer(dim=channels),
        )

    def _create_stem(self):
        conv_block1 = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])
        conv_block2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        conv_block3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        conv_block4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
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
        front = self.stem_front(front)
        back = self.stem_back(back)
        front_enc1 = self.blocks_front[0](front)
        back_enc1 = self.blocks_back[0](back)
        front_fuse1_skip, back_fuse1_skip = self.fuse_1(front_enc1, back_enc1)
        front_fuse1 = self.MaxpoolSpa(front_fuse1_skip)
        back_fuse1 = self.MaxpoolSpa(back_fuse1_skip)
        front_enc2 = self.blocks_front[1](front_fuse1)
        back_enc2 = self.blocks_back[1](back_fuse1)
        front_fuse2_skip, back_fuse2_skip = self.fuse_2(front_enc2, back_enc2)
        front_fuse2 = self.MaxpoolSpa(front_fuse2_skip)
        back_fuse2 = self.MaxpoolSpa(back_fuse2_skip)
        front_enc3 = self.blocks_front[2](front_fuse2)
        back_enc3 = self.blocks_back[2](back_fuse2)
        x_fusion = self.fuse_out(front_enc3, back_enc3)
        x_fusion = self.upsample(x_fusion)
        x_final = self.poolspa(x_fusion)
        x_final = self.ConvBlockLast(x_final)
        rPPG = x_final.view(-1, self.frames)
        outputs['rPPG'] = rPPG
        return outputs
