import math
import pdb

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

class PhysNet_padding_Encoder(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            TNM(),
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            TNM(),
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            TNM(),
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        # 64 64
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)  

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        self.ConvBlock11 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.fusion_net = SummitVitalNet()

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        [batch, channel, length, width, height] = x.shape  

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        # x = self.MaxpoolSpa(x)
        # x = self.ConvBlock8(x)
        # x = self.ConvBlock9(x)
        x = self.upsample(x)
        x = self.upsample2(x)
        x = self.poolspa(x)
        # 16 8192 16 128
        x = self.ConvBlock10(x)   
        # rPPG = x.view(batch, length)
        rPPG = x

        # # for multi-task
        # spo2 = self.ConvBlock11(rPPG.unsqueeze(1)) 
        # spo2 = spo2.view(batch, 1)
        # spo2 = spo2 * 15 + 85

        return rPPG

class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            TNM(),
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            TNM(),
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            TNM(),
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            TNM(),
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        # 64 64
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)  

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        self.ConvBlock11 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.fusion_net = SummitVitalNet()

    def forward(self, x1, x2=None):  # Batch_size*[3, T, 128,128]
        [batch, channel, length, width, height] = x1.shape  
        # print(f"x1.shape: {x1.shape}")    
        if x2 is not None:
            # print(f"x1.shape: {x1.shape}, x2.shape: {x2.shape}")
            x1_visual = self.encode_video(x1)
            x2_visual = self.encode_video(x2)
            #torch.Size([16, 1, 128, 1, 1])
            # print(f"x1_visual.shape: {x1_visual.shape}, x2_visual.shape: {x2_visual.shape}")
            # torch.Size([16, 64, 128, 1, 1])
            x = self.fusion_net(x1_visual, x2_visual)
            # print(f"fusion_net.shape:{x.shape}")
        else:
            # ([16, 1, 128, 1, 1])
            x = self.encode_video(x1)
        # print(f"encode_video: {x.shape}")
        rPPG = x.view(batch, length)  
        # print(f"rPPG.shape: {rPPG.shape}")


        spo2 = self.ConvBlock11(rPPG.unsqueeze(1)) 
        spo2 = spo2.view(batch, 1)
        spo2 = spo2 * 15 + 85


        return rPPG, spo2


    def encode_video(self, x):
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        # x = self.MaxpoolSpa(x)
        # x = self.ConvBlock8(x)
        # x = self.ConvBlock9(x)
        x = self.upsample(x)
        x = self.upsample2(x)
        x = self.poolspa(x)
        # 16 8192 16 128
        x = self.ConvBlock10(x) 
        return x