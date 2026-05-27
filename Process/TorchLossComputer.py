"""
简单的 Loss 函数模块，替代原始的 TorchLossComputer。
包含 Neg_Pearson (负皮尔逊相关系数) 和 RhythmFormer_Loss。
"""

import torch
import torch.nn as nn
import numpy as np


class Neg_Pearson(nn.Module):
    """负皮尔逊相关系数损失"""
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        
    def forward(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()
        
        pred_mean = pred.mean()
        target_mean = target.mean()
        
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum() + 1e-8)
        
        return 1 - numerator / denominator


class RhythmFormer_Loss(nn.Module):
    """RhythmFormer 损失函数 (NegPearson + MSE)"""
    def __init__(self):
        super(RhythmFormer_Loss, self).__init__()
        self.neg_pearson = Neg_Pearson()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, idx=None, fs=None, diff_flag=0):
        """
        Args:
            pred: 预测信号 (L,)
            target: 真实信号 (L,)
            idx: 索引 (未使用)
            fs: 采样率 (未使用)
            diff_flag: 是否使用差分 (0=不使用)
        """
        loss = self.neg_pearson(pred, target)
        
        if diff_flag == 1:
            # 差分损失
            pred_diff = pred[1:] - pred[:-1]
            target_diff = target[1:] - target[:-1]
            loss = loss + 0.5 * self.neg_pearson(pred_diff, target_diff)
        
        return loss
