# F3Mamba BP Estimation Experiments on AZDualCamera

## Dataset Overview
- **Subjects**: 118 unique subjects (visits)
- **Train/Test split**: ~79 train, ~39 test per fold
- **Sequence length**: 160 frames (default), 320 frames (seq_mult=2)
- **Targets**: bp_sys (systolic), bp_dia (diastolic), bvp (heart rate)
- **Label stats**: bp_sys mean=124.6, std=15.1; bp_dia mean=75.3, std=9.8

---

## ⚠️ 核心问题：双视角未优于单视角！

**研究目标**: 双视角融合应该优于单视角 (dual MAE < single MAE)
**实验结果**: 双视角 ≈ 单视角，甚至略差 ❌

### 公平对比（相同数据配置，非overlap）

| 视角 | seq_mult=2 (320帧) | seq_mult=4 (640帧) |
|------|-------------------|-------------------|
| **back** | **11.88** | **11.92** |
| **front** | **11.85** | **11.90** |
| **dual** | **11.91** | **11.96** |

**关键发现**: 
- 双视角始终 ≈ 单视角，无改善
- seq_mult=4 没有比 seq_mult=2 更好
- front 和 back 单视角性能基本相同

---

## 详细实验结果

### 1. 公平对比实验（非overlap数据，统一配置）

#### seq_mult=2 (320帧, batch=2, effective=16)

| 模型 | 视角 | Fold1 sys/dia | Fold2 sys/dia | Fold3 sys/dia | Avg sys | Avg dia |
|------|------|---------------|---------------|---------------|---------|---------|
| InfinitePulseEnhancedBP | **back** | 12.42/8.35 | 12.69/8.52 | 10.52/7.40 | **11.88** | **8.09** |
| InfinitePulseEnhancedBP | **front** | 12.36/8.35 | 12.70/8.52 | 10.48/7.42 | **11.85** | **8.10** |
| LateFusionEnhancedBP sums | front+back | 12.54/8.34 | 12.70/8.52 | 10.50/7.39 | **11.91** | **8.08** |

#### seq_mult=4 (640帧, single batch=2, dual batch=1)

| 模型 | 视角 | Fold1 sys/dia | Fold2 sys/dia | Fold3 sys/dia | Avg sys | Avg dia |
|------|------|---------------|---------------|---------------|---------|---------|
| InfinitePulseEnhancedBP | **back** | 12.36/8.35 | 12.93/8.51 | 10.46/7.37 | **11.92** | **8.08** |
| InfinitePulseEnhancedBP | **front** | 12.34/8.36 | 12.94/8.50 | 10.43/7.40 | **11.90** | **8.09** |
| LateFusionEnhancedBP sums | front+back | 12.51/8.33 | 12.93/8.50 | 10.45/7.37 | **11.96** | **8.07** |

#### seq_mult=6 (960帧) - 待跑
#### seq_mult=8 (1280帧) - 待跑

---

### 2. 历史实验（旧配置，仅供参考）

#### Back Camera Only

| Experiment | Backbone | Data | Frames | Fold1 sys/dia | Fold2 sys/dia | Fold3 sys/dia | Avg sys | Avg dia |
|------------|----------|------|--------|---------------|---------------|---------------|---------|---------|
| back_overlap_seq2 | InfinitePulseEnhancedBP | overlap stride40 | **320** | 12.40/8.34 | 12.65/8.48 | 10.58/7.47 | **11.88** | **8.10** |
| enhancedBP_back_4gpu | InfinitePulseEnhancedBP | overlap stride40 | 160 | 12.24/7.97 | 12.86/8.28 | 10.54/7.58 | **11.88** | **7.94** |
| seq320 | InfinitePulseMultitask | overlap stride40 | 320 | 12.46/8.13 | 12.88/8.65 | 11.28/8.39 | **12.21** | **8.39** |
| seq640 | InfinitePulseMultitask | overlap stride40 | 640 | 12.70/8.29 | 12.70/8.48 | 10.69/7.96 | **12.03** | **8.24** |

#### Dual-View (Front+Back)

| Experiment | Backbone | Fusion | Data | Frames | Avg sys | Avg dia |
|------------|----------|--------|------|--------|---------|---------|
| overlap_seq2 | LateFusionEnhancedBP | sums | overlap stride40 | **320** | **11.90** | **8.04** |
| no_overlap_seq2 | LateFusionEnhancedBP | sums | non-overlap | **320** | **11.91** | **8.08** |
| lf_enhanced_reg | LateFusionEnhancedBP | sums | non-overlap | 160 | **14.74** | **9.20** |
| lf_enhanced_sums_bs8 | LateFusionEnhancedBP | sums | non-overlap | 160 | **12.79** | **8.69** |
| enhanced_demo | LateFusionEnhancedBPDemo | sums | overlap | 160 | 12.57/8.31 | 12.65/8.50 | 10.48/7.41 | **11.90** | **8.07** |
| lf_uncertainty | LateFusionEnhancedBP | sums | overlap | 160 | 12.53/8.30 | 12.65/8.46 | 10.55/7.45 | **11.91** | **8.07** |
| lf_concat_bs8 | LateFusionEnhancedBP | concat | non-overlap | 160 | - | - | - | **~12.8** | **~8.7** |
| lf_freq_bs8 | LateFusionEnhancedBP | freq | non-overlap | 160 | - | - | - | **~12.5** | **~8.5** |

**最佳双视角**: LateFusionEnhancedBP sums + overlap + seq_mult=2 = **bp_sys 11.90 mmHg**

---

### 3. 分类实验

| Experiment | Epochs | Fold1 Acc | Fold2 Acc | Fold3 Acc | Avg Acc |
|------------|--------|-----------|-----------|-----------|---------|
| classifier | 10 | 0.4405 | 0.5127 | 0.5095 | **0.4876** |
| classifier_20ep | 20 | 0.4405 | 0.5567 | 0.5463 | **0.5145** |

---

### 4. 心率预测 (BVP)

训练 Loss 约为 0.33-0.35（normalized），需要转换为实际 RMSE 进行对比。

---

## 问题分析

### 为什么双视角没有优于单视角？

| 可能原因 | 分析 |
|----------|------|
| **融合方式** | Late Fusion "sums" 简单相加可能丢失互补信息，未充分利用双视角优势 |
| **视角冗余** | front/back 视频高度相似，互补性弱，融合收益有限 |
| **模型结构** | 双视角使用 shared_backbone，可能导致两个视角互相干扰 |
| **训练策略** | 双视角需要更多样本、更长训练、特殊 loss 设计 |
| **数据泄漏** | overlap 数据虽然按 subject 划分 fold，但 group 间仍有相关性 |

---

## 解决方案建议

### 1. 改进融合方式
- ❌ sums (当前): `feat = front_feat + back_feat` - 简单相加丢失信息
- ✅ **concat + MLP**: `feat = MLP(concat(front_feat, back_feat))` - 保留完整信息
- ✅ **cross-attention**: 让 front 和 back 互相关注对方的关键区域
- ✅ **gated fusion**: 动态权重融合 `feat = gate_front * front + gate_back * back`

### 2. 增强互补性
- 分析 front/back 视频的差异性（光照、角度、可见特征）
- 设计专门利用差异的模块（如光照补偿、角度校正）

### 3. 训练策略优化
- 视角 curriculum: 先学单视角，再学双视角融合
- 不同视角使用不同 learning rate
- 增加 contrastive loss 让两个视角学习不同特征

---

## 公平对比实验（非overlap，统一配置）

### 实验矩阵

| exp# | Modal | seq_mult | Frames | batch | status |
|------|-------|----------|--------|-------|--------|
| 1 | back | 2 | 320 | 2 | ✅ 11.88 |
| 2 | front | 2 | 320 | 2 | ✅ 11.85 |
| 3 | dual | 2 | 320 | 2 | ✅ 11.91 |
| 4 | back | 4 | 640 | 2 | ✅ 11.92 |
| 5 | front | 4 | 640 | 2 | ✅ 11.90 |
| 6 | dual | 4 | 640 | 1 | ✅ 11.96 |
| 7 | back | 6 | 960 | 1 | 🔄 sweep |
| 8 | front | 6 | 960 | 1 | 🔄 sweep |
| 9 | dual | 6 | 960 | 1 | 🔄 sweep |
| 10 | back | 8 | 1280 | 1 | 🔄 sweep |
| 11 | front | 8 | 1280 | 1 | 🔄 sweep |
| 12 | dual | 8 | 1280 | 1 | 🔄 sweep |

### 运行方式

使用 wandb sweep 管理后续实验 (exp#7-12):

- **Sweep ID**: `q96i39oh`
- **Dashboard**: https://wandb.ai/2796321819-beihang-university/F3Mamba/sweeps/q96i39oh
- **Config**: `sweep_config_direct.yaml`
- **运行命令**: `wandb agent 2796321819-beihang-university/F3Mamba/q96i39oh`

备用脚本（无需 sweep）:
```bash
bash run_remaining_experiments.sh
```

---

## 实验日志文件列表

### 公平对比实验（非overlap，本阶段）
- `train_back_seq2.log` / `train_front_seq2.log` / `train_dual_seq2.log`
- `train_back_seq4.log` / `train_front_seq4.log` / `train_dual_seq4.log`

### 历史实验（旧配置）
- `train_back_overlap_seq2.log` / `train_overlap_seq2.log` / `train_no_overlap_seq2.log`
- `train_enhancedBP_back_4gpu.log` / `train_seq320.log` / `train_seq640.log`
- `train_enhanced_demo.log` / `train_lf_uncertainty.log`

---

*Last updated: May 12, 2026*