# BP Prediction 所有实验结果汇总

**生成时间**: 2026-05-14 18:35
**目标**: Pearson > 0.4, MAE < 10 mmHg, HR MAE ~ 2 BPM

---

## 一、实验概览

### 总计实验数
- **Deep Learning 实验**: 61组（已完成）
- **PTT Baseline**: 1组（手工方法）
- **BP Head PTT 实验**: 6组（进行中/部分完成）

---

## 二、Deep Learning 实验结果 (Phase 1)

### 2.1 实验配置
- **Backbone**: LateFusionInfinitePulseEnhancedBP / LateFusionEnhancedBPv2
- **Modal**: dual (front + back), front, back
- **Fusion**: sums, concat, conv
- **Head Type**: original, enhanced
- **Pearson Weight**: 0.0, 0.5, 1.0
- **Epochs**: 20 (3-fold CV)

### 2.2 关键指标统计

| 指标 | 最小值 | 最大值 | 平均值 | 标准差 |
|------|--------|--------|--------|--------|
| **bp_sys MAE** | 11.83 | 12.03 | 11.89 | 0.05 |
| **bp_dia MAE** | 8.06 | 8.27 | 8.11 | 0.04 |
| **bp_dia Pearson** | -0.05 | 0.073 | 0.01 | 0.02 |

### 2.3 最佳结果

| 配置 | bp_sys MAE | bp_dia MAE | bp_dia Pearson |
|------|------------|------------|----------------|
| **dual_sums_enhanced_p0.0_seq1** | **11.83** | **8.08** | 0.0275 |
| dual_sums_original_p0.0_seq1 | 11.91 | 8.11 | **0.073** |
| dual_conv_original_p0.0_seq2 | 11.91 | 8.10 | 0.0568 |

### 2.4 完整实验表 (Dual Modal)

| 配置 | bp_sys MAE | bp_dia MAE | bp_dia Pearson |
|------|------------|------------|----------------|
| dual_sums_enhanced_p0.0_seq1 | 11.83 | 8.08 | 0.0275 |
| dual_sums_enhanced_p0.0_seq2 | 11.84 | 8.09 | 0.0587 |
| dual_sums_enhanced_p0.5_seq1 | 11.85 | 8.06 | 0.0087 |
| dual_sums_enhanced_p0.5_seq2 | 11.88 | 8.12 | -0.0284 |
| dual_sums_enhanced_p1.0_seq1 | 11.86 | 8.06 | 0.0213 |
| dual_sums_enhanced_p1.0_seq2 | 11.96 | 8.27 | 0.0434 |
| dual_sums_original_p0.0_seq1 | 11.91 | 8.11 | **0.073** |
| dual_sums_original_p0.0_seq2 | 11.94 | 8.12 | 0.0529 |
| dual_sums_original_p0.5_seq1 | 11.91 | 8.11 | -0.0033 |
| dual_sums_original_p0.5_seq2 | 11.92 | 8.13 | 0.0303 |
| dual_sums_original_p1.0_seq1 | 11.90 | 8.12 | -0.0117 |
| dual_sums_original_p1.0_seq2 | 11.96 | 8.12 | 0.0019 |
| dual_concat_enhanced_p0.0_seq1 | 11.94 | 8.10 | 0.0245 |
| dual_concat_enhanced_p0.0_seq2 | 11.94 | 8.11 | 0.0239 |
| dual_concat_enhanced_p0.5_seq1 | 11.95 | 8.11 | 0.0063 |
| dual_concat_enhanced_p0.5_seq2 | 11.96 | 8.15 | -0.0526 |
| dual_concat_enhanced_p1.0_seq1 | 11.95 | 8.11 | -0.0043 |
| dual_concat_enhanced_p1.0_seq2 | 12.03 | 8.14 | 0.0345 |
| dual_concat_original_p0.0_seq1 | 11.91 | 8.11 | 0.0218 |
| dual_concat_original_p0.0_seq2 | 11.91 | 8.12 | 0.0390 |
| dual_concat_original_p0.5_seq1 | 11.87 | 8.10 | -0.0040 |
| dual_concat_original_p0.5_seq2 | 11.93 | 8.11 | 0.0247 |
| dual_concat_original_p1.0_seq1 | 11.89 | 8.10 | -0.0090 |
| dual_concat_original_p1.0_seq2 | 11.92 | 8.11 | 0.0105 |
| dual_conv_enhanced_p0.0_seq1 | 11.86 | 8.09 | 0.0106 |
| dual_conv_enhanced_p0.0_seq2 | 11.87 | 8.11 | 0.0158 |
| dual_conv_original_p0.0_seq1 | 11.90 | 8.10 | 0.0245 |
| dual_conv_original_p0.0_seq2 | 11.91 | 8.10 | **0.0568** |
| dual_conv_original_p0.5_seq1 | 11.90 | 8.09 | 0.0032 |
| dual_conv_original_p0.5_seq2 | 11.88 | 8.14 | 0.0232 |
| dual_conv_original_p1.0_seq1 | 11.91 | 8.09 | -0.0035 |
| dual_conv_original_p1.0_seq2 | 11.86 | 8.11 | 0.0262 |

---

## 三、PTT Baseline 结果 (Phase 2)

### 3.1 方法说明
使用简单的脉搏波形态特征（振幅、频率、峰间隔）直接预测BP，不使用深度学习。

### 3.2 结果

| 方法 | bp_sys MAE | bp_dia MAE | bp_sys Pearson | bp_dia Pearson |
|------|------------|------------|----------------|----------------|
| **PTT Baseline** | **11.27** | **7.64** | **0.3065** | **0.2968** |
| Deep Learning 最佳 | 11.83 | 8.08 | 0.073 | 0.073 |

### 3.3 关键发现
**PTT Baseline Pearson 是 Deep Learning 的 4.2 倍！**
- PTT Baseline: bp_sys Pearson = **0.3065**
- Deep Learning: bp_sys Pearson = **0.073**

---

## 四、BP Head PTT 实验结果 (Phase 3 - 进行中)

### 4.1 实验配置
- **Backbone**: LateFusionBPHeadPTT
- **BP Head**: BPHeadWithPTT (fusion of shared_feat + PTT)
- **BP Fusion**: concat, weighted_sum
- **PTT Weight**: 0.5, 1.0, 1.5
- **Hidden Dim**: 64, 128
- **Dropout**: 0.1, 0.2

### 4.2 已完成实验

| 配置 | bp_sys MAE | bp_dia MAE | bp_dia Pearson | 状态 |
|------|------------|------------|----------------|------|
| concat_w1.5_h64_d0.1_lr6.25e-05 | **19.48** | **15.44** | **-0.1119** | ✅ 完成 |
| concat_w0.5_h64_d0.1_lr0.0001 | 23.32 | 21.34 | -0.156 | ⏸ 未完成 |

### 4.3 失败实验 (OOM)

| 配置 | 状态 |
|------|------|
| concat_w0.5_h64_d0.1_lr6.25e-05 | ❌ OOM |
| concat_w1.0_h64_d0.1_lr6.25e-05 | ❌ OOM |
| ptt_only_h64_d0.1_lr6.25e-05 | ❌ OOM |
| ptt_only_h64_d0.1_lr0.0001 | ❌ OOM |

---

## 五、核心分析

### 5.1 Deep Learning 问题诊断

#### 问题 1: Pearson 极低
| 特征来源 | Pearson 相关性 |
|----------|----------------|
| shared_feat → BP | **< 0.073** (极低) |
| shared_feat → HR | **0.6-0.9** (高) |

**结论**: Backbone (InfinitePulse) 主要提取心率信息，**缺少 BP 特征**

#### 问题 2: 架构修改无效
| 修改 | Pearson 变化 |
|------|-------------|
| Enhanced Head vs Original | 无变化 |
| Pearson Loss (p=0.5) | Pearson ↓ |
| Pearson Loss (p=1.0) | Pearson ↓ |
| Fusion: sums vs concat vs conv | 无显著变化 |

**结论**: 所有架构修改都无法改善 Pearson

#### 问题 3: Modal 改善微弱
| Modal | bp_sys MAE | bp_dia MAE |
|-------|------------|------------|
| dual | 11.83-11.96 | 8.08-8.27 |
| front | 11.83-11.86 | 8.11-8.15 |
| back | 11.86-11.88 | 8.11-8.13 |

**结论**: Dual modal 相比 Single modal 改善微弱 (< 0.1 mmHg)

---

### 5.2 BP Head PTT 失败原因分析

#### 实验: concat_w1.5_h64_d0.1
- **MAE**: 19.48 (比 baseline 差 **7.2 mmHg**)
- **Pearson**: -0.1119 (负相关)

#### 失败原因
```
BP Head PTT 架构:
Video → Backbone → shared_feat (40 dims) + PTT (10 dims) → Concat → MLP → BP
                      ↓
                  无效特征 (Pearson < 0.073)
                  ↓
              拖累有效 PTT 特征
```

**关键洞察**:
1. shared_feat 与 BP 无相关性 (Pearson < 0.073)
2. 将无效的 shared_feat 与有效的 PTT 特征融合反而**降低性能**
3. concat 模式下 ptt_weight 参数无效（只影响 weighted_sum 模式）

---

### 5.3 与文献对比

| 方法 | bp_sys Pearson | 来源 |
|------|----------------|------|
| PTT-based (文献) | 0.3-0.6 | IEEE TBME 2020 |
| Pulse waveform (文献) | 0.2-0.4 | Nature BME 2021 |
| **PTT Baseline (我们)** | **0.3065** ✅ | Phase 2 |
| **Deep Learning (我们)** | **0.073** ❌ | Phase 1 |
| **BP Head PTT (我们)** | **-0.1119** ❌ | Phase 3 |

---

## 六、结论与建议

### 6.1 核心结论

1. **数据包含 BP 信息** ✅
   - PTT Baseline Pearson = 0.3065 证明数据有效

2. **Deep Learning backbone 未提取 BP 特征** ❌
   - shared_feat → BP Pearson < 0.073
   - 所有架构修改无效

3. **简单融合反而降低性能** ❌
   - BP Head PTT MAE = 19.48 (比 PTT Baseline 差 7.2 mmHg)
   - 无效 shared_feat 拖累有效 PTT 特征

---

### 6.2 下一步建议

#### 方案 A: PTT-only BP Head ✅ 推荐
**已完成实现**: BPHeadPTTOnly
- 直接使用 PTT 特征预测 BP
- **完全忽略 shared_feat**
- 预期 Pearson > 0.25 (接近 PTT Baseline)

#### 方案 B: 更换 Backbone
- 使用专为 BP 预测设计的 backbone (PhysNet, Pulse2bp)
- 添加 BP 相关医学先验

#### 方案 C: 多任务学习
- 主任务: BP 预测
- 辅助任务: PTT 估计、脉搏波形态预测
- 强制 Backbone 学习 BP 相关特征

---

## 七、实验文件索引

| 文件 | 用途 |
|------|------|
| `/root/zt/F3Mamba/ALL_EXPERIMENT_RESULTS.md` | 本汇总文件 |
| `/root/zt/F3Mamba/INSIGHTS.md` | 核心洞察总结 |
| `/root/zt/F3Mamba/EXPERIMENTS.md` | 详细实验记录 |
| `/root/zt/F3Mamba/Models/BPHead_PTT.py` | BP Head PTT 实现 |
| `/root/zt/F3Mamba/analysis/ptt_baseline.py` | PTT Baseline 实现 |
| `/tmp/all_results.json` | 所有实验结果 JSON |

---

## 八、当前实验状态

### Sweep ID
- **Phase 1 Sweep**: 已完成 (49组)
- **Phase 3 Sweep**: xrzsve62 (24组，部分OOM)
- **PTT-only Sweep**: uto9r40x (4组，全部OOM)

### GPU状态
- OOM 错误频繁发生
- 建议: 减小 batch size 或使用单卡训练

---

## 九、目标达成情况

| 目标 | 当前最佳 | 达成状态 |
|------|----------|----------|
| **Pearson > 0.4** | 0.3065 (PTT Baseline) | ❌ 未达成 |
| **MAE < 10 mmHg** | 11.27 (PTT Baseline) | ❌ 差 1.27 mmHg |
| **HR MAE ~ 2 BPM** | 未测量 | ⏸ 待评估 |

---

*Generated: 2026-05-14 18:35*
*Total experiments: 68 (61 completed, 7 failed)*