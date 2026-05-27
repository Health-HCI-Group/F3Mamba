# BP Prediction 实验总结与洞察

**更新时间**: 2026-05-14 18:35

---

## 实验全景

| Phase | 方法 | 实验 | 最佳 MAE | 最佳 Pearson |
|-------|------|------|----------|---------------|
| **P1** | Deep Learning | 61组 | 11.83 | 0.073 |
| **P2** | PTT Baseline | 1组 | **11.27** | **0.3065** |
| **P3** | BP Head PTT | 6组 | 19.48 | -0.11 |

---

## 1. Phase 1: Deep Learning Sweep

### 最佳配置

| 方法 | bp_sys MAE | bp_dia MAE | bp_dia Pearson |
|------|------------|------------|----------------|
| **最佳 MAE** | **11.83** | **8.08** | 0.0275 |
| **最佳 Pearson** | 11.91 | 8.11 | **0.073** |

### 关键发现

1. **Enhanced Head 无效**: 与 Original Head 结果完全相同
2. **Pearson Loss 无效**: 反而降低 Pearson 相关性
3. **Fusion 模式影响微弱**: sums/concat/conv 差异 < 0.1 mmHg
4. **Dual modal 改善微弱**: bp_dia 仅改善 < 0.1 mmHg
5. **所有61组实验 Pearson < 0.08**

---

## 2. P1: shared_feat 分析

### 结论

| 特征 | Pearson 相关性 |
|------|----------------|
| **shared_feat → BP** | < 0.073 (极低) |
| **shared_feat → HR** | 0.6-0.9 (较高) |

**验证假设**：
- ✅ Backbone (InfinitePulse) 主要提取 rPPG 信号（心率信息）
- ❌ Backbone 不包含 BP 相关生理特征

---

## 3. P2: PTT Baseline

### 结果

| 方法 | bp_sys MAE | bp_dia MAE | bp_sys Pearson |
|------|------------|------------|----------------|
| **PTT Baseline** | **11.27** | **7.64** | **0.3065** |
| Deep Learning | 11.83 | 8.08 | 0.073 |

### 关键发现

**PTT Baseline Pearson 是 Deep Learning 的 4 倍！**

使用简单的脉搏波形态特征（振幅、频率、峰间隔等）就能达到：
- bp_sys Pearson: **0.3065**
- bp_dia Pearson: **0.2968**

---

## 4. Phase 3: BP Head PTT 实验 (失败)

### 结果

| 方法 | bp_sys MAE | bp_dia MAE | bp_dia Pearson |
|------|------------|------------|----------------|
| **BP Head PTT (concat)** | **19.48** | **15.44** | **-0.1119** |
| PTT Baseline | 11.27 | 7.64 | 0.3065 |

### 关键发现

**BP Head PTT 比 PTT Baseline 差 7.2 mmHg！**

**失败原因**:
1. shared_feat 与 BP Pearson < 0.073（无效）
2. 将无效 shared_feat 与有效 PTT 特征融合**反而降低性能**
3. concat 模式下 ptt_weight 参数无效

### 新方案: PTT-only BP Head

```
BPHeadPTTOnly: rPPG → PTT features (10 dims) → MLP → BP
                    ↓
               完全忽略 shared_feat
```

**预期**: Pearson > 0.25, MAE < 12 mmHg

---

## 5. 核心洞察

### 4.1 问题根源

```
Deep Learning Pipeline:
Video → Backbone (InfinitePulse) → shared_feat → BP Head → BP 预测
                                        ↓
                                  rPPG 信号 (HR)
                                  ❌ 缺少 BP 特征
```

**Backbone 设计目标**：提取 rPPG 信号用于心率预测
**实际效果**：只提取了心率相关特征，缺少 BP 相关特征

### 4.2 为什么 PTT Baseline 更好？

| 方法 | 特征来源 | BP 相关性 |
|------|----------|-----------|
| **PTT Baseline** | 直接从 rPPG 提取脉搏波形态 | **0.3065** |
| **Deep Learning** | Backbone 提取的 shared_feat | 0.073 |

**原因**：
- PTT Baseline 显式提取 BP 相关特征（脉搏波形态、峰间隔）
- Deep Learning 依赖 Backbone 自动学习，但 Backbone 设计不含 BP 特征

### 4.3 文献对比

| 方法 | 典型 Pearson |
|------|--------------|
| PTT-based (文献) | 0.3-0.6 |
| Pulse waveform (文献) | 0.2-0.4 |
| **PTT Baseline (我们)** | **0.3065** ✅ |
| **Deep Learning (我们)** | **0.073** ❌ |

---

## 5. 建议

### 5.1 立即可行的改进

**方案 A: 端到端重新设计 BP Head**

```python
class BPHeadWithPTT(nn.Module):
    def __init__(self):
        self.shared_feat_proj = nn.Linear(32, 16)  # from backbone
        self.ptt_extractor = PTTFeatureExtractor()  # 从 rPPG 提取 PTT
        self.bp_regressor = nn.Linear(16 + 10, 2)  # 混合特征
    
    def forward(self, shared_feat, rPPG):
        feat1 = self.shared_feat_proj(shared_feat.mean(dim=1))
        feat2 = self.ptt_extractor(rPPG)
        hybrid = torch.cat([feat1, feat2], dim=-1)
        return self.bp_regressor(hybrid)
```

**预期效果**: Pearson > 0.2

---

### 5.2 长期改进

**方案 B: 更换 Backbone**

- 使用专门为 BP 预测设计的 backbone（如 PhysNet, Pulse2bp）
- 添加 BP 相关的医学先验（PTT, PAT）

**方案 C: 多任务学习**

- 主任务: BP 预测
- 辅助任务: PTT 估计、脉搏波形态预测
- 强制 Backbone 学习 BP 相关特征

---

## 6. 实验文件

| 文件 | 用途 |
|------|------|
| `/root/zt/F3Mamba/EXPERIMENT_SUMMARY.md` | Sweep 实验总结 |
| `/root/zt/F3Mamba/EXPERIMENTS.md` | 所有实验记录 |
| `/root/zt/F3Mamba/analysis/ptt_baseline.py` | PTT Baseline 实现 |
| `/root/zt/F3Mamba/analysis/hybrid_method.py` | 混合方法实现 |

---

## 7. 结论

1. **PTT Baseline 证明了数据包含 BP 信息** (Pearson = 0.3065)
2. **Deep Learning backbone 没有有效提取 BP 特征** (Pearson = 0.073)
3. **所有架构修改无效** (Enhanced Head, Pearson Loss, Fusion)
4. **根本解决方案**: 重设计 BP Head 或更换 Backbone

---

*Generated: 2026-05-14*
*PTT Baseline: Pearson = 0.3065 (> target 0.2)*