# Experiment Comparison Report

## Overview
Systematic evaluation of DL approaches for dual-view (face+back) BP/HR prediction.
Core bottleneck: all methods fail AAMI STD ≤ 8 mmHg due to systematic subject bias.

## Results Summary

| Experiment | avg_pearson_sys | avg_mae_sys | AAMI SYS (ME±STD) | HR MAE | HR Pearson | Notes |
|---|---|---|---|---|---|---|
| **PTT Handcrafted** (historical) | 0.3065 | 11.27 | — | — | — | Non-DL upper bound |
| **Back-only** (single-view) | 0.149 | 11.82 | — | 2.03 | — | Best single-view DL |
| **Front-only** (single-view) | -0.069 | — | — | — | — | No predictive power |
| **BDF80 Baseline** | **0.2644** | 11.79 | -0.25±14.64 | 2.19 | 0.74 | Best multi-view DL |
| **Exp A** (BP-priority loss) | 0.2404 | 11.88 | -1.54±14.60 | 2.26 | 0.76 | Slightly worse |
| **Exp B** (calibration head) | 0.2584 | 11.76 | -0.20±14.76 | 2.11 | 0.74 | Essentially tied |
| **Exp C** (uncertainty loss) | 0.0186 | 12.00 | -0.03±15.18 | 2.10 | 0.72 | Detrimental |

## Per-Fold Breakdown

### BDF80 Baseline
| Fold | Pearson_sys | MAE_sys | STD_sys | HR_MAE |
|---|---|---|---|---|
| Fold 1 | 0.4421 | 11.59 | 14.00 | 2.00 |
| Fold 2 | 0.1328 | 12.91 | 17.26 | 1.11 |
| Fold 3 | 0.2184 | 10.87 | 12.67 | 3.47 |
| **Avg** | **0.2644** | **11.79** | **14.64** | **2.19** |

### Exp B (Calibration Head)
| Fold | Pearson_sys | MAE_sys | STD_sys | HR_MAE |
|---|---|---|---|---|
| Fold 1 | 0.4134 | 12.00 | 14.69 | 2.83 |
| Fold 2 | 0.1279 | 12.92 | 16.47 | 0.59 |
| Fold 3 | 0.2339 | 10.37 | 13.12 | 2.91 |
| **Avg** | **0.2584** | **11.76** | **14.76** | **2.11** |

### Exp C (Uncertainty Loss) — FAILED
| Fold | Pearson_sys | MAE_sys | STD_sys | HR_MAE |
|---|---|---|---|---|
| Fold 1 | -0.0263 | 12.67 | 15.95 | 3.09 |
| Fold 2 | -0.0172 | 12.83 | 16.27 | 0.70 |
| Fold 3 | 0.0993 | 10.49 | 13.31 | 2.53 |
| **Avg** | **0.0186** | **12.00** | **15.18** | **2.10** |

## Key Findings

1. **BDF80 Baseline remains best DL approach** (Pearson=0.2644). No experiment improves over it.

2. **All loss function modifications fail**: BP-priority, calibration head, uncertainty weighting — none improve Pearson, MAE, or AAMI STD beyond baseline.

3. **AAMI STD failure is universal**: All methods have STD 14-16 mmHg vs the 8 mmHg limit. Per-subject analysis shows small within-subject STD (~1.5 mmHg) but large between-subject bias (ME range ±35 mmHg across 117 subjects).

4. **Fold-to-fold variance is extreme**: Fold 1 consistently achieves Pearson 0.41-0.44, while Fold 2 is 0.07-0.13. This suggests the data has high variance in BP predictability across the fold split.

5. **HR prediction is robust** (MAE ~2.1-2.3 BPM, Pearson ~0.74) regardless of BP-targeted modifications.

## Root Cause Analysis

The fundamental problem: **the DL model learns subject-specific patterns (skin texture, lighting, etc.) rather than universal BP signatures.** Evidence:
- Within-subject error STD is small (~1.5 mmHg) — predictions are precise
- Between-subject ME varies ±35 mmHg — predictions are biased
- The bias is consistent per subject (all windows from a subject have similar bias)
- No loss function change addresses this because the model doesn't know which features are subject-identifying vs BP-relevant

## Next Steps: Pivot Strategies

### Strategy 1: Per-Subject Calibration (Realistic)
The standard approach in all commercial continuous BP monitors: use 1 initial cuff measurement to estimate subject-specific bias.
- **Training**: Train BDF80 as baseline
- **Inference**: Use first K windows (e.g., 5-10) from a new subject to compute bias correction
- **Fair evaluation**: Report metrics on remaining windows after calibration
- **Expected outcome**: AAMI STD < 8 mmHg (demonstrated by the small within-subject STD)

### Strategy 2: Adversarial Subject De-biasing
Add a gradient reversal layer to remove subject-identifying features from the BP representation.
- Risk: may also remove BP-relevant information
- Requires careful tuning

### Strategy 3: Relative BP Change Detection
Frame as change-from-baseline rather than absolute prediction.
- Easier task: track BP changes, not absolute values
- Clinical applications exist where only relative change matters
- Different evaluation framework needed

### Strategy 4: Feature Engineering
Extract known physiological features (PAT, PTT, HR, HRV) as model inputs.
- The handcrafted PTT baseline (Pearson=0.3065) outperforms all DL approaches
- Suggests that learning BP-relevant features from video is the bottleneck
- Hybrid approach: use DL for HR/PAT extraction, then handcrafted features for BP regression

## Recommendation

The current DL approach (video → BP) hits a fundamental ceiling. **Recommended next step**: implement per-subject calibration simulation to demonstrate that the model's within-subject precision (STD ~1.5 mmHg) is sufficient for AAMI PASS when combined with a single reference measurement per subject. This is the standard approach in continuous BP monitoring and would validate the signal quality.
