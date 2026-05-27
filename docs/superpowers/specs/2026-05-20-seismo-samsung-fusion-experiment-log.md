# Seismo + Samsung Fusion Experiment Log

## Scope
- Goal: improve dual-view BP/HR performance by combining Samsung-style cross-view reconstruction ideas with Seismo-style robustness ideas.
- Phase A/B (alignment mechanism): completed. `align=none` wins.
- Phase reweight: completed. Can push Pearson positive but at MAE/HR cost.
- Phase robust-loss (Huber/SmoothL1): completed. All variants degrade or match MSE control.
- Phase quality-fusion (Seismo-inspired): running. A/B test of quality-weighted fusion vs naive cat+Linear.
- Next: task-decoupled architecture (HR=aligned path, BP=robustness-only path).

## Key Working Hypotheses
1. ~~Samsung-style alignment helps HR more than BP in current setup.~~ CONFIRMED.
2. ~~Loss reweighting alone insufficient.~~ CONFIRMED.
3. ~~Robust loss functions (Huber/SmoothL1) alone insufficient for BP Pearson.~~ CONFIRMED.
4. Quality-weighted fusion should let model down-weight noisy views, improving BP robustness.
5. Task-decoupled architecture (HR uses alignment, BP uses separate robust path) will resolve the feature-task mismatch.

## Decision History
- A/B done (`align_mode=xcorr_soft` vs `align_mode=none`):
  - `align_mode=none` is better for BP aggregate stability.
  - `align_mode=xcorr_soft` can improve HR but does not pass BP mechanism gate.
- Chosen BP baseline: `i5tco8op` (`align_mode=none`, `lr=6.25e-5`, `dropout=0.05`, `bvp_w=0.6`, `recon_w=0.1`)
- Reweight sweep: `74bij491` achieved positive Pearson (+0.0284) but unacceptable MAE/HR tradeoff.
- Robust-loss sweep: all Huber/SmoothL1 variants fail to improve BP Pearson vs MSE control.
- **Pivot**: structural experiments only — quality-weighted fusion first, then task-decoupled architecture.

## Active Sweeps
- Quality fusion A/B: `ldkabo34` (running, 1 of 2 runs complete)

## Completed Run Summary

### A/B phase (mechanism diagnosis)
| Run | Core config | avg_mae_sys | avg_pearson_sys | hr_MAE | hr_Pearson | Notes |
|---|---|---|---:|---:|---:|---:|---|
| `dn0cqgo1` | align=xcorr_soft, max_shift=12, min_corr=0.03 | 12.19 | -0.0352 | 2.20 | 0.798 | Align active but negative corr_gain, BP not improved |
| `l9n1roli` | align=none | 12.03 | -0.0288 | 2.78 | 0.689 | BP better than align-on, HR weaker |

### BP mainline sweep `indz3h4p` (align disabled)
| Run | Core config | avg_mae_sys | avg_pearson_sys | hr_MAE | hr_Pearson | Notes |
|---|---|---|---:|---:|---:|---:|---|
| `tt0vpss5` | lr=6.25e-5, dropout=0.1 | 11.98 | -0.1062 | 2.04 | 0.801 | Best HR, BP correlation weak |
| `8sibvbxk` | lr=1e-4, dropout=0.05 | 11.96 | -0.1370 | 3.09 | 0.679 | HR degraded |
| `zznpho4t` | lr=1e-4, dropout=0.1 | 12.30 | -0.0983 | 3.05 | 0.716 | Worst sys MAE |
| `i5tco8op` | lr=6.25e-5, dropout=0.05 | 11.96 | -0.0368 | 2.84 | 0.761 | Best BP-side balance so far |

### BP reweight sweep `n1m8mczw`
| Run | Core config | avg_mae_sys | avg_pearson_sys | hr_MAE | hr_Pearson | Notes |
|---|---|---|---:|---:|---:|---:|---|
| `bc2isap8` | bvp_w=0.4, recon_w=0.1 | 12.08 | -0.0457 | 3.11 | 0.666 | done |
| `us6e557r` | bvp_w=0.6, recon_w=0.1 | 11.96 | -0.0403 | 4.34 | 0.640 | done |
| `m4vazljh` | bvp_w=0.2, recon_w=0.1 | 11.95 | -0.0762 | 2.92 | 0.679 | done |
| `74bij491` | bvp_w=0.6, recon_w=0.05 | 12.17 | **+0.0284** | 3.66 | 0.658 | First positive Pearson but MAE/HR degraded |

### BP robust-only sweep `nlcgmse0`
| Run | Core config | avg_mae_sys | avg_pearson_sys | Notes |
|---|---|---|---:|---:|---|
| `lgaixg94` | mse, delta=2.0 | 11.95 | -0.1025 | MSE control |
| `sudz86rc` | mse, delta=1.0 | 11.95 | -0.1025 | MSE control |
| `xwz3bw8m` | huber, delta=0.5 | 11.90 | -0.1274 | MAE slightly improved, Pearson degraded |
| `pwcwcmbr` | huber, delta=1.0 | 11.94 | -0.1054 | equivalent to MSE |
| `yz9xv8dh` | huber, delta=2.0 | 11.98 | -0.1350 | worst Pearson |
| `1h11c10f` | smooth_l1, delta=0.5 | 11.94 | -0.1054 | equivalent to MSE |
| `awobo7oi` | smooth_l1, delta=2.0 | 11.94 | -0.1054 | equivalent to MSE |
| `4spxxqjp` | smooth_l1, delta=1.0 | 11.94 | -0.1054 | equivalent to MSE |

**Conclusion**: All Huber/SmoothL1 variants failed to improve BP Pearson vs MSE control. Degrades.

### BP quality-weighted fusion sweep `ldkabo34` (running)
- **Hypothesis**: Seismo's quality-weighted fusion (per-view quality score → softmax-weighted feature fusion) can improve BP by allowing the model to down-weight noisy views
- **Baseline**: i5tco8op (`align_mode=none`, `lr=6.25e-5`, `dropout=0.05`, `bvp_w=0.6`, `recon_w=0.1`)
- **Sweep**: `quality_fusion=[False, True]`
- **Implementation**: Replace `cat + Linear(64→32)` with per-view quality heads (global pooling → 8-dim MLP → sigmoid), softmax-weighted fusion
- **Run progress**:
  | Run | quality_fusion | avg_mae_sys | avg_pearson_sys | Status |
  |---|---|---:|---:|---|
  | `6apvvvcy` | False | - | - | training |
  | TBD | True | - | - | pending |
- **Cost**: 546 additional params (two quality heads)

## Update Protocol
- This file is updated after each run completion with:
  - run id
  - key config
  - `final/avg_mae_sys`
  - `final/avg_pearson_sys`
  - `eval/hr_MAE`, `eval/hr_Pearson`
  - one-line conclusion
