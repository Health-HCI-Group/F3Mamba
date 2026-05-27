# V1 / V2 / V3 Experiment Status Record

Date: 2026-05-16
Owner: OpenCode + User
Project: `F3Mamba`

## 1) Executive Summary

- V3 has **not** started yet (no V3-tagged run found in W&B; no V3 config keys detected).
- V2 is still actively running in sweep `7jwusg45` (1 running run at check time).
- V2 improves HR a lot vs older V1-style deep learning runs, and improves BP MAE into a tighter band, but target thresholds are still not fully reached.
- V1 (earlier phases) best Pearson was weaker than PTT baseline and generally did not cross the desired BP Pearson target.

---

## 2) Real-time V3 Run Verification (Process + W&B)

Verification time: 2026-05-16 (current session)

### 2.1 Process-side check

Command used:

```bash
pgrep -af "xmae|sweep_xmae|train_3fold"
```

Observed active process names are all `sweep_xmae.py` (V2 sweep continuation path), no dedicated V3 launcher/process signature.

### 2.2 W&B-side check

Checks performed:

1. Sweep `2796321819-beihang-university/F3Mamba/7jwusg45` running status.
2. Project-wide run scan for V3 signatures:
   - `model_version == v3`
   - run name contains `v3`
   - config keys like `delay_align_weight`, `phase_prior_weight`, `delay_tau_max`, `align_mode`

Result:

- `detected_v3_runs = 0`
- Current running run belongs to V2 sweep (`cjhku4y1`, state=`running`).

Conclusion: **V3 is not running yet.**

---

## 3) V1 Baseline Summary (Historical)

Data source:
- `ALL_EXPERIMENT_RESULTS.md`
- `INSIGHTS.md`

### 3.1 Historical scope

- Deep-learning runs (historical phase): 61
- Main observations: BP Pearson remained low in most configurations.

### 3.2 Reported best historical V1-style deep learning numbers

| Category | bp_sys MAE | bp_dia MAE | Pearson (representative) |
|---|---:|---:|---:|
| Best MAE config | 11.83 | 8.08 | bp_dia Pearson 0.0275 |
| Best Pearson config | 11.91 | 8.11 | bp_dia Pearson 0.073 |

### 3.3 Historical PTT baseline reference (for context)

| Method | bp_sys MAE | bp_dia MAE | bp_sys Pearson | bp_dia Pearson |
|---|---:|---:|---:|---:|
| PTT baseline | 11.27 | 7.64 | 0.3065 | 0.2968 |

Interpretation:
- Historical V1-style deep learning did not match PTT baseline Pearson behavior.
- This is one motivation for V2/V3 evolution.

---

## 4) V2 Sweep Status (Current)

Live data source:
- W&B sweep `2796321819-beihang-university/F3Mamba/7jwusg45`

### 4.1 Sweep health

| Item | Value |
|---|---:|
| Total runs | 17 |
| Finished | 13 |
| Crashed | 3 |
| Running | 1 |

### 4.2 Best BP MAE cluster (finished runs)

Sorted by `test/mae_sys + test/mae_dia` (fallback to `eval/*` where needed):

| Run | mae_sys | mae_dia | Sum | pearson_sys | pearson_dia | hr_MAE |
|---|---:|---:|---:|---:|---:|---:|
| t1hr7pk7 | 10.5329 | 7.5004 | 18.0333 | -0.1908 | -0.1353 | 2.8083 |
| d26p6tda | 10.5191 | 7.5320 | 18.0510 | 0.1239 | 0.2036 | 2.2848 |
| ajprhx3k | 10.5171 | 7.5346 | 18.0516 | 0.0550 | 0.1033 | 4.1127 |

### 4.3 Best HR runs (finished)

| Run | hr_MAE | hr_Pearson | mae_sys | mae_dia |
|---|---:|---:|---:|---:|
| d26p6tda | 2.2848 | 0.7862 | 10.5191 | 7.5320 |
| t1hr7pk7 | 2.8083 | 0.6891 | 10.5329 | 7.5004 |
| z0320eou | 3.0991 | 0.7538 | 11.6190 | 7.0959 |

### 4.4 Target-check against current V2 finished runs

Target set:
- `HR MAE < 2`
- `BP sys/dia MAE < 10`
- `BP Pearson > 0.3`

Observed counts:

| Condition | Count |
|---|---:|
| All targets hit together | 0 |
| `hr_MAE < 2` | 0 |
| `mae_sys < 10` and `mae_dia < 10` | 0 |
| `pearson_sys > 0.3` | 0 |
| `pearson_dia > 0.3` | 1 |

Interpretation:
- V2 clearly moved HR closer to target and stabilized MAE near a narrow BP band.
- Main bottleneck is still BP Pearson, especially systolic Pearson.

---

## 5) V3 Readiness and Current Gap

Current state:
- Codebase has V3 insertion boundary (`align_hook`) in `Models/xMAE_Model_V2.py`.
- But no actual V3 run config/model flag is active in W&B.

Gap to close before claiming V3 started:
1. Add explicit V3 config markers (e.g., `model_version=v3` and delay-prior hyperparams).
2. Launch with clean environment (kill existing V2 sweep processes first as requested).
3. Confirm V3 by both:
   - process command line,
   - W&B run config fields and run naming.

---

## 6) References

- `ALL_EXPERIMENT_RESULTS.md`
- `INSIGHTS.md`
- `sweep_xmae.py`
- `Models/xMAE_Model_V2.py`
- W&B sweep: `2796321819-beihang-university/F3Mamba/7jwusg45`
