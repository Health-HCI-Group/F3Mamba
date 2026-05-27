# V3 Pilot Round-1 Summary and Round-2 Launch Plan

Date: 2026-05-16
Project: `F3Mamba`
Sweep IDs:
- V2 reference: `7jwusg45`
- V3 pilot round-1: `vc9w550o`

## 1) Current Snapshot

### V2 reference (`7jwusg45`)

Sweep state: `FINISHED`  
Run states: `finished=14`, `crashed=4`

Best 3 by BP MAE sum (`sys + dia`):

| Run | test/mae_sys | test/mae_dia | test/pearson_sys | test/pearson_dia | test/hr_MAE |
|---|---:|---:|---:|---:|---:|
| t1hr7pk7 | 10.5329 | 7.5004 | -0.1908 | -0.1353 | 2.8083 |
| d26p6tda | 10.5191 | 7.5320 | 0.1239 | 0.2036 | 2.2848 |
| ajprhx3k | 10.5171 | 7.5346 | 0.0550 | 0.1033 | 4.1127 |

### V3 round-1 (`vc9w550o`, completed 6 runs)

Sweep shows `finished=6` (all complete for this batch).

| Run | test/mae_sys | test/mae_dia | test/pearson_sys | test/pearson_dia | test/hr_MAE | eval/align_corr_gain | eval/align_mech_pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| o07bzpsd | 10.5415 | 7.6489 | 0.0353 | -0.0017 | 3.0991 | -0.0907 | 0 |
| 832z620d | 10.5323 | 7.5219 | -0.0664 | 0.1636 | 2.8415 | -0.0479 | 0 |
| 9woc68hs | 10.5101 | 7.5508 | 0.0918 | 0.0432 | 2.5840 | -0.0397 | 0 |
| 8b2ae3e6 | 10.4974 | 7.5095 | 0.1735 | -0.0055 | 4.1958 | -0.1259 | 0 |
| hti1y8w4 | 10.4998 | 7.5812 | 0.0228 | 0.0200 | 3.6059 | -0.0960 | 0 |
| e8th96jb | 10.5216 | 7.5319 | -0.0179 | 0.1851 | 1.8860 | -0.1349 | 0 |

## 2) Interpretation

1. V3 round-1 kept BP MAE in a stable band near V2 reference quality.
2. Best HR in V3 reached `< 2` once (`e8th96jb`), but BP Pearson remains below target.
3. Critical issue: `eval/align_corr_gain` is negative in all 6 runs and `eval/align_mech_pass=0` in all runs.
4. Therefore current alignment policy is active but not yet beneficial by mechanism criteria.

## 3) Round-2 Plan (already prepared in code)

Second-round design goals:
- Reduce harmful over-alignment.
- Only apply shift when it beats zero-shift by a margin.
- Avoid edge shifts that often indicate unstable delay choice.

Implemented controls in `sweep_xmae.py`:
- `delay_max_shift`: narrowed to `[2, 4]`
- `align_blend`: reduced to `[0.05, 0.1, 0.2]`
- `min_corr_improve`: new threshold `[0.0, 0.01]`
- `avoid_edge_shift`: new guard `[True]`
- New explainability metric: `align_apply_frac`

Mechanism acceptance target for round-2:
- At least some runs with `eval/align_corr_gain >= 0`
- At least some runs with `eval/align_mech_pass = 1`

Task acceptance target remains unchanged:
- BP: improve Pearson (especially `sys`) without MAE regression
- HR: keep low MAE behavior
