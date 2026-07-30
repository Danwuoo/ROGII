# Prefix-Calibrated Multiscale Factor-Graph Inversion — final Stage 0 result

**Stage 0 observability failed；已停止，不值得繼續。**

Final decision:

```text
stop_rogii_modeling_no_competitive_signal
```

沒有建立完整 factor-graph predictor、沒有執行 canonical 100-well quick、
沒有執行 773-well full training，也沒有建立或上傳 Kaggle submission。

## Audited scope

- Stage 0 population: all 773 training wells, fold counts 155/155/155/154/154.
- Prediction-time horizontal view: `MD, X, Y, Z, GR, TVT_input`.
- Prediction-time typewell view: `TVT, GR`.
- Stage 0 calibration reads only the contiguous finite `TVT_input` prefix.
- Target `TVT`, formation surfaces, `Geology`, PNG, `rogii-03` predictions, and
  visible-test labels were not read by the predictor.
- Locked config SHA-256:
  `634245757476f4020424572a8f425a23e6c8c2c6721f3ba6f2f76d03bc781282`.
- Per-well artifact SHA-256:
  `dad22797bfb8d6603f9947d71344408a5a7658e4add92720e30b99d7091c4be7`.

The config was locked before Stage 0: 1 ft resampling, lag
`[-8,-6,-4,-2,0,2,4,6,8]` ft, stretch `[0.98,1.00,1.02]`, fixed 5/15 ft
smooth channels, derivative/contrast/polarity channels, held-out prefix blocks,
and 10,000 paired well bootstraps. No parameters were changed after observing
the 773-well result.

## Stage 0 primary result

Lower likelihood cost is better.

| Pairing | Mean cost | Correct wins / ties / losses | Strict correct-win fraction | correct − control mean | 95% paired well-bootstrap CI |
|---|---:|---:|---:|---:|---:|
| correct | 0.392188 | — | — | — | — |
| nuisance-matched wrong typewell | 0.405900 | 100 / 615 / 58 | **12.94%** | -0.013713 | [-0.018180, -0.009653] |
| phase-destroyed assigned GR | 0.604702 | 748 / 0 / 25 | 96.77% | -0.212514 | [-0.220606, -0.204698] |
| horizontal prefix GR shuffle | 0.440873 | 498 / 0 / 275 | 64.42% | -0.048685 | [-0.058899, -0.038367] |

The required correct-vs-wrong strict win fraction was at least 65%. Actual was
12.94%, because 615/773 wells tied exactly. Even after removing all ties,
correct wins only 63.29% of the 158 non-tied wells, still below 65%.

The ties are not merely duplicate input files: the 773 assigned typewells have
752 distinct full `(TVT,GR)` hashes, and only 34/615 exact cost ties use a wrong
control with the same full typewell hash. The locked prefix likelihood is
therefore unable to identify the assigned typewell for most wells.

Mean correct coherence was 0.723760, mean reliability 0.757959, and median
held-out effective rows 240. Correct lag/stretch touched a configured search
boundary in 15.65% of wells.

## Nuisance-only check

- Nuisance-only baseline preferred correct for 51.36% of wells, below 65%.
- Mean correct-minus-wrong nuisance cost: -0.010007.
- Residualized correct-minus-wrong likelihood intercept at matched nuisance:
  -0.011380; 95% bootstrap CI `[-0.015633, -0.006946]`.
- Nuisance/likelihood delta correlation: 0.2785.

Thus the small negative mean cost margin is not explained solely by TVT range,
row count, missingness, or GR amplitude. It nevertheless fails the primary
per-well identification requirement by a large margin and cannot support a
geological-correspondence claim.

Prefix-length and GR-missingness quartiles all have median wrong-minus-correct
gain exactly zero. Strict correct-win fractions by prefix-length quartile are
11.86%, 12.44%, 9.84%, and 17.62%; by missingness quartile they are 13.92%,
9.84%, 15.03%, and 12.95%. The negative conclusion is not caused only by short
or high-missingness wells.

## Stage 0 locked gate

| Check | Result |
|---|---|
| all 773 wells audited, zero failures | PASS |
| correct beats wrong in at least 65% of wells | **FAIL** |
| correct-minus-wrong CI upper < 0 | PASS |
| correct-minus-phase CI upper < 0 | PASS |
| correct-minus-shuffle CI upper < 0 | PASS |
| nuisance-residualized CI upper < 0 | PASS |
| nuisance-only correct preference < 65% | PASS |

One required condition failed, so Stage 0 failed. Per the locked protocol, no
second similarity metric, window, lag, stretch, threshold, or fallback was
tried on this cohort.

## Canonical quick and prediction metrics

The exact matched quick incumbent remains **12.8117468534 ft** on 100 wells /
489,284 rows. The correct factor-graph candidate was not built or scored, so
its RMSE, paired delta/CI, fold stability, per-well wins, common/slope/shape,
terminal-anchor, tail, catastrophic counts, residual correlation,
observability-vs-improvement relationship, path-boundary hits, and observation
path influence are all **N.A. / not run**.

The quick gate is fail-closed at its Stage 0 prerequisite:

| Locked quick condition | Status |
|---|---|
| 1. improvement vs incumbent ≥1.50 ft | FAIL-CLOSED / N.A. |
| 2. candidate RMSE ≤11.3117468534 | FAIL-CLOSED / N.A. |
| 3. paired bootstrap CI upper <0 | FAIL-CLOSED / N.A. |
| 4. improve ≥4/5 folds | FAIL-CLOSED / N.A. |
| 5. per-well wins ≥60/100 | FAIL-CLOSED / N.A. |
| 6. ≥2 of common/slope/centered-shape improve | FAIL-CLOSED / N.A. |
| 7. terminal-anchor RMSE not worse | FAIL-CLOSED / N.A. |
| 8. p95 worsening ≤0.5 ft | FAIL-CLOSED / N.A. |
| 9. max well-RMSE worsening ≤3 ft | FAIL-CLOSED / N.A. |
| 10. ≥40/≥60 counts do not increase | FAIL-CLOSED / N.A. |
| 11. correct beats every control ≥0.50 ft | FAIL-CLOSED / N.A. |
| 12. phase/shuffle remove ≥50% of gain | FAIL-CLOSED / N.A. |
| 13. finite/coverage/keys/folds/manifest | prediction N.A.; Stage 0 safety PASS |

The CLI prediction stage was explicitly tested and refused to run with:
`Stage 0 did not pass; full inverse prediction is fail-closed`.

## Tests and independent audit

- Focused inverse tests: 11 passed.
- Full repository suite in the existing pinned TTM venv: **474 passed,
  0 failed**, 37 non-fatal warnings, 209.58 s on the final run.
- The system interpreter initially produced three TTM-only dependency failures
  because `granite-tsfm` was absent; all three passed in the already-audited
  `.runs/ttm-r2-transfer-probe-20260729/.venv`, and the complete suite was
  rerun there.
- Cold Stage 0: 145.22 s.
- Exact resume: 10.42 s, 773 checkpoints, zero prediction/metric drift.
- Independent disk audit: passed; checkpoints, config, per-well hash, gate,
  target-free schema, control preservation, and current data manifest verified.
- Five-well CPU smoke: passed in 4.06 s.
- Visible-test offline smoke: passed in 0.68 s; 3 wells dynamically enumerated,
  14,151 sample IDs validated, labels not read, no submission written.
- CUDA was not available; deterministic CPU execution was used.
- Cold-run peak telemetry was unavailable before the Windows fallback was
  added. The exact-resume process, which repeats the full typewell load and
  artifact reconstruction, recorded 212.14 MB peak working set.

## Data immutability

Before and after manifests are identical:

```json
{
  "file_count": 2327,
  "total_bytes": 1325112684,
  "name_size_mtime_sha256": "dec44ac09997b60e900c70064349f0fc9a1d558a9cda5c05aaa3530902fe4514"
}
```

## Version note and final recommendation

External research memory `1.0.0 / archive v9` still reports Flat 15.90985 and
P3 15.45255. Those records predate the audited local production incumbent
14.1708478925 and are not comparable to this matched Stage 0 artifact. The
latest local artifacts were used.

Archive the ROGII modeling research. Retain the production Heel-Registered
Sparse-Anchor Student ensemble and the reusable audit/research assets, but do
not pursue another model pivot, full training, nested ensemble, or submission.

Primary artifacts:

- `.runs/prefix-calibrated-inverse-probe-20260729/stage0_gate.json`
- `.runs/prefix-calibrated-inverse-probe-20260729/stage0_diagnostics.json`
- `.runs/prefix-calibrated-inverse-probe-20260729/summary.json`
- `.runs/prefix-calibrated-inverse-probe-20260729/independent_audit.json`
- `.runs/prefix-calibrated-inverse-probe-20260729/resume_verification.json`
- `.runs/prefix-calibrated-inverse-probe-20260729/visible_test_offline_smoke.json`
