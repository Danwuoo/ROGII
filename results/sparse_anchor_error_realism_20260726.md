# Sparse-anchor realistic measurement-error robustness — final summary

The authoritative report is
`.runs/sparse-anchor-error-realism-full/report.md`; the complete locked matrix
is `.runs/sparse-anchor-error-realism-full/error_realism_curve.csv`.

## Result

The margin-bearing acausal near-terminal k=6 result remains robust under the
predeclared moderate operational scenario:

- tau=1 ft systematic per-well bias;
- sigma=1 ft independent anchor error;
- 10% gross-error probability;
- symmetric M=5 ft gross-error shock.

Its full/non-anchor realization q95 values are `3.883489 / 3.885136`, and its
hierarchical full/non-anchor q97.5 values are `3.967214 / 3.968956`. All six
predefined gates pass. Mild also passes; severe
`tau=2, sigma=2, p=.10, M=10` fails at about 5.03 hierarchical RMSE.

TD k=6 passes mild but fails moderate and severe. Its moderate full/non-anchor
q95 values are `4.008870 / 4.010624`, and hierarchical q97.5 values are
`4.097155 / 4.098988`.

## Direct conclusions

1. The maximum tested robust common-bias tau is **1 ft** for both near_k6 and
   td_k6. Both fail tau=2.
2. TD has slightly smaller incremental sensitivity to common bias, but no
   larger robust tolerance. Its worse absolute result remains an exact-baseline
   margin disadvantage.
3. Common bias is more dangerous than same-scale independent Gaussian noise.
   At 2 ft, paired median full-RMSE bias-minus-Gaussian deltas are about +0.145
   for both configurations.
4. The first outlier-only failure is `p=.10/M=10` for near_k6 and
   `p=.05/M=10` for td_k6. Near_k6 still passes `p=.20/M=5`.
5. One near-terminal outlier affects two adjacent interpolation intervals and
   about 1,475 prediction rows (~1,474 ft MD) on average. TD averages 1.83
   intervals and ~1,497 rows because its final TD anchor has no post-last
   segment.
6. No RMSE ≥40 or ≥60 catastrophic well returns in any of the 4,600
   config/scenario/realization groups.
7. Operational outcomes:
   - near_k6: mild pass, moderate pass, severe fail;
   - td_k6: mild pass, moderate fail, severe fail.
8. Full and non-anchor conclusions agree at every point; excluding observed
   anchor rows changes RMSE by only about 0.001–0.002 ft.
9. Failure comes mainly from coherent tau=2 common shifts and M=10 shocks
   propagated through between-anchor interpolation. Post-last error is locally
   sensitive but too small in row count to dominate. Difficult wells remain an
   ordinary tail, not a catastrophic regime.
10. Near-terminal k=6 is a credible **retrospective** measurement
    specification at the locked moderate condition, with a narrow hierarchical
    margin (~0.031 ft on non-anchor q97.5). It is still target-bearing, acausal,
    non-deployable, and non-submission-eligible.
11. A separate robust anchor-QC/outlier-rejection experiment is worthwhile if
    10-ft gross errors are plausible. It should not be mixed into this result,
    and it cannot replace datum calibration because coherent tau=2 bias also
    fails.

## Propagation and tails

At near_k6 `p=.10/M=10`, about 87% of outlier-added segment SSE occurs between
anchors: 20.46M added SSE over 3,054,922 rows, versus 2.11M pre-first and
0.94M post-last. At p=.10, realized exposure is 10.04% of anchors; 47.1% of
wells have at least one outlier and 11.4% have multiple outliers.

Moderate near_k6 segment median RMSEs are pre-first 4.065, between 3.836, and
post-last 2.999. The maximum-well RMSE q95 is 9.76. Repeated difficult wells
include `d0cb63ae`, `a959858c`, and `f6d009f4`, but no catastrophic tail
reappears.

## Verification

- fixed fold: `priority1_5fold_v1`;
- OOF scope: 773 wells / 3,783,989 rows;
- 100 deterministic paired realizations, seeds 0–99;
- 10,000 hierarchical bootstrap replicates per config/scenario;
- exact terminal metrics reproduced with pooled metric delta 0;
- 1,000 existing Gaussian metric groups reproduced within
  `1.33e-15`;
- 3,555,800 compact per-well rows reconstruct all 4,600 pooled groups within
  `1.78e-15`;
- zero duplicate keys, unexpected numeric NaNs, row/well coverage failures,
  shared-latent mismatches, or leakage-check failures;
- no formation surfaces, `Geology`, PNG, or typewell data read;
- no writes to `data/`;
- complete tests: `184 passed in 85.66s` with `-W error`;
- final `rogii check`: fixed five folds, 773 wells, and 3,783,989 prediction
  rows;
- no Kaggle submission created or uploaded.

## Evidence status

- `deployable: false`
- `submission_eligible: false`
- `counterfactual_information_experiment: true`
- `uses_validation_target_as_revealed_anchor: true`
