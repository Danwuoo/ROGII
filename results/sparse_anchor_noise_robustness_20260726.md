# Sparse-anchor measurement-noise robustness — final summary

The authoritative report is
`.runs/sparse-anchor-noise-robustness-full/report.md`; the complete curve is
`.runs/sparse-anchor-noise-robustness-full/noise_information_curve.csv`.

## Result

Independent Gaussian TVT measurement noise does not remove the margin-bearing
near-terminal k=6 result over the tested range. It remains robustly below 4 at
the largest tested noise, sigma=2 ft:

- full/non-anchor noise-seed q95: `3.799122 / 3.800805`;
- hierarchical full/non-anchor q97.5: `3.888176 / 3.889914`;
- fold-max q95: `3.888561`;
- wells at RMSE >=40 / >=60: `0 / 0` in every realization.

The exact-anchor k=5 point is not robust under the predeclared definition even
at sigma=0: hierarchical full/non-anchor q97.5 are
`4.136827 / 4.138917`, and fold 0 remains above 4. Its fixed-well
noise-seed q95 first crosses 4 at sigma=0.5. At sigma=0.25 the
full/non-anchor q95 are still `3.997583 / 3.999618`, but only 96 of 100
non-anchor realizations are below 4.

TD k=6 remains robust through sigma=1 ft. At sigma=2 its fixed-well q95 is
still below 4 (`3.919813 / 3.921583`), but hierarchical q97.5 reaches
`4.018872 / 4.020735`, so it fails the complete rule.

## Interpretation

TD is not intrinsically more noise-sensitive than 97.5% placement. From
sigma=0 to 2, mean full RMSE rises by 0.328 ft for TD versus 0.344 ft for
near-terminal. Near-terminal stays better at every paired seed because its
exact-anchor margin is larger. The 97.5% anchor carries its noise through the
last 2.5% of rows; TD noise has no later non-anchor rows.

For near-terminal k=6 at sigma=2, median segment RMSE changes are:

- PS boundary to first anchor: `3.879 -> 4.044`;
- between anchors: `3.369 -> 3.744`;
- after the last anchor: `2.145 -> 2.930`.

Post-last is most sensitive per row, but between-anchor rows contribute about
87% of segment-level added SSE because they dominate row count. No
catastrophic tail returns: every tested config/sigma/seed has zero wells at
RMSE >=40 or >=60, and the largest observed well RMSE is about 13.1.

Full and non-anchor conclusions agree throughout. The first robust anchor count
is k=6; near-terminal is the margin-leading configuration. The `<4` result is
therefore not confined to noiseless exact anchors, although it remains an
acausal, target-bearing completed-path reconstruction diagnostic.

Because near-terminal k=6 remains robust at sigma=1 ft and sigma=2 ft, the next
isolated diagnostic is justified: systematic per-well datum bias, mixed
independent noise, and outlier anchors. No new placement search is implied.

## Verification

- fixed fold: `priority1_5fold_v1`;
- OOF scope: 773 wells / 3,783,989 rows;
- noise realizations: 100 deterministic paired seeds for every sigma;
- hierarchical bootstrap: 10,000 replicates;
- sigma=0 reproduces all three locked metrics exactly and terminal-frontier
  predictions row for row;
- compact artifact: 1,159,500 per-well rows and 1,500 metric groups;
- maximum full/non-anchor metric reconstruction error:
  `1.78e-15 / 1.33e-15`;
- unexpected NaNs and row/well coverage failures: `0 / 0`;
- no formation surfaces, `Geology`, PNG, or typewell data were read;
- no output was written to `data/`.
- complete tests: `152 passed in 71.90s` with `-W error`;
- final `rogii check`: fixed five folds, 773 wells, and 3,783,989 prediction
  rows.

## Evidence status

- `deployable: false`
- `submission_eligible: false`
- `counterfactual_information_experiment: true`
- `uses_validation_target_as_revealed_anchor: true`
- `anchor_measurement_noise_simulated: true`
- Kaggle submission created/uploaded: `false / false`
