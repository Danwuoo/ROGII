# Sparse post-PS TVT Anchor VOI — final research summary

The full report is
`.runs/sparse-anchor-voi-full/report.md`; the authoritative information curve
is `.runs/sparse-anchor-voi-full/information_curve.csv`.

## Conclusion

Eight uniform-MD exact TVT anchors are the first successful count on the fixed
`{1,2,4,8}` grid. Uniform piecewise correction reaches full/non-anchor RMSE
`3.078853/3.081372`; the outer-train-only conditional Path-PCA MVP reaches
`3.004626/3.007084`. Both 10,000-well-cluster bootstrap upper 97.5% candidate
RMSE values are below 4 (`3.217953` and `3.143583`).

Disagreement placement fails to reach 4 at k=8 (`4.815486` piecewise,
`4.791693` conditional). P3–Path disagreement is therefore not a useful active
acquisition policy here.

All catastrophic ≥40/≥60 wells are removed under uniform k=8. The remaining
error is concentrated after the last anchor: conditional Path-PCA is about
`2.11–2.51` between anchors but `6.20` after the final one, and several residual
worst wells reach `19–27` in that terminal segment.

This is robust counterfactual target-bearing evidence, not deployable
competition evidence. It is not submission-eligible, and no Kaggle submission
was created or uploaded.

## Status flags

- `deployable: false`
- `submission_eligible: false`
- `counterfactual_information_experiment: true`
- `uses_validation_target_as_revealed_anchor: true`

## Verification

- fixed fold: `priority1_5fold_v1`
- fold SHA256:
  `7F1772817A95277696EB32085DA7618E2671E3ECE49F0DB00FFE1BC03C07C72E`
- OOF: 773 wells / 3,783,989 rows
- complete tests: `116 passed`
- artifact validation: zero NaNs, zero anchor mismatches, zero k=0 mismatches,
  maximum metric reconstruction error `3.55e-15`

