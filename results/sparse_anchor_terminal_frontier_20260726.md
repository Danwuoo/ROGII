# Terminal-aware sparse TVT-anchor frontier — final summary

The authoritative report is
`.runs/sparse-anchor-terminal-frontier-full/report.md`; the complete grid is
`.runs/sparse-anchor-terminal-frontier-full/information_frontier.csv`.

## Result

Fewer than eight exact TVT anchors are sufficient only for retrospective
reconstruction:

- acausal near-terminal k=5 (`[.2,.4,.6,.8,.975]`):
  full/non-anchor `3.989061/3.991099`;
- causal: no passing point through k=8; best is uniform k=8 at
  `5.485900/5.490389`.

Near-terminal 97.5% is sufficient and beats TD at equal k. TD first passes at
k=6 (`3.567297`), while near-terminal passes at k=5. The successful k=5 result
is marginal: its well-bootstrap q97.5 is 4.134912. Near-terminal k=6 is the
margin-bearing reference at 3.432134 with q97.5 3.548700.

Conditional Path-PCA does not improve any selected Pareto point and was not
expanded. At acausal near-terminal k=5 it changes 3.989061 to 3.998216 and
pushes non-anchor RMSE just above 4.

The next justified experiment is isolated measurement-noise robustness with
`sigma={0.25,0.5,1,2 ft}` on locked near-terminal k=5 and near-terminal/TD k=6
placements. This conclusion remains target-bearing, non-deployable, and
non-submission-eligible.

