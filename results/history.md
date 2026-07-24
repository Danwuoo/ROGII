# Historical results

This is the only retained in-repository summary of the pre-reset research
program. The authoritative source at reset time was compiled research memory
version `1.0.0`, archive `v9`, queried with `Worktree reads: 0`.

Absolute scores from different `comparability_group` values are not comparable.
The old STOP/freeze decisions below document history; they are not development
gates in the new open workflow.

| Route | Group | Primary result | Historical decision | Active code |
|---|---|---:|---|---|
| Flat last-known | `full_773_main_v1` | 15.909852870734552 | Retain baseline | Yes |
| P2 residual GBDT | `full_773_main_and_stress_v1` | 15.5970165777 | Not promoted | No |
| P3 alignment | `full_773_main_and_stress_v1` | 15.4525482195 | Frozen comparator; main CI crossed zero | Yes |
| Spatial broad confirmatory | `spatial_confirmatory_743_v1` | 16.424538984 | Stop | No |
| Change-point ranker | `change_point_discovery_60_v1` | 17.104397 | Stop | No |
| Sequence cost volume | `sequence_cost_volume_discovery_60_v1` | 13.9809596795 | Stop; discovery-only and tail-unsafe | No |
| Dense local alignment | `dense_alignment_discovery_60_v1` | 17.0283307957 | Stop | No |
| Audit V2 H01/H06/H11 | separate target-free groups | See CSV | Stop | No |
| Audit V2 TSPulse pilot | `audit_v2_tspulse_target_free_24w_48win_locked_perturbations_v1` | feasibility only | Not promoted | No |

The complete record IDs, decisions and notes are in
[`history.csv`](history.csv). Exact old source, reports and code remain
recoverable from the external legacy Git bundle named in the root README.
