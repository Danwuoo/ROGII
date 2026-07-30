# Granite TTM R2 pretrained-transfer quick result — 2026-07-29

## Decision

**Quick gate failed. Stop this TTM route: do not run 773-well full, adapter,
scratch-full, nested ensemble, or a retuned repeat on the same cohort.**

The primary contrast is the family-averaged
`ttm_pretrained_frozen - ttm_random_frozen`. Pretraining made pooled row-RMSE
**0.300871 ft worse**, not better. The paired well-cluster bootstrap is
directionally inconclusive and centered on harm; only one of five folds favors
pretraining. Three negative controls outperform the correctly paired pretrained
variant, so the experiment provides no evidence that the representation uses the
intended horizontal/typewell GR correspondence.

No Kaggle submission was created or uploaded.

## Audited scope

- Canonical matched quick cohort: 100 validation wells, 20 per outer fold.
- Exact OOF: 489,284 rows, five well-level folds.
- Variants: random-frozen, pretrained-frozen, wrong-typewell, deterministic
  pair shuffle, horizontal/typewell GR-order shuffle, and phase-destroyed GR.
- Head seeds: `20260729`, `20260730`, `20260731`; family average is formed
  before the primary metrics.
- Identical paired head, initialization seeds, 30 epochs, optimizer, batch
  ordering, preprocessing, residual contract, folds, and evaluation code.
- 90 completed checkpoints; matched frozen trainable parameter count: 35,776.
- OOF SHA-256:
  `b71866ef5171a47262fa279e553bfd49d45af92e6cc87e03d53f699a4d4d4b41`.
- Exact-key SHA-256:
  `c6f76cb328197c37562a2892df494390f8578de365bf5a18865b2649a6cbb07b`.
- Locked config SHA-256:
  `605d23c60e157a3f50c67871c4de2b6d13e7b70864fdcfd39389ab7a91f2ffb5`.

The full-population production incumbent RMSE `14.1708478925` is retained as
the current production reference, but it is not directly compared with the
100-well quick values below because the populations differ. The exact matched
quick incumbent is `12.8117468534`.

## Primary result

| Quantity | Result (ft) |
|---|---:|
| quick incumbent | 12.8117468534 |
| `ttm_random_frozen` family average | 13.9618936941 |
| `ttm_pretrained_frozen` family average | 14.2627649676 |
| pretrained − random | **+0.3008712735** |
| pretrained − quick incumbent | **+1.4510181142** |
| paired transfer 95% CI | **[-0.3673589250, +0.9560855459]** |
| P(pretrained better) / P(pretrained worse) | 18.19% / 81.81% |
| pretrained − incumbent 95% CI | [+0.7472916276, +2.1382062153] |

The locked minimum transfer improvement was 0.05 ft. Actual transfer
improvement, defined as random minus pretrained, is `-0.3008712735` ft.

## Fold and seed stability

| Fold | random-frozen | pretrained-frozen | pretrained − random |
|---:|---:|---:|---:|
| 0 | 13.6042577749 | 14.4327676604 | +0.8285098856 |
| 1 | 12.3118506875 | 12.4918737708 | +0.1800230833 |
| 2 | 12.8544966356 | 12.4649135963 | -0.3895830392 |
| 3 | 15.0977488949 | 15.4352472362 | +0.3374983413 |
| 4 | 15.8541504497 | 16.3903086366 | +0.5361581869 |

Only fold 2 favors pretrained, versus the locked requirement of at least 4/5.

| Seed | random-frozen | pretrained-frozen | pretrained − random |
|---:|---:|---:|---:|
| 20260729 | 15.3690515296 | 15.4030811088 | +0.0340295792 |
| 20260730 | 15.1391524988 | 15.2587078991 | +0.1195554003 |
| 20260731 | 14.8637142240 | 14.8714582335 | +0.0077440095 |

All three matched seeds are directionally worse for pretrained. The
family-average RMSE is lower than each individual head because of seed
ensembling; RMSE nonlinearity means the family-average transfer delta need not
equal the average of the three seed deltas. Complete per-seed/per-fold results
are in `.runs/ttm-r2-transfer-probe-20260729/quick/metrics/per_seed_fold.csv`.

## Negative controls

| Pretrained input condition | pooled RMSE | control − correct pretrained |
|---|---:|---:|
| correct pairing | 14.2627649676 | 0.0000000000 |
| wrong typewell | 13.8783367759 | **-0.3844281917** |
| deterministic pair shuffle | 14.2794610925 | +0.0166961249 |
| GR-order shuffle | 13.8650573218 | **-0.3977076458** |
| phase-destroyed GR | 13.9091450480 | **-0.3536199196** |

There is no positive transfer gain for controls to remove. Wrong-typewell,
GR-order shuffle, and phase destruction all beat correct pairing. Pair shuffle
is only 0.0167 ft worse than correct pairing. Consequently the
pairing-sensitivity gate fails and no geological-correspondence claim is
permitted. All controls are marked `eligible_for_model_selection=false` and
were not ensembled.

## Residual decomposition

| Candidate | common | slope | zero-mean smooth shape | centered | terminal |
|---|---:|---:|---:|---:|---:|
| quick incumbent | 9.898497 | 5.173092 | 6.276921 | 8.133918 | 21.244474 |
| random-frozen | 10.600124 | 5.400217 | 7.308181 | 9.086906 | 21.652541 |
| pretrained-frozen | 11.035243 | 5.449851 | 7.207566 | 9.036032 | 22.169969 |

Relative to random-frozen, pretrained improves zero-mean shape by
`0.1006146893` ft and centered RMSE by `0.0508730542` ft, but worsens common
mode by `0.4351189622` ft, slope by `0.0496336146` ft, and terminal RMSE by
`0.5174273663` ft. The locked slope-or-shape condition passes only because of
the modest shape result; it cannot rescue the failed transfer, standalone,
fold, control, and tail conditions.

The reconstructed residual decomposition relative MSE error is at most
`4.76e-16`. Slope and shape row means, slope/shape orthogonality, finite
corrections, and exact row reprojection were validated.

## Tail result

| Candidate | p50 | p90 | p95 | max | ≥40 | ≥60 | worst-10% SSE share |
|---|---:|---:|---:|---:|---:|---:|---:|
| quick incumbent | 9.995564 | 18.791525 | 21.905939 | 35.011139 | 0 | 0 | 0.396687 |
| random-frozen | 11.302503 | 19.536990 | 22.260654 | 37.482841 | 0 | 0 | 0.371927 |
| pretrained-frozen | 11.338755 | 22.354067 | 25.397582 | 33.772255 | 0 | 0 | 0.372474 |

Pretrained has no ≥40/≥60 catastrophes and improves maximum well RMSE, but its
p95 is `+3.491643` ft worse than the quick incumbent, exceeding the locked
`+1.0` ft allowance. The tail gate therefore fails.

## Locked gate

| Condition | Pass |
|---|---|
| ≥0.05 ft pretrained transfer improvement | No |
| bootstrap direction / probability | No |
| at least 4/5 fold directions | No |
| not materially worse than quick incumbent | No |
| slope or shape improvement | Yes |
| tail safety | No |
| negative-control sensitivity | No |
| hash/offline/fold/key/finite/checkpoint safety | Yes |

Final decision stored on disk:
`stop_ttm_no_full_no_adapter_no_retune`.

## Provenance audit

### Confirmed

- Official repository:
  <https://github.com/ibm-granite/granite-tsfm>, audited detached HEAD
  `31c7a6bd640c41e53777c0afe3a13d24e00f7fe7`.
- Package `granite-tsfm==0.3.6`; Apache-2.0.
- Model card:
  <https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2>.
- Paper: [Tiny Time Mixers, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html)
  ([arXiv 2401.03955](https://arxiv.org/abs/2401.03955)).
- Weight revision
  `d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f`; the 3,240,592-byte
  `model.safetensors` hash is
  `a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710`.
- The r2 model card discloses Australian Electricity Demand, Australian
  Weather, Bitcoin, KDD Cup 2018, London Smart Meters, Saugeen River Flow,
  Solar Power, Sunspots, Solar, US Births, Wind Farms Production, and Wind
  Power. It describes roughly 700M pretraining samples.
- No well log, gamma ray, geology, or this competition is named in that
  disclosed r2 list.
- Package inspection confirms mask-aware internal standard scaling for
  `config.scaling="std"`, an exact same-config random constructor, and
  `backbone(...).last_hidden_state` with shape `[batch, channel, patch,
  d_model]`. The probe extracts `[B,1,8,192]` backbone states and never uses
  forecast predictions as embeddings or TVT predictions.

### Not disclosed

- A complete row/file-level pretraining manifest, immutable source hashes, and
  an exhaustive overlap certificate for all roughly 700M samples are not
  provided at the weight level.
- The disclosed dataset list does not prove that no unlisted intermediate,
  derivative, or mirrored source entered preprocessing.

### Inference

There is no positive disclosed evidence of geology/GR/competition exposure, so
direct domain overlap appears unlikely. This is an inference, not absence
proof. The empirical result is negative regardless: the pinned pretrained
weights do not beat their matched random-frozen control.

No external pretraining dataset, MOMENT Pile, GiftEvalPretrain, or artifact
larger than 5 GB was downloaded.

## Integrity and reproducibility

- Online smoke: passed; CPU fallback, shape match, frozen gradients, head
  gradients, serialization, and exact resume.
- Forced-offline smoke: passed with DNS/socket calls blocked.
- Visible-test offline smoke: three test wells dynamically enumerated, 14,151
  IDs in exact `sample_submission.csv` order, 15 fold/seed predictions finite,
  no labels read, no submission written.
- Independent quick re-audit: passed after reading the 154 MB OOF artifact,
  all 90 checkpoints, and all prediction-block hashes from disk.
- Current data manifest equals before/after manifests.
- Focused TTM tests: 11 passed.
- Full repository suite: 463 passed, 0 failed (37 non-fatal warnings).
- Cold CPU quick: 1,117.48 s (18m 37s); exact resume: 344.12 s (5m 44s).
- Peak observed host working set during cold quick: approximately 2.36 GB.
- No CUDA device was present; CUDA code exists but GPU smoke was skipped rather
  than claimed.

## Small artifacts to hand off

- `results/ttm_pretrained_transfer_probe_quick_20260729.md`
- `.runs/ttm-r2-transfer-probe-20260729/quick/quick_gate.json`
- `.runs/ttm-r2-transfer-probe-20260729/quick/independent_audit.json`
- `.runs/ttm-r2-transfer-probe-20260729/quick/visible_test_offline_smoke.json`
- `.runs/ttm-r2-transfer-probe-20260729/quick/metrics/summary.json`
- `.runs/ttm-r2-transfer-probe-20260729/quick/metrics/per_fold.csv`
- `.runs/ttm-r2-transfer-probe-20260729/quick/metrics/per_seed_fold.csv`

The 154 MB OOF, checkpoint tree, embedding cache, and prediction blocks should
remain in ignored `.runs/` and do not need to be pasted into a chat.
