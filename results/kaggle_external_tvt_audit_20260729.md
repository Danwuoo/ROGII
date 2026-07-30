# Kaggle external TVT／physics notebook audit — 2026-07-29

## Decision

The `nina2025/rogii-03` files are **competition-test predictions**, not external
train labels and not strict train OOF predictions. They are therefore ineligible
as supervised targets, validation evidence, or learned blend inputs.

The referenced physics notebook is also ineligible as deployable evidence under
the current project contract. Its final low public-LB trajectory uses same-well
train target `TVT`, formation-surface columns, typewell `Geology`, and a
public-score-driven single-well shift. Those operations violate the explicit
inference-feature and no-post-hoc-LB rules.

No Kaggle submission was created or uploaded. The downloaded CSVs remain in
`.runs/external-tvt-kaggle-audit-20260729/dataset` for audit only.

## Sources and captured versions

### Notebook

- URL:
  <https://www.kaggle.com/code/evgendvorkin/roggi-physics-lb-7-872-v48/notebook>
- title shown by Kaggle: `ROGGI Physics LB 7.872 v48!`
- accessed: 2026-07-29
- rendered state: last successful **Version 68 of 70**; Kaggle warns that the
  current version has an error
- rendered successful result asset: Kaggle kernel result id `338646976`
- page-level best score shown: **6.361, V59**
- runtime shown: 13m 40s, GPU T4 ×2
- notebook code license: Apache-2.0
- attached inputs shown by Kaggle include `ROGII - 03`, `ROGII Model Package`,
  `ROGII v10 Fresh Artifacts`, and `koolbox offline`

Apache-2.0 applies to the notebook code. It does not establish a license for the
attached prediction CSVs, model packages, or training artifacts.

### Dataset

- URL: <https://www.kaggle.com/datasets/nina2025/rogii-03>
- owner/title: `nina2025 / ROGII - 03`
- accessed: 2026-07-29
- Kaggle version: **5**
- displayed size: 3.47 MB; zip download shown as 925 kB
- description: none
- license: **Unknown**
- provenance metadata: blank
- update frequency: unspecified

The eight files were downloaded through the public Kaggle dataset endpoint.
Every file has exactly two columns, `id,tvt`, 14,151 finite rows, unique IDs,
and exact ID set/order equality with the competition `sample_submission.csv`.
This proves that they are predictions for the three competition test wells, not
additional labeled wells.

| File | Bytes | SHA-256 |
|---|---:|---|
| `10.142.csv` | 464,140 | `f7963037b2871b4d040bd6e60ef4b5b69143189235e4f7609cabed11a09bdc6f` |
| `11.284.csv` | 424,186 | `d2301bcb686ef7ab2be972b7e6ad668760ad161b2ae0cf5243d8b66685a4daad` |
| `11.338.csv` | 463,115 | `0a4795ab063f0430afe05980b7726a90843d168522c6a7d8ba003a6a42ae92a8` |
| `9.349.csv` | 426,502 | `99c20ec9cab0df9446953296a2e9cd08f462dce88e4e3b53fdc30ff51d4b8ddb` |
| `9.537.csv` | 426,376 | `3bb5e16cb0162c8a9f6c5be084cfc307e845b0691c5578cc5c1ace14c1b7538f` |
| `9.571.csv` | 338,478 | `3db4e46ad037fa5981117cbf2189cdf8e55fafec5dc67babfd174970c8a55dfc` |
| `9.765.csv` | 464,132 | `35436771bc1cb180bb859a01bdb7018e948f57a31750e98f3caaa4db42d39daf` |
| `9.956.csv` | 464,186 | `da8fb54ee3dee151742a60e9954a7673fca2140bd88e0c1ed5825016397ea99c` |

The names resemble public-LB scores, but the dataset provides no description or
lineage that proves which model or notebook generated each file.

## Target-free prediction analysis

Across the eight external files:

- raw row correlations: **0.999928–0.999998**;
- correlations after removing each well's mean: **0.966394–0.998515**;
- pairwise prediction RMSE: **0.796548–3.664578 ft**.

They should therefore be treated as one highly related prediction family, not
eight independent teachers.

The incumbent visible-test prediction was independently reconstructed from all
25 Heel-Registered checkpoints. Its float64 prediction digest exactly matched
the prior inference audit:

`08d1a700bb398a576fa12c029a339bc55077e47f99afe6a3313e4d528055d9b8`

Against that incumbent prediction, the eight files have:

| External file | raw corr | within-well centered corr | prediction-difference RMSE |
|---|---:|---:|---:|
| `10.142` | 0.999447 | 0.827459 | 9.848184 |
| `11.284` | 0.999396 | 0.825751 | 10.094835 |
| `11.338` | 0.999454 | 0.785065 | 9.659474 |
| `9.349` | 0.999595 | 0.808204 | 8.510063 |
| `9.537` | 0.999576 | 0.821116 | 8.883013 |
| `9.571` | 0.999546 | 0.791570 | 8.889824 |
| `9.765` | 0.999658 | 0.841790 | 7.957684 |
| `9.956` | 0.999449 | 0.779749 | 9.772880 |

The raw correlation is dominated by the three wells' absolute TVT levels.
Within-well centered correlations show potentially material slope/shape
diversity relative to the incumbent. That is useful hypothesis evidence only:
without corresponding strict train OOF predictions or hidden labels, there is no
clean way to estimate a blend weight or prove that the difference is beneficial.

## Notebook leakage and public-LB dependence

The rendered successful notebook contains 53 code cells. A complete scan found
no reference to `rogii-03`, `nina2025`, or any of the eight filenames. The
dataset is attached to the notebook but is not consumed by the rendered
successful source.

The notebook's final scoring path instead enables
`RUN_GUARDED_OVERLAP_OVERRIDE=True` and:

1. detects that all three test well IDs also have train copies;
2. reads the same-well train horizontal file, including target `TVT` and
   formation-surface columns;
3. reads train typewell `Geology` and `TVT`;
4. constructs a formation-contact trajectory, calibrates its offset using train
   target `TVT`, checks it against visible test `TVT_input`, and interpolates it
   over the test rows;
5. overwrites every prediction row.

The recorded execution says:

- candidate same-well copies: **3/3**;
- known-prefix reconstruction RMSE: 0.0101, 0.0090, and 0.0079 ft;
- rows overridden: 3,836 + 6,014 + 4,301 = **14,151/14,151**.

This is direct use of forbidden target and formation/geology information. The
visible-prefix check does not make the train-target reconstruction
inference-safe.

The later `Q0522` layer also records:

- `source_public_score=6.568`;
- target well `00e12e8b`;
- an additional fixed `+0.522 ft` shift on all 4,301 rows of that well.

That is explicit public-leaderboard-driven post-hoc test-well tuning and cannot
be used as OOF or private-safe evidence.

The notebook also loads external saved boosters/artifacts and uses formation
surfaces in its learned feature stack. Their reported OOF scores cannot be
transferred into the current evidence base without the original fold
predictions, training provenance, allowed-feature proof, and exact-row
independent audit.

## What can be salvaged safely

The notebook includes particle-filter and beam-search components that can be
isolated to the allowed columns:

- horizontal `MD, Z, GR, TVT_input`;
- typewell `TVT, GR`.

That representation directly targets the current slope/smooth-shape
bottleneck. A clean follow-up would reimplement only this subset, delete all
same-well target/contact/formation logic, and evaluate it from scratch using
well-level strict OOF. Hard-coded selectors or weights exposed to public-LB
iterations must not be copied as confirmatory settings; they need inner-OOF
learning or fixed predeclared values.

The eight external test predictions may remain as target-free diversity
diagnostics. They should not enter training, weight fitting, conditional routing,
or a submission unless all of the following become available:

- a license permitting reuse;
- source model/code/checkpoints and full input provenance;
- exact strict train OOF predictions on the same keys;
- proof that every outer-well prediction was generated without that well's
  labels or learned statistics;
- a nested/cross-fitted blend evaluation independent of public-LB scores.

