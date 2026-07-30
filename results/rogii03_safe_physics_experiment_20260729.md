# ROGII-03 外部預測與 inference-safe physics alignment 實驗

日期：2026-07-29  
Production comparator：Heel-Registered Sparse-Anchor Student 5-seed ensemble  
Production strict OOF：773 wells、3,783,989 rows、pooled row-RMSE = **14.1708478925**

## Executive decision

**保留 production incumbent。** `rogii-03` 八份檔案仍是 test-only、license
Unknown、沒有 strict train OOF；它們只能標記為 `diagnostic_only`。乾淨重作的
NCC、Particle Filter 與 Beam Search 在固定 100-well quick population 上沒有
通過 full-run gate：raw alignment 全部大幅惡化；cross-fitted Beam/NCC 的小幅
改善只來自 common mode，10,000 次 paired well-bootstrap CI 都跨 0，slope 與
centered-shape 沒有改善，且正確 PF pairing 不優於負控制。因此未執行 773-well
full OOF，也未執行 external teacher、200-well shadow-private simulation 或
Kaggle submission。

本次以 2026-07-28/29 的 audited disk artifacts 為準。可選 research memory
`version 1.0.0 / archive v9` 仍列出舊 flat 15.9098 與 P3 15.4525，和本報告的
同-population incumbent 14.1708478925 不一致；該 memory 沒有被用作 gate，
也沒有被本次實驗改寫。

## 1. `rogii-03` 可證明的來源

### 1.1 Dataset metadata 與版本沿革

[Kaggle dataset `nina2025/rogii-03`](https://www.kaggle.com/datasets/nina2025/rogii-03)
目前為 Version 5，頁面顯示 owner F.A.Nina、license `Unknown`、沒有實質
description、8 files。JSON-LD 的 `dateModified` 是
`2026-06-02T10:04:42.297Z`。逐版比對 Kaggle file tree 得到：

| Dataset version | 日期 | 新增檔案 |
|---|---|---|
| V1 | initial release | `10.142.csv`, `11.284.csv`, `11.338.csv` |
| V2 | 2026-05-11 | `9.956.csv` |
| V3 | 2026-05-13 | `9.537.csv`, `9.765.csv` |
| V4 | 2026-05-15 | `9.571.csv` |
| V5 | 2026-06-02 | `9.349.csv` |

八份檔案都通過本機完整性檢查：

- 每份 14,151 rows，schema 恰為 `id,tvt`。
- 14,151 unique IDs，0 duplicates，0 NaN/Inf。
- ID set 與順序都和動態讀取的 `sample_submission.csv` 完全一致。
- 完整 SHA-256、bytes、版本與 source attribution 固定在
  `configs/external/rogii03_v5_provenance_20260729.json`；本次實驗也在
  `.runs/rogii03-safe-physics-20260729/provenance/` 保存相同 registry。

| 檔案 | bytes | SHA-256 prefix |
|---|---:|---|
| `10.142.csv` | 464,140 | `f7963037b2871b4d` |
| `11.284.csv` | 424,186 | `d2301bcb686ef7ab` |
| `11.338.csv` | 463,115 | `0a4795ab063f0430` |
| `9.349.csv` | 426,502 | `99c20ec9cab0df94` |
| `9.537.csv` | 426,376 | `3bb5e16cb0162c8a` |
| `9.571.csv` | 338,478 | `3db4e46ad037fa59` |
| `9.765.csv` | 464,132 | `35436771bc1cb180` |
| `9.956.csv` | 464,186 | `da8fb54ee3dee151` |

### 1.2 七份檔案的上游 attribution

下載並固定了 Nina 的
[h-blend v1](https://www.kaggle.com/code/nina2025/rogii-h-blend-v1) V4
（SHA-256 `0352c6bb...c370a527`）和
[h-blend v2](https://www.kaggle.com/code/nina2025/rogii-h-blend-v2) V5
（SHA-256 `204bc299...cf5d4`）。兩份 notebook 的 source table 與 embedded
source sections 明確列出：

| Dataset 檔案 | attributed upstream notebook | pinned `scriptVersionId` |
|---|---|---:|
| `10.142.csv` | Roman Tamrazov, `rogii-super-solution-lb-top-3` | 317517099 |
| `11.284.csv` | shinya, `triple-signal-beam-search-dual-pf-lightgbm` | 317309694 |
| `11.338.csv` | Pilkwang Kim, `rogii-eda-leakageriskdiscussion-v2` | 317591796 |
| `9.956.csv` | Roman Tamrazov, `rogii-better-solution-lb-9-956` | 318051097 |
| `9.571.csv` | Svanik Kolli, `aeroridge-engine-v2` | 318498402 |
| `9.537.csv` | Hoang Phuc, `rogii-v10-fresh-artifact-infer` | 318564530 |
| `9.765.csv` | Mark Cooper, `imitation-tasmim-lgb-xgb` | 318708494 |

這是**直接 source attribution 證據**，不是 exact content-hash lineage。
h-blend output preview 中 `9.537/9.765/9.956` 的前列數值和 dataset 在顯示
精度上一致；但 Kaggle 沒有提供原始 source output hash，不能把它升格成
byte-identical 證明。這些檔案也不是單一模型家族：高 correlation 主要反映
同一三口 visible test well 與共同 pipeline heritage，不能由 correlation 推論
相同生成器。

特別重要的直接負證據：h-blend 內嵌的 `9.537` source output 明確印出
`Exact overlap blend: blended 14151/14151 rows with weight=0.280`。這是本專案
禁止的 same-well train-coordinate TVT overlap，故即使日後補到 code/license，
該版本本身仍不能當合法 teacher。

## 2. 仍未知的 provenance

- `9.349.csv` 是 V5 單獨新增；它不在下載的 Nina h-blend v1/v2 source，
  也不在下載的 Evgen source。作者、notebook、checkpoint、feature list、
  offset/blend policy 皆未知。
- 七份有 attribution 的檔案仍沒有 byte-identical upstream output hash、
  exact execution environment、checkpoint hash 或完整 input manifest。
- 八份都沒有 train predictions、strict well-level OOF、fold definitions。
- Dataset license 是 `Unknown`。上游 notebook 的 Apache-2.0 是**程式碼
  license**，不自動授權 dataset 裡的 prediction rows 或外部 model artifacts。
- 無法證明 upstream 沒有 public-LB tuning、formation surfaces、Geology、
  PNG 或 same-well overlap；已有 source scan 反而直接找到多項這些路線。
- Kaggle attached-kernel API 在未登入狀態回傳 401；已用公開 dataset version
  tree、公開 notebook source/profile/output 頁補足可觀測部分，未猜測私有關聯。

因此 registry 的有效狀態仍是：

```text
eligibility = diagnostic_only
train_predictions_available = false
strict_well_level_oof_available = false
fit/oof_score/blend_weight_fit/production_ensemble = forbidden
```

### 與 Evgen physics notebook 的內容 lineage

固定下載的
[Evgen physics notebook](https://www.kaggle.com/code/evgendvorkin/roggi-physics-lb-7-872-v48)
是目前公開 V72，SHA-256
`67c7b924d6e86bd3fdf7cfe3e775e033dd2fe9f70a0f39092c3533a10c30e103`，
code license Apache-2.0。`ROGII-03` 是其 attached input，但：

- source 中沒有八個檔名，也沒有 `rogii-03`/`nina2025` pathname；
- V72 的 52 個 output filenames 沒有這八個檔名；
- V72 current `submission.csv` 顯示的前列 prediction 和八份檔案全部不同；
- V72 source 明確含 formation surfaces、Geology/contact paths、
  guarded same-well override、model packages 與 public-score-derived Q0522。

可證明的是「attached input、不是 V72 current final output」；仍不能排除某個
未固定的舊 version 或未公開 intermediate 有內容關聯。故本報告不把八份 CSV
描述成 Evgen notebook 的中間或最終輸出。

## 3. Notebook 可合法重作的方法

本次建立 `SafePhysicsAlignmentFamily`
（`src/rogii/physics_alignment.py`），只讓以下 prediction-time views 進入演算法：

- horizontal used columns：`MD, Z, GR, TVT_input`；
- typewell used columns：`TVT, GR`；
- `X/Y` 雖在通用 inference schema 中可讀，但 family manifest 明確記錄為
  unused；target `TVT`、formation、Geology、PNG 均不會被讀入 prediction。

乾淨實作包含：

1. **Multi-scale Particle Filter**
   - multi-window normalized GR likelihood；
   - TVT trajectory/velocity transition prior；
   - visible `TVT_input` prefix affine anchor；
   - 5 seeds，輸出 prediction、within/between uncertainty 與 effective
     particle count。
2. **Beam Search**
   - deterministic monotone path；
   - slope 與 curvature transition penalty；
   - single path 與 multi-path soft ensemble。
3. **Multi-scale NCC control**
   - 多窗口 NCC cost、monotone dynamic path 與平滑；
   - 作為 PF/beam 增量價值的便宜 control。

任何低 confidence、exception、shape mismatch、NaN/Inf 都明確回退 exact
incumbent identity，並在 diagnostics 記錄 `identity_fallback`。所有 alignment
residual 都先分解為 `common + slope + zero-mean smooth shape`，再由
`src/rogii/nested_blend.py` 只用 outer-training wells fit bounded component
shrinkage；outer validation labels 不參與 weight fit。

## 4. 禁止或排除的方法

沒有複製或使用：

- ANCC/ASTN/EGFD/BUDA formation contacts/surfaces；
- `Geology`、PNG、train target `TVT`；
- same-well train TVT/contact override；
- hard-coded visible test wells；
- public-LB offset、public score model selection；
- upstream LightGBM/CatBoost/Ridge/model-package artifacts；
- `rogii-03` test-only predictions 作 pseudo-label、teacher 或 blend target。

## 5. Quick / full strict OOF 結果

### 5.1 Baseline 與 matched population

Canonical incumbent artifact：
`.runs/heel-registered-anchor-full-20260728/oof_predictions.csv.gz`，
SHA-256 `4606cf8e07973f...`。本次重算得到 773 wells、3,783,989 rows、
RMSE `14.17084789250194`，和 audited value 的差只有
`3.55e-15`；keys、folds、truth 與 incumbent 都由 independent audit 再比對。

Quick 使用既有、在本次 safe-physics outcome 之前固定的 5 outer folds，
每 fold 20 validation wells，共 100 wells / 489,284 rows。此 matched subset 的
incumbent RMSE 是 `12.8117468534`，**不可和 773-well production RMSE 當成同
population 比較**。Cohort coverage：

- length quartiles：24 / 21 / 30 / 25；
- GR standardized-difference spectral quartiles：27 / 25 / 28 / 20；
- 含 6 個全體 incumbent top-10% tail wells；
- 每 fold 恰 20 wells。

### 5.2 Raw alignment

Raw TVT paths 全部遠差於 incumbent，顯示 clean-room GR alignment 目前沒有可
直接使用的絕對 TVT calibration：

| Raw candidate | pooled RMSE | delta vs quick incumbent |
|---|---:|---:|
| NCC multi-scale | 33.920929 | +21.109182 |
| PF single-scale | 40.678010 | +27.866263 |
| PF multi-scale | 40.753954 | +27.942207 |
| Beam single path | 43.680815 | +30.869068 |
| Beam ensemble | 43.690366 | +30.878620 |
| wrong typewell | 27.444407 | +14.632660 |
| typewell pairing shuffle | 26.269690 | +13.457944 |
| GR order shuffle | 40.498468 | +27.686721 |
| phase-destroy GR | 40.509335 | +27.697588 |
| no prefix anchor | 345.329457 | +332.517710 |
| no cross-series GR interaction | 45.778928 | +32.967181 |

PF raw seed delta（seed order 20260711…20260715）：

| Candidate | five seed deltas |
|---|---|
| PF single-scale | +27.796834, +27.989259, +28.225396, +28.020587, +28.006195 |
| PF multi-scale | +27.805525, +28.297588, +27.784078, +28.058395, +28.457726 |
| wrong typewell | +14.506855, +14.536322, +14.614617, +14.658296, +15.191728 |
| pair shuffle | +13.476762, +13.454246, +13.658585, +13.800725, +13.638312 |
| GR order shuffle | +27.880443, +27.959909, +27.880269, +27.745290, +27.620123 |
| phase destroy | +27.747795, +27.862870, +27.842930, +27.990521, +27.897637 |
| no prefix | +347.514239, +357.053799, +352.783041, +367.647788, +334.498468 |

### 5.3 Strict cross-fitted residual components

下表的 weights 都只在其他四個 outer folds fit；CI 是 10,000 次 paired
well-cluster bootstrap：

| Candidate | RMSE | delta | 95% paired CI | improved folds | wins-losses |
|---|---:|---:|---:|---:|---:|
| nested NCC multi | 12.743537 | -0.068209 | [-0.2902, +0.1506] | 3/5 | 52-48 |
| nested PF single | 12.877326 | +0.065579 | [-0.0446, +0.1734] | 0/5 | 36-44 |
| nested PF multi | 12.888819 | +0.077072 | [-0.0509, +0.2023] | 0/5 | 36-44 |
| nested Beam single | 12.721980 | -0.089767 | [-0.3879, +0.2030] | 3/5 | 53-47 |
| nested Beam ensemble | 12.722066 | -0.089681 | [-0.3877, +0.2032] | 3/5 | 53-47 |
| nested wrong typewell | 12.873304 | +0.061557 | [-0.0076, +0.1463] | 0/5 | 11-11 |
| nested pair shuffle | 12.853855 | +0.042108 | [-0.1776, +0.2852] | 3/5 | 24-18 |
| nested GR order shuffle | 12.900908 | +0.089161 | [-0.0555, +0.2355] | 0/5 | 39-41 |
| nested phase destroy | 12.899930 | +0.088183 | [-0.0481, +0.2252] | 0/5 | 39-41 |
| nested no prefix | 12.845817 | +0.034070 | [-0.0368, +0.1191] | 2/5 | 49-51 |
| nested no cross interaction | 12.924027 | +0.112280 | [-0.0024, +0.2352] | 2/5 | 49-51 |

Best descriptive quick point estimate 是 Beam single `-0.08977`，但 CI 跨 0、
只改善 3/5 folds，且 slope/shape 退步。預先指定的 PF multi gate 更是
`+0.07707`、0/5 folds；其 correct-pairing delta 比最佳負控制差
`0.04300 ft`。因此所有 full-run checks 失敗。

### 5.4 Full OOF 與 200-well subsets

**未執行，這是 protocol-compliant stop，不是缺失的正結果。** Quick 沒有
同時通過 negative delta、CI upper < 0、至少 4/5 folds、slope/shape signal
與 correct pairing > controls。Quick 只有 100 wells，故也不能合法產生
200-well subsets。沒有任何 formal nested candidate，因此 100,000 組
200-well mean delta、P(worse)、q01/q05/q50/q95/q99 均為 `not_run / N.A.`。

## 6. 負控制結果

負控制否定了「模型學到 geological GR correspondence」：

- Correct PF multi nested delta `+0.07707`。
- Wrong-typewell `+0.06156`，pair-shuffle `+0.04211`，都比 correct pairing
  更好；correct pairing 對 best control 的 margin 是 `-0.04300 ft`。
- Wrong/pair shuffle 因 confidence gate 分別在 59/58 wells 安全回退
  incumbent；這不是隱藏失敗，已完整記錄。即使如此，correct pairing 仍未
  顯示勝出。
- GR order shuffle 與 phase-destroy 的 raw/nested 表現都和 correct PF 接近；
  沒有 phase-sensitive signal。
- 移除 prefix anchor 的 raw RMSE 變成 345.33，證明 anchor 是數值穩定性的
  必要條件，但不證明 GR geology correspondence。
- PF single-scale 比 multi-scale raw 好約 0.076 ft；multi-scale 沒有增量。
- Beam ensemble 和 single path 幾乎相同，ensemble raw 反而差 0.0096 ft，
  nested 也差 0.00009 ft；多路徑沒有增量。
- NCC raw 明顯優於 PF/Beam，但仍比 incumbent 差 21.11 ft；複雜 tracker
  沒有超越便宜 control。

## 7. Architecture / physics signal

| Candidate | common RMSE | slope RMSE | centered-shape RMSE | terminal RMSE | residual corr. vs incumbent |
|---|---:|---:|---:|---:|---:|
| incumbent | 9.898 | 5.173 | 6.277 | 21.244 | 1.000000 |
| nested NCC multi | 9.810 | 5.174 | 6.277 | 21.297 | 0.998815 |
| nested PF single | 9.983 | 5.173 | 6.277 | 21.354 | 0.998584 |
| nested PF multi | 9.997 | 5.173 | 6.279 | 21.368 | 0.998087 |
| nested Beam single | 9.764 | 5.188 | 6.292 | 21.321 | 0.996473 |
| nested Beam ensemble | 9.764 | 5.188 | 6.292 | 21.320 | 0.996470 |

Beam/NCC 的 descriptive point gain 完全是 small common-mode correction。
Beam 把 common RMSE 改善約 0.134 ft，但 slope、centered-shape、terminal 都
惡化；NCC 的 shape 實質不變。PF 沒有任何 component signal。這不符合
「至少明確改善 slope 或 centered-shape」的 production rule。

## 8. External teacher 是否真正提供可驗證增益

**沒有可驗證 teacher experiment。** 目前只有 3 visible test wells 的
predictions；拿它們做 distillation、weight fitting 或 descriptive blend
不能產生 773-well strict OOF evidence。因 license、OOF、folds、exact code、
checkpoint 與 allowed-input manifest 都未通過，standalone teacher、
teacher+physics、nested external blend 全部沒有執行。

Target-free geometry 僅作 lineage/shape diagnosis：

- 28 個 pairwise raw correlations：min 0.999928、median 0.999969、
  max 0.999998；
- within-well centered correlations：min 0.966394、median 0.984025、
  max 0.998515；
- pairwise prediction RMSE：0.797–3.665 ft；
- 對 visible-test incumbent 的 raw correlation 為
  0.999396–0.999658、within-well centered correlation 為
  0.779749–0.841790，但 difference RMSE 為 7.958–10.095 ft。

| File | diff RMSE | common | linear slope | zero-mean shape | terminal | high-frequency shape energy |
|---|---:|---:|---:|---:|---:|---:|
| `10.142` | 9.848 | 3.625 | 6.715 | 6.226 | 8.976 | 0.00147 |
| `11.284` | 10.095 | 3.034 | 7.096 | 6.507 | 8.144 | 0.00249 |
| `11.338` | 9.659 | 3.265 | 6.518 | 6.337 | 5.794 | 0.00269 |
| `9.349` | 8.510 | 3.553 | 5.827 | 5.083 | 6.847 | 0.00219 |
| `9.537` | 8.883 | 4.169 | 5.934 | 5.130 | 8.114 | 0.00216 |
| `9.571` | 8.890 | 3.696 | 6.443 | 4.884 | 8.357 | 0.00422 |
| `9.765` | 7.958 | 3.636 | 5.304 | 4.688 | 9.577 | 0.00175 |
| `9.956` | 9.773 | 3.598 | 7.319 | 5.384 | 9.168 | 0.00178 |

這些數字不使用 labels，也不代表 accuracy；沒有用檔名 score 選檔。

## 9. Tail / private-size stability

| Candidate | p50 | p90 | p95 | max | ≥40 | ≥60 | worst-10% SSE share |
|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent | 9.996 | 18.792 | 21.906 | 35.011 | 0 | 0 | 0.397 |
| nested NCC multi | 9.776 | 19.462 | 20.861 | 33.775 | 0 | 0 | 0.398 |
| nested PF single | 9.721 | 19.113 | 21.117 | 35.011 | 0 | 0 | 0.400 |
| nested PF multi | 9.731 | 19.114 | 20.994 | 35.011 | 0 | 0 | 0.399 |
| nested Beam single | 9.518 | 19.782 | 21.398 | 33.570 | 0 | 0 | 0.403 |
| nested Beam ensemble | 9.516 | 19.782 | 21.395 | 33.569 | 0 | 0 | 0.403 |
| nested wrong typewell | 9.996 | 18.792 | 21.091 | 35.011 | 0 | 0 | 0.398 |
| nested pair shuffle | 9.582 | 19.017 | 21.906 | 34.085 | 0 | 0 | 0.389 |
| nested GR order shuffle | 9.787 | 19.113 | 21.205 | 35.011 | 0 | 0 | 0.400 |
| nested phase destroy | 9.717 | 19.115 | 21.077 | 35.011 | 0 | 0 | 0.399 |
| nested no prefix | 10.063 | 18.827 | 22.062 | 35.010 | 0 | 0 | 0.400 |
| nested no cross interaction | 9.967 | 19.097 | 21.149 | 35.012 | 0 | 0 | 0.395 |

Quick Beam/NCC 沒有 catastrophic-count 惡化，但 Beam 的 p90、worst-10% SSE
share 與 terminal 都比 incumbent 差；這和不確定的 mean gain 一起否決
promotion。因為沒有通過 full gate，不能聲稱 private-size stability；尤其
不能把三口 visible test wells 當 hidden/private evidence。

## 10. Production 決策

**Production 維持 Heel-Registered Sparse-Anchor Student 5-seed ensemble。**

決策規則逐項：

| Rule | 結果 |
|---|---|
| strict nested evidence | quick cross-fit 有；full evidence 無 |
| paired CI upper < 0 | fail |
| 不集中於 fold/seed | fail；最多 3/5 folds |
| 200-well P(worse) < 1% | not run，不能證明 |
| p95/max/catastrophic 無惡化 | quick 大致通過，但不足 |
| slope 或 centered-shape 明確改善 | fail |
| correct pairing 優於 controls | fail |
| provenance/license/offline reconstruction | external source fail |

沒有建立或上傳 Kaggle submission。

## 11. Remaining risks

1. Quick 是 100-well matched subset；雖 length/GR quartiles 平衡，只有 6 個
   incumbent top-10% tail wells。它適合做 stop decision，不足以估 production
   effect。
2. Clean-room alignment 參數只做代表性設定，沒有在同一 validation truth
   大量搜尋。負結果可能包含 implementation/parameterization miss，但避免了
   quick overfit。
3. Current PF state model 的 absolute TVT drift 很大；nested shrinkage 多數把
   component weights 壓近 0。更複雜 selector 不會自動修復 identifiability。
4. Wrong-pair controls常觸發 fallback；未來若研究 pairing classifier，應分開
   報告「辨識正確 typewell」和「安全 fallback 後的 RMSE」。
5. Kaggle public pages與 notebook versions 會變；本次已保存 source hashes，
   但 upstream output hashes與 dataset license仍需作者提供。

## 12. 下一個最高資訊價值實驗

建議先做一個**label-free, blinded typewell identification test**，不要直接再跑
大模型：

1. 對固定 outer-training wells，為每口 horizontal GR 建立 1 個正確 typewell
   加 31 個 fold-safe wrong typewells；
2. 只用 prefix anchor 後的 multi-window NCC/phase-sensitive likelihood 排名；
3. 預先固定 top-1 accuracy、mean reciprocal rank、correct-vs-best-wrong margin；
4. 用 typewell shuffle 與 phase destroy 建立 chance distribution；
5. 只有 correct pairing 明顯高於 chance，才把 margin 作 confidence gate，
   在未使用的 outer validation wells 評估 residual components。

這個實驗直接回答目前最關鍵的 identifiability 問題，且不需要 labels 調
alignment path。若連正確 typewell 都無法從 GR 中辨識，繼續加 PF particles、
beam width 或 boosters 的資訊價值很低。

## Reproducibility 與交付

程式：

- `src/rogii/physics_alignment.py`
- `src/rogii/nested_blend.py`
- `src/rogii/external_predictions.py`
- `scripts/run_safe_physics_experiment.py`
- `scripts/audit_safe_physics_artifacts.py`
- `scripts/smoke_safe_physics_visible_test.py`
- `configs/external/rogii03_v5_provenance_20260729.json`

主要 disk artifacts：

- `.runs/rogii03-safe-physics-20260729/quick/oof_predictions.csv.gz`
- `.runs/rogii03-safe-physics-20260729/quick/metrics/`
- `.runs/rogii03-safe-physics-20260729/quick/nested_component_weights.json`
- `.runs/rogii03-safe-physics-20260729/quick/alignment_diagnostics.csv`
- `.runs/rogii03-safe-physics-20260729/provenance/`

驗證狀態：

- `rogii check --data-root data/rogii-wellbore-geology-prediction`：pass；
- focused new tests：9 passed；
- full `python -m pytest -q`：452 passed、37 warnings；
- visible-test offline smoke：3 wells / 14,151 rows，pass；
- independent disk audit：`passed=true`，OOF SHA-256
  `08ba29078c275c8e6ec2fc882daa6d022fe0de3e0c60e83c6cb11992e0be8776`；
  canonical truth/incumbent max delta = 0，nested reconstruction max delta =
  `3.64e-12`，10,000-bootstrap max delta = `5.68e-14`；
- data manifest before/after：identical（2,327 files，
  digest `dec44ac09997b60e900c70064349f0fc9a1d558a9cda5c05aaa3530902fe4514`）；
- submission artifacts：none。

完整逐 fold、逐 seed、per-well、bootstrap、tail、component、correlation 與
geometry 數值均留在上述 `.runs/` artifacts，沒有只保留報告中的摘要。
