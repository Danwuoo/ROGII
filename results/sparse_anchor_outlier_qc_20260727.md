# Sparse-anchor robust anchor QC／outlier rejection — 正式摘要

日期：2026-07-27（Asia/Taipei）  
Primary：`near_k6 = [1/6,2/6,3/6,4/6,5/6,0.975]`  
Secondary：`td_k6 = [1/6,2/6,3/6,4/6,5/6,1.0]`  
Scope：773 wells／3,783,989 prediction rows／100 paired realizations／
10,000 hierarchical bootstraps

完整報告與 artifacts 位於
`.runs/sparse-anchor-outlier-qc-full/`。

## 結論

`local_spike_reject_7p5` **不能安全辨識 M10 anchors，不能採用**。
near_k6 retrospective measurement specification **不能**擴張到
`p=.10/M=10`。

四個 primary gates 全部失敗：

| Gate | 結果 |
| --- | --- |
| A. O5 near_k6 完整 robust `<4` | fail |
| B. C0/C1/C3 control safety | fail |
| C. O5 recall/FPR/precision | fail |
| D. primary tail/fold safety | fail（沒有 >=40/60，但有 persistent fold failure） |

## 直接回答

1. **能否安全辨識 M10 anchors？**  
   不能。O5 anchor-level recall／precision／FPR 是
   `41.7733% / 16.2068% / 24.1058%`，遠離固定 gate
   `>=70% / >=75% / <=2%`。

2. **Clean 與 M5 false-rejection rate？**  
   near_k6 exact-clean 為 `22.9625%`；Gaussian-1 clean 為 `23.4789%`；
   C3 的 clean anchors 為 `23.8551%`。C3 的 M5 anchors 有
   `29.4967%` 被 reject，證明此 rule 無法安全區分 M5 與 M10。

3. **Endpoint、interior、single、multiple detection 差多少？**  
   O5 endpoint recall `60.9932%`，但 clean FPR 高達 `47.3169%`；
   interior recall `32.1417%`、FPR `12.5031%`。Isolated／multiple recall
   為 `43.9740% / 38.5742%`。正／負 M10 recall 幾乎相同：
   `41.9252% / 41.6212%`。

4. **O5 near_k6 能否恢復完整 robust `<4`？**  
   不能。candidate full/non-anchor q95
   `6.924998/6.924769`，hierarchical q97.5
   `7.102102/7.101882`。

5. **對 exact、Gaussian、moderate M5 的傷害？**  
   full q95 相對 no-QC 分別惡化
   `+2.774330 / +2.783921 / +2.651984 ft`；
   non-anchor 分別 `+2.772187 / +2.781901 / +2.650190 ft`。
   三個原本 robust 的 controls 全部失去 robust status。

6. **Reject 一個後的五個 anchors 是否仍足以 `<4`？**  
   不足。O5 恰有一個 outlier 的 well-realization subset 中，
   indicator oracle 移除一個、保留五個後，full/non-anchor q95 仍是
   `4.741017/4.742105`；clean-value oracle 是
   `3.582768/3.584974`。

7. **Candidate 與 indicator oracle gap？**  
   O5 candidate-minus-indicator-oracle paired full-RMSE
   q05/q50/q95 是 `2.3921/2.5413/2.6953 ft`。
   但 indicator oracle 本身也失敗：full q95 `4.375388`、
   hierarchical q97.5 `4.457262`。

8. **主要 failure 是什麼？**  
   有兩層：
   - detector failure：真實 clean residual curve 的 local curvature 被
     one-sided/interior linear leave-one-out 當成 spike，尤其 endpoints；
   - replacement geometry failure：即使知道 true indicator，移除 anchor 後
     information density 仍不足。  
   common bias 不是 score failure：全 anchors 同加 bias 的 score 最大數值差只有
   `3.55e-14`。Specific wells 加重 tail，但 high FPR 與 harm 是 population-wide。

9. **是否重新出現 RMSE >=40 或 >=60 catastrophic wells？**  
   no-QC 與 operational candidate 在所有 scenarios 都是 `0/0`。
   唯一 >=40 是 diagnostic oracle 在 `p=.20/M=10` 移除四個 anchors、
   只剩兩個時：well `ba48188d` RMSE `42.6346`。沒有 >=60。

10. **near_k6 specification 能否擴張到 p=.10/M=10？**  
    不能。candidate 破壞 controls、偵測 gate 失敗、O5 reconstruction
    失敗，而且 indicator oracle 也未恢復 `<4`。

11. **下一步？**  
    不保留此 QC，也不在本輪調 threshold。依預先固定的 oracle 解讀，下一個
    M10 資訊需求應是**增加冗餘 anchors 或 repeat measurements**；
    不是更複雜 detector。Quality metadata 可在有冗餘量測時再研究，但 perfect
    indicator 已證明目前六-anchor rejection geometry 不足。
    `tau=2/sigma=2` datum/noise calibration 仍是另一個獨立問題。

## Severe attribution

S0 沒有 outlier 就已失敗：no-QC full q95 `4.281737`、
hierarchical q97.5 `4.339049`。S1 no-QC 為 `4.980931/5.030689`；
indicator oracle 只降到 `4.867395/4.946138`。因此 S1 不能歸咎 QC；
severe background calibration 本身已不足。

## 驗證

- no-QC 精確重現既有 exact／moderate-M5／outlier p=.10-M10／severe：
  800 pooled groups 最大 RMSE delta `8.88e-16`；
  618,400 per-well groups 最大 SSE delta `2.33e-10`；
- 8,800 metric groups；
- 6,802,400 per-well rows；
- 10,203,600 candidate anchor decisions；
- pooled metric reconstruction delta
  `1.78e-15 / 2.66e-15`；
- duplicate keys `0`；
- unexpected numeric NaNs `0`；
- shared physical-row latent delta `0`；
- accepted anchor observation delta `0`；
- 無 formation surfaces、`Geology`、PNG、typewell；
- 無 `data/` 寫入；
- `python -m pytest -q -W error`：`213 passed in 112.15s`；
- final `rogii check`：5 folds／773 wells／3,783,989 prediction rows；
- 無 Kaggle submission 建立或上傳。

Evidence status：

- `deployable: false`
- `submission_eligible: false`
- `counterfactual_information_experiment: true`
- `uses_validation_target_as_revealed_anchor: true`
