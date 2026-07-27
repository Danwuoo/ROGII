# Sparse-anchor acquisition redundancy frontier — 正式摘要

日期：2026-07-27（Asia/Taipei）  
Primary scenario：`O5_moderate_p10_m10`
（`tau=1/sigma=1/p=.10/M=10`）  
Scope：773 wells／3,783,989 prediction rows／100 paired realizations／
10,000 paired hierarchical bootstraps

完整 artifacts 與英文技術報告位於
`.runs/sparse-anchor-acquisition-redundancy-full/`。

## 結論

O5 下最小 unconditional robust `<4` acquisition design 是
**`near_k8_r1`：八個不同 physical locations、每處一次讀值**：

`[1/8,2/8,3/8,4/8,5/8,6/8,7/8,0.975]`

它的 full/non-anchor realization q95 是
`3.945561/3.946117`，hierarchical q97.5 是
`3.990370/3.991075`。四項數值都小於 4，但 hierarchical margin只有
`0.0089–0.0096 ft`，所以是通過但不寬裕的 retrospective specification。

固定六個 locations 時沒有 unconditional pass：

- `near_k6_r2_mean` 即使 gross errors measurement-independent 仍失敗；
- `near_k6_r3_median` 在 measurement-independent 下通過，但
  anchor-shared 下失敗，只能稱為需要真正獨立 reacquisition failure mode
  的 conditional specification。

`near_k8_r2_mean` 兩種 dependence 都通過，但使用 16 readings，且在相同八個
physical locations 上被 8-reading 的 `near_k8_r1` 支配，不是 Pareto point。
Unconditional passing Pareto frontier只有 `(8 physical, 8 readings)`。

## O5 operational frontier

| design | dependence | physical/readings | full q95 | non-anchor q95 | hier full q97.5 | hier non q97.5 | result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| near_k6_r1 | single-read equivalent | 6/6 | 4.487370 | 4.488427 | 4.538905 | 4.539957 | fail |
| near_k7_r1 | single-read equivalent | 7/7 | 4.182777 | 4.183649 | 4.232230 | 4.233026 | fail |
| **near_k8_r1** | single-read equivalent | **8/8** | **3.945561** | **3.946117** | **3.990370** | **3.991075** | **unconditional pass** |
| near_k9_r1 | single-read equivalent | 9/9 | 3.748617 | 3.748995 | 3.778575 | 3.778918 | pass |
| near_k10_r1 | single-read equivalent | 10/10 | 3.607268 | 3.607362 | 3.632649 | 3.632866 | pass |
| near_k11_r1 | single-read equivalent | 11/11 | 3.510332 | 3.510239 | 3.527550 | 3.527505 | pass |
| near_k12_r1 | single-read equivalent | 12/12 | 3.421152 | 3.420862 | 3.443934 | 3.443695 | pass |
| near_k6_r2_mean | measurement-independent | 6/12 | 4.052689 | 4.054143 | 4.124126 | 4.125605 | fail |
| near_k6_r2_mean | anchor-shared | 6/12 | 4.455975 | 4.457022 | 4.505198 | 4.506328 | fail |
| near_k6_r3_median | measurement-independent | 6/18 | 3.753607 | 3.755415 | 3.840510 | 3.842347 | **conditional pass** |
| near_k6_r3_median | anchor-shared | 6/18 | 4.454979 | 4.455987 | 4.501283 | 4.502390 | fail |
| near_k8_r2_mean | measurement-independent | 8/16 | 3.444156 | 3.445348 | 3.502592 | 3.503895 | pass, dominated |
| near_k8_r2_mean | anchor-shared | 8/16 | 3.914489 | 3.915159 | 3.949557 | 3.950383 | pass, dominated |

Full 與 non-anchor conclusions完全一致；兩者差異只有約千分之一英尺。

## 直接回答

1. **O5 最小 passing single-read spatial k？**  
   `k=8`。`k7`的 q95/hierarchical仍是 `4.1828/4.2322`；`k8`
   首次四項都低於 4。

2. **固定 near_k6，r2 mean 或 r3 median 能否恢復 robust `<4`？**  
   r2不能；independent q95/hierarchical `4.0527/4.1241`。r3只在
   measurement-independent 下能，q95/hierarchical `3.7536/3.8405`；
   shared fault 下仍約 `4.4550/4.5013`。

3. **Repeat benefit 是否只存在於 measurement-independent outliers？**  
   實質上是。Shared fault 下 r2/r3相對 k6-r1 paired median只改善
   `0.0344/0.0375 ft`，主要是 Gaussian averaging；gross shift仍是
   10 ft。Independent 下則改善 `0.4036/0.6950 ft`。

4. **Anchor-shared gross fault 下哪些 designs通過？**  
   Spatial `near_k8_r1` 至 `near_k12_r1`以及 `near_k8_r2_mean`通過。
   六-location r2/r3都失敗。

5. **最少 total readings與最少 physical locations？**  
   對 unconditional operational pass，兩者都是 `near_k8_r1`：
   8 locations／8 readings。若額外假設真正獨立 reacquisition，
   最少 physical locations可降到六，但需 `near_k6_r3_median`
   的 18 readings；這不是 shared-fault-safe答案。

6. **`near_k8_r2_mean`是否為有意義的 hybrid Pareto point？**  
   不是。它兩種 dependence皆通過，但 `(8,16)`被 `near_k8_r1`
   的 `(8,8)`嚴格支配。它只提供額外 margin，不提供 Pareto效率。

7. **M5 controls是否全部保持 robust？**  
   是。所有 spatial/repeat/hybrid designs與兩種 repeat dependence都通過。
   例：k6-r2 independent/shared q95 `3.7355/3.8452`；
   k6-r3 `3.6616/3.8417`。所有 control-harm gates為 true；沒有任何
   comparator q95惡化。

8. **Perfect per-reading indicator增加多少 headroom？**  
   它在有 independent redundancy時有價值：
   - k6-r2 observed→indicator-oracle q95：
     `4.052689→3.723642`，由 fail變 pass；
   - k6-r3：`3.753607→3.631000`；
   - k8-r2 independent：`3.444156→2.993333`；
   - k8-r2 shared：`3.914489→3.554125`。  
   但 k6 shared r2/r3的 oracle仍是 `4.3397/4.3417`，因為同源 fault使
   整個 location被移除，replacement geometry仍不足。Quality metadata只有
   搭配真正 redundancy才可能有足夠 headroom。

9. **Effective M10 shift經 mean/median後降到多少？**  
   - r1：p95 `10 ft`，約 10.04% locations有 10-ft shift；
   - r2 independent mean：p95 `5 ft`，`|shift|>=7.5`僅 `0.514%`；
   - r3 independent median：p90/p95 `0.693/1.352 ft`，
     `|shift|>=5`為 `1.455%`、`>=7.5`為 `1.431%`；
   - anchor-shared r2/r3：p95仍是 `10 ft`。  
   r2/r3 independent約 49.4%／48.2%的 eligible mixed-sign locations互相抵消。

10. **Full與 non-anchor conclusions是否一致？**  
    完全一致。沒有任何 design/dependence在兩個 metrics之間改變 pass/fail。

11. **是否重新出現 RMSE >=40或>=60 wells？**  
    沒有。全部 13,100 metric groups、所有 methods/scenarios/realizations的
    maximum counts都是 `0/0`。O5 operational的最差單一 well-realization
    是 k6-r1 `12.4456`；k8-r1是 `10.6522`。

12. **Failure主要來自什麼？**  
    Primary O5 failure主要是 physical-anchor density／inter-anchor
    reconstruction與 repeat-error correlation，不是少數 wells：
    - k7即使 indicator oracle hierarchical q97.5仍為 `4.0238`；
    - k8增加一個不同 location後 observed aggregate即通過；
    - same-location r3只在 independent fault下通過，shared fault完全失去
      gross-error suppression；
    - k6 moderate-background本身 robust，所以 tau=1 common bias不是 O5
      primary blocker；
    - 沒有 catastrophic wells，worst-10% SSE share約 31%，不是單一 well
      runaway。  
    r1 spatial designs共用 97.5%最後 anchor，所以 post-last q50都約
    `4.0688`；增加 locations主要把 pre-first/between q50從 k6的
    `4.3610/4.4413`降到 k8的 `3.6310/3.9074`。

13. **明確 retrospective measurement specification？**  
    在固定 O5模型與 locked acausal reconstruction下，採用
    **八個不同 physical anchors、每處一次讀值**，fractions為
    `[1/8,2/8,3/8,4/8,5/8,6/8,7/8,0.975]`；PS boundary保持精確；
    不需要本輪已否決的 QC detector。這仍是 target-bearing、acausal、
    retrospective、不可部署的counterfactual specification。

14. **下一輪方向？**  
    Primary O5應轉向**實際八-location anchor acquisition protocol**，
    因為它是最簡單且shared-fault-safe的 passing設計。若現場成本強迫只用
    六個 locations，可另做真正獨立 r3 reacquisition protocol驗證，但必須先
    證明 failure modes獨立，不能把普通同源 repeat當成解答。
    Datum calibration仍是 severe tau=2/sigma=2的獨立問題，不是 primary
    O5通過所需的下一步。

## Exposure與subset

O5實際 outlier-reading fraction維持約 10%。Independent repeats增加
「至少一個 outlier read」的 location exposure，但降低aggregate fault：

| design/dependence | any-outlier location | majority-outlier location | all-outlier location | wells any/multiple |
| --- | ---: | ---: | ---: | ---: |
| k6-r1 | 10.041% | 10.041% | 10.041% | 47.110% / 11.419% |
| k6-r2 independent | 19.032% | 1.018% | 1.018% | 71.781% / 32.110% |
| k6-r3 independent | 27.163% | 2.808% | 0.102% | 85.113% / 51.686% |
| k6 repeat shared | 10.041% | 10.041% | 10.041% | 47.110% / 11.419% |

Endpoint/interior沒有 outlier-QC前輪的端點不對稱。k6-r3 independent
effective-shift p95為 `1.362/1.346 ft`；各 anchor-order p95只在
`1.332–1.376 ft`。r2各 order都是 `5 ft`，shared fault各 order都是
`10 ft`。

Independent k6-r3的 isolated/multiple-contaminated well-realization pooled
full RMSE為 `3.6591/3.7890`；r2則為 `3.9611/4.3957`，multiple exposure
是 r2未能通過完整 gate的主要 residual tail。k8-r1的 exposed subsets仍可
高於 4（isolated/multiple `4.0835/5.1320`），但完整 fixed-well/hierarchical
population gates均通過；沒有 catastrophic tail。

## Margin、stress與severe attribution

- p=.05/M10：single-read最小 passing k是 7；k6-r2/r3 independent通過，
  shared仍失敗。
- p=.20/M10：k6–k12 single-read全部失敗；k8-r2只在 independent模式通過。
- S0：k6-r1、k6-r2、k6-r3都因 severe background本身失敗；k7 point q95
  低於 4但 hierarchical `4.0152`仍失敗；k8起 spatial S0通過。
- S1：所有 single-read k≤12仍失敗；k8-r2 independent的 q95雖為
  `3.9783`，hierarchical `4.0173`仍失敗。

因此 near-k6 severe failure仍優先歸因 tau=2/sigma=2 background
calibration；不能要求 O5 redundancy單獨解決。較高 density可救回 S0，
但本輪 primary gate與 design selection沒有因此改動。

## 驗證

- no-QC prior reproduction：
  - 700 pooled groups最大 RMSE delta `8.88e-16`；
  - 541,100 per-well groups最大 SSE delta `2.33e-10`；
- k6-r1 O5 indicator-oracle pooled delta `8.88e-16`；
  77,300 per-well groups最大 SSE delta `3.73e-9`；
- k7 exact full RMSE精確重現 `3.0151712904359784`，delta `0`；
- replicate-0 legacy、shared physical row/replicate、common bias latent
  maximum deltas全部 `0`；
- physical anchor prediction對aggregate observation maximum delta `0`；
- 131 evaluation groups／13,100 metric groups／10,126,300 compact
  per-well rows；
- pooled full/non-anchor reconstruction最大 delta均 `1.78e-15`；
- measurement exposure 10,500 rows；effective-anchor error 880 rows；
- duplicate keys `0`；unexpected numeric NaNs `0`；
- 無 formation surfaces、`Geology`、PNG或typewell讀取；
- 無 `data/`寫入；
- `python -m pytest -q -W error`：`253 passed in 187.52s`；
- final `rogii check`：5 folds／773 wells／3,783,989 prediction rows；
- 無 Kaggle submission建立或上傳。

Evidence status：

- `deployable: false`
- `submission_eligible: false`
- `counterfactual_information_experiment: true`
- `uses_validation_target_as_revealed_anchor: true`
