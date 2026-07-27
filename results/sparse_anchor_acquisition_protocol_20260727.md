# Sparse-anchor eight-location practical acquisition protocol qualification

日期：2026-07-27（Asia/Taipei）  
Primary protocol tolerance：`T_u=0.001`  
Primary measurement scenario：`O5_moderate_p10_m10`
（`tau=1/sigma=1/p=.10/M=10`）  
Scope：773 wells／3,783,989 OOF prediction rows／100 paired
realizations／10,000 paired hierarchical bootstraps

完整 artifacts 與英文技術報告位於：

`C:\Users\wurre\Desktop\ROGII\.runs\sparse-anchor-acquisition-protocol-full\`

## 正式結論

固定八位置 practical acquisition protocol 在 primary
`T_u=0.001` 下 **unconditional pass**。

名義 normalized post-PS MD positions 完全鎖定為：

```text
[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.975]
```

三種固定 position modes 在 O5 下都通過既有六項 robust `<4` gates：

| position mode | full q95 | non-anchor q95 | hier full q97.5 | hier non-anchor q97.5 | minimum margin | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| coherent early | 3.947951 | 3.948589 | 3.990627 | 3.991265 | 0.008735 | pass |
| coherent late | 3.946383 | 3.946941 | 3.990579 | 3.991367 | 0.008633 | pass |
| independent bounded | 3.944270 | 3.944962 | 3.991363 | 3.992151 | **0.007849** | **pass；limiting** |

最限制 margin 的 mode 是 `independent_bounded`；最限制數值是
hierarchical non-anchor q97.5 `3.9921512903134353`，所以 primary
minimum hierarchical／overall numeric margin 為：

```text
4 - 3.9921512903134353 = 0.007848709686564703 ft
```

這個 margin 仍然很窄。Primary pass 只適用於本輪鎖定的 computational
position tolerance、O5 stochastic envelope、exact PS boundary 與既有
acausal piecewise-residual reconstruction。

## 1. Zero-tolerance 是否精確重現 acquisition-redundancy？

是。`T_u=0` 使用 legacy nominal-row placement 與 measurement keys，重現：

| metric | protocol `T_u=0` | locked acquisition-redundancy |
| --- | ---: | ---: |
| full fixed-well realization q95 | 3.945560597463506 | 3.945560597463506 |
| non-anchor fixed-well realization q95 | 3.946117145448976 | 3.9461171454489756 |
| hierarchical full q97.5 | 3.990370048942644 | 3.990370048942644 |
| hierarchical non-anchor q97.5 | 3.9910746376968422 | 3.9910746376968427 |

正式 regression audit：

- 300 zero-tolerance pooled seed/scenario groups中，maximum metric delta：
  `8.88e-16`；
- 231,900 per-well groups 全部比較；
- per-well full/non-anchor SSE maximum delta：
  `4.66e-10 / 4.66e-10`；
- hierarchical paired-draw maximum delta：`4.44e-16`；
- zero-tolerance selected-row maximum delta：`0`；
- zero-tolerance measurement-latent maximum delta：`0`；
- zero-tolerance full/non-anchor reconstruction row delta：`0`。

因此差異只在 CSV／floating serialization 最後一位，沒有 regression
failure。

## 2. Primary `T_u=0.001` 是否 unconditional pass？

是。`coherent_early`、`coherent_late`、`independent_bounded` 是 strict
AND；三者 O5 都同時滿足：

1. full fixed-well realization q95 `<4`；
2. non-anchor fixed-well realization q95 `<4`；
3. hierarchical full q97.5 `<4`；
4. hierarchical non-anchor q97.5 `<4`；
5. no persistent fold failure；
6. 所有 realizations 的 `wells_ge_40=0` 且 `wells_ge_60=0`。

此外三種 modes 都沒有 boundary、availability、duplicate 或 order
violation，exact-clean 與 M5 controls 也都保持 robust。

## 3. 哪個 position mode 最限制 margin？

`independent_bounded`。

它的 fixed-well q95 反而略低於 zero tolerance，但 paired hierarchical
tail 較高；最限制 gate 是 non-anchor hierarchical q97.5
`3.9921512903134353`。這說明本輪決策不能只看單一 fixed-well q95。

Primary O5 shifted-minus-zero paired RMSE deltas：

| position mode | full delta q50 | full delta q95 | non-anchor delta q50 | non-anchor delta q95 |
| --- | ---: | ---: | ---: | ---: |
| coherent early | +0.000169 | +0.005089 | +0.000169 | +0.005097 |
| coherent late | -0.000066 | +0.006405 | -0.000067 | +0.006414 |
| independent bounded | +0.000335 | +0.007607 | +0.000336 | +0.007618 |

這些 paired deltas 只作 position effect 描述；本輪沒有新增 promotion
threshold。

## 4. Exact-clean 與 M5 controls 是否保持 robust？

是。Primary `T_u=0.001` 的全部三個 modes 都通過。

| mode | exact full/non q95 | exact hier full/non q97.5 | M5 full/non q95 | M5 hier full/non q97.5 |
| --- | --- | --- | --- | --- |
| coherent early | 2.676047 / 2.678236 | 2.786810 / 2.789062 | 3.242048 / 3.243544 | 3.310992 / 3.312579 |
| coherent late | 2.680386 / 2.682578 | 2.791341 / 2.793621 | 3.242779 / 3.244242 | 3.313267 / 3.314876 |
| independent bounded | 2.682565 / 2.684760 | 2.790279 / 2.792536 | 3.243534 / 3.245023 | 3.312460 / 3.314041 |

沒有 control 因 position tolerance 失去 robust status。

## 5. Full 與 non-anchor decisions 是否一致？

是。全部 30 個 tolerance/mode/scenario groups 的 full 與 non-anchor
pass/fail decisions 一致。

Non-anchor mask 依每個 achieved selection 排除八個不同 physical rows，
每 row 只排除一次；沒有 repeated-reading duplicate exclusion。

## 6. 各 tolerance 對應的 achieved MD displacement

以下是 achieved physical row 相對 nominal snapped physical row 的絕對 MD
位移。Anchor 欄是所有 achieved anchor acquisitions 的 pooled 分布；
well-max 欄是在每個 well／position realization 內先取八位置最大絕對位移，
再做分布。

| `T_u` | mode | anchor p50 | p90 | p95 | max | well-max p50 | p90 | p95 | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | nominal | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| .001 | coherent early | 5 | 6 | 7 | 11 | 5 | 7 | 7 | 11 |
| .001 | coherent late | 5 | 7 | 7 | 11 | 5 | 7 | 7 | 11 |
| .001 | independent bounded | 5 | 6 | 7 | 11 | 5 | 7 | 7 | 11 |
| .0025 | coherent early | 12 | 16 | 17 | 26 | 12 | 16 | 18 | 26 |
| .0025 | coherent late | 12 | 16 | 17 | 26 | 13 | 16 | 18 | 26 |
| .0025 | independent bounded | 12 | 16 | 17 | 26 | 12 | 16 | 18 | 26 |
| .005 | coherent early | 24 | 32 | 35 | 51 | 25 | 32 | 35 | 51 |
| .005 | coherent late | 24 | 32 | 35 | 51 | 25 | 32 | 35 | 51 |
| .005 | independent bounded | 24 | 32 | 35 | 51 | 25 | 32 | 35 | 51 |

單位皆為 ft。這些值包含 deterministic nearest-unused row snapping 的離散
效果；每 well 的 horizon 不同，所以相同 `u` 不對應同一個 ft 位移。

`position_exposure.csv` 另含每 tolerance/mode/seed 及每 anchor order 的
requested/achieved `u`、requested/achieved MD displacement、snap error、
violation counts 與 well-max summaries。`position_manifest.csv.gz` 保留每個
實際 achieved row、MD、`u` 與 revealed TVT。

這些是 computational levels，不是經現場 calibration 證明的真實硬體
tolerance。

## 7. 是否有 unavailable、duplicate、order 或 boundary violations？

沒有。

- boundary violations：`0`；
- availability violations：`0`；
- duplicate violations：`0`；
- order violations：`0`；
- wells with any violation：`0`。

所有 actual rows 都是八個 distinct、strictly ordered physical rows；沒有
clipping，也沒有在 violation 後搜尋替代位置。

## 8. 是否重新出現 RMSE >=40 或 >=60 wells？

沒有。全部 3,000 metric groups：

- maximum `wells_ge_40 = 0`；
- maximum `wells_ge_60 = 0`。

所有 primary O5 modes 也沒有 persistent fold failure。

## 9. Primary minimum hierarchical margin

`0.007848709686564703 ft`，來自 `independent_bounded` 的 non-anchor
hierarchical q97.5 `3.9921512903134353`。

此值也等於所有 primary O5 四項數值 gates 中的 overall minimum margin。

## 10. `0.0025/0.005` diagnostics

| tolerance | coherent early | coherent late | independent bounded | diagnostic decision |
| ---: | --- | --- | --- | --- |
| `.0025` | pass | pass | pass | all modes pass；minimum margin `0.005839` |
| `.005` | pass | pass | **fail** | diagnostic fail |

`T_u=.005` 的 `independent_bounded`：

- full/non-anchor fixed-well q95：`3.954901 / 3.955608`，仍 `<4`；
- hierarchical full/non-anchor q97.5：
  `4.000599 / 4.001315`，失敗；
- minimum margin：`-0.001315 ft`；
- 仍沒有 persistent fold 或 `>=40/60` catastrophic well failure。

`0.0025` 與 `0.005` 是事前固定 diagnostics；它們沒有被用來重新定義
primary `0.001`，也沒有發生 tolerance search。

## 11. O5 為何是 stochastic envelope，不是單筆硬 tolerance？

O5 定義的是一個隨機量測誤差族：

- 每 well 一個 common datum bias，`tau=1 ft`；
- 每 reading independent Gaussian error，`sigma=1 ft`；
- 每 reading 以 `p=.10` 發生 symmetric `±10 ft` gross error；
- 八位置各一筆 reading；
- 100 組 deterministic paired realizations 加上 10,000 paired
  hierarchical well/realization bootstraps。

因此 `tau/sigma/p/M` 描述的是 population／realization distribution，
不是「每一筆誤差保證不超過某 ft」的硬規格。尤其 Gaussian error 無有限
hard bound，而 M10 是以 10% 機率發生，不是每次必然或最多只發生一次。

## 12. 哪些 cross-location shared faults 仍未測試？

本輪沒有測試：

- 八個不同 locations 同時遭受共同 `+10` 或 `-10 ft` gross shift；
- 跨 locations 相關的 gross-error indicators 或 signs；
- 超出 `±T_u` bound-edge modes 的共同 position-calibration drift；
- common MD scale、datum、tool、survey 或 decoding faults；
- position error 與 TVT measurement gross error 的相關性；
- unavailable location、失序或硬體重試後的替代 acquisition policy。

先前的「shared-fault-safe」只表示 `near_k8_r1` 不依賴同位置 repeats 的
fault independence；它不代表已證明跨八位置共同 M10 fault 安全。

## 13. 為何仍不是 deployable competition model？

因為：

- revealed anchor value 仍直接取自 validation target `TVT`；
- reconstruction 使用未來 anchors，屬於 acausal reconstruction；
- PS boundary 被假設 exact、unperturbed；
- 本輪沒有建立 test-time 取得真實 TVT anchors 的 competition-safe
  feature pipeline；
- computational position tolerances 尚未由現場硬體 calibration 支持；
- base P3/path/blend OOF predictions 是固定 retrospective evidence；
- 結果只證明 counterfactual information specification，不證明線上
  acquisition、latency、cost、shared-fault 或 hidden-test deployability。

Evidence flags：

```text
deployable: false
submission_eligible: false
counterfactual_information_experiment: true
uses_validation_target_as_revealed_anchor: true
kaggle_submission_created: false
kaggle_submission_uploaded: false
```

## 14. 下一步

Primary protocol 已通過，因此下一步是：

**做獨立的、placement 與 protocol 完全固定的 `near_k8` stress
qualification。**

本輪不調 placement、anchor count、piecewise reconstruction、PS boundary、
QC threshold、outlier detector、measurement envelope 或 primary tolerance。

若未來現場 calibration 顯示 `tau` 或 `sigma` 超過 `1 ft`，則應提高 datum
calibration 的優先度；不能把本輪 O5 pass 外推到更高背景誤差。

## 實作與驗證

新增：

- `src/rogii/diagnostics/anchor_acquisition_protocol.py`
- `scripts/run_anchor_acquisition_protocol.py`
- `tests/test_anchor_acquisition_protocol.py`

Full artifacts：

- `protocol_frontier.csv`
- `per_seed_metrics.csv`
- `per_fold_seed_metrics.csv`
- `per_well_seed_metrics.csv.gz`
- `position_exposure.csv`
- `position_manifest.csv.gz`
- `hierarchical_bootstrap.json`
- `paired_deltas.csv`
- `run_config.json`
- `leakage_check.json`
- `summary.json`
- `report.md`

Artifact audit：

- evaluation groups／seed groups：30／3,000；
- compact per-well rows：2,319,000；
- position exposure rows：9,000；
- position manifest rows：1,898,488；
- pooled full/non-anchor reconstruction maximum delta：
  `1.33e-15 / 1.33e-15`；
- duplicate keys：`0`；
- unexpected numeric NaNs：`0`；
- selected achieved-anchor prediction對 corrupted observation maximum
  delta：`0`；
- analytic interval SSE coefficients 對 direct rowwise audit通過；
- no formation surfaces、`Geology`、PNG 或 typewell；
- horizontal inference columns 只使用
  `MD,X,Y,Z,GR,TVT_input`；
- no writes to `data/`；
- no submission CSV；
- no Kaggle upload。

最終驗證：

- targeted protocol tests：`20 passed in 15.69s`；
- 完整 test suite：`273 passed in 210.90s`；
- artifact audit-only rerun：pass；
- `rogii check --data-root data/rogii-wellbore-geology-prediction`：pass，
  773 train wells、3,783,989 train prediction rows，五 folds 為
  `155/155/155/154/154`；
- 正式 full run 與上述驗證均未建立 submission CSV、未上傳 Kaggle。
