# External Time-Series Model Portfolio

實驗快照：2026-07-28  
初稿組裝：2026-07-29  
狀態：**quick、full TCN、nested production、target-free stress 與完整測試均完成；
所有 external ensembles rejected；正式模型保留 incumbent**

## 1. 結論先行

目前可由完成且獨立重讀的 artifacts 支持的結論只有以下幾點：

1. **可靠基座仍是 Heel-Registered Sparse-Anchor Student 5-seed
   ensemble。**在 773 wells、3,783,989 OOF rows 的 full strict
   well-level CV 上，pooled row-RMSE 為 **14.1708478925 ft**；相對同 keys 的
   historical Anchor Student 改善 **-0.6492321981 ft**，10,000 次 paired
   well-cluster bootstrap CI 為
   **[-0.9247288343, -0.3772842000] ft**。
2. 已建立共同的 fold-local residual sequence harness，以及 Paired
   PatchTST-derived cross-encoder、S4D-inspired long convolution、1D FNO
   和 TCN 四個 architecture-only family。Quick run 沒有使用外部 pretrained
   weights 或外部訓練資料。
3. 在固定 100-well quick cohort 上，**TCN 是最佳 standalone
   architecture-only family**。未縮放 TCN 的 RMSE 是 **12.5655077279**，
   delta 是 **-0.2462391255**，但 CI
   `[-0.6629828233, 0.1536696166]` 跨 0。較保守的 0.25 correction
   shrink 得到 RMSE **12.6864712435**、delta **-0.1252756099**、CI
   **[-0.2368889896, -0.0204374815]**；這是本次 quick screen 唯一 CI
   上界低於 0 的 candidate。
4. **Quick signal 沒有在 full strict OOF 成立。**773 wells、3,783,989 rows、
   5 folds、5 seeds 的 TCN raw ensemble 為 **14.1989291091 ft**，相對
   incumbent **+0.0280812166 ft**，10k paired CI
   **[-0.1164036880, 0.1746635437]**；五個單 seed 全部較差，故 raw 路線失敗。
   固定 0.25 shrink 的 descriptive RMSE 為 **14.1365649645 ft**、delta
   **-0.0342829280 ft**，但 CI **[-0.0709915348, 0.0023756677]** 的上界仍
   大於 0，不能視為可靠 standalone gain。
5. 0.25 shrink 在 100,000 組隨機 200-well subsets 的 empirical
   `P(worse)` 是 **14.171%**，遠高於正式 ensemble 理想門檻 1%。它雖小幅改善
   centered、slope、zero-mean shape、terminal、p95 與 max，卻沒有足夠的
   paired uncertainty／private-size stability 證據。因此**不符合正式
   ensemble 標準**。
6. Paired PatchTST-derived cross-encoder 的未縮放版本明顯劣於 quick
   incumbent；多個破壞 pairing／phase 的 controls 在 0.25 shrink 後也可得到
   類似的小幅負 delta。因此目前**沒有證據可宣稱 paired model 學到可泛化的
   geological correspondence**。
7. 所有完成的 external-family experiments 都是 random-init
   architecture-only。沒有 pretrained checkpoint 被下載、載入或微調，所以目前
   可以談 architecture-only representation signal，**不能宣稱任何 external
   pretraining transfer gain**。
8. Full canonical publication 已用獨立稽核過的 Heel OOF 鎖定 keys、truth 與
   incumbent，並完成第二次 disk read、25 checkpoint provenance、10k bootstrap
   與 100k shadow-private 重算。這修補了 quick bundle 對 truth／incumbent
   reference 自鎖的 P1 限制。
9. Conditional nested run 已完成。Outer-training inner OOF 學到的 TCN weights
   為 0.2301–0.2831；untouched outer predictions
   的 RMSE 是 **14.1386932505 ft**、delta **-0.0321546420 ft**、10k paired CI
   **[-0.0701508816, 0.0057061456]**，100k 200-well `P(worse)` 為
   **16.557%**。Model fit 與 weight fit 是 outer-safe，但 TCN family/config
   是看過 global quick/full OOF 後才被選入，因此這是「給定已選 TCN」的
   conditional cross-fit，不是完整 nested family-selection proof；它同樣不符合
   正式標準。
10. Descriptive simple 0.5 blend 的 point estimate 最低：
    **14.1298180564 ft**、delta **-0.0410298361 ft**；但 CI
    **[-0.1141257306, 0.0319489407]** 且 `P(worse)=26.130%`。固定 0.25
    control 也未通過。**所有 external ensembles 均 rejected，production
    保持 incumbent。**
11. Nested runner 與 independent disk re-audit 均通過；target-free stress
    完成 30 features、139 strata、695 metric rows。Nested 最佳 stratum delta
    為 -0.12288148，最差為 +0.11079374，顯示收益具有明顯 composition
    heterogeneity，進一步支持 rejection。
12. 先前完整 pytest 連續兩次為 430 passed；補入 weight stability、
    noise-floor parser 與 family-selection safety tests 後，最終為
    **443 passed、37 warnings、230.11 s**；delivery-scope
    Ruff、format、compileall 與 diff checks 均 clean。Full-repo Ruff 仍有 22 個
    既有／範圍外 findings，不能描述為 repo-wide lint clean。`data/` manifest
    before／after 相同，沒有建立或上傳 Kaggle submission。

## 2. 證據邊界與 population

本報告把不同 population 明確分開：

| Evidence | Population | Seeds | 用途 |
|---|---:|---:|---|
| Heel-Registered incumbent full audit | 773 wells / 3,783,989 rows / folds 0–4 | 5 | 正式基座 |
| External portfolio quick | 100 wells / 489,284 rows / folds 0–4 | 1 | candidate screening 與 controls |
| TCN full strict OOF | 773 wells / 3,783,989 rows / folds 0–4 | 5 | standalone full evidence；raw fail、0.25 descriptive only |
| Nested incumbent＋TCN | 773 wells / 3,783,989 rows / 5 outer × 4 inner splits | 5/family/split | 給定已選 TCN 的 conditional nested weight evaluation；全部 rejected |
| Visible-test smoke v2 | 3 dynamically enumerated test wells / 14,151 rows | 25 full checkpoints | 離線推論契約；不是 accuracy evidence |

Quick cohort 的 incumbent RMSE 是 **12.8117468534**，只可作同一 quick keys
上的 comparator。它不能和 773-well full RMSE 14.1708478925 作絕對高低解讀。
Noise-floor notebook 對 stochastic public pipelines 量到約 0.03 ft 等級的
rerun scatter。`noise_floor_reaudit_20260729.json` 已從 notebook cells 1–5
獨立重算 A/B sample SD（0.0366833／0.0268701）、variance ratio
1.863804、one-sided F(3,3) 95% critical 9.276628、兩種 SE
0.0523259／0.0413673 與六個 z 值，全部通過。Notebook 的 9.3 是 one-sided
critical；two-sided 5% upper critical 是 15.439182，兩者都不改變定性結論。
這不是本地 OOF 的誤差棒，也不被拿來替代 well-cluster bootstrap；作用只是提醒
單次 leaderboard 或單一 seed 的小差異不能當作穩定 lever。含 train-label
oracle 的其他 notebook cells 沒有被執行或當作 deployable evidence。Re-audit
SHA-256 是
`0397d86667c3c2a88a7f81a6a55a8d836ed52be516c155309d30f676ddc649e5`。

主要本機證據：

- [`external_anchor_research_20260728.md`](external_anchor_research_20260728.md)
- [`external_model_registry_20260728.md`](external_model_registry_20260728.md)
- [`external_model_environment_20260729.json`](external_model_environment_20260729.json)
- [`measure-your-noise-floor-before-believing-a-lever.ipynb`](measure-your-noise-floor-before-believing-a-lever.ipynb)
- [`noise_floor_reaudit_20260729.json`](noise_floor_reaudit_20260729.json)
- `.runs/heel-registered-anchor-full-20260728/summary.json`
- `.runs/heel-registered-anchor-full-20260728/independent_reaudit.json`
- `.runs/external-model-portfolio-quick-20260728/run_config.json`
- `.runs/external-model-portfolio-quick-20260728/model_manifest.json`
- `.runs/external-model-portfolio-quick-20260728/external_model_artifacts/manifest.json`
- `.runs/external-model-portfolio-quick-20260728/independent_portfolio_reaudit.json`
- `.runs/external-model-tcn-controls-quick-20260728/summary.json`
- `.runs/external-model-tcn-full-20260728/summary.json`
- `.runs/external-model-tcn-full-20260728/metrics/`
- `.runs/external-model-tcn-full-20260728/external_model_artifacts/manifest.json`
- `.runs/external-model-tcn-full-20260728/independent_portfolio_reaudit.json`
- `.runs/external-model-tcn-full-20260728/visible_test_smoke_all_folds_v2.json`
- `.runs/nested-external-tcn-full-20260729/summary.json`
- `.runs/nested-external-tcn-full-20260729/nested_weights.csv`
- `.runs/nested-external-tcn-full-20260729/artifact_audit.json`
- `.runs/nested-external-tcn-full-20260729/independent_nested_reaudit.json`
- `.runs/nested-external-tcn-full-20260729/weight_stability_v2/summary.json`
- `.runs/nested-external-tcn-full-20260729/weight_stability_v2/weight_stability.csv`
- `.runs/nested-external-tcn-full-20260729/stress/summary.json`
- `.runs/nested-external-tcn-full-20260729/stress/target_free_stratified_metrics.csv`

所有 full TCN 正式數字以下以 full `independent_portfolio_reaudit.json` 的
canonical disk re-read 為準；nested 數字以 `independent_nested_reaudit.json`
的 disk-only recomputation 為準，而不是 runner 的記憶體內物件。Quick re-read
仍支持 serialization、component algebra 與 matched in-cohort metrics，但其
truth／incumbent 外部鎖有下述 P1 限制，不能冒充 hardened full audit。

## 3. Incumbent baseline audit

### 3.1 Full strict OOF

| Candidate | Pooled row-RMSE | Delta vs historical Anchor |
|---|---:|---:|
| Historical Anchor Student | 14.8200800906 | 0 |
| Matched 5-seed unregistered Anchor ensemble | 14.7880434145 | -0.0320366760 |
| **Heel-Registered 5-seed ensemble** | **14.1708478925** | **-0.6492321981** |

Heel-Registered vs historical Anchor 的 10k paired well bootstrap CI 是
`[-0.9247288343, -0.3772842000]`。為分離 seed averaging 與 registration，
同 seed 的 matched unregistered ensemble 作為 comparator；registration net
delta 是 **-0.6171955220**，CI
`[-0.8924450356, -0.3506084569]`。五個 folds 與五個 seeds 的 matched
方向全部改善。

Independent re-audit 從 disk 重算得到：

- 773 wells、3,783,989 rows；
- duplicate/non-monotone keys：0；
- non-finite values：0；
- 25 個 fold×seed model states 均通過 provenance；
- summary 與重算 RMSE 絕對差：0；
- bootstrap 重算最大絕對差：0；
- test inference capable：true；
- submission CSV created：false。

### 3.2 Full tail 與 residual decomposition

| Metric | Historical Anchor | Heel-Registered |
|---|---:|---:|
| Common-mode RMSE | 12.467902 | **11.613421** |
| Centered-shape RMSE | 8.834977 | 8.825260 |
| Terminal-anchor RMSE | 19.230152 | **18.579584** |
| Per-well p95 RMSE | **26.624218** | 26.690231 |
| Max well RMSE | 63.699058 | **54.779067** |
| Wells RMSE ≥ 40 ft | 12 | **10** |
| Wells RMSE ≥ 60 ft | 2 | **0** |
| Worst-10% SSE share | 0.508357 | **0.490065** |

Registration 的大部分收益來自 common mode；centered shape 幾乎沒有改變。
p95 惡化約 0.066 ft，故不能宣稱所有 tail quantiles 都改善。這也是 external
sequence portfolio 將 slope／zero-mean smooth shape 設為主要 target 的原因。

本節的 common／centered／terminal 是每井 **8 個 sparse-anchor errors** 的
anchor-space diagnostic。Sections 7–8 的 decomposition 則由
`portfolio_metrics` 對全部 3,783,989 OOF row residual 計算 per-well
constant／linear／zero-mean shape，terminal 是每井最後一個 prediction row。
兩者 population 與定義不同，因此 Section 3 的 11.613421／8.825260／18.579584
不可直接和 Sections 7–8 的 incumbent 11.041234／8.882797／19.227242 比較。

## 4. External model registry 與 provenance

完整版本、weight identity、license、pretraining corpus 與 overlap audit 見
[`external_model_registry_20260728.md`](external_model_registry_20260728.md)。
下表只列本輪決策所需的 frozen identities：

| Family | Official snapshot | License | 本輪狀態 |
|---|---|---|---|
| PatchTST | [`yuqinie98/PatchTST@204c21e`](https://github.com/yuqinie98/PatchTST/commit/204c21efe0b39603ad6e2ca640ef5896646ab1a9) | Apache-2.0 | Derived paired cross-encoder；architecture-only quick complete |
| S4 / S4D | [`state-spaces/s4@e757cef`](https://github.com/state-spaces/s4/commit/e757cef57d89e448c413de7325ed5601aceaac13) | Apache-2.0 | S4D-inspired pure-PyTorch quick complete |
| NeuralOperator / FNO | [`neuraloperator@86a8bc7`](https://github.com/neuraloperator/neuraloperator/commit/86a8bc7812a31b42c4f7895693cf4ac11521c066) | MIT | 1D FNO architecture-only quick complete |
| TCN | [`locuslab/TCN@2f8c2b8`](https://github.com/locuslab/TCN/commit/2f8c2b817050206397458dfd1f5a25ce8a32fe65) | MIT | Cheap architecture control；quick signal |
| Mamba | [`state-spaces/mamba@e9594ce`](https://github.com/state-spaces/mamba/commit/e9594ce1c732d97440f0332fdc43170a2294dbfa) | Apache-2.0 | Current Windows CPU route deferred/rejected |
| MOMENT | [`moment@38f7310`](https://github.com/moment-timeseries-foundation-model/moment/commit/38f7310ad594100747ca2a8357e9c7ca7d323e0e) | MIT | Pretrained route held；未下載 |
| Granite TTM | [`granite-tsfm@31c7a6b`](https://github.com/ibm-granite/granite-tsfm/commit/31c7a6bd640c41e53777c0afe3a13d24e00f7fe7) | Apache-2.0 | Pretrained scouting candidate；未執行 |
| TimesFM | [`timesfm@3dae50b`](https://github.com/google-research/timesfm/commit/3dae50b20d7a724981e8ea36cda75578f80dd2dc) | Apache-2.0 | Low-priority scouting；未下載 |
| Chronos | [`chronos@7dc4435`](https://github.com/amazon-science/chronos-forecasting/commit/7dc4435706a4454feb79df44ca9f33631f3027bf) | Apache-2.0 | Low-cost frozen probe candidate；未下載 |

已登錄但未下載的 representative weights 包括：

- MOMENT-1-small revision
  `411e288267f82cce86296dbe4d6c8bc533cc162f`，37.9M parameters，
  151,615,328-byte safetensors；
- MOMENT-1-base revision
  `5e44b0ea26376a176360f87831124e018f876d96`，113.5M parameters，
  453,940,120-byte safetensors；
- Granite TTM R2 revision
  `d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f`，805,280 parameters，
  3,240,592-byte safetensors；
- TimesFM 2.5 revision
  `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`，231.3M Hub-metadata
  parameters，925,181,104-byte safetensors；
- Chronos-Bolt-tiny revision
  `a0e552de83495b5c28c14c71c374f3e33280b340`，8.7M parameters，
  34,622,352-byte safetensors。

Registry 的重要 provenance 限制：

- MOMENT Time-series Pile 明確包含 UCR `Rock` 岩石光譜資料。這不是已知的
  well-log、GR 或 competition overlap，但屬 geology-adjacent exposure；不能把
  pretrained MOMENT 描述為沒有 geology exposure。
- TimesFM 2.5 與 Chronos-Bolt 的 exact training composition 沒有完整披露；
  「未列出 geology」不是 absence proof。
- ChronoLog GR 已被本專案判定為高度 correlation／possible-overlap risk，
  **禁止**用於 training、pretraining、retrieval、features 或 model selection。
- Mamba text weights 的 input stem、task 與 text-Pile provenance 不適合本任務，
  未採用。
- 本輪外部下載為 0 bytes；執行環境為 Windows、Python 3.12.10、
  `torch==2.12.1+cpu`。

## 5. Shared residual sequence harness

共同 prediction contract 是：

```text
prediction
  = heel_registered_incumbent
  + predicted_common
  + predicted_slope
  + predicted_zero_mean_smooth_shape
```

Quick run 的 frozen contract：

- horizontal source columns：`MD, X, Y, Z, GR, TVT_input`；
- typewell source columns：`TVT, GR`；
- formation columns、Geology columns、image inputs、target columns：空；
- horizontal/typewell 都 deterministic resample 至 256 points；
- residual target 是 fold-safe incumbent residual 的
  constant + zero-mean linear + zero-mean slope-orthogonal shape；
- shape 使用 orthonormal DCT-II low-pass，保留 16 modes；
- correction 最大絕對值限制 80 ft；
- validation labels used at prediction：false；
- 每個 outer fold 的 scaler、target representation 與 model 只 fit 該 fold
  training wells；
- model input cache policy：never cached；
- device：CPU；epochs：30；hidden dimension：16；layers：2。

Quick instantiated model sizes：

| Variant | Fold checkpoints | Total parameters | Trainable parameters |
|---|---:|---:|---:|
| Paired Patch cross | 5 | 106,112 | 106,112 |
| S4D | 5 | 13,552 | 13,552 |
| FNO1D | 5 | 25,728 | 25,728 |
| TCN | 5 | 16,016 | 16,016 |
| Each paired negative control | 5 | 106,112 | 106,112 |
| Paired frozen-random encoder | 5 | 106,112 | 73,984 |

`model_manifest.json` 記錄 50 checkpoints、0 failures、0 incomplete family
seeds、`identity_fallbacks_scored=false`。Canonical publication 另要求：

- exact OOF keys、truth 與 incumbent hash；
- 每個 candidate 的 common/slope/shape/correction/final algebra；
- 每個 fold×seed checkpoint 的 path、bytes、SHA-256、fit IDs 與 state manifest；
- controls 和 frozen-random candidate 明確
  `eligible_for_model_selection=false`；
- incomplete checkpoint 或 non-finite output fail closed，不產生完成 bundle。

主要實作：

- `src/rogii/external_models/contracts.py`
- `src/rogii/external_models/targets.py`
- `src/rogii/external_models/torch_families.py`
- `src/rogii/external_models/artifacts.py`
- `src/rogii/external_models/portfolio_publisher.py`
- `src/rogii/external_models/portfolio_reaudit.py`
- `src/rogii/diagnostics/external_model_portfolio_experiment.py`

## 6. Quick matched experiment

設定：

- 100 unique wells、489,284 OOF rows；
- folds 0–4，每 fold 100 training wells、20 validation wells、intersection 0；
- seed `20260728`；
- incumbent、所有 candidates 與 controls 使用相同 keys、folds、rows；
- family mean 在單 seed quick 中等於該 seed prediction；
- 同時保存 full correction 與固定 0.25 shrink；
- paired well-cluster bootstrap：10,000 samples，seed `20260728`。

這是 screening cohort，不是 full evidence，也沒有 seed-stability evidence。

### 6.1 Main families

Quick incumbent：RMSE **12.8117468534**。

| Candidate | RMSE | Delta | 10k paired CI | 判讀 |
|---|---:|---:|---:|---|
| Paired Patch cross | 13.619627 | +0.807880 | [0.092098, 1.556890] | 明顯較差 |
| Paired Patch cross ×0.25 | 12.718559 | -0.093188 | [-0.283569, 0.102778] | CI 跨 0 |
| S4D | 12.730643 | -0.081104 | [-0.505860, 0.352042] | CI 跨 0 |
| S4D ×0.25 | 12.729436 | -0.082311 | [-0.187954, 0.025120] | CI 跨 0 |
| FNO1D | 12.981724 | +0.169977 | [-0.328974, 0.694001] | 無可信 signal |
| FNO1D ×0.25 | 12.772424 | -0.039322 | [-0.166825, 0.093522] | CI 跨 0 |
| **TCN** | **12.565508** | **-0.246239** | **[-0.662983, 0.153670]** | 最佳 point estimate，CI 跨 0 |
| **TCN ×0.25** | **12.686471** | **-0.125276** | **[-0.236889, -0.020437]** | 唯一 quick CI 上界 < 0 |

### 6.2 Negative controls

Paired Patch 的下列 candidates 在 canonical bundle 中均為 ineligible：

| Control | Full delta / CI | 0.25-shrink delta / CI |
|---|---:|---:|
| No cross-series interaction | +1.058979 / [0.179121, 1.999206] | +0.025052 / [-0.206238, 0.269824] |
| Wrong typewell | +2.114419 / [0.937199, 3.419324] | +0.225616 / [-0.016730, 0.497272] |
| Pair shuffle | +1.075934 / [0.083483, 2.061848] | +0.006875 / [-0.234904, 0.250164] |
| GR-order shuffle | +0.616428 / [-0.157814, 1.375240] | -0.121473 / [-0.333200, 0.087348] |
| Phase destroy | +0.866814 / [0.159406, 1.635735] | -0.082330 / [-0.270295, 0.113141] |
| Frozen random encoder | +0.377328 / [-0.356155, 1.078434] | -0.086844 / [-0.275889, 0.095152] |

未縮放 controls 大多惡化，證明大 correction 不安全；但 shrink 後
GR-order/phase/frozen-random controls 也可出現與 paired-cross shrink 相近的負
delta，而且全部 CI 跨 0。這使 paired cross 的小幅 shrunk 改善不能被解讀為
geological correspondence。Controls 不可進入 model selection 或 ensemble。

TCN 另以完全相同的 100-well cohort、folds、rows 與 seed 執行
`.runs/external-model-tcn-controls-quick-20260728`。這個 run 同樣只作
screening，所有 controls 都是 ineligible：

| TCN variant | Full delta / CI | 0.25-shrink delta / CI |
|---|---:|---:|
| **Intact TCN** | **-0.246239 / [-0.662983, 0.153670]** | **-0.125276 / [-0.236889, -0.020437]** |
| Wrong typewell | +0.215570 / [-0.371991, 0.791451] | -0.031903 / [-0.179765, 0.111274] |
| Pair shuffle | +0.120925 / [-0.432119, 0.648940] | -0.040184 / [-0.193409, 0.094394] |
| GR-order shuffle | -0.167619 / [-0.568237, 0.193128] | -0.094903 / [-0.201807, 0.000238] |
| Phase destroy | +0.096754 / [-0.311231, 0.484862] | -0.027971 / [-0.133797, 0.071730] |
| Frozen random encoder | +0.058347 / [-0.331242, 0.458033] | -0.061359 / [-0.161678, 0.039202] |

Intact TCN 的 point gain 大於 wrong-typewell、pair-shuffle、phase-destroy 與
frozen-random controls，支持有限的 learned architecture／pairing signal；
但 GR-order shuffle 仍保留 -0.167619 ft 的 raw point gain，而且所有 raw
control CI 都跨 0。這表示合法 trajectory／incumbent channels 本身可產生部分
signal，不能把 TCN quick gain 全部歸因於 GR geological correspondence。

### 6.3 Tail 與 shape

| Metric | Quick incumbent | TCN | TCN ×0.25 |
|---|---:|---:|---:|
| Pooled RMSE | 12.811747 | **12.565508** | 12.686471 |
| Centered RMSE | 8.133918 | **8.033488** | 8.068172 |
| Slope-component RMSE | 5.173092 | **4.918130** | 5.075958 |
| Zero-mean shape RMSE | **6.276921** | 6.352080 | **6.271368** |
| Terminal RMSE | 21.244474 | **20.526943** | 20.988038 |
| Per-well p95 | 21.905939 | **21.131895** | 21.468840 |
| Max well RMSE | **35.011139** | 35.332932 | 35.089759 |
| Wells ≥40 / ≥60 ft | 0 / 0 | 0 / 0 | 0 / 0 |

未縮放 TCN 改善 centered、slope、terminal 與 p95，但 zero-mean shape 和 max
略差。0.25 shrink 的 shape RMSE 比 incumbent 低約 0.0056 ft，max 僅惡化約
0.0786 ft。這是值得 full 驗證的 signal，但 100-well／1-seed quick 不能證明
private stability。

## 7. Full strict TCN standalone evidence

### 7.1 Scope 與 primary result

Full run `external-model-tcn-full-20260728` 使用 incumbent 的相同 773 wells、
3,783,989 rows 與 folds 0–4。五個 seeds 是 `20260711`、`20260712`、
`20260713`、`20260714`、`20260715`，共 25 個 fold×seed checkpoints；
failure count 為 0，identity fallback rows 為 0。兩個 published candidates
均是同一 TCN family 的 architecture-only 5-seed mean：raw correction 與預先
固定的 0.25 shrink。

| Candidate | Pooled row-RMSE | Delta vs incumbent | 10k paired well-bootstrap CI |
|---|---:|---:|---:|
| Heel-Registered incumbent | 14.1708478925 | 0 | — |
| **TCN raw 5-seed mean** | **14.1989291091** | **+0.0280812166** | **[-0.1164036880, 0.1746635437]** |
| **TCN ×0.25** | **14.1365649645** | **-0.0342829280** | **[-0.0709915348, 0.0023756677]** |

結論沒有模糊空間：**raw TCN 失敗**。0.25 shrink 只提供 descriptive
改善；其 paired CI 上界為正，不能主張穩定 standalone gain，更不能因為 point
estimate 較低就替換 incumbent。

### 7.2 Five-fold 與 five-seed stability

Family-mean 的 fold deltas：

| Outer fold | Incumbent RMSE | TCN raw delta | TCN ×0.25 delta |
|---:|---:|---:|---:|
| 0 | 14.077945 | +0.165791 | +0.008166 |
| 1 | 12.955540 | +0.060955 | -0.041550 |
| 2 | 13.704693 | -0.279985 | -0.103396 |
| 3 | 15.271421 | +0.135447 | -0.014717 |
| 4 | 14.758727 | +0.044065 | -0.023719 |

Raw 只在 fold 2 改善，另外四 folds 變差；0.25 shrink 在四 folds 改善，但
fold 0 仍小幅變差。Raw 單 seed 的 pooled 結果：

| Seed | RMSE | Delta vs incumbent |
|---:|---:|---:|
| 20260711 | 14.250493 | +0.079645 |
| 20260712 | 14.268689 | +0.097841 |
| 20260713 | 14.218343 | +0.047495 |
| 20260714 | 14.290703 | +0.119856 |
| 20260715 | 14.236606 | +0.065758 |

五個單 seed 全部比 incumbent 差。Seed averaging 把 raw delta 降到
`+0.028081`，但沒有把方向翻成改善。0.25 shrink 是對完成的 family mean
correction 作固定縮放；canonical metrics 沒有另造 per-seed shrink 欄位，因此
不把不存在的 per-seed shrink 數字寫入報告。

### 7.3 Tail

| Metric | Incumbent | TCN raw | TCN ×0.25 |
|---|---:|---:|---:|
| Per-well p50 RMSE | **9.796211** | 9.993124 | 9.894366 |
| Per-well p90 RMSE | 20.331417 | **20.266504** | 20.288402 |
| Per-well p95 RMSE | 26.690231 | 26.657243 | **26.632093** |
| Max well RMSE | 54.779067 | 54.472003 | **54.454843** |
| Wells RMSE ≥40 ft | 10 | **9** | 10 |
| Wells RMSE ≥60 ft | 0 | 0 | 0 |
| Worst-10% SSE share | 0.490065 | **0.483050** | 0.489334 |

Raw 的 pooled RMSE 雖失敗，p90／p95／max、≥40 count 與 worst-10% share
沒有 catastrophic 惡化；它的主要代價出現在 p50 與 common mode。0.25 shrink
維持 10／0 catastrophic counts，且 p95 與 max 分別改善約 0.0581／0.3242 ft。
這代表 tail 安全性沒有否決 shrink，但也不能抵消其 CI 與 private-size
stability 失敗。

### 7.4 Common／slope／shape decomposition

| Metric | Incumbent | TCN raw | TCN ×0.25 |
|---|---:|---:|---:|
| Common-mode RMSE | 11.041234 | 11.129932 | **11.030624** |
| Centered RMSE | 8.882797 | **8.816700** | 8.841256 |
| Slope-component RMSE | 5.930760 | **5.864827** | 5.885231 |
| Zero-mean shape RMSE | 6.612879 | **6.583160** | 6.597868 |
| Terminal RMSE | 19.227242 | 19.246977 | **19.166049** |
| Well-slope coefficient RMSE | 10.355509 | **10.196666** | 10.264734 |

TCN 確實提供不同於 incumbent 的 slope／centered-shape representation：
raw centered、slope、zero-mean shape 與 well-slope coefficient 都改善，但
common mode 與 terminal 變差，導致 pooled loss。0.25 shrink 將 common 與
terminal 方向修正，六項 decomposition 都比 incumbent 小；幅度仍不足以產生
paired-CI 明確的總體收益。這是 architecture-only representation signal，
不是 pretrained transfer signal。

### 7.5 Residual diversity、wins 與 200-well shadow-private

Row residual correlation：

| Pair | Correlation |
|---|---:|
| Incumbent vs TCN raw | 0.984499 |
| Incumbent vs TCN ×0.25 | 0.999030 |
| TCN raw vs TCN ×0.25 | 0.991268 |

Raw 有一些 residual diversity，但平均方向不利；shrink 與 incumbent 幾乎完全
相關，故可預期 ensemble marginal headroom 很小。Per-well wins：

| Candidate | Wins / ties / losses | Win fraction | Mean well-RMSE delta | Median well-RMSE delta |
|---|---:|---:|---:|---:|
| TCN raw | 352 / 0 / 421 | 45.537% | +0.064806 | +0.095226 |
| TCN ×0.25 | 386 / 0 / 387 | 49.935% | -0.025058 | +0.000174 |

100,000 組固定 200-well、without-replacement shadow-private subsets：

| Candidate | Mean delta | P(worse) | q01 | q05 | q50 | q95 | q99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TCN raw | +0.028651 | 58.927% | -0.266493 | -0.180339 | +0.028908 | +0.236916 | +0.321407 |
| TCN ×0.25 | -0.034187 | **14.171%** | -0.108569 | -0.086577 | -0.034084 | +0.017817 | +0.039154 |

這個 empirical simulation 只涵蓋現有 train-well composition，不是 domain-shift
保證。即使如此，shrink 的 14.171% 已遠高於預設理想條件 `P(worse)<1%`；
因此它不符合正式 ensemble stability 標準。

### 7.6 Canonical hashes、P1 hardening 與 independent audit

Full canonical bundle：

- path：`.runs/external-model-tcn-full-20260728/external_model_artifacts`；
- rows／wells／folds：3,783,989／773／0–4；
- candidates：2 eligible、0 ineligible；source checkpoints：25；
- manifest SHA-256：
  `669cca7ebc53f78986a94e5abd79d4b5a68753f91d66cd292cce7164ee24c46d`；
- canonical OOF SHA-256：
  `09e780ff7044da750aaa122d97bea9d6e5e987793c974c77ae9ff95be53eaa9d`；
- canonical key SHA-256：
  `9d1d1527eec52b31bd4b23cc81ab582ea6fbb2db7636f9c538aa1721c1e33c47`；
- truth SHA-256：
  `5749d021ddac98c1b697af1d8b33d4aead775f8d6418bd7a9dbc381799a6c615`；
- incumbent-vector SHA-256：
  `b491fa7f809bd1d13fa576882aa9d7c8c7ef5be493fb05f194612c9025fef33c`；
- source wide-OOF SHA-256：
  `796abf6b83158c1652f3779b0a42f76cdb46324941edfd4ddc3eb05deb1a1118`。

P1 hardening 的核心是：publisher 不再從待發布的 candidate frame 自行建立
truth／incumbent expected reference，而是載入既有 Heel full OOF，先驗證其
sibling independent audit，再按 exact keys 比對；任何 drift 先 fail closed，
通過後才用 external reference 固化 canonical truth 與 incumbent。Full manifest
與 re-audit 都鎖定：

- audited incumbent OOF SHA-256：
  `4606cf8e07973f801f3612e9cfe9ecc11588ee7902573f085b350549f0ea60ba`；
- incumbent sibling re-audit SHA-256：
  `dc4ae9c27107de45689c26f7438dc2f7ca8a1b4aa97580f83af6ce4eade15ce0`；
- locked scope：3,783,989 rows、773 wells、folds 0–4、0 duplicate keys；
- runner summary SHA-256：
  `e0b89f76be2226fc543fd5f5ad4fe3526e3f116d7eefa741fec67c9b6f705e72`。

Independent re-audit 從 disk 通過 canonical schema、exact externally locked
keys/truth/incumbent、finite／duplicate、fold coverage/disjointness、component
algebra、25 checkpoint SHA/size/fit scope、fold-config hash、data manifest 與
no-submission checks；第二次 disk read 與 runner-summary comparison 也通過。
Before／after／current `data/` manifest digest 都是
`dec44ac09997b60e900c70064349f0fc9a1d558a9cda5c05aaa3530902fe4514`。
Independent report 本身 SHA-256 為
`50b277e3a83875be096d1cd45ba8685a00601eae37c06e250a5069f47dd1ee67`，
`passed=true`、`submission_created=false`。

Audited Heel artifact 早於 shared canonical-key helper：舊 re-audit 鎖的是
CSV file order `(fold, well_id, row_index)`，新 contract 鎖 canonical
`(well_id, row_index, fold)`。Loader 的 legacy-key 修正只在 exact incumbent
file SHA 與 sibling audit 都已通過後接受這兩種明確記錄的 digest，隨後固定排序
成 canonical keys；其他 digest、stale audit 或 changed bytes 全部 fail closed。
Focused regression：

```text
python -m pytest -q tests/test_external_artifacts.py \
  -k "audited_incumbent_accepts_both_locked_key_conventions or audited_incumbent_wrong_digest_or_changed_file_fails_closed"
3 passed, 8 deselected in 3.32s
```

Nested production 前的 read-only preflight execution log 顯示
`python scripts/run_nested_external_ensemble.py --preflight-only ...` exit 0：
rows 3,783,989、wells 773、`evidence_eligible=true`、`run_created=false`，
source manifest／OOF hashes 分別等於上述 `669cca...`／`09e780...`，並通過
25 outer-checkpoint source gate。Preflight 本身只證明 source 可用；其後完成的
conditional run 與獨立 disk re-audit 才是 Section 8 的 weight-crossfit evidence。

## 8. Conditional nested clean-ensemble evidence

### 8.1 Contract、compute 與 physical-fit accounting

Run `nested-external-tcn-full-20260729` 是給定已選 TCN 的 conditional nested
weight evidence：
`completed=true`、`evidence_eligible=true`，涵蓋 773 wells、3,783,989 rows、
5 個 outer folds 與 20 個 outer-training inner splits。每個 outer fold 的
validation wells 都不參與 inner model fit 或 blend-weight fit；inner predictions
先在 family 內平均五 seeds，再學 nonnegative、sum-to-one 的 incumbent／TCN
weights。TCN weight 上限是 0.5，regularization strength 固定為 **25.0**，
regularization target 才是 incumbent/TCN **0.75/0.25**。

總 elapsed time 為 10,997.546 秒，即 **3.055 hours（約 3.06 h）**。由 reciprocal
inner-split reuse 節省重複 fit：

- incumbent：100 checkpoint records = **50 physical fits + 50 reciprocal
  aliases**；
- TCN：100 checkpoint records = **50 physical fits + 50 reciprocal aliases**；
- 合計 200 records、100 physical fits、100 aliases。

Runner 與 independent audit 都驗證 alias 的 predictions、checkpoint/sidecar
hashes 與原 physical fit 完全等價；alias 不是額外 seed，也沒有被重複計權。

### 8.2 Nested primary 與 fixed controls

| Candidate | Pooled row-RMSE | Delta vs incumbent | 10k paired well-bootstrap CI | Evidence role |
|---|---:|---:|---:|---|
| Heel-Registered incumbent | 14.1708478925 | 0 | — | production baseline |
| Raw TCN | 14.1989291091 | +0.0280812166 | [-0.1164036880, 0.1746635437] | failed standalone |
| **Nested learned weights** | **14.1386932505** | **-0.0321546420** | **[-0.0701508816, 0.0057061456]** | primary nested evidence |
| Fixed incumbent/TCN 0.75/0.25 | 14.1365649645 | -0.0342829280 | [-0.0709915348, 0.0023756677] | preregistered fixed control |
| Simple incumbent/TCN 0.50/0.50 | **14.1298180564** | **-0.0410298361** | **[-0.1141257306, 0.0319489407]** | descriptive simple-average control |

Nested primary 的 CI 跨 0。Simple 0.5 雖有最低 descriptive point estimate，
在三個 blends 中 CI 最寬且明顯跨 0；它不是 learned nested primary，也不能用
point estimate 取代 uncertainty。Fixed 0.25 同樣未通過。**所有 external
blends rejected；production prediction 保持未混入 TCN 的 incumbent。**

### 8.3 Fold weights 與 stability

| Outer fold | Incumbent weight | TCN weight | Inner RMSE | Outer nested RMSE | Outer delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.716859 | 0.283141 | 14.117605 | 14.088869 | +0.010924 |
| 1 | 0.752529 | 0.247471 | 14.459315 | 12.914220 | -0.041319 |
| 2 | 0.769934 | 0.230066 | 14.413138 | 13.608739 | -0.095953 |
| 3 | 0.726565 | 0.273435 | 14.047755 | 15.256991 | -0.014430 |
| 4 | 0.752362 | 0.247638 | 14.106130 | 14.735124 | -0.023603 |

TCN point-weight mean 為 **0.256350**、population SD **0.019263**、range
**[0.230066, 0.283141]**（CV 7.51%）。另由
`weight_stability_v2/summary.json` 只讀 20 個 persisted inner OOF、以 well
cluster 各 fold 重抽 10,000 次；它沒有解析 final/outer OOF labels，point fit 與
independent re-audit weights 的最大差為 `3.08e-13`。TCN weight 分布為：

| Scope | Samples | Mean / SD | q01 | q05 | q50 | q95 | q99 | 0 / 0.5 boundary hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pooled five fold-specific fits | 50,000 | 0.256119 / 0.029080 | 0.190593 | 0.208075 | 0.256373 | 0.303049 | 0.319188 | 0 / 0 |
| Sample-wise five-fold mean | 10,000 | 0.256119 / 0.009699 | 0.233910 | 0.240181 | 0.256150 | 0.272110 | 0.278850 | 0 / 0 |

各 fold bootstrap SD 為 0.018860–0.025095，全部 50,000 draws 都沒有撞到
TCN cap 0.5 或 lower bound 0。Weight 數值確實穩定；但 outer delta 仍只有
4/5 folds 改善、paired CI 跨 0，而且 private-size failure probability 高。
這證明「weight 數值穩定」不等於「outcome 足夠穩定」。此分析也明確標成
`conditional_weight_crossfit`、`family_selection_safe=false`。
Weight-stability summary／CSV SHA-256 分別是
`d4ddc17c603403316fb5c4517d9e16033eab664302d19b3cdeabc2eeac362f1b`／
`edf7e33c302b83e661a6810a9d1b532509347c2b319f3d8bb153bba1445c1934`。
較早的 `weight_stability/` 輸出保留作不可覆寫的 pre-hardening trace，但它沿用
legacy `evidence_eligible=true`，不作正式引用；本報告只以
`weight_stability_v2/` 為權威。

### 8.4 Tail 與 decomposition

| Tail metric | Incumbent | Nested | Fixed 0.25 | Simple 0.5 |
|---|---:|---:|---:|---:|
| Per-well p50 RMSE | **9.796211** | 9.902998 | 9.894366 | 9.974891 |
| Per-well p90 RMSE | 20.331417 | 20.289194 | 20.288402 | **19.976138** |
| Per-well p95 RMSE | 26.690231 | 26.618365 | 26.632093 | **26.555370** |
| Max well RMSE | 54.779067 | **54.454794** | 54.454843 | 54.460188 |
| Wells RMSE ≥40 ft | 10 | 10 | 10 | **9** |
| Wells RMSE ≥60 ft | 0 | 0 | 0 | 0 |
| Worst-10% SSE share | 0.490065 | 0.489325 | 0.489334 | **0.487344** |

沒有任何 blend 造成 ≥60-ft failure，p95 與 max 也沒有惡化；但三個 blends 的
p50 都較 incumbent 差。Tail 沒有否決它們，卻也不能補救 bootstrap 與
shadow-private instability。

| Decomposition metric | Incumbent | Nested | Fixed 0.25 | Simple 0.5 |
|---|---:|---:|---:|---:|
| Common-mode RMSE | 11.041234 | 11.033062 | **11.030624** | 11.041936 |
| Centered RMSE | 8.882797 | 8.841617 | 8.841256 | **8.816314** |
| Slope-component RMSE | 5.930760 | 5.885847 | 5.885231 | **5.858924** |
| Zero-mean shape RMSE | 6.612879 | 6.597802 | 6.597868 | **6.587898** |
| Terminal RMSE | 19.227242 | 19.168431 | 19.166049 | **19.148913** |
| Well-slope coefficient RMSE | 10.355509 | 10.265198 | 10.264734 | **10.207679** |

Nested 與 fixed 0.25 都在 common、centered、slope、shape、terminal 上取得小幅
改善；simple 0.5 的 slope／shape signal 更強，但 common mode 略差。這支持 TCN
具有 architecture-only shape representation，仍不足以支持 production blend，
也不構成 pretrained transfer evidence。

### 8.5 Wins、residual correlation 與 100k shadow-private

| Candidate | Wins / ties / losses | Win fraction | Mean well-RMSE delta | Median well-RMSE delta |
|---|---:|---:|---:|---:|
| Nested | 385 / 0 / 388 | 49.806% | -0.023106 | +0.000689 |
| Fixed 0.25 | 386 / 0 / 387 | 49.935% | -0.025058 | +0.000174 |
| Simple 0.5 | 375 / 0 / 398 | 48.512% | -0.023015 | +0.009073 |

所有 blends 都少於一半 wells 獲勝，median well 反而略差。Row-residual
correlations 也極高：

| Pair | Correlation |
|---|---:|
| Incumbent vs nested | 0.998969 |
| Incumbent vs fixed 0.25 | 0.999030 |
| Incumbent vs simple 0.5 | 0.996110 |
| Nested vs fixed 0.25 | 0.999994 |
| Nested vs simple 0.5 | 0.999073 |

這不是足以支撐 robust ensemble 的 residual diversity。100,000 組固定 200-well
subsets：

| Candidate | Mean delta | P(worse) | q01 | q05 | q50 | q95 | q99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Nested** | -0.032051 | **16.557%** | -0.108867 | -0.086119 | -0.031956 | +0.021632 | +0.043354 |
| Fixed 0.25 | -0.034187 | 14.171% | -0.108569 | -0.086577 | -0.034084 | +0.017817 | +0.039154 |
| Simple 0.5 | -0.040806 | **26.130%** | -0.188817 | -0.145373 | -0.040624 | +0.063200 | +0.105903 |

Nested 的 `P(worse)=16.557%`，simple 0.5 更達 26.130%；兩者都遠高於
formal `<1%` 目標。Simple 的較低 full-population point estimate 是由更寬的
subset outcome 分布換來，不能稱為 private-safe improvement。

### 8.6 Independent disk re-audit

Runner `artifact_audit.json` 與 `independent_nested_reaudit.json` 均
`passed=true`。獨立稽核從 disk：

- 重新驗證 externally locked incumbent 與 full TCN source hashes；
- 重算 20 inner splits 的 weights，並確認 outer validation 未參與 weight fit；
- 確認 family-first seed average、inner residual algebra 與 reciprocal aliases；
- 由 weights 重建每個 outer prediction，maximum absolute delta 為 0；
- 從 final OOF 重算全部 metrics，與 runner summary 最大差為 0；
- 驗 exact 3,783,989-row coverage、0 duplicate keys、0 non-finite values，
  canonical key SHA-256 為
  `9d1d1527eec52b31bd4b23cc81ab582ea6fbb2db7636f9c538aa1721c1e33c47`；
- 驗 `data/` before／after manifest digest 均為
  `dec44ac09997b60e900c70064349f0fc9a1d558a9cda5c05aaa3530902fe4514`；
- 確認 `submission_created=false`。

主要 hashes：

- nested final OOF：
  `589a23e19efbcfc514a10441aa8f3ef6d8eeb9f82430f61da7ae2479e15d4b90`；
- runner summary：
  `2ea892d2d7841099e823d76614919ad656b944f2a654e294fcbaad1d85904a5c`；
- runner artifact audit：
  `a2913ac1ecb9ce8a9e821d3a4b5a3f80e9df5c860b592e542ab62c17c3ed25cd`；
- nested weights：
  `169e7cce1aaef00efb8beef5561d256a342c8836d0644b6b7e2bf107f169812e`；
- independent nested re-audit：
  `c9a9b1053683d4a9e29bfb288d8f0f20d5dfc16c6242ab5a4ed6b5bc1d6546dc`。

Artifact 的 legacy `evidence_eligible=true` 只證明當時 runner 對
model-fit／weight-fit／serialization gates 的判定；它沒有獨立的
family-selection-safe gate。Higher-level audit 因此把這組數字降級為
conditional negative evidence，不作 promotion evidence。依 CI、
shadow-private、win coverage 與這項 selection limitation，正式決策仍是
**reject all external ensembles and retain incumbent**。

### 8.7 Target-free strata stress

正式 stress 使用 30 個 prediction 前可得的 features，涵蓋全部 773 wells；
features 包括 well length／row density、trajectory、horizontal/typewell GR
distribution 與 spectrum，以及 heel-registration posterior diagnostics。
`target_columns_used_for_strata=[]` 且
`prediction_columns_used_for_strata=[]`，所以 strata 本身沒有使用 labels 或
candidate predictions。輸出共有 139 個 feature×quantile strata、5 candidates、
**695 metric rows**。

三個 blends 的 best／worst observed strata：

| Candidate | Best target-free stratum | Best delta | Worst target-free stratum | Worst delta |
|---|---|---:|---|---:|
| Nested | `typewell_gr_std / q03_of_05` | **-0.12288148** | `registration_posterior_coverage_mean / q01_of_02` | **+0.11079374** |
| Fixed 0.25 | `typewell_gr_std / q03_of_05` | **-0.12329080** | `registration_posterior_coverage_mean / q01_of_02` | **+0.09868227** |
| Simple 0.5 | `typewell_gr_std / q03_of_05` | **-0.22170146** | `registration_posterior_coverage_mean / q01_of_02` | **+0.24473380** |

Best stratum 有 155 wells／752,062 rows；worst registration-coverage stratum
有 11 wells／51,870 rows。這是 composition heterogeneity evidence，不是
truth-based routing license。尤其 simple 0.5 同時有最大的局部改善與最大的局部
惡化，與其 26.130% shadow-private failure probability 一致。Stress 因此加強
「全部 rejected」結論，而不是提供 post-hoc gate。

Stress summary 為 `completed=true`、`evidence_eligible=true`、
`submission_created=false`；它重新鎖定並通過下列 exact provenance：

- external source manifest：
  `669cca7ebc53f78986a94e5abd79d4b5a68753f91d66cd292cce7164ee24c46d`；
- external source OOF：
  `09e780ff7044da750aaa122d97bea9d6e5e987793c974c77ae9ff95be53eaa9d`；
- external source run summary：
  `e0b89f76be2226fc543fd5f5ad4fe3526e3f116d7eefa741fec67c9b6f705e72`；
- source target-free feature/stress summary：
  `815220637f91ba8cfe58cabd0640b62ee9512c38f12656bb57b0a0edf3a94070`；
- nested final OOF：
  `589a23e19efbcfc514a10441aa8f3ef6d8eeb9f82430f61da7ae2479e15d4b90`；
- passed independent nested re-audit：
  `c9a9b1053683d4a9e29bfb288d8f0f20d5dfc16c6242ab5a4ed6b5bc1d6546dc`；
- target-free well features：
  `fc79e10e9d5fc2dc0e0ed93918473bab76e1dce6c6b596f0770ac1a107c0073d`；
- target-free strata：
  `ecd29f57f64a14b965773f52f391ef169b4d55b805952aec9e528e781a525f89`；
- registration diagnostics：
  `f8a36b09aab0ca2bf0f646ae366bca37faed13faf1b0c4c3800bddba92a7fb0c`；
- 695-row metrics：
  `0b3e72f0afa5cef52e095df4949398b3bdc8b9d88ec4405f33501a27c8f9c04b`；
- stress summary file：
  `7c60be293cb9a63abd4022f01d15a06d06c6267516490b0195db3b1a0cf193b8`。

這仍不是未知 domain shift 的保證。Observed worst stratum 已足以否決
conditional promotion；沒有使用 stress labels 重新 fit weights、選 gate 或
改寫 nested primary。

## 9. Canonical publication 與 quick disk re-audit（P1 限制）

Canonical bundle：

- path：
  `.runs/external-model-portfolio-quick-20260728/external_model_artifacts`；
- 489,284 rows、100 wells、5 folds；
- 50 source checkpoints；
- 20 candidates，其中 8 eligible、12 ineligible；
- OOF bytes：373,767,567；
- manifest SHA-256：
  `f3f06ee7dcf548f6abcb813f4d59f9f096298f4c7233d26ba8de576e0b2d412c`；
- OOF SHA-256：
  `7e671cde021ab358a9f8ac1d66d8fdc3f809221e79154f9c7fddc55310c28208`；
- normalized row coordinate 由 inference-safe raw `MD` 重建，逐井嚴格遞增、
  位於 `[0,1]` 且 terminal 等於 1；
- data manifest before/after 相同；
- submission created：false。

Independent re-audit 的執行順序是：

1. `independent_reaudit_external_artifacts` 從 disk 驗 manifest、OOF、
   checkpoints、fold provenance、fold config、current data manifest 與 no
   submission；
2. `read_external_artifact_bundle` 再次 disk reread；
3. 對 incumbent 與所有 20 candidates 重算 pooled、per-fold、tail、
   common/slope/shape、10k bootstrap、row/well-mean residual correlation 與 wins；
4. 對 runner summary 中同名 full/shrink metrics 比對。

Summary comparison 的公開容差是：

```text
abs(recomputed - runner)
  <= 5e-9 + 5e-11 * max(abs(recomputed), abs(runner))
```

所有 integer counts 必須 exact。結果：

- first artifact audit：pass；
- second disk read：pass；
- data manifest 在 re-audit 全程相同；
- runner summary comparison：pass；
- 1,815 個 numeric comparisons 通過；
- bootstrap 最大差：0；
- row residual-correlation 最大絕對差：`4.66e-15`；
- report SHA-256：
  `542653b8aa6b76ea93b32ad712512d40276bff4b638e9a8b0c4a0cf365d2973d`。

後續 P1 code audit 發現，這個 archived quick publication 版本的
expected truth／incumbent reference 是從同一份 wide/canonical OOF 建立，因此
上述 disk reread 能驗證 serialization、checkpoint provenance、component
algebra 與內部 matched metric reproduction，卻**不能獨立排除 source
truth／incumbent 已漂移**。Quick 數字仍是同 cohort 的 screening 描述，但不再
被列作 hardened comparator evidence。Section 7.6 的 full publication 已改用
外部 audited Heel OOF lock 修補這個 P1 問題。

100 wells 少於 200，因此 formal 100,000-sample／200-well shadow-private
simulation 在 quick audit 中正確標記為未執行，而不是用較小 subset 冒充。
Full TCN 的正式 100k 結果已列於 Section 7.5。

Target-free strata 已在 Section 8.7 對 length、row density、GR
distribution/spectrum、trajectory、typewell 與 registration uncertainty
完成獨立 stress。它和 100k shadow-private 都不能替代未知 domain shift 保證。

## 10. Visible-test offline inference smoke

正式 smoke artifact 是 full run 的
`.runs/external-model-tcn-full-20260728/visible_test_smoke_all_folds_v2.json`；
它支持的是執行與 key contract，不是 accuracy：

- 動態列舉 3 個 test wells：`000d7d20`、`00bbac68`、`00e12e8b`；
- 依 `sample_submission.csv` 明確 ID join 與原順序輸出 14,151 rows；
- duplicate IDs：0；
- sample contract exact：true；
- finite incumbent/checkpoint/smoke predictions：true；
- labels read：false；
- data manifest before/after 與 run config 相同；
- 使用全部 25 個 full TCN checkpoints（5 folds × 5 seeds）；
- 每個 checkpoint 都使用 matching fold-specific 5-seed incumbent
  conditioning，並逐 checkpoint 驗 finite output；
- ordered sample-ID SHA-256：
  `1fdbb5d981b5fe1e407eafff3edd24f2eff67112a30da7aa2c811a51f5ef73a0`；
- ordered smoke-average prediction SHA-256：
  `4a51506b7c0ebb2073fc783a0d4a26fac54c1d6ea7932df7a2cfe61260cc28f5`；
- smoke JSON SHA-256：
  `f69b565a2eef90c22327b9a3683765b739f5ba4097e053621f66e43d0167e2ed`；
- submission CSV created：false。

此 smoke 證明 full checkpoint portfolio 可在 CPU 離線、無 labels、動態 test
enumeration 下產生 finite、order-aligned predictions；它不是 accuracy
validation，也沒有建立 submission。所有 external blends 已被拒絕，因此沒有把
smoke average 包裝成 production submission；retained incumbent 的 full
independent audit 已另行確認 `test_inference_capable=true`。

## 11. Architecture-only 與 pretrained transfer 結論

本輪只完成 comparison A：

- **A — random-init architecture-only：**Paired Patch、S4D、FNO、TCN；
- **B — random frozen／negative controls：**Paired Patch 與 TCN 都完成
  wrong-typewell／pair-shuffle／GR-order／phase／frozen-random controls；
  no-cross interaction ablation 由 Paired Patch 完成；
- **C — frozen pretrained embeddings／adapter：**未執行。

Artifacts 明確記錄：

```text
architecture_only = true
external_pretrained_weights_used = []
```

所以：

- TCN 的 quick 與 full slope／shape signal 都是 architecture-only residual
  modeling signal；full 與 nested pooled evidence 都未證明它可取代 incumbent；
- 它不是 external-pretraining gain；
- Paired Patch controls 未支持 geological correspondence claim；TCN intact
  優於多數 TCN controls，但 GR-order shuffle 仍有弱 point signal，因此只能
  支持有限的 architecture／pairing signal，不能宣稱純 GR correspondence；
- MOMENT、TTM、TimesFM、Chronos 沒有任何本任務 accuracy evidence；
- 不能把「尚未測」寫成「pretraining 無效」；
- 也不能把 architecture-only model 的改善歸功於其論文或官方 pretrained
  weights。

只有在同 folds／same capacity 下完成 random-init、random-frozen、frozen
pretrained 的 matched comparison，且 pairing-destroying controls 失去收益，
才可宣稱 transfer signal。

## 12. Continue、defer 與 stopped routes

| Route | Evidence-based decision |
|---|---|
| TCN raw correction | **Stop as standalone**；full delta +0.028081、五個單 seed 全差、四個 folds 變差。只保留為 residual-diversity／decomposition diagnostic |
| Nested incumbent＋TCN | **Rejected after conditional weight-crossfit**；delta -0.032155、CI 上界 +0.005706、`P(worse)=16.557%`，且 family selection 未 outer-nest |
| TCN fixed ×0.25 | **Rejected**；delta -0.034283，但 CI 上界 +0.002376、`P(worse)=14.171%` |
| Simple 0.5 blend | **Rejected descriptive control**；最低 point delta -0.041030，但 CI 上界 +0.031949、`P(worse)=26.130%` |
| S4D | Defer；有 centered/slope signal，但 pooled CI 跨 0，尚不足以啟動無限 tuning |
| FNO1D current configuration | Stop before full；full correction 變差，shrink signal 很小且 CI 跨 0 |
| Paired Patch current configuration | Stop before full；未縮放明顯變差，shrunk gain 未通過 negative-control interpretation |
| TS2Vec | Not run；只有在需要新的 lightweight representation control 時再啟動 |
| Small Mamba | Current Windows CPU route deferred/rejected；S4D 是乾淨長序列 fallback |
| MOMENT pretrained | Held；`Rock` geology-adjacent provenance 尚需明確接受，且 architecture-only 尚無 matched signal |
| TTM / Chronos | 未執行的低成本 pretrained probes；不得宣稱 gain |
| TimesFM | Low-priority forecasting-first scouting；未下載、未執行 |
| ChronoLog data | **Prohibited** due correlation／possible-overlap risk |
| More similar seeds without family signal | 不作為獨立 ensemble diversity |

停止是針對目前 configuration/evidence，不等於論文 architecture 永久無效；若未來
重開，必須有新的 inference-safe representation hypothesis，而不是在已曝光 quick
cohort 上無限調參。

## 13. Method 與 validation safeguards

1. `data/` 只讀；incumbent full、quick、canonical publication、visible smoke 與
   independent re-audit 均保存 before/after manifest。
2. Horizontal inference 只取 `MD, X, Y, Z, GR, TVT_input`；typewell 只取
   `TVT, GR`。
3. Target `TVT`、formation surfaces、`Geology`、PNG 與 validation labels 不作
   test-time feature。
4. Outer validation 以 well 分割；每 fold training/validation IDs、intersection
   與 digest 都寫入 artifact。
5. Scaler、normalizer、residual representation 與 model state 都是 fold-local。
6. Primary metric 是所有 OOF rows pooled row-RMSE；bootstrap 以 well 為 paired
   cluster。
7. Canonical OOF 以 `(well_id,row_index,fold)` 作 exact keys，檢查 finite、
   coverage、duplicates、truth/incumbent hashes 與 component algebra。
8. Full publisher 的 truth／incumbent 必須來自有 sibling independent audit 的
   Heel OOF external lock；不得再從 candidate OOF 自行生成 expected reference。
9. Legacy incumbent audit 的 file-order key digest 與 canonical key digest 只在
   exact file SHA／sibling audit 先通過時二擇一接受，其他狀態 fail closed。
10. Negative controls 和 frozen-random outputs 永久標成 ineligible；不能因分數
   好看進入 model selection。
11. Completed candidate 必須涵蓋所有宣告的 fold×seed checkpoints；missing、
   hash mismatch、incomplete 或 NaN 都 fail closed。
12. Test wells 動態列舉，結果用 `sample_submission.csv` 的 `id` join 與排序。
13. Quick evidence 只作 screening；不以 quick cohort 反覆選 blend weights。
14. Final blend 必須 nested/cross-fitted：outer validation 不得參與 family
    selection 或 weight fit。
15. Completed nested run 的 inner OOF weights 已從 disk 重算；outer predictions
    只使用對應 outer-training inner OOF 學得的 weights，source outer predictions
    保持 untouched。
16. Diagnostic oracles 只描述 headroom，不作 deployable evidence。
17. 沒有使用 truth-based per-well routing、post-hoc OOF oracle 或 Kaggle
    leaderboard feedback。

Full artifact audit、legacy-key focused regressions、nested production
independent disk re-audit 與完整 pytest suite 均已通過；完整結果與 delivery
scope checks 見 Section 16。

## 14. Research-memory 版本差異

依 `rogii-research-memory` 的 2026-07-29 external compiled query：

- memory version：`1.0.0`；
- source archive：`v9`；
- reference state 仍把 flat 15.9098528707 視為唯一完全穩定 baseline，並把
  Frozen P3 15.4525482195 記為最低 full strict point estimate；
- query 沒有返回 Heel-Registered 14.1708478925 或本 external portfolio。

這表示 memory 尚未 ingest 2026-07-28 最新 artifacts。Memory 的 historical
records 可能屬不同 `comparability_group`，本報告不把其 absolute RMSE 與新結果
直接相減。當前 incumbent 主張只使用
`.runs/heel-registered-anchor-full-20260728` 內同 population、同 keys、同
comparator 的 summary 與 independent re-audit。

另有一個 schema 命名差異：runner `artifact_audit.json` 的 legacy
`eligible_family_columns` 欄位包含 controls；canonical bundle 才是 selection
eligibility 的權威來源，並明確記錄 8 eligible、12 ineligible candidates。

## 15. 尚未解決的風險

- TCN raw 的五個單 seed 全差，且 family mean 在四個 folds 變差；quick gain
  明顯沒有穩健外推到 full population。
- TCN ×0.25 的 point estimate 較好，但 CI 上界為正，且 200-well
  `P(worse)=14.171%`；它不滿足 formal private-stability 標準。
- Conditional nested weight evaluation 已完成，但 CI 跨 0、
  `P(worse)=16.557%`；simple 0.5 的 `P(worse)=26.130%`，所以所有
  external blends 都被拒絕。
- Nested runner 已證明 model-fit 與 weight-fit outer-safe；但 TCN
  family/config 是由 global quick/full OOF 篩入，沒有 outer-fold-safe family
  selection artifact。Legacy `evidence_eligible=true` 因而不能解讀為完整 nested
  promotion eligibility。
- Fold TCN point weights 與 50,000-draw bootstrap distribution 都集中，沒有
  boundary hits；但 outcome stability 明顯未通過，不能把 weight stability
  誤寫成 prediction stability。
- Full nested portfolio 只有一個 external family（TCN）；沒有真正 multi-family
  leave-one-family-out marginal evidence。
- 100k shadow-private 與 target-free strata stress 已完成，但兩者都來自目前
  train-well empirical composition，不能保證未知 private domain。
- Shrink 的 p95／max 沒有惡化，但 p50 仍比 incumbent 高約 0.0982 ft；wins
  只有 386/773，收益沒有廣泛到足以消除穩定性疑慮。
- External foundation corpus 的 absence-of-overlap 證據有限；MOMENT 有明確
  geology-adjacent exposure，TimesFM／Chronos corpus 披露不完整。
- Full TCN P1 external lock 已通過；archived quick bundle 仍有 self-reference
  限制，不能升級為 hardened evidence。
- Nested artifact 保留訓練時 implementation byte hashes。交付前 formatter 改了
  `data.py`／`ensemble.py` 排版，之後 family-selection fail-closed hardening
  又刻意更新 nested runner；目前 archived/current source-byte mismatch 因而是
  `fixed_grid`、`blend`、`nested_runner` 三項。`--resume` 會依設計 fail closed。
  Artifacts 與 independent re-audit 仍是 immutable evidence，但若要用目前程式碼
  重跑，必須使用新 run ID，不能把舊 run 靜默續跑。
- Executed environment 已記錄 Python 3.12.10、NumPy 2.4.3、Pandas 3.0.1、
  SciPy 1.17.1、PyTorch 2.12.1+cpu；但 `pyproject.toml` 對 SciPy 仍是
  `>=1.12` 而非 exact lock。既有 weights／predictions 有 byte lock，跨環境 fresh
  SLSQP rerun 仍可能有微小數值差異。
- Quick controls 已覆蓋 Paired Patch 與 TCN；TCN 有 same-capacity
  frozen-random negative control，但仍沒有 frozen **pretrained** matched
  transfer experiment。
- Full-repo Ruff 仍有 22 個既有／範圍外 lint findings；delivery-scope checks
  clean 不等於整個 repository lint clean。

## 16. Complete tests 與 final delivery checks

| Check | Result | Scope／interpretation |
|---|---|---|
| Full pytest suite | **443 passed、37 warnings、230.11 s** | 完整 repository test suite；含 weight/noise/family-selection hardening，warnings 保留揭露 |
| Final focused regressions | **45 passed、41.53 s** | nested weights、bootstrap stability、noise-floor parser 與 target-free nested stress |
| Delivery-scope Ruff check | **clean** | 本次 external-model／nested delivery scope |
| Delivery-scope Ruff format check | **clean** | 本次 delivery files |
| Delivery-scope compileall | **clean** | Python source compile validation |
| Diff check | **clean** | 本次交付差異沒有 whitespace/error finding |
| Full-repo Ruff | **22 findings remain** | 既有／範圍外；不得寫成 repo-wide clean |

初次 full-repo Ruff 顯示 23 個 findings；其中
`external_models/incumbent` 的一個本次 unused import 已修正，剩餘 22 個均記為
既有／範圍外 lint debt。本報告不隱藏它們，也沒有為追求全綠而修改無關使用者
檔案。

Final artifact checks 同時確認：

- full TCN 與 nested runner／independent audits 全部 passed；
- target-free stress 695 metric rows 完成且 exact source/audit hashes 通過；
- `data/` before／after/current digest 保持
  `dec44ac09997b60e900c70064349f0fc9a1d558a9cda5c05aaa3530902fe4514`；
- visible-test dynamic enumeration／ID join contract 已由 Section 10 smoke 通過；
- 所有 external blends rejected，沒有 rejected blend 被包裝成 production；
- production 保持已通過 full independent audit 且
  `test_inference_capable=true` 的 Heel-Registered incumbent；
- submission CSV created：false。

Bootstrap weight distribution 已完成；仍沒有第二個 full external family 可做
真正 cross-family leave-one-family-out，且 family selection 沒有 outer-nest。
這些是 evidence limitation，不是可用 post-hoc tuning 補救的 promotion 缺口；
它們不改變本次 completed rejection decision。

## 17. Reproduction commands

Immutable invocation parameters 以各 run 的 `run_config.json` 為權威。以下是實際
runner 形狀；重跑時應換新 `--run-id`，不可覆寫或假裝 resume 已稽核 artifacts。

Quick four-family portfolio＋Paired Patch controls：

```powershell
python scripts/run_external_model_portfolio.py `
  --profile quick `
  --data-root data/rogii-wellbore-geology-prediction `
  --run-id external-model-portfolio-quick-20260728 `
  --variant paired_patch_cross --variant s4d --variant fno1d --variant tcn `
  --variant paired_patch_no_cross --variant paired_patch_wrong_typewell `
  --variant paired_patch_pair_shuffle --variant paired_patch_gr_order_shuffle `
  --variant paired_patch_phase_destroy --variant paired_patch_frozen_random `
  --seed 20260728 --epochs 30 --hidden-dim 16 --layers 2 `
  --device cpu --workers 4 --shrinkage 0.25
```

Matched TCN controls：

```powershell
python scripts/run_external_model_portfolio.py `
  --profile quick `
  --data-root data/rogii-wellbore-geology-prediction `
  --run-id external-model-tcn-controls-quick-20260728 `
  --incumbent-cache-source .runs/external-model-portfolio-quick-20260728/incumbent_cache `
  --variant tcn_wrong_typewell --variant tcn_pair_shuffle `
  --variant tcn_gr_order_shuffle --variant tcn_phase_destroy `
  --variant tcn_frozen_random `
  --seed 20260728 --epochs 30 --hidden-dim 16 --layers 2 `
  --device cpu --workers 4 --shrinkage 0.25
```

Full TCN 5-seed strict OOF：

```powershell
python scripts/run_external_model_portfolio.py `
  --profile full `
  --data-root data/rogii-wellbore-geology-prediction `
  --run-id external-model-tcn-full-20260728 `
  --incumbent-cache-source .runs/external-model-portfolio-quick-20260728/incumbent_cache `
  --variant tcn `
  --seed 20260711 --seed 20260712 --seed 20260713 `
  --seed 20260714 --seed 20260715 `
  --epochs 60 --hidden-dim 16 --layers 2 `
  --device cpu --workers 4 --shrinkage 0.25
```

Conditional nested incumbent＋TCN weight cross-fit：

```powershell
python scripts/run_nested_external_ensemble.py `
  --data-root data/rogii-wellbore-geology-prediction `
  --run-id nested-external-tcn-full-20260729 `
  --source-root .runs/external-model-tcn-full-20260728 `
  --incumbent-oof .runs/heel-registered-anchor-full-20260728/oof_predictions.csv.gz `
  --device cpu --workers 4
```

Full visible-test smoke、noise-floor audit 與 weight stability：

```powershell
python scripts/smoke_external_model_portfolio.py `
  --run-root .runs/external-model-tcn-full-20260728 `
  --output-json visible_test_smoke_all_folds_v2.json

python scripts/audit_noise_floor_notebook.py

python scripts/analyze_nested_blend_weight_stability.py `
  --run-root .runs/nested-external-tcn-full-20260729 `
  --output-name weight_stability_v2
```

已完成 quick canonical publication：

```powershell
python scripts/publish_external_portfolio_artifacts.py `
  --run-root .runs/external-model-portfolio-quick-20260728
```

已完成 independent quick re-audit：

```powershell
python scripts/audit_external_model_portfolio.py `
  --run-root .runs/external-model-portfolio-quick-20260728
```

已完成 hardened full canonical publication 與 independent re-audit：

```powershell
python scripts/publish_external_portfolio_artifacts.py `
  --run-root .runs/external-model-tcn-full-20260728 `
  --incumbent-oof .runs/heel-registered-anchor-full-20260728/oof_predictions.csv.gz

python scripts/audit_external_model_portfolio.py `
  --run-root .runs/external-model-tcn-full-20260728 `
  --incumbent-oof .runs/heel-registered-anchor-full-20260728/oof_predictions.csv.gz
```

已完成 nested production independent disk re-audit：

```powershell
python scripts/audit_nested_external_ensemble.py `
  --run-root .runs/nested-external-tcn-full-20260729
```

已完成 nested target-free stress：

```powershell
python scripts/analyze_nested_external_stress.py `
  --nested-run-root .runs/nested-external-tcn-full-20260729 `
  --external-run-root .runs/external-model-tcn-full-20260728
```

Legacy/canonical key convention focused regression：

```powershell
python -m pytest -q tests/test_external_artifacts.py `
  -k "audited_incumbent_accepts_both_locked_key_conventions or audited_incumbent_wrong_digest_or_changed_file_fails_closed"
```

完整 test suite：

```powershell
python -m pytest -q
```

報告中的 pretrained sources 只作 registry/provenance；以上 commands 均不下載
weights、不建立 submission，也不讀 protected labels。Archived quick bundle
predates P1 hardening；不要把 quick hash 當作 full external-lock provenance。

## 18. 下一個最高資訊價值實驗

不要再增加同一個 TCN configuration 的 seeds。現有 decomposition 顯示 raw TCN
在 centered-shape、slope 與 smooth-shape 有小幅改善，但 common-mode 變差；nested
stress 又顯示低 registration posterior coverage strata 可能反向。因此下一個最有
資訊價值的 inference-safe probe 是預先鎖定的 **component-constrained paired
sequence residual model**：

1. 把 `predicted_common` 固定為 0，只允許輸出 slope 與明確 zero-mean 的 smooth
   shape，直接測試目前唯一可重複觀察到的 representation signal；
2. 比較 matched-capacity TCN 與 S4D，但把 S4D／Mamba 視為同一長序列家族，不以
   多個近似 variants 重複計權；
3. 用 prediction 前可得的 registration posterior coverage／uncertainty 做單調
   attenuation；所有 threshold 或係數只可由 outer-training inner OOF 學習；
4. 同時執行 intact、wrong-typewell、pair shuffle、GR-order/phase-destroy 與
   random-frozen controls；intact 若沒有明顯優於 controls 就停止；
5. 在讀取任何新 run 的 outer-fold labels 前，鎖定 candidate set、loss、seeds、
   stopping rule 與 nested selection code，使用新 run ID；先跑低成本 quick，
   只有 slope/shape、tail 與 controls 同時通過才啟動 full。

這個實驗能判斷 sequence family 是否真的補到 incumbent 的 shape 缺口，而不是再
擬合已近飽和的 common correction。由於本研究流程已看過現有 773 wells 的全體
OOF 結果，即使嚴格 outer/inner cross-fit，任何後續在同 773 wells 上的結果仍只能
視為探索性 evidence；正式 promotion 最終需要真正未觸碰的新 wells 或使用者另行
授權的外部盲測。若 architecture-only probe 通過，再做 MOMENT frozen-pretrained
versus random-init/random-frozen matched comparison；在此之前下載或微調大型
foundation weights 的資訊價值較低。
