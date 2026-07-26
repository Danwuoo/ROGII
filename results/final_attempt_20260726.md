# ROGII Wellbore Geology Prediction：最後一次大規模研究嘗試

日期：2026-07-26（Asia/Taipei）  
工作目錄：`C:\Users\wurre\Desktop\ROGII`  
現行 fold：`priority1_5fold_v1`  
fold SHA256：`7F1772817A95277696EB32085DA7618E2671E3ECE49F0DB00FFE1BC03C07C72E`

## 0. 結論先行

本輪沒有達成 strict full well-level OOF pooled row-RMSE `<4`。

本輪數值最佳的是 **post-hoc、固定 50/50 的 research blend**；權重沒有用
target fit，但它不是獨立未曝光 confirmatory trial：

> **50% P3 + 50% whole-path functional PCA：15.1720614751**

它在相同 773 wells、3,783,989 prediction rows 上：

- 較 Flat `15.9098528707` 改善 `-0.7377913957`；
- 較 P3 `15.4525482195` 改善 `-0.2804867444`；
- well-cluster bootstrap 10,000 次的 Flat-relative delta 95% interval 為
  `[-1.1041, -0.4005]`；
- 451 wells 改善、322 wells 退步；
- p50 / p90 / p95 / max well RMSE 為
  `10.1007 / 22.0525 / 26.8746 / 70.7267`；
- RMSE ≥40 wells 從 Flat 的 15 降到 12，RMSE ≥60 wells 從 4 降到 2；
- 但仍有兩口約 70 ft 的 catastrophic wells，沒有解決 catastrophic tail。

距離 `<4` 仍有 `11.1721 ft`。若以 SSE 計，從 15.1721 降到 4 還需再消除
`93.05%` 的現有誤差。本輪實驗、公開 strict evidence 與一手文獻均沒有顯示
可在現有合法 modality 下跨越這個數量級差距的 scaling law。

因此本輪的研究判斷是：

> **在只有合法的 horizontal `MD,X,Y,Z,GR,TVT_input` 與 typewell
> `TVT,GR` 下，目前沒有可信的 `<4` 路徑。**

這不是證明數學上不可能；它是證據充分的工程／統計結論：低複雜度 target
oracle 的 headroom 無法被 prefix、GR、trajectory 或 cross-well support
安全辨識。要重新開啟研究，最小而有實質價值的變化應是 post-PS 稀疏 TVT
anchors、人工 formation tops/correlation anchors、azimuthal/deep-resistivity
LWD、或 seismic/formation-surface constraints，而不是再換一個較大的相同輸入模型。

本輪沒有建立或上傳 Kaggle submission。

## 1. 規則、資料與評分契約

已完整閱讀根目錄 `AGENTS.md`，並以它覆蓋歷史 audit/research-memory 的舊 gate。
歷史記憶只作結果去重和失敗機制證據，不作探索禁令。

`rogii check` 通過：

- 773 train wells；
- 5,092,255 train rows；
- 3,783,989 prediction rows；
- 3 個本地 test authoring copies；
- 14,151 sample-submission rows；
- fold well counts：155 / 155 / 155 / 154 / 154；
- 每口 well 僅出現在一個 validation fold；
- full manifests 的 training/validation wells 完全互斥；
- `data/` 沒有被寫入。

所有正式 OOF runner 模型都遵守：

- horizontal prepare 只讀 `MD,X,Y,Z,GR,TVT_input`；
- lattice/contextual alignment 的 typewell 只讀 `TVT,GR`；
- target `TVT` 僅在 outer-training fit 或 validation 後評分時讀取；
- scaler、PCA、Ridge、HGB、transition/emission statistics 全在 outer-training
  wells 內 fit；
- 以所有 OOF prediction rows 合併後的 pooled row-RMSE 為唯一主指標；
- paired analyzer 逐列核對 `fold,well_id,row_index,y_true` 完全一致。

## 2. 本地定位：真正的誤差結構

### 2.1 Baseline 與歷史 oracle

| 證據 | 結果 | 能否部署 | 正確解讀 |
|---|---:|---|---|
| Flat full OOF | 15.9098528707 | 是 | 最後已知 TVT 常數延伸 |
| P3 full OOF | 15.4525482195 | 是 | rowwise HGB；alignment features 有部分訊號 |
| P3 stress（歷史） | 16.1361032223 | 是 | main point gain 不穩定 |
| `H=TVT+Z` 2-piece continuous oracle | 3.6890603816 | 否 | 全段 target-bearing breakpoint/slope fit |
| `H` 2-piece discontinuous oracle | 3.1010299342 | 否 | 同上，且允許跳變 |
| `H` 3-piece discontinuous oracle | 2.0931332302 | 否 | 證明輸出低複雜度，不證明可推論 |
| `H` 4-piece discontinuous oracle | 1.5853516059 | 否 | 同上 |
| future TVT 在 prefix TVT 1 ft 內 | 64.42% rows | 否 | range support，不等於候選可選 |
| future TVT 在 prefix TVT 4 ft 內 | 72.33% rows | 否 | 仍有大範圍 unsupported rows |

SLP-E1 的 handcrafted loop-edge V1 提出約 14.13M candidates，但只接受 106
edges、覆蓋 83 unique rows（全部 future rows 的 0.00219%），其中 55 edges
為 catastrophic false edges。它只否定該 edge generator；不否定所有 graph、
learned emission 或 posterior decoder。

### 2.2 Flat 的 tail concentration

- Flat p50 / p90 / p95 / max well RMSE：
  `10.6651 / 22.9725 / 29.0108 / 70.6394`。
- 最差 10% wells 佔全體 SSE 的 `52.4763%`。
- 15 wells 的 RMSE ≥40，4 wells ≥60。
- 由 prediction suffix 的第一到第十距離 decile，Flat row-RMSE 由
  `6.5473` 增至 `19.9224`，誤差是長距離累積，不只是 anchor 附近偏差。
- 最差 well `86454a6f` 的 future GR observed fraction 僅約 18.8%，但全體
  per-well RMSE 與 GR-observed fraction 的 Spearman correlation 約 0.015；
  missingness 是個別 failure mechanism，不是足以安全 gate 所有 catastrophic
  wells 的單一統計。

### 2.3 Oracle、support 與 deployable evidence 的分離

本地證據反覆呈現同一結構：

1. 若直接看 target，路徑可以用很少的 piecewise parameters 重建到 `<4`。
2. 若用合法 GR 建候選，候選附近常有訊號：
   learned-emission fold-0 的 coarse top-10 oracle depth RMSE 為 `6.0975`。
3. 但 top-1、posterior calibration、branch selection 和 unseen-well support
   不能把 oracle headroom 轉成最終 row-RMSE。
4. P3 與 Path-PCA 的 row-level error correlation 為 `0.9392`
   （Flat–P3 `0.9377`、Flat–Path `0.9607`），代表三者大多受相同資訊瓶頸限制。

## 3. 網路研究範圍與 evidence matrix

搜尋涵蓋 geosteering、cross-well stratigraphic correlation、Bayesian inverse
problems、HMM/particle methods、biosequence/speech alignment、differentiable
dynamic programming、retrieval、simulation-based inference、公開 ROGII
資源、官方 GR 技術資料與開源授權。

### 3.1 直接 geoscience / geosteering 證據

| 來源 | 實際輸入／驗證 | Code / license | 對本題的結論 |
|---|---|---|---|
| [Wu et al. 2019, stochastic clustering and pattern matching](https://doi.org/10.1190/geo2018-0781.1)；[Rice copy](https://repository.rice.edu/items/dd6fe8d2-fc4b-498f-91e6-65587bed1071) | Pilot GR、horizontal non-azimuthal GR、survey；SAMC posterior；synthetic、少量 real case、Permian correlations；不是 strict cross-well TVT path RMSE | 未找到可直接重現官方 code | 支持多模態 posterior，但沒有 `<4` 或 unseen-well 證據 |
| [Miao et al. 2020 nonlinear state-space geosteering](https://www.sciencedirect.com/science/article/pii/S0920410520301194) | PF/PGAS/SAEM；synthetic、real case、約 400 wells；主要 metric 是 GR correlation | 論文機制 | state-space 可轉移，但 GR correlation 不是最終 path error |
| [Winkler, exact Bayesian-network geosteering](https://hughw.net/geosteering-bn/geosteering-by-bayesian-network.pdf) | Synthetic GR/path inference | 公開 PDF | 明確暴露 repeated/flat GR 與 fault ambiguity；支持 posterior 診斷 |
| [Alyaev & Elsheikh 2022 multi-modal inversion](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021EA002186)；[arXiv](https://arxiv.org/abs/2201.01871) | 約 28M same-generator synthetic paths；32 ft reference + horizontal GR；報告七個 modes 中 best-path 約 1.13 cells MAE | [MTP loss artifact](https://doi.org/10.5281/zenodo.6531534)；[GWC reference](https://github.com/alin256/gwc2020-semi-unconventional) | 證明 simulator 內 top-K path support 可很強；best-mode oracle 不是 deployable mode selection，也不是 cross-well OOF |
| [Sequential multi-mode inversion poster](https://nfes.org/assets/workshop2022/ambrus_sequential_multi_mode_inversion_poster.pdf) | 以先前 modes 播種、merge/prune，GWC synthetic 單例，約 443 ms | 方法海報 | 適合 sequential multi-hypothesis tracker；無 strict validation |
| [HMM-Match / Lin et al. 2014](https://lorraine-lisiecki.com/Lin_et_al_2014.pdf) | Geological time-series HMM forward-backward；比較 posterior median/mode | 公開論文 | 支持 full posterior 和 Bayes estimator，不支持 ROGII 數值外推 |
| [StratoBayes 2025](https://gchron.copernicus.org/articles/7/545/2025/)；[project](https://stratobayes.github.io/) | Bayesian stratigraphic alignment、age-depth uncertainty | trial/non-commercial 條件需另審 | 方法相關但成本與授權不適合直接納入 |
| [ChronoLog](https://zsylvester.github.io/papers/chronolog/)；[data](https://github.com/zsylvester/chronolog_data) | 將相對/絕對時間整合進 stratigraphic correlation | Data CC BY 4.0 | 支持 anchors 的高價值；本題沒有對應 age/top anchors |
| [Align 2024](https://www.geosociety.org/gsa-today/february-2024/align-a-userfriendly-app-for-numerical-stratigraphic-correlation)；[GitHub](https://github.com/CedricHagen/Align) | Numerical stratigraphic correlation | GPL-3.0 | 可作算法參考；plain alignment 不解 branch identifiability |
| [Samant et al. 2025 constrained DTW](https://doi.org/10.1029/2024PA005082) | Geological series 的 domain constraints | 論文 | 支持 constraint-aware alignment；仍依賴對應訊號 |

### 3.2 相鄰領域可轉移機制

| 來源 | 被驗證的機制 | Code / license | 可轉移與限制 |
|---|---|---|---|
| [DEDAL](https://www.nature.com/articles/s41592-022-01700-2)；[official code](https://github.com/google-research/google-research/tree/master/dedal) | 學 contextual substitution matrix、gap penalties，再做 differentiable alignment；remote-homolog benchmarks | Apache-2.0 | 強力支持 learned emission + structured decoder；protein representation 本身不可直接轉用 |
| [Soft-DTW](https://proceedings.mlr.press/v70/cuturi17a.html)；[Differentiable DP](https://proceedings.mlr.press/v80/mensch18a.html)；[GDTW](https://stanford.edu/~boyd/papers/gdtw.html) | soft path marginalization、generic DAG DP、warp/warp-rate penalties | GDTW code Apache-2.0 | 可學 cost/transition；soft averaging仍可能落在錯誤 modes 中間 |
| [ProbCons](https://pmc.ncbi.nlm.nih.gov/articles/PMC546535/) | pair-HMM posterior consistency、maximum expected accuracy | 公開 source | multi-reference consistency 可消歧；ROGII TVT 必須允許 signed transitions，不能照搬雙索引單調 DTW |
| [Segmental MBR for speech](https://research.ibm.com/publications/segmental-minimum-bayes-risk-decoding-for-automatic-speech-recognition) | 在 lattice 上依 task loss 解碼，不只取 MAP | 論文 | row-squared-loss 下 posterior mean 是直接對應的 Bayes estimator |
| [Differentiable particle filters](https://arxiv.org/abs/1805.11122)；[code](https://github.com/tu-rbo/differentiable-particle-filters) | robot localization 中保留多峰 state posterior | MIT | 適合 regime/fault switching；若 exact banded lattice 已可算，PF 只增加 approximation risk |
| [Stable multi-mode tracking](https://proceedings.neurips.cc/paper_files/paper/2023/hash/85b2ff7574ef265f3a4800db9112ce14-Abstract-Conference.html) | 長序列多峰 tracking、降低 mode collapse | 論文 | 支持 particle rejuvenation；不新增 ROGII observation information |
| [sbi](https://github.com/sbi-dev/sbi)；[misspecification diagnostics](https://sbi.readthedocs.io/en/latest/advanced_tutorials/21_diagnostics_misspecification_checks.html) | NPE/NLE/NRE 與 posterior predictive/misspecification checks | Apache-2.0 | 有可信 simulator 才有價值；sim2real mismatch 是主風險 |
| [TS-RAG](https://arxiv.org/abs/2503.07649)；[code](https://github.com/UConn-DSIS/TS-RAG) | 類似 context 檢索 future 並自適應混合 | MIT | 支持 prefix-only future-path retrieval；本題的 future branch non-uniqueness 更強 |
| [Gromov-DTW](https://proceedings.mlr.press/v130/cohen21a.html) | 以 series 內 relational geometry 對齊不同量測尺度 | 論文/code | 可作 robust emission ablation，不是完整 decoder |
| [Conformal structured prediction 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1868a3c73d0d2a44c42458575fa8514c-Abstract-Conference.html) | DAG structured output prediction sets | 論文 | 可診斷 coverage／safe set；不會自行創造可辨識資訊 |

### 3.3 官方 GR 與公開競賽資源

- [USGS spectral gamma logging](https://www.usgs.gov/mission-areas/water-resources/science/spectral-gamma-borehole-logging)、
  [SLB gamma-ray log](https://glossary.slb.com/terms/g/gamma_ray_log) 與
  [SLB corrected gamma ray](https://glossary.slb.com/en/terms/c/corrected_gamma_ray)
  說明 GR 是天然放射性／頁岩性相關量測，並受 borehole/environmental correction
  影響；它不是 formation TVT 或唯一 lithology identifier。
- 社群整理的 [ROGII task brief](https://github.com/vamseeachanta/kaggle-rogii-2026/blob/main/docs/task-brief.md)
  支持使用 pre-PS GR/TVT self-reference、signed azimuth 與鄰井 dip，但不是
  strict OOF 成績證據。
- [mycarta ROGII geosteering toolkit](https://github.com/mycarta/rogii-geosteering-toolkit)
  為 MIT 社群程式；其自述 Q3D/AEON ablation 未獨立重現，只作線索。
- 公開 [6.928 notebook](https://www.kaggle.com/code/beicicc/rogii-mha200-sep4-20260720/comments)
  的 source 含 train/test overlap override、public-test probe canary 與多次
  leaderboard tuning；其自述 honest GroupKFold stack 約 10.42。故 6.928
  不能當 strict unseen-well OOF evidence，也不能支撐 `<4`。

沒有複製或納入本輪網搜之第三方 repository code、外部資料或預訓練模型；
只使用專案既有的開源 runtime dependencies（含 scikit-learn；研究 probe
另用 XGBoost）。網路資源只作機制研究，因此不產生額外資料公平取得風險。

## 4. 候選版圖與排序

分數 1–5；收益、新穎、與舊失敗差異、可實作性是 5 較佳；成本、leakage
risk、tail risk 是 5 較差。

| 排名 | 候選家族 | 收益 | 新穎 | 差異 | 實作 | 成本 | 洩漏 | Tail |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Whole-path functional PCA / low-rank codebook | 4 | 4 | 5 | 5 | 2 | 2 | 3 |
| 2 | Learned contextual emission + structured decoder | 5 | 4 | 4 | 4 | 3 | 3 | 3 |
| 3 | Signed-state Bayesian lattice + MBR | 4 | 4 | 4 | 4 | 3 | 2 | 3 |
| 4 | Prefix-only retrieval mixture-of-experts | 4 | 3 | 4 | 4 | 3 | 3 | 3 |
| 5 | Artificial-cut / masked-suffix continuation | 4 | 4 | 5 | 3 | 4 | 4 | 4 |
| 6 | Switching state-space / particle smoother | 4 | 4 | 4 | 2 | 5 | 3 | 4 |
| 7 | Multi-reference posterior consistency | 3 | 5 | 5 | 2 | 5 | 3 | 3 |
| 8 | Robust multiscale GR emission | 3 | 3 | 3 | 4 | 2 | 2 | 2 |
| 9 | Optimal transport / Gromov-DTW | 3 | 4 | 5 | 3 | 4 | 2 | 4 |
| 10 | Loop-consistent stratigraphic SLAM graph | 4 | 5 | 5 | 2 | 5 | 4 | 5 |
| 11 | Train-only spatial surface / gradient field | 3 | 4 | 5 | 3 | 4 | 3 | 5 |
| 12 | Train-only simulator + multimodal inverse net | 5 | 5 | 5 | 1 | 5 | 4 | 5 |
| 13 | Prefix-anchored per-well GR calibration | 2 | 2 | 3 | 5 | 1 | 1 | 2 |
| 14 | Cross-fitted uncertainty / conformal tail controller | 2 | 3 | 3 | 4 | 2 | 4 | 1 |

### 4.1 每個候選的機制與最便宜 falsification

| 家族 | inference-safe 新資訊／跨越瓶頸方式 | 合規輸入 | 最可能失敗 | 最便宜且有判別力的 falsification | 若成功如何輸出 TVT |
|---|---|---|---|---|---|
| Whole-path PCA | outer-train 完整 residual paths 的低秩形狀流形；從逐 row 改成一次預測整段 | horizontal 六欄 | rare fault/dip 在 basis 外、平均化 tail | rank oracle 後做 fixed quick5 Ridge；看 pooled/tail | 預測 coefficients、重建 curve、插值回 MD、加 anchor |
| Learned emission | horizontal/typewell 多尺度 context、hard negatives 學相似 GR motif 的消歧 evidence | horizontal 六欄；typewell 兩欄 | top-K 有 support 但 calibration/selection 錯 | fold0 true-state rank、top-K oracle、最終 posterior RMSE | emission cost volume → forward-backward → posterior mean |
| Signed lattice | 保存所有 signed TVT paths；MBR 直接對 squared loss，而非 beam/MAP | 同上 | likelihood misspecified、posterior 貼邊或 diffuse | 同 quick5 比 Flat，檢查 entropy/edge mass/tail | 每 row `E[TVT]`；Viterbi 只作診斷 |
| Retrieval MoE | outer-train prefix-compatible wells 的 empirical future path shapes | query prefix/trajectory；fold-train index | prefix 相似但 future 分岔、support 不足 | top-K oracle vs safe weighted mean | transport residual paths 到 query MD 後取 mixture mean |
| Masked continuation | outer-train 多 artificial cuts 擴增 prefix→suffix episodes | 六欄；pseudo-label 只來自 fold train | artificial cut 與真 PS 不同、deterministic averaging | 小 TCN/cross-attention 與 Path-PCA 同 quick cohort | one-shot residual suffix |
| Switching SSM/PF | regime persistence、fault jump、outlier process、多峰 particles | 六欄＋typewell | particle degeneracy、simulator/dynamics 錯 | synthetic recovery + 20 wells exact-lattice comparator | weighted particle posterior mean |
| Posterior consistency | 多 reference pairwise posterior 的 transitive/cycle evidence | outer-train references | references 共同錯、O(K²–K³) | K=3–5 看 top-K support 和 final RMSE | consistency-fused posterior mean |
| Robust multiscale GR | rank/quantile、derivative、CWT/scattering、Student-t/outlier mixture | GR/MD | normalize 掉 amplitude、motif仍不唯一 | raw vs robust emission rank及同 decoder RMSE | 作 lattice/DP emission |
| OT/Gromov-DTW | series 內 relational geometry，抗 amplitude shift/stretch | GR/MD | diffuse many-to-one coupling、成本高 | 20 wells selected/oracle gap | barycentric TVT map，再作 path constraint |
| Stratigraphic SLAM | motif landmarks、loop closure、multiwell cycle consistency | fold-train graph＋query合法觀測 | 少量錯 loop 扭曲全圖、query無 edges | leave-well-out edge precision/coverage/cycle residual | landmark offsets 插值／積分 |
| Spatial field | outer-train `H=TVT+Z` 或 gradient 的 `(X,Y)` field | query XYZ；target只在 fold train fit | nonstationary structure、稀疏 support、積分漂移 | fold0 surface/gradient predictor vs Flat | `TVT=H-Z` |
| Synthetic multimodal inverse | train-fold path/motif/corruption simulator補 rare regimes | fold-train only | severe sim2real mismatch、mode probability錯 | posterior predictive/MMD，再做 tiny MDN quick | mixture posterior mean |
| Per-well calibration | pre-PS known TVT 對 own typewell 建 gain/offset/noise system ID | query prefix＋typewell | facies差異被誤當儀器差異 | prefix replay及 downstream quick RMSE | 校準 GR 後餵 decoder |
| Uncertainty controller | entropy、disagreement、OOD、coverage 的 cross-fitted risk | prediction-time diagnostics | rare tails 太少，risk model不泛化 | nested quick 比 pooled/p95/worst10 | 安全權重混合 advanced 與 Flat |

### 4.2 選出的 top 3

1. **Whole-path functional PCA**：直接改 representation 和 objective；低成本；
   能利用 training path-shape distribution，而不是只微調 row model。
2. **Learned contextual emission + structured decoder**：SLP-E1 只否定 handcrafted
   edges；learned hard-negative evidence 是真正不同的 candidate-ranking 機制。
3. **Signed-state exact lattice + MBR**：ROGII 的 TVT 可升、降、持平，不能照搬
   單調 DTW；exact posterior 可把「搜尋近似錯」與「模型假設錯」分離。

## 5. 實作與實驗

### 5.1 新實作

- `src/rogii/models/path_pca.py`
  - 64-point prefix representation；
  - 128-point anchor-relative whole-path representation；
  - outer-train PCA（fit 20 components）；
  - 從 prefix/trajectory 預測前 5 coefficients 的 scaled Ridge；
  - 只使用合法 horizontal columns。
- `src/rogii/models/path_pca_anchor.py`
  - 顯式把 residual curve 的 coordinate 0 固定為 anchor residual 0；
  - 用來 falsify endpoint interpolation bias。
- `src/rogii/models/path_pca_nested.py`
  - 每個 outer fold 內做 3-fold well-level inner CV；
  - 在 components `{2,5,8}`、alpha `{10,100,1000}` 中以 pooled rows 選擇；
  - 用來檢查固定 hyperparameter 的研究選擇偏差。
- `src/rogii/probabilistic_lattice.py`
  - signed TVT transitions；
  - robust own-typewell GR emission；
  - exact banded forward-backward；
  - posterior-mean MBR、Viterbi、entropy/variance/edge mass。
- `src/rogii/models/probabilistic_lattice.py`
  - runner wrapper、outer-fold transition/emission statistics；
  - inference-visible prefix affine/rate calibration；
  - missing GR neutral evidence。
- `scripts/contextual_emission_probe.py`
  - XGBoost hard-negative contextual emission；
  - coarse support diagnostic + signed lattice posterior decoder。
- `scripts/analyze_paired_oof.py`
  - exact-key paired OOF、per-well/tail/distance-decile、10k well-cluster bootstrap。
- `scripts/blend_oof.py`
  - 不讀 target 選權重；只做 caller-supplied fixed blend；
  - exact row-key/target identity checks。

### 5.2 Quick / probe 結果

Quick5 固定 cohort：五個 folds 各 100 train、20 validation；合計 100 unique
validation wells、501,440 rows。它不是 full estimate，只作淘汰／晉級。

| 方法 | Scope | Pooled RMSE | vs 同 cohort Flat | 關鍵結果 | 約 wall time |
|---|---:|---:|---:|---|---:|
| Flat | quick5 | 16.2891688456 | — | comparator | 20 s |
| fixed Path-PCA | quick5 | 15.6166617778 | -0.6725070678 | 4/5 folds 改善；bootstrap 95% `[-1.376,-0.007]` | 21 s |
| anchor-inclusive Path-PCA | quick5 | 15.6166622402 | -0.6725066055 | 相對原版只差 `+4.62e-7`；endpoint bias 無實質影響 | 24 s |
| nested Path-PCA | quick5 | 16.3493783633 | +0.0602095177 | inner model selection 未勝 Flat；不跑 full | 21 s |
| exact signed lattice | quick5 | 25.4417940888 | +9.1526252432 | 五折全退步；12 wells ≥40、4 wells ≥60；淘汰 | 204 s |
| learned contextual posterior mean | fold0, 20 wells | 14.8672709621 | +0.7695722077 | top-10 oracle 6.0975，但 posterior 選路失敗；不跑 full | 39.7 s |
| learned contextual marginal MAP | fold0 | 14.8781600259 | +0.7804612715 | 與 posterior mean 同方向失敗 | 同上 |
| P3（僅100 train/fold） | quick5 | 17.2423755293 | +0.9532066837 | 小 train subset 對 P3 不足；不是 full P3 否證 | 143 s |
| fixed 50/50 P3+Path | quick5 | 15.7025784597 | -0.5865903859 | bootstrap 95% 跨 0；方向正、證據弱 | 連同 P3 |

Learned emission 的結果尤其重要：候選 top-10 oracle 明顯優於 top-1
（coarse top-1 depth RMSE `13.6356`），但最終 posterior mean 仍輸 Flat。
因此「再產更多相似 GR candidates」本身不是解答；mode probability 與 branch
selection 才是主瓶頸。

Contextual probe 會讀 validation TVT 來計算 top-1/top-10 oracle diagnostics，
但 emission、posterior 與輸出 prediction 都不使用該 label；oracle 欄位只在
prediction 完成後評估，不是模型輸入。

### 5.3 Full strict OOF

| 模型 | Wells / rows | Pooled RMSE | vs Flat | vs P3 | Bootstrap 95% vs Flat |
|---|---:|---:|---:|---:|---:|
| Flat | 773 / 3,783,989 | 15.9098528707 | — | +0.4573046512 | — |
| P3 rerun | 773 / 3,783,989 | 15.4525482195 | -0.4573046512 | — | `[-0.8840,-0.0481]` |
| fixed Path-PCA | 773 / 3,783,989 | 15.3641531991 | -0.5456996716 | -0.0883950204 | `[-0.9431,-0.1865]` |
| fixed 50/50 P3+Path | 773 / 3,783,989 | **15.1720614751** | **-0.7377913957** | **-0.2804867444** | **`[-1.1041,-0.4005]`** |

Path-PCA 相對 P3 的 95% bootstrap delta 為 `[-0.5210,+0.3391]`，所以不能宣稱
Path-PCA 單獨可靠勝 P3；它的主要價值是與 P3 的互補。

固定 50/50 blend 在看 row-level blend outcome 前指定，沒有用 target fit 權重。
事後 target-fit 的全域 oracle path weight 是 `0.5471`，RMSE 只再從
`15.1720615` 降到 `15.1699467`；這個 oracle 不列為模型，只說明 0.5 已接近
二模型的 quadratic optimum，結果不是靠細緻 weight hunting。

各 fold：

| Fold | Flat | P3 | Path-PCA | 50/50 blend |
|---:|---:|---:|---:|---:|
| 0 | 15.898673 | 14.336629 | 14.263526 | 13.997011 |
| 1 | 14.143107 | 15.016035 | 14.640740 | 14.607048 |
| 2 | 15.712888 | 14.703689 | 14.955136 | 14.614047 |
| 3 | 16.832924 | 16.514088 | 16.276314 | 16.192186 |
| 4 | 16.836888 | 16.580191 | 16.574318 | 16.332951 |

Blend 4/5 folds 改善；fold 1 仍比 Flat 差 `+0.4639`。

### 5.4 Tail 與 catastrophic failures

| 模型 | p50 | p90 | p95 | Max | ≥40 wells | ≥60 wells | Worst 10% SSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat | 10.6651 | 22.9725 | 29.0108 | 70.6394 | 15 | 4 | 52.4763% |
| P3 | 10.4828 | 22.1853 | 27.2829 | 75.9887 | 14 | 2 | 52.4362% |
| Path-PCA | 10.2980 | 22.2334 | 27.6336 | 73.7369 | 13 | 2 | 51.2589% |
| 50/50 blend | **10.1007** | **22.0525** | **26.8746** | 70.7267 | **12** | **2** | 52.0394% |

Blend 在所有 prediction-distance deciles 都勝 Flat：

- decile 0：`6.5473 → 5.5378`；
- decile 1：`11.5284 → 10.1647`；
- decile 5：`16.4996 → 15.7984`；
- decile 9：`19.9224 → 19.4701`。

但遠距 improvement 變小，仍是無 anchor 的長期 drift。

最差 wells：

| Well | Flat | P3 | Path | Blend | 判讀 |
|---|---:|---:|---:|---:|---|
| `86454a6f` | 70.263 | 75.989 | 63.855 | 69.910 | Path救一部分、P3反向；平均仍 catastrophic |
| `1b1eba53` | 70.639 | 67.747 | 73.737 | 70.727 | 兩模型互相抵銷但沒有解決 |
| `5f4d2a52` | 56.915 | 56.388 | 57.230 | 56.804 | 幾乎無新增資訊 |
| `2fd68f7b` | 60.740 | 57.772 | 57.366 | 57.563 | 小幅改善，仍 catastrophic |
| `389ae58f` | 49.763 | 46.034 | 43.536 | 44.768 | 兩者皆有用 |
| `c8d9680c` | 42.770 | 44.540 | 43.899 | 44.206 | 兩者皆錯 |
| `fb03ae90` | 42.888 | 27.281 | 45.396 | 36.154 | P3 特有訊號；Path破壞 |
| `a959858c` | 65.359 | 55.331 | 33.546 | 41.856 | Path 特有強訊號 |

這些反例說明：

- P3 與 whole-path prior 有互補，但不是可靠的 per-well selector；
- validation target 事後選 fallback 會看似大幅改善，卻不合規；
- 現有 entropy/missingness/disagreement 尚不足以在 prediction 前識別哪一模型會錯。

## 6. 為何最有希望的方法仍失敗

### 6.1 Exact posterior 沒有救回 handcrafted likelihood

Signed lattice 不是 global beam 的近似重跑：它保留完整 posterior、允許 TVT
正／負／零 transition，並用 posterior mean 對準 squared loss。它 quick5
仍從 16.289 惡化到 25.442，表示問題不是 beam pruning；own-typewell raw GR
likelihood 與簡化 dynamics 對 unseen wells 是錯誤生成模型。

### 6.2 Learned emission 有 candidate signal，沒有 probability calibration

Hard-negative contextual model 把 top-10 oracle 拉到 6.10，但 posterior mean
仍 14.87。增加 candidate support 尚不足；需要知道哪一支 mode 是真路徑。
目前合法 prefix/trajectory 沒有提供穩定 selector。

### 6.3 Whole-path prior只提供 shrinkage，不提供新 observation

Path-PCA 改善了長距離 drift和多數 folds，證明 path-shape distribution 可用。
但它本質上是 conditional prior：unseen fault/dip regime 沒有 post-PS anchor 時，
係數仍向 training mean shrink。nested model selection也沒有帶來額外訊息。

### 6.4 外部成功案例的條件不對等

- Alyaev 的低誤差是同一 simulator 分布內，且是多 modes 中 best-path；
- geosteering 實務常用 resistivity、azimuthal response、ROP、seismic 或人工
  correlation constraints；
- 公開 ROGII 高 LB 分數含 overlap/probe/leaderboard adaptation，不能替代
  strict unseen-well OOF；
- 這些差異是 observation information 與 validation protocol，不是單純模型大小。

## 7. 對 `<4` 的最終判斷

目前 `<4` 只在 target-bearing whole-path representation oracle 中出現。
本輪最強合法模型 15.1721 還需要再降低 93.05% SSE；所有已測合法模型都沒有
顯示能把 top-K/path-shape oracle轉成可靠 mode selection。

因此，現有資訊下不值得繼續：

- 在同一 GR cost volume 上調 beam、temperature、transition sigma；
- 純粹增加 neural model 參數；
- 事後依 validation disagreement/missingness選 fallback；
- 再做 broad spatial `X,Y→surface`；
- 用 same-generator synthetic 成績推斷 real unseen-well performance。

只有出現以下之一，才值得重新開啟：

1. **同井 post-PS 稀疏 TVT anchors**：尤其在 path regime change/fault 前後；
   這是最小且最直接能把 target-bearing piecewise oracle 變成可辨識問題的新增資訊。
2. **人工 formation tops/correlation anchors**：提供 branch identity，而不只是 GR
   similarity。
3. **Azimuthal/deep resistivity 或多方位 GR/LWD**：新增方向／boundary distance
   observation，能消除 scalar GR alias。
4. **Seismic 或 inference-time formation surfaces**：提供跨井結構場，而不是只從
   稀疏 outer-train wells 外插。
5. **競賽規則明確允許且公平取得的額外 public modality/model**，並能在完全 offline、
   strict well-held-out validation 下重現顯著 improvement。

若只有更多相同格式的 training wells，重開門檻應是先證明：

- prefix-only nearest/support oracle 已接近 `<4`；
- 或 exact top-K posterior coverage 在 unseen wells 大幅上升；
- 且 mode probability 能在 nested well-level CV 校準。

否則更多相同資料可能只改善 prior mean，不會解決 branch non-identifiability。

## 8. 可重現命令與 artifact

### 8.1 驗證與測試

```powershell
rogii check `
  --data-root data/rogii-wellbore-geology-prediction `
  --fold-config configs/folds/main_5fold.csv

python -m pytest -q -W error
```

最終測試：`98 passed in 61.73s`。

環境：

- Python 3.12.10
- NumPy 2.4.3
- pandas 3.0.1
- scikit-learn 1.8.0
- SciPy 1.17.1
- XGBoost 3.2.0
- base Git HEAD `3f1207a777a2a76edd9b9d3cec30e978c0431eb9`
- working tree 原本已有使用者修改；本輪未 reset、未覆寫無關檔案。

### 8.2 Runner 命令

```powershell
python -m rogii run --model flat --profile full `
  --data-root data/rogii-wellbore-geology-prediction `
  --fold-config configs/folds/main_5fold.csv `
  --output-root .runs --run-id repro-flat-full --workers 4

python -m rogii run --model p3 --profile full `
  --data-root data/rogii-wellbore-geology-prediction `
  --fold-config configs/folds/main_5fold.csv `
  --output-root .runs --run-id repro-p3-full --workers 4

python -m rogii run --model path_pca --profile full `
  --data-root data/rogii-wellbore-geology-prediction `
  --fold-config configs/folds/main_5fold.csv `
  --output-root .runs --run-id repro-path-pca-full --workers 4
```

Quick5 使用：

```powershell
$models = @("flat", "p3", "path_pca", "path_pca_anchor",
  "path_pca_nested", "lattice")
foreach ($model in $models) {
  python -m rogii run --model $model --profile quick `
    --data-root data/rogii-wellbore-geology-prediction `
    --fold-config configs/folds/main_5fold.csv `
    --output-root .runs --run-id "repro-$model-quick5" `
    --folds 0,1,2,3,4 `
    --max-train-wells 100 --max-validation-wells 20 `
    --seed 20260711 --workers 4
}

python scripts/contextual_emission_probe.py `
  --data-root data/rogii-wellbore-geology-prediction `
  --fold-config configs/folds/main_5fold.csv `
  --output-root .runs/repro-contextual-emission-fold0 `
  --fold 0 --max-train-wells 100 --max-validation-wells 20 `
  --seed 20260711 --emission-weight 0.5 --transition-sigma 0.5
```

### 8.3 Fixed blend 與 paired analysis

```powershell
python scripts/blend_oof.py `
  --first .runs/repro-path-pca-full/oof_predictions.csv.gz `
  --second .runs/repro-p3-full/oof_predictions.csv.gz `
  --first-weight 0.5 `
  --output .runs/repro-p3-path-blend-full/oof_predictions.csv.gz `
  --metadata .runs/repro-p3-path-blend-full/blend_metadata.json

python scripts/analyze_paired_oof.py `
  --baseline .runs/repro-flat-full/oof_predictions.csv.gz `
  --candidate .runs/repro-p3-path-blend-full/oof_predictions.csv.gz `
  --output-root .runs/repro-analysis/p3-path-blend-full `
  --bootstrap-samples 10000
```

`run-id` 必須指向尚不存在的目錄；上面的 `repro-*` 名稱刻意與本輪既有
`final-*` artifacts 分開。若已重跑過一次，請再換新的 run-id。

### 8.4 主要 artifacts

- `.runs/final-flat-full/summary.json`
- `.runs/final-p3-full/summary.json`
- `.runs/final-path-pca-full/summary.json`
- `.runs/final-p3-path-blend-full/blend_metadata.json`
- `.runs/final-analysis/path-pca-full/paired_summary.json`
- `.runs/final-analysis/p3-full/paired_summary.json`
- `.runs/final-analysis/path-vs-p3-full/paired_summary.json`
- `.runs/final-analysis/p3-path-blend-full/paired_summary.json`
- `.runs/final-analysis/p3-path-blend-full/paired_per_well.csv`
- `.runs/final-analysis/p3-path-blend-full/distance_deciles.csv`
- `.runs/final-contextual-emission-fold0/summary.json`
- `.runs/final-lattice-quick5/summary.json`
- `.runs/final-path-pca-nested-quick5/summary.json`
- `.runs/final-path-pca-anchor-quick5/summary.json`
- `.runs/slp-e0-full/summary.json`
- `.runs/slp-e1-full/summary.json`

注意：`.runs/final-path-pca-quick5` 是早期中斷殘留；有效 fixed quick artifact 是
`.runs/final-path-pca-quick5-v2`。

### 8.5 重要 hashes

- `AGENTS.md`：
  `BFAAB4F9888C924C427179F5C3013FF4942E25E0B058A5389D833E89CA9A250B`
- fixed Path-PCA source：
  `FBC3B56A82364A3647DCF7F4921BCC3C83D903BD0879E50ACCF4E78400EBF31A`
- full Path-PCA OOF gzip：
  `7BA93D23E546EDCE232AB5D3D999E4EEB1FDF95B1879493A28CD497A9E4EE99A`
- paired analyzer：
  `3C1BB0ADC128CE927FE00A0B24DCCEA2453211C9C3BFB1EF168769334D3D73E3`
- fixed blend utility：
  `BEF1C000237EB1EEBDF53427C0DDFC1E404149752CA370D3F80F0687B798C628`

## 9. 限制與誠實聲明

1. fixed Path-PCA 的 fit/predict execution 經獨立 audit，沒有 leakage、row
   misalignment 或 metric error；但其 hypothesis/hyperparameters 是本輪研究過程
   中由 quick evidence 晉級，full point estimate仍有一般研究選擇偏差。
2. 為此另做完全 outer-safe inner-CV hyperparameter variant；它 quick5 沒有勝
   Flat，因此不跑 full。這降低了把 fixed configuration 當普遍最優的信心。
3. 50/50 blend 權重沒有用 target fit，但「嘗試 blend」本身是在看到兩模型
   個別 full 結果後的 follow-up。它是有效 fixed-prediction OOF estimate，不是
   獨立未曝光 confirmatory trial。
4. 本輪沒有重建 Path-PCA/P3 的歷史 anti-twin/spatial stress suite；以 main
   tail、fold heterogeneity、distance deciles和歷史 P3 stress作保守判斷。
5. 公開競賽 notebooks/claims 只作 source audit；沒有把 public LB 或 overlap
   cases 當 strict evidence。
6. 沒有 Kaggle upload，沒有外部大型下載或付費資源，沒有把輸出寫入 `data/`。
