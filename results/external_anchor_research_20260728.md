# External Anchor Research：Heel-Registered Sparse-Anchor Student

日期：2026-07-28  
資料範圍：773 train wells、3,783,989 prediction rows  
驗證：固定 strict 5-fold well-level OOF、5 seeds、10,000 次 paired well-cluster bootstrap

## 1. Executive conclusion

**成功，且通過本輪預先定義的完整成功條件。** 最佳 deployable 候選是
**Heel-Registered Sparse-Anchor Student**：先由既有 Sparse-Anchor Student 預測八個
anchors，再用同一口井可見 heel 的 GR/`TVT_input` 校準 horizontal tool 到該井
typewell GR，最後用 21-quantile 的一維分布註冊後驗平均，估計一個 inference-safe
common-mode TVT shift。

正式 5-seed ensemble 的 pooled row-RMSE 是 **14.1708478925 ft**；歷史 incumbent
Anchor Student 是 **14.8200800906 ft**，paired delta 是 **-0.6492321981 ft**。
10,000 次 bootstrap 的 delta q2.5/q50/q97.5 是
**-0.9247288343 / -0.6451989315 / -0.3772842000 ft**，整個區間低於零。

為避免把 seed ensembling 的收益錯歸因於 registration，另作完全配對的機制消融：
5-seed 未註冊 Anchor ensemble 是 **14.7880434145 ft**；註冊本身的 matched delta
是 **-0.6171955220 ft**，bootstrap 區間
**[-0.8924450356, -0.3506084569] ft**。五個 folds、五個 seeds 的 matched
方向全部改善。因此主要收益確實來自 heel/typewell registration，而不是單純多 seed。

Tail 沒有靠增加災難井換取均值：`>=40 ft` wells 由 12 降至 10，`>=60 ft`
由 2 降至 0，max 由 63.699058 降至 54.779067 ft，worst-10% SSE share 由
0.508357 降至 0.490065。唯一不利的 aggregate tail 指標是 per-well p95 由
26.624218 微升至 26.690231 ft（+0.066014 ft），因此不能宣稱所有 tail quantiles
都改善。

最終模型**沒有使用任何外部資料、外部 pretrained weights、付費 API 或線上服務**；
外部資料只用於候選 scouting/來源風險稽核，沒有進入 fit 或 inference。已在動態列舉的
3 個 visible-test wells、14,151 rows 上完成 25-state finite inference、sample-ID
順序核對；**沒有建立或上傳 Kaggle submission CSV**。

## 2. External sources and evidence

外部證據用來提出機制，不用來替代本專案固定 folds 的實驗證據。

1. ROGII 公開 working note 報告其 762-well 自有 CV 的主要災難來源是 whole-well
   bias，continuous correction 優於 discrete selection，而多種 post-hoc gating、
   self-calibration 與 prefix-to-suffix bias transfer失敗。其 folds/資料清理與本專案
   不同，數字不可直接比較，但 failure mechanism 與本地 common-mode decomposition
   一致：[Daulet Toibazar, *Working Note: Our Solution, the Failures Behind It, and What
   the Data Taught Us*](https://www.kaggle.com/writeups/daulettoibazar/working-note-our-solution-the-failures-behind-it)。
2. 一維 Wasserstein/optimal-transport distance 可由 quantile functions 表達；本方法
   使用離散等距 quantiles 的 clipped squared surrogate，而不是引入 OT library：
   [Peyré & Cuturi, *Computational Optimal Transport*](https://arxiv.org/abs/1803.00567)。
3. USGS 的 gamma-log workflow 明確指出跨井/跨 run 的工具尺度差異，並以平移與縮放
   對齊 type-log mean/std。這支持「先用合法 heel 配對做 per-well tool calibration」：
   [USGS COLORLITH / Collins](https://pubs.usgs.gov/of/1998/of98-487/collins.html)。
4. CWT、normalization、DTW 在 well-log correlation 的先例支持 multiscale/sequence
   候選，但沒有證明會在本競賽改善：
   [Behdad 2019](https://doi.org/10.1016/j.jappgeo.2019.05.007)、
   [Soft-DTW](https://proceedings.mlr.press/v70/cuturi17a.html)、
   [FWHT LWD/EWL boundary enhancement](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2026.1736164/full)。
5. 低頻 basis 候選取材自 interpretable trend/basis forecasting，而不是照搬模型：
   [N-BEATSx](https://arxiv.org/abs/2104.05522)、
   [N-HiTS](https://ojs.aaai.org/index.php/AAAI/article/view/25854)。
6. 後續 cross-tool encoder portfolio 參考
   [TS2Vec](https://ojs.aaai.org/index.php/AAAI/article/download/20881/20640)、
   [TF-C](https://github.com/mims-harvard/TFC-pretraining)、
   [IBM Tiny Time Mixer](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1)
   與 [S4](https://github.com/state-spaces/s4)。這些均未進入本輪最佳模型。
7. Q-3D cumulative tortuosity 是後續低成本 geometry 候選：
   [Jing et al. 2022](https://doi.org/10.1016/j.petlm.2021.03.008)、
   [public ROGII toolkit](https://github.com/mycarta/rogii-geosteering-toolkit)。公開 toolkit
   的自有 ablation 只能當 hypothesis evidence，不能當本專案分數。
8. 研究期間檢查了 681-well Permian Basin GR 的
   [ChronoLog dataset](https://github.com/zsylvester/chronolog_data) 與
   [Sylvester 2023](https://zsylvester.github.io/papers/chronolog/)。可重現 overlap audit
   的 raw / linear-detrended / first-difference 最大 Pearson 分別為
   `0.973114 / 0.959059 / 0.857007`；前兩者在 0.90 threshold 保留
   `20,120 / 3,420` 對。Visible-test maxima 分別為
   `0.927796 / 0.875044 / 0.703212`。因此無法排除來源重疊或低頻形狀巧合；初始未能
   重現的「max 0.897327、無 >0.90」screen 已正式撤回，這份資料不用於本輪模型。

## 3. Competition, license, and offline-availability audit

| 資源 | 精確版本 | License / availability | 本輪用途 | Test inference inputs / leakage judgement |
|---|---|---|---|---|
| 最終 heel registration | 本 repo `heel_quantile_registration_v1` | 本地原始碼；只依賴既有 NumPy/Pandas/scikit-learn | **正式模型** | horizontal `GR,TVT_input`；typewell `TVT,GR`；base prediction。無 target/surface/Geology/PNG |
| ChronoLog GR | commit `336e3e2a20df029ecccbe63611a78550204d64ae`，2026-07-28 暫存 clone | CC BY 4.0；公開、可離線 | overlap scouting，**排除且未 fit** | 若採用只會是 GR-only pretraining、inference 無需外部檔；但目前來源重疊風險未排除，故禁用 |
| IBM TTM-R1 | HF revision `f04bebdd4c13475b006ce72672e74c9dc28871dc`；weight SHA-256 `30f8d9be619518e18c4c15c6ee504e491b9624b48e251661cd5f967bce097731` | Apache-2.0、public/ungated/offline | model-card audit，未下載到正式 runtime、未測 | 若測試需 allowed sequence；未見 ROGII 資料，但 constituent dataset licenses仍須逐項封存 |
| TS2Vec | commit `b0088e14a99706c05451316dc6db8d3da9351163` | MIT | architecture lead，未使用 weights | 必須 outer-fold fit/adapt；無競賽 target可跨 folds |
| TF-C code | commit `96675826e9ef234a9b01cc63d484c66cb0441bc0` | MIT code；部分預訓練資料有 NC 條款 | 只保留 architecture/loss idea | 不使用其 pretrained weights/data |
| ROGII Q-3D toolkit | commit `d83213f97999423908228c85e67a96072c8b2a5a` | MIT、公開 | hypothesis evidence，未 import code | 若重作只讀 allowed XYZ/MD；公開自有 CV 不當作本地證據 |
| N-BEATSx / S4 | commits `fe116d21785fca55670d258756e7c35fcb613eca` / `e757cef57d89e448c413de7325ed5601aceaac13` | MIT / Apache-2.0 | representation leads，未使用 | 競賽資料 adaptation 必須 outer-fold safe |

本輪 production path 沒有 network call、remote API、外部 model file 或 runtime download；
因此符合 offline execution。公開資料與模型的完整機器可讀稽核位於
`.runs/external-anchor-research-scouting-20260728/external_source_audit.json`。競賽規則仍以
[官方 rules page](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules)
為準；若未來採用任何 external weights，必須先把 exact revision/license/weights做成
公平可取得的離線 artifact再重跑。

## 4. Baseline reproduction

獨立 audit 重現 incumbent：

| 項目 | 結果 |
|---|---:|
| OOF wells / rows | 773 / 3,783,989 |
| Anchor Student pooled RMSE | 14.8200800906 |
| P3 / Path-PCA / 50-50 blend | 15.4525482195 / 15.3641531991 / 15.1720614751 |
| Anchor vs blend delta | -0.3519813845 |
| 原報告 bootstrap q2.5/q50/q97.5 | -0.6425530121 / -0.3477493390 / -0.0526412131 |
| Anchor error RMSE | 15.2808833322 |
| common-mode / centered-shape | 12.4679019776 / 8.8349768359 |
| p50 / p90 / p95 / max well-RMSE | 10.110716 / 21.001377 / 26.624218 / 63.699058 |
| wells >=40 / >=60 | 12 / 2 |

落盤 baseline OOF SHA-256 是
`88611a25dd1ee6d86833eafeaf9dbb29b5c6f4f7ac9a3075440f02b881668645`。
Reference seed 重新由八 anchors reconstruction 後，與 incumbent row prediction 的
maximum absolute delta 是 `5.46e-12`；落盤後獨立 chunked audit 的 delta 是精確 0。

基準 provenance caveats：原 Anchor full run 是一次 audit failure 後 resume；descriptor
全部命中 shared cache，而 cache key 未綁 source/data checksum；ExtraTrees 候選由 quick
cohort選出後才在包含該 cohort的 full OOF報告，因此 fold-safe，但不是 nested
model-selection estimate。這些限制沒有被本報告隱藏。

## 5. Hypothesis portfolio

| 候選 | 新增機制 / inference-safe information | 預期 failure mode | 最小否證實驗 | 結果 |
|---|---|---|---|---|
| Heel-calibrated quantile registration | 本井 visible heel 建立 horizontal-tool→typewell-tool affine bridge；suffix GR分布對 shifted predicted path 的 typewell GR | absolute/common mode、whole-well bias | q21 posterior vs MAP/wrong typewell/random shift | **通過 quick、進 full** |
| H-space physical extrapolation | `H=TVT_input+Z` 的末 250 ft robust slope外推 | terminal slope / low-frequency trend | direct line及5% bounded blend | direct catastrophically failed；blend CI跨0，停止 |
| Polynomial low-rank anchor head | 八 anchors改成六個 orthogonal polynomial coefficients | smooth long-horizon shape | ExtraTrees basis head vs shuffled targets | 無 paired signal；shuffle同等，停止 |
| Typewell median denoise variant | horizontal與typewell都 median-7 | tool rotation/noise | quick後單-seed full | quick較好、full較差，視為 selection overfit，停止 |
| Paired-prefix contrastive encoder | 逐井合法 horizontal/typewell paired windows，cross-tool domain bridge | centered shape、aliasing、suffix support | raw NCC / wrong typewell / random-init / phase-shuffle controls | 未執行；列為下一個高資訊實驗 |
| Q-3D cumulative tortuosity | path-integrated steering/oscillation，而非既有 local dogleg aggregates | terminal shape | Q-3D vs dogleg-only/phase-shuffle/random-well feature | 未執行；低成本後續 |
| Frozen TTM/S4 encoder | 外部通用長序列 representation，raw anchor另保留 | long horizon | pretrained vs identical random-init/weight-permuted | 未執行；只有 pretrained勝 control才可進 full |

這次重訪 GR/typewell matching 與既有 lattice/sequence route 的差異是：不再做 row-wise
candidate selection、beam 或 mode pruning；新增了**本井 heel 的合法跨工具配對校準**，
並只估計低維 common-mode shift，以 posterior mean 保留 repeated-bed ambiguity。

## 6. Formal quick experiment

完全沿用既有 quick Anchor OOF 的 100 validation wells、489,284 rows，以及各 fold
outer-training subset。所有 fit 均 well-separated；registration沒有 cross-well fit。

| Candidate | pooled RMSE | delta vs Anchor | bootstrap q2.5 / q50 / q97.5 | improved folds | 判斷 |
|---|---:|---:|---:|---:|---|
| Anchor Student | 13.403931 | 0 | — | — | baseline |
| Heel q21 posterior | **12.877976** | **-0.525956** | -1.327964 / -0.524621 / +0.242442 | 4/5 | point signal大、controls支持；進 full |
| Heel hard MAP | 22.765502 | +9.361570 | +6.029773 / +9.311785 / +12.546967 | 0/5 | mode commitment失敗 |
| Wrong typewell | 14.262856 | +0.858924 | -0.125104 / +0.869189 / +1.834391 | 1/5 | own typewell signal必要 |
| Random common shift | 26.094039 | +12.690108 | +9.619977 / +12.658114 / +15.559222 | 0/5 | 任意平移失敗 |
| H-space direct line | 36.922119 | +23.518188 | +16.790620 / +23.291929 / +30.000274 | 0/5 | 停止 |
| H-space 5% blend | 13.394018 | -0.009913 | -0.280601 / -0.010825 / +0.255992 | 2/5 | CI跨0，停止 |
| Low-rank anchor-6 | 13.434636 | +0.030704 | -0.104300 / +0.032513 / +0.163458 | 2/5 | 停止 |
| Low-rank shuffled target | 13.433655 | +0.029724 | -0.714879 / +0.035329 / +0.727954 | 3/5 | 原候選未勝負控制 |

Heel posterior quick CI 仍跨零；進 full 的理由不是把它包裝成顯著，而是 effect size
0.526 ft、4/5 fold direction、p95由24.99降至21.69，以及三個針對機制的負控制都按
預期崩潰。Hard MAP 同時證明保留 mode uncertainty、輸出 posterior mean 是關鍵。
Distribution objective 的 suffix order-reversal loss delta 精確為 0；這是明確 ablation：
本候選新增的是分布/common-mode訊號，不宣稱學到 temporal shape。

另一個迭代是 typewell也 median-7：quick RMSE 12.774912，看似更好；但單-seed full
是 14.273007，弱於 raw-typewell locked variant 的 14.205713。這是明確的 quick
selection overfit，因此正式模型保留 raw typewell、只平滑 horizontal GR。

## 7. Full strict OOF results

| fold | historical Anchor | 5-seed base ensemble | registered ensemble | delta vs historical | matched registration delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 14.569310 | 14.541696 | 14.077945 | -0.491365 | -0.463751 |
| 1 | 13.697364 | 13.665781 | 12.955540 | -0.741825 | -0.710241 |
| 2 | 14.255198 | 14.323222 | 13.704693 | -0.550505 | -0.618530 |
| 3 | 15.984256 | 15.875778 | 15.271421 | -0.712835 | -0.604357 |
| 4 | 15.508239 | 15.455440 | 14.758727 | -0.749512 | -0.696712 |
| **pooled** | **14.820080** | **14.788043** | **14.170848** | **-0.649232** | **-0.617196** |

五折同向，而且最弱 fold 0仍改善0.464 ft（matched）。這與 Anchor Student 原本
fold 0相對 blend惡化不同；registration 不是只靠原本容易 folds。

## 8. Seed and bootstrap uncertainty

| seed | unregistered Anchor | registered | matched delta | matched bootstrap q2.5 / q50 / q97.5 |
|---:|---:|---:|---:|---:|
| 20260711 | 14.820080 | 14.205713 | -0.614368 | -0.892004 / -0.610783 / -0.342011 |
| 20260712 | 14.763631 | 14.136666 | -0.626965 | -0.901206 / -0.623887 / -0.360442 |
| 20260713 | 14.799153 | 14.191963 | -0.607189 | -0.882353 / -0.603472 / -0.339584 |
| 20260714 | 14.815624 | 14.193214 | -0.622410 | -0.894973 / -0.618528 / -0.357228 |
| 20260715 | 14.814068 | 14.185919 | -0.628149 | -0.902422 / -0.624837 / -0.362583 |

Registered single-seed RMSE 的 mean/std/min/max 是
14.182695 / 0.026718 / 14.136666 / 14.205713。所有 seeds 的 matched bootstrap
上界都低於零，表示效果不依賴某一次 tree randomness。Repeated fit 的 maximum
anchor delta是 `7.11e-15`；不是 bit-for-bit zero，但遠低於固定的 `1e-12`
numerical-determinism tolerance。

## 9. Common-mode, shape, terminal, and tail decomposition

| 指標 | Anchor Student | Registered ensemble | 變化 |
|---|---:|---:|---:|
| anchor RMSE | 15.280883 | 14.586184 | -0.694699 |
| common-mode RMSE | 12.467902 | 11.613421 | **-0.854480** |
| centered-shape RMSE | 8.834977 | 8.825260 | -0.009717 |
| common-mode MSE share | 66.57% | 63.39% | -3.18 pp |
| first anchor RMSE | 8.988449 | 9.003607 | +0.015158 |
| terminal anchor RMSE | 19.230152 | 18.579584 | -0.650567 |

結果與機制一致：幾乎全部 anchor gain 來自 common mode；centered shape基本不動。
Terminal anchor也改善，原因是 constant correction在長 horizon的 Anchor bias較大時有
更高價值，但這不等於模型學會 suffix curvature。

| Tail metric | Anchor | Registered ensemble |
|---|---:|---:|
| p50 / p90 / p95 well-RMSE | 10.110716 / 21.001377 / 26.624218 | 9.796211 / 20.331417 / 26.690231 |
| max | 63.699058 | 54.779067 |
| wells >=40 / >=60 | 12 / 2 | 10 / 0 |
| worst-10% SSE share | 0.508357 | 0.490065 |

三口原本 `>=40` wells（`389ae58f`, `43e16325`, `a959858c`）被救回；一口
`91db7070` 新進 `>=40`，淨減兩口。最大負面 per-well delta仍達 +13.53 ft，說明
posterior correction 並非每井安全；411 wells改善、362 wells惡化，median per-well
RMSE delta只有 -0.0624 ft，pooled gain仍由較大/較差 wells主導。

Candidate-support diagnostics全部在 prediction 前可算，但目前不適合當 gate：最佳
一般 rank signal（minimum cost）對 candidate RMSE 的 Spearman只有約0.18；posterior
shift std約0.13。Signed mean shift 對僅10口 catastrophic wells 的 exploratory AUC約
0.73，但方向是 post-hoc、event數太少，而且對 RMSE delta 的 rank correlation僅約
0.08。故本輪**沒有**用 uncertainty做 shrink、fallback或routing。

## 10. Leakage, artifact, and inference audit

- Horizontal production loader只讀 `MD,X,Y,Z,GR,TVT_input`；registration實際使用
  `GR,TVT_input`與base prediction。
- Typewell production loader只讀 `TVT,GR`。沒有讀 horizontal target `TVT`、formation
  surfaces、`Geology`、PNG或 validation prediction-zone labels。
- 每個 student estimator只 fit outer-training wells：fold 0/1/2各618 wells，fold 3/4
  各619 wells；25 states均逐一 unpickle、核對 seed/fold/training IDs及 validation零交集。
- Registration沒有 cross-well learned state；每井 prefix affine calibration只使用該井
  可見 heel與自己的typewell。
- Descriptor、imputer、clip range、ExtraTrees均維持 outer-fold fit provenance；固定
  P3/Path-PCA states來自同一 outer folds，沒有 refit validation資料。
- OOF 3,783,989 rows無 duplicate/nonfinite/key/truth mismatch。Candidate OOF SHA-256：
  `4606cf8e07973f801f3612e9cfe9ecc11588ee7902573f085b350549f0ea60ba`。
- 獨立 chunked disk re-audit 重讀 candidate與baseline OOF：truth delta=0、historical
  Anchor delta=0、summary RMSE delta=`1.78e-15`；兩組 bootstrap重新抽樣皆精確一致。
- `data/` before/after manifest完全相同：2,327 files、1,325,112,684 bytes、
  name/size/mtime SHA-256
  `dec44ac09997b60e900c70064349f0fc9a1d558a9cda5c05aaa3530902fe4514`。
- 動態 visible test：wells `000d7d20,00bbac68,00e12e8b`，14,151 rows，25-state
  ensemble全部 finite，sample IDs完全同序；prediction SHA-256
  `08d1a700bb398a576fa12c029a339bc55077e47f99afe6a3313e4d528055d9b8`。
- 完整 test suite：`330 passed`；27個 warnings均來自既有 synthetic degeneracy下的
  scikit-learn PCA/PLS/NumPy diagnostics，沒有 test failure。
- 沒有 submission CSV、沒有 Kaggle upload。

主要 artifacts：

- `.runs/heel-registered-anchor-full-20260728/summary.json`
- `.runs/heel-registered-anchor-full-20260728/paired_bootstrap.json`
- `.runs/heel-registered-anchor-full-20260728/artifact_audit.json`
- `.runs/heel-registered-anchor-full-20260728/independent_reaudit.json`
- `.runs/heel-registered-anchor-full-20260728/leakage_check.json`
- `.runs/heel-registered-anchor-full-20260728/test_inference_audit.json`
- `.runs/heel-registered-anchor-full-20260728/pre_prediction_risk_analysis.json`
- `.runs/heel-registered-anchor-full-20260728/oof_predictions.csv.gz`
- `.runs/heel-registered-anchor-full-20260728/models/seed_*/fold_*.pkl`

## 11. Best-model architecture and why it works

對每一個 base Anchor Student suffix prediction：

1. Horizontal prefix與suffix GR各自用 row interpolation補有限缺值，再作 centered
   rolling median，width=7。
2. 在 visible prefix中，以 `TVT_input`插值 own typewell GR，OLS fit
   `typewell_GR = intercept + slope * horizontal_GR`。這把每井工具 gain/bias估計限制在
   合法 heel。
3. 在 `[-40,+40] ft`、step `0.5 ft` 的161個 constant shifts上，把base path平移；
   suffix最多等距抽1,000 rows。
4. 比較 calibrated horizontal suffix GR與shifted path所讀到的typewell GR之21個等距
   quantiles；每個 squared quantile residual cap=2,500。若typewell coverage低於0.95，
   加 `1000*(0.95-coverage)^2`。
5. 用 `exp(-(cost-min_cost)/(2*max(prefix_MSE,1)))`形成posterior，輸出posterior mean
   shift，不輸出MAP。這個shift同時加到row predictions與八個predicted anchors。
6. Full test prediction平均5 seeds × 5 fold states；OOF只平均該well所屬outer fold的5 seeds。

它新增的不是 target proxy，而是 test time已存在、之前沒有被 Anchor Student直接用來
校正絕對datum的訊號：**own heel所提供的跨工具映射 + own suffix GR distribution + own
typewell**。Repeated beds使hard alignment不可靠；quantile objective刻意捨棄順序，
posterior mean再避免commit到錯誤mode。這正好瞄準66.6% anchor MSE的common mode，
同時不假裝解決仍缺乏訊號的centered shape。

## 12. Exact reproduction commands

在 repo root、現有環境：

```powershell
rogii check --data-root data/rogii-wellbore-geology-prediction

python scripts/run_external_anchor_research.py `
  --profile quick `
  --data-root data/rogii-wellbore-geology-prediction `
  --fold-config configs/folds/main_5fold.csv `
  --output-root .runs `
  --run-id external-anchor-research-quick-20260728 `
  --cache-root .runs/cache `
  --workers 4

python scripts/run_external_anchor_research.py `
  --profile full `
  --data-root data/rogii-wellbore-geology-prediction `
  --fold-config configs/folds/main_5fold.csv `
  --output-root .runs `
  --run-id heel-registered-anchor-full-20260728 `
  --cache-root .runs/cache `
  --workers 4

python scripts/audit_external_anchor_artifacts.py `
  --run-root .runs/heel-registered-anchor-full-20260728 `
  --baseline-root .runs/sparse-anchor-student-full `
  --fold-config configs/folds/main_5fold.csv

python -m pytest -q
```

若 full在所有昂貴 artifacts已寫完、只剩final audit時中斷，可用完全相同命令加
`--resume`；finalizer不會重算 OOF，且仍要求 independent disk re-audit。

## 13. Files changed

- `src/rogii/heel_registration.py`：純 inference-safe registration與manifest。
- `src/rogii/models/heel_registered_anchor_student.py`：deployable model wrapper/state。
- `src/rogii/diagnostics/external_anchor_research_experiment.py`：quick/full、5-seed、
  bootstrap、tail/decomposition/leakage/test audit harness。
- `src/rogii/models/__init__.py`：model registry。
- `scripts/run_external_anchor_research.py`：正式 runner。
- `scripts/audit_external_anchor_artifacts.py`：獨立chunked disk/model-state re-audit。
- `scripts/audit_external_gr_overlap.py`：外部GR overlap/source-risk audit。
- `tests/test_heel_registration.py`。
- `tests/test_heel_registered_anchor_student.py`。
- `tests/test_external_anchor_research_experiment.py`。
- `results/external_anchor_research_20260728.md`：本報告。

既有使用者修改的 `cli.py`、`p3.py`、`runner.py`、Sparse-Anchor Student原始碼與原報告
沒有被回退、覆寫或重新格式化。

## 14. Negative results and stopping reasons

1. **Hard MAP**：quick 22.7655，六口>=40；mode commitment直接放大aliasing，停止。
2. **Wrong typewell**：14.2629且CI中心惡化0.87；證明gain不是任意GR distribution
   regularizer，也阻止廣泛跨井prior重訪。
3. **Random shift**：26.0940；排除「任何額外well constant都能改善」的解釋。
4. **H-space line**：36.9221、23口>=40、8口>=60；已知 flat-vs-slope oracle證據再次
   顯示heel slope不可直接外推。5% blend只有-0.010且CI跨0，停止。
5. **Low-rank polynomial anchor head**：+0.031 ft，shuffled-target control相同；output
   basis本身沒有新增可預測information，停止。它不否定未來用新encoder預測basis。
6. **Typewell median denoise**：quick較好但full較差；不採用quick-only選模結果。
7. **Sequence shuffle**：quantile objective精確order-invariant。這不是假negative control，
   而是界定模型能力：它只解common-mode，不解sequence shape。
8. 已知 lattice、beam/ranker、broad spatial prior、dense local field、sequence v1、prefix
   replay等routes沒有因本輪成功而被重新包裝；本輪的新information是heel tool bridge。

## 15. Remaining risks, hidden-test uncertainty, and next step

1. Full OOF的model selection仍非nested：q21/median/search range來自包含於full的quick
   cohort，bootstrap CI是conditional于已選模型，不包含整個研究選擇過程。
2. Visible test三井是authoring copies；finite smoke不能證明hidden geology/tool distribution
   相同。Heel calibration對tool-shift有物理合理性，但hidden shift仍可能改變效益。
3. 362/773 wells仍惡化，p95微升；現有pre-prediction support signals不足以安全gate，
   不應手工fallback。
4. Centered-shape RMSE幾乎未動（8.835→8.825），與degree-6 smooth oracle 3.052仍有
   大gap；14.171不是接近representation ceiling。
5. Descriptor shared cache不是content-addressed；獨立 OOF/state/source audits降低風險，
   但未提供完整原始data content hashes。
6. ChronoLog overlap screen無法排除高相關來源；在來源/provenance釐清前禁止加入。

下一個最高資訊價值實驗是**outer-fold-safe paired-prefix cross-instrument encoder + 顯式
common/slope/zero-mean smooth-shape head**。它保留本輪已驗證的heel bridge，但改用
matched horizontal/typewell prefix windows學跨工具motif，再從allowed suffix GR預測
centered shape；controls必須含 raw affine/NCC、wrong typewell、prefix↔TVT shuffle、
random-init same architecture、phase-shuffled GR。先驗收 held-out anchor terminal與
centered-shape，再與本輪 registration疊加。較便宜的並行 probe是Q-3D tortuosity；
若只改善local geometry而不改善terminal/shape，立即停止。

本輪沒有建立或上傳 Kaggle submission。
