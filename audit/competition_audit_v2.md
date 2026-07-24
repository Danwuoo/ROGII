# ROGII — Wellbore Geology Prediction

# Competition Audit V2

稽核日期：2026-07-22（Asia/Taipei）  
Run ID：`20260722T202906+0800`  
Workspace：`C:\Users\wurre\Desktop\ROGII`  
Git revision：`f115f574b8a927cb9be133f2e877526d2f84c72d`  
Research memory：archive `v4`，registry digest `8cec380adda1cb6b8a7dc10e27323f6d3ec99f5c8ce90bfe40a2f75f2948fb77`

> 本報告不承諾Kaggle排名、獎項或private-leaderboard成績。它區分官方事實、data observation、已驗證實驗、外部證據、推論、假設與recommendation。Protected confirmatory 80 labels未讀；沒有建立或提交competition submission。

## 1. Executive verdict

ROGII現階段沒有一條可宣稱已promotion的後續模型路線。唯一完全穩定的reference是flat last-known：773 wells、3,783,989 prediction rows、pooled row-RMSE `15.909852870734552`。[M:`experiment:p1-flat-baseline`]

Frozen P3是最低full strict OOF point estimate：`15.4525482195`；但相對P2的paired delta `-0.1444683582`之95% CI為`[-0.3306190868, 0.0420698645]`，且spatial/anti-twin stress為`16.1361032223`，所以只保留作comparator，不是promoted model。[M:`experiment:p3-alignment-features`]

後續正式實驗把V1的預設流水線逐層否證或凍結：global beam缺support、broad spatial在disjoint743顯著劣化、change-point objective與summary ranker無法把oracle轉成top1、sequence雖改善ranking但缺tail support與calibrated risk、dense local field在grid coverage正常時仍劣於P3。[M:current-state table]

因此目前最合乎證據的mechanistic synthesis是聯合瓶頸：

1. **Representation/candidate support**：災難井可能根本沒有不劣於P3的候選。
2. **Ranking/calibration**：有oracle headroom不代表inference-safe selector能利用。
3. **Catastrophic-tail safety**：existing risk signals跨fold不可偵測或不可校準。
4. **Domain composition**：main與anti-twin/spatial stress可得相反結論。

下一個protocol必須引入新的inference-safe information，而且在fresh population中同時處理這四層。只換模型容量、loss、temperature、fallback、threshold或在已曝光cohort做post-hoc disagreement gate，都不是新證據。[M:`claim:joint-current-bottleneck`; M:`hypothesis:existing-representation-tail-safe-v2`]

Audit V2最優先的研究問題不是「哪個foundation model最大」，而是：current well自身的known prefix能否提供可重演的sensor/domain calibration與pre-output support certificate。這由`AUD2-H01`與`AUD2-H02`先在target-free artificial cuts上否證；只有兩者通過，才考慮TSPulse representation或horizontal empirical episode retrieval。[H:`AUD2-H01`–`AUD2-H04`]

## 2. Audit scope、方法與限制

### 2.1 實際執行範圍

本次完成：

- 驗證外部compiled research memory，並以planning/status/constraints/cohorts/sources queries重建current state；所有query均顯示`Worktree reads: 0`。
- 盤點git、dirty worktree、Python packages、CPU/RAM/GPU/VRAM、disk與offline dependencies。
- 完整閱讀本機官方competition snapshot與14-slide competition-data PPTX；瀏覽2026-07-22 live Kaggle官方頁與相關官方notebook/dependency文件。
- 建立protected-label-safe profiler，重新讀取全部train/test inference-safe CSV views；只在非protected693 wells讀target作限定scope profile。
- 靜態掃描233個Python files與127個`read_csv` call sites，建立code/leakage review queue。
- 以compiled memory統整14個formal experiments、12個claims與hypotheses、11個cohorts（v4後）。
- 以primary sources審查28項third-party assets與36項外部source records。
- 預註冊並執行一個TSPulse target-free bounded feasibility pilot；完成immutable archive/ingest/rebuild/deep validation。
- 在V2主要論點獨立形成後，才讀external archive中的Audit V1並建立30-row change matrix。

### 2.2 明確未執行

- 未讀protected `cohort:change-point-confirmatory-80`的TVT或任何衍生label統計。
- 未啟動label-bearing TSPulse discovery、full OOF、full stress suite、confirmatory或submission。
- 未下載FORCE/Volve等外部dataset；未fine-tune或訓練任何第三方模型。
- 未把可見test三井作泛化驗證或調參population。
- 未讀工作樹中的舊audit、歷史reports或artifacts以重建current state；V1比較使用external memory archive。
- 未commit、push或建立PR。

### 2.3 重要限制

- 真正hidden test distribution不可由可見三井估計。[O:`OS-LIVE-007`]
- V2 full-target profile故意只覆蓋非protected693 wells；其absolute RMSE不屬於full773 comparability group。[D:`data_profile.json`]
- Third-party landscape受公開文件、model/dataset cards與2026-07-22可取得性限制；「無可用domain FM」是asset audit結論，不是未來永遠不會有。
- TSPulse pilot的retrieval task被兩個raw comparators飽和；它只能回答operational feasibility，不能回答TVT efficacy。[M:`experiment:audit-v2-tspulse-target-free-v1`]

## 3. Source hierarchy與evidence taxonomy

### 3.1 優先序

1. **官方live/snapshot/PPTX**：競賽規格、規則、schema、runtime、license contract。
2. **官方competition data observation**：實際CSV/PPTX/sample的可重現事實。
3. **External compiled research memory**：current experiments、claims、cohort exposure與STOP rules。
4. **Memory archive provenance**：只在精確追溯或V1比較時使用。
5. **Primary external sources**：論文、官方repo、model/dataset card、license。
6. **Inference/hypothesis/recommendation**：明確標記，不能冒充已驗證效果。

### 3.2 標記

| Tag | 類型 | 可支持的主張 |
|---|---|---|
| `O` | Official fact | 規則、評分、deadline、schema、runtime |
| `D` | Data observation | counts、missingness、hash、duplicate、geometry |
| `M` | Memory/formal experiment | RMSE、CI、decision、STOP、cohort exposure |
| `E` | External primary evidence | asset capability、license、training provenance |
| `I` | Inference | 多份證據支持但尚未直接實驗的機制推論 |
| `H` | Hypothesis | 有明確falsification與gate的未驗證提案 |
| `R` | Recommendation | 在constraints與成本下的操作決策 |

一個重要主張只有在相同scope與comparability group下才可比較。Diagnostic oracle不等於deployable evidence；不同cohort的absolute RMSE不表示研究進展。[M:`constraint:no-cross-cohort-absolute-comparison`]

## 4. 官方競賽規格重新審計

完整逐條ledger見 [official_source_ledger.csv](official_source_ledger.csv)。

### 4.1 任務與metric

每個horizontal well提供`MD,X,Y,Z,GR,TVT_input`；known prefix後的`TVT_input`缺失，需要預測所要求rows的`TVT`。每井另有`TVT,GR` typewell。[O:`OS-LIVE-006`–`OS-LIVE-009`; O:`OS-DATA-PPTX-001`]

Evaluation頁的明確文字與公式是root mean squared error over test rows。[O:`OS-LIVE-004`] Live competition metadata的algorithm name顯示MSE，兩者衝突；因Evaluation公式、歷史計分與sample contract一致，本audit以RMSE公式為控制來源，並保留矛盾而不默默合併。

### 4.2 Code competition contract

- submission必須由Kaggle Notebook產生，輸出檔名`submission.csv`。
- CPU與GPU notebook各不超過540分鐘。
- rerun時internet disabled。
- external weights/data/packages須預先作Kaggle Input/Model/Dataset或Dependency Manager/offline wheel可取得，不能在runtime下載。
- generic Kaggle notebook 12h上限不適用；competition-specific 9h override。[O:official ledger Code Requirements rows]

### 4.3 Teams、submissions與timeline

| Contract | Live official value |
|---|---:|
| Maximum team size | 5 |
| Daily submissions | 5 |
| Final selections | 2 |
| Entry deadline | 2026-07-29 23:59 UTC |
| Team merger deadline | 2026-07-29 23:59 UTC |
| Final deadline | 2026-08-05 23:59 UTC |

Live metadata另有`kernelsPublishingDisabledDeadline=2026-07-23T23:59:00Z`，但目前官方頁沒有足夠正文解釋其操作意義。本報告把它列為unknown，不能替代entry/final deadlines。[O:official ledger conflict row]

### 4.4 External data、models與code

Rules允許公開、同等可取得，或符合Reasonableness/minimal-cost標準的external data與pretrained models。Competition-specific code requirement還要求合格的open-source與commercial-use條件；因此「GitHub可見」不等於可用。[O:official ledger external policy rows]

本audit採保守asset gate：

1. 公開且同等可取得；gated/contact-only需額外reasonableness審查。
2. License明確涵蓋data、weights與code；不假設repo license覆蓋weights/data。
3. Training-data provenance足以排除competition contamination或public-prediction lineage。
4. 可pin version/commit/SHA-256並在無網路下重跑。
5. 能在9h與合理RAM/VRAM內部署。
6. 輸入schema在hidden test可取得。

Research-only、NC、無license、training provenance不明、無法公平取得或offline rerun者，標成reject/quarantine。

### 4.5 Competition-data license

Competition Data限競賽使用，不可向未接受rules者重散布。[O:official ledger data-license row] 因此本audit artifacts只保存aggregate、safe-view fingerprints、小型manifests、plots與label-access logs；不複製raw CSV、PNG、大型OOF或competition-derivedweights。

## 5. Submission與hidden-test contract

### 5.1 Authoring test不是holdout

官方Data頁說可見test是少量train examples，rerun會替換成約200口實際hidden wells。[O:`OS-LIVE-007`] V2逐值驗證三口test horizontal/typewell safe views等於同ID train刪欄版本；它們只可測schema、runtime與submission join。[D:`visible_test`]

### 5.2 動態enumeration

Inference必須：

1. 動態配對`*__horizontal_well.csv`與`*__typewell.csv`。
2. 對每井驗證`TVT_input`是non-empty known prefix後接non-empty NaN suffix。
3. 僅對`TVT_input.isna()` rows產生prediction。
4. Parse sample IDs為`well_hash,row_index`，以sample作authoritative key join並保留原順序。
5. 拒絕missing/duplicate/extra IDs、NaN或inf predictions。

不得硬編3 wells、14,151 rows、可見filenames或glob order。[O:`constraint:dynamic-hidden-test`; D:`sample_submission`]

### 5.3 Schema boundary

| Role | Hidden-safe columns | 禁止作feature |
|---|---|---|
| Horizontal | `MD,X,Y,Z,GR,TVT_input` | `TVT`, six formation surfaces, PNG-derived metadata |
| Typewell | `TVT,GR` | `Geology` |
| Output | `id,tvt` | any diagnostic/oracle field |

Train-only surfaces或Geology最多作fold-safe auxiliary supervision；validation推論不能接觸其真值。PNG只可human error analysis。

## 6. Competition data第二次剖析

完整機器可讀結果見 [data_profile.json](../../artifacts/audit_v2/20260722T202906+0800/data_profile.json)，摘要見 [data_profile.md](../../artifacts/audit_v2/20260722T202906+0800/data_profile.md)。

### 6.1 Filesystem、schema與fingerprint

| Asset | Count / rows |
|---|---:|
| Train horizontal CSV | 773 |
| Train typewell CSV | 773 |
| Train PNG | 773 |
| Test horizontal/typewell CSV | 3 / 3 |
| Total dataset files | 2,327 |
| Dataset bytes | 1,325,112,684 |
| Train horizontal rows | 5,092,255 |
| Known-prefix rows | 1,308,266 |
| Prediction rows | 3,783,989 |
| Visible test rows / prediction rows | 19,221 / 14,151 |

Train horizontal schema是`MD,X,Y,Z,ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA,TVT,GR,TVT_input`；test只保留六個hidden-safe columns。Train typewell是`TVT,GR,Geology`；test是`TVT,GR`。所有safe dtypes為`float64`。[D:`schema`]

Safe-scope dataset fingerprint是`49a9fe524467517776ed764c6e25432ae3eac0d3928b67949f36be810ab23565`。Scope只含inference-safe CSV views、path/size/header；PNG contents、protected target與train-only columns故意排除。

### 6.2 Horizontal wells

- 全773 wells的MD嚴格遞增，median step 1 ft；沒有nonpositive step、zero-length segment或safe duplicate rows。
- Prediction fraction median約0.740；prediction horizon median4,840 ft、max10,052 ft。
- GR missingness：overall `0.2961304962`、prefix `0.2347649484`、prediction `0.3173468528`。
- 最大prediction-zone missing run 263 rows；missingness與horizon必須分層報告，不能只用global median imputation。
- 所有prefix mask連續；在非protected693 scope，prefix `TVT_input`逐值等於`TVT`。Protected80不作此target equality read。
- 20對well starts距離<100 ft，7對exact start；exact shared typewell/start/near-start合併後anti-twin components有736個，26個多井component，最大10。

### 6.3 Typewells

Typewell GR無missing，TVT嚴格遞增。Median step categories：

| Median step (ft) | Wells |
|---:|---:|
| 0.10 | 8 |
| 0.25 | 91 |
| 0.50 | 653 |
| 1.00 | 21 |

381 wells的rounded step不只一個，但378只呈現約0.01 ft coordinate quantization；material gaps僅三井：`059c8f24`的max step 2 ft，`14ab73fb`與`eba6605e`的max step 9 ft。[D:`typewells.material_spacing_gap_wells`]

因此所有alignment仍必須在physical TVT coordinate重採樣；同時不能把381誤寫成381口都有重大不規則sampling。

Exact shared typewell有13 groups、34 wells。Heuristic near-duplicate scan得到大量pairs，但它使用fixed standardized resampling threshold，只能作candidate audit queue，不能證明latent template或生成機制。Target-free KMeans最佳k=12、silhouette約0.381也只屬exploratory。

### 6.4 Target與prediction-zone geometry

因protected labels不可讀，V2 target profile明確限定於非protected693 wells：

| Metric | Value |
|---|---:|
| Prediction rows | 3,394,640 |
| Flat pooled RMSE | 15.593406620934621 |
| Worst-decile SSE share | 0.506818 |
| Wells with `abs(diff(TVT))>5` | 42 |
| Prediction-zone jumps >5 | 27 |

Full773 verified memory仍是47 jump wells與30 prediction-zone jumps；V2不為了重算而讀protected labels。兩個scope都顯示hard smoothness/monotonicity不安全，但其absolute metrics不可直接互比。[D:`target_profile_unprotected`; M:full baseline facts]

所有773 last-known anchors落在paired typewell TVT range內；非protected693的prediction target亦完全在typewell range內。這是重要反例：range coverage並不等於candidate path support、ranking正確或tail safety。

### 6.5 Duplicate、latent group與edge cases

Safe-view exact duplicate groups共19：13組train exact shared typewell，加上6組visible train/test copy relationships。沒有train horizontal safe duplicate wells或duplicate rows。

`86454a6f`在非protected合法scope重現：flat RMSE `70.2633851373`、overall GR missing `80.098901099%`、prediction missing `81.17780010%`；它沒有>5 ft target jump，因此tail不能只用jump detector解釋。[D:`regression_case_86454a6f`]

## 7. Data-quality與leakage risk register

完整51項register見 [risk_register.csv](risk_register.csv)。最高風險包括：

| Risk | Evidence | Control |
|---|---|---|
| Protected label被一般test suite讀取 | `tests/test_contracts.py`與`tests/test_baseline.py`會走all-773 target reader | 未建立central capability guard前禁止full pytest；只跑audited safe selection |
| Raw label loaders依caller自律 | residual/spatial/change-point/sequence/dense routes各有直接TVT readers | 建`LabelAccessCapability`與fold context；target只進metric/training capability |
| Comparator file overread | dense/change-point scripts先讀含`y_true`整檔再drop | 按column allowlist讀取；artifact schema把prediction與label分檔 |
| Target-derived cohort reuse | spatial pilot selection使用jump signal | Fresh target-free cohort selector；任何label-selected population明示exposed |
| Validation preprocessing leakage | scaler/imputer/index/pretraining enforcement分散 | 中央`FoldFitContext`、fit provenance與automated negative tests |
| Static authoring constants | code audit找到9個visible constants，多數在tests/docs | production CI拒絕3/14151/visible IDs出現在inference path |
| External contamination/license | weights或dataset provenance可能不完整 | asset register、commit/hash/license/training-data manifest與duplicate guard |
| Pilot evidence overclaim | TSPulse與raw皆48/48 top1 | Claim boundary寫入result、memory與report；需separate incremental protocol |

Code static audit是conservative review queue，不等於每個match都是bug。Production submission core目前動態join正確；主要缺口是enforcement而非每個route都已證實洩漏。[D:`code_risk_audit.json`]

## 8. Validation與metric設計

### 8.1 Primary validation

- 固定5-fold well-level OOF；同一well絕不跨train/validation。
- Fold mapping以spatial location、azimuth、prediction fraction、GR missing、typewell step/range、well length等test-observable features分層並version-lock。
- Outer validation input降為exact hidden schema。
- Scaler、imputer、pretraining、neighbor/retrieval index、target statistics、candidate generator/selector、risk threshold均只fit outer-training wells。
- Early stopping只用outer-training中的inner well split；outer validation只評分一次。
- 每個原始prediction row恰有一個OOF prediction；coverage必須100%。

### 8.2 Primary metric

主分數：

```python
rmse = np.sqrt(np.mean((y_oof - pred_oof) ** 2))
```

不得平均per-well或per-fold RMSE。診斷另報per-well p50/p90/p95/max、worst-decile SSE share、distance-from-PS error curve與預註冊strata。

### 8.3 Mandatory stress

1. Spatial-block holdout。
2. Anti-twin grouping：exact shared typewell、same start或start distance<100 ft綁組。
3. GR missing、missing-run、horizon、typewell step/material gap、jump/no-jump與hard-well strata。
4. Artificial earlier-PS replay作leakage/causality audit，不冒充hidden-test absolute score。

### 8.4 Promotion policy

- Stochastic方法至少5 seeds。
- 以well為cluster做10,000次paired bootstrap；禁止row bootstrap。
- 低成本改動：改善至少0.10 ft且CI上界<0。
- 高成本方法：改善至少0.20 ft，或有事先定義且明顯的tail/stress收益。
- 同時通過pooled point、p95、worst-decile SSE、catastrophic count與stress AND gates。
- Public-LB小幅改善、單seed、oracle gap或無OOF改善都不足以promotion。

## 9. Verified baselines與current reference state

完整current-state table見 [current_state_table.csv](current_state_table.csv)。

| Reference | Main RMSE | Same-group paired evidence | Stress | Decision |
|---|---:|---|---:|---|
| Flat P1 | 15.9098528707 | deterministic reference | 15.9098528707 reference | retain |
| P2 residual GBDT | 15.5970165777 | vs flat -0.3128362930 | 16.2004819005 | not promoted |
| P3 alignment | 15.4525482195 | vs P2 -0.1444683582; CI crosses 0 | 16.1361032223 | freeze comparator |

P2與P3證明hidden-safe geometry/missingness/alignment features含partial signal；它們同時證明main point estimate不足以取代stress與uncertainty evidence。

## 10. 既有研究路線的證據統整

| Route | Scope / comparability | Critical evidence | Decision |
|---|---|---|---|
| Global beam v2/v3 | balanced priority4 60 | beam oracle 42.56/60.63；best 2/60與0/60 | STOP |
| Broad spatial | disjoint743 | blend-flat `+1.0394`; CI `[0.256,1.837]` | STOP |
| Change-point objective | fresh60 | objective top1-flat `+4.8765`; catastrophic20 | STOP |
| Summary ranker | same change60 | headroom utilization1.2336%; CI crosses0; worse thanP3 | STOP |
| Sequence cost-volume | fresh60; 5fold×5seed | paired improvement0.1945 but CI crosses0; p95 +5.331; catastrophic7 | STOP v1 |
| Sequence tail diagnosis | immutable same60 | only3/7 catastrophic supported; risk AUROC0.438–0.475 | STOP hypothesis |
| Dense local field | fresh60; 5fold×5seed | model-P3 `+1.4072`; catastrophic12; out-of-grid0 | STOP |
| TSPulse target-free | fresh safe24; labels unread | operational gate9/9; raw comparators also1.0 top1 | feasible only |

Scope非常重要：spatial30的pilot signal不能抵銷disjoint743 failure；sequence/dense/change60各屬不同discovery populations，absolute RMSE不能互比成method ranking。

## 11. Failure-mode synthesis

### 11.1 Representation/candidate support

Global beam與mode-diverse pruning無法把correct modes放進candidate set。Sequence failure diagnosis中4/7 catastrophic wells連complete candidates都沒有P3-or-better path。Dense modelout-of-grid=0卻仍失敗，顯示「range覆蓋」與「有用representation support」不同。[M]

### 11.2 Ranking

Change-point complete oracle明顯好於deployable top1；summary ranker只利用1.23% headroom。Sequence ranking指標改善但不能穩定轉成RMSE。這表示ranking information存在但不充分，且不能在相同候選與已曝光cohort上無限增加scorer容量。[M]

### 11.3 Calibration與domain composition

P2/P3 main point signals在anti-twin/spatial stress退化；spatial pilot與743 complement方向相反。Risk detector跨fold AUROC低於0.5附近，fixed fallback probe不能升格。這些結果要求pre-output、target-free或outer-train-calibrated support certificate，而不是事後candidate–P3 disagreement。[M]

### 11.4 Catastrophic-tail safety

Full flat worst10% wells佔52.4763% SSE；非protected693仍是50.6818%。少數catastrophic wells足以翻轉pooled RMSE，而其機制不只missingness或jumps。任何point gain沒有p95/worst-decile/stress evidence都不可promotion。[D/M]

### 11.5 Validation與compute

Priority4 v1曾因population imbalance而invalidated；目前codebase的full pytest又會碰protected labels。實驗正確性需要centralized enforcement與label-access logs，不可只靠研究者記得caller convention。[D]

## 12. 現行聯合瓶頸

目前證據支持以下因果圖，而不是單一capacity bottleneck：

```text
hidden-safe observations
        │
        ▼
representation / candidate support ──失敗──► unsupported catastrophe
        │
        ▼
ranking / path selection ───────────失敗──► oracle gap
        │
        ▼
calibration / OOD detectability ────失敗──► unsafe correction or fallback
        │
        ▼
tail/stress robustness ─────────────失敗──► private-LB composition risk
```

新的方法只有在說清楚「增加哪個test-time可得資訊或inductive bias」時值得進場。TSPulse提供外部generic sequence representation，但已完成pilot只證明它能穩定產生embedding；raw comparators同樣完美retrieval，因此它尚未證明新增support。Horizontal empirical episode retrieval則可能新增outer-train真實lateral continuation library，但仍要防duplicate/domain composition與ranking failure。[I/H]

## 13. 第三方資料／模型／論文 landscape

完整primary-source ledger見 [external_research_ledger.csv](external_research_ledger.csv)，asset-level audit見 [third_party_asset_register.csv](third_party_asset_register.csv)。技術主張優先採官方repository、paper、model/dataset card與license，沒有用SEO摘要作核心證據。

### 13.1 Public well-log data

**FORCE 2020**提供118口Norwegian Sea wells、GR與其他logs，約170.1 MB；Zenodo標CC-BY-4.0，description同時引用NOLD/NLOD 2.0，使用時應保留兩組attribution。它可能增加真正domain-specific GR morphology，但vertical/multicurve distribution與ROGII horizontal GR不同；進場前必須只用safe GR fingerprint做exact/near-duplicate與domain audit。[E:`EA-001`]

**Volve**約5 TB、40,000 files，遠超單一下載10 GB回報門檻，access與license對prize-competition reuse仍需澄清；只允許未來在使用者批准後做selected-file manifest，不作現在的bulk corpus。[E:`EA-002`]

EGS Collab 2只有11個geothermal boreholes，適合parser/domain smoke，不足以合理支持pretraining。SEG 2016 dataset的repo與data license分離，直接data/code reuse拒絕，只保留validation lessons。[E:`EA-003`,`EA-004`]

### 13.2 Well-log與geophysical foundation models

WLFM論文描述1,200 wells與stratigraphy-aware masked learning，但未找到可審計weights、data identities、repo與asset license；TGS model使用私有1.1M logs且無公開weights。它們都是architecture evidence，不是可用asset。[E:`EA-006`,`EA-007`]

Seismic FMs（ThinkOnward GFM、NCS 3D、Seismic Foundation Model）即使Apache/MIT，也需要hidden test不存在的seismic cubes/images。Schema mismatch不能用「geophysical」這個領域標籤補足，因此全部reject。[E:`EA-026`–`EA-028`]

### 13.3 Generic time-series representation

**IBM TSPulse r1**是外部模型首選：Apache-2.0、model card明列public training corpora、約1.07M loaded parameters、CPU可行、512-point semantic search。[E:`EA-009`] 它只能作frozen GR embedding/support/OOD probe，不能直接forecast TVT。

**TS2Vec**提供hierarchical contrastive representation，若用competition data必須per outer fold fit，禁止validation/test transductive SSL。[E:`EA-011`] **MOMENT-small**是較重的generic comparator；只在TSPulse signal不明時作一次frozen probe。[E:`EA-010`]

Chronos-Bolt、Tiny Time Mixer與TimesFM最多作GR forecast surprise/predictability diagnostic。直接forecast `TVT_input` suffix會重引入無機制支持的generic slope extrapolation；Moirai R weights的CC-BY-NC增加競賽與redistribution風險。[E:`EA-012`–`EA-015`]

### 13.4 Matching、uncertainty與tail methods

STUMPY可提供deterministic motif/discord support score；FAISS只是retrieval backend，不新增information。Soft-DTW只有warp inductive bias，若單獨重跑raw alignment是STOP route重包裝；只有在新representation先過gate後，才能作同representation內部ablation。[E:`EA-016`–`EA-018`]

MAPIE/conformal risk control必須以well/group為calibration unit，不能row-wise套用exchangeability；PyOD只允許少數preregistered baselines，避免detector/threshold hunting。Group DRO/TERM改loss但不創造support；TERM standalone reject，Group DRO只在新representation已過gate後再談。[E:`EA-019`–`EA-022`]

### 13.5 Synthetic/geostatistical methods

GSTools最先用作covariance、contiguous missingness、sampling與trajectory shift stress generator，不用作broad spatial predictor。[E:`EA-023`] GemPy/Synthoseis可以產生fault/fold concepts，但沒有可信GR forward model時，synthetic TVT pretraining容易引入sim-to-real錯誤；先做static fidelity audit，現在暫緩。[E:`EA-024`,`EA-025`]

## 14. Third-party asset acceptance/rejection table

| Asset / class | Decision | 合格用途 | 拒絕／限制原因 |
|---|---|---|---|
| TSPulse r1 | ACCEPT Tier1 | Frozen GR representation feasibility/support probe | Generic non-geological corpus；incremental value未證明 |
| FORCE 2020 | ACCEPT Tier2 conditional | External-only geological GR SSL | attribution、duplicate/domain audit、training>3h須先回報 |
| TS2Vec | ACCEPT Tier1/2 | Outer-fold-only SSL comparator | old stack compatibility；不能transductive fit validation |
| STUMPY | ACCEPT Tier1 | Bounded motif/discord comparator | full all-pairs runtime不可行；不是新candidate source本身 |
| Chronos-Bolt | ACCEPT diagnostic only | GR surprise/risk ablation | 禁止direct TVT forecast |
| MAPIE/PyOD/FAISS | ACCEPT substrate | group calibration/OOD/retrieval | 不新增support；threshold proliferation risk |
| GSTools | ACCEPT stress | Target-free correlated stress | 不作broad spatial deployable prior |
| MOMENT-small | DEFER comparator | One frozen probe | 較TSPulse重、domain specificity弱 |
| Volve | QUARANTINE | Selected logs only after review | 約5 TB、access/license ambiguity、>10GB |
| WLFM/TGS/TimeGPT | REJECT/QUARANTINE | Literature only | weights/provenance/license/offline access不足 |
| Moirai R | REJECT | None | NC weights與競賽/redistribution risk |
| Seismic FMs | REJECT | None | hidden schema沒有seismic input |
| Standalone soft-DTW/TERM | REJECT route | Conditional internal ablation only | 沒有新information；重包STOP route |
| SEG/FORCE solution repos | Literature/quarantine | Methods/QC lessons | target/schema/license/weights lineage不相容 |

Acceptance只表示asset合規且值得指定用途，不表示model promotion。每個asset仍須version、commit、hash、training provenance、license、offline package與fold-safe usage manifest。

## 15. 新 hypothesis portfolio

完整可機讀protocol見 [hypothesis_register.json](hypothesis_register.json)。Ranking依evidence、expected information gain、private-LB robustness、tail safety、reproducibility與cost，不依novelty或model size。

| Rank | ID | Tier | Core new information / bias | Current decision |
|---:|---|---|---|---|
| 1 | AUD2-H01 | Tier0 | Current-well known-prefix calibration via artificial-cut replay | mandatory pre-label gate |
| 2 | AUD2-H02 | Tier0 | Pre-output support/OOD certificate from replay/coverage/margin | mandatory pre-label gate |
| 3 | AUD2-H03 | Tier1 | Prefix-conditioned TSPulse physical-TVT representation | operational pass only; label work blocked |
| 4 | AUD2-H04 | Tier1 | Outer-train horizontal artificial-cut episode library | conditional secondary route |
| 5 | AUD2-H05 | Tier1 | Prefix-calibrated STUMPY motif/discord | deterministic comparator only |
| 6 | AUD2-H06 | Tier0 | Anti-twin typewell fingerprint coverage | coverage audit only |
| 7 | AUD2-H07 | Tier1 | External generic GR surprise as risk signal | risk-only conditional |
| 8 | AUD2-H08 | Tier2 | FORCE geological-GR external pretraining | deferred, costly |
| 9 | AUD2-H09 | Tier2 | TS2Vec multiscale contrastive GR representation | deferred comparator |
| 10 | AUD2-H10 | Tier2 | Well-block conformal risk control | conditional on new support |
| 11 | AUD2-H11 | Tier0 | Correlated missingness/resampling stress harness | validation infrastructure |
| 12 | AUD2-H12 | Tier3 | MOMENT frozen representation probe | defer |
| 13 | AUD2-H13 | Tier3 | Synthetic geological trajectory pretraining | high-risk defer |
| 14 | AUD2-H14 | Reject | Standalone raw soft-DTW | no new information |
| 15 | AUD2-H15 | Reject | Post-hoc P3/candidate disagreement gate | prohibited reopening |

### 15.1 AUD2-H01 — prefix replay

在known prefix的50/70/85%建立artificial cut；cut後`TVT_input`遮掉，只以cut前GR/TVT fit robust amplitude/shift/dilation，評估被遮段。這新增current-well sensor/domain calibration，不用prediction labels。

AND gate：相對raw matcher pseudo RMSE至少改善5%、median coordinate error至少改善10%、4/5 folds non-worse、p95不惡化>0.5 ft、高missing stratum不惡化>0.20 ft。任一失敗即STOP；不得為了通過改cuts或thresholds。[H:`AUD2-H01`]

### 15.2 AUD2-H02 — support certificate

以H01 replay residual、跨cut stability、alignment entropy/margin、nearest-motif distance、missing-run coverage、horizon ratio形成pre-output risk score。先在target-free replay驗證；只有score/features/threshold完全freeze後，才可能向使用者提議fresh label discovery。

Target-free與future fresh discovery都需Spearman≥0.30、catastrophic AUROC≥0.70、top-risk20%捕獲≥50% excess SSE、至少4 folds monotonic；fixed fallback還要改善≥0.10 ft且tail不壞。[H:`AUD2-H02`]

### 15.3 AUD2-H03 — TSPulse conditional route

本audit已完成的feasibility只通過offline/determinism/non-collapse/stability/runtime。下一個protocol必須在相同decoder中比較TSPulse與locked raw representation，並先由H01/H02提供prefix calibration與support certificate。Fresh population必須排除TSPulse feasibility24、protected80與所有舊route cohorts。

Target-free top-3 recall需相對raw提高至少10 percentage points；candidate support需覆蓋≥80% wells與≥70% predeclared hard wells。任何label-bearing high-cost result仍須至少-0.20 ft、CI<0與全部tail/stress gate。[H:`AUD2-H03`]

### 15.4 AUD2-H04 — horizontal empirical episodes

從outer-training wells的artificial cuts建立真實lateral continuation library；以current prefix GR、trajectory與typewell fingerprint檢索，轉移anchor-relative residual trajectory。它不同於broad spatial kNN：不以absolute XY作主要retrieval，而且candidate support來自outer-train真實future episodes。

Nested replay中similar episode future error需比horizon-matched random低≥10%，candidate support整體≥80%、hard-well≥70%；否則STOP，不以oracle gap接新ranker。[H:`AUD2-H04`]

## 16. 推薦的模型與系統架構方向

推薦架構是可逐層否證的anchored support system：

```text
safe horizontal/typewell inputs
        │
        ├─ current-well prefix replay calibration (H01)
        │
        ├─ raw / TSPulse / TS2Vec representation (paired ablation)
        │
        ├─ physical-TVT candidate field or empirical episodes
        │
        ├─ outer-train-only ranking and calibration
        │
        └─ pre-output support certificate (H02)
                     │
              anchored bounded correction
                     │
              locked flat/P3 comparator
```

核心設計：

- 以last-known`TVT_input`為anchor，預測bounded correction或candidate distribution，不直接外推prefix slope。
- Typewell所有operations在physical TVT grid，顯式保留sampling/missing mask。
- Current-well adapter只使用known prefix；prediction-zone GR可作hidden-safe representation，但不可用其TVT或label-derived statistics。
- Candidate generator先以support metrics否證；selector只在support成立時訓練。
- Support certificate在model output前可計算，不能用candidate–P3 realized error或validation outcome作feature。
- Flat與P3是paired comparators，不是可事後挑選的oracle fallback。
- 所有global transform/index/calibrator皆outer-fold fit；pure per-well deterministic transform另行記錄。

## 17. Development architecture與reproducibility plan

### 17.1 需要補強的中央contracts

1. `LabelAccessCapability`：protected/validation target只有metric與明示training capability可讀。
2. `FoldFitContext`：每個scaler、imputer、encoder adaptation、index、selector與calibrator記錄fit well IDs與fold。
3. Artifact schemas：prediction與`y_true`分檔；consumer以`usecols`讀取，避免comparator overread。
4. Cohort registry：target-free、label-exposed與protected狀態及hash進manifest。
5. Production inference CI：拒絕visible IDs、3 wells、14,151 rows或fixed filenames出現在runtime path。

### 17.2 每次formal run必備

- Git revision、dirty status與user-owned changes。
- Fold version、cohort hash、comparability group、seed與hardware。
- Exact feature list、fit scope、asset commit/hash/license/provenance。
- Pre-run manifest、label-access log、commands與exit codes。
- 100% OOF coverage、pooled metric、per-well/tail/stress與runtime。
- AND gate逐項pass/fail及停止決策。
- Small immutable summary先archive，再ingest/rebuild/deep-validate memory。

### 17.3 Audit V2 code交付

- `src/rogii_baseline/audit_v2/protection.py`：protected fail-closed safe readers。
- `src/rogii_baseline/audit_v2/profiling.py`：全資料safe profile與非protected target scope。
- `scripts/audit_v2/profile_competition_data.py`：reproducible profiler。
- `scripts/audit_v2/audit_code_paths.py`：conservative static review queue。
- `scripts/audit_v2/select_target_free_cohort.py`：stable target-free cohort selector。
- `scripts/audit_v2/run_tspulse_target_free_pilot.py`：offline pinned pilot。
- `tests/audit_v2/`：schema/mask/sample/prefix/protected guard/code audit tests。

## 18. Research gates、stop conditions與promotion policy

每個new protocol都需以下AND gates：

1. **Rule gate**：license、公平取得、offline rerun與schema compatibility全部合格。
2. **Information gate**：說明並target-free測試新增information／inductive bias；matched cheap comparator不能飽和或支配。
3. **Support gate**：overall與predeclared hard/tail strata都有candidate support。
4. **Ranking gate**：paired top1/point metric在fresh cohort改善，不能只靠oracle。
5. **Calibration gate**：risk/uncertainty跨fold monotonic且在output前可得。
6. **Tail gate**：p95、worst-decile SSE、catastrophic count不惡化。
7. **Stress gate**：spatial-block、anti-twin、missing/horizon/typewell strata通過。
8. **Statistical gate**：5 seeds與well-cluster bootstrap CI。
9. **Compute gate**：full hidden-size inference含margin小於9h，可完全offline。

任一AND gate失敗即保存negative result並STOP該版本。禁止在同cohort改loss、temperature、threshold、fallback、features或gate後重稱新證據。通過discovery也只進user review；protected/full/stress/submission不自動授權。

## 19. Compute/runtime/offline deployment plan

### 19.1 本機inventory

| Resource | Observed |
|---|---|
| CPU | Intel Core Ultra 5 125H class；14 cores / 18 logical CPUs |
| RAM | 34,037,383,168 bytes（約31.7 GiB） |
| GPU | NVIDIA RTX 3050 Laptop GPU，6 GiB VRAM |
| Free disk at run start | 約210 GB |
| Main Python | 3.12.10 |

Main environment沒有torch；TSPulse使用隔離venv `C:\Users\wurre\.codex\venvs\rogii-audit-v2-tspulse-py312`，約1.13 GB，內含`granite-tsfm 0.3.6`、`torch 2.10.0`、`transformers 4.57.6`、`scikit-learn 1.7.2`與`pandas 3.0.3`。

Pinned model weight 4,305,624 bytes，SHA-256 `b9332ae796ec7c313f991ed32dbec62c29a8e673281decb7308f955bdda7aae0`；config SHA-256 `4a09cdf00f3146a932954cf5a47ff0e767224de1dce8ae456e73d56bbec81621`。Pilot CPU p95為24.35 ms/window、peak RSS 0.418 GiB，遠低於feasibility gate；full sliding-window architecture仍需另外benchmark，不能由48-window結果線性保證9h deployment。

### 19.2 執行分級

- H01/H02/H06/H11：CPU約0.5–2h，無外部大下載，可在現授權內執行。
- TSPulse harder target-free/limited outer-fold probe：先估0.5–1.5 GPU-h或CPU bounded run；維持exact revision與offline wheelhouse。
- FORCE+TS2Vec或5-seed representation discovery：預估3–8 GPU-h；執行前需向使用者回報runtime、disk/RAM/VRAM、checkpoint與expected information gain。
- Volve/full synthetic或large FM：>10GB或>3h，保持defer，沒有使用者批准不執行。

### 19.3 Kaggle packaging

Final candidate若存在，須把pinned wheels、licenses、model/config hashes與weights作private/public Kaggle Input並測試`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。Production notebook不得依賴local symlink cache、runtime pip internet或mutable `main` branch。

## 20. 立即、兩週與中期 roadmap

詳細roadmap見 [recommended_roadmap.md](recommended_roadmap.md)。摘要：

### 立即（0–48h）

1. 建central label guard/FoldFitContext與safe test orchestration。
2. Lock並執行H01 artificial-cut prefix replay。
3. 在看H01 outcome前freeze H02 feature vector/threshold；執行target-free certificate。
4. 執行H06 anti-twin typewell coverage與H11 stress fidelity audit。
5. H01或H02任一失敗即不啟動TSPulse label protocol。

### 兩週

- 若H01/H02通過，先設計更難的target-free TSPulse-vs-raw comparator；raw不能再100%飽和。
- 只在representation gate提供incremental signal後，向使用者提議fresh label discovery population與budget。
- TSPulse失敗則轉H04 empirical episodes；H05只作deterministic comparator。
- 不在sequence/change/dense/spatial已曝光cohorts重調。

### 中期

- Internal-only representation有一致訊號後才審核FORCE下載與external pretraining。
- 新point/support route成功後才研究well-block conformal calibration。
- Synthetic geology、MOMENT/large generic FM與full geostatistical pretraining保持Tier3。

## 21. Audit V1 → V2 change matrix

完整30-row matrix見 [v1_v2_change_matrix.csv](v1_v2_change_matrix.csv)。

V1在當時workspace沒有model implementation與formal experiments的條件下，正確建立了data contract、flat baseline、validation、tail與inference-schema邊界。V2不否定這些地基。

根本變更是：

- V1的P2/P3/beam/spatial/sequence priorities現在已有formal evidence，不再是未測backlog。
- P2/P3只有partial signal；beam/spatial/change/ranker/sequence-tail/dense routes有明確STOP或freeze。
- Oracle由「可能的selector headroom」降級為diagnostic，必須先證明support與risk detectability。
- 固定48h model sequence改為fresh-population hypothesis portfolio與AND gates。
- External research由一般「可考慮public pretraining」提升成asset-level acceptance/rejection與實際bounded pilot。
- 「validation是最高風險」擴充成validation enforcement加joint representation/ranking/calibration/tail bottleneck。

## 22. Unknowns、disagreements與需使用者決策事項

### 22.1 Unknowns / disagreements

1. Live metric metadata的MSE label與Evaluation RMSE公式衝突；公式控制，但應持續監控。
2. `kernelsPublishingDisabledDeadline`的實務意義未有足夠官方正文；不能推導成final deadline。
3. 真hidden約200 wells的GR missing、typewell groups、spatial density與tail composition未知。
4. Protected80在change-point protocol中的labels仍未讀；本audit沒有估其target distribution。
5. TSPulse是否相對locked raw representation提供incremental support未知。
6. Prefix artificial replay能否外推到真正prediction zone未知；H01/H02專為否證此假設。
7. FORCE與ROGII GR是否有實質domain overlap或near duplicates未知；下載後才可target-free audit。
8. Material typewell gaps三井的原因未知；保留原始資料，不自行修補。

### 22.2 需要使用者明確決定

- H01/H02通過後，是否允許fresh label-bearing discovery protocol。
- 任一訓練>3h或整批>9h時可接受的GPU/CPU budget與runtime。
- 是否下載/使用FORCE或其他public asset；需先交license/provenance/duplicate/runtime manifest。
- 是否及何時允許protected confirmatory80、full OOF、full stress或submission。
- 接近deadline時兩個final submissions的risk allocation；目前沒有已promotion候選，不需現在決定。

## 23. Final recommendations

1. **保留flat為stable baseline、P3為frozen comparator；兩者角色不可混淆。**
2. **先修enforcement。** Full pytest在protected policy下不安全；建立central label/fold capability guard後再擴研究。
3. **先做H01與H02，且在結果前freeze gates。** 它們直接回答current-well新information與risk detectability。
4. **TSPulse只保留為operationally feasible asset。** Raw comparator飽和表示下一個probe必須更難且paired；不得直接進TVT label experiment。
5. **H04 empirical episodes是TSPulse以外最值得的new-support主線。** 它引入outer-train真實lateral continuation，而不是重調舊candidate scorer。
6. **H06/H11作cheap falsification infrastructure。** Coverage不足或synthetic stress不對應真failure時立即關閉。
7. **FORCE/TS2Vec延後。** 先證明internal-only representation有可轉移signal，再付出下載與multi-seed成本。
8. **尊重STOP與cohort exposure。** 不以新model name包裝相同features/loss/fallback/threshold。
9. **不讀protected80、不跑full OOF/stress、不生成submission，直到fresh discovery gate與使用者授權同時成立。**

## 24. Commands、hashes、source ledger與artifact index

### 24.1 主要commands與實際狀態

| Command | Exit / key output |
|---|---|
| `rogii_memory.py validate` | 0；passes=true；Worktree reads=0；startup digest `820c...aa7` |
| planning/status/constraints/cohorts/sources queries | all 0；Worktree reads=0 |
| `profile_competition_data.py ...` | 0；773 wells；5,092,255 rows；protected target reads=0 |
| `audit_code_paths.py --output ...` | 0；233 Python files；127 read_csv calls；conservative queue |
| TSPulse real-window smoke | 0；`SMOKE_PASS`；2 windows |
| TSPulse full target-free pilot | 0；`PASS`；48 windows；gates9/9 |
| research-memory archive/ingest/rebuild | all 0；immutable v4 activated |
| `rogii_memory.py validate --deep` | 0；passes=true；Worktree reads=0；digest `8cec...fb77` |
| `python -m pytest tests/audit_v2 -q` | exit 0；12 passed in 6.75s |
| `python -m pytest tests/test_metrics.py tests/test_folds.py tests/test_priority2_features.py tests/test_priority2_oof.py tests/test_priority3_alignment.py tests/test_priority3_oof.py tests/test_spatial_leakage.py tests/test_change_point_leakage.py tests/test_submission.py -q` | exit 0；45 passed in 79.88s；audited protected-safe legacy selection |
| `python -m compileall -q scripts/audit_v2 src/rogii_baseline/audit_v2 tests/audit_v2` | exit 0 |

### 24.2 Critical hashes

| Object | SHA-256 / ID |
|---|---|
| Git revision | `f115f574b8a927cb9be133f2e877526d2f84c72d` |
| AGENTS.md | `d6d90920f006a5030eac5a5e766921bcf85018969fbf9b97102ce6e831b3ea7d` |
| Local official HTML | `36753e43c4d6f19fef9aae9bcd08817b62732f0482e204cb37f6da338abe4d94` |
| Official PPTX | `c083c59df01f0fdf1fea860bc977fcdcf278eb12a33282a0200044c8abf950fc` |
| Protected cohort safe identity | `1ec8338acf54eac14c11e0625cbd4fcbbb6c6a97e78c8609e81c950ec1e918c4` |
| Safe-scope dataset fingerprint | `49a9fe524467517776ed764c6e25432ae3eac0d3928b67949f36be810ab23565` |
| TSPulse cohort | `afa3842339f2ac3300695a7463174b1947df0034d703c69df982cb1ed7cad7e3` |
| TSPulse weight | `b9332ae796ec7c313f991ed32dbec62c29a8e673281decb7308f955bdda7aae0` |
| Hypothesis register | `e0be69448fd406f257d302baae390b7f13df7d582096703631a007310251927a` |
| Memory v4 registry | `8cec380adda1cb6b8a7dc10e27323f6d3ec99f5c8ce90bfe40a2f75f2948fb77` |

### 24.3 Artifact index

- [manifest.json](../../artifacts/audit_v2/20260722T202906+0800/manifest.json)
- [data_profile.json](../../artifacts/audit_v2/20260722T202906+0800/data_profile.json)
- [data_profile.md](../../artifacts/audit_v2/20260722T202906+0800/data_profile.md)
- [per_well_profile.csv](../../artifacts/audit_v2/20260722T202906+0800/per_well_profile.csv)
- [typewell_profile.csv](../../artifacts/audit_v2/20260722T202906+0800/typewell_profile.csv)
- [duplicate_safe_view_groups.csv](../../artifacts/audit_v2/20260722T202906+0800/duplicate_safe_view_groups.csv)
- [near_duplicate_typewells.csv](../../artifacts/audit_v2/20260722T202906+0800/near_duplicate_typewells.csv)
- [spatial_start_pairs_lt100ft.csv](../../artifacts/audit_v2/20260722T202906+0800/spatial_start_pairs_lt100ft.csv)
- [code_risk_audit.json](../../artifacts/audit_v2/20260722T202906+0800/code_risk_audit.json)
- [label_access_log.json](../../artifacts/audit_v2/20260722T202906+0800/label_access_log.json)
- [TSPulse preregistration](../../artifacts/audit_v2/20260722T202906+0800/pilots/tspulse_target_free_v1/pre_run_manifest.json)
- [TSPulse full result](../../artifacts/audit_v2/20260722T202906+0800/pilots/tspulse_target_free_v1/result.json)
- [TSPulse formal summary](../../artifacts/audit_v2/20260722T202906+0800/pilots/tspulse_target_free_v1/summary.json)
- [TSPulse label access](../../artifacts/audit_v2/20260722T202906+0800/pilots/tspulse_target_free_v1/label_access.json)
- `per_well_profile.csv`與`typewell_profile.csv`提供可重畫的aggregate plot inputs；本報告以tables為正式artifact，未生成raw-data圖像副本。

### 24.4 Report registers

- [executive_summary.md](executive_summary.md)
- [official_source_ledger.csv](official_source_ledger.csv)
- [external_research_ledger.csv](external_research_ledger.csv)
- [third_party_asset_register.csv](third_party_asset_register.csv)
- [hypothesis_register.json](hypothesis_register.json)
- [current_state_table.csv](current_state_table.csv)
- [v1_v2_change_matrix.csv](v1_v2_change_matrix.csv)
- [risk_register.csv](risk_register.csv)
- [recommended_roadmap.md](recommended_roadmap.md)

本報告的結論邊界、cohort exposure與STOP decisions以external compiled memory v4為current authority；本報告中的新hypotheses仍是hypotheses，除了TSPulse target-free feasibility外沒有升格成verified experiment。
