# ROGII — Stratigraphic Program Synthesis and Loop-Closure SLAM

# Formal Experiment Protocol V1

文件日期：2026-07-25（Asia/Taipei）  
文件狀態：Active research protocol  
工作區：`C:\Users\wurre\Desktop\ROGII`  
建議實驗代號：`SLP-V1`

> 本文件把「逐列預測 TVT」改寫為「反推出生成 TVT 的稀疏地質程式」。
> 它是一份可直接實作、驗證與否證的工程研究協議，不是
> preregistration、promotion gate 或 Kaggle 成績承諾。現行開發規則以
> `AGENTS.md` 為準；`competition_audit_v2.md` 與 research memory 僅用來辨認
> 歷史證據與已停止路線。

---

## 1. Executive decision

### 1.1 核心方向

`SLP-V1` 不再訓練另一個 pointwise regressor，也不以多模型 correction、
post-hoc blending 或一般 sequence smoothing 為主體。

它把每口水平井的未知段表示為一個很短的地質程式：

```text
ANCHOR(prefix)
→ DIP(apparent_slope, length)
→ FAULT(optional_throw)
→ DIP(apparent_slope, length)
→ ...
```

程式參數由四類 inference-safe 約束共同決定：

1. 已知 prefix 的 `TVT_input` 絕對錨點；
2. horizontal GR 與 typewell GR 的多尺度地層 landmark；
3. horizontal well 內部的 prefix-to-future 與 future-to-future loop closure；
4. 全井 cycle consistency、少事件先驗與可允許的 fault jump。

### 1.2 主要研究問題

> 在不使用 validation/test 的 target、formation surfaces、`Geology` 或 PNG
> 的前提下，能否從同井 GR 回環與少數地質事件，恢復
> `H(MD) = TVT(MD) + Z(MD)`，使 full well-level OOF pooled row-RMSE
> 低於 `4.0 ft`？

### 1.3 為甚麼值得直接驗證

全訓練集的 representation oracle 顯示，真正的 target path 高度可壓縮。
若 oracle 知道 post-PS target，只用少數等長線性地質段擬合
`H = TVT + Z`，便得到：

| 段數 | 連續 piecewise-linear `H` | 段間可跳躍 `H` |
|---:|---:|---:|
| 1 | 5.9639 | 5.9639 |
| 2 | 3.6891 | 3.1010 |
| 3 | 2.7301 | 2.0931 |
| 4 | 2.0942 | 1.5854 |
| 6 | 1.4988 | 1.0808 |

這不是可部署分數，因為 segment coefficients 是由真實 post-PS `TVT`
估計；它只回答表示法是否有足夠 headroom。答案是肯定的：`<4` 不需要
逐列自由度，只需要可靠推斷少數 slope、change point 與 fault throw。

---

## 2. Scope and non-goals

### 2.1 In scope

- 建立同井高精度 GR loop-closure detector。
- 用少數 `DIP`、`FAULT`、可選 `WARP` primitives 表示未知地質路徑。
- 用 global factor graph／minimum-description-length objective 聯合求解。
- 在固定 well-level folds 上產生完整 OOF prediction。
- 報告 pooled row-RMSE、tail、failure modes 與 solver diagnostics。
- 若同井訊號不足，再加入 fold-safe global RGT typewell atlas。
- 對外部 Texas 公開資料只做身分與合規 audit；未通過 audit 前不接入模型。

### 2.2 Out of scope for V1

- 新的 generic row regressor。
- 對 frozen P3 或其他歷史模型做 residual correction。
- 多模型 voting、stacking 或 leaderboard-driven bias correction。
- 把 TVT path 當一般平滑曲線後只調 regularization。
- 重新調整已停止的 prefix replay、global beam、sequence cost-volume、
  spatial broad prior 或 dense local alignment field。
- 使用 target `TVT`、formation surfaces、`Geology` 或 PNG 作 inference feature。
- 實際提交 Kaggle。

### 2.3 與歷史停止路線的差異

| 歷史路線 | 已知問題 | `SLP-V1` 的不同資訊／表示 |
|---|---|---|
| Prefix replay／nearest match | 單一匹配易 alias，無法校正 tail | 匹配只產生 factor；答案由多回環與 cycle consistency 決定 |
| Sequence cost-volume | 有 ranking signal，但 support 與 catastrophic tail 不足 | 直接搜索低維地質程式，不要求每 row 都有穩定候選 |
| Spatial surface prior | 未見 wells 泛化失敗 | V1 不以 `X,Y → surface` 為主預測器 |
| Dense local alignment | 局部場仍可能累積 drift | 非局部 loop closure 直接消除全井 drift |
| Generic smoothing | 會抹掉 fault，且不能解 GR alias | 稀疏段內規則，段間允許明確 jump |

---

## 3. Evidence base

本節所有數值分成三類：

- **Observed diagnostic**：直接由訓練資料重算；
- **Oracle**：使用 target，只能證明 headroom；
- **External mechanism evidence**：公司文件、專利或論文，不等於本資料已驗證。

### 3.1 Exact structural identity diagnostic

對六個 training-only formation surfaces
`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`，觀察到每個
well／formation 內近似：

\[
TVT_i + Z_i - S_{w,f}(X_i,Y_i) = c_{w,f}
\]

其 within-well pooled standard deviation 約 `0.0053 ft`。

因此定義：

\[
H_i = TVT_i + Z_i
\]

則 target 可寫成：

\[
TVT_i = H_i - Z_i
\]

`H` 是沿井軌跡的局部地層 elevation 加 datum。這個 identity 只用來選擇
state representation；V1 fitting 不讀 formation surfaces。

### 3.2 Structural compressibility oracle

計算範圍：

- 773 training wells；
- 每井 prediction start 為最後一個 non-null `TVT_input` row；
- post-PS 合計 `3,783,989` rows；
- 每井以 normalized post-PS MD 均分成 `K` 段；
- 評分為全部 rows 合併後的 pooled RMSE。

連續模型使用 hinge basis：

\[
H(s)=a+bs+\sum_{k=1}^{K-1}d_k\max(0,s-\kappa_k)
\]

可跳躍模型則讓每一等長區間各自 fit intercept 與 slope。

前述結果顯示：

- 2 個連續段已達 oracle `3.6891`；
- 2 個可跳躍段已達 oracle `3.1010`；
- 3 個可跳躍段已達 oracle `2.0931`；
- 4 個可跳躍段已達 oracle `1.5854`。

這支持「少數事件程式」而不是「高自由度逐列回歸」。

### 3.3 Same-well recurrence oracle

對每個 future row，比較其真實 TVT 與同井 prefix TVT support：

| 診斷 | 結果 |
|---|---:|
| Future TVT 落在 prefix TVT min/max 內 | 61.43% |
| 距最近 prefix TVT ≤ 0.25 ft | 62.12% |
| 距最近 prefix TVT ≤ 0.50 ft | 62.97% |
| 距最近 prefix TVT ≤ 1.00 ft | 64.42% |
| 距最近 prefix TVT ≤ 2.00 ft | 67.29% |
| 距最近 prefix TVT ≤ 4.00 ft | 72.33% |
| ≤ 1 ft 且 future GR observed | 44.31% |

每井 `≤1 ft` recurrence fraction：

- median：74.45%；
- q10：8.79%；
- q90：100%。

因此 recurrence 有大量 headroom，但存在明顯低覆蓋 tail。V1 必須同時回答：

1. GR descriptor 能否以極高 precision 找到正確 recurrence？
2. 無 recurrence 或缺 GR 的 wells，稀疏 structural program 能否維持安全？

### 3.4 Company and literature evidence

- ROGII 官方 AI Tools 說明 Auto-Steer 會產生 alternate interpretations、
  confidence、spectrum，並 auto-pick horizons、dips、faults。
- ROGII Geological Operations 說明 dynamic dip per segment、fault block
  與 throw distance。
- ROGII 的 geosteering patent 描述 variable formation thickness、multi-well、
  segments、faults、dips 與 structural map correction。
- Kuvaev、Uvarov 等人的 assisted geosteering patent 明確描述
  repeated-section、self-correlation 與 similarity local extrema。
- 官方 competition deck 第 9 頁建議使用 horizontal well 在 PS 前的高解析
  GR 與已知 TVT，去 correlate lateral 後段。
- Sylvester（2023）把 pairwise DTW correlations 放入 well graph，再以 global
  optimization 建立 cycle-consistent relative geologic time。

這些來源支持機制可行性，但不構成 `SLP-V1` 的 OOF 證據。

---

## 4. Formal hypothesis set

### SLP-H1 — Sparse structural program

對大部分 wells，post-PS 的：

\[
H_i=TVT_i+Z_i
\]

可由不超過 6 個 piecewise-affine apparent-dip segments，加上少量 fault
jumps 表示；representation error 顯著低於 `4 ft`。

目前狀態：**oracle supported，deployability untested**。

### SLP-H2 — High-precision loop closure

多尺度 horizontal GR descriptor 能在一個 precision-first operating point
識別：

- prefix-to-future 相同／近似 TVT；
- future-to-future 相同／近似 TVT；
- horizontal-to-typewell absolute landmark。

這些 edges 提供一般 smoother 不具備的非局部等式約束。

目前狀態：**untested**。歷史 prefix replay 的失敗不等同於此假設，因為它
沒有測 global cycle-consistent graph。

### SLP-H3 — Cycle consistency resolves aliasing

單一 GR window 可能對應多個 typewell depth；但多個 edges、known prefix、
segment slopes 與 fault events 的共同 cycles，可以排除局部相似度假峰。

目前狀態：**untested**。

### SLP-H4 — Global RGT atlas helps unsupported wells

若某井 prefix recurrence 太少，outer-training wells 的 typewell RGT atlas
可提供更可靠的 absolute landmarks，而不依賴單一 nearest typewell。

目前狀態：**secondary，只有 SLP-H2/H3 顯示局部有效但 coverage 不足時才測**。

### Null hypothesis

即使 target path 可被少數 segments oracle 壓縮，inference-safe GR 仍無法
可靠辨識 program parameters；任何改善只存在於 oracle，實際 OOF 仍受
aliasing、低 recurrence wells 與 false loop closures 支配。

---

## 5. State and generative model

### 5.1 Observed inputs

Horizontal well 可用：

```text
MD, X, Y, Z, GR, TVT_input
```

Typewell 可用：

```text
TVT, GR
```

其他欄位一律不可進入 inference pipeline。

### 5.2 Latent state

對 horizontal row \(i\)：

\[
u_i = TVT_i
\]

\[
h_i = u_i + Z_i
\]

最終輸出：

\[
\widehat{TVT}_i=\hat{h}_i-Z_i
\]

### 5.3 Program primitives

V1 只允許下列 primitives：

#### `DIP`

在 segment \(j\)：

\[
h_i=a_j+b_j(s_i-s_j)
\]

其中 \(s_i\) 預設為 along-hole MD offset。`b_j` 是沿該井方位觀察到的
apparent structural slope，不必先還原完整 3D dip vector。

#### `FAULT`

在 change point \(c_j\) 允許：

\[
h_{c_j^+}=h_{c_j^-}+\delta_j
\]

`FAULT` 是明確離散事件；不得由一般 smoother 自動吸收。

#### `WARP`（可選消融）

允許 typewell GR 的局部 stratigraphic scale 有有限 stretch／squeeze：

\[
t_i^{ref}=\alpha_j+r_j u_i
\]

`WARP` 只作用於 GR observation model，不直接增加 target path 的逐列自由度。

### 5.4 Anchor factor

對已知 prefix：

\[
\mathcal L_{anchor}
=\sum_{i\in prefix}w_i
\rho\left(h_i-(TVT\_input_i+Z_i)\right)
\]

Prefix 是絕對 gauge；不得重新估計或 correction。

### 5.5 Typewell landmark factor

以 fold-safe GR normalization 與 multi-scale descriptor，對 horizontal window
產生少量候選 typewell TVT \(t_m\)：

\[
\mathcal L_{type}
=\sum_{(i,m)}q_{im}\rho(u_i-t_m)
\]

其中 \(q_{im}\) 必須由 outer-training wells 校準。低置信候選不建立 hard
factor。

### 5.6 Same-well loop factor

若 horizontal windows \(i,k\) 被判定為相同 stratigraphic position：

\[
u_i\approx u_k
\]

等價於：

\[
h_i-h_k\approx Z_i-Z_k
\]

factor：

\[
\mathcal L_{loop}
=\sum_{(i,k)\in E}
q_{ik}\rho\left((h_i-h_k)-(Z_i-Z_k)\right)
\]

edge 分三類獨立記錄：

- `PF`：prefix-to-future；
- `FF`：future-to-future；
- `HT`：horizontal-to-typewell。

### 5.7 Program complexity

\[
\mathcal L_{MDL}
=\lambda_K K
+\lambda_F N_{fault}
+\lambda_\Delta\sum_j\rho(b_j-b_{j-1})
+\lambda_T\sum_j\rho(\delta_j)
\]

這裡的 slope penalty 只約束 segment parameters，不是逐 row sequence
smoothing。Fault 可存在，但必須支付明確 description cost。

### 5.8 V1 objective

\[
\min_{\mathcal P,h,E}
\mathcal L_{anchor}
+\mathcal L_{type}
+\mathcal L_{loop}
+\mathcal L_{GR}
+\mathcal L_{MDL}
\]

subject to：

- `1 ≤ K ≤ 6`；
- minimum segment length；
- bounded apparent slope；
- bounded／sparsified fault count；
- prefix exact anchoring；
- missing GR rows不建立 GR factors；
- 所有 confidence calibration 只 fit outer-training wells。

V1 輸出單一 MAP program。可以保留 alternate programs 作 uncertainty
diagnostic，但不做跨模型 blend。

---

## 6. GR representation

### 6.1 Design principle

GR 不視為一般 scalar regression feature，而視為沿 stratigraphic coordinate
觀察到的 barcode。Descriptor 必須同時保留：

- 多尺度峰谷；
- zero／level crossings；
- 一階與二階變化；
- window 內事件順序；
- 對 amplitude shift、scale、局部 missingness 的穩健性。

### 6.2 Candidate descriptors

V1 由低風險至高複雜度測：

1. robust z-normalized raw window；
2. derivative／rank-transformed window；
3. continuous-wavelet extrema hierarchy；
4. persistent landmark barcode；
5. small contrastive encoder。

Learned encoder 若使用，必須：

- 只在 outer-training wells 建立 positives／negatives；
- 至少 5 seeds；
- validation well 不可參與 normalization、index 或 threshold fitting；
- 與 handcrafted descriptor 分開報告，不可只報最佳 seed。

### 6.3 Precision-first edge policy

False positive loop closure 可能把整段 program 拉到錯誤 TVT。因此 edge
selection 優先 precision，不追求 row-level coverage。

每個 edge 必須帶：

- similarity；
- second-best margin；
- scale agreement；
- forward/backward consistency；
- cycle support count；
- missingness；
- calibrated confidence。

---

## 7. Experiment sequence

### 7.1 `SLP-E0` — Representation audit

目的：確認低維 program 的 oracle floor。

狀態：**已完成並固化為可重跑 diagnostic**。

已完成：

- `TVT` 對 MD 的 polynomial／piecewise oracle；
- `H=TVT+Z` 的 continuous／discontinuous equal-segment oracle；
- prefix recurrence coverage oracle；
- formation-surface identity diagnostic。

仍需在實作時固化為可重跑 diagnostic script，輸出到 `.runs/`，不可把 oracle
功能接入 inference code path。

驗收：

- 重跑數值與本文件一致至合理浮點誤差；
- 明確標記 `uses_target=true`；
- unit test 確保 production prediction path 無法 import oracle labels。

驗證紀錄（2026-07-25）：

- implementation：`src/rogii/diagnostics/slp_e0.py`；
- command：`rogii slp-e0 --data-root data/rogii-wellbore-geology-prediction --output-root .runs --run-id slp-e0-full --workers 4`；
- artifacts：`C:\Users\wurre\Desktop\ROGII\.runs\slp-e0-full`；
- full-data validation：773 wells、3,783,989 post-PS rows，flat／polynomial／
  piecewise／recurrence 與本協議一致，六個 formation 的 pooled within-well
  standard deviation 均小於 `0.01 ft`。

### 7.2 `SLP-E1` — Loop-closure feasibility

目的：回答 GR 是否能可靠建立非局部 constraints。

### Data construction

- Outer validation wells 使用既有 PS boundary，完整 post-PS 只作 scoring。
- Outer-training wells 可產生多個 artificial cuts，訓練 descriptor 與
  confidence calibration。
- 所有 pairs 必須在 well-level fold 內建立。

### Pair tasks

1. `PF`：future window 查 prefix windows；
2. `FF`：future window 查同一 future 的非鄰近 windows；
3. `HT`：horizontal window 查該井 typewell；
4. optional `HA`：horizontal window 查 fold-trained RGT atlas。

### Ground-truth edge definition

主要 positive：

\[
|TVT_i-TVT_k|\le1.0\text{ ft}
\]

另報：

- strict：`≤0.5 ft`；
- loose：`≤2.0 ft`；
- catastrophic false edge：`>10 ft`。

### Metrics

- precision–coverage curve；
- average precision；
- strict precision at fixed coverage；
- catastrophic false-edge rate；
- TVT error median／q90／q99；
- per-well edge coverage；
- low-recurrence q10 wells 的獨立結果；
- confidence calibration／reliability curve。

### Engineering decision criteria

以下是後續成本判斷，不是 governance gate：

- **Strong**：precision `≥99%` 且 future-row coverage `≥10%`；
- **Viable**：precision `≥97%` 且 coverage `≥30%`；
- **Unsafe**：catastrophic false-edge rate `>0.5%`，或 confidence 無法跨 fold
  校準。

若只有 typewell matches 有訊號而同井 loops 無訊號，`SLP-H2` 視為部分失敗，
不得把普通 retrieval 改名後直接進 E2。

### Negative controls

- 在 well 內 shuffle GR windows；
- 使用錯誤 typewell；
- 只保留單點 GR、移除 window order；
- 破壞 multi-scale landmark order；
- 把 nearest-neighbor score 當 edge，不做 margin／cycle screening。

真方法必須明顯優於所有 controls。

### 驗證紀錄（2026-07-26）

狀態：**已完成 quick 與固定 five-fold full OOF diagnostic；同井 loop closure
不具工程可行性，不進 `SLP-E2`。**

- inference-safe implementation：
  `src/rogii/stratigraphic/descriptors.py`、
  `src/rogii/stratigraphic/loop_edges.py`、
  `src/rogii/stratigraphic/cycle_checks.py`、
  `src/rogii/stratigraphic/confidence.py`；
- target-bearing harness：`src/rogii/diagnostics/slp_e1.py`；
- quick command：
  `rogii slp-e1 --profile quick --data-root data/rogii-wellbore-geology-prediction --fold-config configs/folds/main_5fold.csv --output-root .runs --run-id slp-e1-quick --seed 20260725 --workers 4`；
- full command：
  `rogii slp-e1 --profile full --data-root data/rogii-wellbore-geology-prediction --fold-config configs/folds/main_5fold.csv --output-root .runs --run-id slp-e1-full --seed 20260725 --workers 4`；
- full artifacts：`C:\Users\wurre\Desktop\ROGII\.runs\slp-e1-full`；
- full scope：5 folds、773 OOF wells、3,783,989 all-future rows、
  2,583,152 GR-observed rows、1,600,549 descriptor-eligible rows，validation
  query 未抽樣；
- `PF`：沒有任何 fold 產生具最低 support 的 97%／99% operating point；
  OOF AP `0.04752`、ECE `0.00211`，因此判為 `unsupported`；
- `FF`：部分 folds 以僅 1–3 個 training edges 得到表面上 100% precision，
  遠低於 1,000 edges／25 wells 的最低 support；套到 OOF 後共放行 106 edges，
  precision `3.77%`、all-future coverage `0.00219%`、catastrophic rate
  `51.89%`、AP `0.13891`、ECE `0.01062`，判為 `unsafe`；
- `HT`：沒有具 support 的 operating point；OOF AP `0.00207`、ECE
  `0.00187`，判為 `unsupported`；
- `PF+FF` same-well 與 all-edge combined 都因 FF 的 catastrophic failures
  判為 `unsafe`；
- low-recurrence q10 cohort 為 78 wells／388,968 rows；只有 4 個 FF edges
  被放行，primary precision `0%`、catastrophic rate `75%`，PF／HT 均為
  zero support；
- 主方法沒有一致優於 controls：`point_gr` 的 FF AP `0.18129` 高於主方法，
  `raw_nn` 與 `no_cycle` 的 PF AP 也略高；shuffle GR 的 FF accepted-edge
  precision 與 catastrophic safety 反而優於主方法。cycle／order features
  未能解決 aliasing；
- 限制：V1 使用 deterministic handcrafted descriptor；training artificial
  cuts 以每 well／cut 48 個 deterministic query rows 控制校準成本，但 full
  outer-validation 使用全部 eligible rows。沒有 HA、learned descriptor、
  TVT path 或 E2 solver。

### 7.3 `SLP-E2` — Program-only end-to-end decoder

目的：不用 P3、correction 或 blend，直接由 program 產生完整 TVT。

### Solver ladder

按下列順序建立，避免一次加入全部自由度：

1. `ANCHOR + 1 DIP`；
2. `ANCHOR + K DIP`，`K=1..6`，連續；
3. 加 `FAULT` jumps；
4. 加 typewell landmark factors；
5. 加 `PF` loop factors；
6. 加 `FF` loop factors；
7. 加 cycle-consistency rejection；
8. optional 加 `WARP`。

### Discrete／continuous optimization

建議採兩層 MAP：

1. minimum-description-length changepoint solver 產生少量 program structures；
2. 對每個 structure 以 robust least squares／factor optimization 求
   slope、throw、warp；
3. 用所有 anchors、loop cycles 與 GR likelihood 選單一 MAP program。

不得以 validation target 選 K、change points、edge threshold 或 fallback。

### Required ablations

| Ablation | 回答問題 |
|---|---|
| No typewell | 同井 loops 本身是否足夠 |
| No `PF` | prefix recurrence 的真正增益 |
| No `FF` | future-to-future loop 是否抑制 drift |
| No cycles | global consistency 是否勝過獨立 match |
| No faults | jump primitive 是否必要 |
| No MDL | 少事件 prior 是否真的提高泛化 |
| Oracle edges | solver 有無足夠 capacity |
| Oracle change points | 瓶頸在 segmentation 還是 continuous parameters |
| Oracle program coefficients | 重現 representation floor |

### Primary metric

全部 OOF prediction rows 的：

\[
RMSE_{pooled}
=\sqrt{\frac{\sum_i(y_i-\hat y_i)^2}{N}}
\]

不得用平均 per-well RMSE 或平均 fold RMSE 代替。

### Secondary metrics

- per-well RMSE median／q75／q90／q95／max；
- worst 20 wells；
- `RMSE > 20`、`>40` wells count；
- missing-GR rows；
- low-recurrence wells；
- long post-PS wells；
- high-curvature Z trajectory；
- inferred-fault wells；
- runtime、peak memory、edge count、program length。

### Result tiers

| Full strict OOF 結果 | 解讀 |
|---|---|
| `RMSE ≥12` 或 catastrophic tail 比 flat 更差 | 主機制沒有實用支援，停止高成本擴張 |
| `<12` 且 tail 不惡化 | 有結構訊號，但不足以主張突破 |
| `<8` | 稀疏 program 是強候選，可進 E3 |
| `<5` | 已接近目標，優先修 edge precision／unsupported wells |
| `<4` | 主要研究目標達成，仍需完整 leakage、seed、tail 與 runtime audit |

所有 tier 必須在相同 wells／rows 上 paired 比較 flat 與 frozen P3；不可跨
comparability group 比絕對 RMSE。

### 7.4 `SLP-E3` — Global RGT synchronization atlas

只在 E2 顯示：

- 有效 edges precision 足夠；
- 但低-recurrence wells support 不足；

時才執行。

### Atlas construction

1. outer-training typewells 為 nodes；
2. fold-safe pairwise multi-scale alignment 為 edges；
3. 以 global optimization 解 relative geologic time；
4. 用 cycle residual 淘汰不一致 pairwise alignments；
5. validation/test typewell 只作 query attachment，不更新 fold-trained atlas。

### Evaluation

- `E2 + atlas` 對 `E2 no atlas` paired delta；
- low-recurrence q10 wells；
- geographically distant／domain-shift wells；
- atlas triangle closure residual；
- absolute landmark error。

Atlas 若只改善典型 wells、惡化 tail，則不進主線。

### 7.5 `SLP-X1` — External public-data identity audit

這是獨立研究線，不與 E1–E3 混合。

### Working inference

formation codes 很可能對應：

```text
ANCC   → Anacacho
ASTNU  → Austin upper
ASTNL  → Austin lower
EGFDU  → Eagle Ford upper
EGFDL  → Eagle Ford lower
BUDA   → Buda
```

公開 Texas stratigraphy 支持這個地層順序，但 dataset location 與 well identity
尚未證實。

### Audit steps

1. 推斷／驗證 `X,Y` CRS，不以單一 plausible transform 當結論；
2. 以 formation order、depth、trajectory geometry 與 GR barcode 做多證據
   identity check；
3. 只查官方／公平公開來源，如 Texas RRC、USGS、BEG；
4. 記錄 license、取得方式、時間點與是否所有參賽者可合理取得；
5. 驗證 Kaggle 規則、offline requirement 與 hidden-test 可重現性；
6. 身分與合規未同時成立前，不把外部 data 寫入 features、atlas 或 cache。

### Stop conditions

- CRS 不唯一；
- 只能靠非公開商業資料匹配；
- public records 只有不可穩定解析的 images；
- 無法在 hidden rerun 離線重建；
- 競賽規則或公平取得性不明。

---

## 8. Validation contract

### 8.1 Data integrity

每次 run 前：

```powershell
rogii check
```

必須動態列舉 train／test wells，不可硬編 well IDs、well count 或 row count。

### 8.2 Fold policy

- 固定 well-level folds；
- 同一 well 不可跨 train／validation；
- scaler、normalizer、descriptor、encoder、threshold、index、atlas、
  program hyperparameters 只 fit outer-training wells；
- validation typewell 可作該 query 的 inference input，但不可更新 learned atlas；
- outer validation post-PS target 只用於最後 scoring。

### 8.3 Profiles

```powershell
rogii run --profile quick
rogii run --profile full
```

- quick：功能、數值穩定、edge precision 與 solver feasibility；
- full：完整固定 folds、完整 OOF、正式 pooled metric。

本協議不要求額外研究審批；是否直接跑 full 由成本與 quick 結果決定。

### 8.4 Stochastic components

若使用 learned descriptor：

- 至少 5 seeds；
- 報 mean、standard deviation 與逐 seed pooled RMSE；
- 不得只保留最佳 seed；
- deterministic handcrafted baseline 必須一併報告。

### 8.5 Comparators

同一 OOF rows 上至少比較：

1. flat last-known；
2. frozen P3；
3. program-only `SLP-V1`；
4. required ablations。

P3 只作 comparator；不得對其 residual 做 correction。

---

## 9. Leakage and inference-safety checklist

每個 run 必須確認：

- [ ] Horizontal features 只含 `MD,X,Y,Z,GR,TVT_input`。
- [ ] Typewell features 只含 `TVT,GR`。
- [ ] 沒有 target `TVT` 進入 fitting、edge generation 或 threshold calibration。
- [ ] 沒有 formation surfaces 進入 V1 fitting。
- [ ] 沒有 `Geology` 或 PNG。
- [ ] Prefix boundary 後的 validation `TVT` 只在 scoring 時讀取。
- [ ] 所有 learned statistics 只 fit outer-training wells。
- [ ] Neighbor／RGT index 不含 outer validation wells。
- [ ] Missing GR 不被 target-aware imputation。
- [ ] Oracle code path 與 deployable predictor 物理分離。
- [ ] Submission IDs 動態列舉並依 `sample_submission.csv` join／排序。
- [ ] 沒有 Kaggle upload side effect。

---

## 10. Failure decomposition

若 E2 失敗，必須用以下 ladder 定位，不得直接換大模型：

| Test | 若通過 | 若失敗 |
|---|---|---|
| Oracle program coefficients | 表示法容量足夠 | primitives／segment cap 不足 |
| Oracle change points + real coefficients inference | change-point search 是瓶頸 | GR likelihood／parameter inference 是瓶頸 |
| Oracle loop edges + real solver | edge detector 是瓶頸 | factor graph／program model 是瓶頸 |
| Real high-confidence edges + oracle program form | program form足夠 | false edges／coverage 是瓶頸 |
| No-loop structural decoder | geometry prior有獨立訊號 | 必須依賴 GR loops／atlas |
| Low-recurrence cohort | 支援可轉移 | 需要 RGT atlas或判定不可解 |

只有 failure decomposition 指向新的 inference-safe information 時，才新增
mechanism；不得回到 model capacity、blend 或 correction 搜尋。

---

## 11. Risk register

| Risk | Consequence | Required mitigation |
|---|---|---|
| GR aliasing | 錯誤 absolute TVT | multi-scale landmarks、margin、cycle checks |
| False loop closure | 整段被拉到錯層 | precision-first edges、robust loss、edge rejection |
| Low recurrence wells | 無足夠 anchors | sparse geometry、typewell landmarks、optional RGT atlas |
| Missing future GR | 局部無 observation | program propagation；不做 target-aware imputation |
| Fault 被平滑掉 | 大段 systematic error | explicit `FAULT` primitive |
| 過多 primitives | oracle 好、OOF 差 | `K≤6`、MDL、minimum segment length |
| 錯誤 typewell landmark | absolute gauge 錯位 | alternate hypotheses、cycle and prefix consistency |
| Fold leakage from atlas | 虛假 OOF | outer-training-only atlas/index |
| External data contamination | 規則／公平性風險 | 獨立 X1 audit，未核准不接模型 |
| Solver local optimum | seed／初始化不穩 | deterministic discrete search、multi-start diagnostics |
| Runtime explosion | Kaggle 不可執行 | edge sparsification、bounded K、offline benchmark |

---

## 12. Required artifacts

大型輸出全部留在 ignored `.runs/`：

```text
.runs/slp_v1/
  config.json
  fold_manifest.csv
  edge_metrics.json
  edges_oof.parquet
  programs_oof.jsonl
  oof_predictions.parquet
  per_well_metrics.csv
  ablations.csv
  runtime.json
  leakage_check.json
  figures/
```

只把小型、可審查摘要交給：

```powershell
rogii record
```

正式摘要至少包含：

- git revision；
- data hash／fold manifest；
- feature schema；
- seed；
- pooled RMSE；
- per-well tail；
- edge precision／coverage；
- program length／fault count；
- ablation deltas；
- decision 與已知限制。

---

## 13. Recommended implementation boundaries

建議把 oracle、edge generation、program solver 與 model wrapper 分離：

```text
src/rogii/stratigraphic/
  descriptors.py
  landmarks.py
  loop_edges.py
  cycle_checks.py
  program.py
  solver.py
  diagnostics.py

src/rogii/models/
  slp.py

tests/
  test_slp_program.py
  test_slp_edges.py
  test_slp_fold_safety.py
  test_slp_submission.py
```

最低測試要求：

- synthetic known-dip well 能恢復 slope；
- synthetic fault 能恢復 jump，且 no-fault ablation 明顯失敗；
- false loop 被 cycle check 拒絕；
- prefix anchors 不可移動；
- validation rows 不參與 fitting；
- shuffled GR 不得產生虛假高 precision；
- output row order 與 sample IDs 完全一致。

---

## 14. Immediate execution order

1. 固化 `SLP-E0` diagnostic script與測試。
2. 實作 handcrafted multi-scale descriptor。
3. 跑 `SLP-E1 PF` precision–coverage。
4. 加 `FF` edges 與 cycle consistency。
5. 實作 `ANCHOR + K DIP` program solver。
6. 加 explicit `FAULT`。
7. 跑 E2 solver ladder 與 oracle ablations。
8. 若 `<8` 且 tail 改善，跑 full E2。
9. 只有 low-recurrence support 仍是主瓶頸時才做 E3 RGT atlas。
10. X1 external identity audit 獨立進行，不影響 E1–E3。

第一個可提交給研究決策的結果不是單一 RMSE，而是：

```text
PF/FF edge precision–coverage
+ oracle-edge solver RMSE
+ real-edge solver RMSE
+ low-recurrence tail
+ program complexity
```

---

## 15. Final falsification statement

`SLP-V1` 在以下任一情況應停止，而不是繼續加模型容量：

1. Oracle edges 下的 real solver 仍無法接近 representation oracle；
2. 高 confidence loop edges 無法跨 folds 達到 precision-first operating point；
3. real-edge program OOF `≥12`，或 catastrophic tail 顯著劣於 flat；
4. 主要收益只能由 validation-specific threshold、fallback 或 correction 取得；
5. 低-recurrence wells 無 inference-safe support，RGT atlas 亦無改善；
6. `<4` 只存在於 oracle、不同 comparability group 或 leakage-prone evaluation。

若停止，結論應是：

> Target path 確實可被低維地質程式表示，但目前 inference-safe GR 與 prefix
> 無法可靠識別該程式。

這比再次宣稱「模型不夠大」更具研究價值。

---

## 16. References

### Local sources

1. `AGENTS.md` — current development and inference-safety rules.
2. `audit/competition_audit_v2.md` — historical audit and stopped-route context.
3. `data/rogii-wellbore-geology-prediction/AI_wellbore_geology_prediction_task_en.pptx`
   — official competition data brief, especially slides 5, 8, 9, 12 and 13.

### Company documents and patents

1. ROGII, “AI Tools for Optimized Workflows.”  
   <https://www.rogii.com/solutions/ai-tools-for-optimized-workflows>
2. ROGII, “Geological Operations.”  
   <https://www.rogii.com/solutions/geological-operations>
3. ROGII, “StarSteer.”  
   <https://www.rogii.com/products/starsteer>
4. ROGII Inc., “Systems and methods for horizontal well geosteering,”
   US20190106974A1.  
   <https://patents.google.com/patent/US20190106974A1/en>
5. Denisenko, Kushmantsev, Uvarov and Kuvaev,
   “Methods for Assisted and Automated Horizontal Well Geosteering,”
   US20230019126A1.  
   <https://patents.google.com/patent/US20230019126A1/en>

### Primary research

1. Sylvester, Z. (2023), “Automated multi-well stratigraphic correlation and
   model building using relative geologic time,” *Basin Research*, 35,
   1961–1984.  
   <https://onlinelibrary.wiley.com/doi/full/10.1111/bre.12787>
2. Grose et al. (2021), “LoopStructural 1.0: time-aware geological modelling,”
   *Geoscientific Model Development*, 14, 3915–3937.  
   <https://gmd.copernicus.org/articles/14/3915/2021/>
3. Pisel et al. (2022), “A recommender system for automatic picking of
   subsurface formation tops.”  
   <https://arxiv.org/abs/2202.08869>

### Public geology and data

1. U.S. Geological Survey, regional stratigraphic units including Anacacho,
   Austin Chalk, Eagle Ford and Buda.  
   <https://pubs.usgs.gov/fs/2016/3002/fs20163002.pdf>
2. Railroad Commission of Texas, Oil & Gas Well Records — GIS Well Logs.  
   <https://www.rrc.texas.gov/oil-and-gas/research-and-statistics/obtaining-commission-records/oil-gas-well-records-gis-well-logs/>
3. Railroad Commission of Texas, public downloadable datasets.  
   <https://www.rrc.texas.gov/resource-center/research/data-sets-available-for-download/>
