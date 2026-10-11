# ROGII

ROGII Wellbore Geology Prediction 的研究與 Kaggle runtime。建議先從 Flat
last-known 與 P3 physical-TVT alignment 審閱；目前
[`model registry`](src/rogii/models/__init__.py)還包含其他模型與診斷路線。
舊路線的摘要留在 [`results/history.md`](results/history.md) 與外部 legacy bundle。

## 教授閱讀入口與展示邊界

此 clone 有可追查的模型來源、well-level validation 與歷史摘要，**未包含競賽
`data/` 或 `.runs/` 的 OOF／models，因此目前不能獨立重算既有 full 分數**。
完整五項評估、公開前問題與歸屬限制見
[`Professor-facing Repository Audit`](PROFESSOR_AUDIT.md)。

1. [`Flat`](src/rogii/models/flat.py)與 [`P3`](src/rogii/models/p3.py)：先理解
   last-known baseline，再讀 physical alignment features 與 residual GBDT。
2. [`Runner`](src/rogii/runner.py)、[`folds`](src/rogii/folds.py)和
   [`submission alignment`](src/rogii/submission.py)：檢查 well 隔離、pooled
   row-RMSE、test IDs 的 join 與排序。
3. [`歷史 full 摘要`](results/recorded_runs.csv)及
   [`P3 run record`](results/recorded/reset-p3-full.json)：記載 Flat/P3 分別為
   `15.90985`／`15.45255`，773 wells、3,783,989 OOF rows。這是本機 CV 歷史
   紀錄，並非新跑結果或 Kaggle 排名；也不足以單獨宣稱穩健優於 baseline。
4. [`第三方 prediction provenance`](configs/external/rogii03_v5_provenance_20260729.json)：
   部分外部產物來源／授權／輸入合規未證，已標 `diagnostic_only`，不能當作本人
   方法的 OOF 或 production ensemble 證據。

以下安裝與 CLI 命令已對照 [`pyproject.toml`](pyproject.toml)及
[`CLI`](src/rogii/cli.py)查實（source checked），本次沒有安裝套件或執行 CV。
安裝前需 Python 3.10 以上；目前 dependency ranges 不是完整環境 lock。

## 安裝

```powershell
python -m pip install -e ".[dev]"
```

本機競賽資料預期位於：

```text
data/rogii-wellbore-geology-prediction/
  train/
  test/
  sample_submission.csv
```

`data/` 完全不納入 Git，程式也拒絕把 run 或 submission 寫入其中。

## 使用

```powershell
$dataRoot = "data/rogii-wellbore-geology-prediction"

rogii check --data-root $dataRoot
rogii slp-e0 --data-root $dataRoot --output-root .runs --run-id slp-e0-full --workers 4
rogii slp-e1 --profile quick --data-root $dataRoot --fold-config configs/folds/main_5fold.csv --output-root .runs --run-id slp-e1-quick --seed 20260725 --workers 4
rogii slp-e1 --profile full --data-root $dataRoot --fold-config configs/folds/main_5fold.csv --output-root .runs --run-id slp-e1-full --seed 20260725 --workers 4
rogii run --model p3 --profile quick --data-root $dataRoot
rogii run --model p3 --profile full --run-id p3-full --data-root $dataRoot
rogii record --run .runs/p3-full --note "full five-fold reference"
rogii submit --run .runs/p3-full --data-root $dataRoot --output submission.csv
```

- `quick` 預設使用 fold 0，以 stable hash 選 100 個 training wells 與 20 個
  validation wells；結果會標記為不可與 full OOF 比較。
- `full` 使用固定 5-fold 與全部 training wells；歷史 full record 為 773 wells。
- `--folds`、`--max-train-wells`、`--max-validation-wells`、`--seed` 與
  `--workers` 都可直接覆寫，不需要 protocol receipt。
- `.runs/` 保存 cache、fold models、OOF 與 diagnostics，但不進 Git。
- `rogii record` 只把小型 JSON/CSV 摘要寫入 `results/`。
- `rogii submit` 只產生本機 CSV，不會上傳 Kaggle。

重現前應先自行依競賽存取條件取得資料，再執行 `rogii check`。建議先跑
`quick` 核查流程；完整歷史分數需另用相同 folds、seed、model configuration
與相容環境重跑 `full`，不能由小型摘要還原 OOF predictions。範例 run ID
若已存在，請換新 ID；[`Runner`](src/rogii/runner.py)會拒絕覆寫既有 run。
額外 sequence／external 路線的環境依賴需另讀各自 runbook，不把基本安裝視為
所有研究路線的完整環境。

`rogii slp-e0` 是獨立的 representation diagnostic，會讀取全部 training
target `TVT` 與 training-only formation surfaces，量測 polynomial、
piecewise、prefix recurrence 與 surface identity 的 oracle floor。它固定標記
`uses_target=true`、`deployable=false`、`submission_eligible=false`，不可用於
model registry、test inference 或 submission。正式定義見
[`audit/stratigraphic_program_slam_experiment_protocol.md`](audit/stratigraphic_program_slam_experiment_protocol.md)
的 7.1 節。

`rogii slp-e1` 將 inference-safe edge generator 與 target-bearing diagnostic
harness 分開。前者只使用 horizontal `MD, X, Y, Z, GR, TVT_input` 與
typewell `TVT, GR` 建立 PF／FF／HT candidates；後者只在 outer-training
wells 的 artificial cuts 上校準 confidence 與選 threshold，並在 candidate、
confidence 和 accept/reject 全部固定後才讀 outer-validation target 評分。
它只輸出 loop-edge feasibility diagnostics，不產生 TVT prediction、不進 model
registry，也不可用於 submission。正式定義與驗證契約見同一協議的
[7.2 節](audit/stratigraphic_program_slam_experiment_protocol.md#72-slp-e1--loop-closure-feasibility)。

## 開發新模型

在 `src/rogii/models/` 實作 `Model` protocol 的 `prepare`、`fit` 與 `predict`，
再加入 `src/rogii/models/__init__.py` registry。Runner 只把 outer-training
targets 傳給 `fit`；validation/test 一律使用 inference-safe view。

## 歷史與備份

- Audit V2 正式報告：[`audit/competition_audit_v2.md`](audit/competition_audit_v2.md)
  （SHA-256 `2A2F1C9C7D3A283AE873351A8E957C1BBE8C00A73547C4DB36EEB156D30680CF`）。
  它是歷史審計，不是現行研究政策。
- 清理前完整 Git bundle：
  `C:\Users\wurre\Desktop\ROGII-legacy-20260725.bundle`
  （SHA-256 `8679EA07C307DF84E51F91EE3DF5BADE0DB569F694370E021C5C3006F7689F94`）。
- Bundle 保存舊 Git history；清理前 ignored `artifacts/` 不在 bundle 內。
