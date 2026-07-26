# ROGII

ROGII Wellbore Geology Prediction 的精簡研究與 Kaggle runtime。Active code
只保留 Flat last-known 與 P3 physical-TVT alignment；舊路線留在
[`results/history.md`](results/history.md) 與外部 legacy bundle，不再增加日常
開發負擔。

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
- `full` 使用固定 5-fold 與全部 773 wells。
- `--folds`、`--max-train-wells`、`--max-validation-wells`、`--seed` 與
  `--workers` 都可直接覆寫，不需要 protocol receipt。
- `.runs/` 保存 cache、fold models、OOF 與 diagnostics，但不進 Git。
- `rogii record` 只把小型 JSON/CSV 摘要寫入 `results/`。
- `rogii submit` 只產生本機 CSV，不會上傳 Kaggle。

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
