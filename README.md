# ROGII

ROGII 是 Wellbore Geology Prediction 的研究與 Kaggle runtime，重點是井級資料隔離、inference-safe features、模型比較與 submission ID 對齊。主要閱讀路線從 last-known baseline 開始，再進入 physical-TVT alignment 與 residual GBDT；[model registry](src/rogii/models/__init__.py) 另保留其他模型和診斷路線。

## 原碼閱讀路線

| 順序 | 入口 | 內容 |
| --- | --- | --- |
| 1 | [Model protocol](src/rogii/models/base.py)、[Flat](src/rogii/models/flat.py) | `prepare`／`fit`／`predict` 與 last-known anchor baseline。 |
| 2 | [P3](src/rogii/models/p3.py)、[Alignment](src/rogii/alignment.py) | Physical alignment features、training-well residual fitting 與 TVT prediction。 |
| 3 | [Runner](src/rogii/runner.py)、[Folds](src/rogii/folds.py) | Well-level split、fold-local fitting 與 pooled row-RMSE。 |
| 4 | [Submission](src/rogii/submission.py) | 動態 test IDs、與 `sample_submission.csv` 的 join 及排序。 |
| 5 | [Fold tests](tests/test_folds_lean.py)、[Metric tests](tests/test_metrics.py) | 資料切分與指標契約。 |

## 保留的結果

[結果索引](results/recorded_runs.csv) 與兩份 [Flat](results/recorded/reset-flat-full.json)／[P3](results/recorded/reset-p3-full.json) record 記錄以下本機 full CV 摘要：

| 方法 | Pooled OOF row-RMSE | Wells | OOF rows |
| --- | ---: | ---: | ---: |
| Flat | 15.90985 | 773 | 3,783,989 |
| P3 | 15.45255 | 773 | 3,783,989 |

這些是歷史 CV 紀錄，不是 Kaggle leaderboard 分數，也不是這次文件更新重新訓練所得。[History](results/history.md) 保留比較的不確定性，包括主要信賴區間跨零；較低的點估計本身不足以證明穩健提升。

一般 clone **不包含競賽 `data/`、原始 OOF 或 trained models**。小型 JSON／CSV 能檢查摘要，但不能還原逐列 predictions 或獨立重算 full 分數。

## 安裝與資料

需要 Python 3.10 以上，依賴與 CLI 定義见 [pyproject.toml](pyproject.toml) 與 [CLI](src/rogii/cli.py)。Dependency ranges 尚不是完整鎖定環境；額外 sequence／pretrained／external 路線可能需要自己的依賴。

```sh
python -m pip install -e ".[dev]"
```

依競賽存取條件自行取得資料，放在本機唯讀輸入目錄：

```text
data/rogii-wellbore-geology-prediction/
  train/
  test/
  sample_submission.csv
```

程式拒絕把 run 或 submission 輸出寫入 `data/`。大型 cache、fold models、OOF 與 diagnostics 寫入 ignored `.runs/`。

## 本機使用

以下為 PowerShell 範例，需先準備資料和依賴：

```powershell
$dataRoot = "data/rogii-wellbore-geology-prediction"
rogii check --data-root $dataRoot
rogii run --model p3 --profile quick --data-root $dataRoot
rogii run --model p3 --profile full --run-id p3-full --data-root $dataRoot
rogii record --run .runs/p3-full --note "full five-fold reference"
rogii submit --run .runs/p3-full --data-root $dataRoot --output submission.csv
```

`quick` 預設使用 fold 0，以 stable hash 選 100 個 training wells 與 20 個 validation wells，結果不可與 full OOF 直接比較。`full` 使用固定五折；歷史 folds 保留於 [main_5fold.csv](configs/folds/main_5fold.csv)。要重現某個舊 run，需核對其資料、folds、seed、model configuration 與環境，而非只選同名 profile。

`rogii record` 僅寫入小型摘要；`rogii submit` 僅產生本機 CSV，不上傳 Kaggle。既有 run ID 不能覆寫，重跑請使用新 ID。Kaggle 執行入口另見 [notebooks/kaggle/README.md](notebooks/kaggle/README.md)。

## 模型與診斷的界線

Horizontal inference 欄位只使用 `MD, X, Y, Z, GR, TVT_input`，typewell 使用 `TVT, GR`。Target `TVT`、formation surfaces、`Geology` 與 PNG 不能作為 test-time features；任何 learned statistic 只能 fit 該 fold 的 training wells。詳細規則見 [AGENTS.md](AGENTS.md)。

`slp-e0` 是讀取 training target 的 representation diagnostic，固定標記 `uses_target=true`、`deployable=false`、`submission_eligible=false`。`slp-e1` 分離 inference-safe edge generator 與 target-bearing diagnostic harness，只輸出 loop-edge feasibility，不產生 submission TVT prediction。兩者的契約見 [SLP experiment protocol](audit/stratigraphic_program_slam_experiment_protocol.md)。Oracle floor 或 loop feasibility 不等同可部署模型的預測成效。

## 外部來源與歷史

[外部 prediction provenance](configs/external/rogii03_v5_provenance_20260729.json) 將來源／授權／輸入合規未證的產物限定為 `diagnostic_only`，不能用作本專案方法的 OOF、teacher fitting、blend weights 或 production ensemble 證據。[外部模型來源](results/external_model_registry_20260728.md) 與本人修改需保持區分。

[歷史競賽流程分析](audit/competition_audit_v2.md) 和 `results/history.*` 是歷史材料，不是現行開發 gate。清理前的 Git bundle 位於作者本機，不包含清理前 ignored artifacts，也不隨 clone 提供。Repository 目前未附根目錄 LICENSE；第三方資料與程式的權利需依來源分別確認。
