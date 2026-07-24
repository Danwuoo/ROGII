# ROGII 開發規則

本專案採開放式開發。任何 inference-safe 想法都可直接用 quick 或 full CV
驗證，不需要 preregistration、cohort ledger、promotion gate、protocol receipt
或 research-memory 查詢。

## 必守底線

- `data/` 是本機唯讀輸入；不要覆寫、刪除或把輸出寫入其中。
- Horizontal inference 欄位只可使用 `MD, X, Y, Z, GR, TVT_input`；
  typewell 只可使用 `TVT, GR`。
- 不得把 target `TVT`、formation surfaces、`Geology` 或 PNG 當作
  test-time feature。
- validation 必須以 well 為單位分割；任何 scaler、imputer、model、
  index 或 learned statistic 只能 fit 該 fold 的 training wells。
- 主指標是全部 OOF prediction rows 的 pooled row-RMSE。
- test wells 與 submission IDs 必須動態列舉，並以
  `sample_submission.csv` 的 `id` join 與排序。
- 可在本機產生 submission CSV；實際上傳 Kaggle 必須由使用者明確要求。
- 外部資料或模型必須符合競賽規則、授權、公平可取得性與離線執行要求。

## 日常流程

1. `rogii check` 驗證資料。
2. `rogii run --profile quick` 快速迭代。
3. 有需要時直接跑 `--profile full`；沒有額外研究審批。
4. `rogii record` 只保留小型摘要；大型 cache、OOF 與 models 留在 ignored
   `.runs/`。

`audit/competition_audit_v2.md` 與 `results/history.*` 是歷史資料，不是現行
開發 gate。外部 research memory 可選用於查舊結果，但不再是必要入口。
