# Professor-facing Repository Audit — ROGII

這個 repo 適合展示「well 隔離、inference-safe features 與模型／診斷分界」的工程與研究思考。主要缺口是 clone 缺競賽資料與原始 OOF，環境沒有完整 lock，且本人、AI 與外部方法的貢獻範圍未確認。歷史分數可引用為紀錄，不能當作本次獨立重現或排名證明。

審查日期：2026-10-11（Asia/Taipei）。檢視 HEAD：`5bfc75c2057fe64c3208024b40efda2c57b85cc4`。初始 Git 工作樹乾淨；本次僅修改 README、新增 audit，未改模型、配置、資料或結果。審查依本地文件與代表性來源，未取得新資料、未跑訓練或查新榜。

本次 README 與 audit 由 AI 協助整理；這項文件協作不代表原專案的研究或程式貢獻已完成歸屬確認。

## 評分

採 1–5 分：1＝入口或證據缺失；2＝部分具備且有重要限制；3＝可沿文件核查但有公開前缺口；4＝易審閱且範圍內可重現；5＝獨立重現與個人歸屬完整。分數是本次審查判斷，不是研究成績。

| 面向 | 分數 | 本地依據與理由 |
| --- | ---: | --- |
| Readiness | 3 | [package/CLI manifest](pyproject.toml)、[CLI](src/rogii/cli.py)與 [Kaggle launchers](notebooks/kaggle/README.md)有具體入口；[ignored inputs/outputs](.gitignore)不在 clone，且 manifest 使用浮動 dependency ranges。命令已查實，尚未獨立執行。 |
| Readability | 4 | [model protocol](src/rogii/models/base.py)、[Flat](src/rogii/models/flat.py)、[P3](src/rogii/models/p3.py)、[Runner](src/rogii/runner.py)與 [submission](src/rogii/submission.py)責任可分別追讀；README 已修正「只剩兩模型」與 [八模型 registry](src/rogii/models/__init__.py)的落差。 |
| Verifiability | 3 | [record index](results/recorded_runs.csv)、[Flat record](results/recorded/reset-flat-full.json)、[P3 record](results/recorded/reset-p3-full.json)保留 seed／fold／rows／summary；缺 `data/` 與 `.runs/` 原始 OOF／model，讀者只能查摘要一致性，不能就地重算。 |
| Technical Depth | 4 | [physical alignment](src/rogii/alignment.py)、[fold-local runner](src/rogii/runner.py)、[fold tests](tests/test_folds_lean.py)及 [SLP protocol](audit/stratigraphic_program_slam_experiment_protocol.md)呈現方法與資料邊界；target-bearing diagnostic 有明確非 submission 限制。效益仍須依各路線的證據範圍判讀。 |
| Individual Contribution | 2 | [外部 provenance registry](configs/external/rogii03_v5_provenance_20260729.json)與 [external architecture registry](results/external_model_registry_20260728.md)有來源紀錄；本人創作、整合、分析、AI 實作／人工驗證分工未確認，Git 作者不能替代此確認。 |

## 核心閱讀路線

從 [Flat](src/rogii/models/flat.py)理解 last-known anchor，再讀 [P3](src/rogii/models/p3.py)的 `prepare → fit → predict`：alignment 與基礎特徵在 prepare 合併，fit 使用 training-well residual，predict 回到 TVT。接著看 [Runner](src/rogii/runner.py)如何把 training target 與 inference view 分開，再讀 [fold partition](src/rogii/folds.py)和 [ID alignment](src/rogii/submission.py)。[Tests](tests/test_folds_lean.py)與 [metric tests](tests/test_metrics.py)可以定位契約；本次未執行。此導覽保留既有核心。

## 公開前必須修復或明確交代

| 優先 | 問題 | 完成條件與現有依據 |
| --- | --- | --- |
| P0 | 個人與 AI 貢獻未確認 | 作者確認 method／implementation／experiment／analysis 各自負責者、AI 協助與人工驗證範圍；[external registry](results/external_model_registry_20260728.md)中的架構啟發需與本人修改分開。 |
| P0 | 根授權與外部權利未釐清 | 根目錄未見 LICENSE。[ROGII-03 registry](configs/external/rogii03_v5_provenance_20260729.json)明載 dataset license `Unknown`、exact lineage 與 input compliance 未證。保留 attribution，不以 notebook 的 Apache-2.0 推定其全部資料也可再散布。 |
| P0 | 外部 prediction 不可充當自身方法成果 | Registry 將外部 CSV 限於 target-free geometry/provenance；禁止 teacher fitting、OOF scoring、blend weights 或 production ensemble。對外材料應沿用此範圍，不能把 filename 中的 LB 數字寫成本人分數。 |
| P1 | 完整重現缺輸入與原始產物 | 依 [README 資料結構](README.md#安裝)交代合法取得方式與所需版本；[recorded JSON](results/recorded/reset-p3-full.json)是 summary，不包含足以重算的 OOF。若舊 `.runs/` 不可取得，明寫不能重算，不能新造替代證據。 |
| P1 | 環境版本未鎖定 | [pyproject](pyproject.toml)基本依賴使用 lower bounds；額外 sequence 與 pretrained probe 各有環境紀錄。發佈前明列實際使用版本／硬體與可用依賴，不宣稱所有路線只靠基本安裝即可重現。 |
| P1 | 歷史 superiority 容易被過度宣稱 | [record index](results/recorded_runs.csv)的 P3 pooled RMSE 較低；[history](results/history.md)仍記載 main CI crossed zero。呈現點估計時保留其比較與不確定性限制，不以較低單一數字宣稱穩健提升。 |
| P1 | 本地路徑與資料標識需確認公開範圍 | [recorded JSON](results/recorded/reset-p3-full.json)保留作者本機絕對 data/fold 路徑及 well-ID lists；[fold config](configs/folds/main_5fold.csv)含競賽井標識。確認可公開範圍與資料權利；本次未刪除或修改原紀錄。 |
| P1 | 歷史外部備份不是 clone 內容 | [README 備份說明](README.md#歷史與備份)指向本機 legacy bundle，且明載 ignored artifacts 不在 bundle。需交代哪些舊路線可恢復，不以歷史摘要宣稱所有舊程式與證據完整附帶。 |

## 可宣稱與不能宣稱

- 可宣稱：repo 實作 Flat/P3、well-level CV、pooled row-RMSE、sample-ID 對齊與 inference-safe feature contract，來源與相關測試可查；也保存對 target-bearing diagnostics 的限制。
- 可引用歷史：Flat `15.9098528707`、P3 `15.4525482195`，各為 773 wells、3,783,989 OOF rows；[run records](results/recorded_runs.csv)支持這些紀錄。本次未重算，也不是 Kaggle leaderboard 分數。
- 不能宣稱：本次 clone 已復現 full CV、P3 已證明穩健優於 Flat、外部 CSV 合規／可用於 ensemble 已確認、或第三方方法均為本人原創。
- [SLP-E0/E1 protocol](audit/stratigraphic_program_slam_experiment_protocol.md)中的 oracle／loop feasibility 不能替代可部署模型的 test prediction 證據。

## 本次驗證範圍

核對 Git 初始狀態、pyproject entrypoint 與 CLI flags、model registry、代表性模型／runner 來源、JSON/CSV summary 及缺失目錄。安裝與研究命令僅 source checked；本地 bundled Python 未具 `sklearn`／`pytest`／`torch`，未為文件整理安裝研究環境或執行 CLI/CV/tests。另檢查 Markdown 本地連結與 diff 空白。這不是模型執行驗證、資料洩漏全量審計或獨立重現。
