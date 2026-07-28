# Inference-safe Sparse-Anchor Student

日期：2026-07-27  
範圍：full_strict_five_fold_oof，固定五折 well-level OOF

## 結論

本研究把 retrospective true-anchor oracle 改成只讀允許欄位的 deployable student。正式主候選 `extra_trees_s20260711` 的 pooled row-RMSE 為 **14.820080 ft**；strict OOF 是否低於 4 ft：**否**。0.5 P3/PathPCA baseline 為 15.172061 ft，paired well-cluster bootstrap 的 RMSE delta q2.5/q50/q97.5 為 -0.642553 / -0.347749 / -0.052641 ft。

八個真實 anchors 的既有 oracle 約 4 ft，只是 information-value diagnostic；本報告所有 student 分數都使用 validation fold 上的**預測 anchors**，沒有讀 validation target anchors。hidden-test 結果仍有 distribution-shift uncertainty，OOF 不構成 hidden score 保證。

## 公平比較（相同 rows/wells）

| 方法 | pooled row-RMSE | 相對 blend delta |
|---|---:|---:|
| flat | 15.909853 | +0.737791 |
| p3 | 15.452548 | +0.280487 |
| path_pca | 15.364153 | +0.192092 |
| blend | 15.172061 | +0.000000 |
| zero | 15.172061 | +0.000000 |
| mean | 15.813523 | +0.641461 |
| ridge_a1000 | 14.998552 | -0.173510 |
| target_pca_ridge_c3_a100 | 15.864242 | +0.692180 |
| extra_trees_s20260711 | 14.820080 | -0.351981 |

`zero` 是 exact zero-correction regression control，逐 row 等於 blend。`mean`、Ridge、target-PCA Ridge 與 ExtraTrees 的 target 都是 outer-training wells 的 `true_anchor_TVT - last_known_TVT_input`。

## 五折主候選

| fold | wells | rows | pooled row-RMSE |
|---:|---:|---:|---:|
| 0 | 155 | 755585 | 14.569310 |
| 1 | 155 | 760926 | 13.697364 |
| 2 | 155 | 767072 | 14.255198 |
| 3 | 154 | 748127 | 15.984256 |
| 4 | 154 | 752279 | 15.508239 |

## Anchor 診斷

主候選 anchor RMSE=15.280883 ft；每 well 八點誤差的 common-mode RMSE=12.467902 ft、bias=+0.001518 ft；扣除 common mean 後 centered shape RMSE=8.834977 ft。

| anchor | normalized suffix fraction | RMSE | bias |
|---:|---:|---:|---:|
| 1 | 0.125 | 8.988449 | -0.013348 |
| 2 | 0.250 | 12.048554 | -0.018633 |
| 3 | 0.375 | 13.971971 | -0.010885 |
| 4 | 0.500 | 14.699137 | +0.002086 |
| 5 | 0.625 | 15.845254 | +0.030971 |
| 6 | 0.750 | 17.069525 | +0.020455 |
| 7 | 0.875 | 17.848339 | +0.013516 |
| 8 | 0.975 | 19.230152 | -0.012019 |

未達 4 ft 的定量歸因：common-mode 佔 anchor error MSE 約 66.6%，centered shape 約 33.4%；位置誤差由第一點 8.988449 ft 增至 0.975 位置的 19.230152 ft，顯示長 horizon representation 支援最弱。真-anchor oracle 約 3.94 ft，證明固定 piecewise reconstruction 有足夠 headroom；但 student anchor RMSE 仍有 15.280883 ft，因此主要瓶頸是允許欄位對 absolute/common-mode anchor delta 與尾端 shape 的表示能力，不是 reconstruction 本身。

Catastrophic tail 沒有靠增加極端 wells 換取均值：student RMSE >=40 的 wells=12、>=60 的 wells=2、max=63.699058 ft；blend 對應為 12、2、70.726654 ft。仍有 12 個 unsupported/catastrophic wells，遠未消除 hidden-test tail uncertainty。

## Inference 與 leakage

- Horizontal prepare 只讀 `MD, X, Y, Z, GR, TVT_input`；typewell 只讀 `TVT, GR`。
- placement、descriptor、imputer、scaler、target PCA 與 regressors 都按 outer fold 隔離；validation well 不在 learned provenance。
- production reconstruction 使用 PS correction=0、固定八 fractions、deterministic nearest-unused、anchor 間線性 interpolation、最後 correction carry。
- 五個 fold state 均可 pickle；artifact audit 在動態列舉的 test authoring copies 上完成 finite inference 與 sample-ID 對齊，但沒有寫 submission CSV。

## Artifact audit

`artifact_audit.json` 從落盤 OOF 重新計算 pooled SSE/RMSE，檢查 key coverage、duplicates、NaN、zero-control、6184 個 production/protocol `T_u=0` anchor rows、anchor exactness、final carry、歷史 baseline 等價與 test inference。`leakage_check.json` 保存各 fold training/validation provenance；`feature_manifest.json` 與 `model_manifest.json` 保存 schema 與模型設定。
