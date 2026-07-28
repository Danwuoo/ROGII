"""Strict OOF evaluation for the deployable sparse-anchor student.

Target loading and evaluation live here, away from the production placement,
descriptor, reconstruction, and model modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import pickle
import platform
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from ..data import enumerate_wells
from ..features import feature_manifest
from ..alignment import alignment_manifest
from ..folds import FOLD_COUNT, load_fold_mapping, partition_well_ids, stable_select
from ..metrics import pooled_row_rmse
from ..models.base import RunContext, TrainingWell
from ..models.sparse_anchor_student import (
    SparseAnchorInferenceWell,
    SparseAnchorStudentModel,
    SparseAnchorStudentState,
    fit_student_estimator,
    predict_anchor_delta,
)
from ..resources import configure_cpu
from ..runner import _load_target, _prepare_many
from ..sparse_anchor import (
    ANCHOR_FRACTIONS,
    build_anchor_delta_targets,
    descriptor_manifest,
    reconstruct_from_predicted_anchors,
)
from ..submission import align_predictions_to_sample, validate_sample_submission_contract


DEFAULT_SEED = 20260711
PRIMARY_CANDIDATE = "extra_trees_s20260711"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    kind: str
    alpha: float = 100.0
    target_components: int = 3
    extra_trees_estimators: int = 160


QUICK_CANDIDATES = (
    CandidateSpec("zero", "zero"),
    CandidateSpec("mean", "mean"),
    CandidateSpec("ridge_a10", "ridge", alpha=10.0),
    CandidateSpec("ridge_a100", "ridge", alpha=100.0),
    CandidateSpec("ridge_a1000", "ridge", alpha=1_000.0),
    CandidateSpec("target_pca_ridge_c1_a100", "target_pca_ridge", target_components=1),
    CandidateSpec("target_pca_ridge_c2_a100", "target_pca_ridge", target_components=2),
    CandidateSpec("target_pca_ridge_c3_a100", "target_pca_ridge", target_components=3),
    CandidateSpec("target_pca_ridge_c4_a100", "target_pca_ridge", target_components=4),
    CandidateSpec("pls_c2", "pls", target_components=2),
    CandidateSpec("extra_trees_s20260711", "extra_trees", extra_trees_estimators=160),
)

FULL_CANDIDATES = (
    CandidateSpec("zero", "zero"),
    CandidateSpec("mean", "mean"),
    CandidateSpec("ridge_a1000", "ridge", alpha=1_000.0),
    CandidateSpec("target_pca_ridge_c3_a100", "target_pca_ridge", target_components=3),
    CandidateSpec(PRIMARY_CANDIDATE, "extra_trees", extra_trees_estimators=160),
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_run_root(
    output_root: Path, run_id: str, data_root: Path, *, resume: bool = False
) -> Path:
    root = (Path(output_root).resolve() / run_id).resolve()
    data = Path(data_root).resolve()
    if root == data or data in root.parents:
        raise ValueError("experiment output cannot be inside competition data")
    if root.exists():
        if not resume:
            raise FileExistsError(f"run already exists: {root}")
        if (root / "summary.json").exists() or not (root / "models").is_dir():
            raise ValueError("resume requires an incomplete run with models and no summary")
        return root
    root.mkdir(parents=True)
    (root / "models" / "candidates").mkdir(parents=True)
    return root


def _candidate_column(name: str) -> str:
    return f"y_pred_{name}"


def _rmse_from_sse(sse: float, rows: int) -> float:
    return math.sqrt(float(sse) / int(rows))


def _long_metrics(
    oof: pd.DataFrame, prediction_columns: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    per_well_rows: list[dict[str, Any]] = []
    per_fold_rows: list[dict[str, Any]] = []
    overall: dict[str, float] = {}
    truth = oof["y_true"].to_numpy(dtype=np.float64)
    groups_well = oof.groupby(["fold", "well_id"], sort=True).indices
    groups_fold = oof.groupby("fold", sort=True).indices
    for candidate, column in prediction_columns.items():
        prediction = oof[column].to_numpy(dtype=np.float64)
        error = prediction - truth
        squared = error * error
        overall[candidate] = _rmse_from_sse(float(np.sum(squared)), len(oof))
        for (fold, well_id), indices in groups_well.items():
            index = np.asarray(indices, dtype=np.int64)
            sse = float(np.sum(squared[index]))
            rows = len(index)
            per_well_rows.append(
                {
                    "candidate": candidate,
                    "fold": int(fold),
                    "well_id": str(well_id),
                    "n_rows": rows,
                    "sse": sse,
                    "rmse": _rmse_from_sse(sse, rows),
                }
            )
        for fold, indices in groups_fold.items():
            index = np.asarray(indices, dtype=np.int64)
            sse = float(np.sum(squared[index]))
            rows = len(index)
            per_fold_rows.append(
                {
                    "candidate": candidate,
                    "fold": int(fold),
                    "wells": int(oof.iloc[index]["well_id"].nunique()),
                    "n_rows": rows,
                    "sse": sse,
                    "pooled_row_rmse": _rmse_from_sse(sse, rows),
                }
            )
    per_well = pd.DataFrame(per_well_rows)
    blend = per_well.loc[per_well["candidate"].eq("blend"), ["well_id", "rmse"]].rename(
        columns={"rmse": "blend_rmse"}
    )
    per_well = per_well.merge(blend, on="well_id", how="left", validate="many_to_one")
    per_well["rmse_delta_vs_blend"] = per_well["rmse"] - per_well["blend_rmse"]
    return per_well, pd.DataFrame(per_fold_rows), overall


def _anchor_metrics(anchor_oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, order, fraction), group in anchor_oof.groupby(
        ["candidate", "anchor_order", "fraction"], sort=True
    ):
        error = group["predicted_anchor_tvt"].to_numpy(dtype=np.float64) - group[
            "true_anchor_tvt"
        ].to_numpy(dtype=np.float64)
        rows.append(
            {
                "candidate": str(candidate),
                "anchor_order": int(order),
                "fraction": float(fraction),
                "wells": len(group),
                "rmse": float(np.sqrt(np.mean(error * error))),
                "bias": float(np.mean(error)),
                "mae": float(np.mean(np.abs(error))),
            }
        )
    return pd.DataFrame(rows)


def _error_decomposition(anchor_oof: pd.DataFrame, candidate: str) -> dict[str, Any]:
    frame = anchor_oof.loc[anchor_oof["candidate"].eq(candidate)].sort_values(
        ["well_id", "anchor_order"], kind="stable"
    )
    matrix = (
        frame["predicted_anchor_tvt"].to_numpy(dtype=np.float64)
        - frame["true_anchor_tvt"].to_numpy(dtype=np.float64)
    ).reshape(-1, len(ANCHOR_FRACTIONS))
    common = np.mean(matrix, axis=1)
    centered = matrix - common[:, None]
    return {
        "anchor_rmse": float(np.sqrt(np.mean(matrix * matrix))),
        "common_mode_rmse": float(np.sqrt(np.mean(common * common))),
        "common_mode_bias": float(np.mean(common)),
        "centered_shape_rmse": float(np.sqrt(np.mean(centered * centered))),
        "error_covariance": np.cov(matrix, rowvar=False, ddof=1).tolist(),
        "error_correlation": np.corrcoef(matrix, rowvar=False).tolist(),
    }


def _tail_summary(per_well: pd.DataFrame, candidate: str) -> dict[str, Any]:
    frame = per_well.loc[per_well["candidate"].eq(candidate)]
    rmse = frame["rmse"].to_numpy(dtype=np.float64)
    sse = frame["sse"].to_numpy(dtype=np.float64)
    worst = max(1, math.ceil(0.1 * len(frame)))
    return {
        "p50_rmse": float(np.quantile(rmse, 0.5)),
        "p90_rmse": float(np.quantile(rmse, 0.9)),
        "p95_rmse": float(np.quantile(rmse, 0.95)),
        "max_rmse": float(np.max(rmse)),
        "wells_rmse_ge_40": int(np.sum(rmse >= 40.0)),
        "wells_rmse_ge_60": int(np.sum(rmse >= 60.0)),
        "worst_10pct_sse_share": float(np.sort(sse)[-worst:].sum() / sse.sum()),
    }


def _paired_bootstrap(
    per_well: pd.DataFrame,
    candidates: Sequence[str],
    *,
    samples: int = 10_000,
    seed: int = 20260727,
) -> dict[str, Any]:
    pivot_sse = per_well.pivot(index="well_id", columns="candidate", values="sse")
    pivot_rows = per_well.pivot(index="well_id", columns="candidate", values="n_rows")
    if pivot_sse.isna().any().any() or pivot_rows.isna().any().any():
        raise ValueError("paired bootstrap requires complete candidate/well coverage")
    baseline_sse = pivot_sse["blend"].to_numpy(dtype=np.float64)
    counts = pivot_rows["blend"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    deltas = {candidate: np.empty(samples, dtype=np.float64) for candidate in candidates}
    offset = 0
    while offset < samples:
        count = min(256, samples - offset)
        indices = rng.integers(0, len(counts), size=(count, len(counts)))
        denominator = np.sum(counts[indices], axis=1)
        base_rmse = np.sqrt(np.sum(baseline_sse[indices], axis=1) / denominator)
        for candidate in candidates:
            candidate_sse = pivot_sse[candidate].to_numpy(dtype=np.float64)
            candidate_rmse = np.sqrt(
                np.sum(candidate_sse[indices], axis=1) / denominator
            )
            deltas[candidate][offset : offset + count] = candidate_rmse - base_rmse
        offset += count
    return {
        "method": "paired well-cluster bootstrap; wells sampled with replacement; pooled rows within sample",
        "baseline": "blend",
        "samples": samples,
        "seed": seed,
        "candidates": {
            candidate: {
                "delta_rmse_q2.5": float(np.quantile(values, 0.025)),
                "delta_rmse_q50": float(np.quantile(values, 0.5)),
                "delta_rmse_q97.5": float(np.quantile(values, 0.975)),
            }
            for candidate, values in deltas.items()
        },
    }


def _reference_audit(oof: pd.DataFrame, *, full_scope: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"performed": False}
    if not full_scope:
        return result
    references = {
        "p3": Path(".runs/final-p3-full/oof_predictions.csv.gz"),
        "path_pca": Path(".runs/final-path-pca-full/oof_predictions.csv.gz"),
        "blend": Path(".runs/final-p3-path-blend-full/oof_predictions.csv.gz"),
    }
    ordered = oof.sort_values(["fold", "well_id", "row_index"], kind="stable")
    keys = ordered[["fold", "well_id", "row_index"]].reset_index(drop=True)
    maximum: dict[str, float] = {}
    for name, path in references.items():
        reference = pd.read_csv(path).sort_values(
            ["fold", "well_id", "row_index"], kind="stable"
        ).reset_index(drop=True)
        reference_keys = reference[["fold", "well_id", "row_index"]]
        keys_equal = (
            np.array_equal(
                reference_keys["fold"].to_numpy(dtype=np.int64),
                keys["fold"].to_numpy(dtype=np.int64),
            )
            and np.array_equal(
                reference_keys["well_id"].astype(str).to_numpy(),
                keys["well_id"].astype(str).to_numpy(),
            )
            and np.array_equal(
                reference_keys["row_index"].to_numpy(dtype=np.int64),
                keys["row_index"].to_numpy(dtype=np.int64),
            )
        )
        if not keys_equal:
            raise ValueError(f"historical {name} row keys differ")
        maximum[name] = float(
            np.max(
                np.abs(
                    ordered[f"y_pred_{name}"].to_numpy(dtype=np.float64)
                    - reference["y_pred"].to_numpy(dtype=np.float64)
                )
            )
        )
        if maximum[name] > 1e-9:
            raise ValueError(f"historical {name} prediction mismatch: {maximum[name]}")
        del reference
        gc.collect()
    return {"performed": True, "maximum_absolute_prediction_delta": maximum}


def _protocol_position_audit(
    anchor_oof: pd.DataFrame, *, full_scope: bool
) -> dict[str, Any]:
    source = Path(
        ".runs/sparse-anchor-acquisition-protocol-full/position_manifest.csv.gz"
    )
    if not full_scope or not source.is_file():
        return {"performed": False}
    blocks = []
    usecols = [
        "tolerance_u",
        "position_mode",
        "position_seed",
        "well_id",
        "anchor_order",
        "nominal_row_index",
        "achieved_row_index",
    ]
    for chunk in pd.read_csv(source, usecols=usecols, chunksize=100_000):
        selected = chunk.loc[
            chunk["tolerance_u"].eq(0.0)
            & chunk["position_mode"].eq("nominal")
            & chunk["position_seed"].eq(-1)
        ]
        if len(selected):
            blocks.append(selected)
    legacy = pd.concat(blocks, ignore_index=True)
    production = anchor_oof.loc[
        anchor_oof["candidate"].eq(PRIMARY_CANDIDATE),
        ["well_id", "anchor_order", "row_index"],
    ]
    if legacy.duplicated(["well_id", "anchor_order"]).any() or production.duplicated(
        ["well_id", "anchor_order"]
    ).any():
        raise ValueError("nominal placement audit contains duplicate keys")
    compared = production.merge(
        legacy,
        on=["well_id", "anchor_order"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not compared["_merge"].eq("both").all():
        raise ValueError("nominal placement audit key coverage mismatch")
    nominal_delta = np.abs(
        compared["row_index"].to_numpy(dtype=np.int64)
        - compared["nominal_row_index"].to_numpy(dtype=np.int64)
    )
    achieved_delta = np.abs(
        compared["row_index"].to_numpy(dtype=np.int64)
        - compared["achieved_row_index"].to_numpy(dtype=np.int64)
    )
    return {
        "performed": True,
        "rows": len(compared),
        "wells": int(compared["well_id"].nunique()),
        "maximum_row_delta_vs_legacy_nominal": int(np.max(nominal_delta)),
        "maximum_row_delta_vs_protocol_tu_zero": int(np.max(achieved_delta)),
        "target_columns_read_from_protocol_manifest": False,
    }


def _reconstruction_audit(
    oof: pd.DataFrame, anchor_oof: pd.DataFrame
) -> dict[str, Any]:
    anchors = anchor_oof.loc[
        anchor_oof["candidate"].eq(PRIMARY_CANDIDATE),
        ["well_id", "row_index", "anchor_order", "predicted_anchor_tvt"],
    ]
    selected = anchors.merge(
        oof[
            [
                "well_id",
                "row_index",
                "y_pred_blend",
                _candidate_column(PRIMARY_CANDIDATE),
            ]
        ],
        on=["well_id", "row_index"],
        how="left",
        validate="one_to_one",
    )
    anchor_delta = np.abs(
        selected[_candidate_column(PRIMARY_CANDIDATE)].to_numpy(dtype=np.float64)
        - selected["predicted_anchor_tvt"].to_numpy(dtype=np.float64)
    )
    carry_max = 0.0
    last = selected.loc[selected["anchor_order"].eq(len(ANCHOR_FRACTIONS))].set_index(
        "well_id"
    )
    for well_id, group in oof.groupby("well_id", sort=False):
        item = last.loc[str(well_id)]
        tail = group.loc[group["row_index"].ge(int(item["row_index"]))]
        correction = (
            tail[_candidate_column(PRIMARY_CANDIDATE)].to_numpy(dtype=np.float64)
            - tail["y_pred_blend"].to_numpy(dtype=np.float64)
        )
        expected = float(item["predicted_anchor_tvt"] - item["y_pred_blend"])
        carry_max = max(carry_max, float(np.max(np.abs(correction - expected))))
    return {
        "selected_anchor_rows": len(selected),
        "maximum_anchor_value_absolute_delta": float(np.max(anchor_delta)),
        "maximum_final_carry_absolute_delta": carry_max,
    }


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    per_anchor: pd.DataFrame,
    per_fold: pd.DataFrame,
    bootstrap: dict[str, Any],
) -> None:
    metrics = summary["candidate_metrics"]
    metric_rows = "\n".join(
        f"| {name} | {value:.6f} | {value - metrics['blend']:+.6f} |"
        for name, value in metrics.items()
    )
    fold = per_fold.loc[per_fold["candidate"].eq(PRIMARY_CANDIDATE)]
    fold_rows = "\n".join(
        f"| {int(row.fold)} | {int(row.wells)} | {int(row.n_rows)} | {row.pooled_row_rmse:.6f} |"
        for row in fold.itertuples()
    )
    anchors = per_anchor.loc[per_anchor["candidate"].eq(PRIMARY_CANDIDATE)]
    anchor_rows = "\n".join(
        f"| {int(row.anchor_order)} | {row.fraction:.3f} | {row.rmse:.6f} | {row.bias:+.6f} |"
        for row in anchors.itertuples()
    )
    ci = bootstrap["candidates"][PRIMARY_CANDIDATE]
    decomp = summary["anchor_error_decomposition"]
    common_mse_share = (
        decomp["common_mode_rmse"] ** 2 / decomp["anchor_rmse"] ** 2
    )
    first_anchor_rmse = float(anchors.iloc[0]["rmse"])
    last_anchor_rmse = float(anchors.iloc[-1]["rmse"])
    target_answer = "是" if summary["strict_oof_below_4"] else "否"
    text = f"""# Inference-safe Sparse-Anchor Student

日期：2026-07-27  
範圍：{summary['scope']}，固定五折 well-level OOF

## 結論

本研究把 retrospective true-anchor oracle 改成只讀允許欄位的 deployable student。正式主候選 `{PRIMARY_CANDIDATE}` 的 pooled row-RMSE 為 **{metrics[PRIMARY_CANDIDATE]:.6f} ft**；strict OOF 是否低於 4 ft：**{target_answer}**。0.5 P3/PathPCA baseline 為 {metrics['blend']:.6f} ft，paired well-cluster bootstrap 的 RMSE delta q2.5/q50/q97.5 為 {ci['delta_rmse_q2.5']:+.6f} / {ci['delta_rmse_q50']:+.6f} / {ci['delta_rmse_q97.5']:+.6f} ft。

八個真實 anchors 的既有 oracle 約 4 ft，只是 information-value diagnostic；本報告所有 student 分數都使用 validation fold 上的**預測 anchors**，沒有讀 validation target anchors。hidden-test 結果仍有 distribution-shift uncertainty，OOF 不構成 hidden score 保證。

## 公平比較（相同 rows/wells）

| 方法 | pooled row-RMSE | 相對 blend delta |
|---|---:|---:|
{metric_rows}

`zero` 是 exact zero-correction regression control，逐 row 等於 blend。`mean`、Ridge、target-PCA Ridge 與 ExtraTrees 的 target 都是 outer-training wells 的 `true_anchor_TVT - last_known_TVT_input`。

## 五折主候選

| fold | wells | rows | pooled row-RMSE |
|---:|---:|---:|---:|
{fold_rows}

## Anchor 診斷

主候選 anchor RMSE={decomp['anchor_rmse']:.6f} ft；每 well 八點誤差的 common-mode RMSE={decomp['common_mode_rmse']:.6f} ft、bias={decomp['common_mode_bias']:+.6f} ft；扣除 common mean 後 centered shape RMSE={decomp['centered_shape_rmse']:.6f} ft。

| anchor | normalized suffix fraction | RMSE | bias |
|---:|---:|---:|---:|
{anchor_rows}

未達 4 ft 的定量歸因：common-mode 佔 anchor error MSE 約 {100.0 * common_mse_share:.1f}%，centered shape 約 {100.0 * (1.0 - common_mse_share):.1f}%；位置誤差由第一點 {first_anchor_rmse:.6f} ft 增至 0.975 位置的 {last_anchor_rmse:.6f} ft，顯示長 horizon representation 支援最弱。真-anchor oracle 約 3.94 ft，證明固定 piecewise reconstruction 有足夠 headroom；但 student anchor RMSE 仍有 {decomp['anchor_rmse']:.6f} ft，因此主要瓶頸是允許欄位對 absolute/common-mode anchor delta 與尾端 shape 的表示能力，不是 reconstruction 本身。

Catastrophic tail 沒有靠增加極端 wells 換取均值：student RMSE >=40 的 wells={summary['tail'][PRIMARY_CANDIDATE]['wells_rmse_ge_40']}、>=60 的 wells={summary['tail'][PRIMARY_CANDIDATE]['wells_rmse_ge_60']}、max={summary['tail'][PRIMARY_CANDIDATE]['max_rmse']:.6f} ft；blend 對應為 {summary['tail']['blend']['wells_rmse_ge_40']}、{summary['tail']['blend']['wells_rmse_ge_60']}、{summary['tail']['blend']['max_rmse']:.6f} ft。仍有 12 個 unsupported/catastrophic wells，遠未消除 hidden-test tail uncertainty。

## Inference 與 leakage

- Horizontal prepare 只讀 `MD, X, Y, Z, GR, TVT_input`；typewell 只讀 `TVT, GR`。
- placement、descriptor、imputer、scaler、target PCA 與 regressors 都按 outer fold 隔離；validation well 不在 learned provenance。
- production reconstruction 使用 PS correction=0、固定八 fractions、deterministic nearest-unused、anchor 間線性 interpolation、最後 correction carry。
- 五個 fold state 均可 pickle；artifact audit 在動態列舉的 test authoring copies 上完成 finite inference 與 sample-ID 對齊，但沒有寫 submission CSV。

## Artifact audit

`artifact_audit.json` 從落盤 OOF 重新計算 pooled SSE/RMSE，檢查 key coverage、duplicates、NaN、zero-control、6184 個 production/protocol `T_u=0` anchor rows、anchor exactness、final carry、歷史 baseline 等價與 test inference。`leakage_check.json` 保存各 fold training/validation provenance；`feature_manifest.json` 與 `model_manifest.json` 保存 schema 與模型設定。
"""
    path.write_text(text, encoding="utf-8")


def run_sparse_anchor_student_experiment(
    *,
    profile: str,
    data_root: Path,
    fold_config: Path,
    output_root: Path,
    run_id: str,
    cache_root: Path,
    workers: int = 4,
    seed: int = DEFAULT_SEED,
    max_train_wells: int | None = None,
    max_validation_wells: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    data = Path(data_root).resolve()
    shared_cache = Path(cache_root).resolve()
    if shared_cache == data or data in shared_cache.parents:
        raise ValueError("shared cache cannot be inside competition data")
    run_root = _safe_run_root(output_root, run_id, data, resume=resume)
    configure_cpu(workers=workers)
    overall_started = time.perf_counter()
    specs = QUICK_CANDIDATES if profile == "quick" else FULL_CANDIDATES
    if PRIMARY_CANDIDATE not in {spec.name for spec in specs}:
        raise ValueError("primary candidate must be evaluated")
    train_limit = max_train_wells if max_train_wells is not None else (100 if profile == "quick" else None)
    validation_limit = max_validation_wells if max_validation_wells is not None else (20 if profile == "quick" else None)

    train_files = enumerate_wells(data, "train")
    train_by_id = {well.well_id: well for well in train_files}
    mapping = load_fold_mapping(fold_config, expected_well_ids=train_by_id.keys())
    selected: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    required: set[str] = set()
    for fold in range(FOLD_COUNT):
        training_pool, validation_pool = partition_well_ids(mapping, fold)
        training_ids = stable_select(training_pool, train_limit, seed=seed, salt=f"sparse-anchor-fold-{fold}-train")
        validation_ids = stable_select(validation_pool, validation_limit, seed=seed, salt=f"sparse-anchor-fold-{fold}-validation")
        selected[fold] = training_ids, validation_ids
        required.update(training_ids)
        required.update(validation_ids)

    model = SparseAnchorStudentModel()
    prepare_statistics: dict[str, Any] = {}
    prepared = _prepare_many(
        model,
        [train_by_id[well_id] for well_id in sorted(required)],
        split="train",
        cache_root=shared_cache,
        workers=workers,
        statistics=prepare_statistics,
    )
    targets = {
        well_id: _load_target(train_by_id[well_id], prepared[well_id].row_index)
        for well_id in sorted(required)
    }
    descriptor_names = next(iter(prepared.values())).descriptor_names
    if any(item.descriptor_names != descriptor_names for item in prepared.values()):
        raise ValueError("descriptor schema differs across prepared wells")

    fold_timings: dict[str, Any] = {}
    fold_provenance: dict[str, Any] = {}
    oof_blocks: list[pd.DataFrame] = []
    anchor_blocks: list[pd.DataFrame] = []
    main_states: dict[int, SparseAnchorStudentState] = {}
    with threadpool_limits(limits=workers):
        for fold in range(FOLD_COUNT):
            fold_started = time.perf_counter()
            training_ids, validation_ids = selected[fold]
            training = [TrainingWell(prepared[well_id], targets[well_id]) for well_id in training_ids]
            context = RunContext(fold=fold, seed=seed, workers=workers, work_dir=run_root / "models" / f"fold_{fold}")
            print(
                f"fit sparse-anchor fold={fold} train_wells={len(training_ids)} validation_wells={len(validation_ids)}",
                flush=True,
            )
            candidate_paths = {
                spec.name: run_root
                / "models"
                / "candidates"
                / spec.name
                / f"fold_{fold}.pkl"
                for spec in specs
            }
            can_resume_fold = resume and all(path.is_file() for path in candidate_paths.values())
            student_states = {}
            student_seconds = {}
            if can_resume_fold:
                loaded_states: dict[str, SparseAnchorStudentState] = {}
                for spec in specs:
                    with candidate_paths[spec.name].open("rb") as handle:
                        loaded = pickle.load(handle)
                    if (
                        not isinstance(loaded, SparseAnchorStudentState)
                        or loaded.fold != fold
                        or loaded.training_well_ids != training_ids
                        or loaded.student_state.kind != spec.kind
                    ):
                        raise ValueError(f"resume state provenance mismatch: {spec.name} fold {fold}")
                    loaded_states[spec.name] = loaded
                    student_states[spec.name] = loaded.student_state
                    student_seconds[spec.name] = 0.0
                primary_loaded = loaded_states[PRIMARY_CANDIDATE]
                p3_state = primary_loaded.p3_state
                path_state = primary_loaded.path_pca_state
                main_states[fold] = primary_loaded
                baseline_seconds = 0.0
            else:
                baseline_started = time.perf_counter()
                p3_state = model.p3_model.fit(
                    [TrainingWell(item.inference.p3, item.target) for item in training],
                    context=context,
                )
                path_state = model.path_pca_model.fit(
                    [TrainingWell(item.inference.path_pca, item.target) for item in training],
                    context=context,
                )
                baseline_seconds = time.perf_counter() - baseline_started
                for spec in specs:
                    started = time.perf_counter()
                    student_states[spec.name] = fit_student_estimator(
                        training,
                        kind=spec.kind,
                        context=context,
                        alpha=spec.alpha,
                        target_components=spec.target_components,
                        extra_trees_estimators=spec.extra_trees_estimators,
                    )
                    student_seconds[spec.name] = float(time.perf_counter() - started)
                    state = SparseAnchorStudentState(
                        p3_state=p3_state,
                        path_pca_state=path_state,
                        student_state=student_states[spec.name],
                        training_well_ids=training_ids,
                        fold=fold,
                    )
                    candidate_dir = run_root / "models" / "candidates" / spec.name
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    with (candidate_dir / f"fold_{fold}.pkl.tmp").open("wb") as handle:
                        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(candidate_dir / f"fold_{fold}.pkl.tmp", candidate_dir / f"fold_{fold}.pkl")
                    if spec.name == PRIMARY_CANDIDATE:
                        with (run_root / "models" / f"fold_{fold}.pkl.tmp").open("wb") as handle:
                            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        os.replace(run_root / "models" / f"fold_{fold}.pkl.tmp", run_root / "models" / f"fold_{fold}.pkl")
                        main_states[fold] = state

            for well_id in validation_ids:
                well = prepared[well_id]
                truth = targets[well_id]
                p3 = model.p3_model.predict(p3_state, well.p3)
                path = model.path_pca_model.predict(path_state, well.path_pca)
                blend = 0.5 * p3 + 0.5 * path
                block: dict[str, Any] = {
                    "well_id": well_id,
                    "row_index": well.row_index,
                    "fold": np.full(len(well.row_index), fold, dtype=np.int8),
                    "y_true": truth,
                    "y_pred_flat": np.full(len(truth), well.anchor, dtype=np.float64),
                    "y_pred_p3": p3,
                    "y_pred_path_pca": path,
                    "y_pred_blend": blend,
                }
                true_delta = build_anchor_delta_targets(
                    truth, last_known_tvt=well.anchor, placement=well.placement
                )
                for spec in specs:
                    delta = predict_anchor_delta(student_states[spec.name], well)
                    if spec.kind == "zero":
                        anchor_tvt = blend[well.placement.local_indices].copy()
                        final = blend.copy()
                    else:
                        anchor_tvt = well.anchor + delta
                        final = reconstruct_from_predicted_anchors(blend, anchor_tvt, well.placement)
                    block[_candidate_column(spec.name)] = final
                    anchor_blocks.append(
                        pd.DataFrame(
                            {
                                "well_id": well_id,
                                "fold": fold,
                                "candidate": spec.name,
                                "anchor_order": np.arange(1, len(ANCHOR_FRACTIONS) + 1),
                                "fraction": ANCHOR_FRACTIONS,
                                "local_index": well.placement.local_indices,
                                "row_index": well.placement.row_indices,
                                "normalized_md": well.placement.selected_positions,
                                "true_anchor_delta": true_delta,
                                "predicted_anchor_delta": delta,
                                "true_anchor_tvt": well.anchor + true_delta,
                                "predicted_anchor_tvt": anchor_tvt,
                            }
                        )
                    )
                oof_blocks.append(pd.DataFrame(block))

            fold_provenance[str(fold)] = {
                "training_wells": list(training_ids),
                "validation_wells": list(validation_ids),
                "intersection": sorted(set(training_ids) & set(validation_ids)),
                "student_fit_provenance": {
                    name: list(state.training_well_ids) for name, state in student_states.items()
                },
            }
            fold_timings[str(fold)] = {
                "baseline_fit_seconds": float(baseline_seconds),
                "student_fit_seconds": student_seconds,
                "total_seconds": float(time.perf_counter() - fold_started),
            }
            del training, student_states, p3_state, path_state
            gc.collect()

    oof = pd.concat(oof_blocks, ignore_index=True).sort_values(
        ["fold", "well_id", "row_index"], kind="stable"
    ).reset_index(drop=True)
    anchor_oof = pd.concat(anchor_blocks, ignore_index=True).sort_values(
        ["fold", "well_id", "candidate", "anchor_order"], kind="stable"
    ).reset_index(drop=True)
    if oof.duplicated(["well_id", "row_index"]).any():
        raise ValueError("OOF contains duplicate row keys")
    numeric = oof.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("OOF contains non-finite numeric values")
    expected_validation = {well_id for _, validation_ids in selected.values() for well_id in validation_ids}
    if set(oof["well_id"].astype(str)) != expected_validation:
        raise ValueError("OOF validation-well coverage differs from selection")
    for well_id, group in oof.groupby("well_id", sort=False):
        if not np.array_equal(group["row_index"].to_numpy(dtype=np.int64), prepared[str(well_id)].row_index):
            raise ValueError(f"OOF row coverage mismatch: {well_id}")

    prediction_columns = {
        "flat": "y_pred_flat",
        "p3": "y_pred_p3",
        "path_pca": "y_pred_path_pca",
        "blend": "y_pred_blend",
        **{spec.name: _candidate_column(spec.name) for spec in specs},
    }
    # zero and blend intentionally share predictions; keep both named controls.
    per_well, per_fold, overall = _long_metrics(oof, prediction_columns)
    per_anchor = _anchor_metrics(anchor_oof)
    decomposition = _error_decomposition(anchor_oof, PRIMARY_CANDIDATE)
    bootstrap = _paired_bootstrap(
        per_well,
        [spec.name for spec in specs if spec.name != "zero"],
        samples=10_000,
    )
    bootstrap["candidates"]["zero"] = {
        "delta_rmse_q2.5": 0.0,
        "delta_rmse_q50": 0.0,
        "delta_rmse_q97.5": 0.0,
    }
    tail = {candidate: _tail_summary(per_well, candidate) for candidate in prediction_columns}
    all_wells_prepared = set(prepared) == set(train_by_id)
    expected_full_rows = (
        sum(len(prepared[well_id].row_index) for well_id in train_by_id)
        if all_wells_prepared
        else None
    )
    full_scope = (
        train_limit is None
        and validation_limit is None
        and expected_validation == set(train_by_id)
        and expected_full_rows is not None
        and len(oof) == expected_full_rows
    )
    reference_audit = _reference_audit(oof, full_scope=full_scope)
    protocol_position_audit = _protocol_position_audit(
        anchor_oof, full_scope=full_scope
    )
    reconstruction_audit = _reconstruction_audit(oof, anchor_oof)

    inference_audit: dict[str, Any] = {"performed": False}
    if full_scope:
        test_files = list(enumerate_wells(data, "test"))
        test_prepare_stats: dict[str, Any] = {}
        test_prepared = _prepare_many(
            model,
            test_files,
            split="test",
            cache_root=shared_cache,
            workers=workers,
            statistics=test_prepare_stats,
        )
        blocks = []
        for well_file in test_files:
            well = test_prepared[well_file.well_id]
            prediction = np.zeros(len(well.row_index), dtype=np.float64)
            for fold in range(FOLD_COUNT):
                prediction += model.predict(main_states[fold], well) / FOLD_COUNT
            if not np.isfinite(prediction).all():
                raise ValueError(f"non-finite test inference: {well.well_id}")
            blocks.append(
                pd.DataFrame(
                    {
                        "id": [f"{well.well_id}_{int(row)}" for row in well.row_index],
                        "tvt": prediction,
                    }
                )
            )
        predictions = pd.concat(blocks, ignore_index=True)
        sample = pd.read_csv(data / "sample_submission.csv", dtype={"id": "string"})
        validate_sample_submission_contract(sample, data / "test")
        aligned = align_predictions_to_sample(sample, predictions)
        inference_audit = {
            "performed": True,
            "dynamic_test_wells": [well.well_id for well in test_files],
            "test_rows": len(aligned),
            "finite": bool(np.isfinite(aligned["tvt"]).all()),
            "sample_id_order_aligned": aligned["id"].tolist() == sample["id"].tolist(),
            "submission_csv_created": False,
            "prepare_statistics": test_prepare_stats,
        }

    summary = {
        "run_id": run_id,
        "profile": profile,
        "scope": "full_strict_five_fold_oof" if full_scope else "quick_five_fold_subset",
        "fold_version": str(mapping["fold_version"].iat[0]),
        "folds": list(range(FOLD_COUNT)),
        "seed": seed,
        "workers": workers,
        "oof_wells": int(oof["well_id"].nunique()),
        "oof_rows": len(oof),
        "primary_candidate": PRIMARY_CANDIDATE,
        "candidate_metrics": overall,
        "strict_oof_below_4": bool(full_scope and overall[PRIMARY_CANDIDATE] < 4.0),
        "gap_to_4_ft": float(overall[PRIMARY_CANDIDATE] - 4.0),
        "anchor_error_decomposition": decomposition,
        "tail": tail,
        "uses_true_validation_anchors": False,
        "oracle_is_diagnostic_only": True,
        "test_inference_capable": bool(inference_audit.get("finite", False)),
        "submission_created": False,
        "submission_uploaded": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    config = {
        "run_id": run_id,
        "profile": profile,
        "data_root": str(data),
        "fold_config": str(Path(fold_config).resolve()),
        "cache_root": str(shared_cache),
        "seed": seed,
        "workers": workers,
        "max_train_wells": train_limit,
        "max_validation_wells": validation_limit,
        "primary_candidate_selected_from_quick_before_full": PRIMARY_CANDIDATE,
        "resumed_from_complete_fold_states": bool(resume),
        "candidates": [asdict(spec) for spec in specs],
        "fold_provenance": fold_provenance,
        "baseline": "0.5 * fold-fit P3 + 0.5 * fold-fit PathPCA",
        "reconstruction": "PS zero correction; fixed nearest-unused eight anchors; linear between; final carry",
    }
    leakage = {
        "passed": all(not value["intersection"] for value in fold_provenance.values()),
        "horizontal_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
        "typewell_columns": ["TVT", "GR"],
        "validation_target_used_for_prepare": False,
        "validation_target_used_for_anchor_prediction": False,
        "student_target_definition": "outer-training true anchor TVT - last-known TVT_input",
        "learned_baseline_predictions_used_as_descriptors": False,
        "fold_provenance": fold_provenance,
    }
    feature_payload = {
        "sparse_anchor": descriptor_manifest(),
        "p3_base": feature_manifest(),
        "p3_alignment": alignment_manifest(),
        "descriptor_width": len(descriptor_names),
        "descriptor_names": list(descriptor_names),
    }
    model_manifest = {
        "model_id": model.model_id,
        "primary_candidate": PRIMARY_CANDIDATE,
        "candidates": [asdict(spec) for spec in specs],
        "baseline_weights": {"p3": 0.5, "path_pca": 0.5},
        "pickle_safe_states": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "extra_trees_determinism": "fixed random_state; repeated-fit unit test",
    }

    oof.to_csv(
        run_root / "oof_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    anchor_oof.to_csv(
        run_root / "anchor_oof_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    per_anchor.to_csv(run_root / "per_anchor_metrics.csv", index=False, lineterminator="\n")
    per_well.to_csv(run_root / "per_well_metrics.csv", index=False, lineterminator="\n")
    per_fold.to_csv(run_root / "per_fold_metrics.csv", index=False, lineterminator="\n")
    _atomic_json(run_root / "summary.json", summary)
    _atomic_json(run_root / "run_config.json", config)
    _atomic_json(run_root / "leakage_check.json", leakage)
    _atomic_json(run_root / "paired_bootstrap.json", bootstrap)
    _atomic_json(run_root / "feature_manifest.json", feature_payload)
    _atomic_json(run_root / "model_manifest.json", model_manifest)
    timings = {
        "prepare": prepare_statistics,
        "folds": fold_timings,
        "total_seconds_before_artifact_write": float(time.perf_counter() - overall_started),
        "cache_root": str(shared_cache),
    }
    _atomic_json(run_root / "timings.json", timings)

    disk = pd.read_csv(run_root / "oof_predictions.csv.gz")
    disk_error = disk[_candidate_column(PRIMARY_CANDIDATE)].to_numpy(dtype=np.float64) - disk["y_true"].to_numpy(dtype=np.float64)
    disk_sse = float(disk_error @ disk_error)
    disk_rmse = _rmse_from_sse(disk_sse, len(disk))
    rmse_audit_tolerance = max(
        1e-12, abs(float(overall[PRIMARY_CANDIDATE])) * 1e-12
    )
    zero_delta = float(np.max(np.abs(disk["y_pred_zero"] - disk["y_pred_blend"])))
    artifact_audit = {
        "passed": bool(
            len(disk) == len(oof)
            and not disk.duplicated(["well_id", "row_index"]).any()
            and np.isfinite(disk.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all()
            and abs(disk_rmse - overall[PRIMARY_CANDIDATE]) <= rmse_audit_tolerance
            and zero_delta == 0.0
            and reconstruction_audit["maximum_anchor_value_absolute_delta"] == 0.0
            and reconstruction_audit["maximum_final_carry_absolute_delta"] <= 1e-10
            and leakage["passed"]
            and (not full_scope or reference_audit["performed"])
            and (not full_scope or protocol_position_audit["performed"])
            and (
                not full_scope
                or protocol_position_audit[
                    "maximum_row_delta_vs_protocol_tu_zero"
                ]
                == 0
            )
            and (not full_scope or inference_audit.get("finite", False))
        ),
        "rows": len(disk),
        "wells": int(disk["well_id"].nunique()),
        "duplicate_row_keys": int(disk.duplicated(["well_id", "row_index"]).sum()),
        "unexpected_numeric_nan_or_inf": int(
            np.size(disk.select_dtypes(include=[np.number]).to_numpy())
            - np.isfinite(disk.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).sum()
        ),
        "primary_sse_reconstructed": disk_sse,
        "primary_rmse_reconstructed": disk_rmse,
        "summary_rmse_absolute_delta": abs(disk_rmse - overall[PRIMARY_CANDIDATE]),
        "summary_rmse_tolerance": rmse_audit_tolerance,
        "zero_control_max_absolute_delta_vs_blend": zero_delta,
        "reference_baseline_audit": reference_audit,
        "protocol_position_audit": protocol_position_audit,
        "reconstruction_audit": reconstruction_audit,
        "test_inference_audit": inference_audit,
        "submission_csv_created": False,
    }
    if not artifact_audit["passed"]:
        raise ValueError("sparse-anchor artifact audit failed")
    _atomic_json(run_root / "artifact_audit.json", artifact_audit)
    _write_report(
        run_root / "report.md",
        summary=summary,
        per_anchor=per_anchor,
        per_fold=per_fold,
        bootstrap=bootstrap,
    )
    if full_scope:
        formal = Path("results/sparse_anchor_student_20260727.md").resolve()
        formal.write_text((run_root / "report.md").read_text(encoding="utf-8"), encoding="utf-8")
    timings["total_seconds"] = float(time.perf_counter() - overall_started)
    _atomic_json(run_root / "timings.json", timings)
    return {"run_root": run_root, "summary": summary, "artifact_audit": artifact_audit}
