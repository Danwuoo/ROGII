"""Open, well-isolated CV runner and local submission generator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import pickle
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import (
    WellFiles,
    enumerate_wells,
    load_horizontal_inference,
    load_typewell_inference,
    validate_prefix,
)
from .folds import FOLD_COUNT, load_fold_mapping, partition_well_ids, stable_select
from .metrics import per_well_diagnostics, pooled_row_rmse, tail_diagnostics
from .models import MODEL_IDS, get_model
from .models.base import InferenceWell, RunContext, TrainingWell
from .resources import configure_cpu
from .submission import align_predictions_to_sample, validate_sample_submission_contract


DEFAULT_SEED = 20260711
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_target(well: WellFiles, row_index: np.ndarray) -> np.ndarray:
    labels = pd.read_csv(well.horizontal_path, usecols=["TVT", "TVT_input"])
    known_rows, _ = validate_prefix(labels["TVT_input"], target=labels["TVT"])
    expected = np.arange(known_rows, len(labels), dtype=np.int64)
    if not np.array_equal(row_index, expected):
        raise ValueError(f"Prediction-row label coverage mismatch: {well.well_id}")
    target = labels["TVT"].iloc[row_index].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError(f"Non-finite prediction target: {well.well_id}")
    return target


def _parse_folds(folds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted(set(map(int, folds))))
    if not values or any(fold not in range(FOLD_COUNT) for fold in values):
        raise ValueError("folds must be a non-empty subset of 0..4")
    return values


def _safe_run_root(output_root: Path, run_id: str, data_root: Path) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id may contain only letters, numbers, dot, underscore, and dash")
    output = Path(output_root).resolve()
    data = Path(data_root).resolve()
    run_root = output / run_id
    if output == data or data in output.parents or run_root == data or data in run_root.parents:
        raise ValueError("Run output cannot be inside competition data")
    if run_root.exists():
        raise FileExistsError(f"Run already exists: {run_root}")
    run_root.mkdir(parents=True)
    return run_root


def check_data(*, data_root: Path, fold_config: Path) -> dict[str, Any]:
    data = Path(data_root).resolve()
    train = enumerate_wells(data, "train")
    test = enumerate_wells(data, "test")
    mapping = load_fold_mapping(
        fold_config, expected_well_ids=[well.well_id for well in train]
    )
    train_rows = 0
    prediction_rows = 0
    for well in train:
        horizontal = load_horizontal_inference(well.horizontal_path)
        load_typewell_inference(well.typewell_path)
        labels = pd.read_csv(well.horizontal_path, usecols=["TVT", "TVT_input"])
        _, predicted = validate_prefix(labels["TVT_input"], target=labels["TVT"])
        train_rows += len(horizontal)
        prediction_rows += predicted
    for well in test:
        horizontal = load_horizontal_inference(well.horizontal_path)
        load_typewell_inference(well.typewell_path)
        validate_prefix(horizontal["TVT_input"])
    sample = pd.read_csv(data / "sample_submission.csv", dtype={"id": "string"})
    validate_sample_submission_contract(sample, data / "test")
    return {
        "data_root": str(data),
        "train_wells": len(train),
        "test_wells": len(test),
        "train_rows": train_rows,
        "train_prediction_rows": prediction_rows,
        "sample_rows": len(sample),
        "fold_version": str(mapping["fold_version"].iat[0]),
        "fold_counts": {
            str(key): int(value)
            for key, value in mapping["fold"].value_counts().sort_index().items()
        },
    }


def _prepare_many(
    model: Any,
    wells: list[WellFiles],
    *,
    split: str,
    cache_root: Path,
    workers: int,
) -> dict[str, InferenceWell]:
    def prepare(well: WellFiles) -> InferenceWell:
        return model.prepare(well, split=split, cache_root=cache_root)

    prepared: dict[str, InferenceWell] = {}
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=f"rogii-{split}"
        ) as executor:
            for count, item in enumerate(executor.map(prepare, wells), start=1):
                prepared[item.well_id] = item
                if count % 25 == 0 or count == len(wells):
                    print(
                        f"prepare model={model.model_id} split={split} "
                        f"wells={count}/{len(wells)}",
                        flush=True,
                    )
    return prepared


def run_experiment(
    *,
    model_id: str,
    profile: str,
    data_root: Path,
    fold_config: Path,
    output_root: Path = Path(".runs"),
    run_id: str | None = None,
    folds: Iterable[int] | None = None,
    max_train_wells: int | None = None,
    max_validation_wells: int | None = None,
    seed: int = DEFAULT_SEED,
    workers: int = 4,
) -> dict[str, Any]:
    if model_id not in MODEL_IDS:
        raise ValueError(f"model_id must be one of {MODEL_IDS}")
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    requested_folds = _parse_folds(
        folds if folds is not None else ((0,) if profile == "quick" else range(FOLD_COUNT))
    )
    train_limit = (
        max_train_wells
        if max_train_wells is not None
        else (100 if profile == "quick" else None)
    )
    validation_limit = (
        max_validation_wells
        if max_validation_wells is not None
        else (20 if profile == "quick" else None)
    )
    configure_cpu(workers=workers)
    model = get_model(model_id)
    data = Path(data_root).resolve()
    identifier = run_id or (
        f"{model_id}-{profile}-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_root = _safe_run_root(output_root, identifier, data)
    cache_root = run_root / "cache"
    model_root = run_root / "models"
    model_root.mkdir()

    train_files = enumerate_wells(data, "train")
    train_by_id = {well.well_id: well for well in train_files}
    mapping = load_fold_mapping(
        fold_config, expected_well_ids=train_by_id.keys()
    )
    selected: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    required_ids: set[str] = set()
    for fold in requested_folds:
        training_pool, validation_pool = partition_well_ids(mapping, fold)
        training_ids = stable_select(
            training_pool, train_limit, seed=seed, salt=f"fold-{fold}-train"
        )
        validation_ids = stable_select(
            validation_pool,
            validation_limit,
            seed=seed,
            salt=f"fold-{fold}-validation",
        )
        selected[fold] = training_ids, validation_ids
        required_ids.update(training_ids)
        required_ids.update(validation_ids)

    required_files = [train_by_id[well_id] for well_id in sorted(required_ids)]
    prepared = _prepare_many(
        model,
        required_files,
        split="train",
        cache_root=cache_root,
        workers=workers,
    )
    targets = {
        well_id: _load_target(train_by_id[well_id], prepared[well_id].row_index)
        for well_id in sorted(required_ids)
    }

    oof_blocks: list[pd.DataFrame] = []
    fold_manifest: dict[str, Any] = {}
    for fold in requested_folds:
        training_ids, validation_ids = selected[fold]
        training = [
            TrainingWell(prepared[well_id], targets[well_id])
            for well_id in training_ids
        ]
        context = RunContext(
            fold=fold,
            seed=seed,
            workers=workers,
            work_dir=model_root / f"fold_{fold}",
        )
        print(
            f"fit model={model_id} fold={fold} train_wells={len(training_ids)} "
            f"validation_wells={len(validation_ids)}",
            flush=True,
        )
        state = model.fit(training, context=context)
        state_path = model_root / f"fold_{fold}.pkl"
        temporary = state_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, state_path)
        validation_blocks: list[pd.DataFrame] = []
        for well_id in validation_ids:
            inference = prepared[well_id]
            prediction = np.asarray(model.predict(state, inference), dtype=np.float64)
            if prediction.shape != targets[well_id].shape or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid validation prediction: {well_id}")
            validation_blocks.append(
                pd.DataFrame(
                    {
                        "well_id": well_id,
                        "row_index": inference.row_index,
                        "fold": np.int8(fold),
                        "y_true": targets[well_id],
                        "y_pred": prediction,
                    }
                )
            )
        fold_frame = pd.concat(validation_blocks, ignore_index=True)
        oof_blocks.append(fold_frame)
        fold_manifest[str(fold)] = {
            "training_wells": list(training_ids),
            "validation_wells": list(validation_ids),
            "validation_rows": len(fold_frame),
            "pooled_row_rmse": pooled_row_rmse(
                fold_frame["y_true"], fold_frame["y_pred"]
            ),
        }
        del state, training, validation_blocks, fold_frame
        gc.collect()

    oof = pd.concat(oof_blocks, ignore_index=True)
    if oof.duplicated(["well_id", "row_index"]).any():
        raise ValueError("OOF contains duplicate well/row predictions")
    if not np.isfinite(oof[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise ValueError("OOF contains non-finite labels or predictions")
    expected_validation = {
        well_id for _, validation_ids in selected.values() for well_id in validation_ids
    }
    if set(oof["well_id"].astype(str)) != expected_validation:
        raise ValueError("OOF validation-well coverage mismatch")
    for well_id, group in oof.groupby("well_id", sort=False):
        expected_rows = prepared[str(well_id)].row_index
        actual_rows = group.sort_values("row_index")["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual_rows, expected_rows):
            raise ValueError(f"OOF row coverage mismatch: {well_id}")

    per_well = per_well_diagnostics(oof)
    per_fold_rows = []
    for fold, group in oof.groupby("fold", sort=True):
        per_fold_rows.append(
            {
                "fold": int(fold),
                "wells": int(group["well_id"].nunique()),
                "rows": len(group),
                "pooled_row_rmse": pooled_row_rmse(
                    group["y_true"], group["y_pred"]
                ),
            }
        )
    per_fold = pd.DataFrame(per_fold_rows)
    full_scope = (
        requested_folds == tuple(range(FOLD_COUNT))
        and train_limit is None
        and validation_limit is None
        and expected_validation == set(train_by_id)
    )
    summary = {
        "run_id": identifier,
        "model_id": model_id,
        "profile_requested": profile,
        "scope": "full" if full_scope else "quick_or_custom",
        "comparable_to_historical_full_oof": full_scope,
        "folds": list(requested_folds),
        "fold_version": str(mapping["fold_version"].iat[0]),
        "seed": seed,
        "workers": workers,
        "oof_wells": int(oof["well_id"].nunique()),
        "oof_rows": len(oof),
        "pooled_row_rmse": pooled_row_rmse(oof["y_true"], oof["y_pred"]),
        "tail": tail_diagnostics(per_well),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    config = {
        "run_id": identifier,
        "model_id": model_id,
        "profile": profile,
        "data_root": str(data),
        "fold_config": str(Path(fold_config).resolve()),
        "folds": list(requested_folds),
        "max_train_wells": train_limit,
        "max_validation_wells": validation_limit,
        "seed": seed,
        "workers": workers,
        "fold_manifest": fold_manifest,
    }
    oof.sort_values(["fold", "well_id", "row_index"], kind="stable").to_csv(
        run_root / "oof_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    per_well.to_csv(run_root / "per_well_metrics.csv", index=False, lineterminator="\n")
    per_fold.to_csv(run_root / "per_fold_metrics.csv", index=False, lineterminator="\n")
    _write_json(run_root / "summary.json", summary)
    _write_json(run_root / "run_config.json", config)
    return {"run_root": run_root, "summary": summary}


def submit_run(
    *,
    run_root: Path,
    data_root: Path,
    output: Path,
    workers: int = 4,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    if summary.get("scope") != "full" or config.get("folds") != list(range(FOLD_COUNT)):
        raise ValueError("Submission requires one complete five-fold run")
    model = get_model(str(config["model_id"]))
    data = Path(data_root).resolve()
    destination = Path(output).resolve()
    if destination == data or data in destination.parents:
        raise ValueError("Submission output cannot be inside competition data")
    configure_cpu(workers=workers)
    test_files = list(enumerate_wells(data, "test"))
    prepared = _prepare_many(
        model,
        test_files,
        split="test",
        cache_root=root / "cache",
        workers=workers,
    )
    states = []
    for fold in range(FOLD_COUNT):
        with (root / "models" / f"fold_{fold}.pkl").open("rb") as handle:
            states.append(pickle.load(handle))

    blocks: list[pd.DataFrame] = []
    for well in test_files:
        inference = prepared[well.well_id]
        prediction = np.zeros(len(inference.row_index), dtype=np.float64)
        for state in states:
            prediction += model.predict(state, inference) / FOLD_COUNT
        blocks.append(
            pd.DataFrame(
                {
                    "id": [
                        f"{well.well_id}_{int(index)}"
                        for index in inference.row_index
                    ],
                    "tvt": prediction,
                }
            )
        )
    predictions = pd.concat(blocks, ignore_index=True)
    sample = pd.read_csv(data / "sample_submission.csv", dtype={"id": "string"})
    validate_sample_submission_contract(sample, data / "test")
    submission = align_predictions_to_sample(sample, predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    submission.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, destination)
    return {
        "run_id": summary["run_id"],
        "model_id": summary["model_id"],
        "submission_path": str(destination),
        "submission_rows": len(submission),
        "submission_wells": int(
            submission["id"].str.rsplit("_", n=1).str[0].nunique()
        ),
    }
