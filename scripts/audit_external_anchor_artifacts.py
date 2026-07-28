from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from rogii.folds import FOLD_COUNT, load_fold_mapping, partition_well_ids
from rogii.models.heel_registered_anchor_student import (
    HeelRegisteredAnchorStudentState,
)


BASELINE_COLUMN = "y_pred_extra_trees_s20260711"
PRIMARY_COLUMN = "y_pred_heel_registered_ensemble"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bootstrap_recompute(
    per_well: pd.DataFrame,
    *,
    baseline_candidate: str = "anchor_student",
    primary_candidate: str = "heel_registered_ensemble",
    samples: int = 10_000,
    seed: int = 20260728,
) -> dict[str, float]:
    baseline = per_well.loc[per_well["candidate"].eq(baseline_candidate)].sort_values(
        "well_id"
    )
    primary = per_well.loc[per_well["candidate"].eq(primary_candidate)].sort_values(
        "well_id"
    )
    if not np.array_equal(
        baseline["well_id"].astype(str).to_numpy(),
        primary["well_id"].astype(str).to_numpy(),
    ):
        raise ValueError("per-well candidate keys differ")
    count = baseline["n_rows"].to_numpy(dtype=np.int64)
    base_sse = baseline["sse"].to_numpy(dtype=np.float64)
    primary_sse = primary["sse"].to_numpy(dtype=np.float64)
    random = np.random.default_rng(seed)
    delta = np.empty(samples, dtype=np.float64)
    offset = 0
    while offset < samples:
        size = min(256, samples - offset)
        indices = random.integers(0, len(count), size=(size, len(count)))
        denominator = np.sum(count[indices], axis=1)
        base = np.sqrt(np.sum(base_sse[indices], axis=1) / denominator)
        candidate = np.sqrt(np.sum(primary_sse[indices], axis=1) / denominator)
        delta[offset : offset + size] = candidate - base
        offset += size
    return {
        "delta_rmse_q2.5": float(np.quantile(delta, 0.025)),
        "delta_rmse_q50": float(np.quantile(delta, 0.50)),
        "delta_rmse_q97.5": float(np.quantile(delta, 0.975)),
    }


def audit(
    run_root: Path,
    baseline_root: Path,
    fold_config: Path,
    *,
    chunk_rows: int = 250_000,
) -> dict[str, object]:
    run_root = run_root.resolve()
    baseline_root = baseline_root.resolve()
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    baseline_oof = baseline_root / "oof_predictions.csv.gz"
    candidate_oof = run_root / "oof_predictions.csv.gz"
    reference_reader = pd.read_csv(
        baseline_oof,
        usecols=["well_id", "row_index", "fold", "y_true", BASELINE_COLUMN],
        dtype={"well_id": "string", "row_index": "int64", "fold": "int8"},
        chunksize=chunk_rows,
    )
    candidate_reader = pd.read_csv(
        candidate_oof,
        usecols=[
            "well_id",
            "row_index",
            "fold",
            "y_true",
            "y_pred_anchor_student",
            PRIMARY_COLUMN,
        ],
        dtype={"well_id": "string", "row_index": "int64", "fold": "int8"},
        chunksize=chunk_rows,
    )
    rows = 0
    base_sse = 0.0
    primary_sse = 0.0
    maximum_truth_delta = 0.0
    maximum_baseline_delta = 0.0
    nonfinite = 0
    key_digest = hashlib.sha256()
    previous_key: tuple[int, str, int] | None = None
    duplicate_or_nonmonotone = 0
    reference_iterator = iter(reference_reader)
    candidate_iterator = iter(candidate_reader)
    while True:
        try:
            reference = next(reference_iterator)
        except StopIteration:
            reference = None
        try:
            candidate = next(candidate_iterator)
        except StopIteration:
            candidate = None
        if reference is None or candidate is None:
            if reference is not None or candidate is not None:
                raise ValueError("baseline and candidate OOF chunk counts differ")
            break
        if len(reference) != len(candidate):
            raise ValueError("baseline and candidate OOF chunk row counts differ")
        for column in ("fold", "well_id", "row_index"):
            left = reference[column].to_numpy()
            right = candidate[column].to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(f"disk OOF key mismatch in {column}")
        truth = candidate["y_true"].to_numpy(dtype=np.float64)
        reference_truth = reference["y_true"].to_numpy(dtype=np.float64)
        base = candidate["y_pred_anchor_student"].to_numpy(dtype=np.float64)
        reference_base = reference[BASELINE_COLUMN].to_numpy(dtype=np.float64)
        primary = candidate[PRIMARY_COLUMN].to_numpy(dtype=np.float64)
        maximum_truth_delta = max(
            maximum_truth_delta, float(np.max(np.abs(truth - reference_truth)))
        )
        maximum_baseline_delta = max(
            maximum_baseline_delta, float(np.max(np.abs(base - reference_base)))
        )
        numeric = np.column_stack([truth, base, primary])
        nonfinite += int(np.size(numeric) - np.isfinite(numeric).sum())
        base_sse += float(np.sum((base - truth) ** 2))
        primary_sse += float(np.sum((primary - truth) ** 2))
        for fold, well_id, row_index in zip(
            candidate["fold"].to_numpy(dtype=np.int8),
            candidate["well_id"].astype(str).to_numpy(),
            candidate["row_index"].to_numpy(dtype=np.int64),
            strict=True,
        ):
            key = (int(fold), str(well_id), int(row_index))
            encoded = f"{key[0]}\0{key[1]}\0{key[2]}\n".encode()
            key_digest.update(encoded)
            if previous_key is not None and key <= previous_key:
                duplicate_or_nonmonotone += 1
            previous_key = key
        rows += len(candidate)

    base_rmse = math.sqrt(base_sse / rows)
    primary_rmse = math.sqrt(primary_sse / rows)
    per_well = pd.read_csv(run_root / "per_well_metrics.csv")
    bootstrap_payload = json.loads(
        (run_root / "paired_bootstrap.json").read_text(encoding="utf-8")
    )
    incumbent_bootstrap = bootstrap_payload.get(
        "vs_historical_anchor_student", bootstrap_payload
    )
    bootstrap = incumbent_bootstrap["candidates"]["heel_registered_ensemble"]
    bootstrap_recomputed = _bootstrap_recompute(per_well)
    bootstrap_maximum_delta = max(
        abs(float(bootstrap[key]) - bootstrap_recomputed[key])
        for key in bootstrap_recomputed
    )
    matched_bootstrap = bootstrap_payload["matched_registration_ensemble"][
        "candidates"
    ]["heel_registered_ensemble"]
    matched_bootstrap_recomputed = _bootstrap_recompute(
        per_well,
        baseline_candidate="anchor_seed_ensemble",
        primary_candidate="heel_registered_ensemble",
    )
    matched_bootstrap_maximum_delta = max(
        abs(float(matched_bootstrap[key]) - matched_bootstrap_recomputed[key])
        for key in matched_bootstrap_recomputed
    )

    mapping = load_fold_mapping(
        fold_config,
        expected_well_ids=per_well.loc[
            per_well["candidate"].eq("anchor_student"), "well_id"
        ].astype(str),
    )
    state_rows: list[dict[str, object]] = []
    provenance_ok = True
    for seed in summary["seeds"]:
        for fold in range(FOLD_COUNT):
            path = run_root / "models" / f"seed_{seed}" / f"fold_{fold}.pkl"
            with path.open("rb") as handle:
                state = pickle.load(handle)
            if not isinstance(state, HeelRegisteredAnchorStudentState):
                raise TypeError(f"unexpected state type: {path}")
            expected_training, expected_validation = partition_well_ids(mapping, fold)
            current_ok = bool(
                state.fold == fold
                and state.random_seed == int(seed)
                and state.training_well_ids == expected_training
                and not set(state.training_well_ids) & set(expected_validation)
            )
            provenance_ok &= current_ok
            state_rows.append(
                {
                    "seed": int(seed),
                    "fold": fold,
                    "training_wells": len(state.training_well_ids),
                    "provenance_ok": current_ok,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

    test_audit = json.loads(
        (run_root / "test_inference_audit.json").read_text(encoding="utf-8")
    )
    result: dict[str, object] = {
        "passed": bool(
            rows == 3_783_989
            and nonfinite == 0
            and duplicate_or_nonmonotone == 0
            and maximum_truth_delta == 0.0
            and maximum_baseline_delta <= 1e-9
            and abs(base_rmse - summary["metrics"]["anchor_student"]) <= 1e-11
            and abs(primary_rmse - summary["metrics"]["heel_registered_ensemble"])
            <= 1e-11
            and bootstrap_maximum_delta <= 1e-12
            and matched_bootstrap_maximum_delta <= 1e-12
            and provenance_ok
            and test_audit["finite"]
            and test_audit["sample_id_order_aligned"]
            and not list(run_root.glob("*submission*.csv"))
        ),
        "disk_oof_rows": rows,
        "disk_oof_key_sha256": key_digest.hexdigest(),
        "disk_oof_duplicate_or_nonmonotone_keys": duplicate_or_nonmonotone,
        "disk_oof_nonfinite_values": nonfinite,
        "maximum_truth_delta_vs_baseline_artifact": maximum_truth_delta,
        "maximum_anchor_student_delta_vs_baseline_artifact": maximum_baseline_delta,
        "base_sse": base_sse,
        "base_rmse": base_rmse,
        "primary_sse": primary_sse,
        "primary_rmse": primary_rmse,
        "summary_primary_rmse_absolute_delta": abs(
            primary_rmse - summary["metrics"]["heel_registered_ensemble"]
        ),
        "bootstrap_recomputed": bootstrap_recomputed,
        "bootstrap_maximum_absolute_delta": bootstrap_maximum_delta,
        "matched_bootstrap_recomputed": matched_bootstrap_recomputed,
        "matched_bootstrap_maximum_absolute_delta": matched_bootstrap_maximum_delta,
        "model_state_provenance_passed": provenance_ok,
        "model_states": state_rows,
        "test_inference_audit": test_audit,
        "candidate_oof_sha256": _sha256(candidate_oof),
        "baseline_oof_sha256": _sha256(baseline_oof),
        "submission_csv_created": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--fold-config", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    args = parser.parse_args()
    result = audit(
        args.run_root,
        args.baseline_root,
        args.fold_config,
        chunk_rows=args.chunk_rows,
    )
    output = args.run_root.resolve() / "independent_reaudit.json"
    _atomic_json(output, result)
    artifact_path = args.run_root.resolve() / "artifact_audit.json"
    if result["passed"] and artifact_path.is_file():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["full_disk_reaudit_required"] = False
        artifact["independent_disk_reaudit_passed"] = True
        artifact["independent_disk_reaudit_path"] = str(output)
        _atomic_json(artifact_path, artifact)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
