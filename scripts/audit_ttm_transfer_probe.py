"""Independent disk re-audit for TTM smoke, quick, or full artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rogii.external_models.artifacts import capture_data_manifest
from rogii.external_models.ttm import sha256_file, verify_ttm_checkpoint
from rogii.external_models.ttm_audit import evaluate_ttm_oof, locked_quick_gate
from rogii.external_models.ttm_embeddings import load_head_checkpoint
from rogii.portfolio_metrics import per_fold_metrics, validate_oof_frame


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid audit JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"audit JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _audit_smoke(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    summary_path = root / "smoke_summary.json"
    summary = _json(summary_path)
    if summary.get("passed") is not True:
        raise ValueError("TTM smoke summary did not pass")
    identity = verify_ttm_checkpoint(
        Path(config["weight_identity"]["weight_path"]).parent
    )
    data_manifest = capture_data_manifest(config["data_root"])
    if data_manifest != config["data_manifest"]:
        raise ValueError("current data manifest differs from TTM smoke")
    checkpoint = root / "checkpoints" / "resume.pt"
    payload = load_head_checkpoint(checkpoint)
    if payload.get("completed") is not True or int(payload.get("epoch", -1)) != 2:
        raise ValueError("TTM smoke resume checkpoint is incomplete")
    return {
        "schema_version": "ttm_independent_audit_v1",
        "profile": "smoke",
        "passed": True,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "weight_identity": identity,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "resume_exact": summary["resume_exact_maximum_absolute_delta"] == 0.0,
        "offline_network_forbidden": (
            summary.get("network_forbidden") is True
            if summary.get("offline") is True
            else None
        ),
        "data_manifest_equal": True,
        "submission_created": False,
    }


def _baseline_reference(config: Mapping[str, Any]) -> pd.DataFrame:
    path = Path(str(config["incumbent_oof"]))
    frame = pd.read_csv(
        path,
        usecols=[
            "well_id",
            "row_index",
            "fold",
            "y_true",
            "y_pred_heel_registered_ensemble",
        ],
        dtype={"well_id": "string", "row_index": "int64", "fold": "int8"},
        float_precision="round_trip",
    ).rename(
        columns={
            "y_pred_heel_registered_ensemble": "y_pred_incumbent",
        }
    )
    validation_ids = {
        str(well_id)
        for record in config["cohort"].values()
        for well_id in record["validation_well_ids"]
    }
    return (
        frame.loc[frame["well_id"].astype(str).isin(validation_ids)]
        .sort_values(["fold", "well_id", "row_index"], kind="stable")
        .reset_index(drop=True)
    )


def _audit_experiment(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    profile = str(config["profile"])
    artifact = _json(root / "artifact_audit.json")
    model_manifest = _json(root / "model_manifest.json")
    summary = _json(root / "summary.json")
    locked_path = Path(str(config["locked_config"]))
    locked = _json(locked_path)
    if sha256_file(locked_path) != config["locked_config_sha256"]:
        raise ValueError("locked TTM config hash differs")
    identity = verify_ttm_checkpoint(config["weight_dir"])
    if identity != config["weight_identity"]:
        raise ValueError("TTM weight identity differs from run config")
    current_data = capture_data_manifest(config["data_root"])
    if current_data != config["data_manifest"]:
        raise ValueError("current competition data differs from TTM run")
    oof_path = root / "oof_predictions.csv.gz"
    if sha256_file(oof_path) != artifact["oof_sha256"]:
        raise ValueError("TTM OOF hash differs from artifact audit")
    oof = pd.read_csv(
        oof_path,
        dtype={"well_id": "string", "row_index": "int64", "fold": "int8"},
        float_precision="round_trip",
    )
    baseline = _baseline_reference(config)
    prediction_columns = [
        "y_pred_incumbent",
        *[
            f"y_pred__{variant}"
            for variant in config["variants"]
        ],
        *[
            f"y_pred__{variant}__seed_{seed}"
            for variant in config["variants"]
            for seed in config["seeds"]
        ],
    ]
    validation = validate_oof_frame(
        oof,
        prediction_columns,
        expected_keys=baseline,
        expected_well_ids=sorted(baseline["well_id"].astype(str).unique()),
        expected_folds=range(5),
    )
    joined = (
        oof[
            [
                "well_id",
                "row_index",
                "fold",
                "y_true",
                "y_pred_incumbent",
            ]
        ]
        .merge(
            baseline,
            on=["well_id", "row_index", "fold"],
            how="inner",
            validate="one_to_one",
            suffixes=("_actual", "_reference"),
        )
    )
    truth_delta = float(
        np.max(
            np.abs(
                joined["y_true_actual"].to_numpy(dtype=np.float64)
                - joined["y_true_reference"].to_numpy(dtype=np.float64)
            )
        )
    )
    incumbent_delta = float(
        np.max(
            np.abs(
                joined["y_pred_incumbent_actual"].to_numpy(dtype=np.float64)
                - joined["y_pred_incumbent_reference"].to_numpy(dtype=np.float64)
            )
        )
    )
    if truth_delta > 0.0 or incumbent_delta > 1.0e-9:
        raise ValueError("TTM OOF truth/incumbent differs from audited reference")
    models = model_manifest["models"]
    expected_model_count = (
        len(config["variants"]) * len(config["seeds"]) * 5
    )
    if len(models) != expected_model_count:
        raise ValueError("TTM checkpoint count differs")
    counts: dict[str, set[int]] = {}
    for record in models:
        checkpoint_path = Path(str(record["checkpoint"]))
        block_path = Path(str(record["prediction_block"]))
        if (
            not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != record["checkpoint_sha256"]
            or not block_path.is_file()
            or sha256_file(block_path) != record["prediction_block_sha256"]
        ):
            raise ValueError("TTM checkpoint/prediction block hash differs")
        checkpoint = load_head_checkpoint(checkpoint_path)
        provenance = checkpoint["provenance"]
        if (
            checkpoint.get("completed") is not True
            or set(provenance["fit_ids"]) & set(provenance["validation_ids"])
            or provenance["fit_ids"] != provenance["scaler_fit_ids"]
            or provenance["outer_validation_labels_used_for_fit"] is not False
        ):
            raise ValueError("TTM checkpoint fold-safety provenance failed")
        counts.setdefault(str(record["variant_id"]), set()).add(
            int(record["trainable_parameter_count"])
        )
    if counts["ttm_random_frozen"] != counts["ttm_pretrained_frozen"] or len(
        counts["ttm_random_frozen"]
    ) != 1:
        raise ValueError("matched random/pretrained trainable counts differ")
    metrics = evaluate_ttm_oof(
        oof,
        variants=tuple(config["variants"]),
        seeds=tuple(config["seeds"]),
        bootstrap_samples=int(config["bootstrap_samples"]),
    )
    if metrics["summary"] != summary["metrics"]:
        raise ValueError("independently recomputed TTM metrics differ from summary")
    seed_columns = [
        f"y_pred__{variant}__seed_{seed}"
        for variant in config["variants"]
        for seed in config["seeds"]
    ]
    per_seed_fold = per_fold_metrics(
        oof,
        ["y_pred_incumbent", *seed_columns],
        baseline_column="y_pred_incumbent",
    )
    per_seed_fold_path = root / "metrics" / "per_seed_fold.csv"
    _write_csv(per_seed_fold_path, per_seed_fold)
    gate_match = None
    recomputed_gate = None
    if profile == "quick":
        safety = dict(artifact["safety_checks"])
        safety.update(
            {
                "finite_exact_oof_keys": True,
                "fold_safety": True,
                "checkpoint_complete": True,
                "matched_trainable_parameter_count": True,
                "weight_identity": True,
                "data_manifest": True,
                "no_submission": True,
            }
        )
        recomputed_gate = locked_quick_gate(
            metrics,
            gate_config=locked["quick_gate"],
            safety_checks=safety,
        )
        disk_gate = _json(root / "quick_gate.json")
        gate_match = recomputed_gate == disk_gate == summary["quick_gate"]
        if not gate_match:
            raise ValueError("independently recomputed TTM quick gate differs")
    submissions = [
        str(path)
        for path in root.rglob("*.csv")
        if "submission" in path.name.lower()
    ]
    if submissions:
        raise ValueError(f"submission CSVs exist in TTM run: {submissions}")
    return {
        "schema_version": "ttm_independent_audit_v1",
        "profile": profile,
        "passed": True,
        "run_config_sha256": sha256_file(root / "run_config.json"),
        "locked_config_sha256": sha256_file(locked_path),
        "weight_identity": identity,
        "oof_path": str(oof_path),
        "oof_sha256": sha256_file(oof_path),
        "oof_validation": validation,
        "truth_maximum_absolute_delta": truth_delta,
        "incumbent_maximum_absolute_delta": incumbent_delta,
        "checkpoint_count": len(models),
        "checkpoint_and_prediction_hashes_verified": True,
        "fold_safety_verified": True,
        "matched_trainable_parameter_count": next(
            iter(counts["ttm_random_frozen"])
        ),
        "metrics_recomputed_from_disk": True,
        "per_seed_fold_path": str(per_seed_fold_path),
        "per_seed_fold_sha256": sha256_file(per_seed_fold_path),
        "per_seed_fold_records": int(len(per_seed_fold)),
        "quick_gate_recomputed": recomputed_gate,
        "quick_gate_matches_disk": gate_match,
        "data_manifest_equal_before_after_current": True,
        "submission_created": False,
        "submission_uploaded": False,
    }


def audit_run(run_dir: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    config = _json(root / "run_config.json")
    result = (
        _audit_smoke(root, config)
        if config.get("profile") == "smoke"
        else _audit_experiment(root, config)
    )
    destination = (
        root / "independent_audit.json"
        if output_path is None
        else Path(output_path).resolve()
    )
    if root not in destination.parents:
        raise ValueError("independent audit output must stay inside the run directory")
    _write_json(destination, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently re-read and audit TTM artifacts from disk."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            audit_run(args.run_dir, output_path=args.output_path),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
