from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rogii.external_models.artifacts import (
    AUDITED_INCUMBENT_SOURCE_COLUMN,
    INCUMBENT_COLUMN,
    FoldProvenance,
    OofCandidateSchema,
    canonical_oof_columns,
    capture_data_manifest,
    make_model_artifact,
    sha256_file,
    well_ids_sha256,
    write_external_artifact_bundle,
)
from rogii.external_models.portfolio_reaudit import (
    DEFAULT_OUTPUT_NAME,
    REAUDIT_SCHEMA_VERSION,
    audit_external_model_portfolio,
)
from rogii.portfolio_metrics import pooled_metrics, validate_oof_frame


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _write_audited_incumbent_reference(
    root: Path,
    reference: pd.DataFrame,
) -> Path:
    directory = root / "audited-incumbent"
    directory.mkdir()
    path = directory / "oof_predictions.csv.gz"
    frame = reference.loc[
        :,
        ["well_id", "row_index", "fold", "y_true", INCUMBENT_COLUMN],
    ].rename(columns={INCUMBENT_COLUMN: AUDITED_INCUMBENT_SOURCE_COLUMN})
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "mtime": 0},
    )
    disk = pd.read_csv(
        path,
        dtype={"well_id": "string", "row_index": "int64", "fold": "int64"},
        float_precision="round_trip",
    ).rename(columns={AUDITED_INCUMBENT_SOURCE_COLUMN: INCUMBENT_COLUMN})
    validation = validate_oof_frame(disk, (INCUMBENT_COLUMN,))
    (directory / "independent_reaudit.json").write_text(
        json.dumps(
            {
                "passed": True,
                "candidate_oof_sha256": sha256_file(path),
                "model_state_provenance_passed": True,
                "submission_csv_created": False,
                "disk_oof_nonfinite_values": 0,
                "disk_oof_duplicate_or_nonmonotone_keys": 0,
                "disk_oof_rows": len(disk),
                "disk_oof_key_sha256": validation["key_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _experiment(tmp_path: Path, *, wells: int = 4) -> dict[str, Any]:
    if wells < 4 or wells % 2:
        raise ValueError("test wells must be an even integer of at least four")
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    (data_root / "immutable.csv").write_text("id,value\n1,2\n", encoding="utf-8")
    fold_config = tmp_path / "folds.csv"
    well_ids = tuple(f"well-{index:03d}" for index in range(wells))
    folds_by_well = [0] * (wells // 2) + [1] * (wells // 2)
    pd.DataFrame({"well_id": well_ids, "fold": folds_by_well}).to_csv(
        fold_config, index=False
    )

    validation_zero = well_ids[: wells // 2]
    validation_one = well_ids[wells // 2 :]
    folds = (
        FoldProvenance(
            outer_fold=0,
            training_well_ids=validation_one,
            validation_well_ids=validation_zero,
        ),
        FoldProvenance(
            outer_fold=1,
            training_well_ids=validation_zero,
            validation_well_ids=validation_one,
        ),
    )
    run_root = tmp_path / "run"
    model_root = run_root / "models"
    model_root.mkdir(parents=True)
    models = []
    for fold in folds:
        checkpoint = model_root / f"fold-{fold.outer_fold}.pt"
        checkpoint.write_bytes(f"checkpoint-{fold.outer_fold}".encode("ascii"))
        models.append(
            make_model_artifact(
                run_root,
                checkpoint,
                model_id=f"tcn-fold-{fold.outer_fold}",
                family_id="tcn",
                variant_id="small",
                fit_scope="outer_training",
                fit_well_ids=fold.training_well_ids,
                outer_fold=fold.outer_fold,
                seed=17,
                state_manifest={
                    "family": "tcn",
                    "variant": "small",
                    "seed": 17,
                    "fit_ids_sha256": well_ids_sha256(fold.training_well_ids),
                    "parameter_sha256": str(fold.outer_fold + 1) * 64,
                },
            )
        )
    candidate = OofCandidateSchema.canonical(
        "tcn",
        family_id="tcn",
        variant_id="small",
        transfer_mode="architecture_only",
        source_model_ids=tuple(model.model_id for model in models),
    )

    rows: list[dict[str, Any]] = []
    coordinate = np.array([0.0, 1.0], dtype=np.float64)
    centered = coordinate - coordinate.mean()
    for well_number, (well_id, fold) in enumerate(
        zip(well_ids, folds_by_well, strict=True)
    ):
        incumbent = 100.0 + 0.01 * well_number + np.array([0.0, 1.0], dtype=np.float64)
        true_correction = (
            0.20 + 0.002 * well_number + (0.04 + 0.0001 * well_number) * centered
        )
        truth = incumbent + true_correction
        common = np.full(2, 0.16 + 0.0015 * well_number, dtype=np.float64)
        slope = (0.035 + 0.0001 * well_number) * centered
        shape = np.zeros(2, dtype=np.float64)
        correction = common + slope + shape
        prediction = incumbent + correction
        for row_index in range(2):
            rows.append(
                {
                    "well_id": well_id,
                    "row_index": row_index,
                    "fold": fold,
                    "row_coordinate": coordinate[row_index],
                    "y_true": truth[row_index],
                    INCUMBENT_COLUMN: incumbent[row_index],
                    candidate.common_column: common[row_index],
                    candidate.slope_column: slope[row_index],
                    candidate.shape_column: shape[row_index],
                    candidate.correction_column: correction[row_index],
                    candidate.prediction_column: prediction[row_index],
                }
            )
    oof = pd.DataFrame(rows).loc[:, canonical_oof_columns((candidate,))]
    reference = oof.loc[
        :,
        [
            "well_id",
            "row_index",
            "fold",
            "row_coordinate",
            "y_true",
            INCUMBENT_COLUMN,
        ],
    ].copy()
    incumbent_oof = _write_audited_incumbent_reference(tmp_path, reference)
    incumbent_manifest = {
        "states": [
            {
                "fold": fold.outer_fold,
                "seed": 20260711,
                "sha256": str(fold.outer_fold + 1) * 64,
                "training_validation_intersection": 0,
            }
            for fold in folds
        ]
    }
    write_external_artifact_bundle(
        run_root,
        oof,
        run_id="reaudit-test",
        profile="quick",
        candidates=(candidate,),
        fold_provenance=folds,
        model_artifacts=tuple(models),
        data_root=data_root,
        data_manifest_before=capture_data_manifest(data_root),
        fold_config_path=fold_config,
        incumbent_manifest=incumbent_manifest,
        expected_reference=reference,
        metadata={"test": True},
    )
    summary = {
        "run_id": "reaudit-test",
        "profile": "quick",
        "metrics": {
            "pooled": _json_records(
                pooled_metrics(
                    oof,
                    (INCUMBENT_COLUMN, candidate.prediction_column),
                    baseline_column=INCUMBENT_COLUMN,
                )
            )
        },
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_root": run_root,
        "data_root": data_root,
        "candidate": candidate,
        "models": tuple(models),
        "summary": summary,
        "incumbent_oof": incumbent_oof,
    }


def test_disk_reaudit_recomputes_metrics_compares_summary_and_writes_atomically(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path, wells=200)
    report = audit_external_model_portfolio(
        experiment["run_root"],
        incumbent_oof=experiment["incumbent_oof"],
        bootstrap_samples=40,
        shadow_samples=30,
    )

    output = experiment["run_root"] / DEFAULT_OUTPUT_NAME
    disk = json.loads(output.read_text(encoding="utf-8"))
    assert disk["schema_version"] == REAUDIT_SCHEMA_VERSION
    assert disk["passed"] is True
    assert disk["artifact_independent_reaudit"]["passed"] is True
    assert disk["second_disk_read_audit"]["passed"] is True
    assert disk["scope"] == {
        "candidate_count": 1,
        "eligible_candidate_count": 1,
        "folds": [0, 1],
        "ineligible_candidate_count": 0,
        "rows": 400,
        "wells": 200,
    }
    assert disk["metrics"]["shadow_private"]["performed"] is True
    assert disk["metrics"]["shadow_private"]["samples"] == 30
    assert disk["metrics"]["shadow_private"]["subset_wells"] == 200
    comparison = disk["runner_summary_comparison"]
    assert comparison["passed"] is True
    assert comparison["sections"]["pooled"]["matched_rows"] == 2
    assert comparison["sections"]["pooled"]["values_compared"] == 10
    assert comparison["tolerance"]["absolute_tolerance"] == 5e-9
    assert comparison["tolerance"]["relative_tolerance"] == 5e-11
    assert not list(experiment["run_root"].glob(f".{DEFAULT_OUTPUT_NAME}.*.tmp"))
    assert report["metrics"]["pooled"] == disk["metrics"]["pooled"]


def test_less_than_200_wells_records_shadow_private_skip(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path, wells=4)
    report = audit_external_model_portfolio(
        experiment["run_root"],
        incumbent_oof=experiment["incumbent_oof"],
        bootstrap_samples=20,
        shadow_samples=25,
    )
    shadow = report["metrics"]["shadow_private"]
    assert shadow["performed"] is False
    assert shadow["available_wells"] == 4
    assert shadow["subset_wells"] == 200
    assert "exceeds available_wells=4" in shadow["reason"]


def test_checkpoint_mutation_fails_before_output(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    checkpoint = experiment["run_root"] / experiment["models"][0].path
    checkpoint.write_bytes(b"mutated-checkpoint")

    with pytest.raises(ValueError, match="size differs|SHA-256 differs"):
        audit_external_model_portfolio(
            experiment["run_root"],
            incumbent_oof=experiment["incumbent_oof"],
            bootstrap_samples=10,
            shadow_samples=10,
        )
    assert not (experiment["run_root"] / DEFAULT_OUTPUT_NAME).exists()


def test_reaudit_rejects_bundle_values_that_differ_from_external_lock(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    incumbent_path = experiment["incumbent_oof"]
    incumbent = pd.read_csv(
        incumbent_path,
        dtype={"well_id": "string", "row_index": "int64", "fold": "int64"},
        float_precision="round_trip",
    )
    incumbent.loc[0, "y_true"] += 0.01
    incumbent.to_csv(
        incumbent_path,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "mtime": 0},
    )
    validation = validate_oof_frame(
        incumbent.rename(columns={AUDITED_INCUMBENT_SOURCE_COLUMN: INCUMBENT_COLUMN}),
        (INCUMBENT_COLUMN,),
    )
    reaudit_path = incumbent_path.parent / "independent_reaudit.json"
    reaudit = json.loads(reaudit_path.read_text(encoding="utf-8"))
    reaudit["candidate_oof_sha256"] = sha256_file(incumbent_path)
    reaudit["disk_oof_key_sha256"] = validation["key_sha256"]
    reaudit_path.write_text(
        json.dumps(reaudit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="audited incumbent reference"):
        audit_external_model_portfolio(
            experiment["run_root"],
            incumbent_oof=incumbent_path,
            bootstrap_samples=10,
            shadow_samples=10,
        )
    assert not (experiment["run_root"] / DEFAULT_OUTPUT_NAME).exists()


def test_same_name_runner_metric_mismatch_fails_without_output(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    summary_path = experiment["run_root"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate_column = experiment["candidate"].prediction_column
    candidate = next(
        row
        for row in summary["metrics"]["pooled"]
        if row["candidate"] == candidate_column
    )
    candidate["pooled_row_rmse"] += 0.01
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="summary comparison failed"):
        audit_external_model_portfolio(
            experiment["run_root"],
            incumbent_oof=experiment["incumbent_oof"],
            bootstrap_samples=10,
            shadow_samples=10,
        )
    assert not (experiment["run_root"] / DEFAULT_OUTPUT_NAME).exists()


def test_absent_runner_summary_is_explicitly_recorded(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    (experiment["run_root"] / "summary.json").unlink()

    report = audit_external_model_portfolio(
        experiment["run_root"],
        incumbent_oof=experiment["incumbent_oof"],
        bootstrap_samples=10,
        shadow_samples=10,
    )
    comparison = report["runner_summary_comparison"]
    assert comparison["performed"] is False
    assert comparison["passed"] is True
    assert comparison["reason"] == "runner summary is absent"


def test_cli_help_is_importable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_external_model_portfolio.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-root" in result.stdout
    assert "--incumbent-oof" in result.stdout
    assert "--bootstrap-samples" in result.stdout
