from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rogii.external_models.artifacts import (
    AUDITED_INCUMBENT_SOURCE_COLUMN,
    DEFAULT_BUNDLE_NAME,
    INCUMBENT_COLUMN,
    MANIFEST_FILENAME,
    MANIFEST_SHA256_FILENAME,
    OOF_FILENAME,
    FoldProvenance,
    OofCandidateSchema,
    canonical_oof_columns,
    capture_data_manifest,
    independent_reaudit_external_artifacts,
    load_audited_incumbent_reference,
    make_model_artifact,
    read_external_artifact_bundle,
    sha256_file,
    well_ids_sha256,
    write_external_artifact_bundle,
)
from rogii.portfolio_metrics import validate_oof_frame


def _experiment(tmp_path: Path) -> dict[str, Any]:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    (data_root / "immutable.csv").write_text("id,value\n1,2\n", encoding="utf-8")
    fold_config = tmp_path / "folds.json"
    fold_config.write_text('{"fold_version":"test-v1"}\n', encoding="utf-8")

    run_root = tmp_path / "run"
    model_root = run_root / "models"
    model_root.mkdir(parents=True)
    folds = (
        FoldProvenance(
            outer_fold=0,
            training_well_ids=("well-2", "well-3"),
            validation_well_ids=("well-0", "well-1"),
        ),
        FoldProvenance(
            outer_fold=1,
            training_well_ids=("well-0", "well-1"),
            validation_well_ids=("well-2", "well-3"),
        ),
    )
    models = []
    for fold in folds:
        path = model_root / f"fold-{fold.outer_fold}.pt"
        path.write_bytes(f"model-state-{fold.outer_fold}".encode())
        fit_ids = fold.training_well_ids
        models.append(
            make_model_artifact(
                run_root,
                path,
                model_id=f"tcn-fold-{fold.outer_fold}",
                family_id="tcn",
                variant_id="small",
                fit_scope="outer_training",
                fit_well_ids=fit_ids,
                outer_fold=fold.outer_fold,
                seed=17,
                state_manifest={
                    "family": "tcn",
                    "variant": "small",
                    "seed": 17,
                    "fit_ids_sha256": well_ids_sha256(fit_ids),
                    "parameter_sha256": "a" * 64,
                },
            )
        )
    candidate = OofCandidateSchema.canonical(
        "tcn-arch-seed17",
        family_id="tcn",
        variant_id="small",
        transfer_mode="architecture_only",
        source_model_ids=tuple(model.model_id for model in models),
    )

    rows: list[dict[str, Any]] = []
    for well_number in range(4):
        well_id = f"well-{well_number}"
        fold = 0 if well_number < 2 else 1
        coordinate = np.array([0.0, 0.4, 1.0], dtype=np.float64)
        common = np.full(3, 0.25 * (well_number + 1), dtype=np.float64)
        slope = 0.4 * (coordinate - coordinate.mean())
        shape = np.zeros(3, dtype=np.float64)
        correction = common + slope + shape
        incumbent = 100.0 + 4.0 * coordinate + well_number
        prediction = incumbent + correction
        truth = incumbent + 0.5 * correction
        for row_index in range(3):
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
    return {
        "data_root": data_root,
        "data_manifest_before": capture_data_manifest(data_root),
        "fold_config": fold_config,
        "run_root": run_root,
        "folds": folds,
        "models": tuple(models),
        "candidate": candidate,
        "oof": oof,
        "reference": reference,
        "incumbent_manifest": incumbent_manifest,
    }


def _write(experiment: dict[str, Any]):
    return write_external_artifact_bundle(
        experiment["run_root"],
        experiment["oof"],
        run_id="artifact-test",
        profile="quick",
        candidates=(experiment["candidate"],),
        fold_provenance=experiment["folds"],
        model_artifacts=experiment["models"],
        data_root=experiment["data_root"],
        data_manifest_before=experiment["data_manifest_before"],
        fold_config_path=experiment["fold_config"],
        incumbent_manifest=experiment["incumbent_manifest"],
        expected_reference=experiment["reference"],
        metadata={
            "sequence_manifest": {"schema_version": "test-sequence"},
            "target_manifest": {"schema_version": "test-target"},
        },
    )


def _legacy_file_order_key_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for well_id, row_index, fold in frame[["well_id", "row_index", "fold"]].itertuples(
        index=False, name=None
    ):
        digest.update(f"{int(fold)}\0{well_id}\0{int(row_index)}\n".encode("utf-8"))
    return digest.hexdigest()


def _write_incumbent_reference_with_key_convention(
    root: Path,
    *,
    key_convention: str,
) -> tuple[Path, Path, str, str]:
    directory = root / "audited-incumbent"
    directory.mkdir(parents=True)
    path = directory / "oof_predictions.csv.gz"
    # Fold-major immutable file order intentionally differs from canonical
    # well_id-major order.
    frame = pd.DataFrame(
        {
            "well_id": ["well-z", "well-z", "well-a", "well-a"],
            "row_index": [0, 1, 0, 1],
            "fold": [0, 0, 1, 1],
            "y_true": [100.0, 101.0, 200.0, 201.0],
            AUDITED_INCUMBENT_SOURCE_COLUMN: [99.5, 100.5, 199.5, 200.5],
        }
    )
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
    )
    normalized = disk.rename(
        columns={AUDITED_INCUMBENT_SOURCE_COLUMN: INCUMBENT_COLUMN}
    )
    canonical_digest = str(
        validate_oof_frame(normalized, (INCUMBENT_COLUMN,))["key_sha256"]
    )
    legacy_digest = _legacy_file_order_key_sha256(disk)
    if canonical_digest == legacy_digest:
        raise AssertionError("test fixture must distinguish key conventions")
    selected_digest = {
        "canonical": canonical_digest,
        "legacy_file_order": legacy_digest,
    }[key_convention]
    reaudit_path = directory / "independent_reaudit.json"
    reaudit_path.write_text(
        json.dumps(
            {
                "passed": True,
                "candidate_oof_sha256": sha256_file(path),
                "model_state_provenance_passed": True,
                "submission_csv_created": False,
                "disk_oof_nonfinite_values": 0,
                "disk_oof_duplicate_or_nonmonotone_keys": 0,
                "disk_oof_rows": len(disk),
                "disk_oof_key_sha256": selected_digest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path, reaudit_path, canonical_digest, legacy_digest


def test_atomic_bundle_roundtrip_and_independent_reaudit(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    written = _write(experiment)
    bundle_root = experiment["run_root"] / DEFAULT_BUNDLE_NAME
    assert written.bundle_root == bundle_root
    assert sorted(path.name for path in bundle_root.iterdir()) == [
        MANIFEST_FILENAME,
        MANIFEST_SHA256_FILENAME,
        OOF_FILENAME,
    ]
    assert not list(experiment["run_root"].glob(f".{DEFAULT_BUNDLE_NAME}.tmp-*"))
    assert written.audit["passed"] is True
    assert written.audit["oof"]["rows"] == 12
    assert written.audit["oof"]["wells"] == 4
    assert written.audit["models"]["checkpoint_sha256_checked"] == 2
    assert all(written.audit["checks"].values())

    loaded = read_external_artifact_bundle(
        experiment["run_root"],
        expected_reference=experiment["reference"],
    )
    pd.testing.assert_frame_equal(loaded.oof, written.oof)
    reaudit = independent_reaudit_external_artifacts(
        experiment["run_root"],
        expected_reference=experiment["reference"],
        data_root=experiment["data_root"],
    )
    assert reaudit["passed"] is True
    assert (
        reaudit["data_read_only_manifest_current"] == experiment["data_manifest_before"]
    )
    with pytest.raises(FileExistsError, match="already exists"):
        _write(experiment)


@pytest.mark.parametrize("failure", ["duplicate", "nonfinite", "algebra"])
def test_invalid_oof_never_commits_bundle(tmp_path: Path, failure: str) -> None:
    experiment = _experiment(tmp_path)
    oof = experiment["oof"].copy()
    candidate = experiment["candidate"]
    if failure == "duplicate":
        oof = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
    elif failure == "nonfinite":
        oof.loc[0, candidate.prediction_column] = np.nan
    else:
        oof.loc[0, candidate.correction_column] += 1.0
    experiment["oof"] = oof
    with pytest.raises(ValueError):
        _write(experiment)
    assert not (experiment["run_root"] / DEFAULT_BUNDLE_NAME).exists()
    assert not list(experiment["run_root"].glob(f".{DEFAULT_BUNDLE_NAME}.tmp-*"))


def test_fold_and_model_fit_provenance_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        FoldProvenance(
            outer_fold=0,
            training_well_ids=("well-a", "well-b"),
            validation_well_ids=("well-b", "well-c"),
        )

    experiment = _experiment(tmp_path)
    bad_path = experiment["run_root"] / "models" / "bad.pt"
    bad_path.write_bytes(b"bad-fit-state")
    bad_model = make_model_artifact(
        experiment["run_root"],
        bad_path,
        model_id="bad-fold-0",
        family_id="tcn",
        variant_id="small",
        fit_scope="outer_training",
        fit_well_ids=("well-0", "well-2"),
        outer_fold=0,
        seed=17,
    )
    candidate = OofCandidateSchema.canonical(
        "tcn-arch-seed17",
        family_id="tcn",
        variant_id="small",
        transfer_mode="architecture_only",
        source_model_ids=(
            bad_model.model_id,
            experiment["models"][1].model_id,
        ),
    )
    experiment["candidate"] = candidate
    experiment["models"] = (bad_model, experiment["models"][1])
    with pytest.raises(ValueError, match="validation wells|exact outer-training"):
        _write(experiment)
    assert not (experiment["run_root"] / DEFAULT_BUNDLE_NAME).exists()


def test_negative_control_is_not_model_selection_eligible() -> None:
    control = OofCandidateSchema.canonical(
        "paired-wrong-typewell",
        family_id="paired_patch_cross",
        variant_id="wrong_typewell",
        transfer_mode="architecture_only",
        control_id="wrong_typewell",
        source_model_ids=("paired-fold-0",),
    )
    assert control.eligible_for_model_selection is False
    with pytest.raises(ValueError, match="negative-control"):
        OofCandidateSchema.canonical(
            "unsafe-control",
            family_id="paired_patch_cross",
            variant_id="wrong_typewell",
            transfer_mode="architecture_only",
            control_id="wrong_typewell",
            source_model_ids=("paired-fold-0",),
            eligible_for_model_selection=True,
        )


def test_reaudit_detects_checkpoint_manifest_and_data_mutation(
    tmp_path: Path,
) -> None:
    checkpoint_experiment = _experiment(tmp_path / "checkpoint")
    _write(checkpoint_experiment)
    checkpoint = (
        checkpoint_experiment["run_root"] / checkpoint_experiment["models"][0].path
    )
    checkpoint.write_bytes(b"mutated checkpoint bytes")
    with pytest.raises(ValueError, match="size differs|SHA-256 differs"):
        independent_reaudit_external_artifacts(checkpoint_experiment["run_root"])

    manifest_experiment = _experiment(tmp_path / "manifest")
    _write(manifest_experiment)
    manifest_path = (
        manifest_experiment["run_root"] / DEFAULT_BUNDLE_NAME / MANIFEST_FILENAME
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        read_external_artifact_bundle(manifest_experiment["run_root"])

    data_experiment = _experiment(tmp_path / "data-change")
    _write(data_experiment)
    (data_experiment["data_root"] / "unexpected.txt").write_text(
        "mutation", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="current data manifest"):
        independent_reaudit_external_artifacts(data_experiment["run_root"])


def test_locked_incumbent_reference_and_no_submission_guard(tmp_path: Path) -> None:
    reference_experiment = _experiment(tmp_path / "reference")
    reference_experiment["oof"].loc[0, INCUMBENT_COLUMN] += 0.01
    with pytest.raises(ValueError, match="locked reference"):
        _write(reference_experiment)

    submission_experiment = _experiment(tmp_path / "submission")
    (submission_experiment["run_root"] / "submission.csv").write_text(
        "id,TVT\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="submission CSV"):
        _write(submission_experiment)
    assert not (submission_experiment["run_root"] / DEFAULT_BUNDLE_NAME).exists()


@pytest.mark.parametrize(
    "key_convention",
    ("canonical", "legacy_file_order"),
)
def test_audited_incumbent_accepts_both_locked_key_conventions(
    tmp_path: Path,
    key_convention: str,
) -> None:
    path, _, canonical_digest, legacy_digest = (
        _write_incumbent_reference_with_key_convention(
            tmp_path,
            key_convention=key_convention,
        )
    )
    loaded, record = load_audited_incumbent_reference(path)
    assert canonical_digest != legacy_digest
    assert loaded["well_id"].astype(str).tolist() == [
        "well-a",
        "well-a",
        "well-z",
        "well-z",
    ]
    assert record["validation"]["key_sha256"] == canonical_digest
    assert record["sha256"] == sha256_file(path)


def test_audited_incumbent_wrong_digest_or_changed_file_fails_closed(
    tmp_path: Path,
) -> None:
    path, reaudit_path, _, legacy_digest = (
        _write_incumbent_reference_with_key_convention(
            tmp_path,
            key_convention="legacy_file_order",
        )
    )
    reaudit = json.loads(reaudit_path.read_text(encoding="utf-8"))
    reaudit["disk_oof_key_sha256"] = "f" * 64
    reaudit_path.write_text(
        json.dumps(reaudit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from its independent re-audit"):
        load_audited_incumbent_reference(path)

    reaudit["disk_oof_key_sha256"] = legacy_digest
    reaudit_path.write_text(
        json.dumps(reaudit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_audited_incumbent_reference(path, expected_sha256="0" * 64)

    changed = pd.read_csv(path, float_precision="round_trip")
    changed.loc[0, AUDITED_INCUMBENT_SOURCE_COLUMN] += 0.25
    changed.to_csv(
        path,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "mtime": 0},
    )
    with pytest.raises(ValueError, match="absent, stale, or failed"):
        load_audited_incumbent_reference(path)
