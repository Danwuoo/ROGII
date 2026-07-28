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
    DEFAULT_BUNDLE_NAME,
    INCUMBENT_COLUMN,
    ROW_COORDINATE_COLUMN,
    capture_data_manifest,
    sha256_file,
    well_ids_sha256,
)
from rogii.external_models.portfolio_publisher import (
    INCUMBENT_MANIFEST_FILENAME,
    MODEL_MANIFEST_FILENAME,
    RUN_CONFIG_FILENAME,
    WIDE_OOF_FILENAME,
    publish_external_portfolio_artifacts,
    reconstruct_row_coordinates,
)
from rogii.external_models.torch_families import (
    FAMILY_PAIRED_PATCH,
    FAMILY_TCN,
    TorchFamilyConfig,
    fit_torch_family,
)
from rogii.portfolio_metrics import validate_oof_frame


SEED = 17
CHANNEL_NAMES = ("horizontal:GR", "typewell:GR")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_audited_incumbent_reference(
    root: Path,
    frame: pd.DataFrame,
) -> Path:
    directory = root / "audited-incumbent"
    directory.mkdir()
    path = directory / "oof_predictions.csv.gz"
    reference = frame.loc[
        :,
        ["well_id", "row_index", "fold", "y_true", INCUMBENT_COLUMN],
    ].rename(columns={INCUMBENT_COLUMN: AUDITED_INCUMBENT_SOURCE_COLUMN})
    reference.to_csv(
        path,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "mtime": 0},
    )
    locked = pd.read_csv(
        path,
        dtype={"well_id": "string", "row_index": "int64", "fold": "int64"},
        float_precision="round_trip",
    ).rename(columns={AUDITED_INCUMBENT_SOURCE_COLUMN: INCUMBENT_COLUMN})
    validation = validate_oof_frame(locked, (INCUMBENT_COLUMN,))
    _write_json(
        directory / "independent_reaudit.json",
        {
            "passed": True,
            "candidate_oof_sha256": sha256_file(path),
            "model_state_provenance_passed": True,
            "submission_csv_created": False,
            "disk_oof_nonfinite_values": 0,
            "disk_oof_duplicate_or_nonmonotone_keys": 0,
            "disk_oof_rows": len(locked),
            "disk_oof_key_sha256": validation["key_sha256"],
        },
    )
    return path


def _write_well(data_root: Path, well_id: str, offset: float) -> None:
    split = data_root / "train"
    split.mkdir(parents=True, exist_ok=True)
    horizontal = pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0, 4.0, 7.0],
            "X": [offset + value for value in (0.0, 1.0, 2.0, 3.0, 4.0)],
            "Y": [0.0, 0.2, 0.4, 0.6, 0.8],
            "Z": [0.0, -1.0, -2.0, -4.0, -7.0],
            "GR": [40.0, 42.0, 44.0, 43.0, 46.0],
            "TVT_input": [10.0, 11.0, np.nan, np.nan, np.nan],
            "TVT": [10.0, 11.0, 12.0, 14.0, 17.0],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": [0.0, 10.0, 20.0],
            "GR": [35.0, 45.0, 55.0],
        }
    )
    horizontal.to_csv(split / f"{well_id}__horizontal_well.csv", index=False)
    typewell.to_csv(split / f"{well_id}__typewell.csv", index=False)


def _checkpoint(
    run_root: Path,
    *,
    variant: dict[str, Any],
    fold: int,
    training_ids: tuple[str, ...],
) -> dict[str, Any]:
    sample_count = len(training_ids)
    coordinate = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    x = np.empty((sample_count, 2, 8), dtype=np.float32)
    targets = np.empty((sample_count, 2), dtype=np.float32)
    for index in range(sample_count):
        x[index, 0] = np.sin(2.0 * np.pi * coordinate + 0.1 * index)
        x[index, 1] = np.cos(2.0 * np.pi * coordinate + 0.1 * index)
        targets[index] = (0.1 * index, -0.05 * index)
    config_values: dict[str, Any] = {
        "family": variant["family"],
        "seed": SEED,
        "device": "cpu",
        "epochs": 1,
        "hidden_dim": 4,
        "layers": 1,
        "attention_heads": 1,
        "patch_sizes": (2, 4),
        "fno_modes": 3,
        "cross_series": variant["cross_series"],
        "frozen_random_encoder": variant["frozen_random_encoder"],
    }
    state = fit_torch_family(
        x,
        targets,
        channel_names=CHANNEL_NAMES,
        ids=training_ids,
        row_counts=np.ones(sample_count, dtype=np.int64),
        config=TorchFamilyConfig(**config_values),
    )
    checkpoint = (
        run_root / "models" / variant["variant_id"] / f"seed_{SEED}" / f"fold_{fold}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state.save(checkpoint)
    return {
        "variant_id": variant["variant_id"],
        "family": variant["family"],
        "control_id": variant["control_id"],
        "fold": fold,
        "seed": SEED,
        "training_wells": len(training_ids),
        "validation_wells": 2,
        "training_validation_intersection": 0,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "resumed": False,
        "elapsed_seconds": 0.01,
        "manifest": state.manifest,
    }


def _experiment(tmp_path: Path) -> dict[str, Any]:
    data_root = tmp_path / "data"
    well_ids = tuple(f"well-{index}" for index in range(4))
    for index, well_id in enumerate(well_ids):
        _write_well(data_root, well_id, float(index))

    fold_config = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            "well_id": well_ids,
            "fold": [0, 0, 1, 1],
        }
    ).to_csv(fold_config, index=False)
    run_root = tmp_path / "run"
    run_root.mkdir()

    variants = (
        {
            "variant_id": "tcn",
            "family": FAMILY_TCN,
            "control_id": "none",
            "cross_series": True,
            "frozen_random_encoder": False,
        },
        {
            "variant_id": "paired_patch_wrong_typewell",
            "family": FAMILY_PAIRED_PATCH,
            "control_id": "wrong_typewell",
            "cross_series": True,
            "frozen_random_encoder": False,
        },
    )
    cohort = {
        0: (("well-2", "well-3"), ("well-0", "well-1")),
        1: (("well-0", "well-1"), ("well-2", "well-3")),
    }
    model_records = [
        _checkpoint(
            run_root,
            variant=variant,
            fold=fold,
            training_ids=cohort[fold][0],
        )
        for variant in variants
        for fold in sorted(cohort)
    ]
    data_manifest = capture_data_manifest(data_root)
    run_config = {
        "run_id": "publisher-test",
        "profile": "quick",
        "data_root": str(data_root.resolve()),
        "fold_config": str(fold_config.resolve()),
        "variants": list(variants),
        "seeds": [SEED],
        "shrinkage": 0.25,
        "cohort": {
            str(fold): {
                "training_well_ids": list(training),
                "validation_well_ids": list(validation),
                "intersection": [],
                "training_ids_sha256": well_ids_sha256(training),
                "validation_ids_sha256": well_ids_sha256(validation),
            }
            for fold, (training, validation) in cohort.items()
        },
        "data_manifest": data_manifest,
        "fixed_grid": {"horizontal_points": 8, "typewell_points": 8},
        "residual_target": {"retained_dct_modes": 2},
        "external_pretrained_weights_used": [],
        "architecture_only": True,
        "submission_created": False,
    }
    model_manifest = {
        "models": model_records,
        "failures": [],
        "incomplete_family_seeds": [],
        "identity_fallbacks_scored": False,
    }
    incumbent_manifest = {
        "states": [
            {
                "fold": fold,
                "seed": 20260711,
                "sha256": str(fold + 1) * 64,
                "training_validation_intersection": 0,
            }
            for fold in sorted(cohort)
        ]
    }

    rows: list[dict[str, Any]] = []
    row_coordinate = np.array([1.0 / 6.0, 0.5, 1.0], dtype=np.float64)
    for well_number, well_id in enumerate(well_ids):
        fold = 0 if well_number < 2 else 1
        row_indices = (2, 3, 4)
        incumbent = 100.0 + well_number + 2.0 * row_coordinate
        truth = incumbent + 0.1
        current_rows = [
            {
                "well_id": well_id,
                "row_index": row_index,
                "fold": fold,
                "y_true": truth[index],
                INCUMBENT_COLUMN: incumbent[index],
            }
            for index, row_index in enumerate(row_indices)
        ]
        for variant_number, variant in enumerate(variants, start=1):
            common = np.full(3, 0.1 * variant_number * (well_number + 1))
            slope = 0.2 * variant_number * (row_coordinate - row_coordinate.mean())
            shape = np.zeros(3, dtype=np.float64)
            correction = common + slope + shape
            prediction = incumbent + correction
            for index, record in enumerate(current_rows):
                variant_id = variant["variant_id"]
                record[f"y_pred__{variant_id}__seed_{SEED}"] = prediction[index]
                record[f"y_pred__{variant_id}"] = prediction[index]
                record[f"y_pred__{variant_id}__shrink_025"] = (
                    incumbent[index] + 0.25 * correction[index]
                )
                record[f"correction__{variant_id}__common"] = common[index]
                record[f"correction__{variant_id}__slope"] = slope[index]
                record[f"correction__{variant_id}__shape"] = shape[index]
        rows.extend(current_rows)
    wide = pd.DataFrame(rows)
    incumbent_oof = _write_audited_incumbent_reference(tmp_path, wide)
    run_config["incumbent_oof"] = str(incumbent_oof.resolve())
    run_config["incumbent_oof_sha256"] = sha256_file(incumbent_oof)

    _write_json(run_root / RUN_CONFIG_FILENAME, run_config)
    _write_json(run_root / MODEL_MANIFEST_FILENAME, model_manifest)
    _write_json(run_root / INCUMBENT_MANIFEST_FILENAME, incumbent_manifest)
    wide.to_csv(
        run_root / WIDE_OOF_FILENAME,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    return {
        "run_root": run_root,
        "data_root": data_root,
        "wide": wide,
        "model_manifest": model_manifest,
        "coordinates": row_coordinate,
        "incumbent_oof": incumbent_oof,
    }


def test_publish_canonical_bundle_reconstructs_md_and_covers_models(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    bundle = publish_external_portfolio_artifacts(experiment["run_root"])

    assert bundle.audit["passed"] is True
    assert bundle.audit["oof"]["rows"] == 12
    assert bundle.audit["oof"]["wells"] == 4
    assert bundle.audit["models"]["checkpoint_sha256_checked"] == 4
    candidates = {
        value["candidate_id"]: value for value in bundle.manifest["oof"]["candidates"]
    }
    assert set(candidates) == {
        "tcn",
        "tcn__shrink_025",
        "paired_patch_wrong_typewell",
        "paired_patch_wrong_typewell__shrink_025",
    }
    assert len(candidates["tcn"]["source_model_ids"]) == 2
    assert bundle.manifest["metadata"]["row_coordinate"]["source"] == "raw_inference_md"
    for _, group in bundle.oof.groupby("well_id"):
        np.testing.assert_allclose(
            group.sort_values("row_index")[ROW_COORDINATE_COLUMN],
            experiment["coordinates"],
        )
    for candidate in candidates.values():
        assert candidate["common_column"] in bundle.oof
        assert candidate["slope_column"] in bundle.oof
        assert candidate["shape_column"] in bundle.oof
        assert candidate["correction_column"] in bundle.oof
        assert candidate["prediction_column"] in bundle.oof


def test_complete_candidate_with_missing_checkpoint_fails_closed(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    missing = experiment["run_root"] / "models" / "tcn" / f"seed_{SEED}" / "fold_1.pt"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
        publish_external_portfolio_artifacts(experiment["run_root"])
    assert not (experiment["run_root"] / DEFAULT_BUNDLE_NAME).exists()


def test_negative_controls_are_explicitly_ineligible(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    coordinate_frame = reconstruct_row_coordinates(
        experiment["wide"],
        data_root=experiment["data_root"],
    )
    coordinate_path = tmp_path / "row-coordinate.csv.gz"
    coordinate_frame.to_csv(
        coordinate_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    bundle = publish_external_portfolio_artifacts(
        experiment["run_root"],
        row_coordinate_path=coordinate_path,
    )
    candidates = bundle.manifest["oof"]["candidates"]
    controls = [
        value for value in candidates if value["control_id"] == "wrong_typewell"
    ]
    assert controls
    assert all(value["eligible_for_model_selection"] is False for value in controls)
    assert all(value["transfer_mode"] == "negative_control" for value in controls)
    normal = [value for value in candidates if value["candidate_id"] == "tcn"]
    assert normal[0]["eligible_for_model_selection"] is True
    assert (
        bundle.manifest["metadata"]["row_coordinate"]["source"] == "explicit_artifact"
    )


def test_nonmonotone_explicit_coordinate_fails_before_commit(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    coordinate_frame = reconstruct_row_coordinates(
        experiment["wide"],
        data_root=experiment["data_root"],
    )
    first_well = coordinate_frame["well_id"].iloc[0]
    selected = coordinate_frame["well_id"].eq(first_well)
    coordinate_frame.loc[selected, ROW_COORDINATE_COLUMN] = [0.4, 0.2, 1.0]
    coordinate_path = tmp_path / "bad-coordinate.csv"
    coordinate_frame.to_csv(coordinate_path, index=False)

    with pytest.raises(ValueError, match="strictly increasing"):
        publish_external_portfolio_artifacts(
            experiment["run_root"],
            row_coordinate_path=coordinate_path,
        )
    assert not (experiment["run_root"] / DEFAULT_BUNDLE_NAME).exists()


@pytest.mark.parametrize("column", ["y_true", INCUMBENT_COLUMN])
def test_publication_rejects_self_consistent_truth_or_incumbent_corruption(
    tmp_path: Path,
    column: str,
) -> None:
    experiment = _experiment(tmp_path)
    wide_path = experiment["run_root"] / WIDE_OOF_FILENAME
    wide = pd.read_csv(
        wide_path,
        dtype={"well_id": "string", "row_index": "int64", "fold": "int64"},
        float_precision="round_trip",
    )
    wide.loc[0, column] += 0.01
    if column == INCUMBENT_COLUMN:
        prediction_columns = [
            current for current in wide.columns if current.startswith("y_pred__")
        ]
        wide.loc[0, prediction_columns] += 0.01
    wide.to_csv(
        wide_path,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "mtime": 0},
    )

    with pytest.raises(ValueError, match="audited incumbent reference"):
        publish_external_portfolio_artifacts(experiment["run_root"])
    assert not (experiment["run_root"] / DEFAULT_BUNDLE_NAME).exists()


def test_publisher_cli_exposes_explicit_incumbent_lock() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/publish_external_portfolio_artifacts.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--incumbent-oof" in result.stdout
