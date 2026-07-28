from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

import rogii.external_models.nested_stress as nested_stress_module
from rogii.diagnostics.nested_external_ensemble import (
    FIXED_050_COLUMN,
    FIXED_075_COLUMN,
    INCUMBENT_COLUMN,
    NESTED_COLUMN,
    TCN_COLUMN,
)
from rogii.external_models.artifacts import sha256_file
from rogii.external_models.nested_reaudit import (
    NESTED_REAUDIT_SCHEMA_VERSION,
)
from rogii.external_models.nested_stress import (
    METRICS_FILENAME,
    NESTED_STRESS_SCHEMA_VERSION,
    NestedStressError,
    PREDICTION_COLUMNS,
    SUMMARY_FILENAME,
    analyze_nested_target_free_stress,
)
from rogii.external_models.stress import TARGET_FREE_FEATURE_COLUMNS
from rogii.portfolio_metrics import make_target_free_strata


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(
    root: Path,
    *,
    include_strata_hash: bool = True,
) -> tuple[Path, Path]:
    external_root = root / "external-full"
    stress_root = external_root / "stress"
    stress_root.mkdir(parents=True)
    nested_root = root / "nested-full"
    nested_root.mkdir()

    well_ids = ("well-z", "well-y", "well-a", "well-b")
    features = pd.DataFrame({"well_id": well_ids})
    for column_index, column in enumerate(TARGET_FREE_FEATURE_COLUMNS):
        features[column] = np.arange(len(well_ids), dtype=np.float64) + 0.01 * (
            column_index + 1
        )
    feature_path = stress_root / "target_free_well_features.csv"
    features.to_csv(
        feature_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    strata = make_target_free_strata(
        features,
        TARGET_FREE_FEATURE_COLUMNS,
        bins=2,
    )
    strata_path = stress_root / "target_free_strata.csv"
    strata.to_csv(
        strata_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    registration = root / "registration_diagnostics.csv.gz"
    pd.DataFrame({"well_id": well_ids, "value": np.arange(4)}).to_csv(
        registration,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    rows: list[dict[str, Any]] = []
    for well_index, well_id in enumerate(well_ids):
        fold = well_index
        for row_index in range(3):
            truth = 100.0 + 10.0 * well_index + row_index
            incumbent = truth + 2.0 - 0.2 * well_index
            raw_tcn = truth + 0.5 + 0.3 * well_index
            rows.append(
                {
                    "well_id": well_id,
                    "row_index": row_index,
                    "fold": fold,
                    "y_true": truth,
                    INCUMBENT_COLUMN: incumbent,
                    TCN_COLUMN: raw_tcn,
                    NESTED_COLUMN: truth + 0.25 + 0.05 * well_index,
                    FIXED_075_COLUMN: 0.75 * incumbent + 0.25 * raw_tcn,
                    FIXED_050_COLUMN: 0.50 * incumbent + 0.50 * raw_tcn,
                }
            )
    oof = pd.DataFrame(rows)
    oof_path = nested_root / "oof_predictions.csv.gz"
    oof.to_csv(
        oof_path,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "mtime": 0},
    )

    _write_json(
        external_root / "summary.json",
        {
            "profile": "full",
            "failure_count": 0,
            "incomplete_family_seeds": [],
            "data_manifest_equal": True,
            "submission_created": False,
            "eligible_family_columns": ["y_pred__tcn"],
            "scope": {
                "rows": len(oof),
                "wells": len(well_ids),
                "folds": [0, 1, 2, 3, 4],
            },
        },
    )
    stress_summary: dict[str, Any] = {
        "run_root": str(external_root.resolve()),
        "rows": len(oof),
        "wells": len(well_ids),
        "bins": 2,
        "feature_columns": list(TARGET_FREE_FEATURE_COLUMNS),
        "feature_sha256": sha256_file(feature_path),
        "strata_rows": len(strata),
        "registration_diagnostics": str(registration.resolve()),
        "registration_diagnostics_sha256": sha256_file(registration),
        "target_columns_used_for_strata": [],
        "prediction_columns_used_for_strata": [],
    }
    if include_strata_hash:
        stress_summary["strata_sha256"] = sha256_file(strata_path)
    _write_json(stress_root / "summary.json", stress_summary)

    _write_json(
        nested_root / "independent_nested_reaudit.json",
        {
            "schema_version": NESTED_REAUDIT_SCHEMA_VERSION,
            "passed": True,
            "run_root": str(nested_root.resolve()),
            "evidence_mode": "production",
            "evidence_eligible": True,
            "scope": {"rows": len(oof), "wells": len(well_ids)},
            "checks": {"submission_created": False},
            "source": {
                "mode": "production",
                "root": str(external_root.resolve()),
                "oof_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
                "independently_locked": True,
            },
            "artifacts": {
                "final_oof": {
                    "path": str(oof_path.resolve()),
                    "sha256": sha256_file(oof_path),
                }
            },
            "submission_created": False,
        },
    )
    return external_root, nested_root


@pytest.mark.parametrize("include_strata_hash", (True, False))
def test_nested_stress_uses_frozen_strata_and_writes_atomic_summary(
    tmp_path: Path,
    include_strata_hash: bool,
) -> None:
    external_root, nested_root = _fixture(
        tmp_path,
        include_strata_hash=include_strata_hash,
    )
    summary = analyze_nested_target_free_stress(
        nested_root,
        external_run_root=external_root,
    )
    output = nested_root / "stress"
    assert summary["schema_version"] == NESTED_STRESS_SCHEMA_VERSION
    assert summary["completed"] is True
    assert summary["evidence_eligible"] is True
    assert summary["prediction_columns"] == list(PREDICTION_COLUMNS)
    assert summary["target_columns_used_for_strata"] == []
    assert summary["prediction_columns_used_for_strata"] == []
    assert summary["submission_created"] is False
    assert set(summary["candidate_best_delta"]) == set(PREDICTION_COLUMNS[1:])
    assert set(summary["candidate_worst_delta"]) == set(PREDICTION_COLUMNS[1:])
    assert summary["feature_sha256"] == sha256_file(
        external_root / "stress" / "target_free_well_features.csv"
    )
    assert summary["strata_sha256"] == sha256_file(
        external_root / "stress" / "target_free_strata.csv"
    )
    assert summary["independent_nested_reaudit_sha256"] == sha256_file(
        nested_root / "independent_nested_reaudit.json"
    )
    assert summary["final_oof_sha256"] == sha256_file(
        nested_root / "oof_predictions.csv.gz"
    )
    assert summary["metrics_sha256"] == sha256_file(output / METRICS_FILENAME)
    assert (
        json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8")) == summary
    )
    metrics = pd.read_csv(output / METRICS_FILENAME)
    assert set(metrics["candidate"]) == set(PREDICTION_COLUMNS)
    assert not list(nested_root.glob(".stress.tmp-*"))


@pytest.mark.parametrize(
    "corruption",
    (
        "feature_hash",
        "strata_hash",
        "target_marker",
        "prediction_marker",
        "strata_target_column",
        "strata_semantics",
        "registration_bytes",
        "audit_passed",
        "audit_evidence",
        "audit_submission",
        "audit_source",
        "submission_file",
        "oof_hash",
        "oof_bytes",
        "oof_fixed_control",
    ),
)
def test_nested_stress_corruption_fails_without_output(
    tmp_path: Path,
    corruption: str,
) -> None:
    external_root, nested_root = _fixture(tmp_path)
    stress_root = external_root / "stress"
    stress_summary_path = stress_root / "summary.json"
    stress_summary = json.loads(stress_summary_path.read_text(encoding="utf-8"))
    audit_path = nested_root / "independent_nested_reaudit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    if corruption == "feature_hash":
        stress_summary["feature_sha256"] = "0" * 64
        _write_json(stress_summary_path, stress_summary)
    elif corruption == "strata_hash":
        stress_summary["strata_sha256"] = "0" * 64
        _write_json(stress_summary_path, stress_summary)
    elif corruption == "target_marker":
        stress_summary["target_columns_used_for_strata"] = ["y_true"]
        _write_json(stress_summary_path, stress_summary)
    elif corruption == "prediction_marker":
        stress_summary["prediction_columns_used_for_strata"] = [INCUMBENT_COLUMN]
        _write_json(stress_summary_path, stress_summary)
    elif corruption in {"strata_target_column", "strata_semantics"}:
        strata_path = stress_root / "target_free_strata.csv"
        strata = pd.read_csv(strata_path)
        if corruption == "strata_target_column":
            strata["y_true"] = 1.0
        else:
            strata.loc[0, "stratum"] = "tampered"
        strata.to_csv(
            strata_path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        stress_summary["strata_sha256"] = sha256_file(strata_path)
        _write_json(stress_summary_path, stress_summary)
    elif corruption == "registration_bytes":
        Path(stress_summary["registration_diagnostics"]).write_bytes(b"changed")
    elif corruption in {
        "audit_passed",
        "audit_evidence",
        "audit_submission",
        "audit_source",
        "oof_hash",
    }:
        if corruption == "audit_passed":
            audit["passed"] = False
        elif corruption == "audit_evidence":
            audit["evidence_eligible"] = False
        elif corruption == "audit_submission":
            audit["submission_created"] = True
        elif corruption == "audit_source":
            audit["source"]["root"] = str((tmp_path / "other").resolve())
        else:
            audit["artifacts"]["final_oof"]["sha256"] = "0" * 64
        _write_json(audit_path, audit)
    elif corruption == "submission_file":
        (nested_root / "submission.csv").write_text(
            "id,TVT\n1,2\n",
            encoding="utf-8",
        )
    elif corruption == "oof_bytes":
        oof_path = nested_root / "oof_predictions.csv.gz"
        oof_path.write_bytes(oof_path.read_bytes() + b"changed")
    else:
        oof_path = nested_root / "oof_predictions.csv.gz"
        oof = pd.read_csv(oof_path, float_precision="round_trip")
        oof.loc[0, FIXED_075_COLUMN] += 1.0
        oof.to_csv(
            oof_path,
            index=False,
            float_format="%.17g",
            compression={"method": "gzip", "mtime": 0},
        )
        audit["artifacts"]["final_oof"]["sha256"] = sha256_file(oof_path)
        _write_json(audit_path, audit)

    with pytest.raises(
        (NestedStressError, FileNotFoundError, ValueError),
    ):
        analyze_nested_target_free_stress(
            nested_root,
            external_run_root=external_root,
        )
    assert not (nested_root / "stress").exists()
    assert not list(nested_root.glob(".stress.tmp-*"))


def test_nested_oof_is_not_opened_until_strata_are_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_root, nested_root = _fixture(tmp_path)
    strata_path = external_root / "stress" / "target_free_strata.csv"
    strata = pd.read_csv(strata_path)
    strata.loc[0, "stratum"] = "tampered"
    strata.to_csv(strata_path, index=False, lineterminator="\n")
    stress_summary_path = external_root / "stress" / "summary.json"
    summary = json.loads(stress_summary_path.read_text(encoding="utf-8"))
    summary["strata_sha256"] = sha256_file(strata_path)
    _write_json(stress_summary_path, summary)

    real_read_csv = nested_stress_module.pd.read_csv
    opened: list[Path] = []

    def recording_read_csv(path: Any, *args: Any, **kwargs: Any):
        resolved = Path(path).resolve()
        opened.append(resolved)
        if resolved == (nested_root / "oof_predictions.csv.gz").resolve():
            raise AssertionError("nested OOF opened before frozen strata validation")
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(nested_stress_module.pd, "read_csv", recording_read_csv)
    with pytest.raises(NestedStressError, match="deterministic frozen"):
        analyze_nested_target_free_stress(
            nested_root,
            external_run_root=external_root,
        )
    assert (nested_root / "oof_predictions.csv.gz").resolve() not in opened


def test_nested_stress_cli_uses_disk_artifacts(tmp_path: Path) -> None:
    external_root, nested_root = _fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_nested_external_stress.py",
            "--nested-run-root",
            str(nested_root),
            "--external-run-root",
            str(external_root),
            "--output-name",
            "cli-stress",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["completed"] is True
    assert output["evidence_eligible"] is True
    assert (nested_root / "cli-stress" / SUMMARY_FILENAME).is_file()
