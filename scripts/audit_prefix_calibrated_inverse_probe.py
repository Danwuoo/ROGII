"""Independent disk-only re-audit for the prefix-calibrated inverse probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rogii.external_models.artifacts import capture_data_manifest, sha256_file
from rogii.inverse_solver.contracts import PrefixCalibrationConfig
from rogii.inverse_solver.stage0 import audit_stage0_frame


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_per_well(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    identifiers = {
        column: "string"
        for column in columns
        if column == "well_id" or column.endswith("_id")
    }
    return pd.read_csv(path, dtype=identifiers, float_precision="round_trip")


def _quartile_sensitivity(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    values = pd.to_numeric(frame[column], errors="raise")
    bucket = pd.qcut(values.rank(method="first"), 4, labels=False)
    gain = frame["wrong_cost"].to_numpy(dtype=np.float64) - frame[
        "correct_cost"
    ].to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    for quartile in range(4):
        mask = bucket.to_numpy(dtype=np.int64) == quartile
        records.append(
            {
                "quartile": quartile + 1,
                "rows": int(np.sum(mask)),
                "range_min": float(np.min(values.to_numpy()[mask])),
                "range_max": float(np.max(values.to_numpy()[mask])),
                "mean_wrong_minus_correct_gain": float(np.mean(gain[mask])),
                "median_wrong_minus_correct_gain": float(np.median(gain[mask])),
                "strict_correct_win_fraction": float(np.mean(gain[mask] > 0.0)),
            }
        )
    return records


def _typewell_hashes(data_root: Path, well_ids: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for well_id in well_ids:
        path = data_root / "train" / f"{well_id}__typewell.csv"
        frame = pd.read_csv(path, usecols=["TVT", "GR"])
        values = np.asarray(frame[["TVT", "GR"]], dtype="<f8")
        digest = hashlib.sha256()
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
        result[well_id] = digest.hexdigest()
    return result


def audit(run_dir: Path, output: Path | None = None) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    run_config_path = root / "run_config.json"
    summary_path = root / "summary.json"
    gate_path = root / "stage0_gate.json"
    per_well_path = root / "stage0_observability_per_well.csv.gz"
    for path in (run_config_path, summary_path, gate_path, per_well_path):
        if not path.exists():
            raise FileNotFoundError(f"required Stage 0 artifact is missing: {path}")
    run_config = _read_json(run_config_path)
    summary = _read_json(summary_path)
    recorded_gate = _read_json(gate_path)
    config_path = Path(run_config["config"])
    if sha256_file(config_path) != run_config["config_sha256"]:
        raise ValueError("locked config SHA-256 differs")
    locked = _read_json(config_path)
    settings = PrefixCalibrationConfig.from_dict(locked["prefix_calibration"])
    current_manifest = capture_data_manifest(run_config["data_root"])
    if (
        current_manifest != run_config["data_manifest"]
        or current_manifest != summary["data_manifest_before"]
        or current_manifest != summary["data_manifest_after"]
    ):
        raise ValueError("data manifest differs from Stage 0 provenance")
    if sha256_file(per_well_path) != summary["per_well_sha256"]:
        raise ValueError("Stage 0 per-well artifact SHA-256 differs")
    if sha256_file(run_config_path) != summary["run_config_sha256"]:
        raise ValueError("Stage 0 run-config SHA-256 differs")
    frame = _read_per_well(per_well_path)
    failures = _read_json(root / "stage0_failures.json")["failures"]
    rebuilt = audit_stage0_frame(
        frame,
        config=settings,
        expected_wells=len(run_config["selected_well_ids"]),
        failure_count=len(failures),
    )
    if rebuilt != recorded_gate or summary["stage0"] != recorded_gate:
        raise ValueError("independent Stage 0 metric reconstruction differs")
    expected_ids = tuple(sorted(map(str, run_config["selected_well_ids"])))
    actual_ids = tuple(sorted(frame["well_id"].astype(str)))
    failed_ids = tuple(sorted(str(item["well_id"]) for item in failures))
    if tuple(sorted((*actual_ids, *failed_ids))) != expected_ids:
        raise ValueError("Stage 0 completed/failed well keys differ")
    checkpoint_rows: list[dict[str, Any]] = []
    for well_id in expected_ids:
        checkpoint = _read_json(root / "checkpoints" / "stage0" / f"{well_id}.json")
        if (
            checkpoint["config_sha256"] != run_config["config_sha256"]
            or checkpoint["data_manifest"] != current_manifest
        ):
            raise ValueError(f"checkpoint provenance differs: {well_id}")
        if checkpoint["status"] == "completed":
            checkpoint_rows.append(checkpoint["row"])
    checkpoint_frame = (
        pd.DataFrame(checkpoint_rows)
        .sort_values("well_id", kind="stable")
        .reset_index(drop=True)
    )
    comparable_columns = sorted(set(frame.columns) & set(checkpoint_frame.columns))
    pd.testing.assert_frame_equal(
        frame[comparable_columns],
        checkpoint_frame[comparable_columns],
        check_dtype=False,
        check_exact=False,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    if (
        not frame["phase_marginal_preserved"].astype(bool).all()
        or not frame["shuffle_marginal_preserved"].astype(bool).all()
        or summary.get("submission_created") is not False
        or summary.get("submission_uploaded") is not False
        or summary.get("full_training_executed") is not False
    ):
        raise ValueError("control preservation or execution-scope audit failed")
    correct_minus_wrong = frame["correct_cost"].to_numpy(dtype=np.float64) - frame[
        "wrong_cost"
    ].to_numpy(dtype=np.float64)
    well_ids = tuple(frame["well_id"].astype(str))
    typewell_hash = _typewell_hashes(Path(run_config["data_root"]), well_ids)
    wrong_ids = tuple(frame["wrong_typewell_id"].astype(str))
    same_typewell = np.array(
        [
            typewell_hash[well_id] == typewell_hash[wrong_id]
            for well_id, wrong_id in zip(well_ids, wrong_ids, strict=True)
        ],
        dtype=bool,
    )
    exact_tie = correct_minus_wrong == 0.0
    diagnostics = {
        "correct_wrong_exact_ties": int(np.sum(exact_tie)),
        "non_tied_wells": int(np.sum(~exact_tie)),
        "strict_correct_win_fraction_among_non_ties": float(
            np.mean(correct_minus_wrong[~exact_tie] < 0.0)
        ),
        "unique_typewell_tvt_gr_hashes": len(set(typewell_hash.values())),
        "wrong_control_same_typewell_tvt_gr_hash": int(np.sum(same_typewell)),
        "ties_with_same_typewell_tvt_gr_hash": int(np.sum(exact_tie & same_typewell)),
        "same_hash_fraction": float(np.mean(same_typewell)),
        "prefix_length_sensitivity": _quartile_sensitivity(
            frame, "correct_prefix_length_ft"
        ),
        "gr_missingness_sensitivity": _quartile_sensitivity(
            frame, "correct_gr_missing_fraction"
        ),
        "typewell_coverage_sensitivity": _quartile_sensitivity(
            frame, "correct_typewell_coverage_fraction"
        ),
        "correct_lag_or_stretch_boundary_fraction": float(
            np.mean(frame["correct_boundary_fraction"].to_numpy(dtype=np.float64))
        ),
        "stage1_search_band_clipping": "not_applicable_stage1_not_run",
        "stage1_low_observability_catastrophic_concentration": "not_applicable_stage1_not_run",
        "stage1_observation_path_influence": "not_applicable_stage1_not_run",
    }
    result = {
        "passed": True,
        "profile": run_config["profile"],
        "audited_wells": len(frame),
        "failed_wells": len(failures),
        "stage0_passed": bool(recorded_gate["passed"]),
        "decision": recorded_gate["decision"],
        "config_sha256": run_config["config_sha256"],
        "run_config_sha256": summary["run_config_sha256"],
        "per_well_sha256": summary["per_well_sha256"],
        "data_manifest_equal_before_after_current": True,
        "checkpoint_reconstruction_passed": True,
        "marginal_and_missingness_controls_passed": True,
        "target_columns_read_for_prediction": [],
        "submission_created": False,
        "submission_uploaded": False,
        "full_training_executed": False,
        "diagnostics": diagnostics,
    }
    _write_json(output or root / "independent_audit.json", result)
    _write_json(root / "stage0_diagnostics.json", diagnostics)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(".runs/prefix-calibrated-inverse-probe-20260729"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
