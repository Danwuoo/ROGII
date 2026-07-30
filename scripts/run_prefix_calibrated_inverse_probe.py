"""Run the locked target-free prefix observability gate and downstream probe."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping
from ctypes import wintypes

import numpy as np
import pandas as pd

from rogii.data import (
    enumerate_wells,
    load_horizontal_inference,
    load_typewell_inference,
    validate_prefix,
)
from rogii.external_models.artifacts import capture_data_manifest, sha256_file
from rogii.folds import load_fold_mapping
from rogii.inverse_solver.contracts import (
    CONTROL_CORRECT,
    CONTROL_GR_ORDER_SHUFFLE,
    CONTROL_PHASE_DESTROY,
    CONTROL_WRONG_TYPEWELL,
    PrefixCalibrationConfig,
)
from rogii.inverse_solver.controls import (
    deterministic_matched_wrong_typewell,
    nuisance_only_cost,
    phase_destroy_preserve_marginal,
    shuffle_gr_preserve_missingness,
    stable_seed,
    summarize_typewell,
)
from rogii.inverse_solver.prefix_calibration import calibrate_prefix_likelihood
from rogii.inverse_solver.stage0 import audit_stage0_frame
from rogii.submission import validate_sample_submission_contract


DEFAULT_DATA_ROOT = Path("data/rogii-wellbore-geology-prediction")
DEFAULT_FOLDS = Path("configs/folds/main_5fold.csv")
DEFAULT_CONFIG = Path("configs/inverse/prefix_calibrated_probe_20260729.json")
DEFAULT_RUN_DIR = Path(".runs/prefix-calibrated-inverse-probe-20260729")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_per_well(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    identifiers = {
        column: "string"
        for column in columns
        if column == "well_id" or column.endswith("_id")
    }
    return pd.read_csv(
        path,
        dtype=identifiers,
        float_precision="round_trip",
    )


def _peak_memory_mb() -> float:
    try:
        import psutil

        memory = psutil.Process(os.getpid()).memory_info()
        value = getattr(memory, "peak_wset", memory.rss)
        return float(value / (1024.0 * 1024.0))
    except Exception:
        try:
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel.GetCurrentProcess()
            success = psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            if not success:
                raise OSError("GetProcessMemoryInfo failed")
            return float(counters.PeakWorkingSetSize / (1024.0 * 1024.0))
        except Exception:
            return float("nan")


def _resolved_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    return "cuda" if available else "cpu"


def _load_locked_config(path: Path) -> tuple[dict[str, Any], PrefixCalibrationConfig]:
    payload = _read_json(path)
    if payload.get("locked_before_stage0") is not True:
        raise ValueError("inverse-probe config is not locked")
    settings = PrefixCalibrationConfig.from_dict(payload["prefix_calibration"])
    return payload, settings


def _checkpoint_path(root: Path, well_id: str) -> Path:
    return root / "checkpoints" / "stage0" / f"{well_id}.json"


def _preservation_check(before: np.ndarray, after: np.ndarray) -> bool:
    left = np.asarray(before, dtype=np.float64)
    right = np.asarray(after, dtype=np.float64)
    return bool(
        np.array_equal(np.isfinite(left), np.isfinite(right))
        and np.array_equal(
            np.sort(left[np.isfinite(left)], kind="stable"),
            np.sort(right[np.isfinite(right)], kind="stable"),
        )
    )


def _one_well(
    *,
    well_id: str,
    fold: int,
    horizontal: pd.DataFrame,
    typewells: Mapping[str, pd.DataFrame],
    summaries: Mapping[str, Any],
    config: PrefixCalibrationConfig,
    base_seed: int,
) -> dict[str, Any]:
    known_count, _ = validate_prefix(horizontal["TVT_input"])
    prefix_tvt = horizontal["TVT_input"].iloc[:known_count].to_numpy(dtype=np.float64)
    wrong_id, matched_distance = deterministic_matched_wrong_typewell(
        well_id, summaries, prefix_tvt=prefix_tvt
    )
    correct = calibrate_prefix_likelihood(
        horizontal,
        typewells[well_id],
        well_id=well_id,
        reference_well_id=well_id,
        control_id=CONTROL_CORRECT,
        config=config,
    )
    wrong = calibrate_prefix_likelihood(
        horizontal,
        typewells[wrong_id],
        well_id=well_id,
        reference_well_id=wrong_id,
        control_id=CONTROL_WRONG_TYPEWELL,
        config=config,
    )
    phase_frame = typewells[well_id].copy()
    phase_before = phase_frame["GR"].to_numpy(dtype=np.float64)
    phase_after = phase_destroy_preserve_marginal(
        phase_before,
        seed=stable_seed(well_id, "phase", base_seed=base_seed),
    )
    if not _preservation_check(phase_before, phase_after):
        raise RuntimeError("phase control failed marginal/missingness preservation")
    phase_frame["GR"] = phase_after
    phase = calibrate_prefix_likelihood(
        horizontal,
        phase_frame,
        well_id=well_id,
        reference_well_id=well_id,
        control_id=CONTROL_PHASE_DESTROY,
        config=config,
    )
    shuffle_frame = horizontal.copy()
    shuffle_before = shuffle_frame["GR"].iloc[:known_count].to_numpy(dtype=np.float64)
    shuffle_after = shuffle_gr_preserve_missingness(
        shuffle_before,
        seed=stable_seed(well_id, "shuffle", base_seed=base_seed),
    )
    if not _preservation_check(shuffle_before, shuffle_after):
        raise RuntimeError("shuffle control failed marginal/missingness preservation")
    shuffle_frame.loc[shuffle_frame.index[:known_count], "GR"] = shuffle_after
    shuffle = calibrate_prefix_likelihood(
        shuffle_frame,
        typewells[well_id],
        well_id=well_id,
        reference_well_id=well_id,
        control_id=CONTROL_GR_ORDER_SHUFFLE,
        config=config,
    )
    correct_nuisance = nuisance_only_cost(horizontal, summaries[well_id])
    wrong_nuisance = nuisance_only_cost(horizontal, summaries[wrong_id])
    costs = np.array([correct.cost, wrong.cost, phase.cost, shuffle.cost])
    row: dict[str, Any] = {
        "well_id": well_id,
        "fold": int(fold),
        "wrong_typewell_id": wrong_id,
        "wrong_matched_nuisance_distance": matched_distance,
        "correct_rank": int(1 + np.sum(costs[1:] < costs[0])),
        "phase_marginal_preserved": True,
        "shuffle_marginal_preserved": True,
    }
    for label, result in (
        ("correct", correct),
        ("wrong", wrong),
        ("phase", phase),
        ("shuffle", shuffle),
    ):
        for key, value in result.to_dict().items():
            if key in {"well_id", "reference_well_id", "control_id"}:
                row[f"{label}_{key}"] = value
            else:
                row[f"{label}_{key}"] = value
    for key, value in correct_nuisance.items():
        row[f"correct_{key}"] = value
    for key, value in wrong_nuisance.items():
        row[f"wrong_{key}"] = value
    return row


def _select_wells(profile: str, wells: tuple[Any, ...], fold_map: Mapping[str, int]) -> tuple[Any, ...]:
    if profile != "smoke":
        return wells
    selected: list[Any] = []
    for fold in range(5):
        candidates = [well for well in wells if int(fold_map[well.well_id]) == fold]
        if not candidates:
            raise ValueError(f"smoke selection lacks fold {fold}")
        selected.append(candidates[0])
    return tuple(selected)


def run_observability(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = Path(args.data_root).resolve()
    config_path = Path(args.config).resolve()
    run_root = Path(args.run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    locked, config = _load_locked_config(config_path)
    config_sha = sha256_file(config_path)
    data_manifest = capture_data_manifest(data_root)
    before_path = run_root / "data_manifest_before.json"
    if before_path.exists() and _read_json(before_path) != data_manifest:
        raise ValueError("current data differs from before manifest")
    _write_json(before_path, data_manifest)
    all_wells = enumerate_wells(data_root, "train")
    mapping_frame = load_fold_mapping(
        Path(args.fold_config),
        expected_well_ids=[well.well_id for well in all_wells],
    )
    fold_map = mapping_frame.set_index("well_id")["fold"].astype(int).to_dict()
    selected = _select_wells(args.profile, all_wells, fold_map)
    if args.profile == "quick" and len(selected) != 773:
        raise ValueError("canonical Stage 0 quick profile must audit all 773 wells")
    device = _resolved_device(args.device)
    typewells = {
        well.well_id: load_typewell_inference(well.typewell_path)
        for well in all_wells
    }
    summaries = {
        well_id: summarize_typewell(well_id, frame)
        for well_id, frame in typewells.items()
    }
    run_contract = {
        "schema_version": "prefix_inverse_stage0_run_v1",
        "stage": "observability",
        "profile": args.profile,
        "data_root": str(data_root),
        "data_manifest": data_manifest,
        "fold_config": str(Path(args.fold_config).resolve()),
        "config": str(config_path),
        "config_sha256": config_sha,
        "locked_config": locked,
        "seed": int(args.seed),
        "device_requested": args.device,
        "device_resolved": device,
        "num_workers": int(args.num_workers),
        "grid_chunk_rows": int(args.grid_chunk_rows),
        "memory_limit_mb": int(args.memory_limit_mb),
        "selected_well_ids": [well.well_id for well in selected],
        "target_columns_read_for_prediction": [],
        "formation_columns_read": [],
        "geology_columns_read": [],
        "png_inputs_read": False,
        "submission_created": False,
        "submission_uploaded": False,
    }
    run_config_path = run_root / "run_config.json"
    if run_config_path.exists():
        existing = _read_json(run_config_path)
        if existing != run_contract:
            raise ValueError("run directory contains a different deterministic manifest")
        if not args.resume:
            raise FileExistsError("run already exists; pass --resume for exact continuation")
    else:
        _write_json(run_config_path, run_contract)

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for position, well in enumerate(selected, start=1):
        checkpoint = _checkpoint_path(run_root, well.well_id)
        if checkpoint.exists():
            if not args.resume:
                raise FileExistsError(f"checkpoint already exists: {checkpoint}")
            payload = _read_json(checkpoint)
            if (
                payload.get("config_sha256") != config_sha
                or payload.get("data_manifest") != data_manifest
            ):
                raise ValueError(f"stale Stage 0 checkpoint: {well.well_id}")
            if payload.get("status") == "completed":
                completed.append(dict(payload["row"]))
            else:
                failures.append(dict(payload["failure"]))
            continue
        try:
            horizontal = load_horizontal_inference(well.horizontal_path)
            row = _one_well(
                well_id=well.well_id,
                fold=int(fold_map[well.well_id]),
                horizontal=horizontal,
                typewells=typewells,
                summaries=summaries,
                config=config,
                base_seed=int(args.seed),
            )
            completed.append(row)
            checkpoint_payload: dict[str, Any] = {
                "status": "completed",
                "config_sha256": config_sha,
                "data_manifest": data_manifest,
                "row": row,
            }
        except Exception as exc:
            failure = {
                "well_id": well.well_id,
                "exception_type": type(exc).__name__,
                "reason": str(exc),
            }
            failures.append(failure)
            checkpoint_payload = {
                "status": "failed",
                "config_sha256": config_sha,
                "data_manifest": data_manifest,
                "failure": failure,
            }
        _write_json(checkpoint, checkpoint_payload)
        _write_json(
            run_root / "progress.json",
            {
                "completed_or_failed": position,
                "total": len(selected),
                "last_well_id": well.well_id,
                "resume_supported": True,
            },
        )
    frame = (
        pd.DataFrame(completed)
        .sort_values("well_id", kind="stable")
        .reset_index(drop=True)
    )
    frame = frame.reindex(sorted(frame.columns), axis=1)
    per_well_path = run_root / "stage0_observability_per_well.csv.gz"
    preserve_existing = False
    if args.resume and per_well_path.exists():
        existing_frame = _read_per_well(per_well_path)
        preserve_existing = (
            tuple(existing_frame.columns) == tuple(frame.columns)
            and tuple(existing_frame["well_id"].astype(str))
            == tuple(frame["well_id"].astype(str))
        )
        if preserve_existing:
            pd.testing.assert_frame_equal(
                existing_frame,
                frame,
                check_dtype=False,
                check_exact=False,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
    if not preserve_existing:
        frame.to_csv(per_well_path, index=False, compression="gzip")
    _write_json(run_root / "stage0_failures.json", {"failures": failures})
    gate = audit_stage0_frame(
        frame,
        config=config,
        expected_wells=len(selected),
        failure_count=len(failures),
    )
    _write_json(run_root / "stage0_gate.json", gate)
    after_manifest = capture_data_manifest(data_root)
    if after_manifest != data_manifest:
        raise RuntimeError("competition data manifest changed during Stage 0")
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": "prefix_inverse_probe_summary_v1",
        "stage0": gate,
        "stage1": {
            "status": "not_run" if not gate["passed"] else "authorized_not_yet_run",
            "quick_incumbent_rmse": 12.8117468534,
            "correct_candidate_rmse": None,
        },
        "decision": (
            "stop_rogii_modeling_no_competitive_signal"
            if not gate["passed"]
            else "stage0_passed_proceed_to_quick"
        ),
        "runtime_seconds": elapsed,
        "peak_memory_mb": _peak_memory_mb(),
        "config_sha256": config_sha,
        "per_well_sha256": sha256_file(per_well_path),
        "run_config_sha256": sha256_file(run_config_path),
        "data_manifest_before": data_manifest,
        "data_manifest_after": after_manifest,
        "data_manifest_equal": True,
        "submission_created": False,
        "submission_uploaded": False,
        "full_training_executed": False,
    }
    summary_path = run_root / "summary.json"
    if args.resume and summary_path.exists():
        original = _read_json(summary_path)
        if (
            original.get("stage0") != gate
            or original.get("per_well_sha256") != summary["per_well_sha256"]
            or original.get("config_sha256") != config_sha
            or original.get("data_manifest_after") != after_manifest
        ):
            raise ValueError("resume reconstruction differs from original Stage 0 run")
        resume_record = {
            "passed": True,
            "runtime_seconds": elapsed,
            "peak_memory_mb": _peak_memory_mb(),
            "per_well_sha256": summary["per_well_sha256"],
            "config_sha256": config_sha,
            "data_manifest_equal": True,
            "checkpoint_count": len(selected),
            "prediction_or_metric_drift": False,
        }
        _write_json(run_root / "resume_verification.json", resume_record)
        result = dict(original)
        result["resume_verification"] = resume_record
    else:
        _write_json(summary_path, summary)
        result = summary
    _write_json(run_root / "data_manifest_after.json", after_manifest)
    return result


def run_visible_test_smoke(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = Path(args.data_root).resolve()
    run_root = Path(args.run_dir).resolve()
    _, config = _load_locked_config(Path(args.config).resolve())
    before = capture_data_manifest(data_root)
    test_wells = enumerate_wells(data_root, "test")
    train_wells = {well.well_id: well for well in enumerate_wells(data_root, "train")}
    rows: list[dict[str, Any]] = []
    for well in test_wells:
        if well.well_id not in train_wells:
            raise ValueError(f"visible test well lacks train-matched typewell: {well.well_id}")
        result = calibrate_prefix_likelihood(
            load_horizontal_inference(well.horizontal_path),
            load_typewell_inference(well.typewell_path),
            well_id=well.well_id,
            control_id=CONTROL_CORRECT,
            config=config,
        )
        rows.append(result.to_dict())
    sample = pd.read_csv(data_root / "sample_submission.csv", dtype={"id": "string"})
    parsed = validate_sample_submission_contract(sample, data_root / "test")
    after = capture_data_manifest(data_root)
    if before != after:
        raise RuntimeError("competition data changed during visible-test smoke")
    audit = {
        "passed": True,
        "dynamic_test_wells": [well.well_id for well in test_wells],
        "test_well_count": len(test_wells),
        "sample_rows": len(parsed),
        "prefix_likelihoods": rows,
        "sample_id_order_validated": True,
        "labels_read": False,
        "submission_created": False,
        "submission_uploaded": False,
        "data_manifest_equal": True,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(run_root / "visible_test_offline_smoke.json", audit)
    return audit


def run_environment(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "requested_device": args.device,
        "resolved_device": _resolved_device(args.device),
        "data_manifest": capture_data_manifest(Path(args.data_root)),
        "train_wells": len(enumerate_wells(Path(args.data_root), "train")),
        "test_wells": len(enumerate_wells(Path(args.data_root), "test")),
    }
    _write_json(Path(args.run_dir) / "environment.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locked prefix-calibrated factor-graph research pivot"
    )
    parser.add_argument(
        "--stage",
        choices=("environment", "observability", "prediction", "visible-test-smoke"),
        default="observability",
    )
    parser.add_argument("--profile", choices=("smoke", "quick", "full"), default="smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path(".runs/cache/prefix-inverse"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--grid-chunk-rows", type=int, default=2048)
    parser.add_argument("--memory-limit-mb", type=int, default=4096)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fold-config", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.profile == "full" and "--profile" not in sys.argv:
        parser.error("full profile must be explicitly requested")
    if args.stage == "prediction":
        gate_path = Path(args.run_dir) / "stage0_gate.json"
        if not gate_path.exists() or _read_json(gate_path).get("passed") is not True:
            raise SystemExit("Stage 0 did not pass; full inverse prediction is fail-closed")
        raise SystemExit("prediction stage is unavailable until the passed Stage 0 gate is audited")
    if args.stage == "environment":
        result = run_environment(args)
    elif args.stage == "visible-test-smoke":
        result = run_visible_test_smoke(args)
    else:
        result = run_observability(args)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
