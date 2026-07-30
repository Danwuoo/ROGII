from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.external_models.artifacts import capture_data_manifest, sha256_file


def _write_well(root: Path, split: str, well_id: str, phase: float) -> None:
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    rows = 150
    known = 110
    md = np.arange(rows, dtype=np.float64)
    tvt = 1000.0 + 0.8 * md
    signal = lambda value: 70.0 + 10.0 * np.sin(value / 5.0 + phase)
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": md + 100.0,
            "Y": md + 200.0,
            "Z": -md,
            "GR": 5.0 + 1.2 * signal(tvt + 2.0),
            "TVT_input": np.where(md < known, tvt, np.nan),
            "TVT": tvt,
            "ANCC": tvt + 999.0,
        }
    )
    tw_tvt = np.arange(970.0, 1160.0, 0.5)
    typewell = pd.DataFrame(
        {"TVT": tw_tvt, "GR": signal(tw_tvt), "Geology": "forbidden"}
    )
    horizontal.to_csv(directory / f"{well_id}__horizontal_well.csv", index=False)
    typewell.to_csv(directory / f"{well_id}__typewell.csv", index=False)


def test_smoke_checkpoint_resume_cpu_and_data_manifest(tmp_path: Path) -> None:
    script_path = Path("scripts/run_prefix_calibrated_inverse_probe.py").resolve()
    specification = importlib.util.spec_from_file_location(
        "run_prefix_calibrated_inverse_probe", script_path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    run_observability = module.run_observability

    data_root = tmp_path / "data"
    well_ids = [f"{index + 1:08x}" for index in range(5)]
    for index, well_id in enumerate(well_ids):
        _write_well(data_root, "train", well_id, phase=0.17 * index)
    folds = pd.DataFrame(
        {
            "well_id": well_ids,
            "fold": range(5),
            "fold_version": "toy_v1",
        }
    )
    fold_path = tmp_path / "folds.csv"
    folds.to_csv(fold_path, index=False)
    source_config = json.loads(
        Path("configs/inverse/prefix_calibrated_probe_20260729.json").read_text(
            encoding="utf-8"
        )
    )
    source_config["prefix_calibration"]["bootstrap_replicates"] = 200
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(source_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    args = argparse.Namespace(
        data_root=data_root,
        config=config_path,
        run_dir=run_dir,
        fold_config=fold_path,
        profile="smoke",
        device="cpu",
        seed=20260729,
        num_workers=1,
        grid_chunk_rows=256,
        memory_limit_mb=512,
        resume=False,
    )
    before = capture_data_manifest(data_root)
    first = run_observability(args)
    first_hash = sha256_file(run_dir / "stage0_observability_per_well.csv.gz")
    args.resume = True
    second = run_observability(args)
    assert first["stage0"] == second["stage0"]
    assert first_hash == sha256_file(run_dir / "stage0_observability_per_well.csv.gz")
    assert capture_data_manifest(data_root) == before
    assert first["submission_created"] is False
    assert first["full_training_executed"] is False
    assert len(list((run_dir / "checkpoints" / "stage0").glob("*.json"))) == 5
    audit_script = Path(
        "scripts/audit_prefix_calibrated_inverse_probe.py"
    ).resolve()
    audit_specification = importlib.util.spec_from_file_location(
        "audit_prefix_calibrated_inverse_probe", audit_script
    )
    assert (
        audit_specification is not None
        and audit_specification.loader is not None
    )
    audit_module = importlib.util.module_from_spec(audit_specification)
    audit_specification.loader.exec_module(audit_module)
    independent = audit_module.audit(run_dir)
    assert independent["passed"] is True
    assert independent["checkpoint_reconstruction_passed"] is True
