from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.records import record_run
from rogii.runner import check_data, run_experiment, submit_run


def _write_well(root: Path, split: str, well_id: str, offset: float) -> list[str]:
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    rows = 9
    md = np.arange(rows, dtype=float)
    truth = 150.0 + offset + 0.2 * md
    tvt_input = truth.copy()
    tvt_input[5:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": md + offset,
            "Y": 2.0 * md,
            "Z": -md,
            "GR": 50.0 + np.sin(md),
            "TVT_input": tvt_input,
        }
    )
    if split == "train":
        horizontal["TVT"] = truth
    horizontal.to_csv(
        directory / f"{well_id}__horizontal_well.csv", index=False
    )
    tvt = np.arange(0.0, 301.0, 1.0)
    pd.DataFrame({"TVT": tvt, "GR": 50.0 + np.sin(tvt / 20.0)}).to_csv(
        directory / f"{well_id}__typewell.csv", index=False
    )
    return [f"{well_id}_{index}" for index in range(5, rows)]


def _synthetic(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "competition"
    train_ids = [f"{index:08x}" for index in range(10)]
    for index, well_id in enumerate(train_ids):
        _write_well(data, "train", well_id, float(index))
    test_ids = _write_well(data, "test", "000000ff", 20.0)
    pd.DataFrame({"id": test_ids, "tvt": 0.0}).to_csv(
        data / "sample_submission.csv", index=False
    )
    folds = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            "well_id": train_ids,
            "fold": [index % 5 for index in range(10)],
            "fold_version": "synthetic_5fold_v1",
        }
    ).to_csv(folds, index=False)
    return data, folds


def test_check_flat_full_submit_and_record(tmp_path: Path) -> None:
    data, folds = _synthetic(tmp_path)
    checked = check_data(data_root=data, fold_config=folds)
    assert checked["train_wells"] == 10
    assert checked["test_wells"] == 1

    result = run_experiment(
        model_id="flat",
        profile="full",
        data_root=data,
        fold_config=folds,
        output_root=tmp_path / "runs",
        run_id="flat-full",
        workers=2,
    )
    assert result["summary"]["scope"] == "full"
    assert result["summary"]["oof_wells"] == 10

    submitted = submit_run(
        run_root=result["run_root"],
        data_root=data,
        output=tmp_path / "submission.csv",
        workers=2,
    )
    assert submitted["submission_rows"] == 4

    recorded = record_run(
        run_root=result["run_root"],
        results_root=tmp_path / "results",
        note="synthetic",
    )
    payload = json.loads(Path(recorded["record_path"]).read_text(encoding="utf-8"))
    assert payload["summary"]["run_id"] == "flat-full"


def test_p3_quick_is_explicitly_not_full_comparable(tmp_path: Path) -> None:
    data, folds = _synthetic(tmp_path)
    result = run_experiment(
        model_id="p3",
        profile="quick",
        data_root=data,
        fold_config=folds,
        output_root=tmp_path / "runs",
        run_id="p3-quick",
        workers=2,
    )
    assert result["summary"]["scope"] == "quick_or_custom"
    assert result["summary"]["comparable_to_historical_full_oof"] is False
    assert np.isfinite(result["summary"]["pooled_row_rmse"])
