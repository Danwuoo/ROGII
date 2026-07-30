from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.external_predictions import (
    audit_external_prediction_files,
    external_prediction_geometry,
    split_submission_id,
)


def _write_inputs(root: Path) -> tuple[Path, Path]:
    sample = pd.DataFrame(
        {
            "id": [
                "alpha_3",
                "alpha_4",
                "alpha_5",
                "long-dynamic-well_9",
                "long-dynamic-well_10",
            ],
            "tvt": 0.0,
        }
    )
    sample_path = root / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)
    dataset = root / "external"
    dataset.mkdir()
    for name, offset in (("model-a.csv", 0.0), ("model-b.csv", 0.2)):
        pd.DataFrame(
            {
                "id": sample["id"],
                "tvt": np.linspace(100.0, 104.0, len(sample)) + offset,
            }
        ).to_csv(dataset / name, index=False)
    return sample_path, dataset


def test_dynamic_ids_and_diagnostic_only_registry(tmp_path: Path) -> None:
    sample_path, dataset = _write_inputs(tmp_path)
    parsed = split_submission_id(pd.read_csv(sample_path)["id"])
    assert set(parsed["well_id"]) == {"alpha", "long-dynamic-well"}
    family, inventory = audit_external_prediction_files(
        dataset_dir=dataset,
        sample_submission_path=sample_path,
    )
    assert len(inventory) == 2
    assert family.manifest()["eligibility"] == "diagnostic_only"
    with pytest.raises(PermissionError):
        family.fit()
    with pytest.raises(PermissionError):
        family.predict_oof()


def test_target_free_geometry_uses_no_labels(tmp_path: Path) -> None:
    sample_path, dataset = _write_inputs(tmp_path)
    family, _ = audit_external_prediction_files(
        dataset_dir=dataset,
        sample_submission_path=sample_path,
    )
    sample = pd.read_csv(sample_path)
    incumbent = pd.DataFrame(
        {
            "id": sample["id"],
            "tvt_incumbent": np.linspace(99.5, 103.5, len(sample)),
        }
    )
    geometry = external_prediction_geometry(
        family=family,
        sample_submission_path=sample_path,
        incumbent=incumbent,
    )
    assert len(geometry["pairwise"]) == 1
    assert len(geometry["vs_incumbent_summary"]) == 2
    assert set(geometry["vs_incumbent_per_well"]["well_id"]) == {
        "alpha",
        "long-dynamic-well",
    }
