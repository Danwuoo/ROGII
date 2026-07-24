from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.submission import (
    align_predictions_to_sample,
    parse_sample_ids,
    validate_sample_submission_contract,
)


def test_parse_sample_ids_rejects_malformed_id() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_sample_ids(pd.Series(["not-an-id"], name="id"))


def test_sample_ids_exactly_match_test_prediction_mask(data_root: Path) -> None:
    sample = pd.read_csv(data_root / "sample_submission.csv", dtype={"id": "string"})
    parsed = validate_sample_submission_contract(sample, data_root / "test")
    assert len(parsed) == 14_151
    assert parsed["well_id"].nunique() == 3


def test_prediction_alignment_uses_ids_not_input_order(data_root: Path) -> None:
    sample = pd.read_csv(data_root / "sample_submission.csv", dtype={"id": "string"})
    predictions = pd.DataFrame(
        {"id": sample["id"].iloc[::-1].to_numpy(),
         "tvt": np.arange(len(sample), dtype=float)[::-1]}
    )
    aligned = align_predictions_to_sample(sample, predictions)
    assert aligned["id"].tolist() == sample["id"].tolist()
    np.testing.assert_array_equal(aligned["tvt"].to_numpy(), np.arange(len(sample)))


def test_prediction_alignment_rejects_missing_ids(data_root: Path) -> None:
    sample = pd.read_csv(data_root / "sample_submission.csv", dtype={"id": "string"})
    predictions = sample[["id"]].iloc[:-1].assign(tvt=0.0)
    with pytest.raises(ValueError, match="missing"):
        align_predictions_to_sample(sample, predictions)


def test_sample_contract_rejects_noncontiguous_prediction_mask(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    pd.DataFrame(
        {
            "MD": range(4), "X": range(4), "Y": range(4), "Z": range(4),
            "GR": [1.0] * 4, "TVT_input": [1.0, np.nan, 2.0, np.nan],
        }
    ).to_csv(test_dir / "deadbeef__horizontal_well.csv", index=False)
    pd.DataFrame({"TVT": [0.0, 0.5], "GR": [1.0, 2.0]}).to_csv(
        test_dir / "deadbeef__typewell.csv", index=False
    )
    sample = pd.DataFrame({"id": ["deadbeef_1", "deadbeef_3"], "tvt": [0.0, 0.0]})
    with pytest.raises(ValueError, match="contiguous"):
        validate_sample_submission_contract(sample, test_dir)
