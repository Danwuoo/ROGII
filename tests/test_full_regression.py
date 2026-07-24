from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.data import enumerate_wells, load_horizontal_inference, validate_prefix
from rogii.metrics import pooled_row_rmse


@pytest.mark.slow
def test_flat_last_known_exact_full_train_regression(data_root: Path) -> None:
    truth_blocks: list[np.ndarray] = []
    prediction_blocks: list[np.ndarray] = []
    for well in enumerate_wells(data_root, "train"):
        horizontal = load_horizontal_inference(well.horizontal_path)
        labels = pd.read_csv(well.horizontal_path, usecols=["TVT", "TVT_input"])
        known, predicted = validate_prefix(
            labels["TVT_input"], target=labels["TVT"]
        )
        truth_blocks.append(
            labels["TVT"].iloc[known:].to_numpy(dtype=np.float64)
        )
        prediction_blocks.append(
            np.full(predicted, float(horizontal["TVT_input"].iloc[known - 1]))
        )
    truth = np.concatenate(truth_blocks)
    prediction = np.concatenate(prediction_blocks)
    assert len(truth) == 3_783_989
    assert pooled_row_rmse(truth, prediction) == pytest.approx(
        15.909852870734552, abs=1e-12
    )
