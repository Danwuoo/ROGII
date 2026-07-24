import math

import numpy as np
import pandas as pd
import pytest

from rogii.metrics import per_well_diagnostics, pooled_row_rmse, tail_diagnostics


def test_pooled_row_rmse_matches_hand_calculation() -> None:
    actual = pooled_row_rmse([0.0, 2.0, 4.0], [0.0, 1.0, 2.0])
    assert actual == math.sqrt(5.0 / 3.0)


@pytest.mark.parametrize(
    ("truth", "prediction", "message"),
    [([], [], "empty"), ([1.0], [1.0, 2.0], "length"), ([1.0], [np.nan], "finite")],
)
def test_pooled_row_rmse_rejects_invalid_inputs(truth, prediction, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pooled_row_rmse(truth, prediction)


def test_per_well_metrics_are_diagnostics_not_an_averaged_primary_metric() -> None:
    frame = pd.DataFrame(
        {"well_id": ["a", "b", "b", "b"], "y_true": [0.0, 0.0, 0.0, 0.0],
         "y_pred": [2.0, 1.0, 1.0, 1.0]}
    )
    per_well = per_well_diagnostics(frame)
    tail = tail_diagnostics(per_well)
    assert pooled_row_rmse(frame.y_true, frame.y_pred) == math.sqrt(7.0 / 4.0)
    assert per_well.set_index("well_id").loc["a", "rmse"] == 2.0
    assert tail["p50_rmse"] == 1.5


def test_per_well_metrics_reject_null_well_ids() -> None:
    frame = pd.DataFrame({"well_id": ["a", None], "y_true": [0.0, 0.0], "y_pred": [0.0, 1.0]})
    with pytest.raises(ValueError, match="null well_id"):
        per_well_diagnostics(frame)
