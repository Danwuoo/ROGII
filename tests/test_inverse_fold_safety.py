from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.data import select_inference_view, validate_prefix


def test_prediction_view_excludes_suffix_target_and_formation_columns() -> None:
    rows = 12
    frame = pd.DataFrame(
        {
            "MD": np.arange(rows),
            "X": np.arange(rows),
            "Y": np.arange(rows),
            "Z": np.arange(rows),
            "GR": np.arange(rows),
            "TVT_input": [*range(8), *([np.nan] * 4)],
            "TVT": np.arange(rows) + 99_000,
            "ANCC": np.arange(rows) + 88_000,
        }
    )
    view = select_inference_view(frame, kind="horizontal")
    assert list(view.columns) == ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    assert "TVT" not in view and "ANCC" not in view
    assert validate_prefix(view["TVT_input"]) == (8, 4)


def test_fold_local_fit_ids_reject_validation_overlap_by_contract() -> None:
    fit_ids = {"a", "b", "c"}
    validation_ids = {"d", "e"}
    assert not fit_ids & validation_ids
    validation_ids.add("a")
    assert fit_ids & validation_ids
