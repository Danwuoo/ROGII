from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.data import (
    FORBIDDEN_FEATURE_COLUMNS,
    HORIZONTAL_INFERENCE_COLUMNS,
    TYPEWELL_INFERENCE_COLUMNS,
    assert_feature_columns_allowed,
    enumerate_wells,
    load_horizontal_inference,
    load_typewell_inference,
    select_inference_view,
    validate_prefix,
)


def test_inference_views_drop_train_only_columns() -> None:
    horizontal = pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "X": [1.0, 2.0],
            "Y": [3.0, 4.0],
            "Z": [5.0, 6.0],
            "GR": [7.0, np.nan],
            "TVT_input": [10.0, np.nan],
            "TVT": [10.0, 10.5],
            "ANCC": [1.0, 1.0],
        }
    )
    typewell = pd.DataFrame(
        {"TVT": [0.0, 1.0], "GR": [1.0, 2.0], "Geology": ["a", "b"]}
    )
    assert tuple(select_inference_view(horizontal, kind="horizontal").columns) == (
        HORIZONTAL_INFERENCE_COLUMNS
    )
    assert tuple(select_inference_view(typewell, kind="typewell").columns) == (
        TYPEWELL_INFERENCE_COLUMNS
    )


@pytest.mark.parametrize("column", sorted(FORBIDDEN_FEATURE_COLUMNS | {"Geology"}))
def test_forbidden_features_are_rejected(column: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_feature_columns_allowed(["MD", column])


def test_prefix_contract_requires_contiguous_known_rows() -> None:
    assert validate_prefix(
        pd.Series([1.0, 2.0, np.nan, np.nan]),
        target=pd.Series([1.0, 2.0, 3.0, 4.0]),
    ) == (2, 2)
    with pytest.raises(ValueError, match="contiguous"):
        validate_prefix(pd.Series([1.0, np.nan, 2.0, np.nan]))


def test_real_data_loaders_are_inference_only(data_root: Path) -> None:
    wells = enumerate_wells(data_root, "train")
    assert len(wells) == 773
    horizontal = load_horizontal_inference(wells[0].horizontal_path)
    typewell = load_typewell_inference(wells[0].typewell_path)
    assert tuple(horizontal.columns) == HORIZONTAL_INFERENCE_COLUMNS
    assert tuple(typewell.columns) == TYPEWELL_INFERENCE_COLUMNS
