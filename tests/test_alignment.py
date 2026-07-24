from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.alignment import (
    ALIGNMENT_FEATURE_COLUMNS,
    AlignmentConfig,
    build_alignment_features,
    resample_typewell_physical,
)


def _horizontal(secret: float = 0.0) -> pd.DataFrame:
    n = 12
    return pd.DataFrame(
        {
            "MD": np.arange(100.0, 112.0),
            "X": np.arange(n, dtype=float),
            "Y": np.linspace(0.0, 2.0, n),
            "Z": np.linspace(10.0, 11.0, n),
            "GR": [10.0, 12.0, np.nan, 18.0, 20.0, np.nan, 28.0, 31.0, 35.0, np.nan, 42.0, 45.0],
            "TVT_input": [50.0, 50.2, 50.4, 50.6, 50.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "TVT": [50.0, 50.2, 50.4, 50.6, 50.8] + [secret + i for i in range(7)],
            "ANCC": secret,
            "ASTNU": secret + 1,
            "ASTNL": secret + 2,
            "EGFDU": secret + 3,
            "EGFDL": secret + 4,
            "BUDA": secret + 5,
        }
    )


def _typewell(secret: str = "x") -> pd.DataFrame:
    tvt = np.arange(40.0, 61.0, 0.5)
    return pd.DataFrame(
        {
            "TVT": tvt,
            "GR": 20.0 + 8.0 * np.sin(tvt / 2.0),
            "Geology": [secret] * len(tvt),
        }
    )


def _config() -> AlignmentConfig:
    return AlignmentConfig()


def test_alignment_contract_is_locked_float32_and_train_test_symmetric() -> None:
    train = build_alignment_features(_horizontal(), _typewell(), config=_config())
    test = build_alignment_features(
        _horizontal().drop(columns=["TVT", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]),
        _typewell().drop(columns=["Geology"]),
        config=_config(),
    )
    assert tuple(train.features.columns) == ALIGNMENT_FEATURE_COLUMNS
    assert set(map(str, train.features.dtypes)) == {"float32"}
    pd.testing.assert_frame_equal(train.features, test.features)
    np.testing.assert_array_equal(train.candidate_tvt, test.candidate_tvt)
    np.testing.assert_array_equal(train.candidate_cost, test.candidate_cost)


def test_alignment_ignores_post_ps_target_formations_geology_and_never_uses_target_for_topk() -> None:
    first = build_alignment_features(_horizontal(0.0), _typewell("a"), config=_config())
    second = build_alignment_features(_horizontal(1_000_000.0), _typewell("b"), config=_config())
    pd.testing.assert_frame_equal(first.features, second.features)
    np.testing.assert_array_equal(first.candidate_tvt, second.candidate_tvt)
    np.testing.assert_array_equal(first.candidate_cost, second.candidate_cost)


def test_typewell_resampling_uses_physical_tvt_not_row_index() -> None:
    irregular = pd.DataFrame({"TVT": [0.0, 1.0, 4.0], "GR": [0.0, 10.0, 40.0]})
    grid, gr, coverage = resample_typewell_physical(irregular, grid_step_ft=1.0)
    np.testing.assert_array_equal(grid, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(gr, [0.0, 10.0, 20.0, 30.0, 40.0])
    np.testing.assert_array_equal(coverage, np.ones(5, dtype=np.float32))


def test_nan_gr_behavior_is_finite_deterministic_and_reports_coverage() -> None:
    horizontal = _horizontal().copy()
    horizontal.loc[5:9, "GR"] = np.nan
    first = build_alignment_features(horizontal, _typewell(), config=_config())
    second = build_alignment_features(horizontal, _typewell(), config=_config())
    assert np.isfinite(first.features.to_numpy()).all()
    pd.testing.assert_frame_equal(first.features, second.features)
    assert (first.features.filter(like="horizontal_coverage").to_numpy() < 1.0).any()


def test_candidate_positions_are_physical_grid_values_and_not_targets() -> None:
    built = build_alignment_features(_horizontal(9999.0), _typewell(), config=_config())
    assert built.candidate_tvt.shape == (7, 3)
    assert built.candidate_cost.shape == (7, 3)
    assert np.isfinite(built.candidate_tvt).all()
    assert np.isfinite(built.candidate_cost).all()
    assert np.allclose(np.mod(built.candidate_tvt, 1.0), 0.0)
    assert not np.isin(built.candidate_tvt, _horizontal(9999.0)["TVT"].iloc[5:]).all()
