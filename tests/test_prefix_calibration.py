from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from rogii.inverse_solver.contracts import PrefixCalibrationConfig, PrefixLikelihoodResult
from rogii.inverse_solver.prefix_calibration import calibrate_prefix_likelihood


def _frames(
    *,
    missing: bool = False,
    typewell_lower: float = 980.0,
    typewell_upper: float = 1150.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 180
    known = 132
    md = np.arange(rows, dtype=np.float64)
    tvt = 1_000.0 + 0.7 * md
    base = lambda value: 75.0 + 14.0 * np.sin(value / 5.5) + 5.0 * np.cos(value / 13.0)
    gr = 8.0 + 1.25 * base(tvt + 2.0)
    if missing:
        gr[35:48] = np.nan
        gr[80:87] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 2_000.0 + md,
            "Y": 5_000.0 - md,
            "Z": -8_000.0 - md * 0.2,
            "GR": gr,
            "TVT_input": np.where(md < known, tvt, np.nan),
            "TVT": tvt,
            "ANCC": np.arange(rows) + 1.0e9,
        }
    )
    type_tvt = np.arange(typewell_lower, typewell_upper, 0.5)
    typewell = pd.DataFrame(
        {
            "TVT": type_tvt,
            "GR": base(type_tvt),
            "Geology": "forbidden",
        }
    )
    return horizontal, typewell


def _config() -> PrefixCalibrationConfig:
    return PrefixCalibrationConfig(
        minimum_prefix_pairs=24,
        minimum_evaluation_pairs=10,
        maximum_resampled_points=512,
        lag_values_ft=(-4.0, -2.0, 0.0, 2.0, 4.0),
        smoothing_scales_ft=(5.0, 11.0),
        bootstrap_replicates=200,
    )


def test_prefix_calibration_is_finite_deterministic_and_serializable() -> None:
    horizontal, typewell = _frames(missing=True)
    first = calibrate_prefix_likelihood(
        horizontal,
        typewell,
        well_id="deadbeef",
        config=_config(),
    )
    second = calibrate_prefix_likelihood(
        horizontal,
        typewell,
        well_id="deadbeef",
        config=_config(),
    )
    assert first == second
    assert np.isfinite(list(first.to_dict().values())[3:]).all()
    restored = PrefixLikelihoodResult.from_dict(first.to_dict())
    assert restored == first
    assert pickle.loads(pickle.dumps(first)) == first


def test_suffix_target_and_forbidden_columns_never_change_prefix_likelihood() -> None:
    horizontal, typewell = _frames()
    expected = calibrate_prefix_likelihood(
        horizontal, typewell, well_id="deadbeef", config=_config()
    )
    changed_h = horizontal.copy()
    changed_h["TVT"] = np.linspace(-1.0e12, 1.0e12, len(changed_h))
    changed_h["ANCC"] = -changed_h["ANCC"]
    changed_t = typewell.copy()
    changed_t["Geology"] = "secret-changed"
    actual = calibrate_prefix_likelihood(
        changed_h, changed_t, well_id="deadbeef", config=_config()
    )
    assert actual == expected


def test_prefix_mask_must_be_contiguous_and_suffix_is_not_consumed() -> None:
    horizontal, typewell = _frames()
    horizontal.loc[20, "TVT_input"] = np.nan
    with pytest.raises(ValueError, match="contiguous"):
        calibrate_prefix_likelihood(
            horizontal, typewell, well_id="deadbeef", config=_config()
        )


def test_missing_gr_blocks_are_supported_but_insufficient_coverage_fails_closed() -> None:
    horizontal, typewell = _frames(missing=True)
    result = calibrate_prefix_likelihood(
        horizontal, typewell, well_id="deadbeef", config=_config()
    )
    assert result.gr_missing_fraction > 0.0
    _, uncovered = _frames(typewell_lower=2_000.0, typewell_upper=2_100.0)
    with pytest.raises(ValueError, match="lag/stretch"):
        calibrate_prefix_likelihood(
            horizontal, uncovered, well_id="deadbeef", config=_config()
        )
