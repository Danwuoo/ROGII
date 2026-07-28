from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from rogii.heel_registration import (
    HEEL_REGISTRATION_VERSION,
    HeelRegistrationConfig,
    clipped_quantile_loss,
    estimate_heel_registration,
    estimate_heel_shift,
    heel_registration_manifest,
)


TRUE_SHIFT = 12.5


def _typewell_signal(tvt: np.ndarray) -> np.ndarray:
    return 30.0 + 0.08 * tvt + 7.0 * np.sin(tvt / 11.0) + 3.0 * np.sin(tvt / 3.7)


def _synthetic_frames(
    *,
    target_secret: float = 0.0,
    geology_secret: str = "secret-a",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    typewell_tvt = np.arange(0.0, 301.0, 0.25)
    typewell_gr = _typewell_signal(typewell_tvt)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": typewell_gr,
            "Geology": [geology_secret] * len(typewell_tvt),
        }
    )

    prefix_count = 80
    suffix_count = 240
    prefix_tvt = np.linspace(70.0, 110.0, prefix_count)
    base_prediction = np.linspace(100.0, 220.0, suffix_count)
    true_suffix_tvt = base_prediction + TRUE_SHIFT
    # The exact horizontal->typewell tool response is y = 5 + 1.6*x.
    horizontal_gr = np.concatenate(
        (
            (_typewell_signal(prefix_tvt) - 5.0) / 1.6,
            (_typewell_signal(true_suffix_tvt) - 5.0) / 1.6,
        )
    )
    row_count = prefix_count + suffix_count
    target = np.concatenate((prefix_tvt, true_suffix_tvt))
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(row_count, dtype=np.float64),
            "X": np.linspace(10.0, 20.0, row_count),
            "Y": np.linspace(-5.0, 5.0, row_count),
            "Z": np.linspace(100.0, 90.0, row_count),
            "GR": horizontal_gr,
            "TVT_input": np.concatenate((prefix_tvt, np.full(suffix_count, np.nan))),
            "TVT": target + target_secret,
            "ANCC": np.full(row_count, target_secret + 100.0),
            "ASTNU": np.full(row_count, target_secret + 200.0),
            "ASTNL": np.full(row_count, target_secret + 300.0),
            "EGFDU": np.full(row_count, target_secret + 400.0),
            "EGFDL": np.full(row_count, target_secret + 500.0),
            "BUDA": np.full(row_count, target_secret + 600.0),
        }
    )
    return horizontal, typewell, base_prediction


@pytest.mark.parametrize(
    "overrides",
    [
        {"shift_min_ft": 1.0, "shift_max_ft": 1.0},
        {"shift_min_ft": "-40"},
        {"shift_step_ft": 0.0},
        {"shift_step_ft": 0.3},
        {"rolling_median_width": 2},
        {"rolling_median_width": True},
        {"max_suffix_rows": 0},
        {"quantile_count": 1},
        {"squared_loss_cap": 0.0},
        {"minimum_coverage": 1.1},
        {"coverage_penalty_scale": -1.0},
        {"posterior_temperature_multiplier": 0.0},
        {"posterior_mse_floor": np.nan},
        {"minimum_prefix_pairs": 1},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HeelRegistrationConfig(**overrides)


def test_inference_schema_ignores_targets_formations_geology_and_png(
    tmp_path,
) -> None:
    first_horizontal, first_typewell, base = _synthetic_frames(
        target_secret=0.0, geology_secret="not-used-a"
    )
    first = estimate_heel_registration(first_horizontal, first_typewell, base)

    (tmp_path / "forbidden-diagnostic.png").write_bytes(
        b"PNG data that must never be read"
    )
    second_horizontal, second_typewell, second_base = _synthetic_frames(
        target_secret=1_000_000.0, geology_secret="not-used-b"
    )
    second = estimate_heel_registration(second_horizontal, second_typewell, second_base)

    assert first.posterior_mean_shift == pytest.approx(
        second.posterior_mean_shift, abs=0.0
    )
    assert first.map_shift == second.map_shift
    assert first.calibration == second.calibration
    np.testing.assert_array_equal(first.candidate_cost, second.candidate_cost)
    np.testing.assert_array_equal(first.posterior_weight, second.posterior_weight)
    manifest = heel_registration_manifest()
    assert manifest["version"] == HEEL_REGISTRATION_VERSION
    assert manifest["horizontal_columns_used"] == ["GR", "TVT_input"]
    assert manifest["typewell_source_columns"] == ["TVT", "GR"]


def test_synthetic_shift_is_recovered_and_diagnostics_are_finite() -> None:
    horizontal, typewell, base = _synthetic_frames()
    result = estimate_heel_registration(horizontal, typewell, base)

    assert result.map_shift == pytest.approx(TRUE_SHIFT, abs=0.5)
    assert result.posterior_mean_shift == pytest.approx(TRUE_SHIFT, abs=0.75)
    assert estimate_heel_shift(horizontal, typewell, base) == pytest.approx(
        result.posterior_mean_shift, abs=0.0
    )
    assert result.calibration.pair_count == 80
    assert result.calibration.prefix_coverage == 1.0
    assert result.calibration.prefix_mse >= 0.0
    assert result.shift_grid.shape == (161,)
    assert result.candidate_cost.shape == result.shift_grid.shape
    assert result.candidate_coverage.shape == result.shift_grid.shape
    assert result.candidate_support_count.shape == result.shift_grid.shape
    assert result.posterior_weight.shape == result.shift_grid.shape
    assert np.isfinite(result.candidate_cost).all()
    assert np.isfinite(result.posterior_weight).all()
    assert result.posterior_weight.sum() == pytest.approx(1.0, abs=1e-14)
    assert result.best_coverage == 1.0
    assert result.posterior_mean_coverage == pytest.approx(1.0, abs=1e-14)
    assert 1.0 <= result.effective_support <= len(result.shift_grid)
    assert 0.0 <= result.edge_mass <= 1.0


def test_distributional_loss_is_invariant_to_arbitrary_suffix_shuffle() -> None:
    rng = np.random.default_rng(20260728)
    horizontal_values = rng.normal(size=500)
    typewell_values = rng.normal(loc=0.2, scale=1.3, size=500)
    first = clipped_quantile_loss(horizontal_values, typewell_values)

    left_order = rng.permutation(len(horizontal_values))
    right_order = rng.permutation(len(typewell_values))
    shuffled = clipped_quantile_loss(
        horizontal_values[left_order], typewell_values[right_order]
    )
    assert shuffled == pytest.approx(first, rel=0.0, abs=1e-15)


def test_complete_suffix_reversal_preserves_end_to_end_distributional_result() -> None:
    horizontal, typewell, base = _synthetic_frames()
    original = estimate_heel_registration(horizontal, typewell, base)
    known_count = int(horizontal["TVT_input"].notna().sum())
    reversed_horizontal = horizontal.copy()
    reversed_horizontal.loc[known_count:, "GR"] = horizontal.loc[
        known_count:, "GR"
    ].to_numpy()[::-1]
    reversed_result = estimate_heel_registration(
        reversed_horizontal, typewell, base[::-1]
    )
    np.testing.assert_allclose(
        reversed_result.candidate_cost,
        original.candidate_cost,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        reversed_result.posterior_weight,
        original.posterior_weight,
        rtol=0.0,
        atol=1e-14,
    )
    assert reversed_result.posterior_mean_shift == pytest.approx(
        original.posterior_mean_shift, abs=1e-13
    )


def test_wrong_typewell_breaks_synthetic_shift_recovery() -> None:
    horizontal, typewell, base = _synthetic_frames()
    correct = estimate_heel_registration(horizontal, typewell, base)
    wrong = typewell.copy()
    tvt = wrong["TVT"].to_numpy(dtype=np.float64)
    wrong["GR"] = 30.0 + 8.0 * np.cos(tvt / 17.0) + 2.0 * np.sin(tvt / 2.5)
    wrong_result = estimate_heel_registration(horizontal, wrong, base)

    assert abs(correct.posterior_mean_shift - TRUE_SHIFT) < 0.75
    assert abs(wrong_result.posterior_mean_shift - TRUE_SHIFT) > 5.0
    assert abs(wrong_result.map_shift - TRUE_SHIFT) > 5.0


def test_missing_gr_is_interpolated_and_large_suffix_is_sampled_deterministically() -> (
    None
):
    horizontal, typewell, base = _synthetic_frames()
    horizontal.loc[[2, 3, 20, 95, 96, 150, 270], "GR"] = np.nan
    config = HeelRegistrationConfig(max_suffix_rows=100)
    first = estimate_heel_registration(horizontal, typewell, base, config=config)
    second = estimate_heel_registration(horizontal, typewell, base, config=config)

    assert len(first.sampled_suffix_row_index) == 100
    assert first.sampled_suffix_row_index[0] == 0
    assert first.sampled_suffix_row_index[-1] == len(base) - 1
    assert np.all(np.diff(first.sampled_suffix_row_index) > 0)
    assert np.isfinite(first.posterior_mean_shift)
    np.testing.assert_array_equal(
        first.sampled_suffix_row_index, second.sampled_suffix_row_index
    )
    np.testing.assert_array_equal(first.candidate_cost, second.candidate_cost)


def test_zero_candidate_support_has_finite_symmetric_posterior_fallback() -> None:
    horizontal, typewell, base = _synthetic_frames()
    outside = np.full_like(base, 10_000.0)
    result = estimate_heel_registration(horizontal, typewell, outside)

    assert np.all(result.candidate_coverage == 0.0)
    assert np.all(result.candidate_support_count == 0)
    assert np.all(result.candidate_cost == result.candidate_cost[0])
    np.testing.assert_allclose(
        result.posterior_weight,
        np.full(161, 1.0 / 161.0),
        rtol=0.0,
        atol=1e-16,
    )
    assert result.posterior_mean_shift == pytest.approx(0.0, abs=1e-13)
    assert result.posterior_mean_coverage == 0.0
    assert np.isfinite(result.posterior_entropy)


def test_public_dataclasses_are_pickle_safe() -> None:
    horizontal, typewell, base = _synthetic_frames()
    config = HeelRegistrationConfig(max_suffix_rows=50)
    result = estimate_heel_registration(horizontal, typewell, base, config=config)
    restored_config, restored_result = pickle.loads(pickle.dumps((config, result)))

    assert restored_config == config
    assert restored_result.calibration == result.calibration
    assert restored_result.posterior_mean_shift == result.posterior_mean_shift
    np.testing.assert_array_equal(
        restored_result.posterior_weight, result.posterior_weight
    )


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("missing_column", "Missing required horizontal"),
        ("noncontiguous_prefix", "contiguous known prefix"),
        ("base_wrong_length", "length must equal"),
        ("base_matrix", "one-dimensional"),
        ("base_nan", "must be finite"),
        ("horizontal_infinity", "must not contain infinity"),
        ("prefix_all_nan_gr", "requires at least one finite"),
        ("suffix_all_nan_gr", "requires at least one finite"),
        ("typewell_unsorted", "strictly increasing"),
        ("typewell_all_nan_gr", "at least two finite"),
        ("prefix_outside_typewell", "fewer supported"),
    ],
)
def test_invalid_inputs_are_rejected(mutation: str, match: str) -> None:
    horizontal, typewell, base = _synthetic_frames()
    if mutation == "missing_column":
        horizontal = horizontal.drop(columns=["Y"])
    elif mutation == "noncontiguous_prefix":
        horizontal.loc[10, "TVT_input"] = np.nan
    elif mutation == "base_wrong_length":
        base = base[:-1]
    elif mutation == "base_matrix":
        base = base[:, None]
    elif mutation == "base_nan":
        base = base.copy()
        base[0] = np.nan
    elif mutation == "horizontal_infinity":
        horizontal.loc[0, "GR"] = np.inf
    elif mutation == "prefix_all_nan_gr":
        known_count = int(horizontal["TVT_input"].notna().sum())
        horizontal.loc[: known_count - 1, "GR"] = np.nan
    elif mutation == "suffix_all_nan_gr":
        known_count = int(horizontal["TVT_input"].notna().sum())
        horizontal.loc[known_count:, "GR"] = np.nan
    elif mutation == "typewell_unsorted":
        typewell.loc[1, "TVT"] = typewell.loc[0, "TVT"]
    elif mutation == "typewell_all_nan_gr":
        typewell["GR"] = np.nan
    elif mutation == "prefix_outside_typewell":
        known_count = int(horizontal["TVT_input"].notna().sum())
        horizontal.loc[: known_count - 1, "TVT_input"] += 10_000.0
    else:
        raise AssertionError(f"unknown test mutation: {mutation}")

    with pytest.raises(ValueError, match=match):
        estimate_heel_registration(horizontal, typewell, base)


def test_duplicate_columns_are_rejected() -> None:
    horizontal, typewell, base = _synthetic_frames()
    duplicated = pd.concat((horizontal, horizontal[["GR"]]), axis=1)
    with pytest.raises(ValueError, match="Duplicate columns"):
        estimate_heel_registration(duplicated, typewell, base)
