from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.data import WellFiles
from rogii.models import get_model
from rogii.models.base import RunContext, TrainingWell
from rogii.models.probabilistic_lattice import ProbabilisticLatticeModel
from rogii.probabilistic_lattice import (
    LatticeConfig,
    LatticeStatistics,
    PrefixCalibration,
    build_emission_log_potential,
    derive_prefix_calibration,
    infer_signed_tvt_lattice,
    transition_log_weights,
)


def _calibration(
    *, rate_location: float = 0.0, rate_count: int = 0
) -> PrefixCalibration:
    return PrefixCalibration(
        gr_intercept=0.0,
        gr_slope=1.0,
        gr_pair_count=10,
        rate_location=rate_location,
        rate_count=rate_count,
    )


def _statistics(
    *,
    transition_location: float = 0.0,
    transition_scale: float = 1.0,
    emission_scale: float = 0.5,
    emission_floor: float = 0.01,
) -> LatticeStatistics:
    return LatticeStatistics(
        transition_rate_location=transition_location,
        transition_rate_scale=transition_scale,
        transition_sample_count=100,
        emission_residual_location=0.0,
        emission_residual_scale=emission_scale,
        emission_floor_probability=emission_floor,
        emission_sample_count=100,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_move_bins": 4},
        {"max_move_bins": 1.5},
        {"edge_bins": 1.5},
        {"emission_chunk_rows": 0},
        {"emission_floor_probability": 0.5},
        {"transition_df": np.nan},
    ],
)
def test_malformed_lattice_config_is_rejected(
    overrides: dict[str, float],
) -> None:
    parameters: dict[str, float] = {
        "state_step_ft": 1.0,
        "state_radius_ft": 3.0,
        "max_move_bins": 1,
    }
    parameters.update(overrides)
    with pytest.raises(ValueError):
        LatticeConfig(**parameters)


def test_forward_backward_posterior_normalizes_and_matches_bruteforce() -> None:
    config = LatticeConfig(
        state_step_ft=1.0,
        state_radius_ft=2.0,
        max_move_bins=1,
        transition_rate_scale_floor=0.01,
        emission_scale_floor=0.01,
        emission_floor_probability=0.001,
        edge_bins=1,
    )
    statistics = _statistics(
        transition_location=0.15,
        transition_scale=0.8,
        emission_scale=0.35,
        emission_floor=0.001,
    )
    calibration = _calibration()
    typewell_tvt = np.arange(-3.0, 4.0)
    typewell_gr = np.array([2.0, -1.0, 1.5, 0.0, 0.8, -0.5, 2.5])
    md = np.array([0.75, 2.0, 3.5])
    observed = np.array([0.7, -0.2, 1.2])

    result = infer_signed_tvt_lattice(
        md,
        observed,
        anchor_md=0.0,
        anchor_tvt=0.0,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        statistics=statistics,
        calibration=calibration,
        config=config,
    )
    np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-12)
    assert np.isfinite(result.posterior).all()
    assert np.isfinite(result.log_evidence)

    emission = build_emission_log_potential(
        observed,
        result.state_grid,
        typewell_tvt,
        typewell_gr,
        statistics,
        calibration,
        config=config,
    )
    delta_md = np.diff(np.concatenate(([0.0], md)))
    weights_by_row: list[dict[int, float]] = []
    for step in delta_md:
        moves, log_weights = transition_log_weights(
            float(step), statistics, calibration, config=config
        )
        weights_by_row.append(
            {
                int(move): float(weight)
                for move, weight in zip(moves, log_weights, strict=True)
            }
        )
    anchor_index = len(result.state_grid) // 2
    paths: list[tuple[tuple[int, ...], float]] = []
    for indices in product(range(len(result.state_grid)), repeat=len(md)):
        deltas = (indices[0] - anchor_index,) + tuple(
            indices[row] - indices[row - 1] for row in range(1, len(md))
        )
        if any(
            delta not in weights_by_row[row]
            for row, delta in enumerate(deltas)
        ):
            continue
        log_weight = sum(
            weights_by_row[row][delta] for row, delta in enumerate(deltas)
        )
        log_weight += sum(emission[row, state] for row, state in enumerate(indices))
        paths.append((indices, float(np.exp(log_weight))))
    normalizer = sum(weight for _, weight in paths)
    brute = np.zeros_like(result.posterior)
    for indices, weight in paths:
        for row, state in enumerate(indices):
            brute[row, state] += weight / normalizer
    np.testing.assert_allclose(result.posterior, brute, rtol=1e-11, atol=1e-12)
    assert result.log_evidence == pytest.approx(np.log(normalizer), abs=1e-11)
    best_indices, _ = max(paths, key=lambda item: item[1])
    np.testing.assert_allclose(
        result.map_path, result.state_grid[np.asarray(best_indices)]
    )


def test_transition_band_explicitly_contains_signed_and_zero_moves() -> None:
    config = LatticeConfig(
        state_step_ft=1.0,
        state_radius_ft=4.0,
        max_move_bins=2,
        transition_rate_scale_floor=0.01,
    )
    moves, log_weights = transition_log_weights(
        1.5,
        _statistics(transition_scale=0.7),
        _calibration(rate_location=-0.25, rate_count=30),
        config=config,
    )
    assert np.array_equal(moves, np.array([-2, -1, 0, 1, 2]))
    assert np.any(moves < 0) and np.any(moves == 0) and np.any(moves > 0)
    np.testing.assert_allclose(np.exp(log_weights).sum(), 1.0, atol=1e-14)

    typewell_tvt = np.arange(-5.0, 6.0)
    result = infer_signed_tvt_lattice(
        np.array([1.0, 2.0, 3.0]),
        np.array([-1.0, -2.0, -3.0]),
        anchor_md=0.0,
        anchor_tvt=0.0,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_tvt,
        statistics=_statistics(transition_scale=2.0, emission_scale=0.05),
        calibration=_calibration(),
        config=LatticeConfig(
            state_step_ft=1.0,
            state_radius_ft=4.0,
            max_move_bins=1,
            transition_rate_scale_floor=0.01,
            emission_scale_floor=0.01,
            emission_floor_probability=1e-5,
        ),
    )
    np.testing.assert_allclose(result.map_path, [-1.0, -2.0, -3.0])


def test_visible_prefix_calibrates_gr_and_signed_rate_deterministically() -> None:
    config = LatticeConfig(
        state_step_ft=1.0,
        state_radius_ft=3.0,
        max_move_bins=1,
    )
    md = np.arange(6.0)
    tvt_input = np.array([-2.0, -1.0, 0.0, np.nan, np.nan, np.nan])
    typewell_tvt = np.arange(-5.0, 6.0)
    typewell_gr = 10.0 + 2.0 * typewell_tvt
    horizontal_gr = 3.0 + 1.5 * np.interp(
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
        typewell_tvt,
        typewell_gr,
    )
    calibration = derive_prefix_calibration(
        md,
        horizontal_gr,
        tvt_input,
        typewell_tvt,
        typewell_gr,
        config=config,
    )
    assert calibration.gr_slope == pytest.approx(1.5)
    assert calibration.gr_intercept == pytest.approx(3.0)
    assert calibration.rate_location == pytest.approx(1.0)
    assert calibration.rate_count == 2
    assert calibration.gr_pair_count == 3


def test_missing_horizontal_and_typewell_gr_are_neutral_not_nan() -> None:
    config = LatticeConfig(
        state_step_ft=1.0,
        state_radius_ft=3.0,
        max_move_bins=1,
        transition_rate_scale_floor=0.01,
        emission_scale_floor=0.01,
    )
    result = infer_signed_tvt_lattice(
        np.array([1.0, 2.0, 3.0]),
        np.array([np.nan, 12.0, np.nan]),
        anchor_md=0.0,
        anchor_tvt=10.0,
        typewell_tvt=np.arange(0.0, 21.0),
        typewell_gr=np.full(21, np.nan),
        statistics=_statistics(),
        calibration=_calibration(),
        config=config,
    )
    assert np.isfinite(result.posterior).all()
    assert np.isfinite(result.posterior_mean).all()
    assert np.isfinite(result.entropy).all()
    assert np.isfinite(result.edge_mass).all()
    np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-12)


def test_exact_anchor_prior_prevents_first_row_teleportation() -> None:
    config = LatticeConfig(
        state_step_ft=1.0,
        state_radius_ft=5.0,
        max_move_bins=1,
        transition_rate_scale_floor=0.01,
        emission_scale_floor=0.01,
        emission_floor_probability=1e-6,
    )
    typewell_tvt = np.arange(-6.0, 7.0)
    typewell_gr = np.zeros_like(typewell_tvt)
    typewell_gr[typewell_tvt == 5.0] = 100.0
    result = infer_signed_tvt_lattice(
        np.array([1.0, 2.0, 3.0]),
        np.array([100.0, 100.0, 100.0]),
        anchor_md=0.0,
        anchor_tvt=0.0,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        statistics=_statistics(transition_scale=10.0, emission_scale=0.05),
        calibration=_calibration(),
        config=config,
    )
    unreachable = np.abs(result.state_grid) > 1.0
    assert np.all(result.posterior[0, unreachable] == 0.0)
    assert abs(result.map_path[0]) <= 1.0
    complete_map = np.concatenate(([0.0], result.map_path))
    assert np.all(np.abs(np.diff(complete_map)) <= 1.0)


def test_posterior_mean_is_squared_loss_mbr_and_can_differ_from_map() -> None:
    config = LatticeConfig(
        state_step_ft=1.0,
        state_radius_ft=2.0,
        max_move_bins=2,
        transition_rate_scale_floor=0.001,
        emission_scale_floor=0.001,
        emission_floor_probability=1e-8,
        edge_bins=1,
    )
    states = np.arange(-2.0, 3.0)
    result = infer_signed_tvt_lattice(
        np.array([1.0]),
        np.array([1.0]),
        anchor_md=0.0,
        anchor_tvt=0.0,
        typewell_tvt=states,
        typewell_gr=states * states,
        statistics=_statistics(
            transition_scale=100.0,
            emission_scale=0.02,
            emission_floor=1e-8,
        ),
        calibration=_calibration(),
        config=config,
    )
    assert result.posterior_mean[0] == pytest.approx(0.0, abs=1e-12)
    assert abs(result.map_path[0]) == pytest.approx(1.0)
    assert not np.isclose(result.posterior_mean[0], result.map_path[0])
    probabilities = result.posterior[0]
    mean_loss = np.sum(
        probabilities * (result.state_grid - result.posterior_mean[0]) ** 2
    )
    map_loss = np.sum(
        probabilities * (result.state_grid - result.map_path[0]) ** 2
    )
    assert mean_loss < map_loss


def test_model_wrapper_fits_only_training_statistics_and_predicts_mbr(
    tmp_path: Path,
) -> None:
    assert get_model("lattice").model_id == "lattice"
    horizontal_path = tmp_path / "well__horizontal_well.csv"
    typewell_path = tmp_path / "well__typewell.csv"
    md = np.arange(7.0)
    visible_tvt = np.array([0.0, 0.5, 1.0, np.nan, np.nan, np.nan, np.nan])
    truth = np.arange(7.0) * 0.5
    horizontal_frame = pd.DataFrame(
        {
            "MD": md,
            "X": md,
            "Y": md * 2.0,
            "Z": -md,
            "GR": truth,
            "TVT_input": visible_tvt,
            # A target column is present, but prepare reads only the safe schema.
            "TVT": truth + 9999.0,
            "ANCC": truth - 500.0,
        }
    )
    horizontal_frame.to_csv(horizontal_path, index=False)
    typewell_tvt = np.arange(-5.0, 11.0, 0.5)
    typewell_frame = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": typewell_tvt,
            "Geology": ["forbidden-a"] * len(typewell_tvt),
        }
    )
    typewell_frame.to_csv(typewell_path, index=False)
    model = ProbabilisticLatticeModel(
        config=LatticeConfig(
            state_step_ft=0.5,
            state_radius_ft=3.0,
            max_move_bins=2,
            transition_rate_scale_floor=0.01,
            emission_scale_floor=0.01,
            emission_floor_probability=1e-5,
        )
    )
    inference = model.prepare(
        WellFiles("well", horizontal_path, typewell_path),
        split="train",
        cache_root=tmp_path / "unused-cache",
    )
    horizontal_frame["TVT"] = truth - 12345.0
    horizontal_frame["ANCC"] = truth + 777.0
    horizontal_frame.to_csv(horizontal_path, index=False)
    typewell_frame["Geology"] = ["forbidden-b"] * len(typewell_tvt)
    typewell_frame.to_csv(typewell_path, index=False)
    (tmp_path / "well.png").write_bytes(b"unrelated forbidden image content")
    changed_forbidden = model.prepare(
        WellFiles("well", horizontal_path, typewell_path),
        split="train",
        cache_root=tmp_path / "different-unused-cache",
    )
    assert inference.calibration == changed_forbidden.calibration
    assert inference.anchor == changed_forbidden.anchor
    assert inference.anchor_md == changed_forbidden.anchor_md
    np.testing.assert_array_equal(inference.row_index, changed_forbidden.row_index)
    np.testing.assert_allclose(inference.features, changed_forbidden.features)
    np.testing.assert_allclose(
        inference.typewell_tvt, changed_forbidden.typewell_tvt
    )
    np.testing.assert_allclose(
        inference.typewell_gr, changed_forbidden.typewell_gr
    )
    training_target = truth[inference.row_index]
    state = model.fit(
        [TrainingWell(inference, training_target)],
        context=RunContext(fold=0, seed=7, workers=1, work_dir=tmp_path),
    )
    diagnostics = model.predict_with_diagnostics(state, inference)
    prediction = model.predict(state, inference)
    np.testing.assert_allclose(prediction, diagnostics.posterior_mean)
    assert state.statistics.transition_sample_count == len(training_target)
    assert state.statistics.emission_sample_count == len(training_target)
    assert np.isfinite(prediction).all()
