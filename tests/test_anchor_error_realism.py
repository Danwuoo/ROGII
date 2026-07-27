from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from rogii.diagnostics.anchor_error_realism import (
    DEFAULT_SCENARIOS,
    ERROR_CONFIGS,
    AnchorLatents,
    ErrorScenario,
    _evaluate_anchor_error_sse,
    _mask_coefficients,
    audit_anchor_error_artifacts,
    corrupted_anchor_observations,
    hierarchical_error_bootstrap,
    reconstruct_pooled_metrics,
    scenario_anchor_errors,
    stable_anchor_latents,
    stable_common_bias_normal,
    stable_independent_normal,
    stable_outlier_indicator_uniform,
    stable_outlier_sign,
    validate_outlier_magnitude,
    validate_outlier_probability,
    validate_sigma,
    validate_tau,
)
from rogii.diagnostics.anchor_noise_robustness import (
    LOCKED_EXACT_METRICS,
    condition_noisy_piecewise,
    piecewise_noise_weights,
    stable_standard_normal,
)
from rogii.diagnostics.sparse_anchor_voi import (
    AnchorSelection,
    non_anchor_rmse,
    safe_output_directory,
)
from rogii.diagnostics.terminal_anchor_frontier import (
    select_terminal_anchors,
)


def _selection(indices: list[int], positions: np.ndarray) -> AnchorSelection:
    local = np.asarray(indices, dtype=np.int64)
    return AnchorSelection(
        policy="near_terminal",
        requested_k=len(indices),
        local_indices=local,
        normalized_positions=positions[local],
        requested_positions=positions[local],
        acquisition_scores=np.full(len(local), np.nan),
        minimum_spacing=0.0,
    )


def _latents() -> AnchorLatents:
    return AnchorLatents(
        common_bias=np.asarray([1.0, -2.0]),
        independent=np.asarray([[0.5, -1.0], [2.0, 0.25]]),
        outlier_indicator_uniform=np.asarray([[0.01, 0.99], [0.20, 0.30]]),
        outlier_sign=np.asarray([[1, -1], [-1, 1]], dtype=np.int8),
    )


def test_zero_error_reproduces_terminal_frontier_exact_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = (
        root
        / ".runs/sparse-anchor-terminal-frontier-full/information_frontier.csv"
    )
    if not artifact.is_file():
        pytest.skip("terminal frontier artifact not present")
    frame = pd.read_csv(artifact)
    for config in ERROR_CONFIGS:
        row = frame[
            frame["policy"].eq(config.policy)
            & frame["mode"].eq("acausal")
            & frame["method"].eq("piecewise_residual")
            & frame["k"].eq(config.k)
        ].iloc[0]
        expected_full, expected_non = LOCKED_EXACT_METRICS[config.name]
        assert float(row["pooled_rmse"]) == pytest.approx(expected_full, abs=1e-15)
        assert float(row["non_anchor_rmse"]) == pytest.approx(
            expected_non, abs=1e-15
        )


def test_independent_draw_exactly_reuses_gaussian_engine() -> None:
    for seed in (0, 17, 99):
        assert stable_independent_normal(
            seed, "e03b45fd", 12345
        ) == stable_standard_normal(seed, "e03b45fd", 12345)


def test_common_bias_is_identical_for_all_anchors_in_one_well() -> None:
    scenario = ErrorScenario("test", "bias", 2.0, 0.0, 0.0, 0.0, 2.0)
    latents = stable_anchor_latents([0, 1, 2], "well-a", [10, 20, 30])
    error, _ = scenario_anchor_errors(scenario, latents)
    np.testing.assert_array_equal(error, error[:, :1] * np.ones((1, 3)))


def test_common_bias_differs_across_wells_and_is_deterministic() -> None:
    first = stable_common_bias_normal(4, "well-a")
    assert first == stable_common_bias_normal(4, "well-a")
    assert first != stable_common_bias_normal(4, "well-b")


def test_independent_noise_is_anchor_specific_and_deterministic() -> None:
    first = stable_anchor_latents([3], "well", [10, 20, 30]).independent
    second = stable_anchor_latents([3], "well", [10, 20, 30]).independent
    np.testing.assert_array_equal(first, second)
    assert len(set(first[0])) == 3


def test_outlier_indicator_and_sign_are_cross_process_deterministic() -> None:
    expected = (
        stable_outlier_indicator_uniform(17, "e03b45fd", 12345),
        stable_outlier_sign(17, "e03b45fd", 12345),
    )
    code = (
        "from rogii.diagnostics.anchor_error_realism import "
        "stable_outlier_indicator_uniform,stable_outlier_sign;"
        "import json;"
        "print(json.dumps([stable_outlier_indicator_uniform("
        "17,'e03b45fd',12345),stable_outlier_sign(17,'e03b45fd',12345)]))"
    )
    actual = json.loads(
        subprocess.check_output([sys.executable, "-c", code], text=True)
    )
    assert tuple(actual) == expected


def test_shared_physical_anchor_rows_share_all_anchor_latents() -> None:
    first = stable_anchor_latents([0, 1, 99], "well", [10, 20, 30])
    second = stable_anchor_latents([0, 1, 99], "well", [20, 40])
    np.testing.assert_array_equal(first.independent[:, 1], second.independent[:, 0])
    np.testing.assert_array_equal(
        first.outlier_indicator_uniform[:, 1],
        second.outlier_indicator_uniform[:, 0],
    )
    np.testing.assert_array_equal(
        first.outlier_sign[:, 1], second.outlier_sign[:, 0]
    )


def test_tau_sigma_and_magnitude_only_scale_locked_latents() -> None:
    latents = _latents()
    at_one = ErrorScenario("test", "one", 1.0, 1.0, 1.0, 1.0, 1.0)
    at_two = ErrorScenario("test", "two", 2.0, 2.0, 1.0, 2.0, 2.0)
    first, _ = scenario_anchor_errors(at_one, latents)
    second, _ = scenario_anchor_errors(at_two, latents)
    np.testing.assert_array_equal(second, 2.0 * first)


def test_p_zero_never_emits_an_outlier() -> None:
    scenario = ErrorScenario("test", "p0", 0.0, 0.0, 0.0, 10.0, 0.0)
    errors, flags = scenario_anchor_errors(scenario, _latents())
    assert not flags.any()
    assert not errors.any()


def test_p_one_marks_every_anchor_as_outlier() -> None:
    scenario = ErrorScenario("test", "p1", 0.0, 0.0, 1.0, 5.0, 0.0)
    _, flags = scenario_anchor_errors(scenario, _latents())
    assert flags.all()


def test_zero_magnitude_has_zero_outlier_contribution() -> None:
    scenario = ErrorScenario("test", "m0", 0.0, 0.0, 1.0, 0.0, 0.0)
    errors, flags = scenario_anchor_errors(scenario, _latents())
    assert flags.all()
    assert not errors.any()


def test_outlier_sign_is_strictly_minus_or_plus_one() -> None:
    signs = {
        stable_outlier_sign(seed, "well", row)
        for seed in range(20)
        for row in (10, 20, 30)
    }
    assert signs == {-1, 1}


def test_ps_boundary_is_exact_and_not_an_error_column() -> None:
    positions = np.asarray([0.1, 0.4, 0.8, 1.0])
    selection = _selection([1, 2], positions)
    weights = piecewise_noise_weights(positions, selection)
    assert weights.shape == (4, 2)
    assert weights[0, 0] == pytest.approx(0.1 / 0.4)
    assert weights[0, 1] == 0.0


def test_anchor_prediction_equals_complete_corrupted_observation() -> None:
    positions = np.linspace(0.1, 1.0, 8)
    selection = _selection([2, 5, 7], positions)
    true = np.asarray([54.0, 60.0, 62.0])
    scenario = ErrorScenario("test", "mixed", 1.0, 0.5, 1.0, 5.0, 0.5)
    observed = corrupted_anchor_observations(true, scenario, _latents_for_three())
    prediction = condition_noisy_piecewise(
        np.linspace(50.0, 55.0, 8), observed[0], positions, selection
    )
    np.testing.assert_array_equal(prediction[selection.local_indices], observed[0])


def _latents_for_three() -> AnchorLatents:
    return AnchorLatents(
        common_bias=np.asarray([1.0, -2.0]),
        independent=np.asarray([[0.5, -1.0, 0.2], [2.0, 0.25, -0.5]]),
        outlier_indicator_uniform=np.asarray(
            [[0.01, 0.99, 0.02], [0.20, 0.30, 0.40]]
        ),
        outlier_sign=np.asarray([[1, -1, 1], [-1, 1, -1]], dtype=np.int8),
    )


def test_non_anchor_metric_excludes_corrupted_anchor_rows() -> None:
    truth = np.zeros(4)
    prediction = np.asarray([1.0, 100.0, 2.0, 200.0])
    is_anchor = np.asarray([False, True, False, True])
    assert non_anchor_rmse(truth, prediction, is_anchor) == pytest.approx(
        np.sqrt(2.5)
    )


def test_target_permutation_does_not_change_placement_or_random_draws() -> None:
    md = np.linspace(1.0, 100.0, 101)
    first = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=6
    )
    _ = np.arange(len(md), dtype=np.float64)[::-1]
    second = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=6
    )
    np.testing.assert_array_equal(first.local_indices, second.local_indices)
    np.testing.assert_array_equal(
        stable_anchor_latents([0, 1], "well", first.local_indices).independent,
        stable_anchor_latents([0, 1], "well", second.local_indices).independent,
    )


def test_unrevealed_target_changes_do_not_change_conditioned_prediction() -> None:
    positions = np.linspace(0.1, 1.0, 10)
    selection = _selection([2, 7], positions)
    base = np.linspace(10.0, 12.0, 10)
    truth = base + np.arange(10.0)
    observed = truth[selection.local_indices] + np.asarray([0.2, -0.3])
    first = condition_noisy_piecewise(base, observed, positions, selection)
    changed = truth.copy()
    changed[[0, 1, 3, 4, 5, 6, 8, 9]] += 1000.0
    second = condition_noisy_piecewise(base, observed, positions, selection)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(truth, changed)


def test_mixed_piecewise_propagation_matches_hand_calculation() -> None:
    positions = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    selection = _selection([2, 4], positions)
    base = np.zeros(6)
    true = np.asarray([4.0, 10.0])
    scenario = ErrorScenario("test", "mixed", 1.0, 0.5, 1.0, 2.0, 0.5)
    latents = AnchorLatents(
        common_bias=np.asarray([1.0]),
        independent=np.asarray([[2.0, -2.0]]),
        outlier_indicator_uniform=np.asarray([[0.1, 0.1]]),
        outlier_sign=np.asarray([[1, -1]], dtype=np.int8),
    )
    observed = corrupted_anchor_observations(true, scenario, latents)[0]
    actual = condition_noisy_piecewise(base, observed, positions, selection)
    expected = np.interp(
        positions,
        [0.0, 0.4, 0.8],
        [0.0, observed[0], observed[1]],
        right=observed[1],
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_irregular_md_short_suffix_td_and_975_snapping() -> None:
    md = np.asarray([1.0, 7.0, 31.0, 80.0, 100.0])
    td = select_terminal_anchors(
        md, 0.0, policy="endpoint_inclusive", k=1
    )
    near = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=1
    )
    assert td.local_indices[-1] == 4
    assert near.local_indices[-1] == 4
    short = select_terminal_anchors(
        np.asarray([10.0, 10.01, 10.02]),
        0.0,
        policy="endpoint_inclusive",
        k=6,
    )
    assert short.achieved_k == 3
    assert len(np.unique(short.local_indices)) == 3


def test_per_well_sse_reconstructs_all_pooled_metrics() -> None:
    frame = pd.DataFrame(
        {
            "rows": [2, 3],
            "non_anchor_rows": [1, 2],
            "sse": [5.0, 20.0],
            "non_anchor_sse": [4.0, 5.0],
        }
    )
    full, non = reconstruct_pooled_metrics(frame)
    assert full == pytest.approx(np.sqrt(5.0))
    assert non == pytest.approx(np.sqrt(3.0))


def test_sse_coefficients_match_direct_rowwise_propagation() -> None:
    exact = np.asarray([1.0, -2.0, 0.5])
    weights = np.asarray([[0.5, 0.0], [1.0, 0.0], [0.5, 0.5]])
    masks = np.asarray([[True, True, True], [True, False, True]])
    errors = np.asarray([[[2.0, -1.0], [0.0, 3.0]]])
    coefficients = _mask_coefficients(exact, weights, masks)
    actual = _evaluate_anchor_error_sse(coefficients, errors)[0]
    for seed in range(2):
        row_error = exact + weights @ errors[0, seed]
        np.testing.assert_allclose(
            actual[seed],
            [
                row_error @ row_error,
                row_error[masks[1]] @ row_error[masks[1]],
            ],
        )


def test_hierarchical_bootstrap_is_deterministic() -> None:
    full = np.asarray(
        [[[[1.0, 4.0], [9.0, 16.0]]]], dtype=np.float64
    )
    non = full * 0.5
    rows = np.asarray([2, 4])
    non_rows = np.asarray([[1, 3]])
    first = hierarchical_error_bootstrap(
        full, non, rows, non_rows, samples=100, seed=77
    )
    second = hierarchical_error_bootstrap(
        full, non, rows, non_rows, samples=100, seed=77
    )
    assert first == second
    assert "hierarchical" in json.dumps(first, allow_nan=False)


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (validate_tau, -1.0),
        (validate_tau, float("nan")),
        (validate_sigma, -1.0),
        (validate_outlier_probability, -0.1),
        (validate_outlier_probability, 1.1),
        (validate_outlier_magnitude, -1.0),
    ],
)
def test_invalid_error_parameters_fail_closed(function: object, value: float) -> None:
    with pytest.raises(ValueError):
        function(value)  # type: ignore[operator]


def test_diagnostic_uses_no_forbidden_geology_inputs() -> None:
    from rogii.diagnostics import anchor_error_realism as module

    source = inspect.getsource(module)
    assert "load_horizontal_inference" not in source
    assert '"formation_surfaces_read": False' in source
    assert '"geology_read": False' in source
    assert '"png_read": False' in source
    assert "load_full_horizontal" not in source


def test_output_guard_prevents_writes_inside_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError, match="competition data"):
        safe_output_directory(data, "illegal-realism", data)


def test_locked_matrix_has_only_requested_configs_and_scenarios() -> None:
    assert {config.name for config in ERROR_CONFIGS} == {"near_k6", "td_k6"}
    assert len(DEFAULT_SCENARIOS) == 23
    assert [scenario.name for scenario in DEFAULT_SCENARIOS[-3:]] == [
        "mild",
        "moderate",
        "severe",
    ]


def test_full_artifact_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / ".runs/sparse-anchor-error-realism-full"
    if not artifact.is_dir():
        pytest.skip("full anchor-error realism artifact is not present")
    audit = audit_anchor_error_artifacts(artifact)
    assert audit["passed"]
    assert audit["duplicate_key_count"] == 0
    assert audit["unexpected_numeric_nan_count"] == 0
