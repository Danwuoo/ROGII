from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from rogii.diagnostics.anchor_acquisition_redundancy import (
    DEPENDENCE_ANCHOR_SHARED,
    DEPENDENCE_CLEAN,
    DEPENDENCE_MEASUREMENT_INDEPENDENT,
    DEPENDENCE_SINGLE,
    LOCKED_DESIGNS,
    METHODS,
    O5_SCENARIO,
    REDUNDANCY_SCENARIOS,
    AcquisitionDesign,
    ReplicateLatents,
    _aggregate_variable_clean_replicates,
    _dependences_for,
    _retained_mask_values,
    aggregate_replicates,
    audit_anchor_acquisition_redundancy_artifacts,
    build_evaluation_groups,
    hierarchical_redundancy_bootstrap,
    reconstruct_pooled_metrics,
    scenario_replicate_errors,
    select_designs,
    select_redundancy_scenarios,
    stable_replicate_independent_normal,
    stable_replicate_latents,
    stable_replicate_outlier_indicator_uniform,
    stable_replicate_outlier_sign,
    validate_acquisition_design,
    validate_dependence,
)
from rogii.diagnostics.anchor_error_realism import (
    ErrorScenario,
    stable_anchor_latents,
    stable_common_bias_normal,
)
from rogii.diagnostics.anchor_noise_robustness import (
    condition_noisy_piecewise,
    piecewise_noise_weights,
)
from rogii.diagnostics.sparse_anchor_voi import AnchorSelection, anchor_mask
from rogii.diagnostics.terminal_anchor_frontier import (
    requested_fractions,
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


def _manual_latents(replicates: int = 3) -> ReplicateLatents:
    independent = np.asarray(
        [
            [[1.0, -1.0, 0.0], [0.5, 1.5, -0.5]],
            [[-2.0, 2.0, 1.0], [0.25, -0.25, 0.75]],
        ],
        dtype=np.float64,
    )[:, :, :replicates]
    indicator = np.asarray(
        [
            [[0.01, 0.99, 0.99], [0.01, 0.02, 0.99]],
            [[0.99, 0.01, 0.02], [0.99, 0.99, 0.99]],
        ],
        dtype=np.float64,
    )[:, :, :replicates]
    signs = np.asarray(
        [
            [[1, -1, 1], [1, -1, 1]],
            [[-1, 1, -1], [-1, 1, -1]],
        ],
        dtype=np.int8,
    )[:, :, :replicates]
    return ReplicateLatents(
        common_bias=np.asarray([2.0, -3.0]),
        independent=independent,
        outlier_indicator_uniform=indicator,
        outlier_sign=signs,
    )


def test_locked_designs_scenarios_and_methods_are_exact() -> None:
    assert [design.name for design in LOCKED_DESIGNS] == [
        "near_k6_r1",
        "near_k7_r1",
        "near_k8_r1",
        "near_k9_r1",
        "near_k10_r1",
        "near_k11_r1",
        "near_k12_r1",
        "near_k6_r2_mean",
        "near_k6_r3_median",
        "near_k8_r2_mean",
    ]
    assert [scenario.name for scenario in REDUNDANCY_SCENARIOS] == [
        "exact_clean",
        "gaussian_1_clean",
        "moderate_background",
        "moderate_m5_control",
        "moderate_p05_m10",
        O5_SCENARIO,
        "moderate_p20_m10",
        "S0_severe_background",
        "S1_severe_p10_m10",
    ]
    assert METHODS == (
        "observed_aggregate",
        "oracle_drop_outlier_replicates",
        "oracle_clean_values",
    )


def test_spatial_fractions_are_locked_for_k6_through_k12() -> None:
    for design in LOCKED_DESIGNS:
        expected = np.concatenate(
            (
                np.arange(1, design.k, dtype=np.float64) / design.k,
                np.asarray([0.975]),
            )
        )
        np.testing.assert_array_equal(np.asarray(design.fractions), expected)


def test_replicate_zero_is_exactly_legacy_compatible() -> None:
    rows = np.asarray([10, 20, 30])
    current = stable_replicate_latents([0, 17, 99], "well", rows, 3)
    legacy = stable_anchor_latents([0, 17, 99], "well", rows)
    np.testing.assert_array_equal(current.common_bias, legacy.common_bias)
    np.testing.assert_array_equal(current.independent[:, :, 0], legacy.independent)
    np.testing.assert_array_equal(
        current.outlier_indicator_uniform[:, :, 0],
        legacy.outlier_indicator_uniform,
    )
    np.testing.assert_array_equal(
        current.outlier_sign[:, :, 0], legacy.outlier_sign
    )


def test_replicate_keys_are_cross_process_deterministic() -> None:
    expected = [
        stable_replicate_independent_normal(17, "well", 123, 2),
        stable_replicate_outlier_indicator_uniform(17, "well", 123, 2),
        stable_replicate_outlier_sign(17, "well", 123, 2),
    ]
    code = (
        "import json;"
        "from rogii.diagnostics.anchor_acquisition_redundancy import "
        "stable_replicate_independent_normal,"
        "stable_replicate_outlier_indicator_uniform,"
        "stable_replicate_outlier_sign;"
        "print(json.dumps([stable_replicate_independent_normal(17,'well',123,2),"
        "stable_replicate_outlier_indicator_uniform(17,'well',123,2),"
        "stable_replicate_outlier_sign(17,'well',123,2)]))"
    )
    assert json.loads(
        subprocess.check_output([sys.executable, "-c", code], text=True)
    ) == expected


def test_same_physical_row_and_replicate_share_latents_across_configs() -> None:
    first = stable_replicate_latents([0, 1], "well", [10, 20, 30], 3)
    second = stable_replicate_latents([0, 1], "well", [20, 40], 3)
    np.testing.assert_array_equal(first.independent[:, 1], second.independent[:, 0])
    np.testing.assert_array_equal(
        first.outlier_indicator_uniform[:, 1],
        second.outlier_indicator_uniform[:, 0],
    )
    np.testing.assert_array_equal(
        first.outlier_sign[:, 1], second.outlier_sign[:, 0]
    )


def test_common_bias_is_shared_within_well_and_differs_across_wells() -> None:
    first = stable_replicate_latents([0, 1], "well-a", [10, 20], 3)
    second = stable_replicate_latents([0, 1], "well-a", [30], 2)
    np.testing.assert_array_equal(first.common_bias, second.common_bias)
    assert stable_common_bias_normal(0, "well-a") != stable_common_bias_normal(
        0, "well-b"
    )


def test_gaussian_noise_is_independent_between_replicates() -> None:
    latents = stable_replicate_latents([0, 1, 2], "well", [10, 20], 3)
    assert not np.array_equal(
        latents.independent[:, :, 0], latents.independent[:, :, 1]
    )
    np.testing.assert_array_equal(
        latents.independent,
        stable_replicate_latents([0, 1, 2], "well", [10, 20], 3).independent,
    )


def test_measurement_independent_and_anchor_shared_outlier_contracts() -> None:
    scenario = ErrorScenario("test", "p50", 0.0, 0.0, 0.5, 10.0, 0.0)
    latents = _manual_latents(3)
    independent = scenario_replicate_errors(
        scenario, latents, DEPENDENCE_MEASUREMENT_INDEPENDENT, "median"
    )
    shared = scenario_replicate_errors(
        scenario, latents, DEPENDENCE_ANCHOR_SHARED, "median"
    )
    assert np.any(
        independent["flags"] != independent["flags"][:, :, :1]
    )
    np.testing.assert_array_equal(
        shared["flags"],
        np.broadcast_to(shared["flags"][:, :, :1], shared["flags"].shape),
    )
    np.testing.assert_array_equal(
        shared["signs"],
        np.broadcast_to(shared["signs"][:, :, :1], shared["signs"].shape),
    )


@pytest.mark.parametrize(
    ("probability", "magnitude", "expected_any", "expected_all"),
    [(0.0, 10.0, False, False), (1.0, 10.0, True, True), (1.0, 0.0, False, True)],
)
def test_probability_and_zero_magnitude_edges(
    probability: float,
    magnitude: float,
    expected_any: bool,
    expected_all: bool,
) -> None:
    scenario = ErrorScenario(
        "test", "edge", 0.0, 0.0, probability, magnitude, 0.0
    )
    dependence = (
        DEPENDENCE_CLEAN
        if probability == 0.0
        else DEPENDENCE_MEASUREMENT_INDEPENDENT
    )
    outcome = scenario_replicate_errors(
        scenario, _manual_latents(3), dependence, "median"
    )
    assert bool(np.any(outcome["effective_outlier_shift"])) == expected_any
    assert bool(np.all(outcome["flags"])) == expected_all


def test_signs_are_strictly_plus_or_minus_one() -> None:
    signs = {
        stable_replicate_outlier_sign(seed, "well", row, replicate)
        for seed in range(20)
        for row in (10, 20)
        for replicate in (0, 1, 2)
    }
    assert signs == {-1, 1}


def test_different_scales_only_scale_the_same_latents() -> None:
    first = ErrorScenario("test", "one", 1.0, 1.0, 1.0, 1.0, 1.0)
    second = ErrorScenario("test", "two", 2.0, 2.0, 1.0, 2.0, 2.0)
    one = scenario_replicate_errors(
        first, _manual_latents(2), DEPENDENCE_ANCHOR_SHARED, "mean"
    )
    two = scenario_replicate_errors(
        second, _manual_latents(2), DEPENDENCE_ANCHOR_SHARED, "mean"
    )
    np.testing.assert_array_equal(
        two["aggregate_error"], 2.0 * one["aggregate_error"]
    )


def test_r1_identity_and_r2_mean_are_hand_checkable() -> None:
    values = np.asarray([[[2.0]], [[-3.0]]])
    np.testing.assert_array_equal(
        aggregate_replicates(values, "identity"), np.asarray([[2.0], [-3.0]])
    )
    mean_values = np.asarray([[[2.0, 6.0], [-3.0, 1.0]]])
    np.testing.assert_array_equal(
        aggregate_replicates(mean_values, "mean"),
        np.asarray([[4.0, -1.0]]),
    )


def test_r3_median_handles_one_and_two_gross_errors() -> None:
    values = np.asarray(
        [[[1.0, 1.5, 11.0], [1.0, 11.0, 12.0]]], dtype=np.float64
    )
    aggregate = aggregate_replicates(values, "median")
    np.testing.assert_array_equal(aggregate, np.asarray([[1.5, 11.0]]))


def test_common_bias_survives_mean_and_median() -> None:
    scenario = ErrorScenario("test", "bias", 1.0, 0.0, 0.0, 0.0, 0.0)
    for replicates, aggregation in ((2, "mean"), (3, "median")):
        outcome = scenario_replicate_errors(
            scenario, _manual_latents(replicates), DEPENDENCE_CLEAN, aggregation
        )
        expected = np.broadcast_to(
            _manual_latents(replicates).common_bias[:, None],
            outcome["aggregate_error"].shape,
        )
        np.testing.assert_array_equal(outcome["aggregate_error"], expected)


def test_physical_anchor_prediction_equals_aggregate_observation() -> None:
    positions = np.linspace(0.1, 1.0, 9)
    selection = _selection([1, 3, 5, 8], positions)
    base = np.linspace(100.0, 105.0, 9)
    truth = np.asarray([101.0, 102.0, 106.0, 108.0])
    replicate_observations = np.stack((truth + 1.0, truth + 3.0), axis=1)
    aggregate = aggregate_replicates(replicate_observations, "mean")
    prediction = condition_noisy_piecewise(
        base, aggregate, positions, selection
    )
    np.testing.assert_array_equal(
        prediction[selection.local_indices], aggregate
    )


def test_non_anchor_excludes_each_physical_row_once() -> None:
    positions = np.linspace(0.1, 1.0, 20)
    selection = _selection([2, 5, 8, 11, 14, 18], positions)
    mask = anchor_mask(len(positions), selection)
    assert int(mask.sum()) == 6
    assert int((~mask).sum()) == 14


def test_ps_boundary_is_not_a_replicate_or_anchor_weight_column() -> None:
    positions = np.asarray([0.1, 0.3, 0.6, 1.0])
    selection = _selection([1, 2], positions)
    weights = piecewise_noise_weights(positions, selection)
    assert weights.shape == (4, 2)
    assert weights[0, 0] == pytest.approx(0.1 / 0.3)
    assert weights[0, 1] == 0.0


def test_target_permutation_does_not_change_placement_or_draws() -> None:
    md = np.asarray([1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0])
    first = select_terminal_anchors(md, 0.0, policy="near_terminal", k=6)
    _ = np.arange(len(md), dtype=np.float64)[::-1]
    second = select_terminal_anchors(md, 0.0, policy="near_terminal", k=6)
    np.testing.assert_array_equal(first.local_indices, second.local_indices)
    np.testing.assert_array_equal(
        stable_replicate_latents(
            [0, 1], "well", first.local_indices, 2
        ).independent,
        stable_replicate_latents(
            [0, 1], "well", second.local_indices, 2
        ).independent,
    )


def test_unrevealed_target_does_not_enter_conditioner() -> None:
    positions = np.linspace(0.1, 1.0, 9)
    selection = _selection([1, 3, 5, 8], positions)
    base = np.linspace(100.0, 105.0, 9)
    observed = np.asarray([101.0, 102.0, 106.0, 108.0])
    first = condition_noisy_piecewise(base, observed, positions, selection)
    hidden_target = base + 999.0
    hidden_target[selection.local_indices] = observed
    second = condition_noisy_piecewise(base, observed, positions, selection)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(hidden_target, base)


def test_irregular_md_snapping_and_k6_through_k12_counts() -> None:
    rng = np.random.default_rng(7)
    md = np.cumsum(rng.uniform(0.1, 3.0, size=100))
    for k in range(6, 13):
        selection = select_terminal_anchors(
            md, md[0] - 1.0, policy="near_terminal", k=k
        )
        assert selection.achieved_k == k
        assert len(np.unique(selection.local_indices)) == k
        assert selection.requested_positions[-1] == 0.975


def test_short_suffix_fails_closed_to_achieved_unique_rows() -> None:
    md = np.asarray([1.0, 2.0, 3.0, 4.0])
    selection = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=6
    )
    assert selection.achieved_k == 4
    assert len(np.unique(selection.local_indices)) == 4


def test_oracle_drop_preserves_clean_reading_bias_and_noise() -> None:
    values = np.asarray([[[11.0, 12.0, 99.0]]])
    clean = np.asarray([[[True, True, False]]])
    aggregate, retained = _aggregate_variable_clean_replicates(
        values, clean, "median"
    )
    assert retained[0, 0]
    assert aggregate[0, 0] == pytest.approx(11.5)


def test_oracle_drop_removes_location_when_all_replicates_outlier() -> None:
    values = np.asarray([[[11.0, 12.0], [20.0, 21.0]]])
    clean = np.asarray([[[False, False], [True, False]]])
    aggregate, retained = _aggregate_variable_clean_replicates(
        values, clean, "mean"
    )
    np.testing.assert_array_equal(retained, np.asarray([[False, True]]))
    assert aggregate[0, 0] == 0.0
    assert aggregate[0, 1] == 20.0
    np.testing.assert_array_equal(
        _retained_mask_values(retained), np.asarray([2], dtype=np.uint64)
    )


def test_effective_outlier_shift_matches_hand_calculation() -> None:
    scenario = ErrorScenario("test", "m10", 0.0, 0.0, 0.5, 10.0, 0.0)
    outcome = scenario_replicate_errors(
        scenario,
        _manual_latents(2),
        DEPENDENCE_MEASUREMENT_INDEPENDENT,
        "mean",
    )
    expected = aggregate_replicates(
        outcome["contributions"], "mean"
    )
    np.testing.assert_array_equal(
        outcome["effective_outlier_shift"], expected
    )


def test_per_well_sse_reconstructs_pooled_metrics() -> None:
    frame = pd.DataFrame(
        {
            "rows": [2, 3],
            "non_anchor_rows": [1, 2],
            "sse": [8.0, 12.0],
            "non_anchor_sse": [4.0, 5.0],
        }
    )
    full, non = reconstruct_pooled_metrics(frame)
    assert full == pytest.approx(2.0)
    assert non == pytest.approx(np.sqrt(3.0))


def test_hierarchical_bootstrap_is_deterministic_and_paired() -> None:
    full = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4) + 1.0
    non = full + 0.5
    rows = np.asarray([10, 11, 12, 13])
    non_rows = np.asarray([[8, 9, 10, 11], [7, 8, 9, 10]])
    design_index = np.asarray([0, 1])
    first = hierarchical_redundancy_bootstrap(
        full,
        non,
        rows,
        non_rows,
        design_index,
        samples=50,
        seed=7,
    )
    second = hierarchical_redundancy_bootstrap(
        full,
        non,
        rows,
        non_rows,
        design_index,
        samples=50,
        seed=7,
    )
    assert first == second


def test_group_building_is_order_stable_and_r1_is_single_equivalent() -> None:
    designs = select_designs(["near_k6_r1", "near_k6_r2_mean"])
    scenarios = select_redundancy_scenarios(
        ["exact_clean", O5_SCENARIO]
    )
    first = build_evaluation_groups(designs, scenarios)
    second = build_evaluation_groups(designs, scenarios)
    assert first == second
    r1_o5 = [
        group
        for group in first
        if group.design_index == 0 and group.scenario_index == 1
    ]
    assert {group.dependence for group in r1_o5} == {DEPENDENCE_SINGLE}
    assert len(r1_o5) == 3


@pytest.mark.parametrize(
    "call",
    [
        lambda: validate_acquisition_design(0, 1, "identity", "single_read_spatial"),
        lambda: validate_acquisition_design(6, 4, "median", "same_location_repeat"),
        lambda: validate_acquisition_design(6, 2, "median", "same_location_repeat"),
        lambda: validate_acquisition_design(7, 2, "mean", "same_location_repeat"),
        lambda: validate_dependence("unknown"),
        lambda: select_designs(["near_k13_r1"]),
        lambda: select_redundancy_scenarios(["new_scenario"]),
        lambda: aggregate_replicates(np.ones((2, 2)), "identity"),
    ],
)
def test_invalid_configuration_fails_closed(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_no_forbidden_data_sources_are_referenced() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/rogii/diagnostics/anchor_acquisition_redundancy.py"
    ).read_text(encoding="utf-8").lower()
    assert "load_formation" not in source
    assert "read_geology" not in source
    assert "read_png" not in source
    assert "load_typewell" not in source


def test_existing_exact_and_regression_artifacts_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    frontier_path = (
        root
        / ".runs/sparse-anchor-terminal-frontier-full/information_frontier.csv"
    )
    qc_path = root / ".runs/sparse-anchor-outlier-qc-full/per_seed_metrics.csv"
    if not frontier_path.is_file() or not qc_path.is_file():
        pytest.skip("prior sparse-anchor artifacts not present")
    frontier = pd.read_csv(frontier_path)
    k7 = frontier[
        frontier["policy"].eq("near_terminal")
        & frontier["mode"].eq("acausal")
        & frontier["method"].eq("piecewise_residual")
        & frontier["k"].eq(7)
    ]
    assert float(k7.iloc[0]["pooled_rmse"]) == pytest.approx(
        3.015171, abs=1e-6
    )
    qc = pd.read_csv(qc_path)
    assert len(
        qc[
            qc["config"].eq("near_k6")
            & qc["method"].eq("no_qc")
            & qc["scenario"].isin(
                [
                    "exact_clean",
                    "gaussian_1_clean",
                    "moderate_m5_control",
                    "moderate_background_p10_m10",
                    "severe_background_no_outlier",
                    "severe",
                ]
            )
        ]
    ) == 600


def test_full_redundancy_artifact_audit_when_available() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / ".runs/sparse-anchor-acquisition-redundancy-full"
    if not (artifact / "summary.json").is_file():
        pytest.skip("full redundancy artifact not present")
    audit = audit_anchor_acquisition_redundancy_artifacts(artifact)
    assert audit["passed"]
