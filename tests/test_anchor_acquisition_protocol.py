from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from rogii.diagnostics.anchor_acquisition_protocol import (
    EVALUATION_GROUPS,
    EXPECTED_ZERO_METRICS,
    NOMINAL_FRACTIONS,
    O5_SCENARIO,
    PLACEMENT_SPECS,
    POSITION_MODES,
    POSITION_TOLERANCES_U,
    PRIMARY_PROTOCOL_TOLERANCE_U,
    PROTOCOL_SCENARIOS,
    ZERO_POSITION_MODE,
    analytic_piecewise_coefficients,
    audit_anchor_acquisition_protocol_artifacts,
    build_piecewise_prefix_sums,
    direct_piecewise_coefficients,
    nearest_unused_indices_equivalent,
    requested_protocol_positions,
    select_protocol_anchors,
    stable_position_sign,
    stable_position_sign_matrix,
)
from rogii.diagnostics.anchor_acquisition_redundancy import (
    DEPENDENCE_SINGLE,
    stable_replicate_latents,
)
from rogii.diagnostics.anchor_noise_robustness import (
    NoiseWellContext,
    condition_noisy_piecewise,
)
from rogii.diagnostics.sparse_anchor_voi import (
    _nearest_unused_indices,
    anchor_mask,
    safe_output_directory,
)
from rogii.diagnostics.terminal_anchor_frontier import (
    select_terminal_anchors,
)


def _context() -> NoiseWellContext:
    positions = np.asarray(
        [0.04, 0.12, 0.19, 0.26, 0.37, 0.51, 0.63, 0.74, 0.88, 0.97, 1.0]
    )
    base = np.linspace(100.0, 108.0, len(positions))
    truth = base + np.sin(positions * 4.0) * 3.0
    return NoiseWellContext(
        well_id="well",
        fold=0,
        row_index=np.arange(100, 100 + len(positions), dtype=np.int64),
        normalized_md=positions,
        prediction_md=positions * 1_000.0,
        y_true=truth,
        base_prediction=base,
    )


def test_nominal_fractions_and_tolerances_are_completely_locked() -> None:
    assert NOMINAL_FRACTIONS == (
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        0.975,
    )
    assert POSITION_TOLERANCES_U == (0.0, 0.001, 0.0025, 0.005)
    assert PRIMARY_PROTOCOL_TOLERANCE_U == 0.001
    assert POSITION_MODES == (
        "coherent_early",
        "coherent_late",
        "independent_bounded",
    )
    assert [scenario.name for scenario in PROTOCOL_SCENARIOS] == [
        "exact_clean",
        "moderate_m5_control",
        O5_SCENARIO,
    ]
    assert len(PLACEMENT_SPECS) == 10
    assert len(EVALUATION_GROUPS) == 30


def test_coherent_early_and_late_apply_exact_bound_edge() -> None:
    nominal = np.asarray(NOMINAL_FRACTIONS)
    early = requested_protocol_positions(0.001, "coherent_early")
    late = requested_protocol_positions(0.001, "coherent_late")
    np.testing.assert_array_equal(early, nominal - 0.001)
    np.testing.assert_array_equal(late, nominal + 0.001)


def test_independent_bounded_signs_are_deterministic_and_cross_process() -> None:
    expected = [
        stable_position_sign(17, "well", order)
        for order in range(1, 9)
    ]
    code = (
        "import json;"
        "from rogii.diagnostics.anchor_acquisition_protocol import "
        "stable_position_sign;"
        "print(json.dumps([stable_position_sign(17,'well',i) "
        "for i in range(1,9)]))"
    )
    assert json.loads(
        subprocess.check_output([sys.executable, "-c", code], text=True)
    ) == expected
    assert set(expected).issubset({-1, 1})


def test_config_order_and_tolerance_do_not_change_position_signs() -> None:
    first = stable_position_sign_matrix([0, 3, 9], "well")
    reordered = stable_position_sign_matrix([9, 0, 3], "well")
    np.testing.assert_array_equal(first[0], reordered[1])
    np.testing.assert_array_equal(first[1], reordered[2])
    np.testing.assert_array_equal(first[2], reordered[0])
    for tolerance in POSITION_TOLERANCES_U[1:]:
        requested = requested_protocol_positions(
            tolerance, "independent_bounded", signs=first[0]
        )
        np.testing.assert_array_equal(
            np.sign(requested - np.asarray(NOMINAL_FRACTIONS)),
            first[0],
        )


def test_position_selection_is_target_independent_and_permutation_invariant() -> None:
    assert "target" not in inspect.signature(select_protocol_anchors).parameters
    positions = np.linspace(0.01, 1.0, 250)
    requested = requested_protocol_positions(
        0.001,
        "independent_bounded",
        signs=stable_position_sign_matrix([4], "well")[0],
    )
    first = select_protocol_anchors(positions, requested)
    _ = np.arange(len(positions), dtype=np.float64)[::-1]
    second = select_protocol_anchors(positions, requested)
    assert first.valid and second.valid
    assert first.selection is not None and second.selection is not None
    np.testing.assert_array_equal(
        first.selection.local_indices, second.selection.local_indices
    )


def test_achieved_rows_are_distinct_and_strictly_ordered() -> None:
    positions = np.linspace(0.002, 1.0, 1_000)
    for tolerance in POSITION_TOLERANCES_U:
        modes = (ZERO_POSITION_MODE,) if tolerance == 0 else POSITION_MODES
        for mode in modes:
            signs = (
                stable_position_sign_matrix([2], "well")[0]
                if mode == "independent_bounded"
                else None
            )
            requested = requested_protocol_positions(
                tolerance, mode, signs=signs
            )
            outcome = select_protocol_anchors(positions, requested)
            assert outcome.valid
            assert outcome.selection is not None
            assert len(np.unique(outcome.selection.local_indices)) == 8
            assert np.all(np.diff(outcome.selection.local_indices) > 0)


def test_fast_nearest_unused_is_exactly_legacy_equivalent() -> None:
    rng = np.random.default_rng(20260727)
    for row_count in (8, 9, 25, 250):
        for _ in range(100):
            positions = np.cumsum(rng.uniform(0.001, 1.0, row_count))
            positions /= positions[-1]
            requested = np.sort(rng.uniform(0.001, 1.0, 8))
            np.testing.assert_array_equal(
                nearest_unused_indices_equivalent(positions, requested),
                _nearest_unused_indices(positions, requested),
            )
    collision_positions = np.asarray(
        [0.1, 0.2, 0.9, 0.91, 0.92, 0.93, 0.94, 1.0]
    )
    collision_requested = np.asarray(
        [0.19, 0.21, 0.90, 0.91, 0.92, 0.93, 0.94, 0.99]
    )
    np.testing.assert_array_equal(
        nearest_unused_indices_equivalent(
            collision_positions, collision_requested
        ),
        _nearest_unused_indices(
            collision_positions, collision_requested
        ),
    )


def test_no_clipping_and_boundary_violation_fails_closed() -> None:
    requested = np.asarray(NOMINAL_FRACTIONS) + 0.1
    requested[-1] = 1.01
    outcome = select_protocol_anchors(np.linspace(0.01, 1.0, 100), requested)
    assert not outcome.valid
    assert outcome.selection is None
    assert outcome.boundary_violations == 1
    assert outcome.requested_positions[-1] == 1.01


def test_availability_and_order_violations_fail_closed() -> None:
    short = select_protocol_anchors(
        np.linspace(0.1, 1.0, 7), NOMINAL_FRACTIONS
    )
    assert not short.valid
    assert short.selection is None
    assert short.availability_violations == 1
    sparse_positions = np.asarray(
        [0.1, 0.2, 0.9, 0.91, 0.92, 0.93, 0.94, 1.0]
    )
    requested = np.asarray([0.19, 0.21, 0.90, 0.91, 0.92, 0.93, 0.94, 0.99])
    out_of_order = select_protocol_anchors(sparse_positions, requested)
    assert not out_of_order.valid
    assert out_of_order.selection is None
    assert out_of_order.order_violations >= 1


def test_actual_achieved_positions_drive_interpolation() -> None:
    context = _context()
    requested = requested_protocol_positions(0.005, "coherent_late")
    outcome = select_protocol_anchors(context.normalized_md, requested)
    assert outcome.valid and outcome.selection is not None
    selection = outcome.selection
    observed = context.y_true[selection.local_indices] + np.linspace(0.1, 0.8, 8)
    prediction = condition_noisy_piecewise(
        context.base_prediction,
        observed,
        context.normalized_md,
        selection,
    )
    np.testing.assert_array_equal(
        prediction[selection.local_indices], observed
    )
    nominal = select_terminal_anchors(
        context.prediction_md,
        0.0,
        policy="near_terminal",
        k=8,
    )
    if not np.array_equal(nominal.local_indices, selection.local_indices):
        nominal_prediction = condition_noisy_piecewise(
            context.base_prediction,
            observed,
            context.normalized_md,
            nominal,
        )
        assert not np.array_equal(prediction, nominal_prediction)


def test_only_selected_achieved_targets_enter_conditioner() -> None:
    context = _context()
    outcome = select_protocol_anchors(
        context.normalized_md,
        requested_protocol_positions(0.005, "coherent_early"),
    )
    assert outcome.selection is not None
    selection = outcome.selection
    observed = context.y_true[selection.local_indices].copy()
    first = condition_noisy_piecewise(
        context.base_prediction,
        observed,
        context.normalized_md,
        selection,
    )
    changed = context.y_true.copy()
    unselected = ~anchor_mask(len(changed), selection)
    changed[unselected] += 10_000.0
    second = condition_noisy_piecewise(
        context.base_prediction,
        observed,
        context.normalized_md,
        selection,
    )
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(changed, context.y_true)


def test_non_anchor_mask_excludes_each_physical_row_once() -> None:
    positions = np.linspace(0.01, 1.0, 200)
    outcome = select_protocol_anchors(positions, NOMINAL_FRACTIONS)
    assert outcome.selection is not None
    mask = anchor_mask(len(positions), outcome.selection)
    assert int(mask.sum()) == 8
    assert int((~mask).sum()) == len(positions) - 8


def test_measurement_latents_are_paired_to_nominal_rows_across_modes() -> None:
    nominal_rows = np.asarray([10, 20, 30, 40, 50, 60, 70, 80])
    first = stable_replicate_latents([0, 1, 2], "well", nominal_rows, 1)
    _ = requested_protocol_positions(0.001, "coherent_early")
    second = stable_replicate_latents([0, 1, 2], "well", nominal_rows, 1)
    np.testing.assert_array_equal(first.common_bias, second.common_bias)
    np.testing.assert_array_equal(first.independent, second.independent)
    np.testing.assert_array_equal(
        first.outlier_indicator_uniform,
        second.outlier_indicator_uniform,
    )
    np.testing.assert_array_equal(first.outlier_sign, second.outlier_sign)


def test_analytic_coefficients_match_direct_rowwise_reconstruction() -> None:
    positions = np.linspace(0.001, 1.0, 2_000)
    base = 100.0 + 5.0 * positions + np.sin(positions * 9.0)
    truth = base + np.cos(positions * 6.0) * 2.0
    context = NoiseWellContext(
        well_id="well",
        fold=0,
        row_index=np.arange(len(positions), dtype=np.int64),
        normalized_md=positions,
        prediction_md=positions * 8_000.0,
        y_true=truth,
        base_prediction=base,
    )
    outcome = select_protocol_anchors(
        positions,
        requested_protocol_positions(
            0.005,
            "independent_bounded",
            signs=stable_position_sign_matrix([9], "well")[0],
        ),
    )
    assert outcome.selection is not None
    prefix = build_piecewise_prefix_sums(positions, base, truth)
    analytic = analytic_piecewise_coefficients(prefix, outcome.selection)
    direct, _, _ = direct_piecewise_coefficients(context, outcome.selection)
    np.testing.assert_allclose(analytic[0], direct[0], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(analytic[1], direct[1], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(analytic[2], direct[2], rtol=0.0, atol=1e-9)
    np.testing.assert_array_equal(analytic[3], direct[3])


def test_pooled_metrics_are_reconstructible_from_per_well_values() -> None:
    frame = pd.DataFrame(
        {
            "rows": [2, 3],
            "non_anchor_rows": [1, 2],
            "sse": [8.0, 12.0],
            "non_anchor_sse": [4.0, 5.0],
        }
    )
    full = float(np.sqrt(frame["sse"].sum() / frame["rows"].sum()))
    non = float(
        np.sqrt(
            frame["non_anchor_sse"].sum()
            / frame["non_anchor_rows"].sum()
        )
    )
    assert full == pytest.approx(2.0)
    assert non == pytest.approx(np.sqrt(3.0))


def test_zero_tolerance_legacy_metrics_when_full_artifact_is_present() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = (
        root / ".runs/sparse-anchor-acquisition-redundancy-full"
    )
    if not (artifact / "acquisition_frontier.csv").is_file():
        pytest.skip("legacy acquisition-redundancy artifact not present")
    frame = pd.read_csv(artifact / "acquisition_frontier.csv")
    row = frame[
        frame["config"].eq("near_k8_r1")
        & frame["method"].eq("observed_aggregate")
        & frame["outlier_dependence"].eq(DEPENDENCE_SINGLE)
        & frame["scenario"].eq(O5_SCENARIO)
    ].iloc[0]
    for column, expected in EXPECTED_ZERO_METRICS.items():
        assert float(row[column]) == pytest.approx(expected, abs=1e-15)


def test_no_forbidden_data_sources_are_referenced() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/rogii/diagnostics/anchor_acquisition_protocol.py"
    ).read_text(encoding="utf-8").lower()
    assert "load_formation" not in source
    assert "read_geology" not in source
    assert "read_png" not in source
    assert "load_typewell" not in source
    assert '"formation_surfaces_read": false' in source
    assert '"geology_read": false' in source
    assert '"png_read": false' in source


def test_output_guard_prevents_writes_inside_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError, match="competition data"):
        safe_output_directory(data, "illegal-protocol", data)


def test_evidence_flags_are_locked_false_or_true_as_required() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/rogii/diagnostics/anchor_acquisition_protocol.py"
    ).read_text(encoding="utf-8")
    assert '"deployable": False' in source
    assert '"submission_eligible": False' in source
    assert '"counterfactual_information_experiment": True' in source
    assert '"uses_validation_target_as_revealed_anchor": True' in source
    assert '"kaggle_submission_created": False' in source
    assert '"kaggle_submission_uploaded": False' in source


def test_full_protocol_artifact_audit_when_available() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / ".runs/sparse-anchor-acquisition-protocol-full"
    if not (artifact / "summary.json").is_file():
        pytest.skip("full acquisition-protocol artifact not present")
    audit = audit_anchor_acquisition_protocol_artifacts(artifact)
    assert audit["passed"]
    assert audit["duplicate_key_count"] == 0
    assert audit["unexpected_numeric_nan_count"] == 0
