from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from rogii.diagnostics.anchor_error_realism import stable_anchor_latents
from rogii.diagnostics.anchor_outlier_qc import (
    METHODS,
    PRIMARY_METHOD,
    QC_MAX_REJECTS,
    QC_SCENARIOS,
    QC_THRESHOLD,
    _batch_local_spike_decisions,
    anchor_confusion_counts,
    audit_anchor_outlier_qc_artifacts,
    condition_with_accepted_anchors,
    hierarchical_qc_bootstrap,
    local_leave_one_out_scores,
    local_spike_reject_7p5,
    reconstruct_pooled_metrics,
    validate_qc_contract,
)
from rogii.diagnostics.anchor_noise_robustness import (
    LOCKED_EXACT_METRICS,
    condition_noisy_piecewise,
)
from rogii.diagnostics.sparse_anchor_voi import AnchorSelection
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


def test_locked_matrix_and_methods_are_exact() -> None:
    assert [scenario.name for scenario in QC_SCENARIOS] == [
        "exact_clean",
        "gaussian_1_clean",
        "moderate_background_no_outlier",
        "moderate_m5_control",
        "outlier_p05_m10",
        "outlier_p10_m10",
        "outlier_p20_m10",
        "moderate_background_p05_m10",
        "moderate_background_p10_m10",
        "severe_background_no_outlier",
        "severe",
    ]
    assert METHODS == (
        "no_qc",
        "local_spike_reject_7p5",
        "oracle_indicator_reject",
        "oracle_clean_values",
    )
    assert PRIMARY_METHOD == "local_spike_reject_7p5"


def test_qc_signature_excludes_truth_indicator_and_seed() -> None:
    parameters = inspect.signature(local_spike_reject_7p5).parameters
    forbidden = {
        "true_outlier_indicator",
        "true_target",
        "target",
        "seed",
        "realization_seed",
    }
    assert not forbidden & set(parameters)


def test_scores_are_exactly_common_bias_invariant() -> None:
    x = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.975])
    residual = np.asarray([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
    first = local_leave_one_out_scores(x, residual)
    second = local_leave_one_out_scores(x, residual + 123.5)
    np.testing.assert_allclose(first, second, rtol=0.0, atol=2e-14)
    base = np.linspace(100.0, 105.0, 6)
    first_decision = local_spike_reject_7p5(base + residual, base, x)
    second_decision = local_spike_reject_7p5(
        base + residual + 123.5, base, x
    )
    np.testing.assert_array_equal(
        first_decision.accepted, second_decision.accepted
    )


def test_scores_are_cross_process_deterministic() -> None:
    code = (
        "import json,numpy as np;"
        "from rogii.diagnostics.anchor_outlier_qc import "
        "local_leave_one_out_scores;"
        "x=np.array([.1,.2,.4,.6,.8,.975]);"
        "r=np.array([1.,2.,3.,5.,8.,13.]);"
        "print(json.dumps(local_leave_one_out_scores(x,r).tolist()))"
    )
    expected = local_leave_one_out_scores(
        np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.975]),
        np.asarray([1.0, 2.0, 3.0, 5.0, 8.0, 13.0]),
    )
    actual = np.asarray(
        json.loads(
            subprocess.check_output([sys.executable, "-c", code], text=True)
        )
    )
    np.testing.assert_array_equal(actual, expected)


def test_deterministic_tie_break_prefers_lower_anchor_order() -> None:
    x = np.arange(6, dtype=np.float64)
    base = np.zeros(6)
    # Equal endpoint scores of 10; anchor order 1 must be removed first.
    observed = np.asarray([10.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    decision = local_spike_reject_7p5(observed, base, x)
    assert decision.rejected[0]
    assert decision.decision_iteration[0] == 1


def test_single_interior_m10_spike_is_detected() -> None:
    x = np.arange(6, dtype=np.float64)
    base = np.full(6, 100.0)
    observed = base.copy()
    observed[3] += 10.0
    decision = local_spike_reject_7p5(observed, base, x)
    assert decision.rejected[3]
    assert decision.score_at_decision[3] == pytest.approx(10.0)


def test_endpoint_one_sided_scores_match_hand_calculation() -> None:
    x = np.arange(6, dtype=np.float64)
    residual = np.asarray([10.0, 1.0, 2.0, 3.0, 4.0, -5.0])
    score = local_leave_one_out_scores(x, residual)
    # First extrapolates the line through (1,1),(2,2) to x=0 -> 0.
    assert score[0] == pytest.approx(10.0)
    # Last extrapolates the line through (3,3),(4,4) to x=5 -> 5.
    assert score[5] == pytest.approx(10.0)


def test_flat_m5_is_not_rejected_and_flat_m10_is_rejected() -> None:
    x = np.arange(6, dtype=np.float64)
    base = np.zeros(6)
    m5 = np.zeros(6)
    m5[3] = 5.0
    m10 = np.zeros(6)
    m10[3] = 10.0
    assert not local_spike_reject_7p5(m5, base, x).rejected.any()
    assert local_spike_reject_7p5(m10, base, x).rejected[3]


def test_threshold_is_strictly_greater_than_7p5() -> None:
    x = np.arange(6, dtype=np.float64)
    base = np.zeros(6)
    observed = np.zeros(6)
    observed[2] = QC_THRESHOLD
    assert not local_spike_reject_7p5(observed, base, x).rejected.any()


def test_maximum_two_rejects_and_scores_recompute() -> None:
    x = np.arange(6, dtype=np.float64)
    residual = np.asarray([20.0, 0.0, 10.0, 0.0, -10.0, 0.0])
    decision = local_spike_reject_7p5(residual, np.zeros(6), x)
    assert decision.rejected.sum() == QC_MAX_REJECTS
    rejected_iterations = sorted(
        decision.decision_iteration[decision.rejected].tolist()
    )
    assert rejected_iterations == [1, 2]
    batch = _batch_local_spike_decisions(x, residual[None, :])
    np.testing.assert_array_equal(
        batch["rejected"][0], decision.rejected
    )


def test_conditioner_preserves_accepted_and_not_rejected_observation() -> None:
    positions = np.linspace(0.1, 1.0, 9)
    selection = _selection([1, 3, 4, 5, 6, 8], positions)
    base = np.zeros(9)
    observed = np.asarray([1.0, 3.0, 20.0, 5.0, 6.0, 8.0])
    accepted = np.asarray([True, True, False, True, True, True])
    prediction = condition_with_accepted_anchors(
        base, observed, positions, selection, accepted
    )
    np.testing.assert_array_equal(
        prediction[selection.local_indices[accepted]], observed[accepted]
    )
    assert prediction[selection.local_indices[2]] != observed[2]


def test_rejected_interior_gap_interpolation_matches_hand_calculation() -> None:
    positions = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    selection = _selection([0, 1, 2, 3, 4, 5], positions)
    base = np.zeros(6)
    observed = np.asarray([1.0, 2.0, 99.0, 4.0, 5.0, 6.0])
    accepted = np.asarray([True, True, False, True, True, True])
    prediction = condition_with_accepted_anchors(
        base, observed, positions, selection, accepted
    )
    assert prediction[2] == pytest.approx(3.0)


def test_reject_final_anchor_carries_previous_accepted_residual() -> None:
    positions = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    selection = _selection([0, 1, 2, 3, 4, 6], positions)
    base = np.zeros(8)
    observed = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    accepted = np.asarray([True, True, True, True, True, False])
    prediction = condition_with_accepted_anchors(
        base, observed, positions, selection, accepted
    )
    assert prediction[-1] == pytest.approx(5.0)


def test_changing_unrevealed_target_does_not_change_qc_or_prediction() -> None:
    positions = np.linspace(0.1, 1.0, 10)
    selection = _selection([1, 2, 4, 5, 7, 9], positions)
    base = np.linspace(100.0, 101.0, 10)
    observed = base[selection.local_indices].copy()
    observed[2] += 10.0
    decision = local_spike_reject_7p5(
        observed, base[selection.local_indices], positions[selection.local_indices]
    )
    first = condition_with_accepted_anchors(
        base, observed, positions, selection, decision.accepted
    )
    unrevealed_target = base + 1000.0
    unrevealed_target[selection.local_indices] = observed
    second = condition_with_accepted_anchors(
        base, observed, positions, selection, decision.accepted
    )
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(unrevealed_target, base)


def test_target_permutation_does_not_change_placement_or_latents() -> None:
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


def test_shared_physical_anchor_rows_keep_same_latents() -> None:
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


def test_confusion_matrix_and_endpoint_multiplicity_inputs() -> None:
    rejected = np.asarray([True, True, False, False, True])
    truth = np.asarray([True, False, True, False, True])
    assert anchor_confusion_counts(rejected, truth) == {
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "tn": 1,
    }
    orders = np.asarray([1, 2, 3, 4, 6])
    assert int(np.sum(truth & np.isin(orders, [1, 6]))) == 2
    assert int(np.sum(truth & np.isin(orders, [2, 3, 4, 5]))) == 1


def test_per_well_sse_reconstructs_pooled_metrics() -> None:
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


def test_hierarchical_bootstrap_is_deterministic_and_paired() -> None:
    full = np.zeros((1, 2, 1, 2, 2), dtype=np.float64)
    full[0, 0, 0] = [[1.0, 4.0], [9.0, 16.0]]
    full[0, 1, 0] = [[2.0, 8.0], [18.0, 32.0]]
    non = full * 0.5
    rows = np.asarray([2, 4])
    non_rows = np.asarray([[1, 3]])
    first = hierarchical_qc_bootstrap(
        full, non, rows, non_rows, samples=100, seed=77
    )
    second = hierarchical_qc_bootstrap(
        full, non, rows, non_rows, samples=100, seed=77
    )
    assert first == second
    assert len(first) == 2


@pytest.mark.parametrize(
    ("threshold", "rejects"),
    [
        (6.0, 2),
        (8.0, 2),
        (float("nan"), 2),
        (7.5, 1),
        (7.5, 3),
    ],
)
def test_invalid_qc_contract_fails_closed(
    threshold: float, rejects: int
) -> None:
    with pytest.raises(ValueError):
        validate_qc_contract(threshold, rejects)


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
    with pytest.raises(ValueError):
        local_spike_reject_7p5(
            np.zeros(2), np.zeros(2), np.asarray([0.5, 1.0])
        )


def test_exact_no_error_reference_matches_terminal_frontier_artifact() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = (
        root
        / ".runs/sparse-anchor-terminal-frontier-full/information_frontier.csv"
    )
    if not artifact.is_file():
        pytest.skip("terminal frontier artifact not present")
    frame = pd.read_csv(artifact)
    for config, policy in (
        ("near_k6", "near_terminal"),
        ("td_k6", "endpoint_inclusive"),
    ):
        row = frame[
            frame["policy"].eq(policy)
            & frame["mode"].eq("acausal")
            & frame["method"].eq("piecewise_residual")
            & frame["k"].eq(6)
        ].iloc[0]
        expected = LOCKED_EXACT_METRICS[config]
        assert float(row["pooled_rmse"]) == pytest.approx(expected[0], abs=1e-15)
        assert float(row["non_anchor_rmse"]) == pytest.approx(
            expected[1], abs=1e-15
        )


def test_no_qc_conditioner_is_exact_prior_piecewise_conditioner() -> None:
    positions = np.linspace(0.1, 1.0, 8)
    selection = _selection([0, 1, 2, 4, 5, 7], positions)
    base = np.linspace(50.0, 55.0, 8)
    observed = np.asarray([51.0, 52.0, 53.0, 56.0, 57.0, 60.0])
    accepted = np.ones(6, dtype=bool)
    actual = condition_with_accepted_anchors(
        base, observed, positions, selection, accepted
    )
    expected = condition_noisy_piecewise(
        base, observed, positions, selection
    )
    np.testing.assert_array_equal(actual, expected)


def test_module_does_not_read_forbidden_inputs() -> None:
    from rogii.diagnostics import anchor_outlier_qc as module

    source = inspect.getsource(module)
    assert "load_full_horizontal" not in source
    assert '"formation_surfaces_read": False' in source
    assert '"geology_read": False' in source
    assert '"png_read": False' in source
    assert '"typewell_columns_read": []' in source


def test_full_artifact_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / ".runs/sparse-anchor-outlier-qc-full"
    if not artifact.is_dir():
        pytest.skip("full outlier-QC artifact is not present")
    audit = audit_anchor_outlier_qc_artifacts(artifact)
    assert audit["passed"]
    assert audit["duplicate_key_count"] == 0
    assert audit["unexpected_numeric_nan_count"] == 0
