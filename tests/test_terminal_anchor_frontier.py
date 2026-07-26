from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.diagnostics.sparse_anchor_voi import (
    AnchorSelection,
    condition_piecewise_residual,
    non_anchor_rmse,
)
from rogii.diagnostics.terminal_anchor_frontier import (
    condition_causal_piecewise,
    frontier_segment_regions,
    requested_fractions,
    select_terminal_anchors,
)


def _selection(indices: list[int], positions: np.ndarray) -> AnchorSelection:
    local = np.asarray(indices, dtype=np.int64)
    return AnchorSelection(
        policy="endpoint_inclusive",
        requested_k=len(indices),
        local_indices=local,
        normalized_positions=positions[local],
        requested_positions=positions[local],
        acquisition_scores=np.full(len(local), np.nan),
        minimum_spacing=0.0,
    )


def test_terminal_placement_is_target_free_and_permutation_invariant() -> None:
    assert "target" not in inspect.signature(select_terminal_anchors).parameters
    md = np.linspace(1.0, 100.0, 100)
    target = np.arange(100, dtype=np.float64)
    first = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=5
    )
    _ = target[::-1]
    second = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=5
    )
    np.testing.assert_array_equal(first.local_indices, second.local_indices)


def test_locked_fraction_rules_are_exact() -> None:
    np.testing.assert_allclose(
        requested_fractions("uniform", 3), [0.25, 0.5, 0.75]
    )
    np.testing.assert_allclose(
        requested_fractions("endpoint_inclusive", 3),
        [1 / 3, 2 / 3, 1.0],
    )
    np.testing.assert_allclose(
        requested_fractions("near_terminal", 1), [0.975]
    )
    np.testing.assert_allclose(
        requested_fractions("near_terminal", 4),
        [0.25, 0.5, 0.75, 0.975],
    )


def test_causal_prediction_is_invariant_to_all_future_anchor_values() -> None:
    positions = np.linspace(0.1, 1.0, 10)
    selection = _selection([2, 7], positions)
    base = np.linspace(100.0, 101.0, 10)
    truth = base.copy()
    truth[2] += 3.0
    truth[7] -= 9.0
    changed = truth.copy()
    changed[7] += 1_000.0
    first = condition_causal_piecewise(base, truth, selection)
    second = condition_causal_piecewise(base, changed, selection)
    np.testing.assert_array_equal(first[:7], second[:7])


def test_changing_not_yet_observed_anchor_cannot_change_earlier_rows() -> None:
    positions = np.linspace(0.05, 1.0, 20)
    selection = _selection([3, 9, 16], positions)
    base = np.zeros(20)
    truth = np.zeros(20)
    truth[[3, 9, 16]] = [2.0, 4.0, 6.0]
    changed = truth.copy()
    changed[16] = -500.0
    first = condition_causal_piecewise(base, truth, selection)
    second = condition_causal_piecewise(base, changed, selection)
    np.testing.assert_array_equal(first[:16], second[:16])


def test_acausal_interpolation_and_causal_carry_match_hand_calculation() -> None:
    positions = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    selection = _selection([2, 4], positions)
    base = np.zeros(6)
    truth = np.zeros(6)
    truth[2] = 4.0
    truth[4] = 10.0
    acausal = condition_piecewise_residual(
        base, truth, positions, selection
    )
    expected_acausal = np.interp(
        positions, [0.0, 0.4, 0.8], [0.0, 4.0, 10.0], right=10.0
    )
    np.testing.assert_allclose(acausal, expected_acausal)
    causal = condition_causal_piecewise(base, truth, selection)
    np.testing.assert_array_equal(causal, [0.0, 0.0, 4.0, 4.0, 10.0, 10.0])
    assert not np.array_equal(acausal, causal)


def test_k_zero_exactly_reproduces_fixed_base_for_both_modes() -> None:
    positions = np.linspace(0.1, 1.0, 5)
    selection = _selection([], positions)
    base = np.arange(5, dtype=np.float64)
    truth = base + 10.0
    assert np.array_equal(
        condition_piecewise_residual(base, truth, positions, selection), base
    )
    assert np.array_equal(
        condition_causal_piecewise(base, truth, selection), base
    )


def test_anchor_rows_are_exact_for_acausal_and_causal() -> None:
    positions = np.linspace(0.1, 1.0, 8)
    selection = _selection([1, 5, 7], positions)
    base = np.full(8, 50.0)
    truth = np.linspace(55.0, 62.0, 8)
    acausal = condition_piecewise_residual(
        base, truth, positions, selection
    )
    causal = condition_causal_piecewise(base, truth, selection)
    np.testing.assert_array_equal(
        acausal[selection.local_indices], truth[selection.local_indices]
    )
    np.testing.assert_array_equal(
        causal[selection.local_indices], truth[selection.local_indices]
    )


def test_non_anchor_metric_excludes_td_anchor() -> None:
    truth = np.zeros(3)
    prediction = np.asarray([3.0, 4.0, 100.0])
    is_anchor = np.asarray([False, False, True])
    assert non_anchor_rmse(truth, prediction, is_anchor) == pytest.approx(5 / np.sqrt(2))


def test_td_and_975_snap_correctly_on_irregular_md() -> None:
    md = np.asarray([1.0, 7.0, 31.0, 80.0, 100.0])
    td = select_terminal_anchors(
        md, 0.0, policy="endpoint_inclusive", k=1
    )
    near = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=1
    )
    np.testing.assert_array_equal(td.local_indices, [4])
    np.testing.assert_array_equal(near.local_indices, [4])
    assert td.normalized_positions[0] == 1.0


def test_short_suffix_and_duplicate_snaps_use_unique_nearest_rows() -> None:
    md = np.asarray([10.0, 10.01, 10.02])
    selection = select_terminal_anchors(
        md, 0.0, policy="endpoint_inclusive", k=7
    )
    assert selection.achieved_k == 3
    assert len(np.unique(selection.local_indices)) == 3


def test_frontier_segments_separate_pre_between_post_and_anchor() -> None:
    positions = np.linspace(0.1, 1.0, 8)
    selection = _selection([2, 5], positions)
    labels = frontier_segment_regions(8, selection)
    assert labels.tolist() == [
        "pre_first",
        "pre_first",
        "anchor",
        "between",
        "between",
        "anchor",
        "post_last",
        "post_last",
    ]


def test_full_locked_artifact_has_exact_row_well_fold_coverage() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / ".runs/final-p3-path-blend-full/oof_predictions.csv.gz"
    if not artifact.is_file():
        pytest.skip("locked full OOF artifact is not present")
    frame = pd.read_csv(
        artifact,
        usecols=["well_id", "row_index", "fold"],
        dtype={"well_id": "string", "row_index": "int64", "fold": "int8"},
    )
    assert len(frame) == 3_783_989
    assert frame["well_id"].nunique() == 773
    assert set(frame["fold"].unique()) == {0, 1, 2, 3, 4}
    assert not frame.duplicated(["well_id", "row_index"]).any()


def test_terminal_placement_and_causal_prediction_are_deterministic() -> None:
    md = np.linspace(1.0, 100.0, 50)
    first = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=6
    )
    second = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=6
    )
    np.testing.assert_array_equal(first.local_indices, second.local_indices)
    base = np.linspace(10.0, 20.0, 50)
    truth = base + np.sin(md)
    np.testing.assert_array_equal(
        condition_causal_piecewise(base, truth, first),
        condition_causal_piecewise(base, truth, second),
    )

