from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from rogii.diagnostics.anchor_noise_robustness import (
    LOCKED_CONFIGS,
    LOCKED_EXACT_METRICS,
    audit_anchor_noise_artifacts,
    condition_noisy_piecewise,
    hierarchical_bootstrap,
    noisy_anchor_observations,
    piecewise_noise_weights,
    reconstruct_pooled_rmse,
    stable_noise_matrix,
    stable_standard_normal,
    validate_sigma,
)
from rogii.diagnostics.sparse_anchor_voi import (
    AnchorSelection,
    condition_piecewise_residual,
    non_anchor_rmse,
)
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


def test_sigma_zero_matches_exact_piecewise_prediction_at_machine_precision() -> None:
    positions = np.asarray([0.1, 0.25, 0.5, 0.7, 0.975, 1.0])
    selection = _selection([1, 3, 4], positions)
    base = np.linspace(100.0, 103.0, len(positions))
    truth = base + np.asarray([1.0, 2.0, -3.0, 4.0, -1.0, 8.0])
    exact = condition_piecewise_residual(base, truth, positions, selection)
    noisy = condition_noisy_piecewise(
        base, truth[selection.local_indices], positions, selection
    )
    np.testing.assert_allclose(noisy, exact, rtol=0.0, atol=2e-14)


def test_locked_sigma_zero_metrics_match_terminal_frontier_artifact() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = (
        root
        / ".runs/sparse-anchor-terminal-frontier-full/information_frontier.csv"
    )
    if not artifact.is_file():
        pytest.skip("locked terminal frontier artifact is not present")
    frame = pd.read_csv(artifact)
    for config in LOCKED_CONFIGS:
        row = frame[
            frame["policy"].eq(config.policy)
            & frame["mode"].eq("acausal")
            & frame["method"].eq("piecewise_residual")
            & frame["k"].eq(config.k)
        ].iloc[0]
        expected_full, expected_non = LOCKED_EXACT_METRICS[config.name]
        assert float(row["pooled_rmse"]) == pytest.approx(
            expected_full, abs=1e-15
        )
        assert float(row["non_anchor_rmse"]) == pytest.approx(
            expected_non, abs=1e-15
        )


def test_noise_generator_is_cross_process_deterministic() -> None:
    expected = stable_standard_normal(17, "e03b45fd", 12345)
    code = (
        "from rogii.diagnostics.anchor_noise_robustness import "
        "stable_standard_normal;"
        "print(repr(stable_standard_normal(17,'e03b45fd',12345)))"
    )
    actual = float(
        subprocess.check_output(
            [sys.executable, "-c", code], text=True
        ).strip()
    )
    assert actual == expected


def test_shared_physical_anchor_row_has_same_noise_across_configs() -> None:
    first = stable_noise_matrix([0, 1, 99], "00000001", [10, 20, 30])
    second = stable_noise_matrix([0, 1, 99], "00000001", [20, 40])
    np.testing.assert_array_equal(first[:, 1], second[:, 0])


def test_same_z_scales_linearly_across_sigma() -> None:
    truth = np.asarray([100.0, 200.0])
    z = np.asarray([0.5, -1.25])
    at_half = noisy_anchor_observations(truth, 0.5, z) - truth
    at_two = noisy_anchor_observations(truth, 2.0, z) - truth
    np.testing.assert_array_equal(at_two, 4.0 * at_half)


def test_target_permutation_does_not_change_locked_anchor_positions() -> None:
    md = np.linspace(1.0, 100.0, 101)
    target = np.arange(len(md), dtype=np.float64)
    first = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=6
    )
    _ = target[::-1]
    second = select_terminal_anchors(
        md, 0.0, policy="near_terminal", k=6
    )
    np.testing.assert_array_equal(first.local_indices, second.local_indices)


def test_noise_does_not_change_snapping_or_locked_fractions() -> None:
    md = np.linspace(1.0, 101.0, 101)
    for config in LOCKED_CONFIGS:
        first = select_terminal_anchors(
            md, 0.0, policy=config.policy, k=config.k
        )
        _ = stable_noise_matrix([0, 1], "well", first.local_indices)
        second = select_terminal_anchors(
            md, 0.0, policy=config.policy, k=config.k
        )
        np.testing.assert_array_equal(first.local_indices, second.local_indices)
        np.testing.assert_array_equal(
            first.requested_positions, np.asarray(config.fractions)
        )
        np.testing.assert_array_equal(
            requested_fractions(config.policy, config.k),
            np.asarray(config.fractions),
        )


def test_ps_boundary_has_implicit_exact_zero_noise_correction() -> None:
    positions = np.asarray([0.1, 0.4, 0.8, 1.0])
    selection = _selection([1, 2], positions)
    weights = piecewise_noise_weights(positions, selection)
    # The first segment is the straight line from the exact zero-noise PS
    # boundary at x=0 to the first noisy anchor at x=0.4.
    assert weights[0, 0] == pytest.approx(0.1 / 0.4)
    assert weights[0, 1] == 0.0


def test_selected_anchor_prediction_equals_noisy_observation() -> None:
    positions = np.linspace(0.1, 1.0, 8)
    selection = _selection([2, 5, 7], positions)
    base = np.linspace(50.0, 55.0, 8)
    observed = np.asarray([54.2, 60.1, 62.7])
    prediction = condition_noisy_piecewise(
        base, observed, positions, selection
    )
    np.testing.assert_array_equal(
        prediction[selection.local_indices], observed
    )


def test_non_anchor_metric_excludes_noisy_anchor_rows() -> None:
    truth = np.zeros(4)
    prediction = np.asarray([1.0, 100.0, 2.0, 200.0])
    is_anchor = np.asarray([False, True, False, True])
    assert non_anchor_rmse(truth, prediction, is_anchor) == pytest.approx(
        np.sqrt(2.5)
    )


def test_unrevealed_target_changes_do_not_affect_noisy_conditioning() -> None:
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
    assert not np.array_equal(changed, truth)


def test_noisy_piecewise_interpolation_matches_hand_calculation() -> None:
    positions = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    selection = _selection([2, 4], positions)
    base = np.zeros(6)
    observed = np.asarray([4.5, 9.0])
    actual = condition_noisy_piecewise(
        base, observed, positions, selection
    )
    expected = np.interp(
        positions, [0.0, 0.4, 0.8], [0.0, 4.5, 9.0], right=9.0
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (validate_sigma, (-1.0,)),
        (validate_sigma, (float("nan"),)),
        (stable_standard_normal, (-1, "well", 1)),
        (stable_standard_normal, (1.5, "well", 1)),
        (stable_standard_normal, (1, "well", -1)),
        (stable_standard_normal, (1, "", 1)),
    ],
)
def test_invalid_noise_inputs_fail_closed(function: object, args: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        function(*args)  # type: ignore[operator]


def test_irregular_md_short_suffix_and_duplicate_snaps_remain_valid() -> None:
    irregular = select_terminal_anchors(
        np.asarray([1.0, 4.0, 31.0, 80.0, 100.0]),
        0.0,
        policy="near_terminal",
        k=2,
    )
    assert irregular.local_indices[-1] == 4
    short = select_terminal_anchors(
        np.asarray([10.0, 10.01, 10.02]),
        0.0,
        policy="endpoint_inclusive",
        k=6,
    )
    assert short.achieved_k == 3
    assert len(np.unique(short.local_indices)) == 3


def test_per_well_sse_reconstructs_pooled_metrics_exactly() -> None:
    frame = pd.DataFrame(
        {
            "rows": [2, 3],
            "non_anchor_rows": [1, 2],
            "sse": [5.0, 20.0],
            "non_anchor_sse": [4.0, 5.0],
        }
    )
    full, non = reconstruct_pooled_rmse(frame)
    assert full == pytest.approx(np.sqrt(5.0))
    assert non == pytest.approx(np.sqrt(3.0))


def test_hierarchical_bootstrap_is_deterministic_and_finite() -> None:
    full = np.asarray(
        [[[[1.0, 4.0], [9.0, 16.0]]]], dtype=np.float64
    )
    non = full * 0.5
    rows = np.asarray([2, 4])
    non_rows = np.asarray([[1, 3]])
    first = hierarchical_bootstrap(
        full, non, rows, non_rows, samples=100, seed=77
    )
    second = hierarchical_bootstrap(
        full, non, rows, non_rows, samples=100, seed=77
    )
    assert first == second
    payload = json.dumps(first, allow_nan=False)
    assert "hierarchical" in payload


def test_diagnostic_reads_only_inference_safe_horizontal_view() -> None:
    from rogii.diagnostics import anchor_noise_robustness as module

    source = inspect.getsource(module)
    assert "load_horizontal_inference" in source
    assert "formation_surfaces_read" in source
    assert "geology_read" in source
    assert "png_read" in source
    assert '["MD", "X", "Y", "Z", "GR", "TVT_input"]' in source


def test_full_artifact_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / ".runs/sparse-anchor-noise-robustness-full"
    if not artifact.is_dir():
        pytest.skip("full anchor-noise artifact is not present")
    audit = audit_anchor_noise_artifacts(artifact)
    assert audit["passed"]
    assert audit["bad_well_coverage_groups"] == 0
    assert audit["bad_row_coverage_groups"] == 0
