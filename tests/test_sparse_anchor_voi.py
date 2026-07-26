from __future__ import annotations

import gzip
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rogii.diagnostics.sparse_anchor_voi import (
    AnchorSelection,
    _DeterministicGzipCsvWriter,
    _build_well_contexts,
    _read_aligned_baselines,
    anchor_mask,
    condition_path_pca_posterior,
    condition_piecewise_residual,
    fit_conditional_calibration,
    non_anchor_rmse,
    normalized_md_positions,
    pooled_rmse,
    safe_output_directory,
    segment_regions,
    select_disagreement_anchors,
    select_uniform_anchors,
    validate_complete_coverage,
)
from rogii.models.path_pca import PathPCAState


def _conditional_fixture() -> tuple[
    PathPCAState, object, dict[str, np.ndarray], dict[str, np.ndarray]
]:
    rng = np.random.default_rng(41)
    identifiers = [f"{index:08d}" for index in range(12)]
    features = {well_id: rng.normal(size=7) for well_id in identifiers}
    curves = {
        well_id: np.cumsum(rng.normal(scale=0.2, size=128))
        for well_id in identifiers
    }
    x = np.stack(list(features.values()))
    y = np.stack(list(curves.values()))
    pca = PCA(n_components=5, svd_solver="full").fit(y)
    coefficients = pca.transform(y)
    regressor = Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))]
    ).fit(x, coefficients[:, :3])
    state = PathPCAState(pca=pca, regressor=regressor, predicted_components=3)
    calibration = fit_conditional_calibration(
        state,
        identifiers[:10],
        identifiers[10:],
        features,
        curves,
    )
    return state, calibration, features, curves


def _selection(indices: list[int], positions: np.ndarray) -> AnchorSelection:
    local = np.asarray(indices, dtype=np.int64)
    return AnchorSelection(
        policy="uniform",
        requested_k=len(indices),
        local_indices=local,
        normalized_positions=positions[local],
        requested_positions=positions[local],
        acquisition_scores=np.full(len(local), np.nan),
        minimum_spacing=0.0,
    )


def test_uniform_anchor_positions_do_not_accept_or_read_target() -> None:
    signature = inspect.signature(select_uniform_anchors)
    assert "target" not in signature.parameters
    first = select_uniform_anchors(np.array([2.0, 7.0, 20.0]), 0.0, 2)
    second = select_uniform_anchors(np.array([2.0, 7.0, 20.0]), 0.0, 2)
    np.testing.assert_array_equal(first.local_indices, second.local_indices)


def test_validation_target_permutation_does_not_change_anchor_positions() -> None:
    md = np.linspace(1.0, 101.0, 101)
    score = np.sin(md) ** 2
    target = np.arange(len(md), dtype=np.float64)
    first = select_disagreement_anchors(md, 0.0, score, 8)
    _ = target[::-1]  # A target permutation is deliberately irrelevant to the API.
    second = select_disagreement_anchors(md, 0.0, score, 8)
    np.testing.assert_array_equal(first.local_indices, second.local_indices)


def test_unrevealed_target_changes_do_not_change_piecewise_prediction() -> None:
    positions = np.linspace(0.1, 1.0, 10)
    selection = _selection([2, 7], positions)
    base = np.linspace(10.0, 11.0, 10)
    truth = base + np.linspace(0.0, 4.0, 10)
    changed = truth.copy()
    changed[[0, 1, 3, 4, 5, 6, 8, 9]] += 1_000.0
    np.testing.assert_array_equal(
        condition_piecewise_residual(base, truth, positions, selection),
        condition_piecewise_residual(base, changed, positions, selection),
    )


def test_revealed_anchor_value_changes_only_through_conditioning_path() -> None:
    positions = np.linspace(0.1, 1.0, 10)
    selection = _selection([4], positions)
    base = np.zeros(10)
    first_truth = np.zeros(10)
    second_truth = first_truth.copy()
    second_truth[4] = 5.0
    first = condition_piecewise_residual(
        base, first_truth, positions, selection
    )
    second = condition_piecewise_residual(
        base, second_truth, positions, selection
    )
    expected = np.interp(positions, [0.0, positions[4]], [0.0, 5.0], right=5.0)
    np.testing.assert_allclose(second - first, expected)


def test_k_zero_exactly_reproduces_each_method_baseline() -> None:
    positions = np.linspace(0.1, 1.0, 6)
    empty = _selection([], positions)
    base = np.arange(6, dtype=np.float64)
    truth = base + 1.0
    assert np.array_equal(
        condition_piecewise_residual(base, truth, positions, empty), base
    )
    state, calibration, features, _ = _conditional_fixture()
    path = base + 10.0
    actual = condition_path_pca_posterior(
        state,
        calibration,
        features["00000010"],
        100.0,
        path,
        truth,
        positions,
        empty,
    )
    assert np.array_equal(actual, path)
    disagreement_empty = select_disagreement_anchors(
        np.arange(1.0, 7.0), 0.0, np.arange(6.0), 0
    )
    assert disagreement_empty.achieved_k == 0


def test_selected_anchor_rows_equal_revealed_tvt_for_both_methods() -> None:
    positions = np.linspace(0.1, 1.0, 8)
    selection = _selection([2, 6], positions)
    truth = np.linspace(100.0, 110.0, 8)
    base = np.full(8, 99.0)
    piecewise = condition_piecewise_residual(base, truth, positions, selection)
    assert np.array_equal(piecewise[selection.local_indices], truth[selection.local_indices])

    state, calibration, features, _ = _conditional_fixture()
    coefficients = np.zeros(state.pca.components_.shape[0])
    coefficients[: state.predicted_components] = state.regressor.predict(
        features["00000010"][None, :]
    )[0]
    curve = state.pca.mean_ + coefficients @ state.pca.components_
    path_base = 100.0 + np.interp(
        positions, np.linspace(0.0, 1.0, len(curve)), curve
    )
    posterior = condition_path_pca_posterior(
        state,
        calibration,
        features["00000010"],
        100.0,
        path_base,
        truth,
        positions,
        selection,
    )
    assert np.array_equal(posterior[selection.local_indices], truth[selection.local_indices])


def test_non_anchor_metric_excludes_revealed_rows() -> None:
    truth = np.zeros(4)
    prediction = np.array([100.0, 1.0, 2.0, 3.0])
    mask = np.array([True, False, False, False])
    assert non_anchor_rmse(truth, prediction, mask) == pytest.approx(
        np.sqrt((1.0 + 4.0 + 9.0) / 3.0)
    )


def test_irregular_md_snapping_and_interpolation_use_physical_md() -> None:
    md = np.array([1.0, 2.0, 50.0, 100.0])
    positions = normalized_md_positions(md, 0.0)
    selection = select_uniform_anchors(md, 0.0, 2)
    # 1/3 and 2/3 of the MD horizon snap to 50 and 100, not row thirds.
    np.testing.assert_array_equal(selection.local_indices, np.array([2, 3]))
    base = np.zeros(4)
    truth = np.array([0.0, 0.0, 5.0, 10.0])
    corrected = condition_piecewise_residual(base, truth, positions, selection)
    assert corrected[1] < corrected[2]


def test_short_suffix_duplicate_snap_uses_unique_nearest_unused_rows() -> None:
    selection = select_uniform_anchors(np.array([10.0, 10.1]), 0.0, 8)
    assert selection.achieved_k == 2
    assert len(np.unique(selection.local_indices)) == 2


def test_conditional_fit_state_uses_only_declared_outer_training_wells() -> None:
    state, first, features, curves = _conditional_fixture()
    changed = dict(curves)
    changed["00000010"] = changed["00000010"] + 10_000.0
    second = fit_conditional_calibration(
        state,
        [f"{index:08d}" for index in range(10)],
        ["00000010", "00000011"],
        features,
        changed,
    )
    assert first.state_sha256 == second.state_sha256
    np.testing.assert_array_equal(
        first.coefficient_covariance, second.coefficient_covariance
    )


def test_outer_validation_wells_are_rejected_from_inner_fit() -> None:
    state, _, features, curves = _conditional_fixture()
    with pytest.raises(ValueError, match="disjoint"):
        fit_conditional_calibration(
            state,
            ["00000000", "00000010"],
            ["00000010"],
            features,
            curves,
        )


def test_oof_artifact_keys_and_targets_must_match_exactly(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["00000001", "00000001"],
            "row_index": [3, 4],
            "fold": [0, 0],
            "y_true": [10.0, 11.0],
            "y_pred": [9.0, 10.0],
        }
    )
    paths = [tmp_path / f"{name}.csv.gz" for name in ("blend", "p3", "path")]
    for path in paths:
        frame.to_csv(path, index=False)
    aligned = _read_aligned_baselines(paths[0], paths[1], paths[2])
    assert len(aligned) == 2
    changed = frame.copy()
    changed.loc[1, "row_index"] = 5
    changed.to_csv(paths[2], index=False)
    with pytest.raises(ValueError, match="keys"):
        _read_aligned_baselines(paths[0], paths[1], paths[2])


def test_diagnostic_well_loader_uses_inference_safe_horizontal_view() -> None:
    source = inspect.getsource(_build_well_contexts)
    assert "load_horizontal_inference" in source
    assert "Geology" not in source
    assert "ANCC" not in source
    assert "PNG" not in source


def test_output_guard_prevents_writes_under_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError, match="competition data"):
        safe_output_directory(data, "illegal-run", data)


def test_anchor_selection_and_gzip_output_are_deterministic(tmp_path: Path) -> None:
    md = np.linspace(1.0, 50.0, 50)
    score = np.cos(md) ** 2
    first = select_disagreement_anchors(md, 0.0, score, 4)
    second = select_disagreement_anchors(md, 0.0, score, 4)
    np.testing.assert_array_equal(first.local_indices, second.local_indices)
    frame = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
    outputs = [tmp_path / "one.csv.gz", tmp_path / "two.csv.gz"]
    for output in outputs:
        with _DeterministicGzipCsvWriter(output) as writer:
            writer.write(frame)
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_pooled_metric_matches_manual_sse() -> None:
    truth = np.array([0.0, 0.0, 0.0])
    prediction = np.array([1.0, 2.0, 3.0])
    assert pooled_rmse(truth, prediction) == pytest.approx(np.sqrt(14.0 / 3.0))


def test_every_well_requires_complete_prediction_row_coverage() -> None:
    validate_complete_coverage(np.array([3, 4, 5]), np.array([3, 4, 5]))
    with pytest.raises(ValueError, match="coverage"):
        validate_complete_coverage(np.array([3, 5]), np.array([3, 4, 5]))
    with pytest.raises(ValueError, match="coverage"):
        validate_complete_coverage(np.array([3, 3, 4]), np.array([3, 4, 5]))


def test_target_permutation_changes_only_revealed_values_and_downstream_output() -> None:
    positions = np.linspace(0.05, 1.0, 20)
    md = positions * 100.0
    score = np.linspace(1.0, 2.0, 20)
    selection = select_disagreement_anchors(md, 0.0, score, 2)
    truth = np.arange(20, dtype=np.float64)
    permuted = truth[::-1].copy()
    repeated_selection = select_disagreement_anchors(md, 0.0, score, 2)
    np.testing.assert_array_equal(
        selection.local_indices, repeated_selection.local_indices
    )
    first = condition_piecewise_residual(
        np.zeros(20), truth, positions, selection
    )
    second = condition_piecewise_residual(
        np.zeros(20), permuted, positions, repeated_selection
    )
    assert not np.array_equal(
        truth[selection.local_indices], permuted[selection.local_indices]
    )
    assert not np.array_equal(first, second)
    assert segment_regions(positions, selection).tolist() == segment_regions(
        positions, repeated_selection
    ).tolist()
