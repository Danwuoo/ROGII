from __future__ import annotations

from pathlib import Path

import numpy as np

from rogii.models.base import InferenceWell, RunContext, TrainingWell
from rogii.models.path_pca import PathPCAModel, _prefix_features
from rogii.models.path_pca_anchor import (
    AnchorPathPCAModel,
    _anchor_inclusive_target_curve,
)
from rogii.models.path_pca_nested import (
    ALPHA_GRID,
    COMPONENT_GRID,
    NestedPathPCAModel,
)


def _well(index: int, *, rows: int = 320, known: int = 120) -> TrainingWell:
    md = np.arange(rows, dtype=np.float64)
    x = 1_000.0 * index + md
    y = 2_000.0 - 0.4 * md
    z = -9_000.0 + 0.02 * md
    gr = 80.0 + np.sin(md / 17.0 + index)
    slope = (index - 2.0) * 0.002
    full_tvt = 11_000.0 + 5.0 * index + slope * md
    tvt_input = full_tvt.copy()
    tvt_input[known:] = np.nan
    features = np.column_stack([md, x, y, z, gr, tvt_input])
    inference = InferenceWell(
        well_id=f"{index:08x}",
        row_index=np.arange(known, rows, dtype=np.int64),
        anchor=float(full_tvt[known - 1]),
        features=features,
    )
    return TrainingWell(inference=inference, target=full_tvt[known:])


def test_prefix_encoding_never_reads_hidden_tvt_input() -> None:
    training = _well(0)
    features = training.inference.features.copy()
    known = int(training.inference.row_index[0])
    baseline = _prefix_features(features, known)
    features[known:, 5] = 1e12
    np.testing.assert_array_equal(_prefix_features(features, known), baseline)


def test_path_pca_fit_predicts_one_finite_value_per_suffix_row(
    tmp_path: Path,
) -> None:
    training = [_well(index) for index in range(5)]
    model = PathPCAModel()
    state = model.fit(
        training,
        context=RunContext(
            fold=0,
            seed=20260711,
            workers=1,
            work_dir=tmp_path,
        ),
    )
    prediction = model.predict(state, training[2].inference)
    assert prediction.shape == training[2].target.shape
    assert np.isfinite(prediction).all()
    assert np.sqrt(np.mean((prediction - training[2].target) ** 2)) < 1.0


def test_nested_path_pca_selects_only_declared_candidates(
    tmp_path: Path,
) -> None:
    training = [_well(index) for index in range(9)]
    model = NestedPathPCAModel()
    context = RunContext(
        fold=2,
        seed=20260711,
        workers=1,
        work_dir=tmp_path,
    )
    state = model.fit(training, context=context)
    assert state.predicted_components in COMPONENT_GRID
    assert state.selected_alpha in ALPHA_GRID
    assert np.isfinite(state.inner_pooled_rmse)
    assert 0 < len(state.inner_candidate_rmse) <= len(COMPONENT_GRID) * len(ALPHA_GRID)

    prediction = model.predict(state, training[4].inference)
    assert prediction.shape == training[4].target.shape
    assert np.isfinite(prediction).all()


def test_nested_path_pca_selection_is_deterministic(tmp_path: Path) -> None:
    training = [_well(index) for index in range(9)]
    model = NestedPathPCAModel()
    context = RunContext(
        fold=1,
        seed=20260711,
        workers=1,
        work_dir=tmp_path,
    )
    first = model.fit(training, context=context)
    second = model.fit(training, context=context)
    assert first.predicted_components == second.predicted_components
    assert first.selected_alpha == second.selected_alpha
    assert first.inner_pooled_rmse == second.inner_pooled_rmse


def test_anchor_path_curve_starts_at_visible_anchor(tmp_path: Path) -> None:
    training = [_well(index) for index in range(6)]
    assert _anchor_inclusive_target_curve(training[0])[0] == 0.0
    model = AnchorPathPCAModel()
    state = model.fit(
        training,
        context=RunContext(
            fold=0,
            seed=20260711,
            workers=1,
            work_dir=tmp_path,
        ),
    )
    assert abs(float(state.pca.mean_[0])) < 1e-12
    assert np.max(np.abs(state.pca.components_[:, 0])) < 1e-12
