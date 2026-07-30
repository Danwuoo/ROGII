from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.nested_blend import (
    COMPONENT_NAMES,
    cross_fit_component_shrinkage,
    decompose_candidate_correction,
    fit_component_shrinkage,
)


def _frame() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for well_number in range(2):
            well_id = f"well-{fold}-{well_number}"
            for row_index in range(12):
                x = (row_index - 5.5) / 5.5
                truth = 100.0 + fold + 0.5 * x
                incumbent = truth + 2.0 + 0.3 * x
                candidate = incumbent - 1.0 - 0.2 * x + 0.1 * np.sin(row_index)
                rows.append(
                    {
                        "well_id": well_id,
                        "row_index": row_index,
                        "fold": fold,
                        "y_true": truth,
                        "incumbent": incumbent,
                        "physics": candidate,
                    }
                )
    return pd.DataFrame(rows)


def test_component_decomposition_reconstructs_candidate() -> None:
    frame = _frame()
    components = decompose_candidate_correction(
        frame,
        incumbent_column="incumbent",
        candidate_column="physics",
    )
    reconstructed = (
        frame["incumbent"].to_numpy()
        + components.loc[:, list(COMPONENT_NAMES)].sum(axis=1).to_numpy()
    )
    assert np.allclose(reconstructed, frame["physics"].to_numpy(), atol=1e-12)


def test_outer_validation_labels_do_not_affect_its_fitted_state() -> None:
    frame = _frame()
    first = fit_component_shrinkage(
        frame,
        outer_fold=2,
        incumbent_column="incumbent",
        candidate_column="physics",
    )
    modified = frame.copy()
    modified.loc[modified["fold"].eq(2), "y_true"] += 1_000_000.0
    second = fit_component_shrinkage(
        modified,
        outer_fold=2,
        incumbent_column="incumbent",
        candidate_column="physics",
    )
    assert first.weights == second.weights
    assert set(first.training_well_ids).isdisjoint(first.validation_well_ids)


def test_cross_fitted_component_prediction_is_complete_and_fold_safe() -> None:
    frame = _frame()
    result, provenance = cross_fit_component_shrinkage(
        frame,
        incumbent_column="incumbent",
        candidate_columns=("physics",),
    )
    assert np.isfinite(result["nested_physics"]).all()
    assert len(provenance["physics"]) == 5
    for record in provenance["physics"]:
        assert record["training_validation_intersection"] == []
        assert record["outer_validation_labels_used_for_fit"] is False
