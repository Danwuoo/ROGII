from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.external_models.components import validate_residual_components
from rogii.external_models.targets import grid_prediction_to_row_components
from rogii.external_models.ttm_audit import (
    paired_transfer_bootstrap,
    prediction_column,
)
from rogii.portfolio_metrics import validate_oof_frame
from test_ttm_embeddings import synthetic_input


def test_residual_projection_constraints_and_finite_output() -> None:
    inference = synthetic_input("well-a")
    components = grid_prediction_to_row_components(
        np.sin(np.linspace(0.0, np.pi, 512)),
        inference,
    )
    validated = validate_residual_components(
        components,
        inference.incumbent_prediction,
        inference.row_coordinate,
    )
    assert validated is components
    assert np.isfinite(components.final).all()
    assert abs(float(np.mean(components.slope))) < 1.0e-10
    assert abs(float(np.mean(components.shape))) < 1.0e-10
    centered = inference.row_coordinate - np.mean(inference.row_coordinate)
    assert abs(float(components.shape @ centered)) < 1.0e-9


def test_exact_oof_keys_duplicate_missing_and_transfer_bootstrap() -> None:
    rows = []
    for fold in range(5):
        for well in range(2):
            well_id = f"{fold}-{well}"
            for row in range(3):
                truth = float(fold + well + row)
                rows.append(
                    {
                        "well_id": well_id,
                        "row_index": row,
                        "fold": fold,
                        "y_true": truth,
                        "y_pred__ttm_random_frozen": truth + 1.0,
                        "y_pred__ttm_pretrained_frozen": truth + 0.5,
                    }
                )
    frame = pd.DataFrame(rows)
    validation = validate_oof_frame(
        frame,
        (
            prediction_column("ttm_random_frozen"),
            prediction_column("ttm_pretrained_frozen"),
        ),
        expected_folds=range(5),
    )
    assert validation["rows"] == 30
    bootstrap = paired_transfer_bootstrap(frame, samples=100, seed=7)
    assert bootstrap["point_delta_rmse"] < 0.0
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_oof_frame(
            duplicate,
            (
                prediction_column("ttm_random_frozen"),
                prediction_column("ttm_pretrained_frozen"),
            ),
        )
    with pytest.raises(ValueError, match="coverage differs"):
        validate_oof_frame(
            frame.iloc[1:],
            (
                prediction_column("ttm_random_frozen"),
                prediction_column("ttm_pretrained_frozen"),
            ),
            expected_keys=frame,
        )
