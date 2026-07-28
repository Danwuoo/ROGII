from __future__ import annotations

import numpy as np

from rogii.external_models.components import validate_residual_components
from rogii.external_models.contracts import (
    FixedGridSequenceWell,
    ResidualTrainingWell,
    SequenceModelInput,
)
from rogii.external_models.targets import (
    SmoothTargetConfig,
    grid_prediction_to_row_components,
    low_pass_shape,
    make_grid_residual_target,
)


def _input() -> SequenceModelInput:
    row_index = np.arange(3, 14, dtype=np.int64)
    row_coordinate = np.linspace(0.0, 1.0, len(row_index)) ** 1.2
    horizontal = np.column_stack(
        [
            np.linspace(0.0, 1.0, 16),
            np.sin(np.linspace(0.0, 4.0, 16)),
        ]
    )
    typewell = np.column_stack(
        [
            np.linspace(0.0, 1.0, 16),
            np.cos(np.linspace(0.0, 4.0, 16)),
        ]
    )
    sequence = FixedGridSequenceWell(
        well_id="synthetic",
        row_index=row_index,
        known_count=3,
        anchor_tvt=100.0,
        horizontal=horizontal,
        typewell=typewell,
        output_coordinate=row_coordinate,
        horizontal_feature_names=("horizontal:coordinate", "horizontal:gr"),
        typewell_feature_names=("typewell:coordinate", "typewell:gr"),
        source_horizontal_rows=14,
        source_typewell_rows=16,
        schema_version="test",
    )
    grid = np.linspace(0.0, 1.0, 16)
    x = np.stack(
        [
            np.sin(2.0 * np.pi * grid),
            np.cos(2.0 * np.pi * grid),
            grid,
            1.0 - grid,
        ]
    )
    incumbent = 100.0 + 2.0 * row_coordinate
    return SequenceModelInput(
        sequence=sequence,
        incumbent_prediction=incumbent,
        x=x,
        grid_coordinate=grid,
        channel_names=(
            "horizontal:gr",
            "typewell:gr",
            "horizontal:coordinate",
            "typewell:coordinate",
        ),
    )


def test_low_pass_shape_is_identifiable() -> None:
    coordinate = np.linspace(0.0, 1.0, 64)
    raw = (
        3.0
        + 4.0 * coordinate
        + np.sin(2.0 * np.pi * coordinate)
        + 0.5 * np.sin(24.0 * np.pi * coordinate)
    )
    result = low_pass_shape(raw, coordinate, retained_modes=8)
    basis = coordinate - coordinate.mean()
    assert abs(float(result.mean())) < 1e-12
    assert abs(float(result @ basis)) < 1e-10
    assert np.std(result) < np.std(raw)


def test_grid_target_and_row_reconstruction_are_finite_and_identifiable() -> None:
    inference = _input()
    coordinate = inference.row_coordinate
    residual = (
        2.5
        - 3.0 * (coordinate - coordinate.mean())
        + 1.2 * np.sin(2.0 * np.pi * coordinate)
        + 0.4 * np.sin(18.0 * np.pi * coordinate)
    )
    target = inference.incumbent_prediction + residual
    training = ResidualTrainingWell(
        inference=inference,
        incumbent_prediction=inference.incumbent_prediction,
        target=target,
    )
    config = SmoothTargetConfig(retained_dct_modes=8, maximum_absolute_correction=20.0)
    grid_components = make_grid_residual_target(training, config=config)
    assert grid_components.correction.shape == inference.grid_coordinate.shape
    assert np.isfinite(grid_components.correction).all()

    row_components = grid_prediction_to_row_components(
        grid_components.correction,
        inference,
        config=config,
    )
    validate_residual_components(
        row_components,
        inference.incumbent_prediction,
        inference.row_coordinate,
    )
    assert row_components.eligible
    assert row_components.final.shape == inference.row_index.shape
    assert np.isfinite(row_components.final).all()
    assert abs(float(row_components.shape.mean())) < 1e-10
    basis = coordinate - coordinate.mean()
    assert abs(float(row_components.shape @ basis)) < 1e-8
