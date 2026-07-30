from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from rogii.physics_alignment import (
    ALIGNMENT_STATUS_IDENTITY,
    ALIGNMENT_STATUS_OK,
    CONTROL_GR_ORDER_SHUFFLE,
    METHOD_BEAM,
    METHOD_NCC,
    METHOD_PARTICLE_FILTER,
    SafePhysicsAlignmentConfig,
    SafePhysicsAlignmentFamily,
)


def _synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rows = 72
    known = 28
    md = np.arange(rows, dtype=np.float64)
    tvt = 10_000.0 + 0.15 * md
    gr = 70.0 + 12.0 * np.sin(tvt * 0.35) + 2.0 * np.cos(md * 0.2)
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 100.0 + md,
            "Y": 200.0 - md,
            "Z": -8_000.0 - 0.1 * md,
            "GR": gr,
            "TVT_input": np.where(md < known, tvt, np.nan),
            "TVT": tvt,
            "ANCC": np.arange(rows) + 999_000.0,
        }
    )
    tw_tvt = np.linspace(9_990.0, 10_030.0, 241)
    typewell = pd.DataFrame(
        {
            "TVT": tw_tvt,
            "GR": 70.0 + 12.0 * np.sin(tw_tvt * 0.35),
            "Geology": "forbidden",
        }
    )
    incumbent = np.full(rows - known, tvt[known - 1], dtype=np.float64)
    return horizontal, typewell, incumbent


def _family() -> SafePhysicsAlignmentFamily:
    return SafePhysicsAlignmentFamily(
        SafePhysicsAlignmentConfig(
            state_step_ft=1.0,
            state_radius_ft=20.0,
            maximum_state_count=64,
            minimum_steps=8,
            maximum_steps=12,
            rows_per_step=6,
            ncc_windows=(3, 5),
            beam_width=8,
            beam_ensemble_paths=3,
            particles=32,
            smoothing_rows=3,
        )
    )


def test_safe_physics_methods_are_finite_deterministic_and_serializable() -> None:
    horizontal, typewell, incumbent = _synthetic_well()
    family = _family()
    for method in (METHOD_NCC, METHOD_PARTICLE_FILTER, METHOD_BEAM):
        first = family.predict(
            method=method,
            horizontal_frame=horizontal,
            typewell_frame=typewell,
            incumbent_prediction=incumbent,
            well_id="dynamic-well-a",
            seeds=(11, 12),
        )
        second = family.predict(
            method=method,
            horizontal_frame=horizontal,
            typewell_frame=typewell,
            incumbent_prediction=incumbent,
            well_id="dynamic-well-a",
            seeds=(11, 12),
        )
        assert first.status == ALIGNMENT_STATUS_OK
        assert np.array_equal(first.prediction, second.prediction)
        assert np.isfinite(first.prediction).all()
        assert np.isfinite(first.uncertainty).all()
        assert np.isfinite(first.effective_particle_count).all()
        restored = pickle.loads(pickle.dumps(first))
        assert np.array_equal(restored.prediction, first.prediction)
        assert restored.well_id == "dynamic-well-a"


def test_unsafe_extra_columns_never_change_prediction() -> None:
    horizontal, typewell, incumbent = _synthetic_well()
    family = _family()
    first = family.predict(
        method=METHOD_PARTICLE_FILTER,
        horizontal_frame=horizontal,
        typewell_frame=typewell,
        incumbent_prediction=incumbent,
        well_id="dynamic-well-b",
        seeds=(21, 22),
    )
    changed_horizontal = horizontal.copy()
    changed_horizontal["TVT"] = np.linspace(-1e9, 1e9, len(horizontal))
    changed_horizontal["ANCC"] = np.linspace(1e9, -1e9, len(horizontal))
    changed_typewell = typewell.copy()
    changed_typewell["Geology"] = "changed-secret"
    second = family.predict(
        method=METHOD_PARTICLE_FILTER,
        horizontal_frame=changed_horizontal,
        typewell_frame=changed_typewell,
        incumbent_prediction=incumbent,
        well_id="dynamic-well-b",
        seeds=(21, 22),
    )
    assert np.array_equal(first.prediction, second.prediction)


def test_corrupted_alignment_explicitly_falls_back_to_incumbent() -> None:
    horizontal, typewell, incumbent = _synthetic_well()
    horizontal["GR"] = np.nan
    horizontal.loc[0, "GR"] = 1.0
    result = _family().predict(
        method=METHOD_NCC,
        horizontal_frame=horizontal,
        typewell_frame=typewell,
        incumbent_prediction=incumbent,
        well_id="dynamic-well-c",
    )
    assert result.status == ALIGNMENT_STATUS_IDENTITY
    assert result.reason
    assert np.array_equal(result.prediction, incumbent)


def test_gr_order_shuffle_is_reproducible_and_labeled_as_control() -> None:
    horizontal, typewell, incumbent = _synthetic_well()
    family = _family()
    result = family.predict(
        method=METHOD_PARTICLE_FILTER,
        horizontal_frame=horizontal,
        typewell_frame=typewell,
        incumbent_prediction=incumbent,
        well_id="dynamic-well-d",
        control_id=CONTROL_GR_ORDER_SHUFFLE,
        seeds=(31, 32),
    )
    repeat = family.predict(
        method=METHOD_PARTICLE_FILTER,
        horizontal_frame=horizontal,
        typewell_frame=typewell,
        incumbent_prediction=incumbent,
        well_id="dynamic-well-d",
        control_id=CONTROL_GR_ORDER_SHUFFLE,
        seeds=(31, 32),
    )
    assert result.control_id == CONTROL_GR_ORDER_SHUFFLE
    assert np.array_equal(result.prediction, repeat.prediction)
