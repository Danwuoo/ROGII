from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.data import (
    HORIZONTAL_INFERENCE_COLUMNS,
    TYPEWELL_INFERENCE_COLUMNS,
    WellFiles,
)
from rogii.heel_registration import HeelRegistrationConfig
from rogii.models.base import InferenceWell, RunContext, TrainingWell
from rogii.models.heel_registered_anchor_student import (
    HeelRegisteredAnchorStudentModel,
    heel_registered_inference_schema,
    heel_registered_state_provenance,
    wrap_sparse_anchor_student_state,
)
from rogii.models.sparse_anchor_student import (
    SparseAnchorInferenceWell,
    SparseAnchorPrediction,
    SparseAnchorStudentState,
    StudentEstimatorState,
)
from rogii.sparse_anchor import select_anchor_rows


_CONFIG = HeelRegistrationConfig(
    shift_min_ft=-20.0,
    shift_max_ft=20.0,
    shift_step_ft=0.5,
    rolling_median_width=1,
)


def _write_well(
    root: Path,
    *,
    unsafe_target_offset: float = 0.0,
    unsafe_label: str = "unsafe-a",
) -> tuple[WellFiles, np.ndarray, int]:
    root.mkdir(parents=True, exist_ok=True)
    well_id = "00000001"
    rows, known = 36, 12
    md = np.arange(rows, dtype=np.float64)
    truth = 100.0 + md
    tvt_input = truth.copy()
    tvt_input[known:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 1_000.0 + md,
            "Y": 2_000.0 - md,
            "Z": -8_000.0 + 0.1 * md,
            "GR": truth,
            "TVT_input": tvt_input,
            "TVT": truth + unsafe_target_offset,
            "Geology": unsafe_label,
            "ANCC": unsafe_target_offset + md,
        }
    )
    horizontal_path = root / f"{well_id}__horizontal_well.csv"
    horizontal.to_csv(horizontal_path, index=False)
    typewell_tvt = np.arange(0.0, 301.0, dtype=np.float64)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": typewell_tvt,
            "Geology": unsafe_label,
        }
    )
    typewell_path = root / f"{well_id}__typewell.csv"
    typewell.to_csv(typewell_path, index=False)
    (root / f"{well_id}.png").write_bytes(unsafe_label.encode("utf-8"))
    return WellFiles(well_id, horizontal_path, typewell_path), truth, known


def _base_payload(
    well_id: str, truth: np.ndarray, known: int
) -> SparseAnchorInferenceWell:
    row_index = np.arange(known, len(truth), dtype=np.int64)
    anchor = float(truth[known - 1])
    placement = select_anchor_rows(
        np.arange(known, len(truth), dtype=np.float64),
        float(known - 1),
        row_indices=row_index,
    )
    engine_payload = InferenceWell(
        well_id=well_id,
        row_index=row_index,
        anchor=anchor,
        features=np.zeros((len(row_index), 1), dtype=np.float64),
    )
    return SparseAnchorInferenceWell(
        well_id=well_id,
        row_index=row_index,
        anchor=anchor,
        features=np.asarray([1.0, 2.0], dtype=np.float64),
        descriptor_names=("one", "two"),
        p3=engine_payload,
        path_pca=engine_payload,
        placement=placement,
    )


def _base_prediction(
    payload: SparseAnchorInferenceWell, final: np.ndarray
) -> SparseAnchorPrediction:
    values = np.asarray(final, dtype=np.float64)
    return SparseAnchorPrediction(
        p3=values.copy(),
        path_pca=values.copy(),
        baseline=values.copy(),
        anchor_delta=np.zeros(8, dtype=np.float64),
        anchor_tvt=values[payload.placement.local_indices].copy(),
        final=values.copy(),
    )


def _base_state(well_id: str = "00000001") -> SparseAnchorStudentState:
    student = StudentEstimatorState(
        kind="zero",
        estimator=None,
        target_pca=None,
        training_well_ids=(well_id,),
        descriptor_width=2,
        target_components=None,
        alpha=None,
        random_seed=20260728,
    )
    return SparseAnchorStudentState(
        p3_state={"engine": "p3"},
        path_pca_state={"engine": "path"},
        student_state=student,
        training_well_ids=(well_id,),
        fold=3,
    )


def _prepared_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    base_shift: float,
) -> tuple[
    HeelRegisteredAnchorStudentModel,
    object,
    SparseAnchorPrediction,
    np.ndarray,
]:
    well, truth, known = _write_well(tmp_path / "data")
    payload = _base_payload(well.well_id, truth, known)
    base_prediction = _base_prediction(payload, truth[known:] - base_shift)
    model = HeelRegisteredAnchorStudentModel(
        student_kind="zero", registration_config=_CONFIG
    )
    monkeypatch.setattr(
        model.base_model,
        "prepare",
        lambda actual_well, *, split, cache_root: payload,
    )
    monkeypatch.setattr(
        model.base_model,
        "predict_components",
        lambda state, actual_payload: base_prediction,
    )
    prepared = model.prepare(well, split="test", cache_root=tmp_path / "cache")
    state = model.from_base_state(_base_state())
    return model, (state, prepared), base_prediction, truth[known:]


def test_synthetic_common_mode_shift_updates_rows_and_anchors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model, pair, base, truth = _prepared_model(monkeypatch, tmp_path, base_shift=8.0)
    state, prepared = pair
    prediction = model.predict_components(state, prepared)
    assert prediction.posterior_mean_shift == pytest.approx(8.0, abs=1e-8)
    np.testing.assert_allclose(prediction.final, truth, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(
        prediction.anchor_tvt,
        base.anchor_tvt + prediction.posterior_mean_shift,
        rtol=0.0,
        atol=0.0,
    )
    assert prediction.final.shape == prepared.row_index.shape
    assert np.isfinite(prediction.final).all()
    assert np.isfinite(prediction.registration.candidate_cost).all()
    assert np.isfinite(prediction.registration.posterior_weight).all()
    assert prediction.registration.posterior_weight.sum() == pytest.approx(1.0)


def test_noop_registration_preserves_base_prediction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model, pair, base, _ = _prepared_model(monkeypatch, tmp_path, base_shift=0.0)
    state, prepared = pair
    prediction = model.predict_components(state, prepared)
    assert prediction.posterior_mean_shift == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(prediction.final, base.final, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        prediction.anchor_tvt, base.anchor_tvt, rtol=0.0, atol=1e-10
    )


def test_prepare_ignores_unsafe_extra_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_well, truth, known = _write_well(
        tmp_path / "first", unsafe_target_offset=0.0, unsafe_label="first"
    )
    second_well, _, _ = _write_well(
        tmp_path / "second", unsafe_target_offset=1e9, unsafe_label="second"
    )
    payload = _base_payload(first_well.well_id, truth, known)
    model = HeelRegisteredAnchorStudentModel(
        student_kind="zero", registration_config=_CONFIG
    )
    monkeypatch.setattr(
        model.base_model,
        "prepare",
        lambda actual_well, *, split, cache_root: payload,
    )
    first = model.prepare(first_well, split="train", cache_root=tmp_path / "cache-a")
    second = model.prepare(second_well, split="train", cache_root=tmp_path / "cache-b")
    assert tuple(first.horizontal.columns) == HORIZONTAL_INFERENCE_COLUMNS
    assert tuple(first.typewell.columns) == TYPEWELL_INFERENCE_COLUMNS
    pd.testing.assert_frame_equal(first.horizontal, second.horizontal)
    pd.testing.assert_frame_equal(first.typewell, second.typewell)
    assert "TVT" not in first.horizontal
    assert "Geology" not in first.typewell


def test_fit_delegates_only_base_payload_and_records_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    well, truth, known = _write_well(tmp_path / "data")
    payload = _base_payload(well.well_id, truth, known)
    model = HeelRegisteredAnchorStudentModel(
        student_kind="zero", registration_config=_CONFIG
    )
    monkeypatch.setattr(
        model.base_model,
        "prepare",
        lambda actual_well, *, split, cache_root: payload,
    )
    prepared = model.prepare(well, split="train", cache_root=tmp_path / "cache")
    context = RunContext(fold=3, seed=20260728, workers=1, work_dir=tmp_path / "work")
    seen: dict[str, object] = {}

    def fake_fit(training_wells: list[TrainingWell], *, context: RunContext):
        seen["training"] = training_wells
        seen["context"] = context
        return _base_state(well.well_id)

    monkeypatch.setattr(model.base_model, "fit", fake_fit)
    state = model.fit(
        [TrainingWell(prepared, truth[known:])],
        context=context,
    )
    delegated = seen["training"]
    assert isinstance(delegated, list)
    assert delegated[0].inference is payload
    assert seen["context"] is context
    provenance = heel_registered_state_provenance(state)
    assert provenance["fold"] == 3
    assert provenance["random_seed"] == 20260728
    assert provenance["training_well_ids"] == [well.well_id]
    assert provenance["registration_fit_wells"] == []
    assert not provenance["registration_uses_cross_well_learned_statistics"]


def test_loaded_base_state_pickle_schema_and_strict_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model, pair, _, _ = _prepared_model(monkeypatch, tmp_path, base_shift=4.0)
    state, prepared = pair
    restored = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    first = model.predict(restored, prepared)
    second = model.predict(state, prepared)
    np.testing.assert_array_equal(first, second)
    direct = wrap_sparse_anchor_student_state(_base_state(), config=_CONFIG)
    assert direct.training_well_ids == state.training_well_ids
    schema = heel_registered_inference_schema()
    assert schema["horizontal_source_columns"] == list(HORIZONTAL_INFERENCE_COLUMNS)
    assert schema["typewell_source_columns"] == list(TYPEWELL_INFERENCE_COLUMNS)
    assert schema["target_columns_used"] == []
    assert model.manifest()["registration"]["config"]["quantile_count"] == 21

    bad = _base_prediction(prepared.base, second)
    bad.final[0] = np.nan
    monkeypatch.setattr(
        model.base_model,
        "predict_components",
        lambda base_state, base_payload: bad,
    )
    with pytest.raises(ValueError, match="invalid base prediction"):
        model.predict(state, prepared)


def test_prepare_rejects_base_row_misalignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    well, truth, known = _write_well(tmp_path / "data")
    payload = _base_payload(well.well_id, truth, known)
    payload.row_index[0] += 1
    model = HeelRegisteredAnchorStudentModel(
        student_kind="zero", registration_config=_CONFIG
    )
    monkeypatch.setattr(
        model.base_model,
        "prepare",
        lambda actual_well, *, split, cache_root: payload,
    )
    with pytest.raises(ValueError, match="row alignment"):
        model.prepare(well, split="test", cache_root=tmp_path / "cache")
