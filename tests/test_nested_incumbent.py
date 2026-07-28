from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.data import WellFiles, validate_prefix
from rogii.models.base import InferenceWell, RunContext, TrainingWell
from rogii.models.heel_registered_anchor_student import (
    HeelRegisteredAnchorStudentModel,
)
from rogii.models.sparse_anchor_student import (
    SparseAnchorStudentState,
    StudentEstimatorState,
)
from rogii.external_models.nested_incumbent import (
    PREDICTION_COLUMN,
    NestedHeelIncumbentRefitter,
    NestedIncumbentError,
)


def _write_dataset(
    root: Path,
    *,
    wells_per_fold: int = 2,
    missing_target_folds: set[int] | None = None,
    realistic: bool = False,
) -> tuple[Path, Path, dict[int, tuple[str, ...]]]:
    data_root = root / "competition"
    train = data_root / "train"
    train.mkdir(parents=True)
    missing = set() if missing_target_folds is None else set(missing_target_folds)
    records: list[dict[str, object]] = []
    by_fold: dict[int, list[str]] = {fold: [] for fold in range(5)}
    rows = 48 if realistic else 18
    known = 18 if realistic else 7

    for fold in range(5):
        for member in range(wells_per_fold):
            well_id = f"{fold + 1:02d}{member + 1:06d}"
            by_fold[fold].append(well_id)
            md = np.cumsum(np.linspace(4.0, 6.0, rows))
            offset = 3.0 * fold + 0.5 * member
            truth = 1_000.0 + offset + 0.18 * md + 0.0008 * md**2
            tvt_input = truth.copy()
            tvt_input[known:] = np.nan
            gr = 65.0 + 7.0 * np.sin(truth / 13.0 + 0.1 * member)
            horizontal_data: dict[str, object] = {
                "MD": md,
                "X": 500.0 + offset + 0.85 * md,
                "Y": 1_500.0 - 0.25 * md,
                "Z": -8_000.0 + 0.12 * md,
                "GR": gr,
                "TVT_input": tvt_input,
            }
            if fold not in missing:
                horizontal_data["TVT"] = truth
            horizontal = pd.DataFrame(horizontal_data)
            horizontal_path = train / f"{well_id}__horizontal_well.csv"
            horizontal.to_csv(horizontal_path, index=False)

            typewell_tvt = np.linspace(900.0, 1_200.0, 121)
            typewell = pd.DataFrame(
                {
                    "TVT": typewell_tvt,
                    "GR": 65.0 + 7.0 * np.sin(typewell_tvt / 13.0 + 0.1 * member),
                }
            )
            typewell.to_csv(
                train / f"{well_id}__typewell.csv",
                index=False,
            )
            records.append(
                {
                    "well_id": well_id,
                    "fold": fold,
                    "fold_version": "nested-test-v1",
                }
            )

    fold_config = root / "folds.csv"
    pd.DataFrame(records).to_csv(fold_config, index=False)
    return (
        data_root,
        fold_config,
        {fold: tuple(sorted(ids)) for fold, ids in by_fold.items()},
    )


def _mocked_current_model(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    HeelRegisteredAnchorStudentModel,
    list[tuple[tuple[str, ...], RunContext]],
]:
    model = HeelRegisteredAnchorStudentModel(student_kind="zero")
    fit_calls: list[tuple[tuple[str, ...], RunContext]] = []

    def prepare(
        well: WellFiles,
        *,
        split: str,
        cache_root: Path,
    ) -> InferenceWell:
        del split, cache_root
        # The mock intentionally asks only for TVT_input.  Validation fixtures
        # omit TVT entirely, so an accidental validation-label read fails.
        frame = pd.read_csv(well.horizontal_path, usecols=["TVT_input"])
        known_count, prediction_count = validate_prefix(frame["TVT_input"])
        row_index = np.arange(
            known_count,
            known_count + prediction_count,
            dtype=np.int64,
        )
        return InferenceWell(
            well_id=well.well_id,
            row_index=row_index,
            anchor=float(known_count),
            features=np.arange(prediction_count, dtype=np.float64)[:, None],
        )

    def fit(
        training_wells: list[TrainingWell],
        *,
        context: RunContext,
    ):
        ids = tuple(str(item.inference.well_id) for item in training_wells)
        fit_calls.append((ids, context))
        student = StudentEstimatorState(
            kind="zero",
            estimator=None,
            target_pca=None,
            training_well_ids=ids,
            descriptor_width=1,
            target_components=None,
            alpha=None,
            random_seed=context.seed,
        )
        base_state = SparseAnchorStudentState(
            p3_state={"seed": context.seed},
            path_pca_state={"seed": context.seed},
            student_state=student,
            training_well_ids=ids,
            fold=context.fold,
        )
        return model.from_base_state(base_state)

    def predict(state: object, well: InferenceWell) -> np.ndarray:
        seed = float(getattr(state, "random_seed"))
        return seed + np.arange(len(well.row_index), dtype=np.float64)

    monkeypatch.setattr(model, "prepare", prepare)
    monkeypatch.setattr(model, "fit", fit)
    monkeypatch.setattr(model, "predict", predict)
    return model, fit_calls


def _refitter(
    root: Path,
    *,
    data_root: Path,
    fold_config: Path,
    model: HeelRegisteredAnchorStudentModel,
    seeds: tuple[int, ...],
    reciprocal_state_reuse: bool = False,
) -> NestedHeelIncumbentRefitter:
    return NestedHeelIncumbentRefitter(
        data_root=data_root,
        fold_config=fold_config,
        cache_root=root / "cache",
        checkpoint_root=root / "checkpoints",
        seeds=seeds,
        workers=1,
        model=model,
        reciprocal_state_reuse=reciprocal_state_reuse,
    )


def test_inner_refit_excludes_outer_and_inner_labels_and_resumes_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root, fold_config, by_fold = _write_dataset(
        tmp_path,
        missing_target_folds={0, 1},
    )
    model, fit_calls = _mocked_current_model(monkeypatch)
    refitter = _refitter(
        tmp_path,
        data_root=data_root,
        fold_config=fold_config,
        model=model,
        seeds=(11, 13),
    )
    spec = refitter.fold_spec(outer_fold=0, inner_validation_fold=1)
    expected_training = tuple(sorted((*by_fold[2], *by_fold[3], *by_fold[4])))
    assert spec.training_well_ids == expected_training
    assert spec.validation_well_ids == by_fold[1]
    assert spec.excluded_outer_well_ids == by_fold[0]

    first = refitter.predict_inner_validation(
        outer_fold=0,
        inner_validation_fold=1,
    )
    assert [call[0] for call in fit_calls] == [
        expected_training,
        expected_training,
    ]
    assert [call[1].fold for call in fit_calls] == [1, 1]
    assert [call[1].seed for call in fit_calls] == [11, 13]
    assert all(not record.reused for record in first.checkpoints)
    assert all(Path(record.checkpoint_path).is_file() for record in first.checkpoints)
    assert all(Path(record.manifest_path).is_file() for record in first.checkpoints)
    for record in first.checkpoints:
        manifest = json.loads(Path(record.manifest_path).read_text(encoding="utf-8"))
        assert manifest["checkpoint_sha256"] == record.checkpoint_sha256
        assert (
            tuple(manifest["provenance"]["ordered_training_well_ids"])
            == expected_training
        )
    frame = first.to_frame()
    assert tuple(frame["well_id"].drop_duplicates()) == by_fold[1]
    assert set(frame["outer_fold"]) == {0}
    assert set(frame["inner_validation_fold"]) == {1}
    for _, group in frame.groupby("well_id"):
        np.testing.assert_array_equal(
            group[PREDICTION_COLUMN],
            12.0 + np.arange(len(group), dtype=np.float64),
        )

    calls_before_resume = len(fit_calls)
    second = refitter.predict_inner_validation(
        outer_fold=0,
        inner_validation_fold=1,
        resume=True,
    )
    assert len(fit_calls) == calls_before_resume
    assert all(record.reused for record in second.checkpoints)
    assert tuple(record.checkpoint_sha256 for record in first.checkpoints) == tuple(
        record.checkpoint_sha256 for record in second.checkpoints
    )
    pd.testing.assert_frame_equal(first.to_frame(), second.to_frame())
    with pytest.raises(FileExistsError, match="already exists"):
        refitter.fit_or_load_seed(
            outer_fold=0,
            inner_validation_fold=1,
            seed=11,
            resume=False,
        )
    monkeypatch.setattr(
        model,
        "predict",
        lambda state, well: np.full(len(well.row_index), np.nan),
    )
    with pytest.raises(NestedIncumbentError, match="invalid rows"):
        refitter.predict_inner_validation(
            outer_fold=0,
            inner_validation_fold=1,
            resume=True,
        )


def test_checkpoint_corruption_and_provenance_change_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root, fold_config, _ = _write_dataset(tmp_path)
    model, fit_calls = _mocked_current_model(monkeypatch)
    refitter = _refitter(
        tmp_path,
        data_root=data_root,
        fold_config=fold_config,
        model=model,
        seeds=(17,),
    )
    artifact = refitter.fit_or_load_seed(
        outer_fold=0,
        inner_validation_fold=2,
        seed=17,
    )
    checkpoint = Path(artifact.record.checkpoint_path)
    with checkpoint.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(NestedIncumbentError, match="manifest provenance mismatch"):
        refitter.fit_or_load_seed(
            outer_fold=0,
            inner_validation_fold=2,
            seed=17,
            resume=True,
        )
    assert len(fit_calls) == 1

    other_root = tmp_path / "mapping-change"
    data_root, fold_config, _ = _write_dataset(other_root)
    model, _ = _mocked_current_model(monkeypatch)
    first_refitter = _refitter(
        other_root,
        data_root=data_root,
        fold_config=fold_config,
        model=model,
        seeds=(19,),
    )
    first_refitter.fit_or_load_seed(
        outer_fold=1,
        inner_validation_fold=3,
        seed=19,
    )
    mapping = pd.read_csv(fold_config, dtype={"well_id": "string"})
    mapping["fold_version"] = "nested-test-v2"
    mapping.to_csv(fold_config, index=False)
    changed = _refitter(
        other_root,
        data_root=data_root,
        fold_config=fold_config,
        model=model,
        seeds=(19,),
    )
    with pytest.raises(NestedIncumbentError, match="manifest provenance mismatch"):
        changed.fit_or_load_seed(
            outer_fold=1,
            inner_validation_fold=3,
            seed=19,
            resume=True,
        )


def test_outer_crossfit_covers_each_outer_training_well_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root, fold_config, by_fold = _write_dataset(tmp_path)
    model, fit_calls = _mocked_current_model(monkeypatch)
    refitter = _refitter(
        tmp_path,
        data_root=data_root,
        fold_config=fold_config,
        model=model,
        seeds=(23,),
    )
    result = refitter.predict_outer_training_oof(outer_fold=0)
    frame = result.to_frame()
    assert len(fit_calls) == 4
    assert set(frame["well_id"]) == set().union(
        by_fold[1],
        by_fold[2],
        by_fold[3],
        by_fold[4],
    )
    assert not frame.duplicated(["well_id", "row_index"]).any()
    for (training_ids, context), inner in zip(fit_calls, (1, 2, 3, 4)):
        expected = tuple(
            sorted(
                well_id
                for fold, fold_ids in by_fold.items()
                if fold not in {0, inner}
                for well_id in fold_ids
            )
        )
        assert training_ids == expected
        assert context.fold == inner


def test_scoped_predictions_reject_outer_before_fit_and_preserve_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root, fold_config, by_fold = _write_dataset(tmp_path)
    model, fit_calls = _mocked_current_model(monkeypatch)
    refitter = _refitter(
        tmp_path,
        data_root=data_root,
        fold_config=fold_config,
        model=model,
        seeds=(31, 37),
    )

    with pytest.raises(NestedIncumbentError, match="outer-validation wells"):
        refitter.predict_wells(
            outer_fold=0,
            inner_validation_fold=1,
            well_ids=(by_fold[0][0],),
        )
    assert fit_calls == []

    with pytest.raises(NestedIncumbentError, match="outside allowed_scope"):
        refitter.predict_wells(
            outer_fold=0,
            inner_validation_fold=1,
            well_ids=(by_fold[1][0],),
            allowed_scope="inner_training",
        )
    assert fit_calls == []

    requested = (by_fold[2][1], by_fold[1][0], by_fold[4][0])
    result = refitter.predict_wells(
        outer_fold=0,
        inner_validation_fold=1,
        well_ids=requested,
        allowed_scope="inner_training_or_validation",
    )
    assert tuple(prediction.well_id for prediction in result.predictions) == requested
    assert tuple(result.requested_well_ids) == requested
    assert len(fit_calls) == 2
    assert all(set(call[0]).isdisjoint(by_fold[0]) for call in fit_calls)
    assert all(set(call[0]).isdisjoint(by_fold[1]) for call in fit_calls)

    frame = result.to_frame()
    assert tuple(frame["well_id"].drop_duplicates()) == requested
    assert not frame.duplicated(["well_id", "row_index"]).any()
    assert np.isfinite(frame[PREDICTION_COLUMN]).all()
    for _, group in frame.groupby("well_id", sort=False):
        np.testing.assert_array_equal(
            group["row_index"].to_numpy(),
            np.arange(7, 18, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            group[PREDICTION_COLUMN].to_numpy(),
            34.0 + np.arange(len(group), dtype=np.float64),
        )


def test_reciprocal_state_reuse_is_exact_auditable_and_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root, fold_config, by_fold = _write_dataset(tmp_path)
    model, fit_calls = _mocked_current_model(monkeypatch)
    original_predict = model.predict
    predict_calls: list[str] = []

    def counted_predict(state: object, well: InferenceWell) -> np.ndarray:
        predict_calls.append(well.well_id)
        return original_predict(state, well)

    monkeypatch.setattr(model, "predict", counted_predict)
    refitter = _refitter(
        tmp_path,
        data_root=data_root,
        fold_config=fold_config,
        model=model,
        seeds=(41, 43),
        reciprocal_state_reuse=True,
    )
    common_ids = (by_fold[2][0], by_fold[3][1], by_fold[4][0])
    forward = refitter.predict_wells(
        outer_fold=0,
        inner_validation_fold=1,
        well_ids=common_ids,
    )
    calls_after_forward = len(predict_calls)
    reciprocal = refitter.predict_wells(
        outer_fold=1,
        inner_validation_fold=0,
        well_ids=common_ids,
    )
    assert len(fit_calls) == 2
    assert calls_after_forward == len(common_ids) * 2
    assert len(predict_calls) == calls_after_forward
    assert all(
        record.reciprocal_source_checkpoint_path is None
        for record in forward.checkpoints
    )
    assert all(
        record.reciprocal_source_checkpoint_path is not None
        for record in reciprocal.checkpoints
    )
    assert all(
        Path(record.reciprocal_source_checkpoint_path).is_file()
        for record in reciprocal.checkpoints
    )
    assert all(
        Path(record.reciprocal_source_manifest_path).is_file()
        for record in reciprocal.checkpoints
    )
    for left, right in zip(
        forward.predictions,
        reciprocal.predictions,
        strict=True,
    ):
        np.testing.assert_array_equal(left.row_index, right.row_index)
        np.testing.assert_array_equal(left.prediction, right.prediction)
    reciprocal_manifests = [
        json.loads(Path(record.manifest_path).read_text(encoding="utf-8"))
        for record in reciprocal.checkpoints
    ]
    assert all(
        manifest["reciprocal_state_reuse"]["source_outer_fold"] == 0
        and manifest["reciprocal_state_reuse"]["source_inner_validation_fold"] == 1
        for manifest in reciprocal_manifests
    )

    calls_before_resume = len(fit_calls)
    resumed = refitter.predict_wells(
        outer_fold=1,
        inner_validation_fold=0,
        well_ids=common_ids,
        resume=True,
    )
    assert len(fit_calls) == calls_before_resume
    assert len(predict_calls) == calls_after_forward
    assert all(record.reused for record in resumed.checkpoints)
    for first, second in zip(
        reciprocal.predictions,
        resumed.predictions,
        strict=True,
    ):
        np.testing.assert_array_equal(first.prediction, second.prediction)

    source = Path(reciprocal.checkpoints[0].reciprocal_source_checkpoint_path)
    with source.open("ab") as handle:
        handle.write(b"stale")
    with pytest.raises(NestedIncumbentError, match="reciprocal reuse source"):
        refitter.predict_wells(
            outer_fold=1,
            inner_validation_fold=0,
            well_ids=common_ids,
            resume=True,
        )


def test_actual_small_heel_model_fit_predict_and_checkpoint_smoke(
    tmp_path: Path,
) -> None:
    data_root, fold_config, by_fold = _write_dataset(
        tmp_path,
        realistic=True,
    )
    model = HeelRegisteredAnchorStudentModel(student_kind="mean")
    refitter = _refitter(
        tmp_path,
        data_root=data_root,
        fold_config=fold_config,
        model=model,
        seeds=(29,),
        reciprocal_state_reuse=True,
    )
    result = refitter.predict_inner_validation(
        outer_fold=0,
        inner_validation_fold=1,
    )
    assert tuple(prediction.well_id for prediction in result.predictions) == by_fold[1]
    assert result.checkpoints[0].provenance.ordered_training_well_ids == tuple(
        sorted((*by_fold[2], *by_fold[3], *by_fold[4]))
    )
    assert np.isfinite(result.to_frame()[PREDICTION_COLUMN]).all()
    resumed = refitter.predict_inner_validation(
        outer_fold=0,
        inner_validation_fold=1,
        resume=True,
    )
    assert resumed.checkpoints[0].reused
    pd.testing.assert_frame_equal(result.to_frame(), resumed.to_frame())

    common_ids = (by_fold[2][0], by_fold[3][0])
    forward = refitter.predict_wells(
        outer_fold=0,
        inner_validation_fold=1,
        well_ids=common_ids,
    )
    reciprocal = refitter.predict_wells(
        outer_fold=1,
        inner_validation_fold=0,
        well_ids=common_ids,
    )
    assert reciprocal.checkpoints[0].reciprocal_source_checkpoint_path
    reciprocal_validation = refitter.predict_wells(
        outer_fold=1,
        inner_validation_fold=0,
        well_ids=(by_fold[0][0],),
        allowed_scope="inner_validation",
    )

    independent_refitter = NestedHeelIncumbentRefitter(
        data_root=data_root,
        fold_config=fold_config,
        cache_root=tmp_path / "cache",
        checkpoint_root=tmp_path / "independent-checkpoints",
        seeds=(29,),
        workers=1,
        model=HeelRegisteredAnchorStudentModel(student_kind="mean"),
        reciprocal_state_reuse=False,
    )
    independent = independent_refitter.predict_wells(
        outer_fold=1,
        inner_validation_fold=0,
        well_ids=common_ids,
    )
    for aliased, fresh in zip(
        reciprocal.predictions,
        independent.predictions,
        strict=True,
    ):
        np.testing.assert_array_equal(aliased.row_index, fresh.row_index)
        np.testing.assert_array_equal(aliased.prediction, fresh.prediction)
    for original, aliased in zip(
        forward.predictions,
        reciprocal.predictions,
        strict=True,
    ):
        np.testing.assert_array_equal(original.prediction, aliased.prediction)
    independent_validation = independent_refitter.predict_wells(
        outer_fold=1,
        inner_validation_fold=0,
        well_ids=(by_fold[0][0],),
        allowed_scope="inner_validation",
    )
    np.testing.assert_array_equal(
        reciprocal_validation.predictions[0].prediction,
        independent_validation.predictions[0].prediction,
    )
