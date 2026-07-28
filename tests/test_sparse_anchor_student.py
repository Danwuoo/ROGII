from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.data import WellFiles, enumerate_wells
from rogii.diagnostics.anchor_acquisition_protocol import select_protocol_anchors
from rogii.diagnostics.sparse_anchor_student_experiment import (
    run_sparse_anchor_student_experiment,
)
from rogii.models.base import InferenceWell, RunContext, TrainingWell
from rogii.models.sparse_anchor_student import (
    SparseAnchorInferenceWell,
    SparseAnchorStudentModel,
    fit_student_estimator,
    predict_anchor_delta,
)
from rogii.sparse_anchor import (
    ANCHOR_FRACTIONS,
    AnchorPlacement,
    reconstruct_from_predicted_anchors,
    select_anchor_rows,
)
from rogii.submission import align_predictions_to_sample


def _placement(rows: int = 24, start: int = 16) -> AnchorPlacement:
    prediction_md = np.arange(start, start + rows, dtype=np.float64)
    return select_anchor_rows(
        prediction_md,
        float(start - 1),
        row_indices=np.arange(start, start + rows, dtype=np.int64),
    )


def _write_well(
    root: Path,
    split: str,
    well_id: str,
    offset: float,
    *,
    unsafe_shift: float = 0.0,
) -> WellFiles:
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    rows, known = 40, 16
    md = np.arange(rows, dtype=np.float64)
    truth = 150.0 + offset + 0.15 * md + 0.004 * md**2
    tvt_input = truth.copy()
    tvt_input[known:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 1_000.0 + offset + 0.9 * md,
            "Y": 2_000.0 - 0.3 * md,
            "Z": -8_000.0 + 0.05 * md,
            "GR": 70.0 + np.sin(md / 4.0 + offset / 10.0),
            "TVT_input": tvt_input,
            "TVT": truth + unsafe_shift,
            "Geology": np.where(md < 20, "A", "B"),
            "ANCC": unsafe_shift + md,
        }
    )
    horizontal_path = directory / f"{well_id}__horizontal_well.csv"
    horizontal.to_csv(horizontal_path, index=False)
    typewell_tvt = np.arange(0.0, 401.0, 1.0)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": 70.0 + np.sin(typewell_tvt / 4.0 + offset / 10.0),
            "Geology": "unsafe",
        }
    )
    typewell_path = directory / f"{well_id}__typewell.csv"
    typewell.to_csv(typewell_path, index=False)
    (directory / f"{well_id}.png").write_bytes(
        b"unsafe-image" + str(unsafe_shift).encode("ascii")
    )
    return WellFiles(well_id, horizontal_path, typewell_path)


def _target(well: WellFiles) -> np.ndarray:
    frame = pd.read_csv(well.horizontal_path)
    known = int(frame["TVT_input"].notna().sum())
    return frame["TVT"].iloc[known:].to_numpy(dtype=np.float64)


def test_locked_fractions_and_placement_are_deterministic() -> None:
    assert ANCHOR_FRACTIONS == (
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        0.975,
    )
    first = _placement()
    second = _placement()
    np.testing.assert_array_equal(first.local_indices, second.local_indices)
    assert len(np.unique(first.local_indices)) == 8
    assert np.all(np.diff(first.local_indices) > 0)
    assert np.all(np.diff(first.row_indices) > 0)


def test_production_nominal_rows_equal_protocol_tu_zero() -> None:
    rng = np.random.default_rng(20260727)
    for row_count in (8, 17, 101, 1_003):
        md = np.cumsum(rng.uniform(0.1, 3.0, size=row_count)) + 100.0
        production = select_anchor_rows(md, 99.5)
        diagnostic = select_protocol_anchors(
            production.normalized_positions, ANCHOR_FRACTIONS
        )
        assert diagnostic.valid
        assert diagnostic.selection is not None
        np.testing.assert_array_equal(
            production.local_indices, diagnostic.selection.local_indices
        )


def test_zero_correction_exactly_reproduces_baseline_and_anchor_values() -> None:
    placement = _placement()
    baseline = np.linspace(100.0, 140.0, len(placement.normalized_positions))
    anchors = baseline[placement.local_indices]
    prediction = reconstruct_from_predicted_anchors(baseline, anchors, placement)
    np.testing.assert_array_equal(prediction, baseline)
    np.testing.assert_array_equal(prediction[placement.local_indices], anchors)


def test_reconstruction_hits_anchors_and_carries_last_correction() -> None:
    placement = _placement()
    baseline = np.linspace(100.0, 140.0, len(placement.normalized_positions))
    corrections = np.arange(1.0, 9.0)
    anchors = baseline[placement.local_indices] + corrections
    prediction = reconstruct_from_predicted_anchors(baseline, anchors, placement)
    np.testing.assert_allclose(prediction[placement.local_indices], anchors, rtol=0, atol=0)
    last = int(placement.local_indices[-1])
    np.testing.assert_allclose(prediction[last:] - baseline[last:], corrections[-1])


def test_prepare_ignores_target_formations_geology_and_png(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _write_well(first_root, "train", "00000001", 1.0, unsafe_shift=0.0)
    second = _write_well(second_root, "train", "00000001", 1.0, unsafe_shift=1e6)
    model = SparseAnchorStudentModel(student_kind="ridge")
    one = model.prepare(first, split="train", cache_root=tmp_path / "cache-one")
    two = model.prepare(second, split="train", cache_root=tmp_path / "cache-two")
    np.testing.assert_array_equal(one.row_index, two.row_index)
    np.testing.assert_allclose(one.features, two.features, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(one.p3.features, two.p3.features, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(one.placement.local_indices, two.placement.local_indices)


def test_cached_and_uncached_prepare_match_and_corruption_rebuilds(tmp_path: Path) -> None:
    well = _write_well(tmp_path / "data", "train", "00000002", 2.0)
    model = SparseAnchorStudentModel(student_kind="ridge")
    cache = tmp_path / "cache"
    uncached = model.prepare(well, split="train", cache_root=cache)
    cached = model.prepare(well, split="train", cache_root=cache)
    np.testing.assert_allclose(uncached.features, cached.features, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(uncached.p3.features, cached.p3.features)
    descriptor_path = model.cache_path(well, split="train", cache_root=cache)
    descriptor_path.write_bytes(b"corrupt")
    rebuilt = model.prepare(well, split="train", cache_root=cache)
    np.testing.assert_allclose(uncached.features, rebuilt.features, rtol=0, atol=0, equal_nan=True)
    p3_path = model.p3_model.cache_path(well, split="train", cache_root=cache)
    p3_path.write_bytes(b"corrupt")
    rebuilt_p3 = model.prepare(well, split="train", cache_root=cache)
    np.testing.assert_allclose(uncached.p3.features, rebuilt_p3.p3.features, rtol=0, atol=0, equal_nan=True)
    np.savez(
        p3_path,
        features=np.zeros((1, 1), dtype=np.float32),
        row_index=np.asarray([16], dtype=np.int64),
        anchor=np.asarray([uncached.anchor], dtype=np.float64),
    )
    shape_rebuilt = model.prepare(well, split="train", cache_root=cache)
    np.testing.assert_allclose(
        uncached.p3.features,
        shape_rebuilt.p3.features,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_outer_fit_provenance_pickle_and_target_free_validation(tmp_path: Path) -> None:
    data = tmp_path / "data"
    wells = [_write_well(data, "train", f"{index:08x}", float(index)) for index in range(6)]
    test_well = _write_well(data, "test", "000000ff", 9.0)
    # Test files have no target in the actual competition contract.
    frame = pd.read_csv(test_well.horizontal_path).drop(
        columns=["TVT", "Geology", "ANCC"]
    )
    frame.to_csv(test_well.horizontal_path, index=False)
    model = SparseAnchorStudentModel(
        student_kind="target_pca_ridge", alpha=10.0, target_components=2
    )
    cache = tmp_path / "cache"
    prepared = [model.prepare(well, split="train", cache_root=cache) for well in wells]
    training = [TrainingWell(item, _target(well)) for item, well in zip(prepared[:5], wells[:5])]
    context = RunContext(fold=0, seed=20260727, workers=1, work_dir=tmp_path)
    state = model.fit(training, context=context)
    assert state.training_well_ids == tuple(well.well_id for well in wells[:5])
    assert wells[5].well_id not in state.training_well_ids
    assert state.student_state.target_pca is not None
    assert state.student_state.target_pca.n_samples_ == 5
    assert state.student_state.estimator.named_steps["scale"].n_samples_seen_ == 5
    validation = model.prepare(wells[5], split="train", cache_root=cache)
    prediction = model.predict(state, validation)
    assert prediction.shape == validation.row_index.shape
    assert np.isfinite(prediction).all()
    test_payload = model.prepare(test_well, split="test", cache_root=cache)
    assert np.isfinite(model.predict(state, test_payload)).all()
    restored = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(model.predict(restored, validation), prediction)


def _manual_training_well(index: int, descriptor: np.ndarray) -> TrainingWell:
    placement = _placement(rows=32, start=10)
    anchor = 1_000.0
    p3 = InferenceWell(
        well_id=f"{index:08x}",
        row_index=np.arange(10, 42, dtype=np.int64),
        anchor=anchor,
        features=np.zeros((32, 1), dtype=np.float64),
    )
    payload = SparseAnchorInferenceWell(
        well_id=p3.well_id,
        row_index=p3.row_index,
        anchor=anchor,
        features=descriptor,
        descriptor_names=tuple(f"x_{j}" for j in range(len(descriptor))),
        p3=p3,
        path_pca=p3,
        placement=placement,
    )
    coefficients = np.arange(1.0, 9.0)
    delta = 5.0 + descriptor[0] * coefficients + descriptor[1] * 0.5
    target = np.full(32, anchor, dtype=np.float64)
    target[placement.local_indices] = anchor + delta
    return TrainingWell(payload, target)


def test_synthetic_linear_mapping_beats_zero_student(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    all_wells = [
        _manual_training_well(index, rng.normal(size=4)) for index in range(60)
    ]
    context = RunContext(fold=0, seed=11, workers=1, work_dir=tmp_path)
    state = fit_student_estimator(
        all_wells[:45], kind="ridge", context=context, alpha=1e-6
    )
    predicted = []
    truth = []
    for item in all_wells[45:]:
        predicted.append(predict_anchor_delta(state, item.inference))
        truth.append(
            item.target[item.inference.placement.local_indices] - item.inference.anchor
        )
    predicted_array = np.stack(predicted)
    truth_array = np.stack(truth)
    learned_rmse = float(np.sqrt(np.mean((predicted_array - truth_array) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(truth_array**2)))
    assert learned_rmse < 0.01
    assert learned_rmse < zero_rmse * 0.01


def test_extra_trees_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    wells = [_manual_training_well(index, rng.normal(size=4)) for index in range(30)]
    context = RunContext(fold=1, seed=20260711, workers=1, work_dir=tmp_path)
    first = fit_student_estimator(
        wells, kind="extra_trees", context=context, extra_trees_estimators=24
    )
    second = fit_student_estimator(
        wells, kind="extra_trees", context=context, extra_trees_estimators=24
    )
    np.testing.assert_array_equal(
        predict_anchor_delta(first, wells[0].inference),
        predict_anchor_delta(second, wells[0].inference),
    )


def test_dynamic_test_enumeration_and_sample_alignment(tmp_path: Path) -> None:
    root = tmp_path / "competition"
    for index, well_id in enumerate(("0000000b", "0000000a")):
        _write_well(root, "test", well_id, float(index))
    wells = enumerate_wells(root, "test")
    assert [well.well_id for well in wells] == ["0000000a", "0000000b"]
    ids = [f"{well.well_id}_{row}" for well in wells for row in range(16, 40)]
    sample = pd.DataFrame({"id": list(reversed(ids)), "tvt": 0.0})
    predictions = pd.DataFrame({"id": ids, "tvt": np.arange(len(ids), dtype=float)})
    aligned = align_predictions_to_sample(sample, predictions)
    assert aligned["id"].tolist() == sample["id"].tolist()
    assert np.isfinite(aligned["tvt"]).all()


def test_quick_artifacts_reconstruct_pooled_metric(tmp_path: Path) -> None:
    data = tmp_path / "competition"
    well_ids = [f"{index:08x}" for index in range(10)]
    for index, well_id in enumerate(well_ids):
        _write_well(data, "train", well_id, float(index))
    folds = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            "well_id": well_ids,
            "fold": [index % 5 for index in range(10)],
            "fold_version": "synthetic_5fold_v1",
        }
    ).to_csv(folds, index=False)
    result = run_sparse_anchor_student_experiment(
        profile="quick",
        data_root=data,
        fold_config=folds,
        output_root=tmp_path / "runs",
        run_id="sparse-quick",
        cache_root=tmp_path / "cache",
        workers=1,
    )
    run_root = Path(result["run_root"])
    expected = {
        "summary.json",
        "run_config.json",
        "timings.json",
        "leakage_check.json",
        "oof_predictions.csv.gz",
        "anchor_oof_predictions.csv.gz",
        "per_anchor_metrics.csv",
        "per_well_metrics.csv",
        "per_fold_metrics.csv",
        "paired_bootstrap.json",
        "feature_manifest.json",
        "model_manifest.json",
        "artifact_audit.json",
        "report.md",
    }
    assert expected <= {path.name for path in run_root.iterdir()}
    audit = result["artifact_audit"]
    assert audit["passed"]
    assert audit["duplicate_row_keys"] == 0
    assert audit["summary_rmse_absolute_delta"] <= 1e-12
