from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from rogii.data import WellFiles
from rogii.external_models import (
    CONTROL_GR_ORDER_SHUFFLE,
    CONTROL_NO_CROSS,
    CONTROL_PAIR_SHUFFLE,
    CONTROL_PHASE_DESTROY,
    CONTROL_WRONG_TYPEWELL,
    ExternalResidualFamily,
    FamilyRunContext,
    FixedGridConfig,
    MODEL_INPUT_CHANNEL_NAMES,
    PredictionFailure,
    ResidualComponents,
    ResidualTrainingWell,
    SequenceModelInput,
    apply_individual_control,
    build_fixed_grid_sequence,
    build_sequence_model_input,
    compose_residual_components,
    decompose_residual_target,
    fixed_grid_manifest,
    guarded_residual_prediction,
    identity_residual_components,
    paired_control_inputs,
    prepare_fixed_grid_well,
    sequence_cache_path,
    uncontrolled_input,
    validate_residual_components,
)
from rogii.external_models.controls import ControlSpec
from rogii.external_models.data import (
    HORIZONTAL_FEATURE_NAMES,
    TYPEWELL_FEATURE_NAMES,
)


def _write_well(
    data_root: Path,
    *,
    well_id: str,
    unsafe_offset: float = 0.0,
    signal_phase: float = 0.0,
    split: str = "train",
) -> WellFiles:
    directory = data_root / split
    directory.mkdir(parents=True, exist_ok=True)
    rows = 44
    known = 15
    md = np.cumsum(np.linspace(8.0, 12.0, rows))
    truth = 10_000.0 + 0.04 * md + unsafe_offset
    tvt_input = truth.copy()
    tvt_input[known:] = np.nan
    gr = 60.0 + 8.0 * np.sin(md / 31.0 + signal_phase)
    gr[[3, 22]] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 1_000.0 + 0.8 * md,
            "Y": 2_000.0 - 0.2 * md,
            "Z": -8_000.0 + 0.1 * md,
            "GR": gr,
            "TVT_input": tvt_input,
            # Every column below is forbidden at model inference.
            "TVT": truth + unsafe_offset,
            "ANCC": unsafe_offset + np.arange(rows),
            "Geology": f"unsafe-{unsafe_offset}",
            "PNG": f"unsafe-image-{unsafe_offset}",
        }
    )
    horizontal_path = directory / f"{well_id}__horizontal_well.csv"
    horizontal.to_csv(horizontal_path, index=False)

    typewell_tvt = np.linspace(9_950.0, 10_150.0, 79)
    typewell_gr = 55.0 + 9.0 * np.sin(typewell_tvt / 23.0 + signal_phase)
    typewell_gr[11] = np.nan
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": typewell_gr,
            "Geology": f"unsafe-{unsafe_offset}",
        }
    )
    typewell_path = directory / f"{well_id}__typewell.csv"
    typewell.to_csv(typewell_path, index=False)
    return WellFiles(well_id, horizontal_path, typewell_path)


def _prepared_cohort(tmp_path: Path) -> list[Any]:
    data = tmp_path / "competition"
    config = FixedGridConfig(horizontal_points=48, typewell_points=40)
    result = []
    for index, well_id in enumerate(("0000000a", "0000000b", "0000000c")):
        well = _write_well(
            data,
            well_id=well_id,
            signal_phase=0.7 * index,
        )
        result.append(
            prepare_fixed_grid_well(
                well,
                split="train",
                cache_root=tmp_path / "cache",
                config=config,
            )
        )
    return result


def _model_input(
    sequence: Any,
    *,
    grid_points: int = 48,
) -> SequenceModelInput:
    incumbent = sequence.anchor_tvt + np.linspace(
        1.0,
        80.0,
        sequence.prediction_rows,
    )
    return build_sequence_model_input(
        sequence,
        incumbent,
        grid_points=grid_points,
    )


def test_fixed_grid_prepare_uses_only_inference_columns_and_is_finite(
    tmp_path: Path,
) -> None:
    first = _write_well(
        tmp_path / "first" / "competition",
        well_id="00000001",
        unsafe_offset=0.0,
    )
    second = _write_well(
        tmp_path / "second" / "competition",
        well_id="00000001",
        unsafe_offset=1e6,
    )
    # Make the inference-safe values identical while leaving forbidden values different.
    first_horizontal = pd.read_csv(first.horizontal_path)
    second_horizontal = pd.read_csv(second.horizontal_path)
    safe_horizontal = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    second_horizontal.loc[:, safe_horizontal] = first_horizontal[safe_horizontal]
    second_horizontal.to_csv(second.horizontal_path, index=False)
    first_typewell = pd.read_csv(first.typewell_path)
    second_typewell = pd.read_csv(second.typewell_path)
    second_typewell.loc[:, ["TVT", "GR"]] = first_typewell[["TVT", "GR"]]
    second_typewell.to_csv(second.typewell_path, index=False)

    config = FixedGridConfig(horizontal_points=32, typewell_points=24)
    one = prepare_fixed_grid_well(first, split="train", config=config)
    two = prepare_fixed_grid_well(second, split="train", config=config)
    assert one.row_index.tolist() == list(range(15, 44))
    assert one.horizontal.shape == (32, len(HORIZONTAL_FEATURE_NAMES))
    assert one.typewell.shape == (24, len(TYPEWELL_FEATURE_NAMES))
    np.testing.assert_array_equal(one.horizontal, two.horizontal)
    np.testing.assert_array_equal(one.typewell, two.typewell)
    np.testing.assert_array_equal(one.output_coordinate, two.output_coordinate)
    assert np.isfinite(one.horizontal).all()
    assert np.isfinite(one.typewell).all()
    assert not one.horizontal.flags.writeable
    assert not one.typewell.flags.writeable

    manifest = fixed_grid_manifest(config)
    assert manifest["target_columns"] == []
    assert manifest["formation_columns"] == []
    assert manifest["geology_columns"] == []
    assert manifest["image_inputs"] is False
    assert manifest["global_learned_statistics"] is False
    assert manifest["model_input_cache_policy"] == "never_cached"
    assert manifest["fold_specific_model_input_channels"] == list(
        MODEL_INPUT_CHANNEL_NAMES
    )

    direct = build_fixed_grid_sequence(
        pd.read_csv(first.horizontal_path),
        pd.read_csv(first.typewell_path),
        well_id=first.well_id,
        config=config,
    )
    np.testing.assert_array_equal(direct.horizontal, one.horizontal)
    np.testing.assert_array_equal(direct.typewell, one.typewell)


def test_fold_specific_model_input_has_calibrated_pair_and_safe_derivatives(
    tmp_path: Path,
) -> None:
    sequence = _prepared_cohort(tmp_path)[0]
    horizontal_before = np.array(sequence.horizontal, copy=True)
    typewell_before = np.array(sequence.typewell, copy=True)
    inference = _model_input(sequence, grid_points=64)

    assert inference.x.shape == (len(MODEL_INPUT_CHANNEL_NAMES), 64)
    assert inference.channel_names == MODEL_INPUT_CHANNEL_NAMES
    assert inference.row_index.shape == (sequence.prediction_rows,)
    assert inference.row_coordinate.shape == (sequence.prediction_rows,)
    np.testing.assert_allclose(
        inference.grid_coordinate,
        np.linspace(0.0, 1.0, 64),
    )
    np.testing.assert_allclose(
        inference.channel("gr_cross_difference"),
        inference.channel("horizontal_gr_calibrated")
        - inference.channel("typewell_gr_at_incumbent"),
        rtol=0.0,
        atol=1e-6,
    )
    assert np.isfinite(inference.channel("incumbent_d1_md")).all()
    assert np.isfinite(inference.channel("incumbent_d2_md2")).all()
    assert np.isfinite(inference.channel("trajectory_curvature")).all()
    assert set(np.unique(inference.channel("typewell_support"))) <= {0.0, 1.0}
    assert not inference.x.flags.writeable

    # Fold-specific incumbent channels are built after the immutable raw cache.
    shifted = build_sequence_model_input(
        sequence,
        inference.incumbent_prediction + 5.0,
        grid_points=64,
    )
    assert not np.array_equal(
        inference.channel("incumbent_relative_anchor"),
        shifted.channel("incumbent_relative_anchor"),
    )
    np.testing.assert_array_equal(sequence.horizontal, horizontal_before)
    np.testing.assert_array_equal(sequence.typewell, typewell_before)


def test_fixed_grid_cache_roundtrip_corruption_rebuild_and_data_guard(
    tmp_path: Path,
) -> None:
    data = tmp_path / "competition"
    well = _write_well(data, well_id="00000002")
    config = FixedGridConfig(horizontal_points=40, typewell_points=32)
    cache = tmp_path / "cache"
    first = prepare_fixed_grid_well(
        well, split="train", cache_root=cache, config=config
    )
    second = prepare_fixed_grid_well(
        well, split="train", cache_root=cache, config=config
    )
    np.testing.assert_array_equal(first.horizontal, second.horizontal)
    np.testing.assert_array_equal(first.typewell, second.typewell)

    path = sequence_cache_path(
        cache, split="train", well_id=well.well_id, config=config
    )
    path.write_bytes(b"corrupt-cache")
    rebuilt = prepare_fixed_grid_well(
        well, split="train", cache_root=cache, config=config
    )
    np.testing.assert_array_equal(first.horizontal, rebuilt.horizontal)

    # A syntactically valid cache must still be rejected after either raw
    # source changes.  The rebuilt grid therefore reflects the new GR bytes.
    changed = pd.read_csv(well.horizontal_path)
    changed["GR"] = changed["GR"] + np.linspace(0.0, 25.0, len(changed))
    changed.to_csv(well.horizontal_path, index=False)
    refreshed = prepare_fixed_grid_well(
        well, split="train", cache_root=cache, config=config
    )
    assert not np.array_equal(rebuilt.horizontal, refreshed.horizontal)

    with pytest.raises(ValueError, match="inside competition data"):
        prepare_fixed_grid_well(
            well,
            split="train",
            cache_root=data / "unsafe-cache",
            config=config,
        )


def test_residual_decomposition_is_exact_identifiable_and_serializable() -> None:
    coordinate = np.linspace(0.02, 1.0, 101) ** 1.3
    incumbent = 11_000.0 + 4.0 * coordinate
    nonlinear = 2.0 * np.sin(4.0 * np.pi * coordinate)
    residual = 3.5 - 6.0 * (coordinate - coordinate.mean()) + nonlinear
    decomposition = decompose_residual_target(
        residual,
        coordinate,
        incumbent_prediction=incumbent,
    )
    np.testing.assert_allclose(decomposition.correction, residual, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        decomposition.final, incumbent + residual, rtol=0.0, atol=1e-10
    )
    centered_coordinate = coordinate - coordinate.mean()
    assert abs(float(np.mean(decomposition.shape))) < 1e-12
    assert abs(float(decomposition.shape @ centered_coordinate)) < 1e-10
    assert abs(float(np.mean(decomposition.slope))) < 1e-12
    assert decomposition.eligible
    restored = pickle.loads(
        pickle.dumps(decomposition, protocol=pickle.HIGHEST_PROTOCOL)
    )
    validate_residual_components(restored, incumbent, coordinate)

    composed = compose_residual_components(
        incumbent,
        coordinate,
        common_coefficient=4.0,
        slope_coefficient=-2.0,
        raw_shape=np.cos(5.0 * coordinate) + 9.0 * coordinate + 7.0,
    )
    assert abs(float(np.mean(composed.shape))) < 1e-12
    assert abs(float(composed.shape @ centered_coordinate)) < 1e-10


def test_identity_and_guarded_prediction_are_explicit_and_fail_closed() -> None:
    incumbent = np.linspace(100.0, 110.0, 20)
    coordinate = np.linspace(0.0, 1.0, 20)
    identity = identity_residual_components(
        incumbent, coordinate, reason="checkpoint incomplete"
    )
    assert not identity.eligible
    assert identity.status == "identity_fallback"
    assert identity.reason == "checkpoint incomplete"
    np.testing.assert_array_equal(identity.final, incumbent)
    np.testing.assert_array_equal(identity.correction, np.zeros_like(incumbent))

    def broken() -> ResidualComponents:
        return compose_residual_components(
            incumbent,
            coordinate,
            common_coefficient=np.nan,
            slope_coefficient=0.0,
        )

    with pytest.raises(PredictionFailure, match="failed validation"):
        guarded_residual_prediction(broken, incumbent, coordinate)
    fallback = guarded_residual_prediction(
        broken,
        incumbent,
        coordinate,
        allow_identity_fallback=True,
        failure_label="toy-family",
    )
    assert not fallback.eligible
    assert "toy-family" in str(fallback.reason)

    invalid = ResidualComponents(
        common=np.r_[0.0, np.ones(19)],
        slope=np.zeros(20),
        shape=np.zeros(20),
        correction=np.r_[0.0, np.ones(19)],
        final=incumbent + np.r_[0.0, np.ones(19)],
        status="ok",
        eligible=True,
    )
    with pytest.raises(ValueError, match="common component"):
        validate_residual_components(invalid, incumbent, coordinate)


def test_pair_controls_are_deterministic_deranged_and_target_free(
    tmp_path: Path,
) -> None:
    sequences = _prepared_cohort(tmp_path)
    cohort = [_model_input(sequence) for sequence in sequences]
    wrong_first = paired_control_inputs(
        cohort, control_id=CONTROL_WRONG_TYPEWELL, seed=71
    )
    wrong_second = paired_control_inputs(
        cohort, control_id=CONTROL_WRONG_TYPEWELL, seed=71
    )
    shuffled = paired_control_inputs(cohort, control_id=CONTROL_PAIR_SHUFFLE, seed=81)
    by_id = {well.well_id: well for well in cohort}
    for well_id in sorted(by_id):
        assert wrong_first[well_id].paired_typewell_id != well_id
        assert shuffled[well_id].paired_typewell_id != well_id
        assert (
            wrong_first[well_id].paired_typewell_id
            == wrong_second[well_id].paired_typewell_id
        )
        source_id = str(wrong_first[well_id].paired_typewell_id)
        np.testing.assert_array_equal(
            wrong_first[well_id].sequence.typewell,
            by_id[source_id].sequence.typewell,
        )
        np.testing.assert_array_equal(
            wrong_first[well_id].sequence.horizontal,
            by_id[well_id].sequence.horizontal,
        )
        np.testing.assert_array_equal(
            wrong_first[well_id].incumbent_prediction,
            by_id[well_id].incumbent_prediction,
        )
        np.testing.assert_allclose(
            wrong_first[well_id].channel("gr_cross_difference"),
            wrong_first[well_id].channel("horizontal_gr_calibrated")
            - wrong_first[well_id].channel("typewell_gr_at_incumbent"),
            rtol=0.0,
            atol=1e-6,
        )
        assert wrong_first[well_id].cross_series_enabled


def test_order_phase_and_no_cross_controls_preserve_contract(
    tmp_path: Path,
) -> None:
    sequence = _prepared_cohort(tmp_path)[0]
    inference = _model_input(sequence)
    order_one = apply_individual_control(
        inference,
        spec=ControlSpec(CONTROL_GR_ORDER_SHUFFLE, seed=99),
    )
    order_two = apply_individual_control(
        inference,
        spec=ControlSpec(CONTROL_GR_ORDER_SHUFFLE, seed=99),
    )
    np.testing.assert_array_equal(order_one.x, order_two.x)
    h_gr = MODEL_INPUT_CHANNEL_NAMES.index("horizontal_gr_calibrated")
    h_non_gr = [
        index
        for index, name in enumerate(MODEL_INPUT_CHANNEL_NAMES)
        if name
        not in {
            "horizontal_gr_calibrated",
            "horizontal_gr_observed",
            "typewell_gr_at_incumbent",
            "typewell_support",
            "gr_cross_difference",
        }
    ]
    assert not np.array_equal(
        order_one.x[h_gr],
        inference.x[h_gr],
    )
    np.testing.assert_array_equal(
        order_one.x[h_non_gr],
        inference.x[h_non_gr],
    )
    np.testing.assert_array_equal(order_one.row_index, sequence.row_index)

    phase = apply_individual_control(
        inference,
        spec=ControlSpec(CONTROL_PHASE_DESTROY, seed=101),
    )
    original = inference.x[h_gr].astype(np.float64)
    transformed = phase.x[h_gr].astype(np.float64)
    assert not np.array_equal(original, transformed)
    original_power = np.abs(np.fft.rfft(original - original.mean()))
    transformed_power = np.abs(np.fft.rfft(transformed - transformed.mean()))
    np.testing.assert_allclose(transformed_power, original_power, rtol=2e-5, atol=2e-4)

    no_cross = apply_individual_control(
        inference,
        spec=ControlSpec(CONTROL_NO_CROSS, seed=0),
    )
    assert not no_cross.cross_series_enabled
    assert no_cross.control_id == CONTROL_NO_CROSS
    cross = MODEL_INPUT_CHANNEL_NAMES.index("gr_cross_difference")
    np.testing.assert_array_equal(no_cross.x[cross], np.zeros(no_cross.x.shape[1]))
    non_cross = [index for index in range(no_cross.x.shape[0]) if index != cross]
    np.testing.assert_array_equal(no_cross.x[non_cross], inference.x[non_cross])
    np.testing.assert_array_equal(no_cross.sequence.horizontal, sequence.horizontal)
    np.testing.assert_array_equal(no_cross.sequence.typewell, sequence.typewell)
    with pytest.raises(ValueError, match="cannot be erased"):
        uncontrolled_input(no_cross)


class _ToyFamily:
    family_id = "toy"
    variant_id = "mean"

    def fit(
        self,
        training_wells: Sequence[ResidualTrainingWell],
        *,
        context: FamilyRunContext,
    ) -> float:
        del context
        return float(
            np.mean([np.mean(training.residual_target) for training in training_wells])
        )

    def predict_components(
        self,
        state: float,
        inference: SequenceModelInput,
        incumbent_prediction: np.ndarray,
    ) -> ResidualComponents:
        return compose_residual_components(
            incumbent_prediction,
            inference.sequence.output_coordinate,
            common_coefficient=state,
            slope_coefficient=0.0,
        )

    def manifest(self) -> Mapping[str, Any]:
        return {"family_id": self.family_id, "variant_id": self.variant_id}


def test_family_context_training_contract_and_protocol_are_pickle_safe(
    tmp_path: Path,
) -> None:
    sequence = _prepared_cohort(tmp_path)[0]
    incumbent = np.linspace(100.0, 120.0, sequence.prediction_rows)
    inference = build_sequence_model_input(sequence, incumbent)
    inference = uncontrolled_input(inference)
    target = incumbent + 2.5
    training = ResidualTrainingWell(inference, incumbent, target)
    context = FamilyRunContext(
        family_id="toy",
        variant_id="mean",
        profile="quick",
        outer_fold=0,
        inner_fold=1,
        seed=20260728,
        device="cpu",
        workers=1,
        checkpoint_dir=tmp_path / "checkpoints",
        resume=True,
    )
    restored_context, restored_training = pickle.loads(
        pickle.dumps((context, training), protocol=pickle.HIGHEST_PROTOCOL)
    )
    assert restored_context == context
    np.testing.assert_array_equal(
        restored_training.residual_target, np.full(sequence.prediction_rows, 2.5)
    )
    family = _ToyFamily()
    assert isinstance(family, ExternalResidualFamily)
    state = family.fit([restored_training], context=restored_context)
    prediction = family.predict_components(state, inference, incumbent)
    validate_residual_components(prediction, incumbent, sequence.output_coordinate)
    np.testing.assert_allclose(prediction.final, target)
