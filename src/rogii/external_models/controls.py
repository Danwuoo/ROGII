"""Deterministic, target-free negative controls for paired sequence families."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np

from .contracts import (
    FixedGridSequenceWell,
    SequenceModelInput,
    uncontrolled_input,
)
from .data import build_sequence_model_input


CONTROL_NONE = "none"
CONTROL_WRONG_TYPEWELL = "wrong_typewell"
CONTROL_PAIR_SHUFFLE = "pair_shuffle"
CONTROL_GR_ORDER_SHUFFLE = "gr_order_shuffle"
CONTROL_PHASE_DESTROY = "phase_destroy"
CONTROL_NO_CROSS = "no_cross_interaction"
CONTROL_IDS = (
    CONTROL_NONE,
    CONTROL_WRONG_TYPEWELL,
    CONTROL_PAIR_SHUFFLE,
    CONTROL_GR_ORDER_SHUFFLE,
    CONTROL_PHASE_DESTROY,
    CONTROL_NO_CROSS,
)


@dataclass(frozen=True)
class ControlSpec:
    control_id: str
    seed: int

    def __post_init__(self) -> None:
        control = str(self.control_id)
        if control not in CONTROL_IDS:
            raise ValueError(f"control_id must be one of {CONTROL_IDS}")
        object.__setattr__(self, "control_id", control)
        object.__setattr__(self, "seed", int(self.seed))


def _stable_uint64(seed: int, *parts: str) -> int:
    payload = "\0".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _validated_ids(well_ids: Iterable[str]) -> tuple[str, ...]:
    values = tuple(sorted(map(str, well_ids)))
    if len(values) < 2 or len(set(values)) != len(values) or any(not x for x in values):
        raise ValueError("pair controls require at least two unique non-empty well IDs")
    return values


def wrong_typewell_mapping(well_ids: Iterable[str], *, seed: int = 0) -> dict[str, str]:
    """Return a deterministic no-self rotation for the explicit wrong-well control."""

    values = _validated_ids(well_ids)
    offset = 1 + _stable_uint64(seed, CONTROL_WRONG_TYPEWELL) % (len(values) - 1)
    return {
        well_id: values[(index + offset) % len(values)]
        for index, well_id in enumerate(values)
    }


def pair_shuffle_mapping(well_ids: Iterable[str], *, seed: int) -> dict[str, str]:
    """Return a seeded deterministic derangement for pairing-shuffle controls."""

    values = _validated_ids(well_ids)
    ordered = tuple(
        sorted(
            values,
            key=lambda value: (
                _stable_uint64(seed, CONTROL_PAIR_SHUFFLE, value),
                value,
            ),
        )
    )
    offset = 1 + _stable_uint64(seed, "pair-shuffle-offset") % (len(values) - 1)
    mapping = {
        ordered[index]: ordered[(index + offset) % len(ordered)]
        for index in range(len(ordered))
    }
    if any(source == target for source, target in mapping.items()):
        raise RuntimeError("pair shuffle unexpectedly produced a self pairing")
    return mapping


def _replace_sequence(
    source: FixedGridSequenceWell,
    *,
    horizontal: np.ndarray | None = None,
    typewell: np.ndarray | None = None,
    typewell_source: FixedGridSequenceWell | None = None,
) -> FixedGridSequenceWell:
    paired = source if typewell_source is None else typewell_source
    return FixedGridSequenceWell(
        well_id=source.well_id,
        row_index=source.row_index,
        known_count=source.known_count,
        anchor_tvt=source.anchor_tvt,
        horizontal=source.horizontal if horizontal is None else horizontal,
        typewell=source.typewell if typewell is None else typewell,
        output_coordinate=source.output_coordinate,
        horizontal_feature_names=source.horizontal_feature_names,
        typewell_feature_names=paired.typewell_feature_names,
        source_horizontal_rows=source.source_horizontal_rows,
        source_typewell_rows=paired.source_typewell_rows,
        schema_version=source.schema_version,
    )


def _permutation(length: int, *, seed: int, well_id: str, salt: str) -> np.ndarray:
    random = np.random.default_rng(_stable_uint64(seed, well_id, salt))
    return random.permutation(length)


def _model_channel(input_well: SequenceModelInput, name: str) -> int:
    try:
        return input_well.channel_names.index(name)
    except ValueError as error:
        raise ValueError(f"model input is missing required channel {name!r}") from error


def _replace_model_input(
    input_well: SequenceModelInput,
    *,
    x: np.ndarray | None = None,
    sequence: FixedGridSequenceWell | None = None,
    control_id: str,
    paired_typewell_id: str | None = None,
    cross_series_enabled: bool = True,
) -> SequenceModelInput:
    return SequenceModelInput(
        sequence=input_well.sequence if sequence is None else sequence,
        incumbent_prediction=input_well.incumbent_prediction,
        x=input_well.x if x is None else x,
        grid_coordinate=input_well.grid_coordinate,
        channel_names=input_well.channel_names,
        control_id=control_id,
        paired_typewell_id=(
            input_well.paired_typewell_id
            if paired_typewell_id is None
            else paired_typewell_id
        ),
        cross_series_enabled=cross_series_enabled,
    )


def gr_order_shuffle(
    input_well: SequenceModelInput, *, seed: int
) -> SequenceModelInput:
    """Shuffle the two final GR streams without moving geometry or base channels."""

    if not isinstance(input_well, SequenceModelInput):
        raise TypeError("input_well must be a SequenceModelInput")
    x = np.array(input_well.x, copy=True)
    h_columns = [
        _model_channel(input_well, "horizontal_gr_calibrated"),
        _model_channel(input_well, "horizontal_gr_observed"),
    ]
    t_columns = [
        _model_channel(input_well, "typewell_gr_at_incumbent"),
        _model_channel(input_well, "typewell_support"),
    ]
    h_order = _permutation(
        x.shape[1],
        seed=seed,
        well_id=input_well.well_id,
        salt="horizontal-order",
    )
    t_order = _permutation(
        x.shape[1],
        seed=seed,
        well_id=input_well.well_id,
        salt="typewell-order",
    )
    x[h_columns] = x[h_columns][:, h_order]
    x[t_columns] = x[t_columns][:, t_order]
    cross = _model_channel(input_well, "gr_cross_difference")
    x[cross] = x[h_columns[0]] - x[t_columns[0]]
    return _replace_model_input(
        input_well,
        x=x,
        control_id=CONTROL_GR_ORDER_SHUFFLE,
        paired_typewell_id=input_well.paired_typewell_id,
        cross_series_enabled=True,
    )


def _destroy_phase(values: np.ndarray, *, random: np.random.Generator) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    center = float(np.mean(source))
    spectrum = np.fft.rfft(source - center)
    if len(spectrum) > 1:
        phase = random.uniform(-np.pi, np.pi, size=len(spectrum))
        phase[0] = 0.0
        if len(source) % 2 == 0:
            phase[-1] = 0.0
        spectrum = np.abs(spectrum) * np.exp(1j * phase)
        spectrum[0] = 0.0
        if len(source) % 2 == 0:
            spectrum[-1] = complex(float(np.real(spectrum[-1])), 0.0)
    transformed = np.fft.irfft(spectrum, n=len(source)) + center
    if not np.isfinite(transformed).all():
        raise ValueError("phase destruction produced non-finite values")
    return transformed


def phase_destroy(input_well: SequenceModelInput, *, seed: int) -> SequenceModelInput:
    """Independently randomize horizontal/typewell GR phase, preserving spectra."""

    if not isinstance(input_well, SequenceModelInput):
        raise TypeError("input_well must be a SequenceModelInput")
    x = np.array(input_well.x, copy=True)
    h_gr = _model_channel(input_well, "horizontal_gr_calibrated")
    t_gr = _model_channel(input_well, "typewell_gr_at_incumbent")
    x[h_gr] = _destroy_phase(
        x[h_gr],
        random=np.random.default_rng(
            _stable_uint64(seed, input_well.well_id, "horizontal-phase")
        ),
    )
    x[t_gr] = _destroy_phase(
        x[t_gr],
        random=np.random.default_rng(
            _stable_uint64(seed, input_well.well_id, "typewell-phase")
        ),
    )
    cross = _model_channel(input_well, "gr_cross_difference")
    x[cross] = x[h_gr] - x[t_gr]
    return _replace_model_input(
        input_well,
        x=x,
        control_id=CONTROL_PHASE_DESTROY,
        paired_typewell_id=input_well.paired_typewell_id,
        cross_series_enabled=True,
    )


def no_cross_interaction(input_well: SequenceModelInput) -> SequenceModelInput:
    """Zero explicit cross features and mark cross-attention as disabled."""

    if not isinstance(input_well, SequenceModelInput):
        raise TypeError("input_well must be a SequenceModelInput")
    x = np.array(input_well.x, copy=True)
    x[_model_channel(input_well, "gr_cross_difference")] = 0.0
    return _replace_model_input(
        input_well,
        x=x,
        control_id=CONTROL_NO_CROSS,
        paired_typewell_id=input_well.paired_typewell_id,
        cross_series_enabled=False,
    )


def paired_control_inputs(
    inputs: Sequence[SequenceModelInput],
    *,
    control_id: str,
    seed: int,
) -> dict[str, SequenceModelInput]:
    """Apply normal, wrong-typewell, or pairing-shuffle routing to a cohort."""

    if control_id not in {
        CONTROL_NONE,
        CONTROL_WRONG_TYPEWELL,
        CONTROL_PAIR_SHUFFLE,
    }:
        raise ValueError("paired_control_inputs requires a pairing control")
    if any(not isinstance(item, SequenceModelInput) for item in inputs):
        raise TypeError("inputs must contain only SequenceModelInput instances")
    by_id = {item.well_id: item for item in inputs}
    if len(by_id) != len(inputs):
        raise ValueError("model-input cohort contains duplicate well IDs")
    if control_id == CONTROL_NONE:
        return {
            well_id: uncontrolled_input(input_well)
            for well_id, input_well in by_id.items()
        }
    mapping = (
        wrong_typewell_mapping(by_id, seed=seed)
        if control_id == CONTROL_WRONG_TYPEWELL
        else pair_shuffle_mapping(by_id, seed=seed)
    )
    result: dict[str, SequenceModelInput] = {}
    for well_id, input_well in by_id.items():
        paired_id = mapping[well_id]
        paired = by_id[paired_id]
        replaced = _replace_sequence(
            input_well.sequence,
            typewell=paired.sequence.typewell,
            typewell_source=paired.sequence,
        )
        rebuilt = build_sequence_model_input(
            replaced,
            input_well.incumbent_prediction,
            grid_points=len(input_well.grid_coordinate),
        )
        result[well_id] = _replace_model_input(
            rebuilt,
            sequence=replaced,
            control_id=control_id,
            paired_typewell_id=paired_id,
            cross_series_enabled=True,
        )
    return result


def apply_individual_control(
    input_well: SequenceModelInput,
    *,
    spec: ControlSpec,
) -> SequenceModelInput:
    """Apply a non-pairing control to one well."""

    if spec.control_id == CONTROL_NONE:
        return uncontrolled_input(input_well)
    if spec.control_id == CONTROL_GR_ORDER_SHUFFLE:
        return gr_order_shuffle(input_well, seed=spec.seed)
    if spec.control_id == CONTROL_PHASE_DESTROY:
        return phase_destroy(input_well, seed=spec.seed)
    if spec.control_id == CONTROL_NO_CROSS:
        return no_cross_interaction(input_well)
    raise ValueError(
        "wrong-typewell and pair-shuffle controls require paired_control_inputs"
    )


__all__ = [
    "CONTROL_GR_ORDER_SHUFFLE",
    "CONTROL_IDS",
    "CONTROL_NONE",
    "CONTROL_NO_CROSS",
    "CONTROL_PAIR_SHUFFLE",
    "CONTROL_PHASE_DESTROY",
    "CONTROL_WRONG_TYPEWELL",
    "ControlSpec",
    "apply_individual_control",
    "gr_order_shuffle",
    "no_cross_interaction",
    "pair_shuffle_mapping",
    "paired_control_inputs",
    "phase_destroy",
    "wrong_typewell_mapping",
]
