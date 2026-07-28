"""Serializable contracts for inference-safe external residual families.

The contracts in this module intentionally keep raw, fold-independent sequence
preparation separate from anything learned.  Fold-local normalizers, encoders,
and heads belong in a family state returned by ``fit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


PROFILE_NAMES = ("quick", "full")
PREDICTION_STATUS_OK = "ok"
PREDICTION_STATUS_IDENTITY = "identity_fallback"


def _readonly_vector(
    values: Any,
    *,
    name: str,
    dtype: np.dtype[Any] | type[np.generic] = np.float64,
    finite: bool = True,
) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _readonly_matrix(values: Any, *, name: str) -> np.ndarray:
    result = np.array(values, dtype=np.float32, copy=True)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FixedGridSequenceWell:
    """One inference-only horizontal/typewell pair on deterministic fixed grids."""

    well_id: str
    row_index: np.ndarray
    known_count: int
    anchor_tvt: float
    horizontal: np.ndarray
    typewell: np.ndarray
    output_coordinate: np.ndarray
    horizontal_feature_names: tuple[str, ...]
    typewell_feature_names: tuple[str, ...]
    source_horizontal_rows: int
    source_typewell_rows: int
    schema_version: str

    def __post_init__(self) -> None:
        well_id = str(self.well_id)
        if not well_id:
            raise ValueError("well_id must not be empty")
        row_index = _readonly_vector(self.row_index, name="row_index", dtype=np.int64)
        if np.any(row_index < 0) or np.any(np.diff(row_index) != 1):
            raise ValueError("row_index must be non-negative and contiguous")
        known_count = int(self.known_count)
        if known_count < 1 or int(row_index[0]) != known_count:
            raise ValueError("known_count must equal the first prediction row")
        anchor = float(self.anchor_tvt)
        if not math.isfinite(anchor):
            raise ValueError("anchor_tvt must be finite")

        horizontal = _readonly_matrix(self.horizontal, name="horizontal")
        typewell = _readonly_matrix(self.typewell, name="typewell")
        horizontal_names = tuple(map(str, self.horizontal_feature_names))
        typewell_names = tuple(map(str, self.typewell_feature_names))
        if (
            not horizontal_names
            or len(set(horizontal_names)) != len(horizontal_names)
            or horizontal.shape[1] != len(horizontal_names)
        ):
            raise ValueError("horizontal feature schema does not match its matrix")
        if (
            not typewell_names
            or len(set(typewell_names)) != len(typewell_names)
            or typewell.shape[1] != len(typewell_names)
        ):
            raise ValueError("typewell feature schema does not match its matrix")

        output_coordinate = _readonly_vector(
            self.output_coordinate, name="output_coordinate"
        )
        if output_coordinate.shape != row_index.shape:
            raise ValueError("output_coordinate must align with prediction rows")
        tolerance = 32.0 * np.finfo(np.float64).eps
        if (
            output_coordinate[0] < -tolerance
            or output_coordinate[-1] > 1.0 + tolerance
            or np.any(np.diff(output_coordinate) < 0.0)
        ):
            raise ValueError("output_coordinate must be monotone and lie within [0, 1]")

        source_horizontal_rows = int(self.source_horizontal_rows)
        source_typewell_rows = int(self.source_typewell_rows)
        if (
            source_horizontal_rows <= known_count
            or source_typewell_rows < 2
            or source_horizontal_rows != known_count + len(row_index)
        ):
            raise ValueError("source row counts are inconsistent")
        schema_version = str(self.schema_version)
        if not schema_version:
            raise ValueError("schema_version must not be empty")

        object.__setattr__(self, "well_id", well_id)
        object.__setattr__(self, "row_index", row_index)
        object.__setattr__(self, "known_count", known_count)
        object.__setattr__(self, "anchor_tvt", anchor)
        object.__setattr__(self, "horizontal", horizontal)
        object.__setattr__(self, "typewell", typewell)
        object.__setattr__(self, "output_coordinate", output_coordinate)
        object.__setattr__(self, "horizontal_feature_names", horizontal_names)
        object.__setattr__(self, "typewell_feature_names", typewell_names)
        object.__setattr__(self, "source_horizontal_rows", source_horizontal_rows)
        object.__setattr__(self, "source_typewell_rows", source_typewell_rows)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def prediction_rows(self) -> int:
        return len(self.row_index)


@dataclass(frozen=True)
class SequenceModelInput:
    """Fold-specific channel-first model input plus control-routing metadata."""

    sequence: FixedGridSequenceWell
    incumbent_prediction: np.ndarray
    x: np.ndarray
    grid_coordinate: np.ndarray
    channel_names: tuple[str, ...]
    control_id: str = "none"
    paired_typewell_id: str | None = None
    cross_series_enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, FixedGridSequenceWell):
            raise TypeError("sequence must be a FixedGridSequenceWell")
        incumbent = _readonly_vector(
            self.incumbent_prediction, name="incumbent_prediction"
        )
        if incumbent.shape != self.sequence.row_index.shape:
            raise ValueError("incumbent_prediction must align with prediction rows")
        x = _readonly_matrix(self.x, name="x")
        grid = _readonly_vector(self.grid_coordinate, name="grid_coordinate")
        names = tuple(map(str, self.channel_names))
        if (
            x.shape != (len(names), len(grid))
            or not names
            or len(set(names)) != len(names)
        ):
            raise ValueError("x must have shape [len(channel_names), len(grid)]")
        tolerance = 32.0 * np.finfo(np.float64).eps
        if (
            grid[0] < -tolerance
            or grid[-1] > 1.0 + tolerance
            or np.any(np.diff(grid) <= 0.0)
        ):
            raise ValueError(
                "grid_coordinate must be strictly increasing within [0, 1]"
            )
        control_id = str(self.control_id)
        if not control_id:
            raise ValueError("control_id must not be empty")
        paired = (
            self.sequence.well_id
            if self.paired_typewell_id is None
            else str(self.paired_typewell_id)
        )
        if not paired:
            raise ValueError("paired_typewell_id must not be empty")
        if not isinstance(self.cross_series_enabled, bool):
            raise TypeError("cross_series_enabled must be bool")
        if control_id == "none" and (
            paired != self.sequence.well_id or not self.cross_series_enabled
        ):
            raise ValueError(
                "uncontrolled input must use its own typewell and cross-series path"
            )
        object.__setattr__(self, "incumbent_prediction", incumbent)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "grid_coordinate", grid)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "control_id", control_id)
        object.__setattr__(self, "paired_typewell_id", paired)

    @property
    def well_id(self) -> str:
        return self.sequence.well_id

    @property
    def row_index(self) -> np.ndarray:
        return self.sequence.row_index

    @property
    def row_coordinate(self) -> np.ndarray:
        return self.sequence.output_coordinate

    def channel(self, name: str) -> np.ndarray:
        try:
            index = self.channel_names.index(str(name))
        except ValueError as error:
            raise KeyError(f"unknown model-input channel {name!r}") from error
        return self.x[index]


def uncontrolled_input(inference: SequenceModelInput) -> SequenceModelInput:
    """Return an equivalent own-typewell input with normal control metadata."""

    if not isinstance(inference, SequenceModelInput):
        raise TypeError("inference must be a SequenceModelInput")
    if (
        inference.control_id != "none"
        or inference.paired_typewell_id != inference.sequence.well_id
        or not inference.cross_series_enabled
    ):
        raise ValueError("control transforms cannot be erased by relabeling metadata")
    return SequenceModelInput(
        sequence=inference.sequence,
        incumbent_prediction=inference.incumbent_prediction,
        x=inference.x,
        grid_coordinate=inference.grid_coordinate,
        channel_names=inference.channel_names,
        control_id="none",
        paired_typewell_id=inference.sequence.well_id,
        cross_series_enabled=True,
    )


@dataclass(frozen=True)
class ResidualTrainingWell:
    """One fold-local training example and its incumbent residual target."""

    inference: SequenceModelInput
    incumbent_prediction: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.inference, SequenceModelInput):
            raise TypeError("inference must be a SequenceModelInput")
        incumbent = _readonly_vector(
            self.incumbent_prediction, name="incumbent_prediction"
        )
        target = _readonly_vector(self.target, name="target")
        expected = self.inference.row_index.shape
        if incumbent.shape != expected or target.shape != expected:
            raise ValueError(
                "incumbent_prediction and target must align with prediction rows"
            )
        if not np.array_equal(incumbent, self.inference.incumbent_prediction):
            raise ValueError(
                "ResidualTrainingWell incumbent differs from SequenceModelInput"
            )
        object.__setattr__(self, "incumbent_prediction", incumbent)
        object.__setattr__(self, "target", target)

    @property
    def residual_target(self) -> np.ndarray:
        result = np.asarray(self.target - self.incumbent_prediction, dtype=np.float64)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class FamilyRunContext:
    """Serializable execution identity for outer and optional inner fits."""

    family_id: str
    variant_id: str
    profile: str
    outer_fold: int
    inner_fold: int | None
    seed: int
    device: str
    workers: int
    checkpoint_dir: Path
    resume: bool = False

    def __post_init__(self) -> None:
        family_id = str(self.family_id)
        variant_id = str(self.variant_id)
        if not family_id or not variant_id:
            raise ValueError("family_id and variant_id must not be empty")
        profile = str(self.profile)
        if profile not in PROFILE_NAMES:
            raise ValueError(f"profile must be one of {PROFILE_NAMES}")
        outer = int(self.outer_fold)
        if outer not in range(5):
            raise ValueError("outer_fold must be in 0..4")
        inner = None if self.inner_fold is None else int(self.inner_fold)
        if inner is not None and (inner not in range(5) or inner == outer):
            raise ValueError("inner_fold must be another fold in 0..4")
        seed = int(self.seed)
        device = str(self.device)
        workers = int(self.workers)
        if not device:
            raise ValueError("device must not be empty")
        if workers < 1:
            raise ValueError("workers must be positive")
        checkpoint = Path(self.checkpoint_dir)
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be bool")
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "variant_id", variant_id)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "outer_fold", outer)
        object.__setattr__(self, "inner_fold", inner)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "checkpoint_dir", checkpoint)


@runtime_checkable
class ExternalResidualFamily(Protocol):
    """Shared family surface; states must be serializable or checkpoint-backed."""

    family_id: str
    variant_id: str

    def fit(
        self,
        training_wells: Sequence[ResidualTrainingWell],
        *,
        context: FamilyRunContext,
    ) -> Any: ...

    def predict_components(
        self,
        state: Any,
        inference: SequenceModelInput,
        incumbent_prediction: np.ndarray,
    ) -> Any: ...

    def manifest(self) -> Mapping[str, Any]: ...


__all__ = [
    "ExternalResidualFamily",
    "FamilyRunContext",
    "FixedGridSequenceWell",
    "PREDICTION_STATUS_IDENTITY",
    "PREDICTION_STATUS_OK",
    "PROFILE_NAMES",
    "ResidualTrainingWell",
    "SequenceModelInput",
    "uncontrolled_input",
]
