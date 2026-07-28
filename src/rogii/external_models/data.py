"""Inference-only fixed-grid sequence preparation and validated caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data import (
    WellFiles,
    load_horizontal_inference,
    load_typewell_inference,
    select_inference_view,
    validate_prefix,
)
from .contracts import FixedGridSequenceWell, SequenceModelInput


SEQUENCE_SCHEMA_VERSION = "external_fixed_grid_sequence_v1"
RAW_SOURCE_PROVENANCE_SCHEMA_VERSION = "external_raw_source_provenance_v1"
HORIZONTAL_FEATURE_NAMES = (
    "MD",
    "X",
    "Y",
    "Z",
    "GR_filled",
    "GR_observed",
    "TVT_input_filled",
    "TVT_input_observed",
    "prediction_zone",
)
TYPEWELL_FEATURE_NAMES = ("TVT", "GR_filled", "GR_observed")
MODEL_INPUT_CHANNEL_NAMES = (
    "horizontal_gr_calibrated",
    "horizontal_gr_observed",
    "typewell_gr_at_incumbent",
    "typewell_support",
    "gr_cross_difference",
    "incumbent_relative_anchor",
    "incumbent_d1_md",
    "incumbent_d2_md2",
    "dx_dmd",
    "dy_dmd",
    "dz_dmd",
    "inclination_rad",
    "trajectory_curvature",
    "suffix_fraction",
    "prefix_calibration_coverage",
    "prefix_calibration_rmse",
)


@dataclass(frozen=True)
class FixedGridConfig:
    horizontal_points: int = 256
    typewell_points: int = 256

    def __post_init__(self) -> None:
        for name, value in (
            ("horizontal_points", self.horizontal_points),
            ("typewell_points", self.typewell_points),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < 8:
                raise ValueError(f"{name} must be an integer >= 8")

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": SEQUENCE_SCHEMA_VERSION,
            **asdict(self),
            "horizontal_features": HORIZONTAL_FEATURE_NAMES,
            "typewell_features": TYPEWELL_FEATURE_NAMES,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


DEFAULT_FIXED_GRID_CONFIG = FixedGridConfig()


def _inference_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash only the numeric inference view, with canonical missing values."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            list(map(str, frame.columns)),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    digest.update(b"\0")
    for column in frame.columns:
        values = frame[column].to_numpy(dtype="<f8")
        observed = np.isfinite(values)
        canonical = np.where(observed, values, 0.0).astype("<f8", copy=False)
        digest.update(str(column).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(observed, dtype=np.uint8).tobytes())
        digest.update(np.ascontiguousarray(canonical).tobytes())
    return digest.hexdigest()


def _load_source_with_provenance(
    path: Path,
    *,
    kind: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and fingerprint one inference-safe CSV view."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"well source file does not exist: {source}")
    before = source.stat()
    if kind == "horizontal":
        frame = load_horizontal_inference(source)
    elif kind == "typewell":
        frame = load_typewell_inference(source)
    else:
        raise ValueError("source provenance kind must be horizontal or typewell")
    after = source.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"well source changed while being fingerprinted: {source}")
    return (
        frame,
        {
            "file_name": source.name,
            "size_bytes": int(after.st_size),
            "mtime_ns": int(after.st_mtime_ns),
            "inference_rows": int(len(frame)),
            "inference_columns": list(map(str, frame.columns)),
            "inference_view_sha256": _inference_frame_sha256(frame),
        },
    )


def _load_well_sources_with_provenance(
    well: WellFiles,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    horizontal, horizontal_provenance = _load_source_with_provenance(
        well.horizontal_path,
        kind="horizontal",
    )
    typewell, typewell_provenance = _load_source_with_provenance(
        well.typewell_path,
        kind="typewell",
    )
    provenance = {
        "schema_version": RAW_SOURCE_PROVENANCE_SCHEMA_VERSION,
        "well_id": str(well.well_id),
        "horizontal": horizontal_provenance,
        "typewell": typewell_provenance,
        "horizontal_inference_columns": [
            "MD",
            "X",
            "Y",
            "Z",
            "GR",
            "TVT_input",
        ],
        "typewell_inference_columns": ["TVT", "GR"],
    }
    return horizontal, typewell, provenance


def raw_well_source_provenance(well: WellFiles) -> dict[str, Any]:
    """Fingerprint both raw inputs used by fixed-grid preparation.

    The fingerprints are provenance only and are never supplied as model
    features.  Exact inference-view hashes keep cache reuse safe even if a
    source is replaced while retaining the same file name or byte count; train
    targets and geology columns are never requested for this fingerprint.
    """

    if not isinstance(well, WellFiles):
        raise TypeError("well must be a WellFiles instance")
    _, _, provenance = _load_well_sources_with_provenance(well)
    return provenance


def raw_source_provenance_digest(provenance: Any) -> str:
    """Hash a raw-source provenance object using canonical JSON."""

    encoded = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sequence_cache_path(
    cache_root: Path,
    *,
    split: str,
    well_id: str,
    config: FixedGridConfig = DEFAULT_FIXED_GRID_CONFIG,
) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    identifier = str(well_id)
    if not identifier or any(character in identifier for character in "\\/:"):
        raise ValueError("well_id is not cache-path safe")
    return (
        Path(cache_root)
        / f"{SEQUENCE_SCHEMA_VERSION}-{config.digest[:16]}"
        / split
        / f"{identifier}.npz"
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _strict_coordinate(values: Any, *, name: str) -> np.ndarray:
    coordinate = np.asarray(values, dtype=np.float64)
    if (
        coordinate.ndim != 1
        or len(coordinate) < 2
        or not np.isfinite(coordinate).all()
        or np.any(np.diff(coordinate) <= 0.0)
    ):
        raise ValueError(f"{name} must be finite and strictly increasing")
    span = float(coordinate[-1] - coordinate[0])
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError(f"{name} has no positive span")
    return (coordinate - coordinate[0]) / span


def _fill(
    values: Any,
    coordinate: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.shape != coordinate.shape:
        raise ValueError(f"{name} must align with its coordinate")
    if np.isinf(vector).any():
        raise ValueError(f"{name} must not contain infinity")
    observed = np.isfinite(vector)
    if not observed.any():
        raise ValueError(f"{name} requires at least one finite observation")
    filled = np.interp(
        coordinate,
        coordinate[observed],
        vector[observed],
    ).astype(np.float64)
    return filled, observed.astype(np.float64)


def _resample(coordinate: np.ndarray, values: np.ndarray, points: int) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    columns = [
        np.interp(grid, coordinate, values[:, column])
        for column in range(values.shape[1])
    ]
    output = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("fixed-grid resampling produced non-finite values")
    return output


def build_fixed_grid_sequence(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    well_id: str,
    config: FixedGridConfig = DEFAULT_FIXED_GRID_CONFIG,
) -> FixedGridSequenceWell:
    """Build a target-free sequence from defensive inference-only frame views."""

    if not isinstance(horizontal, pd.DataFrame):
        raise TypeError("horizontal must be a pandas DataFrame")
    if not isinstance(typewell, pd.DataFrame):
        raise TypeError("typewell must be a pandas DataFrame")
    if not isinstance(config, FixedGridConfig):
        raise TypeError("config must be a FixedGridConfig")
    identifier = str(well_id)
    if not identifier:
        raise ValueError("well_id must not be empty")
    horizontal = select_inference_view(horizontal, kind="horizontal")
    typewell = select_inference_view(typewell, kind="typewell")
    known_count, prediction_count = validate_prefix(horizontal["TVT_input"])

    horizontal_coordinate = _strict_coordinate(horizontal["MD"], name="horizontal MD")
    md = horizontal["MD"].to_numpy(dtype=np.float64)
    x, _ = _fill(horizontal["X"], horizontal_coordinate, name="horizontal X")
    y, _ = _fill(horizontal["Y"], horizontal_coordinate, name="horizontal Y")
    z, _ = _fill(horizontal["Z"], horizontal_coordinate, name="horizontal Z")
    gr, gr_observed = _fill(
        horizontal["GR"], horizontal_coordinate, name="horizontal GR"
    )
    tvt_input = horizontal["TVT_input"].to_numpy(dtype=np.float64)
    anchor = float(tvt_input[known_count - 1])
    if not math.isfinite(anchor):
        raise ValueError("last visible TVT_input must be finite")
    tvt_filled = np.where(np.isfinite(tvt_input), tvt_input, anchor)
    tvt_observed = np.isfinite(tvt_input).astype(np.float64)
    prediction_zone = np.zeros(len(horizontal), dtype=np.float64)
    prediction_zone[known_count:] = 1.0
    horizontal_values = np.column_stack(
        [
            md,
            x,
            y,
            z,
            gr,
            gr_observed,
            tvt_filled,
            tvt_observed,
            prediction_zone,
        ]
    )
    horizontal_grid = _resample(
        horizontal_coordinate, horizontal_values, config.horizontal_points
    )

    typewell_coordinate = _strict_coordinate(typewell["TVT"], name="typewell TVT")
    typewell_tvt = typewell["TVT"].to_numpy(dtype=np.float64)
    typewell_gr, typewell_gr_observed = _fill(
        typewell["GR"], typewell_coordinate, name="typewell GR"
    )
    typewell_values = np.column_stack([typewell_tvt, typewell_gr, typewell_gr_observed])
    typewell_grid = _resample(
        typewell_coordinate, typewell_values, config.typewell_points
    )

    row_index = np.arange(known_count, known_count + prediction_count, dtype=np.int64)
    anchor_md = float(md[known_count - 1])
    suffix_md = md[row_index]
    horizon = float(md[-1] - anchor_md)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("prediction suffix must have positive MD horizon")
    output_coordinate = (suffix_md - anchor_md) / horizon
    return FixedGridSequenceWell(
        well_id=identifier,
        row_index=row_index,
        known_count=known_count,
        anchor_tvt=anchor,
        horizontal=horizontal_grid,
        typewell=typewell_grid,
        output_coordinate=output_coordinate,
        horizontal_feature_names=HORIZONTAL_FEATURE_NAMES,
        typewell_feature_names=TYPEWELL_FEATURE_NAMES,
        source_horizontal_rows=len(horizontal),
        source_typewell_rows=len(typewell),
        schema_version=SEQUENCE_SCHEMA_VERSION,
    )


def _build_sequence_well(
    well: WellFiles,
    *,
    config: FixedGridConfig,
) -> FixedGridSequenceWell:
    horizontal = load_horizontal_inference(well.horizontal_path)
    typewell = load_typewell_inference(well.typewell_path)
    return build_fixed_grid_sequence(
        horizontal,
        typewell,
        well_id=well.well_id,
        config=config,
    )


def _named_column(
    values: np.ndarray,
    names: tuple[str, ...],
    name: str,
) -> np.ndarray:
    try:
        index = names.index(name)
    except ValueError as error:
        raise ValueError(f"prepared sequence is missing feature {name!r}") from error
    return np.asarray(values[:, index], dtype=np.float64)


def _heel_grid_coordinate(sequence: FixedGridSequenceWell) -> float:
    """Approximate the visible/prediction boundary on the cached full-well grid."""

    zone = _named_column(
        sequence.horizontal,
        sequence.horizontal_feature_names,
        "prediction_zone",
    )
    coordinate = np.linspace(0.0, 1.0, len(zone), dtype=np.float64)
    upper_candidates = np.flatnonzero(zone >= 0.5)
    if len(upper_candidates) == 0 or int(upper_candidates[0]) == 0:
        raise ValueError("prepared sequence has no valid heel transition")
    upper = int(upper_candidates[0])
    lower = upper - 1
    left = float(zone[lower])
    right = float(zone[upper])
    if right <= left:
        boundary = float(coordinate[lower])
    else:
        weight = float(np.clip((0.5 - left) / (right - left), 0.0, 1.0))
        boundary = float(
            coordinate[lower] + weight * (coordinate[upper] - coordinate[lower])
        )
    if not math.isfinite(boundary) or not 0.0 < boundary < 1.0:
        raise ValueError("prepared sequence produced an invalid heel coordinate")
    return boundary


def _typewell_support(
    sequence: FixedGridSequenceWell,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    tvt = _named_column(
        sequence.typewell,
        sequence.typewell_feature_names,
        "TVT",
    )
    gr = _named_column(
        sequence.typewell,
        sequence.typewell_feature_names,
        "GR_filled",
    )
    observed = _named_column(
        sequence.typewell,
        sequence.typewell_feature_names,
        "GR_observed",
    )
    if np.any(np.diff(tvt) <= 0.0):
        raise ValueError("prepared typewell TVT must be strictly increasing")
    supported = np.flatnonzero(observed > 0.5)
    if len(supported) < 2:
        supported = np.arange(len(tvt), dtype=np.int64)
    support_min = float(tvt[int(supported[0])])
    support_max = float(tvt[int(supported[-1])])
    return tvt, gr, support_min, support_max


def _prefix_affine_calibration(
    sequence: FixedGridSequenceWell,
) -> tuple[float, float, float, float]:
    """Fit a per-well affine GR map using visible TVT_input only."""

    horizontal_gr = _named_column(
        sequence.horizontal,
        sequence.horizontal_feature_names,
        "GR_filled",
    )
    horizontal_observed = _named_column(
        sequence.horizontal,
        sequence.horizontal_feature_names,
        "GR_observed",
    )
    prefix_tvt = _named_column(
        sequence.horizontal,
        sequence.horizontal_feature_names,
        "TVT_input_filled",
    )
    prefix_observed = _named_column(
        sequence.horizontal,
        sequence.horizontal_feature_names,
        "TVT_input_observed",
    )
    typewell_tvt, typewell_gr, support_min, support_max = _typewell_support(sequence)
    visible = prefix_observed > 0.5
    paired = (
        visible
        & (horizontal_observed > 0.5)
        & (prefix_tvt >= support_min)
        & (prefix_tvt <= support_max)
    )
    pair_count = int(paired.sum())
    visible_count = max(int(visible.sum()), 1)

    if pair_count > 0:
        horizontal_values = horizontal_gr[paired]
        typewell_values = np.interp(
            prefix_tvt[paired],
            typewell_tvt,
            typewell_gr,
        )
    else:
        horizontal_values = np.empty(0, dtype=np.float64)
        typewell_values = np.empty(0, dtype=np.float64)

    if pair_count >= 2:
        horizontal_center = float(np.mean(horizontal_values))
        typewell_center = float(np.mean(typewell_values))
        centered = horizontal_values - horizontal_center
        denominator = float(centered @ centered)
        scale = max(float(np.max(np.abs(horizontal_values))), 1.0)
        threshold = np.finfo(np.float64).eps * scale * scale * pair_count
        slope = (
            0.0
            if denominator <= threshold
            else float(centered @ (typewell_values - typewell_center) / denominator)
        )
        intercept = float(typewell_center - slope * horizontal_center)
    elif pair_count == 1:
        slope = 1.0
        intercept = float(typewell_values[0] - horizontal_values[0])
    else:
        # A zero-overlap well is still a valid inference case.  Keep a finite,
        # explicitly low-coverage location calibration instead of borrowing a
        # cross-well statistic or silently dropping the well.
        visible_gr = horizontal_gr[visible]
        if len(visible_gr) == 0:
            visible_gr = horizontal_gr
        slope = 1.0
        intercept = float(np.median(typewell_gr) - np.median(visible_gr))

    residual = intercept + slope * horizontal_values - typewell_values
    rmse = float(np.sqrt(np.mean(residual * residual))) if pair_count > 0 else 0.0
    coverage = float(pair_count / visible_count)
    if not np.isfinite([intercept, slope, coverage, rmse]).all():
        raise ValueError("prefix affine calibration produced non-finite values")
    return intercept, slope, coverage, rmse


def _gradient(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    edge_order = 2 if len(values) >= 3 else 1
    result = np.gradient(values, coordinate, edge_order=edge_order)
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("sequence derivative produced non-finite values")
    return result


def build_sequence_model_input(
    prepared: FixedGridSequenceWell,
    incumbent_prediction: Any,
    *,
    grid_points: int | None = None,
) -> SequenceModelInput:
    """Create fold-specific ``x[C,L]`` without touching the raw sequence cache.

    ``incumbent_prediction`` is deliberately supplied at call time: callers
    must use the incumbent state appropriate to the current outer/inner fit.
    The function performs no I/O and learns no statistic across wells.
    """

    if not isinstance(prepared, FixedGridSequenceWell):
        raise TypeError("prepared must be a FixedGridSequenceWell")
    incumbent = np.asarray(incumbent_prediction, dtype=np.float64)
    if (
        incumbent.ndim != 1
        or incumbent.shape != prepared.row_index.shape
        or not np.isfinite(incumbent).all()
    ):
        raise ValueError(
            "incumbent_prediction must be finite and align with prediction rows"
        )
    points = len(prepared.horizontal) if grid_points is None else grid_points
    if isinstance(points, bool) or int(points) != points or int(points) < 8:
        raise ValueError("grid_points must be an integer >= 8")
    model_grid = np.linspace(0.0, 1.0, int(points), dtype=np.float64)

    horizontal_coordinate = np.linspace(
        0.0,
        1.0,
        len(prepared.horizontal),
        dtype=np.float64,
    )
    heel_coordinate = _heel_grid_coordinate(prepared)
    suffix_coordinate = heel_coordinate + (1.0 - heel_coordinate) * model_grid

    def suffix_feature(name: str) -> np.ndarray:
        return np.interp(
            suffix_coordinate,
            horizontal_coordinate,
            _named_column(
                prepared.horizontal,
                prepared.horizontal_feature_names,
                name,
            ),
        )

    intercept, slope, calibration_coverage, calibration_rmse = (
        _prefix_affine_calibration(prepared)
    )
    horizontal_gr = intercept + slope * suffix_feature("GR_filled")
    horizontal_gr_observed = np.clip(
        suffix_feature("GR_observed"),
        0.0,
        1.0,
    )

    row_coordinate = np.asarray(prepared.output_coordinate, dtype=np.float64)
    incumbent_grid = np.interp(model_grid, row_coordinate, incumbent)
    typewell_tvt, typewell_gr, support_min, support_max = _typewell_support(prepared)
    typewell_at_incumbent = np.interp(
        incumbent_grid,
        typewell_tvt,
        typewell_gr,
    )
    typewell_support = (
        (incumbent_grid >= support_min) & (incumbent_grid <= support_max)
    ).astype(np.float64)

    md = suffix_feature("MD")
    if np.any(np.diff(md) <= 0.0):
        raise ValueError("suffix MD must be strictly increasing")
    x_position = suffix_feature("X")
    y_position = suffix_feature("Y")
    z_position = suffix_feature("Z")
    dx_dmd = _gradient(x_position, md)
    dy_dmd = _gradient(y_position, md)
    dz_dmd = _gradient(z_position, md)
    speed = np.sqrt(dx_dmd * dx_dmd + dy_dmd * dy_dmd + dz_dmd * dz_dmd)
    speed = np.maximum(speed, np.finfo(np.float64).eps)
    inclination = np.arctan2(
        np.hypot(dx_dmd, dy_dmd),
        np.maximum(np.abs(dz_dmd), np.finfo(np.float64).eps),
    )
    tangent = np.vstack([dx_dmd, dy_dmd, dz_dmd]) / speed
    tangent_derivative = np.vstack(
        [_gradient(tangent[index], md) for index in range(3)]
    )
    curvature = np.sqrt(np.sum(tangent_derivative * tangent_derivative, axis=0))

    incumbent_d1 = _gradient(incumbent_grid, md)
    incumbent_d2 = _gradient(incumbent_d1, md)
    channels = np.vstack(
        [
            horizontal_gr,
            horizontal_gr_observed,
            typewell_at_incumbent,
            typewell_support,
            horizontal_gr - typewell_at_incumbent,
            incumbent_grid - float(prepared.anchor_tvt),
            incumbent_d1,
            incumbent_d2,
            dx_dmd,
            dy_dmd,
            dz_dmd,
            inclination,
            curvature,
            model_grid,
            np.full(len(model_grid), calibration_coverage, dtype=np.float64),
            np.full(len(model_grid), calibration_rmse, dtype=np.float64),
        ]
    ).astype(np.float32)
    cross_index = MODEL_INPUT_CHANNEL_NAMES.index("gr_cross_difference")
    horizontal_index = MODEL_INPUT_CHANNEL_NAMES.index("horizontal_gr_calibrated")
    typewell_index = MODEL_INPUT_CHANNEL_NAMES.index("typewell_gr_at_incumbent")
    channels[cross_index] = channels[horizontal_index] - channels[typewell_index]
    if channels.shape != (len(MODEL_INPUT_CHANNEL_NAMES), len(model_grid)):
        raise RuntimeError("model-input channel assembly has an invalid shape")
    if not np.isfinite(channels).all():
        raise ValueError("model-input channel assembly produced non-finite values")
    return SequenceModelInput(
        sequence=prepared,
        incumbent_prediction=incumbent,
        x=channels,
        grid_coordinate=model_grid,
        channel_names=MODEL_INPUT_CHANNEL_NAMES,
        control_id="none",
        paired_typewell_id=prepared.well_id,
        cross_series_enabled=True,
    )


def _load_cache(
    path: Path,
    *,
    well_id: str,
    config: FixedGridConfig,
    source_provenance: dict[str, Any],
) -> FixedGridSequenceWell | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            cached_well_id = str(cached["well_id"][0])
            cached_digest = str(cached["config_digest"][0])
            cached_schema = str(cached["schema_version"][0])
            cached_source_json = str(cached["source_provenance_json"][0])
            cached_source_digest = str(cached["source_provenance_digest"][0])
            cached_source = json.loads(cached_source_json)
            payload = FixedGridSequenceWell(
                well_id=cached_well_id,
                row_index=cached["row_index"],
                known_count=int(cached["known_count"][0]),
                anchor_tvt=float(cached["anchor_tvt"][0]),
                horizontal=cached["horizontal"],
                typewell=cached["typewell"],
                output_coordinate=cached["output_coordinate"],
                horizontal_feature_names=HORIZONTAL_FEATURE_NAMES,
                typewell_feature_names=TYPEWELL_FEATURE_NAMES,
                source_horizontal_rows=int(cached["source_horizontal_rows"][0]),
                source_typewell_rows=int(cached["source_typewell_rows"][0]),
                schema_version=cached_schema,
            )
        valid = (
            cached_well_id == str(well_id)
            and cached_digest == config.digest
            and cached_schema == SEQUENCE_SCHEMA_VERSION
            and cached_source == source_provenance
            and cached_source_digest == raw_source_provenance_digest(source_provenance)
            and payload.horizontal.shape
            == (config.horizontal_points, len(HORIZONTAL_FEATURE_NAMES))
            and payload.typewell.shape
            == (config.typewell_points, len(TYPEWELL_FEATURE_NAMES))
        )
        return payload if valid else None
    except (OSError, ValueError, KeyError, EOFError, TypeError):
        return None


def prepare_fixed_grid_well(
    well: WellFiles,
    *,
    split: str,
    cache_root: Path | None = None,
    config: FixedGridConfig = DEFAULT_FIXED_GRID_CONFIG,
) -> FixedGridSequenceWell:
    """Prepare one well without ever requesting target or geology columns."""

    if not isinstance(well, WellFiles):
        raise TypeError("well must be a WellFiles instance")
    if not isinstance(config, FixedGridConfig):
        raise TypeError("config must be a FixedGridConfig")
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if cache_root is None:
        return _build_sequence_well(well, config=config)

    cache = Path(cache_root).resolve()
    data_root = Path(well.horizontal_path).resolve().parent.parent
    if cache == data_root or data_root in cache.parents:
        raise ValueError("sequence cache cannot be inside competition data")
    horizontal, typewell, source_provenance = _load_well_sources_with_provenance(well)
    path = sequence_cache_path(cache, split=split, well_id=well.well_id, config=config)
    prepared = _load_cache(
        path,
        well_id=well.well_id,
        config=config,
        source_provenance=source_provenance,
    )
    if prepared is not None:
        return prepared
    prepared = build_fixed_grid_sequence(
        horizontal,
        typewell,
        well_id=well.well_id,
        config=config,
    )
    source_provenance_json = json.dumps(
        source_provenance,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    _atomic_npz(
        path,
        well_id=np.asarray([prepared.well_id], dtype="U"),
        config_digest=np.asarray([config.digest], dtype="U"),
        schema_version=np.asarray([prepared.schema_version], dtype="U"),
        source_provenance_json=np.asarray([source_provenance_json], dtype="U"),
        source_provenance_digest=np.asarray(
            [raw_source_provenance_digest(source_provenance)],
            dtype="U",
        ),
        row_index=prepared.row_index,
        known_count=np.asarray([prepared.known_count], dtype=np.int64),
        anchor_tvt=np.asarray([prepared.anchor_tvt], dtype=np.float64),
        horizontal=prepared.horizontal,
        typewell=prepared.typewell,
        output_coordinate=prepared.output_coordinate,
        source_horizontal_rows=np.asarray(
            [prepared.source_horizontal_rows], dtype=np.int64
        ),
        source_typewell_rows=np.asarray(
            [prepared.source_typewell_rows], dtype=np.int64
        ),
    )
    restored = _load_cache(
        path,
        well_id=well.well_id,
        config=config,
        source_provenance=source_provenance,
    )
    if restored is None:
        raise ValueError(
            f"sequence cache validation failed after write: {well.well_id}"
        )
    return restored


def fixed_grid_manifest(
    config: FixedGridConfig = DEFAULT_FIXED_GRID_CONFIG,
) -> dict[str, Any]:
    if not isinstance(config, FixedGridConfig):
        raise TypeError("config must be a FixedGridConfig")
    return {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "config": asdict(config),
        "config_digest": config.digest,
        "horizontal_source_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
        "typewell_source_columns": ["TVT", "GR"],
        "horizontal_features": list(HORIZONTAL_FEATURE_NAMES),
        "typewell_features": list(TYPEWELL_FEATURE_NAMES),
        "fold_specific_model_input_channels": list(MODEL_INPUT_CHANNEL_NAMES),
        "model_input_cache_policy": "never_cached",
        "raw_cache_provenance_schema": RAW_SOURCE_PROVENANCE_SCHEMA_VERSION,
        "raw_cache_reuse_requires_exact_inference_view_sha256": True,
        "raw_cache_mismatch_behavior": "rebuild from inference-safe source columns",
        "incumbent_requirement": (
            "row-aligned predictions from the fold-appropriate incumbent state"
        ),
        "target_columns": [],
        "formation_columns": [],
        "geology_columns": [],
        "image_inputs": False,
        "global_learned_statistics": False,
    }


__all__ = [
    "DEFAULT_FIXED_GRID_CONFIG",
    "FixedGridConfig",
    "HORIZONTAL_FEATURE_NAMES",
    "MODEL_INPUT_CHANNEL_NAMES",
    "RAW_SOURCE_PROVENANCE_SCHEMA_VERSION",
    "SEQUENCE_SCHEMA_VERSION",
    "TYPEWELL_FEATURE_NAMES",
    "build_fixed_grid_sequence",
    "build_sequence_model_input",
    "fixed_grid_manifest",
    "prepare_fixed_grid_well",
    "raw_source_provenance_digest",
    "raw_well_source_provenance",
    "sequence_cache_path",
]
