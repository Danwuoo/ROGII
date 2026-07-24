"""Inference-safe per-well features used by the P3 residual model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    HORIZONTAL_INFERENCE_COLUMNS,
    TYPEWELL_INFERENCE_COLUMNS,
    select_inference_view,
    validate_prefix,
)


FEATURE_VERSION = "p3_base_test_safe_features_v1"
GR_WINDOWS = (11, 51, 201)

FEATURE_COLUMNS = (
    "last_known_tvt",
    "known_prefix_length",
    "row_index",
    "md_from_ps",
    "normalized_distance_from_ps",
    "remaining_normalized_horizon",
    "total_well_length",
    "prediction_fraction",
    "dx_from_ps",
    "dy_from_ps",
    "dz_from_ps",
    "displacement_3d_from_ps",
    "tangent_x",
    "tangent_y",
    "tangent_z",
    "azimuth_sin",
    "azimuth_cos",
    "inclination_proxy",
    "dogleg_proxy",
    "gr_raw",
    "gr_is_missing",
    "gr_missing_run_length",
    "gr_nearest_observed_distance",
    "gr_first_difference",
    "gr_first_difference_is_missing",
    "gr_roll_mean_11",
    "gr_roll_std_11",
    "gr_roll_min_11",
    "gr_roll_max_11",
    "gr_roll_observed_fraction_11",
    "gr_roll_mean_51",
    "gr_roll_std_51",
    "gr_roll_min_51",
    "gr_roll_max_51",
    "gr_roll_observed_fraction_51",
    "gr_roll_mean_201",
    "gr_roll_std_201",
    "gr_roll_min_201",
    "gr_roll_max_201",
    "gr_roll_observed_fraction_201",
    "prefix_gr_mean",
    "prefix_gr_std",
    "prefix_gr_min",
    "prefix_gr_max",
    "prefix_gr_missing_fraction",
    "prefix_tvt_slope_50",
    "prefix_tvt_diff_mad_50",
    "prefix_tvt_slope_200",
    "prefix_tvt_diff_mad_200",
    "typewell_tvt_min",
    "typewell_tvt_max",
    "typewell_tvt_range",
    "typewell_tvt_step",
    "typewell_row_count",
    "typewell_gr_mean",
    "typewell_gr_std",
    "typewell_gr_min",
    "typewell_gr_max",
    "typewell_gr_q10",
    "typewell_gr_q25",
    "typewell_gr_q50",
    "typewell_gr_q75",
    "typewell_gr_q90",
)


@dataclass(frozen=True)
class WellFeatureFrame:
    """Features and row keys for one well's original prediction zone."""

    features: pd.DataFrame
    row_index: np.ndarray
    anchor: float
    metadata: dict[str, float | int]


def _nearest_observed_distance(values: np.ndarray) -> np.ndarray:
    observed = np.isfinite(values)
    n_rows = len(values)
    if not observed.any():
        return np.full(n_rows, float(n_rows), dtype=np.float32)
    indices = np.arange(n_rows)
    previous = np.maximum.accumulate(np.where(observed, indices, -n_rows))
    following = np.minimum.accumulate(np.where(observed, indices, 2 * n_rows)[::-1])[::-1]
    return np.minimum(indices - previous, following - indices).astype(np.float32)


def _missing_run_length(values: np.ndarray) -> np.ndarray:
    missing = ~np.isfinite(values)
    if not missing.any():
        return np.zeros(len(values), dtype=np.float32)
    series = pd.Series(missing)
    groups = series.ne(series.shift()).cumsum()
    lengths = series.groupby(groups).transform("sum").where(series, 0)
    return lengths.to_numpy(dtype=np.float32)


def _prefix_tvt_context(md: np.ndarray, tvt: np.ndarray, window: int) -> tuple[float, float]:
    start = max(0, len(tvt) - window)
    local_md = md[start:]
    local_tvt = tvt[start:]
    denominator = float(local_md[-1] - local_md[0]) if len(local_md) > 1 else 0.0
    slope = float((local_tvt[-1] - local_tvt[0]) / denominator) if denominator else 0.0
    differences = np.diff(local_tvt)
    if len(differences):
        median = float(np.median(differences))
        mad = float(np.median(np.abs(differences - median)))
    else:
        mad = 0.0
    return slope, mad


def _finite_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return math.nan, math.nan, math.nan, math.nan
    return (
        float(np.mean(finite)),
        float(np.std(finite)),
        float(np.min(finite)),
        float(np.max(finite)),
    )


def build_prediction_features(
    horizontal: pd.DataFrame, typewell: pd.DataFrame
) -> WellFeatureFrame:
    """Build fixed test-safe features for only the original prediction rows.

    Defensive inference views are selected before any computation, so extra
    training labels or train-only columns cannot affect the result.
    """
    h = select_inference_view(horizontal, kind="horizontal")
    t = select_inference_view(typewell, kind="typewell")
    known_count, prediction_count = validate_prefix(h["TVT_input"])
    n_rows = len(h)
    prediction_index = np.arange(known_count, n_rows, dtype=np.int64)

    md = h["MD"].to_numpy(dtype=np.float64)
    x = h["X"].to_numpy(dtype=np.float64)
    y = h["Y"].to_numpy(dtype=np.float64)
    z = h["Z"].to_numpy(dtype=np.float64)
    gr = h["GR"].to_numpy(dtype=np.float64)
    if not all(np.isfinite(values).all() for values in (md, x, y, z)):
        raise ValueError("MD and XYZ must contain only finite values")
    if np.any(np.diff(md) <= 0):
        raise ValueError("MD must be strictly increasing")

    ps_index = known_count - 1
    anchor = float(h["TVT_input"].iloc[ps_index])
    md_ps = float(md[ps_index])
    horizon = float(md[-1] - md_ps)
    if horizon <= 0:
        raise ValueError("Prediction horizon must have positive MD length")

    tangent_x = np.gradient(x, md)
    tangent_y = np.gradient(y, md)
    tangent_z = np.gradient(z, md)
    tangent_norm = np.sqrt(tangent_x**2 + tangent_y**2 + tangent_z**2)
    safe_norm = np.where(tangent_norm > 0, tangent_norm, np.nan)
    unit_x = tangent_x / safe_norm
    unit_y = tangent_y / safe_norm
    unit_z = tangent_z / safe_norm
    dogleg = np.sqrt(
        np.gradient(unit_x, md) ** 2
        + np.gradient(unit_y, md) ** 2
        + np.gradient(unit_z, md) ** 2
    )
    azimuth = np.arctan2(tangent_y, tangent_x)

    gr_series = pd.Series(gr)
    gr_diff = np.empty(n_rows, dtype=np.float64)
    gr_diff[0] = np.nan
    gr_diff[1:] = np.diff(gr)
    missing_run = _missing_run_length(gr)
    nearest_observed = _nearest_observed_distance(gr)

    prefix_gr = gr[:known_count]
    prefix_gr_mean, prefix_gr_std, prefix_gr_min, prefix_gr_max = _finite_summary(prefix_gr)
    prefix_tvt = h["TVT_input"].iloc[:known_count].to_numpy(dtype=np.float64)
    slope_50, mad_50 = _prefix_tvt_context(md[:known_count], prefix_tvt, 50)
    slope_200, mad_200 = _prefix_tvt_context(md[:known_count], prefix_tvt, 200)

    typewell_tvt = t["TVT"].to_numpy(dtype=np.float64)
    typewell_gr = t["GR"].to_numpy(dtype=np.float64)
    if len(typewell_tvt) < 2 or not np.isfinite(typewell_tvt).all() or np.any(np.diff(typewell_tvt) <= 0):
        raise ValueError("Typewell TVT must be finite and strictly increasing")
    if not np.isfinite(typewell_gr).all():
        raise ValueError("Typewell GR must be finite")
    typewell_quantiles = np.quantile(typewell_gr, [0.10, 0.25, 0.50, 0.75, 0.90])

    all_features: dict[str, Any] = {
        "last_known_tvt": anchor,
        "known_prefix_length": known_count,
        "row_index": np.arange(n_rows),
        "md_from_ps": md - md_ps,
        "normalized_distance_from_ps": (md - md_ps) / horizon,
        "remaining_normalized_horizon": (md[-1] - md) / horizon,
        "total_well_length": float(md[-1] - md[0]),
        "prediction_fraction": prediction_count / n_rows,
        "dx_from_ps": x - x[ps_index],
        "dy_from_ps": y - y[ps_index],
        "dz_from_ps": z - z[ps_index],
        "displacement_3d_from_ps": np.sqrt(
            (x - x[ps_index]) ** 2 + (y - y[ps_index]) ** 2 + (z - z[ps_index]) ** 2
        ),
        "tangent_x": tangent_x,
        "tangent_y": tangent_y,
        "tangent_z": tangent_z,
        "azimuth_sin": np.sin(azimuth),
        "azimuth_cos": np.cos(azimuth),
        "inclination_proxy": unit_z,
        "dogleg_proxy": dogleg,
        "gr_raw": gr,
        "gr_is_missing": ~np.isfinite(gr),
        "gr_missing_run_length": missing_run,
        "gr_nearest_observed_distance": nearest_observed,
        "gr_first_difference": gr_diff,
        "gr_first_difference_is_missing": ~np.isfinite(gr_diff),
        "prefix_gr_mean": prefix_gr_mean,
        "prefix_gr_std": prefix_gr_std,
        "prefix_gr_min": prefix_gr_min,
        "prefix_gr_max": prefix_gr_max,
        "prefix_gr_missing_fraction": float(np.mean(~np.isfinite(prefix_gr))),
        "prefix_tvt_slope_50": slope_50,
        "prefix_tvt_diff_mad_50": mad_50,
        "prefix_tvt_slope_200": slope_200,
        "prefix_tvt_diff_mad_200": mad_200,
        "typewell_tvt_min": float(typewell_tvt[0]),
        "typewell_tvt_max": float(typewell_tvt[-1]),
        "typewell_tvt_range": float(typewell_tvt[-1] - typewell_tvt[0]),
        "typewell_tvt_step": float(np.median(np.diff(typewell_tvt))),
        "typewell_row_count": len(typewell_tvt),
        "typewell_gr_mean": float(np.mean(typewell_gr)),
        "typewell_gr_std": float(np.std(typewell_gr)),
        "typewell_gr_min": float(np.min(typewell_gr)),
        "typewell_gr_max": float(np.max(typewell_gr)),
        "typewell_gr_q10": float(typewell_quantiles[0]),
        "typewell_gr_q25": float(typewell_quantiles[1]),
        "typewell_gr_q50": float(typewell_quantiles[2]),
        "typewell_gr_q75": float(typewell_quantiles[3]),
        "typewell_gr_q90": float(typewell_quantiles[4]),
    }
    for window in GR_WINDOWS:
        rolling = gr_series.rolling(window=window, center=True, min_periods=1)
        all_features[f"gr_roll_mean_{window}"] = rolling.mean().to_numpy()
        all_features[f"gr_roll_std_{window}"] = rolling.std(ddof=0).to_numpy()
        all_features[f"gr_roll_min_{window}"] = rolling.min().to_numpy()
        all_features[f"gr_roll_max_{window}"] = rolling.max().to_numpy()
        all_features[f"gr_roll_observed_fraction_{window}"] = (
            gr_series.notna().astype(float).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
        )

    frame = pd.DataFrame(index=np.arange(n_rows))
    for column in FEATURE_COLUMNS:
        values = all_features[column]
        if np.isscalar(values):
            frame[column] = np.float32(values)
        else:
            frame[column] = np.asarray(values, dtype=np.float32)
    prediction_features = frame.iloc[prediction_index].reset_index(drop=True)
    metadata = {
        "known_rows": known_count,
        "prediction_rows": prediction_count,
        "gr_missing_fraction": float(np.mean(~np.isfinite(gr))),
        "typewell_step": float(np.median(np.diff(typewell_tvt))),
    }
    return WellFeatureFrame(prediction_features, prediction_index, anchor, metadata)


def feature_manifest() -> dict[str, object]:
    """Return the explicit, serializable P3 base-feature contract."""
    return {
        "feature_version": FEATURE_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "dtype": "float32",
        "horizontal_source_columns": list(HORIZONTAL_INFERENCE_COLUMNS),
        "typewell_source_columns": list(TYPEWELL_INFERENCE_COLUMNS),
        "gr_windows": list(GR_WINDOWS),
        "target_representation": "TVT - last_known_TVT_input",
    }
