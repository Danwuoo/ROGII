"""Target-free within-well prefix calibration and multiscale likelihood."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..data import select_inference_view, validate_prefix
from .contracts import PrefixCalibrationConfig, PrefixLikelihoodResult


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        raise ValueError("robust scale requires at least two finite values")
    center = float(np.median(finite))
    q25, q75 = np.quantile(finite, (0.25, 0.75))
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale < 1.0e-6:
        scale = float(np.std(finite))
    return center, max(scale, 1.0e-6)


def _collapse_coordinate(
    coordinate: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    finite_coordinate = np.isfinite(coordinate)
    x = np.asarray(coordinate[finite_coordinate], dtype=np.float64)
    y = np.asarray(values[finite_coordinate], dtype=np.float64)
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    finite_y = np.isfinite(y)
    sums = np.bincount(inverse[finite_y], weights=y[finite_y], minlength=len(unique))
    counts = np.bincount(inverse[finite_y], minlength=len(unique))
    result = np.full(len(unique), np.nan, dtype=np.float64)
    available = counts > 0
    result[available] = sums[available] / counts[available]
    return unique, result


def _interpolate_finite(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 2:
        raise ValueError("interpolation requires at least two finite pairs")
    return np.interp(grid, x[finite], y[finite])


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    actual = min(int(width), len(values) if len(values) % 2 else len(values) - 1)
    actual = max(1, actual)
    if actual % 2 == 0:
        actual -= 1
    if actual <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    half = actual // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.full(actual, 1.0 / actual), mode="valid")


def _student_cost(residual: np.ndarray, scale: float, df: float) -> float:
    vector = np.asarray(residual, dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if len(vector) == 0:
        return float("inf")
    z = vector / max(float(scale), 1.0e-6)
    return float(np.mean(0.5 * (df + 1.0) * np.log1p((z * z) / df)))


def _robust_affine(
    reference: np.ndarray,
    horizontal: np.ndarray,
    mask: np.ndarray,
    config: PrefixCalibrationConfig,
) -> tuple[float, float]:
    selected = mask & np.isfinite(reference) & np.isfinite(horizontal)
    if int(np.sum(selected)) < config.minimum_evaluation_pairs:
        raise ValueError("affine prefix calibration has insufficient rows")
    x = reference[selected]
    y = horizontal[selected]
    x_center, x_scale = _robust_location_scale(x)
    y_center, y_scale = _robust_location_scale(y)
    slope = float(
        np.clip(
            y_scale / x_scale,
            config.amplitude_scale_min,
            config.amplitude_scale_max,
        )
    )
    offset = float(np.median(y - slope * x))
    for _ in range(3):
        residual = y - (offset + slope * x)
        _, residual_scale = _robust_location_scale(residual)
        weight = 1.0 / np.maximum(1.0, np.abs(residual) / (1.5 * residual_scale))
        design = np.column_stack([np.ones(len(x)), x])
        weighted = design * np.sqrt(weight[:, None])
        target = y * np.sqrt(weight)
        coefficient = np.linalg.lstsq(weighted, target, rcond=None)[0]
        offset = float(coefficient[0])
        slope = float(
            np.clip(
                coefficient[1],
                config.amplitude_scale_min,
                config.amplitude_scale_max,
            )
        )
        offset = float(np.median(y - slope * x))
    return offset, slope


def _reference_at(
    grid: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    center: float,
    lag_ft: float,
    stretch: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = center + (grid - center) * float(stretch) + float(lag_ft)
    valid = (
        np.isfinite(candidate)
        & (candidate >= typewell_tvt[0])
        & (candidate <= typewell_tvt[-1])
    )
    result = np.full(len(grid), np.nan, dtype=np.float64)
    result[valid] = np.interp(candidate[valid], typewell_tvt, typewell_gr)
    return result, valid


def _selection_cost(
    horizontal: np.ndarray,
    calibrated_reference: np.ndarray,
    mask: np.ndarray,
    *,
    horizontal_scale: float,
    df: float,
) -> float:
    selected = mask & np.isfinite(calibrated_reference)
    if not selected.any():
        return float("inf")
    raw = _student_cost(
        horizontal[selected] - calibrated_reference[selected],
        horizontal_scale,
        df,
    )
    smooth_h = _smooth(horizontal, 5)
    filled_reference = np.interp(
        np.arange(len(calibrated_reference)),
        np.flatnonzero(np.isfinite(calibrated_reference)),
        calibrated_reference[np.isfinite(calibrated_reference)],
    )
    smooth_r = _smooth(filled_reference, 5)
    smooth = _student_cost(
        (smooth_h - smooth_r)[selected],
        horizontal_scale,
        df,
    )
    return 0.6 * raw + 0.4 * smooth


def calibrate_prefix_likelihood(
    horizontal_frame: pd.DataFrame,
    typewell_frame: pd.DataFrame,
    *,
    well_id: str,
    reference_well_id: str | None = None,
    control_id: str = "correct",
    config: PrefixCalibrationConfig | None = None,
) -> PrefixLikelihoodResult:
    """Fit lag/stretch/amplitude on calibration blocks and score held-out blocks.

    Only the horizontal inference view and typewell inference view are copied.
    In particular, a training frame's suffix target ``TVT`` is never requested.
    """

    settings = config or PrefixCalibrationConfig()
    horizontal = select_inference_view(horizontal_frame, kind="horizontal")
    typewell = select_inference_view(typewell_frame, kind="typewell")
    known_count, _ = validate_prefix(horizontal["TVT_input"])
    prefix = horizontal.iloc[:known_count]
    prefix_tvt_raw = prefix["TVT_input"].to_numpy(dtype=np.float64)
    prefix_gr_raw = prefix["GR"].to_numpy(dtype=np.float64)
    finite_gr_count = int(np.sum(np.isfinite(prefix_gr_raw)))
    if finite_gr_count < settings.minimum_prefix_pairs:
        raise ValueError("prefix has insufficient finite GR pairs")
    prefix_missing = float(1.0 - finite_gr_count / len(prefix_gr_raw))
    collapsed_tvt, collapsed_gr = _collapse_coordinate(prefix_tvt_raw, prefix_gr_raw)
    finite_collapsed = np.isfinite(collapsed_gr)
    if int(np.sum(finite_collapsed)) < settings.minimum_prefix_pairs:
        raise ValueError("prefix TVT-axis support is insufficient")
    lower = float(np.min(collapsed_tvt[finite_collapsed]))
    upper = float(np.max(collapsed_tvt[finite_collapsed]))
    prefix_length = upper - lower
    if prefix_length <= settings.resample_step_ft * settings.minimum_evaluation_pairs:
        raise ValueError("prefix physical length is insufficient")
    step = max(
        settings.resample_step_ft,
        prefix_length / max(settings.maximum_resampled_points - 1, 1),
    )
    grid = np.arange(lower, upper + 0.5 * step, step, dtype=np.float64)
    horizontal_gr = _interpolate_finite(collapsed_tvt, collapsed_gr, grid)
    _, horizontal_scale = _robust_location_scale(horizontal_gr)

    tw_tvt_raw = typewell["TVT"].to_numpy(dtype=np.float64)
    tw_gr_raw = typewell["GR"].to_numpy(dtype=np.float64)
    tw_tvt, tw_gr = _collapse_coordinate(tw_tvt_raw, tw_gr_raw)
    finite_tw = np.isfinite(tw_gr)
    tw_tvt = tw_tvt[finite_tw]
    tw_gr = tw_gr[finite_tw]
    if len(tw_tvt) < 16 or np.any(np.diff(tw_tvt) <= 0.0):
        raise ValueError("typewell TVT/GR support is insufficient")

    block = np.floor((grid - grid[0]) / settings.calibration_block_ft).astype(int)
    evaluation_mask = block % 3 == 2
    calibration_mask = ~evaluation_mask
    if int(np.sum(evaluation_mask)) < settings.minimum_evaluation_pairs:
        evaluation_mask = np.arange(len(grid)) % 3 == 2
        calibration_mask = ~evaluation_mask
    center = float(np.median(grid))
    best: tuple[float, float, float, float, float, np.ndarray, np.ndarray] | None = None
    for stretch in settings.stretch_values:
        for lag in settings.lag_values_ft:
            reference, coverage = _reference_at(
                grid,
                tw_tvt,
                tw_gr,
                center=center,
                lag_ft=lag,
                stretch=stretch,
            )
            fit_mask = calibration_mask & coverage
            if int(np.sum(fit_mask)) < settings.minimum_evaluation_pairs:
                continue
            offset, amplitude = _robust_affine(
                reference, horizontal_gr, fit_mask, settings
            )
            calibrated = offset + amplitude * reference
            cost = _selection_cost(
                horizontal_gr,
                calibrated,
                fit_mask,
                horizontal_scale=horizontal_scale,
                df=settings.student_t_df,
            )
            candidate = (
                cost,
                abs(lag),
                abs(stretch - 1.0),
                float(lag),
                float(stretch),
                calibrated,
                coverage,
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise ValueError("no lag/stretch candidate has sufficient typewell coverage")
    _, _, _, lag, stretch, calibrated, coverage = best
    fit_mask = calibration_mask & coverage
    # Rebuild the selected raw reference so recorded affine parameters are exact.
    raw_reference, coverage = _reference_at(
        grid,
        tw_tvt,
        tw_gr,
        center=center,
        lag_ft=lag,
        stretch=stretch,
    )
    offset, amplitude = _robust_affine(
        raw_reference, horizontal_gr, calibration_mask & coverage, settings
    )
    calibrated = offset + amplitude * raw_reference
    evaluation = evaluation_mask & coverage
    if int(np.sum(evaluation)) < settings.minimum_evaluation_pairs:
        raise ValueError("selected likelihood has insufficient held-out prefix rows")

    raw_cost = _student_cost(
        (horizontal_gr - calibrated)[evaluation],
        horizontal_scale,
        settings.student_t_df,
    )
    filled_reference = np.interp(
        np.arange(len(calibrated)),
        np.flatnonzero(np.isfinite(calibrated)),
        calibrated[np.isfinite(calibrated)],
    )
    smooth_costs: list[float] = []
    for physical_scale in settings.smoothing_scales_ft:
        width = max(3, int(round(physical_scale / step)))
        if width % 2 == 0:
            width += 1
        smooth_costs.append(
            _student_cost(
                (_smooth(horizontal_gr, width) - _smooth(filled_reference, width))[
                    evaluation
                ],
                horizontal_scale,
                settings.student_t_df,
            )
        )
    smooth_cost = float(np.mean(smooth_costs))
    h_derivative = np.gradient(horizontal_gr, step)
    r_derivative = np.gradient(filled_reference, step)
    _, derivative_scale = _robust_location_scale(h_derivative[calibration_mask])
    derivative_cost = _student_cost(
        (h_derivative - r_derivative)[evaluation],
        derivative_scale,
        settings.student_t_df,
    )
    contrast_width = max(3, int(round(settings.smoothing_scales_ft[-1] / step)))
    if contrast_width % 2 == 0:
        contrast_width += 1
    h_contrast = horizontal_gr - _smooth(horizontal_gr, contrast_width)
    r_contrast = filled_reference - _smooth(filled_reference, contrast_width)
    _, contrast_scale = _robust_location_scale(h_contrast[calibration_mask])
    contrast_cost = _student_cost(
        (h_contrast - r_contrast)[evaluation],
        contrast_scale,
        settings.student_t_df,
    )
    texture_threshold = 0.25 * derivative_scale
    informative = (
        evaluation
        & (np.abs(h_derivative) >= texture_threshold)
        & (np.abs(r_derivative) >= texture_threshold)
    )
    polarity_cost = (
        float(np.mean(np.signbit(h_derivative[informative]) != np.signbit(r_derivative[informative])))
        if informative.any()
        else 1.0
    )
    total_cost = float(
        settings.raw_weight * raw_cost
        + settings.smooth_weight * smooth_cost
        + settings.derivative_weight * derivative_cost
        + settings.contrast_weight * contrast_cost
        + settings.polarity_weight * polarity_cost
    )
    coverage_fraction = float(np.mean(coverage))
    texture_reliability = float(
        np.clip(np.std(h_derivative) / max(derivative_scale, 1.0e-6), 0.0, 1.0)
    )
    reliability = float(
        np.clip(
            (1.0 - prefix_missing)
            * coverage_fraction
            * texture_reliability,
            0.0,
            1.0,
        )
    )
    boundary = float(
        (lag in (settings.lag_values_ft[0], settings.lag_values_ft[-1]))
        or (
            stretch
            in (settings.stretch_values[0], settings.stretch_values[-1])
        )
    )
    return PrefixLikelihoodResult(
        well_id=str(well_id),
        reference_well_id=str(reference_well_id or well_id),
        control_id=str(control_id),
        cost=total_cost,
        coherence=float(1.0 / (1.0 + total_cost)),
        reliability=reliability,
        effective_rows=int(np.sum(evaluation)),
        prefix_rows=known_count,
        prefix_length_ft=prefix_length,
        gr_missing_fraction=prefix_missing,
        typewell_coverage_fraction=coverage_fraction,
        lag_ft=lag,
        stretch=stretch,
        amplitude_offset=offset,
        amplitude_scale=amplitude,
        raw_cost=raw_cost,
        smooth_cost=smooth_cost,
        derivative_cost=derivative_cost,
        contrast_cost=contrast_cost,
        polarity_cost=polarity_cost,
        calibration_rows=int(np.sum(calibration_mask & coverage)),
        evaluation_rows=int(np.sum(evaluation)),
        boundary_fraction=boundary,
    )
