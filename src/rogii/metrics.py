"""Exact pooled row-RMSE and secondary well-level diagnostics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _validated_pair(
    y_true: Iterable[float], y_pred: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(list(y_true) if not hasattr(y_true, "__len__") else y_true, dtype=float)
    prediction = np.asarray(
        list(y_pred) if not hasattr(y_pred, "__len__") else y_pred, dtype=float
    )
    if truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError("Metric inputs must be one-dimensional")
    if truth.size == 0:
        raise ValueError("Metric inputs must not be empty")
    if truth.size != prediction.size:
        raise ValueError(f"Metric input length mismatch: {truth.size} != {prediction.size}")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs and predictions must contain only finite values")
    return truth, prediction


def pooled_row_rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Compute the competition's primary metric over all pooled rows once."""
    truth, prediction = _validated_pair(y_true, y_pred)
    error = truth - prediction
    return math.sqrt(float(error @ error) / len(error))


def per_well_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute secondary per-well RMSE/SSE diagnostics; never average as primary."""
    required = {"well_id", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Per-well diagnostic input missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Per-well diagnostic input must not be empty")
    if frame["well_id"].isna().any():
        raise ValueError("Per-well diagnostic input contains null well_id values")
    rows: list[dict[str, float | int | str]] = []
    for well_id, group in frame.groupby("well_id", sort=True):
        truth, prediction = _validated_pair(group["y_true"], group["y_pred"])
        error = truth - prediction
        sse = float(error @ error)
        rows.append(
            {"well_id": str(well_id), "n_rows": len(error), "sse": sse,
             "rmse": math.sqrt(sse / len(error))}
        )
    return pd.DataFrame(rows)


def tail_diagnostics(per_well: pd.DataFrame) -> dict[str, float]:
    """Summarize well-RMSE quantiles and worst-decile SSE concentration."""
    required = {"rmse", "sse"}
    missing = required - set(per_well.columns)
    if missing or per_well.empty:
        raise ValueError(f"Tail diagnostics require non-empty columns {sorted(required)}")
    values = per_well["rmse"].to_numpy(dtype=float)
    sse = per_well["sse"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(sse).all() or sse.sum() <= 0:
        raise ValueError("Tail diagnostic values must be finite with positive total SSE")
    worst_count = max(1, math.ceil(len(per_well) * 0.10))
    worst_sse = float(np.sort(sse)[-worst_count:].sum())
    return {
        "p50_rmse": float(np.quantile(values, 0.50)),
        "p90_rmse": float(np.quantile(values, 0.90)),
        "p95_rmse": float(np.quantile(values, 0.95)),
        "max_rmse": float(values.max()),
        "worst_10pct_sse_share": worst_sse / float(sse.sum()),
    }
