"""Deterministic matched controls for target-free prefix observability."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np
import pandas as pd

from ..data import select_inference_view, validate_prefix


def stable_seed(*parts: object, base_seed: int = 20260729) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(base_seed)).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") & 0x7FFFFFFF


def phase_destroy_preserve_marginal(
    values: np.ndarray, *, seed: int
) -> np.ndarray:
    """Destroy phase while preserving finite-value multiset and missing positions."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("phase-destroy input must be one-dimensional")
    result = source.copy()
    finite = np.isfinite(source)
    observed = source[finite]
    if len(observed) < 4:
        raise ValueError("phase destroy requires at least four finite values")
    centered = observed - float(np.mean(observed))
    spectrum = np.fft.rfft(centered)
    random = np.random.default_rng(int(seed))
    if len(spectrum) > 2:
        phases = random.uniform(-np.pi, np.pi, len(spectrum) - 2)
        spectrum[1:-1] = np.abs(spectrum[1:-1]) * np.exp(1j * phases)
    surrogate = np.fft.irfft(spectrum, n=len(observed))
    order = np.argsort(surrogate, kind="stable")
    ranked = np.empty(len(observed), dtype=np.float64)
    ranked[order] = np.sort(observed, kind="stable")
    if np.array_equal(ranked, observed):
        ranked = np.roll(ranked, max(1, len(ranked) // 3))
    result[finite] = ranked
    return result


def shuffle_gr_preserve_missingness(values: np.ndarray, *, seed: int) -> np.ndarray:
    """Deterministically shuffle finite GR values without moving missingness."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("GR shuffle input must be one-dimensional")
    result = source.copy()
    finite = np.isfinite(source)
    observed = source[finite]
    if len(observed) < 2:
        raise ValueError("GR shuffle requires at least two finite values")
    random = np.random.default_rng(int(seed))
    permutation = random.permutation(len(observed))
    if np.array_equal(permutation, np.arange(len(observed))):
        permutation = np.roll(permutation, 1)
    result[finite] = observed[permutation]
    return result


@dataclass(frozen=True)
class TypewellNuisance:
    well_id: str
    tvt_min: float
    tvt_max: float
    tvt_span: float
    row_count: int
    gr_missing_fraction: float
    gr_mean: float
    gr_scale: float


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return 1.0
    q25, q75 = np.quantile(finite, (0.25, 0.75))
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale < 1.0e-6:
        scale = float(np.std(finite))
    return max(scale, 1.0e-6)


def summarize_typewell(well_id: str, frame: pd.DataFrame) -> TypewellNuisance:
    view = select_inference_view(frame, kind="typewell")
    tvt = view["TVT"].to_numpy(dtype=np.float64)
    gr = view["GR"].to_numpy(dtype=np.float64)
    finite_tvt = tvt[np.isfinite(tvt)]
    finite_gr = gr[np.isfinite(gr)]
    if len(finite_tvt) < 16 or len(finite_gr) < 16:
        raise ValueError(f"typewell support is insufficient: {well_id}")
    lower = float(np.min(finite_tvt))
    upper = float(np.max(finite_tvt))
    return TypewellNuisance(
        well_id=str(well_id),
        tvt_min=lower,
        tvt_max=upper,
        tvt_span=max(upper - lower, 1.0e-6),
        row_count=len(view),
        gr_missing_fraction=float(1.0 - len(finite_gr) / len(gr)),
        gr_mean=float(np.mean(finite_gr)),
        gr_scale=_robust_scale(finite_gr),
    )


def _coverage(prefix_tvt: np.ndarray, summary: TypewellNuisance) -> float:
    values = np.asarray(prefix_tvt, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    lower = float(np.min(values[finite]))
    upper = float(np.max(values[finite]))
    if upper <= lower:
        return float(summary.tvt_min <= lower <= summary.tvt_max)
    overlap = max(0.0, min(upper, summary.tvt_max) - max(lower, summary.tvt_min))
    return float(np.clip(overlap / (upper - lower), 0.0, 1.0))


def nuisance_distance(
    correct: TypewellNuisance,
    candidate: TypewellNuisance,
    *,
    prefix_tvt: np.ndarray,
) -> float:
    """Scale-free matching distance using no GR order information."""

    amplitude = max(correct.gr_scale, candidate.gr_scale, 1.0)
    return float(
        2.0 * abs(_coverage(prefix_tvt, correct) - _coverage(prefix_tvt, candidate))
        + abs(math.log(candidate.row_count / correct.row_count))
        + abs(math.log(candidate.tvt_span / correct.tvt_span))
        + abs(candidate.gr_missing_fraction - correct.gr_missing_fraction)
        + abs(candidate.gr_mean - correct.gr_mean) / amplitude
        + abs(math.log(candidate.gr_scale / correct.gr_scale))
    )


def deterministic_matched_wrong_typewell(
    well_id: str,
    summaries: Mapping[str, TypewellNuisance],
    *,
    prefix_tvt: np.ndarray,
) -> tuple[str, float]:
    """Choose the nearest nuisance-matched non-self typewell deterministically."""

    identifier = str(well_id)
    if identifier not in summaries:
        raise KeyError(f"missing correct typewell summary: {identifier}")
    correct = summaries[identifier]
    candidates = [
        (
            nuisance_distance(correct, summary, prefix_tvt=prefix_tvt),
            candidate_id,
        )
        for candidate_id, summary in summaries.items()
        if candidate_id != identifier
    ]
    if not candidates:
        raise ValueError("wrong-typewell control requires at least two wells")
    distance, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return selected, float(distance)


def nuisance_only_cost(
    horizontal: pd.DataFrame,
    summary: TypewellNuisance,
) -> dict[str, float]:
    """Target-free nuisance score with no GR order or correspondence channel."""

    view = select_inference_view(horizontal, kind="horizontal")
    known_count, _ = validate_prefix(view["TVT_input"])
    prefix_tvt = view["TVT_input"].iloc[:known_count].to_numpy(dtype=np.float64)
    prefix_gr = view["GR"].iloc[:known_count].to_numpy(dtype=np.float64)
    finite_tvt = prefix_tvt[np.isfinite(prefix_tvt)]
    finite_gr = prefix_gr[np.isfinite(prefix_gr)]
    if len(finite_tvt) < 2 or len(finite_gr) < 2:
        raise ValueError("prefix nuisance summary is insufficient")
    prefix_span = max(float(np.max(finite_tvt) - np.min(finite_tvt)), 1.0e-6)
    gr_scale = _robust_scale(finite_gr)
    coverage = _coverage(prefix_tvt, summary)
    row_density = summary.row_count / max(summary.tvt_span, 1.0e-6)
    prefix_density = len(finite_tvt) / prefix_span
    amplitude_mean_delta = abs(float(np.mean(finite_gr)) - summary.gr_mean) / max(
        gr_scale, summary.gr_scale, 1.0
    )
    amplitude_scale_delta = abs(math.log(summary.gr_scale / gr_scale))
    cost = (
        2.0 * (1.0 - coverage)
        + 0.5 * abs(math.log(max(row_density, 1.0e-6) / max(prefix_density, 1.0e-6)))
        + abs(summary.gr_missing_fraction - (1.0 - len(finite_gr) / len(prefix_gr)))
        + amplitude_mean_delta
        + amplitude_scale_delta
    )
    return {
        "nuisance_cost": float(cost),
        "nuisance_coverage": coverage,
        "nuisance_row_density": float(row_density),
        "nuisance_gr_missing_fraction": summary.gr_missing_fraction,
        "nuisance_gr_mean": summary.gr_mean,
        "nuisance_gr_scale": summary.gr_scale,
    }
