"""Inference-safe GR sequence alignment against a well's paired typewell.

This module is a clean-room implementation inspired by generic particle-filter,
beam-search, and normalized-cross-correlation alignment ideas.  It does not
import code or parameters from the referenced Kaggle notebook.  The only data
views accepted by the public family are the competition's inference-time
horizontal and typewell schemas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import select_inference_view, validate_prefix


ALIGNMENT_SCHEMA_VERSION = "safe_physics_alignment_v1"
ALIGNMENT_STATUS_OK = "ok"
ALIGNMENT_STATUS_IDENTITY = "identity_fallback"

CONTROL_NONE = "none"
CONTROL_WRONG_TYPEWELL = "wrong_typewell"
CONTROL_PAIR_SHUFFLE = "pair_shuffle"
CONTROL_GR_ORDER_SHUFFLE = "gr_order_shuffle"
CONTROL_PHASE_DESTROY = "phase_destroy"
CONTROL_NO_PREFIX_ANCHOR = "no_prefix_anchor"
CONTROL_NO_CROSS_INTERACTION = "no_cross_interaction"
CONTROL_IDS = (
    CONTROL_NONE,
    CONTROL_WRONG_TYPEWELL,
    CONTROL_PAIR_SHUFFLE,
    CONTROL_GR_ORDER_SHUFFLE,
    CONTROL_PHASE_DESTROY,
    CONTROL_NO_PREFIX_ANCHOR,
    CONTROL_NO_CROSS_INTERACTION,
)

METHOD_NCC = "ncc"
METHOD_PARTICLE_FILTER = "particle_filter"
METHOD_BEAM = "beam"
METHOD_IDS = (METHOD_NCC, METHOD_PARTICLE_FILTER, METHOD_BEAM)


def _readonly_vector(
    values: Any,
    *,
    name: str,
    dtype: np.dtype[Any] | type[np.generic] = np.float64,
    finite: bool = True,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SafePhysicsAlignmentConfig:
    """Numerical configuration shared by NCC, PF, and beam candidates."""

    state_step_ft: float = 1.0
    state_radius_ft: float = 60.0
    maximum_state_count: int = 321
    minimum_steps: int = 48
    maximum_steps: int = 128
    rows_per_step: int = 36
    ncc_windows: tuple[int, ...] = (5, 11, 21)
    point_cost_weight: float = 0.15
    maximum_transition_states: int = 2
    slope_penalty: float = 0.08
    curvature_penalty: float = 0.12
    beam_width: int = 32
    beam_ensemble_paths: int = 8
    beam_temperature: float = 0.08
    particles: int = 192
    process_noise_ft: float = 0.35
    velocity_noise_ft: float = 0.12
    velocity_memory: float = 0.85
    observation_temperature: float = 0.16
    resample_ess_fraction: float = 0.45
    minimum_mean_best_ncc: float = -0.05
    smoothing_rows: int = 9

    def __post_init__(self) -> None:
        if not math.isfinite(self.state_step_ft) or self.state_step_ft <= 0.0:
            raise ValueError("state_step_ft must be positive")
        if not math.isfinite(self.state_radius_ft) or self.state_radius_ft <= 0.0:
            raise ValueError("state_radius_ft must be positive")
        if self.maximum_state_count < 16:
            raise ValueError("maximum_state_count is too small")
        if (
            self.minimum_steps < 8
            or self.maximum_steps < self.minimum_steps
            or self.rows_per_step < 1
        ):
            raise ValueError("invalid alignment step limits")
        windows = tuple(map(int, self.ncc_windows))
        if not windows or any(value < 3 or value % 2 == 0 for value in windows):
            raise ValueError("ncc_windows must contain positive odd widths >= 3")
        if not 0.0 <= self.point_cost_weight <= 1.0:
            raise ValueError("point_cost_weight must lie in [0, 1]")
        if self.maximum_transition_states < 1:
            raise ValueError("maximum_transition_states must be positive")
        if self.beam_width < 1 or self.beam_ensemble_paths < 1:
            raise ValueError("beam sizes must be positive")
        if self.particles < 16:
            raise ValueError("particles must be at least 16")
        if not 0.0 < self.velocity_memory < 1.0:
            raise ValueError("velocity_memory must lie in (0, 1)")
        if not 0.0 < self.resample_ess_fraction <= 1.0:
            raise ValueError("resample_ess_fraction must lie in (0, 1]")
        if self.smoothing_rows < 1 or self.smoothing_rows % 2 == 0:
            raise ValueError("smoothing_rows must be a positive odd number")
        object.__setattr__(self, "ncc_windows", windows)


@dataclass(frozen=True)
class SafeAlignmentPrediction:
    """A finite suffix prediction with explicit alignment diagnostics."""

    family_id: str
    method: str
    control_id: str
    well_id: str
    paired_typewell_id: str
    row_index: np.ndarray
    prediction: np.ndarray
    uncertainty: np.ndarray
    effective_particle_count: np.ndarray
    status: str
    reason: str
    confidence: float
    seed_predictions: Mapping[int, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in METHOD_IDS:
            raise ValueError(f"unknown method: {self.method!r}")
        if self.control_id not in CONTROL_IDS:
            raise ValueError(f"unknown control: {self.control_id!r}")
        if self.status not in {ALIGNMENT_STATUS_OK, ALIGNMENT_STATUS_IDENTITY}:
            raise ValueError(f"unknown prediction status: {self.status!r}")
        row_index = _readonly_vector(self.row_index, name="row_index", dtype=np.int64)
        prediction = _readonly_vector(self.prediction, name="prediction")
        uncertainty = _readonly_vector(self.uncertainty, name="uncertainty")
        ess = _readonly_vector(
            self.effective_particle_count,
            name="effective_particle_count",
        )
        if (
            len(row_index) == 0
            or prediction.shape != row_index.shape
            or uncertainty.shape != row_index.shape
            or ess.shape != row_index.shape
            or np.any(np.diff(row_index) != 1)
            or np.any(uncertainty < 0.0)
            or np.any(ess < 0.0)
        ):
            raise ValueError("alignment output arrays are inconsistent")
        seeds: dict[int, np.ndarray] = {}
        for seed, values in self.seed_predictions.items():
            current = _readonly_vector(values, name=f"seed_prediction[{seed}]")
            if current.shape != row_index.shape:
                raise ValueError("seed prediction does not align with row_index")
            seeds[int(seed)] = current
        confidence = float(self.confidence)
        if not math.isfinite(confidence):
            raise ValueError("confidence must be finite")
        object.__setattr__(self, "row_index", row_index)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "effective_particle_count", ess)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "seed_predictions", seeds)


@dataclass(frozen=True)
class _PreparedAlignment:
    well_id: str
    paired_typewell_id: str
    control_id: str
    row_index: np.ndarray
    incumbent: np.ndarray
    suffix_coordinate: np.ndarray
    step_coordinate: np.ndarray
    step_md: np.ndarray
    state_tvt: np.ndarray
    observation_cost: np.ndarray
    mean_best_ncc: float
    anchor_tvt: float
    anchor_state_index: int
    expected_step_ft: float
    use_anchor: bool


def _fill_series(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    finite = np.isfinite(array)
    if int(np.sum(finite)) < 2:
        raise ValueError(f"{name} has fewer than two finite observations")
    coordinate = np.arange(len(array), dtype=np.float64)
    return np.interp(coordinate, coordinate[finite], array[finite])


def _unique_typewell(tvt: np.ndarray, gr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(tvt, kind="stable")
    ordered_tvt = np.asarray(tvt[order], dtype=np.float64)
    ordered_gr = np.asarray(gr[order], dtype=np.float64)
    unique, inverse = np.unique(ordered_tvt, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    mean_gr = np.bincount(inverse, weights=ordered_gr) / counts
    if len(unique) < 16 or np.any(np.diff(unique) <= 0.0):
        raise ValueError("typewell TVT support is insufficient")
    return unique, mean_gr


def _window_matrix(values: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), (half, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    center = np.mean(windows, axis=1, keepdims=True)
    scale = np.std(windows, axis=1, keepdims=True)
    safe = np.where(scale > 1e-8, scale, 1.0)
    normalized = (windows - center) / safe
    normalized[scale[:, 0] <= 1e-8] = 0.0
    return normalized


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    q25, q75 = np.quantile(values, (0.25, 0.75))
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values))
    return center, max(scale, 1e-6)


def _affine_typewell_gr(
    *,
    horizontal_gr: np.ndarray,
    tvt_input: np.ndarray,
    known_count: int,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    use_anchor: bool,
) -> np.ndarray:
    if use_anchor:
        prefix_tvt = tvt_input[:known_count]
        prefix_gr = horizontal_gr[:known_count]
        mask = (
            np.isfinite(prefix_tvt)
            & np.isfinite(prefix_gr)
            & (prefix_tvt >= typewell_tvt[0])
            & (prefix_tvt <= typewell_tvt[-1])
        )
        if int(np.sum(mask)) >= 16:
            reference = np.interp(prefix_tvt[mask], typewell_tvt, typewell_gr)
            design = np.column_stack(
                [reference, np.ones(len(reference), dtype=np.float64)]
            )
            coefficient, intercept = np.linalg.lstsq(
                design, prefix_gr[mask], rcond=None
            )[0]
            coefficient = float(np.clip(coefficient, 0.25, 4.0))
            intercept = float(np.median(prefix_gr[mask] - coefficient * reference))
            calibrated = coefficient * typewell_gr + intercept
            if np.isfinite(calibrated).all():
                return calibrated
    h_center, h_scale = _robust_scale(horizontal_gr)
    t_center, t_scale = _robust_scale(typewell_gr)
    return (typewell_gr - t_center) * (h_scale / t_scale) + h_center


def _phase_destroy(values: np.ndarray, random: np.random.Generator) -> np.ndarray:
    centered = values - float(np.mean(values))
    spectrum = np.fft.rfft(centered)
    if len(spectrum) > 2:
        phase = random.uniform(-np.pi, np.pi, size=len(spectrum) - 2)
        spectrum[1:-1] = np.abs(spectrum[1:-1]) * np.exp(1j * phase)
    result = np.fft.irfft(spectrum, n=len(values))
    return result + float(np.mean(values))


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(values) < 3:
        return np.asarray(values, dtype=np.float64).copy()
    actual = min(width, len(values) if len(values) % 2 else len(values) - 1)
    if actual <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    half = actual // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.full(actual, 1.0 / actual, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _systematic_resample(
    weights: np.ndarray, random: np.random.Generator
) -> np.ndarray:
    positions = (random.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


class SafePhysicsAlignmentFamily:
    """Shared inference-safe NCC, particle-filter, and beam-search family."""

    family_id = "safe_physics_alignment"

    def __init__(
        self, config: SafePhysicsAlignmentConfig = SafePhysicsAlignmentConfig()
    ) -> None:
        if not isinstance(config, SafePhysicsAlignmentConfig):
            raise TypeError("config must be SafePhysicsAlignmentConfig")
        self.config = config

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": ALIGNMENT_SCHEMA_VERSION,
            "family_id": self.family_id,
            "config": asdict(self.config),
            "horizontal_source_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
            "horizontal_columns_used": ["MD", "Z", "GR", "TVT_input"],
            "typewell_source_columns": ["TVT", "GR"],
            "target_columns_used_for_prediction": [],
            "formation_columns_used": [],
            "geology_columns_used": [],
            "image_inputs": False,
            "same_well_train_target_overlap": False,
            "public_lb_parameters": False,
            "fallback": "exact incumbent identity",
        }

    def _identity(
        self,
        *,
        method: str,
        control_id: str,
        well_id: str,
        paired_typewell_id: str,
        row_index: np.ndarray,
        incumbent: np.ndarray,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> SafeAlignmentPrediction:
        return SafeAlignmentPrediction(
            family_id=self.family_id,
            method=method,
            control_id=control_id,
            well_id=well_id,
            paired_typewell_id=paired_typewell_id,
            row_index=row_index,
            prediction=incumbent,
            uncertainty=np.zeros(len(row_index), dtype=np.float64),
            effective_particle_count=np.zeros(len(row_index), dtype=np.float64),
            status=ALIGNMENT_STATUS_IDENTITY,
            reason=reason,
            confidence=0.0,
            metadata={} if metadata is None else dict(metadata),
        )

    def _trajectory_only(
        self,
        *,
        method: str,
        control_id: str,
        well_id: str,
        paired_typewell_id: str,
        horizontal: pd.DataFrame,
        incumbent: np.ndarray,
    ) -> SafeAlignmentPrediction:
        known_count, suffix_count = validate_prefix(horizontal["TVT_input"])
        md = _fill_series(horizontal["MD"], name="horizontal MD")
        tvt_input = horizontal["TVT_input"].to_numpy(dtype=np.float64)
        tail_start = max(0, known_count - min(256, known_count))
        tail_md = md[tail_start:known_count]
        tail_tvt = tvt_input[tail_start:known_count]
        x = tail_md - float(np.mean(tail_md))
        denominator = float(x @ x)
        slope = (
            float(x @ (tail_tvt - float(np.mean(tail_tvt)))) / denominator
            if denominator > 0.0
            else 0.0
        )
        slope = float(np.clip(slope, -0.02, 0.02))
        anchor = float(tvt_input[known_count - 1])
        prediction = anchor + slope * (md[known_count:] - md[known_count - 1])
        if not np.isfinite(prediction).all() or len(prediction) != suffix_count:
            return self._identity(
                method=method,
                control_id=control_id,
                well_id=well_id,
                paired_typewell_id=paired_typewell_id,
                row_index=np.arange(known_count, len(horizontal), dtype=np.int64),
                incumbent=incumbent,
                reason="trajectory-only control failed",
            )
        return SafeAlignmentPrediction(
            family_id=self.family_id,
            method=method,
            control_id=control_id,
            well_id=well_id,
            paired_typewell_id=paired_typewell_id,
            row_index=np.arange(known_count, len(horizontal), dtype=np.int64),
            prediction=prediction,
            uncertainty=np.zeros(suffix_count, dtype=np.float64),
            effective_particle_count=np.zeros(suffix_count, dtype=np.float64),
            status=ALIGNMENT_STATUS_OK,
            reason="trajectory-only control",
            confidence=0.0,
            metadata={"visible_prefix_slope_tvt_per_md": slope},
        )

    def _prepare(
        self,
        *,
        horizontal_frame: pd.DataFrame,
        typewell_frame: pd.DataFrame,
        incumbent_prediction: np.ndarray,
        well_id: str,
        paired_typewell_id: str,
        control_id: str,
        seed: int,
        multiscale: bool,
    ) -> _PreparedAlignment:
        horizontal = select_inference_view(horizontal_frame, kind="horizontal")
        typewell = select_inference_view(typewell_frame, kind="typewell")
        known_count, suffix_count = validate_prefix(horizontal["TVT_input"])
        row_index = np.arange(known_count, len(horizontal), dtype=np.int64)
        incumbent = np.asarray(incumbent_prediction, dtype=np.float64)
        if (
            incumbent.ndim != 1
            or incumbent.shape != row_index.shape
            or not np.isfinite(incumbent).all()
        ):
            raise ValueError("incumbent prediction is not finite and row aligned")

        md = _fill_series(horizontal["MD"], name="horizontal MD")
        horizontal_gr = _fill_series(horizontal["GR"], name="horizontal GR")
        tvt_input = horizontal["TVT_input"].to_numpy(dtype=np.float64)
        tw_tvt_raw = typewell["TVT"].to_numpy(dtype=np.float64)
        tw_gr_raw = _fill_series(typewell["GR"], name="typewell GR")
        finite_tvt = np.isfinite(tw_tvt_raw)
        if int(np.sum(finite_tvt)) < 16:
            raise ValueError("typewell TVT has insufficient finite support")
        tw_tvt, tw_gr = _unique_typewell(tw_tvt_raw[finite_tvt], tw_gr_raw[finite_tvt])

        random = np.random.default_rng(int(seed))
        if control_id == CONTROL_GR_ORDER_SHUFFLE:
            shuffled = horizontal_gr[known_count:].copy()
            random.shuffle(shuffled)
            horizontal_gr[known_count:] = shuffled
        elif control_id == CONTROL_PHASE_DESTROY:
            horizontal_gr[known_count:] = _phase_destroy(
                horizontal_gr[known_count:], random
            )

        use_anchor = control_id != CONTROL_NO_PREFIX_ANCHOR
        calibrated_tw_gr = _affine_typewell_gr(
            horizontal_gr=horizontal_gr,
            tvt_input=tvt_input,
            known_count=known_count,
            typewell_tvt=tw_tvt,
            typewell_gr=tw_gr,
            use_anchor=use_anchor,
        )
        if use_anchor:
            anchor_tvt = float(tvt_input[known_count - 1])
            lower = max(float(tw_tvt[0]), anchor_tvt - self.config.state_radius_ft)
            upper = min(float(tw_tvt[-1]), anchor_tvt + self.config.state_radius_ft)
        else:
            anchor_tvt = float(np.median(tw_tvt))
            lower = float(tw_tvt[0])
            upper = float(tw_tvt[-1])
        if not lower < upper:
            raise ValueError("typewell has no usable TVT support around the anchor")
        state_count = int(math.floor((upper - lower) / self.config.state_step_ft)) + 1
        state_count = min(state_count, self.config.maximum_state_count)
        state_tvt = np.linspace(lower, upper, state_count, dtype=np.float64)
        if len(state_tvt) < 16:
            raise ValueError("alignment state grid is too small")
        anchor_state_index = int(np.argmin(np.abs(state_tvt - anchor_tvt)))

        steps = int(
            np.clip(
                math.ceil(suffix_count / self.config.rows_per_step),
                self.config.minimum_steps,
                self.config.maximum_steps,
            )
        )
        suffix_coordinate = np.linspace(0.0, 1.0, suffix_count, dtype=np.float64)
        step_coordinate = np.linspace(0.0, 1.0, steps, dtype=np.float64)
        suffix_md = md[known_count:]
        step_md = np.interp(step_coordinate, suffix_coordinate, suffix_md)
        suffix_gr = horizontal_gr[known_count:]
        step_gr = np.interp(step_coordinate, suffix_coordinate, suffix_gr)
        state_gr = np.interp(state_tvt, tw_tvt, calibrated_tw_gr)

        h_center, h_scale = _robust_scale(step_gr)
        t_center, t_scale = _robust_scale(state_gr)
        h_standard = (step_gr - h_center) / h_scale
        t_standard = (state_gr - t_center) / t_scale
        point_cost = (
            np.minimum(np.square(h_standard[:, None] - t_standard[None, :]), 9.0) / 9.0
        )

        widths = (
            self.config.ncc_windows
            if multiscale
            else (self.config.ncc_windows[len(self.config.ncc_windows) // 2],)
        )
        correlations: list[np.ndarray] = []
        for requested_width in widths:
            maximum = min(len(step_gr), len(state_gr))
            width = min(int(requested_width), maximum if maximum % 2 else maximum - 1)
            if width < 3:
                continue
            horizontal_windows = _window_matrix(step_gr, width)
            typewell_windows = _window_matrix(state_gr, width)
            current = np.clip(
                horizontal_windows @ typewell_windows.T / width, -1.0, 1.0
            )
            correlations.append(current)
        if not correlations:
            raise ValueError("no valid NCC scale")
        mean_ncc = np.mean(np.stack(correlations), axis=0)
        ncc_cost = 0.5 * (1.0 - mean_ncc)
        weight = self.config.point_cost_weight
        observation_cost = (1.0 - weight) * ncc_cost + weight * point_cost
        mean_best_ncc = float(np.mean(np.max(mean_ncc, axis=1)))

        expected_step_ft = 0.0
        if use_anchor and known_count >= 8:
            tail = min(256, known_count)
            tail_md = md[known_count - tail : known_count]
            tail_tvt = tvt_input[known_count - tail : known_count]
            centered_md = tail_md - float(np.mean(tail_md))
            denominator = float(centered_md @ centered_md)
            slope = (
                float(centered_md @ (tail_tvt - float(np.mean(tail_tvt))) / denominator)
                if denominator > 0.0
                else 0.0
            )
            step_md_delta = float(np.median(np.diff(step_md)))
            expected_step_ft = float(
                np.clip(
                    slope * step_md_delta,
                    -self.config.maximum_transition_states * self.config.state_step_ft,
                    self.config.maximum_transition_states * self.config.state_step_ft,
                )
            )

        return _PreparedAlignment(
            well_id=str(well_id),
            paired_typewell_id=str(paired_typewell_id),
            control_id=str(control_id),
            row_index=row_index,
            incumbent=incumbent,
            suffix_coordinate=suffix_coordinate,
            step_coordinate=step_coordinate,
            step_md=step_md,
            state_tvt=state_tvt,
            observation_cost=observation_cost,
            mean_best_ncc=mean_best_ncc,
            anchor_tvt=anchor_tvt,
            anchor_state_index=anchor_state_index,
            expected_step_ft=expected_step_ft,
            use_anchor=use_anchor,
        )

    def _interpolate_path(
        self, prepared: _PreparedAlignment, step_path: np.ndarray
    ) -> np.ndarray:
        path = np.asarray(step_path, dtype=np.float64)
        if path.shape != prepared.step_coordinate.shape:
            raise ValueError("alignment path shape mismatch")
        path = _smooth(path, self.config.smoothing_rows)
        result = np.interp(prepared.suffix_coordinate, prepared.step_coordinate, path)
        if prepared.use_anchor:
            start_delta = float(result[0] - prepared.anchor_tvt)
            taper = np.exp(-prepared.suffix_coordinate * 12.0)
            result = result - start_delta * taper
        result = np.clip(
            result, float(prepared.state_tvt[0]), float(prepared.state_tvt[-1])
        )
        if not np.isfinite(result).all():
            raise ValueError("interpolated alignment path is non-finite")
        return result

    def _monotone_dp_branch(
        self, prepared: _PreparedAlignment, direction: int
    ) -> tuple[np.ndarray, float]:
        if direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1")
        cost = prepared.observation_cost
        steps, states = cost.shape
        maximum_delta = self.config.maximum_transition_states
        expected_states = abs(prepared.expected_step_ft) / max(
            self.config.state_step_ft, 1e-12
        )
        previous = np.full(states, np.inf, dtype=np.float64)
        if prepared.use_anchor:
            previous[prepared.anchor_state_index] = 0.0
        else:
            previous[:] = cost[0]
        back = np.full((steps, states), -1, dtype=np.int16)
        start = 0 if prepared.use_anchor else 1
        for step_index in range(start, steps):
            current = np.full(states, np.inf, dtype=np.float64)
            current_back = np.full(states, -1, dtype=np.int16)
            for delta in range(maximum_delta + 1):
                shift = direction * delta
                source_start = max(0, -shift)
                source_stop = min(states, states - shift)
                target_start = source_start + shift
                target_stop = source_stop + shift
                if source_start >= source_stop:
                    continue
                transition = self.config.slope_penalty * (delta - expected_states) ** 2
                candidate = previous[source_start:source_stop] + transition
                existing = current[target_start:target_stop]
                improve = candidate < existing
                existing[improve] = candidate[improve]
                current_back[target_start:target_stop][improve] = np.arange(
                    source_start, source_stop, dtype=np.int16
                )[improve]
            current += cost[step_index]
            previous = current
            back[step_index] = current_back
        final_state = int(np.argmin(previous))
        if not math.isfinite(float(previous[final_state])):
            raise ValueError("monotone NCC path is unreachable")
        indices = np.empty(steps, dtype=np.int64)
        indices[-1] = final_state
        for step_index in range(steps - 1, start, -1):
            parent = int(back[step_index, indices[step_index]])
            if parent < 0:
                raise ValueError("monotone NCC backtrace is incomplete")
            indices[step_index - 1] = parent
        if prepared.use_anchor:
            indices[0] = prepared.anchor_state_index
        score = float(previous[final_state] / steps)
        return prepared.state_tvt[indices], score

    def _ncc_path(self, prepared: _PreparedAlignment) -> tuple[np.ndarray, float]:
        candidates = [
            self._monotone_dp_branch(prepared, -1),
            self._monotone_dp_branch(prepared, 1),
        ]
        return min(candidates, key=lambda item: item[1])

    def _beam_paths(
        self, prepared: _PreparedAlignment
    ) -> list[tuple[float, np.ndarray]]:
        cost = prepared.observation_cost
        steps, states = cost.shape
        expected_states = abs(prepared.expected_step_ft) / max(
            self.config.state_step_ft, 1e-12
        )
        completed: list[tuple[float, np.ndarray]] = []
        for direction in (-1, 1):
            if prepared.use_anchor:
                beams: list[tuple[float, int, int, tuple[int, ...]]] = [
                    (0.0, prepared.anchor_state_index, 0, ())
                ]
            else:
                initial = np.argsort(cost[0], kind="stable")[: self.config.beam_width]
                beams = [
                    (float(cost[0, index]), int(index), 0, (int(index),))
                    for index in initial
                ]
            start = 0 if prepared.use_anchor else 1
            for step_index in range(start, steps):
                expanded: list[tuple[float, int, int, tuple[int, ...]]] = []
                for total, state, previous_delta, path in beams:
                    for delta in range(self.config.maximum_transition_states + 1):
                        next_state = state + direction * delta
                        if not 0 <= next_state < states:
                            continue
                        transition = (
                            self.config.slope_penalty * (delta - expected_states) ** 2
                            + self.config.curvature_penalty
                            * (delta - previous_delta) ** 2
                        )
                        expanded.append(
                            (
                                total
                                + float(cost[step_index, next_state])
                                + transition,
                                next_state,
                                delta,
                                (*path, next_state),
                            )
                        )
                if not expanded:
                    break
                expanded.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
                beams = expanded[: self.config.beam_width]
            for total, _, _, path in beams:
                if len(path) == steps:
                    completed.append(
                        (
                            float(total / steps),
                            prepared.state_tvt[np.asarray(path, dtype=np.int64)],
                        )
                    )
        if not completed:
            raise ValueError("beam search returned no complete path")
        completed.sort(key=lambda item: item[0])
        return completed

    def _particle_path(
        self, prepared: _PreparedAlignment, *, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        random = np.random.default_rng(int(seed))
        count = self.config.particles
        lower = float(prepared.state_tvt[0])
        upper = float(prepared.state_tvt[-1])
        if prepared.use_anchor:
            position = prepared.anchor_tvt + random.normal(0.0, 0.5, count)
        else:
            position = random.uniform(lower, upper, count)
        position = np.clip(position, lower, upper)
        velocity = np.full(count, prepared.expected_step_ft, dtype=np.float64)
        weights = np.full(count, 1.0 / count, dtype=np.float64)
        mean_path = np.empty(len(prepared.step_coordinate), dtype=np.float64)
        std_path = np.empty_like(mean_path)
        ess_path = np.empty_like(mean_path)
        for step_index in range(len(mean_path)):
            if step_index > 0 or not prepared.use_anchor:
                velocity = (
                    self.config.velocity_memory * velocity
                    + (1.0 - self.config.velocity_memory) * prepared.expected_step_ft
                    + random.normal(0.0, self.config.velocity_noise_ft, count)
                )
                position = (
                    position
                    + velocity
                    + random.normal(0.0, self.config.process_noise_ft, count)
                )
                position = np.clip(position, lower, upper)
            observation = np.interp(
                position,
                prepared.state_tvt,
                prepared.observation_cost[step_index],
            )
            log_weight = np.log(np.maximum(weights, 1e-300)) - (
                observation / self.config.observation_temperature
            )
            log_weight -= float(np.max(log_weight))
            weights = np.exp(log_weight)
            total = float(np.sum(weights))
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError("particle weights collapsed")
            weights /= total
            mean = float(weights @ position)
            variance = float(weights @ np.square(position - mean))
            ess = float(1.0 / (weights @ weights))
            mean_path[step_index] = mean
            std_path[step_index] = math.sqrt(max(variance, 0.0))
            ess_path[step_index] = ess
            if ess < self.config.resample_ess_fraction * count:
                indices = _systematic_resample(weights, random)
                position = position[indices]
                velocity = velocity[indices]
                weights.fill(1.0 / count)
        return mean_path, std_path, ess_path

    def predict(
        self,
        *,
        method: str,
        horizontal_frame: pd.DataFrame,
        typewell_frame: pd.DataFrame,
        incumbent_prediction: np.ndarray,
        well_id: str,
        paired_typewell_id: str | None = None,
        control_id: str = CONTROL_NONE,
        seeds: Sequence[int] = (20260711,),
        multiscale: bool = True,
        beam_ensemble: bool = True,
    ) -> SafeAlignmentPrediction:
        """Predict one suffix without ever reading the train target column."""

        method = str(method)
        control_id = str(control_id)
        well_id = str(well_id)
        paired = well_id if paired_typewell_id is None else str(paired_typewell_id)
        if method not in METHOD_IDS:
            raise ValueError(f"method must be one of {METHOD_IDS}")
        if control_id not in CONTROL_IDS:
            raise ValueError(f"control_id must be one of {CONTROL_IDS}")
        seed_tuple = tuple(map(int, seeds))
        if not seed_tuple or len(set(seed_tuple)) != len(seed_tuple):
            raise ValueError("seeds must be non-empty and unique")

        horizontal = select_inference_view(horizontal_frame, kind="horizontal")
        known_count, _ = validate_prefix(horizontal["TVT_input"])
        row_index = np.arange(known_count, len(horizontal), dtype=np.int64)
        incumbent = np.asarray(incumbent_prediction, dtype=np.float64)
        if control_id == CONTROL_NO_CROSS_INTERACTION:
            return self._trajectory_only(
                method=method,
                control_id=control_id,
                well_id=well_id,
                paired_typewell_id=paired,
                horizontal=horizontal,
                incumbent=incumbent,
            )

        try:
            control_seed_payload = (
                f"{well_id}\0{paired}\0{control_id}\0{seed_tuple[0]}"
            ).encode("utf-8")
            control_seed = int.from_bytes(
                hashlib.sha256(control_seed_payload).digest()[:8], "little"
            )
            prepared = self._prepare(
                horizontal_frame=horizontal,
                typewell_frame=typewell_frame,
                incumbent_prediction=incumbent,
                well_id=well_id,
                paired_typewell_id=paired,
                control_id=control_id,
                seed=control_seed,
                multiscale=bool(multiscale),
            )
            if (
                not math.isfinite(prepared.mean_best_ncc)
                or prepared.mean_best_ncc < self.config.minimum_mean_best_ncc
            ):
                return self._identity(
                    method=method,
                    control_id=control_id,
                    well_id=well_id,
                    paired_typewell_id=paired,
                    row_index=row_index,
                    incumbent=incumbent,
                    reason=(
                        "alignment confidence below threshold: "
                        f"{prepared.mean_best_ncc:.6f}"
                    ),
                    metadata={"mean_best_ncc": prepared.mean_best_ncc},
                )

            if method == METHOD_NCC:
                step_path, score = self._ncc_path(prepared)
                prediction = self._interpolate_path(prepared, step_path)
                uncertainty = np.zeros(len(prediction), dtype=np.float64)
                ess = np.zeros(len(prediction), dtype=np.float64)
                seed_predictions: dict[int, np.ndarray] = {}
                metadata = {
                    "path_cost": score,
                    "multiscale": bool(multiscale),
                    "monotonic_direction": int(np.sign(step_path[-1] - step_path[0])),
                }
            elif method == METHOD_BEAM:
                paths = self._beam_paths(prepared)
                selected = paths[
                    : self.config.beam_ensemble_paths if beam_ensemble else 1
                ]
                scores = np.asarray([item[0] for item in selected], dtype=np.float64)
                weights = np.exp(
                    -(scores - float(np.min(scores))) / self.config.beam_temperature
                )
                weights /= float(np.sum(weights))
                step_path = np.sum(
                    np.stack([item[1] for item in selected]) * weights[:, None],
                    axis=0,
                )
                prediction = self._interpolate_path(prepared, step_path)
                uncertainty_steps = np.sqrt(
                    np.sum(
                        weights[:, None]
                        * np.square(
                            np.stack([item[1] for item in selected])
                            - step_path[None, :]
                        ),
                        axis=0,
                    )
                )
                uncertainty = np.interp(
                    prepared.suffix_coordinate,
                    prepared.step_coordinate,
                    uncertainty_steps,
                )
                ess = np.zeros(len(prediction), dtype=np.float64)
                seed_predictions = {}
                metadata = {
                    "best_path_cost": float(scores[0]),
                    "ensemble_paths": len(selected),
                    "multiscale": bool(multiscale),
                }
            else:
                seed_predictions = {}
                step_predictions: list[np.ndarray] = []
                step_uncertainties: list[np.ndarray] = []
                step_ess: list[np.ndarray] = []
                for seed in seed_tuple:
                    step_path, step_std, current_ess = self._particle_path(
                        prepared, seed=seed
                    )
                    current_prediction = self._interpolate_path(prepared, step_path)
                    seed_predictions[seed] = current_prediction
                    step_predictions.append(step_path)
                    step_uncertainties.append(step_std)
                    step_ess.append(current_ess)
                ensemble_step = np.mean(np.stack(step_predictions), axis=0)
                prediction = self._interpolate_path(prepared, ensemble_step)
                within = np.mean(np.stack(step_uncertainties), axis=0)
                between = np.std(np.stack(step_predictions), axis=0)
                uncertainty = np.interp(
                    prepared.suffix_coordinate,
                    prepared.step_coordinate,
                    np.sqrt(np.square(within) + np.square(between)),
                )
                ess = np.interp(
                    prepared.suffix_coordinate,
                    prepared.step_coordinate,
                    np.mean(np.stack(step_ess), axis=0),
                )
                metadata = {
                    "seeds": list(seed_tuple),
                    "particles": self.config.particles,
                    "multiscale": bool(multiscale),
                    "mean_uncertainty_ft": float(np.mean(uncertainty)),
                    "mean_effective_particle_count": float(np.mean(ess)),
                }

            if (
                prediction.shape != incumbent.shape
                or uncertainty.shape != incumbent.shape
                or ess.shape != incumbent.shape
                or not np.isfinite(prediction).all()
                or not np.isfinite(uncertainty).all()
                or not np.isfinite(ess).all()
            ):
                raise ValueError("alignment output is incomplete or non-finite")
            return SafeAlignmentPrediction(
                family_id=self.family_id,
                method=method,
                control_id=control_id,
                well_id=well_id,
                paired_typewell_id=paired,
                row_index=prepared.row_index,
                prediction=prediction,
                uncertainty=uncertainty,
                effective_particle_count=ess,
                status=ALIGNMENT_STATUS_OK,
                reason="",
                confidence=prepared.mean_best_ncc,
                seed_predictions=seed_predictions,
                metadata=metadata,
            )
        except Exception as error:
            return self._identity(
                method=method,
                control_id=control_id,
                well_id=well_id,
                paired_typewell_id=paired,
                row_index=row_index,
                incumbent=incumbent,
                reason=f"{type(error).__name__}: {error}",
            )


__all__ = [
    "ALIGNMENT_SCHEMA_VERSION",
    "ALIGNMENT_STATUS_IDENTITY",
    "ALIGNMENT_STATUS_OK",
    "CONTROL_GR_ORDER_SHUFFLE",
    "CONTROL_IDS",
    "CONTROL_NONE",
    "CONTROL_NO_CROSS_INTERACTION",
    "CONTROL_NO_PREFIX_ANCHOR",
    "CONTROL_PAIR_SHUFFLE",
    "CONTROL_PHASE_DESTROY",
    "CONTROL_WRONG_TYPEWELL",
    "METHOD_BEAM",
    "METHOD_IDS",
    "METHOD_NCC",
    "METHOD_PARTICLE_FILTER",
    "SafeAlignmentPrediction",
    "SafePhysicsAlignmentConfig",
    "SafePhysicsAlignmentFamily",
]
