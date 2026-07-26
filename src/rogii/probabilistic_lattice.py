"""Exact, bounded probabilistic TVT-path inference.

The lattice treats measured depth (MD) as time and physical TVT as a discrete
state.  Its transition band contains negative, zero, and positive TVT moves.
Horizontal GR is compared with the well's own typewell GR after a deterministic
affine calibration from the visible ``TVT_input`` prefix.

The implementation is deliberately small and dependency-free (beyond NumPy).
It performs exact sum-product and max-product inference for the finite banded
factor graph:

    p(s_1:T | x) proportional to
        psi(anchor, s_1) phi(x_1, s_1)
        product_t psi(s_t, s_t+1) phi(x_t+1, s_t+1)

The returned posterior mean is the minimum-Bayes-risk estimate for row-wise
squared loss.  The Viterbi path is also returned for diagnostics, but is not
the default prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LatticeConfig:
    """Numerical and robustness settings for a finite signed-TVT lattice."""

    state_step_ft: float = 2.0
    state_radius_ft: float = 96.0
    max_move_bins: int = 3
    transition_df: float = 4.0
    transition_rate_scale_floor: float = 0.02
    prefix_rate_shrinkage: float = 20.0
    emission_df: float = 3.0
    emission_scale_floor: float = 1.0
    emission_floor_probability: float = 0.02
    emission_chunk_rows: int = 4096
    calibration_slope_min: float = 0.25
    calibration_slope_max: float = 4.0
    edge_bins: int = 2

    def __post_init__(self) -> None:
        numeric = (
            self.state_step_ft,
            self.state_radius_ft,
            self.transition_df,
            self.transition_rate_scale_floor,
            self.prefix_rate_shrinkage,
            self.emission_df,
            self.emission_scale_floor,
            self.emission_floor_probability,
            self.calibration_slope_min,
            self.calibration_slope_max,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("Lattice numeric configuration must be finite")
        if self.state_step_ft <= 0 or self.state_radius_ft < self.state_step_ft:
            raise ValueError("State step and radius must define at least three states")
        radius_bins = self.state_radius_ft / self.state_step_ft
        if not np.isclose(radius_bins, round(radius_bins), rtol=0.0, atol=1e-9):
            raise ValueError("state_radius_ft must be an integer multiple of state_step_ft")
        if (
            isinstance(self.max_move_bins, bool)
            or not isinstance(self.max_move_bins, (int, np.integer))
            or self.max_move_bins < 1
            or self.max_move_bins > int(round(radius_bins))
        ):
            raise ValueError("max_move_bins must allow negative and positive moves")
        if self.transition_df <= 0 or self.transition_rate_scale_floor <= 0:
            raise ValueError("Invalid transition distribution")
        if self.prefix_rate_shrinkage < 0:
            raise ValueError("prefix_rate_shrinkage cannot be negative")
        if self.emission_df <= 0 or self.emission_scale_floor <= 0:
            raise ValueError("Invalid emission distribution")
        if not 0 < self.emission_floor_probability <= 0.25:
            raise ValueError(
                "emission_floor_probability must be in the interval (0, 0.25]"
            )
        if (
            isinstance(self.emission_chunk_rows, bool)
            or not isinstance(self.emission_chunk_rows, (int, np.integer))
            or self.emission_chunk_rows < 1
        ):
            raise ValueError("emission_chunk_rows must be a positive integer")
        if (
            self.calibration_slope_min <= 0
            or self.calibration_slope_max < self.calibration_slope_min
        ):
            raise ValueError("Invalid calibration slope bounds")
        if (
            isinstance(self.edge_bins, bool)
            or not isinstance(self.edge_bins, (int, np.integer))
            or self.edge_bins < 1
            or self.edge_bins > int(round(radius_bins))
        ):
            raise ValueError("edge_bins must be a positive non-overlapping edge width")


@dataclass(frozen=True)
class PrefixCalibration:
    """Inference-visible, per-well calibration learned only from the prefix."""

    gr_intercept: float
    gr_slope: float
    gr_pair_count: int
    rate_location: float
    rate_count: int

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.gr_intercept)
            or not np.isfinite(self.gr_slope)
            or self.gr_slope <= 0
            or not np.isfinite(self.rate_location)
        ):
            raise ValueError("Prefix calibration values must be finite and valid")
        if (
            isinstance(self.gr_pair_count, bool)
            or not isinstance(self.gr_pair_count, (int, np.integer))
            or self.gr_pair_count < 0
            or isinstance(self.rate_count, bool)
            or not isinstance(self.rate_count, (int, np.integer))
            or self.rate_count < 0
        ):
            raise ValueError("Prefix calibration counts must be non-negative integers")


@dataclass(frozen=True)
class LatticeStatistics:
    """Fold-training-only transition and emission statistics."""

    transition_rate_location: float
    transition_rate_scale: float
    transition_sample_count: int
    emission_residual_location: float
    emission_residual_scale: float
    emission_floor_probability: float
    emission_sample_count: int

    def __post_init__(self) -> None:
        numeric = (
            self.transition_rate_location,
            self.transition_rate_scale,
            self.emission_residual_location,
            self.emission_residual_scale,
            self.emission_floor_probability,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("Lattice statistics must be finite")
        if self.transition_rate_scale <= 0 or self.emission_residual_scale <= 0:
            raise ValueError("Lattice statistic scales must be positive")
        if not 0 < self.emission_floor_probability <= 0.25:
            raise ValueError(
                "Lattice emission floor probability must be in the interval (0, 0.25]"
            )
        for count in (
            self.transition_sample_count,
            self.emission_sample_count,
        ):
            if (
                isinstance(count, bool)
                or not isinstance(count, (int, np.integer))
                or count < 0
            ):
                raise ValueError(
                    "Lattice statistic counts must be non-negative integers"
                )


@dataclass(frozen=True)
class LatticeResult:
    """Exact posterior, MBR prediction, MAP path, and uncertainty diagnostics."""

    state_grid: np.ndarray
    posterior: np.ndarray
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    map_path: np.ndarray
    marginal_map: np.ndarray
    entropy: np.ndarray
    edge_mass: np.ndarray
    log_evidence: float


def _as_float_vector(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _validate_coordinate(values: np.ndarray, name: str) -> None:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain at least two finite values")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")


def _robust_location_scale(
    values: Iterable[float] | np.ndarray, *, scale_floor: float
) -> tuple[float, float, int]:
    array = _as_float_vector(values, "robust-statistic values")
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return 0.0, float(scale_floor), 0
    location = float(np.median(finite))
    mad_scale = 1.4826 * float(np.median(np.abs(finite - location)))
    if mad_scale <= scale_floor:
        q25, q75 = np.percentile(finite, [25.0, 75.0])
        iqr_scale = float(q75 - q25) / 1.349
    else:
        iqr_scale = 0.0
    if max(mad_scale, iqr_scale) <= scale_floor and len(finite) > 1:
        standard_scale = float(np.std(finite))
    else:
        standard_scale = 0.0
    scale = max(mad_scale, iqr_scale, standard_scale, float(scale_floor))
    return location, scale, int(len(finite))


def interpolate_typewell_gr(
    typewell_tvt: Iterable[float] | np.ndarray,
    typewell_gr: Iterable[float] | np.ndarray,
    query_tvt: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Interpolate finite typewell GR without extrapolating beyond support."""

    coordinate = _as_float_vector(typewell_tvt, "typewell_tvt")
    values = _as_float_vector(typewell_gr, "typewell_gr")
    query = np.asarray(query_tvt, dtype=np.float64)
    if len(coordinate) != len(values):
        raise ValueError("typewell_tvt and typewell_gr lengths differ")
    _validate_coordinate(coordinate, "typewell_tvt")
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return np.full(query.shape, np.nan, dtype=np.float64)
    return np.interp(
        query,
        coordinate[finite],
        values[finite],
        left=np.nan,
        right=np.nan,
    ).astype(np.float64, copy=False)


def derive_prefix_calibration(
    md: Iterable[float] | np.ndarray,
    horizontal_gr: Iterable[float] | np.ndarray,
    tvt_input: Iterable[float] | np.ndarray,
    typewell_tvt: Iterable[float] | np.ndarray,
    typewell_gr: Iterable[float] | np.ndarray,
    *,
    config: LatticeConfig,
) -> PrefixCalibration:
    """Derive deterministic GR and signed-rate calibration from known rows only."""

    md_array = _as_float_vector(md, "md")
    gr_array = _as_float_vector(horizontal_gr, "horizontal_gr")
    input_array = _as_float_vector(tvt_input, "tvt_input")
    if not (len(md_array) == len(gr_array) == len(input_array)):
        raise ValueError("md, horizontal_gr, and tvt_input lengths differ")
    _validate_coordinate(md_array, "md")
    known = np.isfinite(input_array)
    known_count = int(known.sum())
    if known_count < 1 or known_count == len(input_array):
        raise ValueError("tvt_input must have a non-empty known prefix and suffix")
    if not np.array_equal(known, np.arange(len(input_array)) < known_count):
        raise ValueError("tvt_input must be one contiguous known prefix")

    known_tvt = input_array[:known_count]
    if not np.isfinite(known_tvt).all():
        raise ValueError("Known TVT prefix must be finite")
    paired_type_gr = interpolate_typewell_gr(
        typewell_tvt, typewell_gr, known_tvt
    )
    paired = np.isfinite(gr_array[:known_count]) & np.isfinite(paired_type_gr)
    pair_count = int(paired.sum())
    if pair_count == 0:
        slope = 1.0
        intercept = 0.0
    else:
        observed = gr_array[:known_count][paired]
        reference = paired_type_gr[paired]
        observed_center = float(np.median(observed))
        reference_center = float(np.median(reference))
        if pair_count >= 3:
            observed_scale = 1.4826 * float(
                np.median(np.abs(observed - observed_center))
            )
            reference_scale = 1.4826 * float(
                np.median(np.abs(reference - reference_center))
            )
        else:
            observed_scale = 0.0
            reference_scale = 0.0
        if observed_scale > 1e-9 and reference_scale > 1e-9:
            slope = float(
                np.clip(
                    observed_scale / reference_scale,
                    config.calibration_slope_min,
                    config.calibration_slope_max,
                )
            )
        else:
            slope = 1.0
        intercept = float(np.median(observed - slope * reference))

    if known_count >= 2:
        delta_md = np.diff(md_array[:known_count])
        rates = np.diff(known_tvt) / delta_md
        finite_rates = rates[np.isfinite(rates)]
    else:
        finite_rates = np.empty(0, dtype=np.float64)
    rate_location = float(np.median(finite_rates)) if len(finite_rates) else 0.0
    return PrefixCalibration(
        gr_intercept=intercept,
        gr_slope=slope,
        gr_pair_count=pair_count,
        rate_location=rate_location,
        rate_count=int(len(finite_rates)),
    )


def fit_lattice_statistics(
    transition_rates: Iterable[float] | np.ndarray,
    emission_residuals: Iterable[float] | np.ndarray,
    *,
    config: LatticeConfig,
) -> LatticeStatistics:
    """Fit robust scalar distributions using outer-fold training samples only."""

    transition_location, transition_scale, transition_count = (
        _robust_location_scale(
            transition_rates, scale_floor=config.transition_rate_scale_floor
        )
    )
    emission_location, emission_scale, emission_count = _robust_location_scale(
        emission_residuals, scale_floor=config.emission_scale_floor
    )
    residual_array = _as_float_vector(emission_residuals, "emission_residuals")
    finite_residual = residual_array[np.isfinite(residual_array)]
    if len(finite_residual):
        standardized = np.abs(
            (finite_residual - emission_location) / emission_scale
        )
        empirical_outlier = float(np.mean(standardized > 3.0))
        floor_probability = float(
            np.clip(
                max(config.emission_floor_probability, empirical_outlier),
                config.emission_floor_probability,
                0.25,
            )
        )
    else:
        floor_probability = config.emission_floor_probability
    return LatticeStatistics(
        transition_rate_location=transition_location,
        transition_rate_scale=transition_scale,
        transition_sample_count=transition_count,
        emission_residual_location=emission_location,
        emission_residual_scale=emission_scale,
        emission_floor_probability=floor_probability,
        emission_sample_count=emission_count,
    )


def build_state_grid(anchor: float, *, config: LatticeConfig) -> np.ndarray:
    """Return a symmetric finite grid containing the anchor exactly."""

    if not np.isfinite(anchor):
        raise ValueError("anchor must be finite")
    radius_bins = int(round(config.state_radius_ft / config.state_step_ft))
    offsets = np.arange(-radius_bins, radius_bins + 1, dtype=np.float64)
    return float(anchor) + config.state_step_ft * offsets


def blended_transition_location(
    statistics: LatticeStatistics,
    calibration: PrefixCalibration,
    *,
    config: LatticeConfig,
) -> float:
    """Shrink a visible-prefix rate toward the fold-training global rate."""

    if calibration.rate_count <= 0:
        return statistics.transition_rate_location
    denominator = calibration.rate_count + config.prefix_rate_shrinkage
    if denominator <= 0:
        return calibration.rate_location
    prefix_weight = calibration.rate_count / denominator
    return float(
        prefix_weight * calibration.rate_location
        + (1.0 - prefix_weight) * statistics.transition_rate_location
    )


def transition_log_weights(
    delta_md: float,
    statistics: LatticeStatistics,
    calibration: PrefixCalibration,
    *,
    config: LatticeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized log weights for every signed move in the band."""

    if not np.isfinite(delta_md) or delta_md <= 0:
        raise ValueError("delta_md must be finite and positive")
    moves = np.arange(
        -config.max_move_bins, config.max_move_bins + 1, dtype=np.int64
    )
    rates = moves.astype(np.float64) * config.state_step_ft / float(delta_md)
    location = blended_transition_location(
        statistics, calibration, config=config
    )
    scale = max(
        float(statistics.transition_rate_scale),
        config.transition_rate_scale_floor,
    )
    z = (rates - location) / scale
    log_weight = -0.5 * (config.transition_df + 1.0) * np.log1p(
        z * z / config.transition_df
    )
    maximum = float(np.max(log_weight))
    log_normalizer = maximum + math.log(
        float(np.sum(np.exp(log_weight - maximum)))
    )
    return moves, log_weight - log_normalizer


def build_emission_log_potential(
    horizontal_gr: Iterable[float] | np.ndarray,
    state_grid: Iterable[float] | np.ndarray,
    typewell_tvt: Iterable[float] | np.ndarray,
    typewell_gr: Iterable[float] | np.ndarray,
    statistics: LatticeStatistics,
    calibration: PrefixCalibration,
    *,
    config: LatticeConfig,
) -> np.ndarray:
    """Build robust GR emission factors; missing observations are neutral."""

    observed = _as_float_vector(horizontal_gr, "horizontal_gr")
    states = _as_float_vector(state_grid, "state_grid")
    reference = interpolate_typewell_gr(typewell_tvt, typewell_gr, states)
    expected = calibration.gr_intercept + calibration.gr_slope * reference
    reference_valid = np.isfinite(expected)[None, :]
    scale = max(
        float(statistics.emission_residual_scale), config.emission_scale_floor
    )
    floor_probability = float(
        np.clip(
            statistics.emission_floor_probability,
            config.emission_floor_probability,
            0.25,
        )
    )
    log_floor = math.log(floor_probability)
    output = np.zeros((len(observed), len(states)), dtype=np.float64)
    for begin in range(0, len(observed), config.emission_chunk_rows):
        end = min(begin + config.emission_chunk_rows, len(observed))
        observed_chunk = observed[begin:end]
        observed_valid = np.isfinite(observed_chunk)[:, None]
        valid = observed_valid & reference_valid
        residual = observed_chunk[:, None] - expected[None, :]
        safe_residual = np.where(
            valid, residual, statistics.emission_residual_location
        )
        z = (safe_residual - statistics.emission_residual_location) / scale
        core = -0.5 * (config.emission_df + 1.0) * np.log1p(
            z * z / config.emission_df
        )
        robust = np.logaddexp(
            math.log1p(-floor_probability) + core,
            log_floor,
        )
        block = output[begin:end]
        # A missing horizontal observation is neutral because it contains no
        # evidence.  A state lacking typewell support receives only the robust
        # contamination floor, preventing unsupported states from being more
        # attractive than a well-matched supported state.
        block[observed_valid & ~reference_valid] = log_floor
        block[valid] = robust[valid]
    return output


def _banded_forward_step(
    previous: np.ndarray, moves: np.ndarray, log_weights: np.ndarray
) -> np.ndarray:
    state_count = len(previous)
    output = np.full(state_count, -np.inf, dtype=np.float64)
    for move, log_weight in zip(moves, log_weights, strict=True):
        delta = int(move)
        if delta >= 0:
            source = slice(0, state_count - delta)
            destination = slice(delta, state_count)
        else:
            source = slice(-delta, state_count)
            destination = slice(0, state_count + delta)
        output[destination] = np.logaddexp(
            output[destination], previous[source] + log_weight
        )
    return output


def _banded_backward_step(
    following: np.ndarray,
    following_emission: np.ndarray,
    moves: np.ndarray,
    log_weights: np.ndarray,
) -> np.ndarray:
    state_count = len(following)
    output = np.full(state_count, -np.inf, dtype=np.float64)
    future = following + following_emission
    for move, log_weight in zip(moves, log_weights, strict=True):
        delta = int(move)
        if delta >= 0:
            source = slice(0, state_count - delta)
            destination = slice(delta, state_count)
        else:
            source = slice(-delta, state_count)
            destination = slice(0, state_count + delta)
        output[source] = np.logaddexp(
            output[source], future[destination] + log_weight
        )
    return output


def _logsumexp(vector: np.ndarray) -> float:
    maximum = float(np.max(vector))
    if not np.isfinite(maximum):
        return -np.inf
    return maximum + math.log(float(np.sum(np.exp(vector - maximum))))


def _viterbi_path(
    emission: np.ndarray,
    delta_md: np.ndarray,
    anchor_index: int,
    statistics: LatticeStatistics,
    calibration: PrefixCalibration,
    *,
    config: LatticeConfig,
) -> np.ndarray:
    row_count, state_count = emission.shape
    backpointer = np.full((row_count, state_count), -1, dtype=np.int32)
    moves, weights = transition_log_weights(
        delta_md[0], statistics, calibration, config=config
    )
    score = np.full(state_count, -np.inf, dtype=np.float64)
    for move, weight in zip(moves, weights, strict=True):
        state = anchor_index + int(move)
        if 0 <= state < state_count:
            score[state] = weight + emission[0, state]

    for row in range(1, row_count):
        moves, weights = transition_log_weights(
            delta_md[row], statistics, calibration, config=config
        )
        next_score = np.full(state_count, -np.inf, dtype=np.float64)
        for move, weight in zip(moves, weights, strict=True):
            delta = int(move)
            if delta >= 0:
                source = np.arange(0, state_count - delta, dtype=np.int32)
                destination = source + delta
            else:
                destination = np.arange(0, state_count + delta, dtype=np.int32)
                source = destination - delta
            candidate = score[source] + weight
            better = candidate > next_score[destination]
            selected_destination = destination[better]
            next_score[selected_destination] = candidate[better]
            backpointer[row, selected_destination] = source[better]
        score = next_score + emission[row]

    path_index = np.empty(row_count, dtype=np.int32)
    path_index[-1] = int(np.argmax(score))
    for row in range(row_count - 1, 0, -1):
        predecessor = backpointer[row, path_index[row]]
        if predecessor < 0:
            raise RuntimeError("Viterbi backtrace encountered an unreachable state")
        path_index[row - 1] = predecessor
    return path_index


def _prepare_inference(
    prediction_md: Iterable[float] | np.ndarray,
    horizontal_gr: Iterable[float] | np.ndarray,
    *,
    anchor_md: float,
    anchor_tvt: float,
    typewell_tvt: Iterable[float] | np.ndarray,
    typewell_gr: Iterable[float] | np.ndarray,
    statistics: LatticeStatistics,
    calibration: PrefixCalibration,
    config: LatticeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    md = _as_float_vector(prediction_md, "prediction_md")
    observed = _as_float_vector(horizontal_gr, "horizontal_gr")
    if len(md) == 0 or len(md) != len(observed):
        raise ValueError("prediction_md and horizontal_gr must have equal non-zero length")
    if not np.isfinite(anchor_md):
        raise ValueError("anchor_md must be finite")
    complete_md = np.concatenate(([float(anchor_md)], md))
    if not np.isfinite(complete_md).all() or np.any(np.diff(complete_md) <= 0):
        raise ValueError("Prediction MD must be finite, increasing, and after anchor_md")
    delta_md = np.diff(complete_md)
    states = build_state_grid(anchor_tvt, config=config)
    anchor_index = len(states) // 2
    emission = build_emission_log_potential(
        observed,
        states,
        typewell_tvt,
        typewell_gr,
        statistics,
        calibration,
        config=config,
    )
    return states, emission, delta_md, anchor_index


def _forward_messages(
    emission: np.ndarray,
    delta_md: np.ndarray,
    anchor_index: int,
    statistics: LatticeStatistics,
    calibration: PrefixCalibration,
    *,
    config: LatticeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    row_count, state_count = emission.shape
    alpha = np.full((row_count, state_count), -np.inf, dtype=np.float64)
    log_scales = np.empty(row_count, dtype=np.float64)
    moves, weights = transition_log_weights(
        delta_md[0], statistics, calibration, config=config
    )
    for move, weight in zip(moves, weights, strict=True):
        state = anchor_index + int(move)
        if 0 <= state < state_count:
            alpha[0, state] = weight + emission[0, state]
    log_scales[0] = _logsumexp(alpha[0])
    if not np.isfinite(log_scales[0]):
        raise RuntimeError("No reachable first-row lattice state")
    alpha[0] -= log_scales[0]

    for row in range(1, row_count):
        moves, weights = transition_log_weights(
            delta_md[row], statistics, calibration, config=config
        )
        alpha[row] = (
            _banded_forward_step(alpha[row - 1], moves, weights)
            + emission[row]
        )
        log_scales[row] = _logsumexp(alpha[row])
        if not np.isfinite(log_scales[row]):
            raise RuntimeError(f"No reachable lattice state at row {row}")
        alpha[row] -= log_scales[row]
    return alpha, log_scales


def infer_signed_tvt_posterior_mean(
    prediction_md: Iterable[float] | np.ndarray,
    horizontal_gr: Iterable[float] | np.ndarray,
    *,
    anchor_md: float,
    anchor_tvt: float,
    typewell_tvt: Iterable[float] | np.ndarray,
    typewell_gr: Iterable[float] | np.ndarray,
    statistics: LatticeStatistics,
    calibration: PrefixCalibration,
    config: LatticeConfig,
) -> np.ndarray:
    """Compute exact squared-loss MBR predictions with rolling backward state.

    Unlike :func:`infer_signed_tvt_lattice`, this production path does not
    retain a dense posterior, beta matrix, or Viterbi backpointers.
    """

    states, emission, delta_md, anchor_index = _prepare_inference(
        prediction_md,
        horizontal_gr,
        anchor_md=anchor_md,
        anchor_tvt=anchor_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        statistics=statistics,
        calibration=calibration,
        config=config,
    )
    alpha, log_scales = _forward_messages(
        emission,
        delta_md,
        anchor_index,
        statistics,
        calibration,
        config=config,
    )
    row_count = len(alpha)
    posterior_mean = np.empty(row_count, dtype=np.float64)
    beta = np.zeros(len(states), dtype=np.float64)
    for row in range(row_count - 1, -1, -1):
        log_posterior = alpha[row] + beta
        maximum = float(np.max(log_posterior))
        probability = np.exp(log_posterior - maximum)
        probability /= np.sum(probability)
        posterior_mean[row] = float(probability @ states)
        if row > 0:
            moves, weights = transition_log_weights(
                delta_md[row], statistics, calibration, config=config
            )
            beta = _banded_backward_step(
                beta, emission[row], moves, weights
            )
            beta -= log_scales[row]
    return posterior_mean


def infer_signed_tvt_lattice(
    prediction_md: Iterable[float] | np.ndarray,
    horizontal_gr: Iterable[float] | np.ndarray,
    *,
    anchor_md: float,
    anchor_tvt: float,
    typewell_tvt: Iterable[float] | np.ndarray,
    typewell_gr: Iterable[float] | np.ndarray,
    statistics: LatticeStatistics,
    calibration: PrefixCalibration,
    config: LatticeConfig,
) -> LatticeResult:
    """Run exact banded forward-backward and Viterbi inference."""

    states, emission, delta_md, anchor_index = _prepare_inference(
        prediction_md,
        horizontal_gr,
        anchor_md=anchor_md,
        anchor_tvt=anchor_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        statistics=statistics,
        calibration=calibration,
        config=config,
    )
    state_count = len(states)
    alpha, log_scales = _forward_messages(
        emission,
        delta_md,
        anchor_index,
        statistics,
        calibration,
        config=config,
    )
    row_count = len(alpha)
    beta = np.zeros((row_count, state_count), dtype=np.float64)
    for row in range(row_count - 2, -1, -1):
        moves, weights = transition_log_weights(
            delta_md[row + 1], statistics, calibration, config=config
        )
        beta[row] = _banded_backward_step(
            beta[row + 1], emission[row + 1], moves, weights
        )
        beta[row] -= log_scales[row + 1]

    log_posterior = alpha + beta
    row_maximum = np.max(log_posterior, axis=1, keepdims=True)
    posterior = np.exp(log_posterior - row_maximum)
    posterior /= np.sum(posterior, axis=1, keepdims=True)
    posterior_mean = posterior @ states
    centered = states[None, :] - posterior_mean[:, None]
    posterior_variance = np.sum(posterior * centered * centered, axis=1)
    marginal_map = states[np.argmax(posterior, axis=1)]
    map_index = _viterbi_path(
        emission,
        delta_md,
        anchor_index,
        statistics,
        calibration,
        config=config,
    )
    map_path = states[map_index]
    entropy = -np.sum(
        posterior * np.log(np.maximum(posterior, np.finfo(np.float64).tiny)),
        axis=1,
    )
    edge_width = min(config.edge_bins, state_count // 2)
    edge_mass = (
        np.sum(posterior[:, :edge_width], axis=1)
        + np.sum(posterior[:, -edge_width:], axis=1)
    )
    return LatticeResult(
        state_grid=states,
        posterior=posterior,
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        map_path=map_path,
        marginal_map=marginal_map,
        entropy=entropy,
        edge_mass=edge_mass,
        log_evidence=float(np.sum(log_scales)),
    )


__all__ = [
    "LatticeConfig",
    "LatticeResult",
    "LatticeStatistics",
    "PrefixCalibration",
    "blended_transition_location",
    "build_emission_log_potential",
    "build_state_grid",
    "derive_prefix_calibration",
    "fit_lattice_statistics",
    "infer_signed_tvt_lattice",
    "infer_signed_tvt_posterior_mean",
    "interpolate_typewell_gr",
    "transition_log_weights",
]
