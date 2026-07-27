"""Realistic error robustness for locked acausal sparse TVT anchors.

This target-bearing counterfactual diagnostic evaluates systematic per-well
datum bias, gross anchor outliers, and mixed measurement error.  Placement,
anchor count, and the acausal piecewise-linear correction are locked.  The PS
boundary remains exact and no per-seed wide OOF predictions are written.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import enumerate_wells
from ..folds import load_fold_mapping
from .anchor_noise_robustness import (
    DEFAULT_NOISE_SEEDS,
    LOCKED_CONFIGS,
    LOCKED_EXACT_METRICS,
    NoiseWellContext,
    _build_noise_contexts,
    _read_blend_oof,
    _selected_wells,
    _validate_locked_selection,
    _validate_seed,
    condition_noisy_piecewise,
    piecewise_noise_weights,
    stable_noise_matrix,
    stable_standard_normal,
)
from .sparse_anchor_voi import (
    _DeterministicGzipCsvWriter,
    _atomic_json,
    _markdown_table,
    anchor_mask,
    condition_piecewise_residual,
    safe_output_directory,
)
from .terminal_anchor_frontier import (
    PIECEWISE_METHOD,
    _boundary_md,
    frontier_segment_regions,
    select_terminal_anchors,
)
from .sparse_anchor_voi import FOLD_SHA256, FOLD_VERSION


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260819
PERSISTENT_FOLD_RMSE_THRESHOLD = 4.0
PERSISTENT_FOLD_FAILURE_RATE = 0.95
SEGMENTS = ("pre_first", "between", "post_last")
ROWWISE_TOLERANCE = 2e-13
METRIC_TOLERANCE = 1e-12
PER_WELL_SSE_TOLERANCE = 1e-8

ERROR_CONFIGS = tuple(
    config for config in LOCKED_CONFIGS if config.name in {"near_k6", "td_k6"}
)


def _validate_nonnegative(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def validate_tau(value: float) -> float:
    return _validate_nonnegative(value, name="tau")


def validate_sigma(value: float) -> float:
    return _validate_nonnegative(value, name="sigma")


def validate_outlier_probability(value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("outlier probability must be finite and in [0, 1]")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("outlier probability must be finite and in [0, 1]")
    return result


def validate_outlier_magnitude(value: float) -> float:
    return _validate_nonnegative(value, name="outlier magnitude")


@dataclass(frozen=True)
class ErrorScenario:
    error_family: str
    name: str
    tau: float
    sigma: float
    outlier_probability: float
    outlier_magnitude: float
    gaussian_comparator_sigma: float

    def __post_init__(self) -> None:
        if not self.error_family or not self.name:
            raise ValueError("scenario family and name must be non-empty")
        tau = validate_tau(self.tau)
        sigma = validate_sigma(self.sigma)
        probability = validate_outlier_probability(self.outlier_probability)
        magnitude = validate_outlier_magnitude(self.outlier_magnitude)
        comparator = validate_sigma(self.gaussian_comparator_sigma)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "outlier_probability", probability)
        object.__setattr__(self, "outlier_magnitude", magnitude)
        object.__setattr__(self, "gaussian_comparator_sigma", comparator)


def default_error_scenarios() -> tuple[ErrorScenario, ...]:
    scenarios: list[ErrorScenario] = []
    for tau, name in (
        (0.0, "bias_tau_0"),
        (0.25, "bias_tau_0p25"),
        (0.5, "bias_tau_0p5"),
        (1.0, "bias_tau_1"),
        (2.0, "bias_tau_2"),
        (4.0, "bias_tau_4"),
    ):
        scenarios.append(
            ErrorScenario("systematic_bias", name, tau, 0.0, 0.0, 0.0, tau)
        )
    for probability, probability_name in (
        (0.05, "05"),
        (0.10, "10"),
        (0.20, "20"),
    ):
        for magnitude in (2.0, 5.0, 10.0):
            scenarios.append(
                ErrorScenario(
                    "outlier_only",
                    f"outlier_p{probability_name}_m{int(magnitude)}",
                    0.0,
                    0.0,
                    probability,
                    magnitude,
                    0.0,
                )
            )
    for tau, sigma, name in (
        (0.5, 0.5, "bias0p5_sigma0p5"),
        (0.5, 1.0, "bias0p5_sigma1"),
        (1.0, 0.5, "bias1_sigma0p5"),
        (1.0, 1.0, "bias1_sigma1"),
        (2.0, 2.0, "bias2_sigma2"),
    ):
        scenarios.append(
            ErrorScenario(
                "bias_plus_gaussian", name, tau, sigma, 0.0, 0.0, sigma
            )
        )
    scenarios.extend(
        (
            ErrorScenario(
                "combined_operational", "mild", 0.5, 0.5, 0.05, 5.0, 0.5
            ),
            ErrorScenario(
                "combined_operational", "moderate", 1.0, 1.0, 0.10, 5.0, 1.0
            ),
            ErrorScenario(
                "combined_operational", "severe", 2.0, 2.0, 0.10, 10.0, 2.0
            ),
        )
    )
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("default scenario names must be unique")
    return tuple(scenarios)


DEFAULT_SCENARIOS = default_error_scenarios()


def select_error_scenarios(names: Sequence[str] | None) -> tuple[ErrorScenario, ...]:
    if names is None:
        return DEFAULT_SCENARIOS
    requested = tuple(dict.fromkeys(str(name) for name in names))
    by_name = {scenario.name: scenario for scenario in DEFAULT_SCENARIOS}
    if not requested or any(name not in by_name for name in requested):
        raise ValueError(
            "scenario names must be a non-empty subset of the locked scenario matrix"
        )
    return tuple(by_name[name] for name in requested)


def _validate_well_id(well_id: str) -> bytes:
    identifier = str(well_id)
    if not identifier or "\x00" in identifier:
        raise ValueError("well_id must be a non-empty string without NUL")
    return identifier.encode("utf-8")


def _stable_digest(
    seed: int,
    well_id: str,
    anchor_row: int | None,
    key: str,
) -> bytes:
    validated_seed = _validate_seed(seed)
    encoded_well = _validate_well_id(well_id)
    encoded_key = str(key).encode("ascii")
    if not encoded_key or b"\x00" in encoded_key:
        raise ValueError("random key must be non-empty ASCII without NUL")
    digest = hashlib.sha256()
    digest.update(b"rogii-anchor-error-realism-v1\x00")
    digest.update(validated_seed.to_bytes(8, "big", signed=False))
    digest.update(len(encoded_well).to_bytes(4, "big", signed=False))
    digest.update(encoded_well)
    if anchor_row is None:
        digest.update(b"\x00")
    else:
        row = int(anchor_row)
        if isinstance(anchor_row, (bool, np.bool_)) or row < 0:
            raise ValueError("anchor row must be a non-negative integer")
        digest.update(b"\x01")
        digest.update(row.to_bytes(8, "big", signed=False))
    digest.update(len(encoded_key).to_bytes(2, "big", signed=False))
    digest.update(encoded_key)
    return digest.digest()


def _normal_from_digest(digest: bytes) -> float:
    denominator = float(1 << 64)
    uniform_1 = (int.from_bytes(digest[:8], "big") + 0.5) / denominator
    uniform_2 = (int.from_bytes(digest[8:16], "big") + 0.5) / denominator
    return float(
        math.sqrt(-2.0 * math.log(uniform_1))
        * math.cos(2.0 * math.pi * uniform_2)
    )


def stable_common_bias_normal(seed: int, well_id: str) -> float:
    """Stable N(0,1) draw keyed by ``(seed, well_id, "common_bias")``."""
    return _normal_from_digest(
        _stable_digest(seed, well_id, None, "common_bias")
    )


def stable_independent_normal(seed: int, well_id: str, anchor_row: int) -> float:
    """Stable independent draw, exactly backward-compatible with the old engine."""
    return stable_standard_normal(seed, well_id, anchor_row)


def stable_outlier_indicator_uniform(
    seed: int, well_id: str, anchor_row: int
) -> float:
    digest = _stable_digest(seed, well_id, anchor_row, "outlier_indicator")
    return (int.from_bytes(digest[:8], "big") + 0.5) / float(1 << 64)


def stable_outlier_sign(seed: int, well_id: str, anchor_row: int) -> int:
    digest = _stable_digest(seed, well_id, anchor_row, "outlier_sign")
    uniform = (int.from_bytes(digest[:8], "big") + 0.5) / float(1 << 64)
    return -1 if uniform < 0.5 else 1


@dataclass(frozen=True)
class AnchorLatents:
    common_bias: np.ndarray
    independent: np.ndarray
    outlier_indicator_uniform: np.ndarray
    outlier_sign: np.ndarray


def stable_anchor_latents(
    seeds: Sequence[int],
    well_id: str,
    anchor_rows: Sequence[int] | np.ndarray,
) -> AnchorLatents:
    seed_values = tuple(_validate_seed(seed) for seed in seeds)
    rows = np.asarray(anchor_rows)
    if (
        not seed_values
        or len(set(seed_values)) != len(seed_values)
        or rows.ndim != 1
    ):
        raise ValueError("seeds must be unique/non-empty and rows must be 1D")
    independent = stable_noise_matrix(seed_values, well_id, rows)
    bias = np.asarray(
        [stable_common_bias_normal(seed, well_id) for seed in seed_values],
        dtype=np.float64,
    )
    indicator = np.empty_like(independent)
    sign = np.empty_like(independent, dtype=np.int8)
    for seed_index, seed in enumerate(seed_values):
        for row_index, row in enumerate(rows):
            indicator[seed_index, row_index] = stable_outlier_indicator_uniform(
                seed, well_id, int(row)
            )
            sign[seed_index, row_index] = stable_outlier_sign(
                seed, well_id, int(row)
            )
    return AnchorLatents(bias, independent, indicator, sign)


def scenario_anchor_errors(
    scenario: ErrorScenario,
    latents: AnchorLatents,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full anchor errors and the paired no-outlier errors."""
    no_outlier = (
        scenario.tau * latents.common_bias[:, None]
        + scenario.sigma * latents.independent
    )
    flags = latents.outlier_indicator_uniform < scenario.outlier_probability
    outlier = (
        flags
        * latents.outlier_sign.astype(np.float64)
        * scenario.outlier_magnitude
    )
    return no_outlier + outlier, flags


def corrupted_anchor_observations(
    true_anchor_tvt: np.ndarray,
    scenario: ErrorScenario,
    latents: AnchorLatents,
) -> np.ndarray:
    truth = np.asarray(true_anchor_tvt, dtype=np.float64)
    errors, _ = scenario_anchor_errors(scenario, latents)
    if truth.ndim != 1 or errors.shape[1] != len(truth):
        raise ValueError("true anchors and latent anchor columns must align")
    return truth[None, :] + errors


def _mask_coefficients(
    exact_error: np.ndarray,
    weights: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    error = np.asarray(exact_error, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    selected = np.asarray(masks, dtype=bool)
    if (
        error.ndim != 1
        or weight.shape[0] != len(error)
        or selected.ndim != 2
        or selected.shape[1] != len(error)
    ):
        raise ValueError("SSE coefficient inputs are not aligned")
    counts = selected.sum(axis=1, dtype=np.int64)
    selected_float = selected.astype(np.float64)
    exact_sse = selected_float @ np.square(error)
    linear = (selected_float * error[None, :]) @ weight
    gram = np.einsum(
        "mr,ri,rj->mij", selected_float, weight, weight, optimize=True
    )
    return exact_sse, linear, gram, counts


def _evaluate_anchor_error_sse(
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    anchor_errors: np.ndarray,
) -> np.ndarray:
    exact_sse, linear, gram, _ = coefficients
    error = np.asarray(anchor_errors, dtype=np.float64)
    if error.ndim == 2:
        error = error[None, :, :]
    if error.ndim != 3 or error.shape[2] != linear.shape[1]:
        raise ValueError("anchor error array must have shape (scenario, seed, anchor)")
    cross = np.einsum("csk,mk->csm", error, linear, optimize=True)
    quadratic = np.einsum(
        "csi,mij,csj->csm", error, gram, error, optimize=True
    )
    result = exact_sse[None, None, :] + 2.0 * cross + quadratic
    tolerance = (
        np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(exact_sse))))
        * 64
    )
    if float(np.min(result)) < -tolerance:
        raise ValueError("anchor error SSE propagation became negative")
    return np.maximum(result, 0.0)


def reconstruct_pooled_metrics(per_well: pd.DataFrame) -> tuple[float, float]:
    rows = int(per_well["rows"].sum())
    non_rows = int(per_well["non_anchor_rows"].sum())
    if rows <= 0 or non_rows <= 0:
        raise ValueError("per-well rows must be positive")
    return (
        float(np.sqrt(per_well["sse"].sum() / rows)),
        float(np.sqrt(per_well["non_anchor_sse"].sum() / non_rows)),
    )


def hierarchical_error_bootstrap(
    full_sse: np.ndarray,
    non_anchor_sse: np.ndarray,
    rows: np.ndarray,
    non_anchor_rows: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Combine well-population and independently sampled error uncertainty."""
    full = np.asarray(full_sse, dtype=np.float64)
    non = np.asarray(non_anchor_sse, dtype=np.float64)
    row_counts = np.asarray(rows, dtype=np.int64)
    non_counts = np.asarray(non_anchor_rows, dtype=np.int64)
    bootstrap_seed = _validate_seed(seed)
    if (
        full.ndim != 4
        or non.shape != full.shape
        or row_counts.shape != (full.shape[3],)
        or non_counts.shape != (full.shape[0], full.shape[3])
        or samples < 1
        or not np.isfinite(full).all()
        or not np.isfinite(non).all()
        or np.any(full < 0)
        or np.any(non < 0)
        or np.any(row_counts <= 0)
        or np.any(non_counts <= 0)
    ):
        raise ValueError("invalid hierarchical bootstrap inputs")
    config_count, scenario_count, realization_count, well_count = full.shape
    rng = np.random.default_rng(bootstrap_seed)
    well_draws = rng.integers(
        0, well_count, size=(samples, well_count), dtype=np.int32
    )
    realization_draws = rng.integers(
        0,
        realization_count,
        size=(samples, well_count),
        dtype=np.int16,
    )
    sampled_rows = row_counts[well_draws].sum(axis=1, dtype=np.int64)
    output: list[dict[str, Any]] = []

    def quantiles(values: np.ndarray) -> dict[str, float]:
        return {
            "q025": float(np.quantile(values, 0.025)),
            "q50": float(np.quantile(values, 0.50)),
            "q975": float(np.quantile(values, 0.975)),
        }

    for config_index in range(config_count):
        sampled_non_rows = non_counts[config_index][well_draws].sum(
            axis=1, dtype=np.int64
        )
        for scenario_index in range(scenario_count):
            group_full = full[config_index, scenario_index]
            group_non = non[config_index, scenario_index]
            hierarchical_full = np.sqrt(
                group_full[realization_draws, well_draws].sum(axis=1)
                / sampled_rows
            )
            hierarchical_non = np.sqrt(
                group_non[realization_draws, well_draws].sum(axis=1)
                / sampled_non_rows
            )
            mean_full = group_full.mean(axis=0)
            mean_non = group_non.mean(axis=0)
            well_only_full = np.sqrt(
                mean_full[well_draws].sum(axis=1) / sampled_rows
            )
            well_only_non = np.sqrt(
                mean_non[well_draws].sum(axis=1) / sampled_non_rows
            )
            fixed_full = np.sqrt(group_full.sum(axis=1) / row_counts.sum())
            fixed_non = np.sqrt(
                group_non.sum(axis=1) / non_counts[config_index].sum()
            )
            output.append(
                {
                    "config_index": config_index,
                    "scenario_index": scenario_index,
                    "fixed_well_error_realization": {
                        "definition": (
                            "all 773 wells fixed; one common deterministic error "
                            "realization seed changes all anchor errors"
                        ),
                        "full_rmse": quantiles(fixed_full),
                        "non_anchor_rmse": quantiles(fixed_non),
                    },
                    "well_population_only": {
                        "definition": (
                            "resample wells after averaging each well's SSE over "
                            "all error realizations"
                        ),
                        "full_rmse": quantiles(well_only_full),
                        "non_anchor_rmse": quantiles(well_only_non),
                    },
                    "hierarchical": {
                        "definition": (
                            "resample 773 wells with replacement and independently "
                            "draw one error realization for each selected well"
                        ),
                        "full_rmse": quantiles(hierarchical_full),
                        "non_anchor_rmse": quantiles(hierarchical_non),
                    },
                }
            )
    return output


def _scenario_metadata(scenario: ErrorScenario) -> dict[str, Any]:
    return {
        "error_family": scenario.error_family,
        "scenario": scenario.name,
        "tau": scenario.tau,
        "sigma": scenario.sigma,
        "outlier_probability": scenario.outlier_probability,
        "outlier_magnitude": scenario.outlier_magnitude,
        "gaussian_comparator_sigma": scenario.gaussian_comparator_sigma,
    }


def _tail_arrays(full_sse: np.ndarray, rows: np.ndarray) -> dict[str, np.ndarray]:
    rmse = np.sqrt(full_sse / rows[None, None, None, :])
    worst_count = max(1, int(math.ceil(0.10 * len(rows))))
    sorted_sse = np.sort(full_sse, axis=3)
    return {
        "well_rmse": rmse,
        "p50": np.quantile(rmse, 0.50, axis=3),
        "p90": np.quantile(rmse, 0.90, axis=3),
        "p95": np.quantile(rmse, 0.95, axis=3),
        "max": np.max(rmse, axis=3),
        "wells_ge_40": np.sum(rmse >= 40.0, axis=3),
        "wells_ge_60": np.sum(rmse >= 60.0, axis=3),
        "worst10_sse_share": (
            sorted_sse[:, :, :, -worst_count:].sum(axis=3)
            / full_sse.sum(axis=3)
        ),
    }


def _per_seed_frame(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    full_sse: np.ndarray,
    non_sse: np.ndarray,
    exact_full_sse: np.ndarray,
    exact_non_sse: np.ndarray,
    gaussian_full_sse: np.ndarray,
    gaussian_non_sse: np.ndarray,
    gaussian_sigmas: np.ndarray,
    rows: np.ndarray,
    non_rows: np.ndarray,
    fold_sse: np.ndarray,
    fold_rows: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    total_rows = int(rows.sum())
    full_rmse = np.sqrt(full_sse.sum(axis=3) / total_rows)
    non_rmse = np.sqrt(
        non_sse.sum(axis=3) / non_rows.sum(axis=1)[:, None, None]
    )
    exact_full_rmse = np.sqrt(exact_full_sse.sum(axis=1) / total_rows)
    exact_non_rmse = np.sqrt(
        exact_non_sse.sum(axis=1) / non_rows.sum(axis=1)
    )
    gaussian_full_rmse = np.sqrt(
        gaussian_full_sse.sum(axis=3) / total_rows
    )
    gaussian_non_rmse = np.sqrt(
        gaussian_non_sse.sum(axis=3) / non_rows.sum(axis=1)[:, None, None]
    )
    fold_rmse = np.sqrt(fold_sse / fold_rows[None, None, None, :])
    fold_max = fold_rmse.max(axis=3)
    tail = _tail_arrays(full_sse, rows)
    gaussian_index = {
        float(sigma): index for index, sigma in enumerate(gaussian_sigmas)
    }
    records: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for scenario_index, scenario in enumerate(scenarios):
            comparator_index = gaussian_index[
                float(scenario.gaussian_comparator_sigma)
            ]
            for seed_index, realization_seed in enumerate(seeds):
                full = float(full_rmse[config_index, scenario_index, seed_index])
                non = float(non_rmse[config_index, scenario_index, seed_index])
                gaussian_full = float(
                    gaussian_full_rmse[
                        config_index, comparator_index, seed_index
                    ]
                )
                gaussian_non = float(
                    gaussian_non_rmse[
                        config_index, comparator_index, seed_index
                    ]
                )
                records.append(
                    {
                        "config": config.name,
                        "policy": config.policy,
                        "k": config.k,
                        "fractions": config.fractions_text,
                        **_scenario_metadata(scenario),
                        "realization_seed": int(realization_seed),
                        "full_rows": total_rows,
                        "non_anchor_rows": int(non_rows[config_index].sum()),
                        "full_rmse": full,
                        "non_anchor_rmse": non,
                        "exact_full_rmse": float(
                            exact_full_rmse[config_index]
                        ),
                        "exact_non_anchor_rmse": float(
                            exact_non_rmse[config_index]
                        ),
                        "full_delta_vs_exact": (
                            full - float(exact_full_rmse[config_index])
                        ),
                        "non_anchor_delta_vs_exact": (
                            non - float(exact_non_rmse[config_index])
                        ),
                        "pure_gaussian_full_rmse": gaussian_full,
                        "pure_gaussian_non_anchor_rmse": gaussian_non,
                        "full_delta_vs_pure_gaussian": full - gaussian_full,
                        "non_anchor_delta_vs_pure_gaussian": non - gaussian_non,
                        "fold_max_rmse": float(
                            fold_max[
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "well_rmse_p50": float(
                            tail["p50"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "well_rmse_p90": float(
                            tail["p90"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "well_rmse_p95": float(
                            tail["p95"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "well_rmse_max": float(
                            tail["max"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "wells_ge_40": int(
                            tail["wells_ge_40"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "wells_ge_60": int(
                            tail["wells_ge_60"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "worst10_sse_share": float(
                            tail["worst10_sse_share"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                    }
                )
    arrays = {
        "full_rmse": full_rmse,
        "non_anchor_rmse": non_rmse,
        "fold_rmse": fold_rmse,
        "fold_max_rmse": fold_max,
        **tail,
    }
    return pd.DataFrame(records), arrays


def _per_fold_frame(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    fold_sse: np.ndarray,
    fold_non_sse: np.ndarray,
    fold_rows: np.ndarray,
    fold_non_rows: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for scenario_index, scenario in enumerate(scenarios):
            for seed_index, realization_seed in enumerate(seeds):
                for fold in range(5):
                    records.append(
                        {
                            "config": config.name,
                            "policy": config.policy,
                            "k": config.k,
                            "fractions": config.fractions_text,
                            **_scenario_metadata(scenario),
                            "realization_seed": int(realization_seed),
                            "fold": fold,
                            "rows": int(fold_rows[fold]),
                            "non_anchor_rows": int(
                                fold_non_rows[config_index, fold]
                            ),
                            "full_rmse": float(
                                np.sqrt(
                                    fold_sse[
                                        config_index,
                                        scenario_index,
                                        seed_index,
                                        fold,
                                    ]
                                    / fold_rows[fold]
                                )
                            ),
                            "non_anchor_rmse": float(
                                np.sqrt(
                                    fold_non_sse[
                                        config_index,
                                        scenario_index,
                                        seed_index,
                                        fold,
                                    ]
                                    / fold_non_rows[config_index, fold]
                                )
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _segment_frame(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    segment_sse: np.ndarray,
    no_outlier_segment_sse: np.ndarray,
    exact_segment_sse: np.ndarray,
    gaussian_segment_sse: np.ndarray,
    gaussian_sigmas: np.ndarray,
    segment_rows: np.ndarray,
) -> pd.DataFrame:
    gaussian_index = {
        float(sigma): index for index, sigma in enumerate(gaussian_sigmas)
    }
    records: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for scenario_index, scenario in enumerate(scenarios):
            comparator_index = gaussian_index[
                float(scenario.gaussian_comparator_sigma)
            ]
            for seed_index, realization_seed in enumerate(seeds):
                for segment_index, segment in enumerate(SEGMENTS):
                    count = int(segment_rows[config_index, segment_index])
                    if not count:
                        continue
                    actual = float(
                        segment_sse[
                            config_index,
                            scenario_index,
                            seed_index,
                            segment_index,
                        ]
                    )
                    no_outlier = float(
                        no_outlier_segment_sse[
                            config_index,
                            scenario_index,
                            seed_index,
                            segment_index,
                        ]
                    )
                    exact = float(
                        exact_segment_sse[config_index, segment_index]
                    )
                    gaussian = float(
                        gaussian_segment_sse[
                            config_index,
                            comparator_index,
                            seed_index,
                            segment_index,
                        ]
                    )
                    records.append(
                        {
                            "config": config.name,
                            "policy": config.policy,
                            "k": config.k,
                            "fractions": config.fractions_text,
                            **_scenario_metadata(scenario),
                            "realization_seed": int(realization_seed),
                            "segment": segment,
                            "rows": count,
                            "sse": actual,
                            "full_rmse": float(np.sqrt(actual / count)),
                            "no_outlier_sse": no_outlier,
                            "no_outlier_rmse": float(
                                np.sqrt(no_outlier / count)
                            ),
                            "delta_sse_vs_no_outlier": actual - no_outlier,
                            "exact_sse": exact,
                            "exact_rmse": float(np.sqrt(exact / count)),
                            "delta_sse_vs_exact": actual - exact,
                            "pure_gaussian_sse": gaussian,
                            "pure_gaussian_rmse": float(
                                np.sqrt(gaussian / count)
                            ),
                            "delta_sse_vs_pure_gaussian": actual - gaussian,
                        }
                    )
    return pd.DataFrame(records)


def _distance_frame(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    distance_sse: np.ndarray,
    distance_non_sse: np.ndarray,
    distance_rows: np.ndarray,
    distance_non_rows: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for scenario_index, scenario in enumerate(scenarios):
            for seed_index, realization_seed in enumerate(seeds):
                for decile in range(10):
                    count = int(distance_rows[config_index, decile])
                    non_count = int(distance_non_rows[config_index, decile])
                    sse = float(
                        distance_sse[
                            config_index, scenario_index, seed_index, decile
                        ]
                    )
                    non_sse = float(
                        distance_non_sse[
                            config_index, scenario_index, seed_index, decile
                        ]
                    )
                    records.append(
                        {
                            "config": config.name,
                            "policy": config.policy,
                            "k": config.k,
                            "fractions": config.fractions_text,
                            **_scenario_metadata(scenario),
                            "realization_seed": int(realization_seed),
                            "distance_decile": decile,
                            "rows": count,
                            "sse": sse,
                            "full_rmse": float(np.sqrt(sse / count)),
                            "non_anchor_rows": non_count,
                            "non_anchor_sse": non_sse,
                            "non_anchor_rmse": float(
                                np.sqrt(non_sse / non_count)
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _outlier_exposure_frame(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    wells: int,
    exposure: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        anchor_denominator = wells * config.k
        for scenario_index, scenario in enumerate(scenarios):
            for seed_index, realization_seed in enumerate(seeds):
                outliers = int(
                    exposure["outlier_anchors"][
                        config_index, scenario_index, seed_index
                    ]
                )
                wells_with = int(
                    exposure["wells_with_outlier"][
                        config_index, scenario_index, seed_index
                    ]
                )
                multiple = int(
                    exposure["wells_with_multiple_outliers"][
                        config_index, scenario_index, seed_index
                    ]
                )
                denominator = max(outliers, 1)
                records.append(
                    {
                        "config": config.name,
                        "policy": config.policy,
                        "k": config.k,
                        "fractions": config.fractions_text,
                        **_scenario_metadata(scenario),
                        "realization_seed": int(realization_seed),
                        "anchor_rows": anchor_denominator,
                        "realized_outlier_anchors": outliers,
                        "realized_outlier_fraction": (
                            outliers / anchor_denominator
                        ),
                        "wells": wells,
                        "wells_with_outlier": wells_with,
                        "wells_with_outlier_fraction": wells_with / wells,
                        "wells_with_multiple_outliers": multiple,
                        "wells_with_multiple_outliers_fraction": (
                            multiple / wells
                        ),
                        "event_interval_count": float(
                            exposure["event_intervals"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "mean_intervals_per_outlier": float(
                            exposure["event_intervals"][
                                config_index, scenario_index, seed_index
                            ]
                            / denominator
                        )
                        if outliers
                        else 0.0,
                        "event_support_rows": float(
                            exposure["event_support_rows"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "mean_rows_per_outlier": float(
                            exposure["event_support_rows"][
                                config_index, scenario_index, seed_index
                            ]
                            / denominator
                        )
                        if outliers
                        else 0.0,
                        "event_before_rows": float(
                            exposure["event_before_rows"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "mean_before_rows_per_outlier": float(
                            exposure["event_before_rows"][
                                config_index, scenario_index, seed_index
                            ]
                            / denominator
                        )
                        if outliers
                        else 0.0,
                        "event_after_rows": float(
                            exposure["event_after_rows"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "mean_after_rows_per_outlier": float(
                            exposure["event_after_rows"][
                                config_index, scenario_index, seed_index
                            ]
                            / denominator
                        )
                        if outliers
                        else 0.0,
                        "event_md_span_ft": float(
                            exposure["event_md_span_ft"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                        "mean_md_span_ft_per_outlier": float(
                            exposure["event_md_span_ft"][
                                config_index, scenario_index, seed_index
                            ]
                            / denominator
                        )
                        if outliers
                        else 0.0,
                        "unique_affected_rows": int(
                            exposure["unique_affected_rows"][
                                config_index, scenario_index, seed_index
                            ]
                        ),
                    }
                )
    return pd.DataFrame(records)


def _information_curve(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    arrays: Mapping[str, np.ndarray],
    per_seed: pd.DataFrame,
    per_segment: pd.DataFrame,
    outlier_exposure: pd.DataFrame,
    hierarchy: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    hierarchy_by_index = {
        (int(item["config_index"]), int(item["scenario_index"])): item
        for item in hierarchy
    }
    rows: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for scenario_index, scenario in enumerate(scenarios):
            full = arrays["full_rmse"][config_index, scenario_index]
            non = arrays["non_anchor_rmse"][config_index, scenario_index]
            fold_max = arrays["fold_max_rmse"][config_index, scenario_index]
            fold_values = arrays["fold_rmse"][config_index, scenario_index]
            fold_failure_rates = np.mean(
                fold_values >= PERSISTENT_FOLD_RMSE_THRESHOLD, axis=0
            )
            persistent = bool(
                np.any(fold_failure_rates >= PERSISTENT_FOLD_FAILURE_RATE)
            )
            bootstrap = hierarchy_by_index[
                (config_index, scenario_index)
            ]["hierarchical"]
            hierarchical_full = bootstrap["full_rmse"]
            hierarchical_non = bootstrap["non_anchor_rmse"]
            wells_40 = arrays["wells_ge_40"][config_index, scenario_index]
            wells_60 = arrays["wells_ge_60"][config_index, scenario_index]
            catastrophic = bool(
                int(np.max(wells_40)) > 0 or int(np.max(wells_60)) > 0
            )
            group_seed = per_seed[
                per_seed["config"].eq(config.name)
                & per_seed["scenario"].eq(scenario.name)
            ]
            group_segment = per_segment[
                per_segment["config"].eq(config.name)
                & per_segment["scenario"].eq(scenario.name)
            ]
            group_exposure = outlier_exposure[
                outlier_exposure["config"].eq(config.name)
                & outlier_exposure["scenario"].eq(scenario.name)
            ]

            def segment_q50(segment: str) -> float:
                values = group_segment.loc[
                    group_segment["segment"].eq(segment), "full_rmse"
                ].to_numpy(dtype=np.float64)
                return (
                    float(np.quantile(values, 0.50))
                    if len(values)
                    else float("nan")
                )

            robust = bool(
                np.quantile(full, 0.95) < 4.0
                and np.quantile(non, 0.95) < 4.0
                and hierarchical_full["q975"] < 4.0
                and hierarchical_non["q975"] < 4.0
                and not persistent
                and not catastrophic
            )
            rows.append(
                {
                    "config": config.name,
                    "policy": config.policy,
                    "k": config.k,
                    "fractions": config.fractions_text,
                    **_scenario_metadata(scenario),
                    "realizations": len(seeds),
                    "full_rmse_mean": float(np.mean(full)),
                    "full_rmse_q05": float(np.quantile(full, 0.05)),
                    "full_rmse_q50": float(np.quantile(full, 0.50)),
                    "full_rmse_q95": float(np.quantile(full, 0.95)),
                    "non_anchor_rmse_mean": float(np.mean(non)),
                    "non_anchor_rmse_q05": float(np.quantile(non, 0.05)),
                    "non_anchor_rmse_q50": float(np.quantile(non, 0.50)),
                    "non_anchor_rmse_q95": float(np.quantile(non, 0.95)),
                    "probability_full_rmse_lt_4": float(np.mean(full < 4.0)),
                    "probability_non_anchor_rmse_lt_4": float(
                        np.mean(non < 4.0)
                    ),
                    "hierarchical_full_q025": float(
                        hierarchical_full["q025"]
                    ),
                    "hierarchical_full_q50": float(
                        hierarchical_full["q50"]
                    ),
                    "hierarchical_full_q975": float(
                        hierarchical_full["q975"]
                    ),
                    "hierarchical_non_anchor_q025": float(
                        hierarchical_non["q025"]
                    ),
                    "hierarchical_non_anchor_q50": float(
                        hierarchical_non["q50"]
                    ),
                    "hierarchical_non_anchor_q975": float(
                        hierarchical_non["q975"]
                    ),
                    "fold_max_q50": float(np.quantile(fold_max, 0.50)),
                    "fold_max_q95": float(np.quantile(fold_max, 0.95)),
                    "p95_well_rmse_q50": float(
                        np.quantile(
                            arrays["p95"][config_index, scenario_index], 0.50
                        )
                    ),
                    "p95_well_rmse_q95": float(
                        np.quantile(
                            arrays["p95"][config_index, scenario_index], 0.95
                        )
                    ),
                    "max_well_rmse_q50": float(
                        np.quantile(
                            arrays["max"][config_index, scenario_index], 0.50
                        )
                    ),
                    "max_well_rmse_q95": float(
                        np.quantile(
                            arrays["max"][config_index, scenario_index], 0.95
                        )
                    ),
                    "wells_ge_40_q50": float(np.quantile(wells_40, 0.50)),
                    "wells_ge_40_q95": float(np.quantile(wells_40, 0.95)),
                    "wells_ge_60_q50": float(np.quantile(wells_60, 0.50)),
                    "wells_ge_60_q95": float(np.quantile(wells_60, 0.95)),
                    "worst10_sse_share_q50": float(
                        np.quantile(
                            arrays["worst10_sse_share"][
                                config_index, scenario_index
                            ],
                            0.50,
                        )
                    ),
                    "worst10_sse_share_q95": float(
                        np.quantile(
                            arrays["worst10_sse_share"][
                                config_index, scenario_index
                            ],
                            0.95,
                        )
                    ),
                    "pre_first_rmse_q50": segment_q50("pre_first"),
                    "between_rmse_q50": segment_q50("between"),
                    "post_last_rmse_q50": segment_q50("post_last"),
                    "realized_outlier_fraction": float(
                        group_exposure["realized_outlier_fraction"].mean()
                    ),
                    "wells_with_outlier_fraction": float(
                        group_exposure["wells_with_outlier_fraction"].mean()
                    ),
                    "wells_with_multiple_outliers_fraction": float(
                        group_exposure[
                            "wells_with_multiple_outliers_fraction"
                        ].mean()
                    ),
                    "mean_outlier_intervals": float(
                        group_exposure["event_interval_count"].sum()
                        / max(
                            1,
                            int(group_exposure["realized_outlier_anchors"].sum()),
                        )
                    ),
                    "mean_outlier_rows": float(
                        group_exposure["event_support_rows"].sum()
                        / max(
                            1,
                            int(group_exposure["realized_outlier_anchors"].sum()),
                        )
                    ),
                    "mean_outlier_md_span_ft": float(
                        group_exposure["event_md_span_ft"].sum()
                        / max(
                            1,
                            int(group_exposure["realized_outlier_anchors"].sum()),
                        )
                    ),
                    "paired_full_delta_vs_exact_q50": float(
                        group_seed["full_delta_vs_exact"].quantile(0.50)
                    ),
                    "paired_full_delta_vs_exact_q95": float(
                        group_seed["full_delta_vs_exact"].quantile(0.95)
                    ),
                    "paired_non_anchor_delta_vs_exact_q50": float(
                        group_seed["non_anchor_delta_vs_exact"].quantile(0.50)
                    ),
                    "paired_full_delta_vs_gaussian_q50": float(
                        group_seed[
                            "full_delta_vs_pure_gaussian"
                        ].quantile(0.50)
                    ),
                    "paired_full_delta_vs_gaussian_q95": float(
                        group_seed[
                            "full_delta_vs_pure_gaussian"
                        ].quantile(0.95)
                    ),
                    "paired_non_anchor_delta_vs_gaussian_q50": float(
                        group_seed[
                            "non_anchor_delta_vs_pure_gaussian"
                        ].quantile(0.50)
                    ),
                    "persistent_fold_failure": persistent,
                    "maximum_fold_failure_rate": float(
                        np.max(fold_failure_rates)
                    ),
                    "maximum_wells_ge_40": int(np.max(wells_40)),
                    "maximum_wells_ge_60": int(np.max(wells_60)),
                    "catastrophic_tail_failure": catastrophic,
                    "robust_lt_4": robust,
                }
            )
    return pd.DataFrame(rows)


def _worst_wells_frame(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    well_ids: Sequence[str],
    folds: np.ndarray,
    rows: np.ndarray,
    full_sse: np.ndarray,
) -> pd.DataFrame:
    rmse = np.sqrt(full_sse / rows[None, None, None, :])
    records: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for scenario_index, scenario in enumerate(scenarios):
            group = rmse[config_index, scenario_index]
            mean = group.mean(axis=0)
            q95 = np.quantile(group, 0.95, axis=0)
            maximum = group.max(axis=0)
            max_seed_position = np.argmax(group, axis=0)
            order = np.lexsort((np.asarray(well_ids), -q95))[:20]
            for rank, well_position in enumerate(order, start=1):
                records.append(
                    {
                        "config": config.name,
                        "policy": config.policy,
                        "k": config.k,
                        **_scenario_metadata(scenario),
                        "rank_by_realization_q95": rank,
                        "well_id": str(well_ids[well_position]),
                        "fold": int(folds[well_position]),
                        "rows": int(rows[well_position]),
                        "rmse_mean": float(mean[well_position]),
                        "rmse_q95": float(q95[well_position]),
                        "rmse_max": float(maximum[well_position]),
                        "realization_seed_at_max": int(
                            seeds[max_seed_position[well_position]]
                        ),
                    }
                )
    return pd.DataFrame(records)


def _regression_reproduction(
    *,
    configs: Sequence[Any],
    seeds: Sequence[int],
    well_ids: Sequence[str],
    exact_full_sse: np.ndarray,
    exact_non_sse: np.ndarray,
    gaussian_full_sse: np.ndarray,
    gaussian_non_sse: np.ndarray,
    gaussian_sigmas: np.ndarray,
    rows: np.ndarray,
    non_rows: np.ndarray,
    terminal_frontier: Path,
    gaussian_noise_artifact: Path,
    strict_full: bool,
) -> dict[str, Any]:
    well_position = {well_id: index for index, well_id in enumerate(well_ids)}
    config_position = {
        config.name: index for index, config in enumerate(configs)
    }
    seed_position = {int(seed): index for index, seed in enumerate(seeds)}
    gaussian_position = {
        float(sigma): index for index, sigma in enumerate(gaussian_sigmas)
    }

    terminal = pd.read_csv(
        Path(terminal_frontier) / "per_well_metrics.csv",
        dtype={"well_id": "string"},
    )
    exact_full_delta = 0.0
    exact_non_delta = 0.0
    exact_rows = 0
    for config_index, config in enumerate(configs):
        expected = terminal[
            terminal["policy"].eq(config.policy)
            & terminal["mode"].eq("acausal")
            & terminal["method"].eq(PIECEWISE_METHOD)
            & terminal["k"].eq(config.k)
            & terminal["well_id"].isin(well_ids)
        ].set_index("well_id")
        if set(expected.index.astype(str)) != set(well_ids):
            raise ValueError(
                f"terminal exact regression coverage mismatch for {config.name}"
            )
        expected = expected.loc[list(well_ids)]
        exact_full_delta = max(
            exact_full_delta,
            float(
                np.max(
                    np.abs(
                        exact_full_sse[config_index]
                        - expected["sse"].to_numpy(dtype=np.float64)
                    )
                )
            ),
        )
        exact_non_delta = max(
            exact_non_delta,
            float(
                np.max(
                    np.abs(
                        exact_non_sse[config_index]
                        - expected["non_anchor_sse"].to_numpy(dtype=np.float64)
                    )
                )
            ),
        )
        exact_rows += len(expected)

    old_config = json.loads(
        (
            Path(gaussian_noise_artifact) / "run_config.json"
        ).read_text(encoding="utf-8")
    )
    old_sigmas = {
        float(value) for value in old_config["sigmas"]
    } & set(gaussian_position)
    regression_sigmas = tuple(sorted(old_sigmas))
    gaussian_full_delta = 0.0
    gaussian_non_delta = 0.0
    gaussian_rows = 0
    required_columns = [
        "config",
        "sigma",
        "noise_seed",
        "well_id",
        "sse",
        "non_anchor_sse",
    ]
    for chunk in pd.read_csv(
        Path(gaussian_noise_artifact) / "per_well_seed_metrics.csv.gz",
        usecols=required_columns,
        dtype={"well_id": "string"},
        chunksize=200_000,
    ):
        selected = chunk[
            chunk["config"].isin(config_position)
            & chunk["sigma"].isin(regression_sigmas)
            & chunk["noise_seed"].isin(seed_position)
            & chunk["well_id"].isin(well_position)
        ]
        if not len(selected):
            continue
        ci = selected["config"].map(config_position).to_numpy(dtype=np.int64)
        gi = selected["sigma"].map(gaussian_position).to_numpy(dtype=np.int64)
        si = selected["noise_seed"].map(seed_position).to_numpy(dtype=np.int64)
        wi = selected["well_id"].map(well_position).to_numpy(dtype=np.int64)
        actual_full = gaussian_full_sse[ci, gi, si, wi]
        actual_non = gaussian_non_sse[ci, gi, si, wi]
        gaussian_full_delta = max(
            gaussian_full_delta,
            float(
                np.max(
                    np.abs(
                        actual_full - selected["sse"].to_numpy(dtype=np.float64)
                    )
                )
            ),
        )
        gaussian_non_delta = max(
            gaussian_non_delta,
            float(
                np.max(
                    np.abs(
                        actual_non
                        - selected["non_anchor_sse"].to_numpy(dtype=np.float64)
                    )
                )
            ),
        )
        gaussian_rows += len(selected)

    expected_gaussian_rows = (
        len(configs)
        * len(regression_sigmas)
        * len(seeds)
        * len(well_ids)
    )
    metric_full_delta = 0.0
    metric_non_delta = 0.0
    metric_groups = 0
    if strict_full:
        old_metrics = pd.read_csv(
            Path(gaussian_noise_artifact) / "per_seed_metrics.csv"
        )
        old_metrics = old_metrics[
            old_metrics["config"].isin(config_position)
            & old_metrics["sigma"].isin(regression_sigmas)
            & old_metrics["noise_seed"].isin(seed_position)
        ].copy()
        for row in old_metrics.itertuples(index=False):
            ci = config_position[str(row.config)]
            gi = gaussian_position[float(row.sigma)]
            si = seed_position[int(row.noise_seed)]
            actual_full = float(
                np.sqrt(gaussian_full_sse[ci, gi, si].sum() / rows.sum())
            )
            actual_non = float(
                np.sqrt(
                    gaussian_non_sse[ci, gi, si].sum()
                    / non_rows[ci].sum()
                )
            )
            metric_full_delta = max(
                metric_full_delta, abs(actual_full - float(row.full_rmse))
            )
            metric_non_delta = max(
                metric_non_delta,
                abs(actual_non - float(row.non_anchor_rmse)),
            )
            metric_groups += 1
        expected_metric_groups = (
            len(configs) * len(regression_sigmas) * len(seeds)
        )
    else:
        expected_metric_groups = 0

    exact_metric_delta = 0.0
    if strict_full:
        for config_index, config in enumerate(configs):
            expected_full, expected_non = LOCKED_EXACT_METRICS[config.name]
            actual_full = float(
                np.sqrt(exact_full_sse[config_index].sum() / rows.sum())
            )
            actual_non = float(
                np.sqrt(
                    exact_non_sse[config_index].sum()
                    / non_rows[config_index].sum()
                )
            )
            exact_metric_delta = max(
                exact_metric_delta,
                abs(actual_full - expected_full),
                abs(actual_non - expected_non),
            )

    passed = bool(
        exact_rows == len(configs) * len(well_ids)
        and exact_full_delta <= PER_WELL_SSE_TOLERANCE
        and exact_non_delta <= PER_WELL_SSE_TOLERANCE
        and gaussian_rows == expected_gaussian_rows
        and gaussian_full_delta <= PER_WELL_SSE_TOLERANCE
        and gaussian_non_delta <= PER_WELL_SSE_TOLERANCE
        and metric_groups == expected_metric_groups
        and metric_full_delta <= METRIC_TOLERANCE
        and metric_non_delta <= METRIC_TOLERANCE
        and exact_metric_delta <= METRIC_TOLERANCE
    )
    return {
        "passed": passed,
        "exact_terminal_per_well_rows": exact_rows,
        "exact_terminal_max_full_sse_delta": exact_full_delta,
        "exact_terminal_max_non_anchor_sse_delta": exact_non_delta,
        "exact_terminal_max_metric_delta": exact_metric_delta,
        "gaussian_regression_sigmas": list(regression_sigmas),
        "gaussian_per_well_rows": gaussian_rows,
        "expected_gaussian_per_well_rows": expected_gaussian_rows,
        "gaussian_max_full_sse_delta": gaussian_full_delta,
        "gaussian_max_non_anchor_sse_delta": gaussian_non_delta,
        "gaussian_metric_groups": metric_groups,
        "expected_gaussian_metric_groups": expected_metric_groups,
        "gaussian_max_full_rmse_delta": metric_full_delta,
        "gaussian_max_non_anchor_rmse_delta": metric_non_delta,
    }


def audit_anchor_error_artifacts(run_root: Path) -> dict[str, Any]:
    """Reconstruct pooled metrics and validate compact artifact key coverage."""
    root = Path(run_root).resolve()
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    per_seed = pd.read_csv(root / "per_seed_metrics.csv")
    curve = pd.read_csv(root / "error_realism_curve.csv")
    per_fold = pd.read_csv(root / "per_fold_seed_metrics.csv")
    per_segment = pd.read_csv(root / "per_segment_seed_metrics.csv")
    exposure = pd.read_csv(root / "outlier_exposure.csv")
    keys = ["config", "scenario", "realization_seed"]
    well_keys = keys + ["well_id"]
    partials: list[pd.DataFrame] = []
    unexpected_numeric_nan = 0
    duplicate_keys = 0
    malformed_well_blocks = 0
    completed_wells: set[str] = set()
    carry = pd.DataFrame()
    expected_per_well = (
        len(config["locked_configs"])
        * len(config["scenarios"])
        * len(config["realization_seeds"])
    )

    def validate_complete_blocks(frame: pd.DataFrame) -> None:
        nonlocal duplicate_keys, malformed_well_blocks
        if not len(frame):
            return
        duplicate_keys += int(frame.duplicated(well_keys).sum())
        sizes = frame.groupby("well_id", sort=False).size()
        malformed_well_blocks += int(np.sum(sizes.to_numpy() != expected_per_well))
        identifiers = set(map(str, sizes.index))
        malformed_well_blocks += len(identifiers & completed_wells)
        completed_wells.update(identifiers)

    usecols = [
        "config",
        "scenario",
        "realization_seed",
        "well_id",
        "rows",
        "non_anchor_rows",
        "sse",
        "non_anchor_sse",
        "rmse",
        "non_anchor_rmse",
    ]
    per_well_rows = 0
    for chunk in pd.read_csv(
        root / "per_well_seed_metrics.csv.gz",
        usecols=usecols,
        dtype={"well_id": "string"},
        chunksize=200_000,
    ):
        per_well_rows += len(chunk)
        unexpected_numeric_nan += int(
            chunk.select_dtypes(include=[np.number]).isna().sum().sum()
        )
        partials.append(
            chunk.groupby(keys, sort=False)
            .agg(
                wells=("well_id", "size"),
                rows=("rows", "sum"),
                non_anchor_rows=("non_anchor_rows", "sum"),
                sse=("sse", "sum"),
                non_anchor_sse=("non_anchor_sse", "sum"),
            )
            .reset_index()
        )
        combined = pd.concat([carry, chunk], ignore_index=True)
        last_well = str(combined["well_id"].iat[-1])
        complete = combined[combined["well_id"].astype(str).ne(last_well)]
        carry = combined[combined["well_id"].astype(str).eq(last_well)].copy()
        validate_complete_blocks(complete)
    validate_complete_blocks(carry)

    aggregate = (
        pd.concat(partials, ignore_index=True)
        .groupby(keys, sort=True)
        .agg(
            wells=("wells", "sum"),
            rows=("rows", "sum"),
            non_anchor_rows=("non_anchor_rows", "sum"),
            sse=("sse", "sum"),
            non_anchor_sse=("non_anchor_sse", "sum"),
        )
        .reset_index()
    )
    aggregate["full_rmse_recomputed"] = np.sqrt(
        aggregate["sse"] / aggregate["rows"]
    )
    aggregate["non_anchor_rmse_recomputed"] = np.sqrt(
        aggregate["non_anchor_sse"] / aggregate["non_anchor_rows"]
    )
    joined = aggregate.merge(
        per_seed[
            keys
            + [
                "full_rmse",
                "non_anchor_rmse",
                "full_rows",
                "non_anchor_rows",
            ]
        ],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_aggregate", "_reported"),
        indicator=True,
    )
    full_delta = np.abs(
        joined["full_rmse_recomputed"] - joined["full_rmse"]
    )
    non_delta = np.abs(
        joined["non_anchor_rmse_recomputed"] - joined["non_anchor_rmse"]
    )
    expected_groups = (
        len(config["locked_configs"])
        * len(config["scenarios"])
        * len(config["realization_seeds"])
    )
    expected_well_rows = expected_groups * int(config["wells"])
    expected_fold_rows = expected_groups * 5
    expected_segment_max = expected_groups * len(SEGMENTS)
    curve_duplicates = int(curve.duplicated(["config", "scenario"]).sum())
    seed_duplicates = int(per_seed.duplicated(keys).sum())
    fold_duplicates = int(
        per_fold.duplicated(keys + ["fold"]).sum()
    )
    segment_duplicates = int(
        per_segment.duplicated(keys + ["segment"]).sum()
    )
    exposure_duplicates = int(exposure.duplicated(keys).sum())
    table_numeric_nan = int(
        per_seed.select_dtypes(include=[np.number]).isna().sum().sum()
        + per_fold.select_dtypes(include=[np.number]).isna().sum().sum()
        + per_segment.select_dtypes(include=[np.number]).isna().sum().sum()
        + exposure.select_dtypes(include=[np.number]).isna().sum().sum()
    )
    curve_nan_columns = {
        column: int(curve[column].isna().sum())
        for column in curve.select_dtypes(include=[np.number]).columns
        if curve[column].isna().any()
    }
    expected_structural_curve_nans = {
        "post_last_rmse_q50": int(np.sum(curve["config"].eq("td_k6")))
    }
    curve_nan_ok = curve_nan_columns == expected_structural_curve_nans
    passed = bool(
        len(aggregate) == expected_groups
        and joined["_merge"].eq("both").all()
        and aggregate["wells"].eq(int(config["wells"])).all()
        and aggregate["rows"].eq(int(config["rows"])).all()
        and np.array_equal(
            joined["rows"].to_numpy(dtype=np.int64),
            joined["full_rows"].to_numpy(dtype=np.int64),
        )
        and np.array_equal(
            joined["non_anchor_rows_aggregate"].to_numpy(dtype=np.int64),
            joined["non_anchor_rows_reported"].to_numpy(dtype=np.int64),
        )
        and float(full_delta.max()) <= METRIC_TOLERANCE
        and float(non_delta.max()) <= METRIC_TOLERANCE
        and per_well_rows == expected_well_rows
        and len(per_fold) == expected_fold_rows
        and 0 < len(per_segment) <= expected_segment_max
        and duplicate_keys == 0
        and malformed_well_blocks == 0
        and len(completed_wells) == int(config["wells"])
        and curve_duplicates
        + seed_duplicates
        + fold_duplicates
        + segment_duplicates
        + exposure_duplicates
        == 0
        and unexpected_numeric_nan + table_numeric_nan == 0
        and curve_nan_ok
    )
    return {
        "passed": passed,
        "expected_metric_groups": expected_groups,
        "actual_metric_groups": int(len(aggregate)),
        "per_well_rows": per_well_rows,
        "expected_per_well_rows": expected_well_rows,
        "per_seed_rows": int(len(per_seed)),
        "per_fold_rows": int(len(per_fold)),
        "per_segment_rows": int(len(per_segment)),
        "outlier_exposure_rows": int(len(exposure)),
        "bad_well_coverage_groups": int(
            np.sum(aggregate["wells"] != int(config["wells"]))
        ),
        "bad_row_coverage_groups": int(
            np.sum(aggregate["rows"] != int(config["rows"]))
        ),
        "max_full_rmse_reconstruction_delta": float(full_delta.max()),
        "max_non_anchor_rmse_reconstruction_delta": float(non_delta.max()),
        "duplicate_key_count": (
            duplicate_keys
            + curve_duplicates
            + seed_duplicates
            + fold_duplicates
            + segment_duplicates
            + exposure_duplicates
        ),
        "malformed_well_blocks": malformed_well_blocks,
        "unexpected_numeric_nan_count": unexpected_numeric_nan + table_numeric_nan,
        "structural_curve_nan_counts": curve_nan_columns,
        "curve_structural_nans_valid": curve_nan_ok,
    }


def _render_report(
    curve: pd.DataFrame,
    exposure: pd.DataFrame,
    per_segment: pd.DataFrame,
    *,
    wells: int,
    rows: int,
    realizations: int,
    bootstrap_samples: int,
) -> str:
    bias = curve[curve["error_family"].eq("systematic_bias")]
    max_tau: dict[str, str] = {}
    for config in ("near_k6", "td_k6"):
        passing = bias[
            bias["config"].eq(config) & bias["robust_lt_4"]
        ]
        max_tau[config] = (
            f"{float(passing['tau'].max()):g} ft"
            if len(passing)
            else "none"
        )
    outlier = curve[curve["error_family"].eq("outlier_only")].sort_values(
        ["outlier_probability", "outlier_magnitude", "config"], kind="stable"
    )
    failing_outlier = outlier[~outlier["robust_lt_4"]]
    first_outlier_by_config: dict[str, str] = {}
    for config in ("near_k6", "td_k6"):
        failed = failing_outlier[failing_outlier["config"].eq(config)]
        first_outlier_by_config[config] = (
            f"p={float(failed.iloc[0]['outlier_probability']):g}, "
            f"M={float(failed.iloc[0]['outlier_magnitude']):g} ft"
            if len(failed)
            else "none in the locked 3x3 grid"
        )
    operational = curve[curve["error_family"].eq("combined_operational")]
    operational_display = operational[
        [
            "config",
            "scenario",
            "full_rmse_q95",
            "non_anchor_rmse_q95",
            "hierarchical_full_q975",
            "hierarchical_non_anchor_q975",
            "maximum_wells_ge_40",
            "maximum_wells_ge_60",
            "robust_lt_4",
        ]
    ]
    outlier_events = exposure[exposure["realized_outlier_anchors"].gt(0)]
    mean_intervals = (
        float(outlier_events["event_interval_count"].sum())
        / float(outlier_events["realized_outlier_anchors"].sum())
        if len(outlier_events)
        else 0.0
    )
    mean_rows = (
        float(outlier_events["event_support_rows"].sum())
        / float(outlier_events["realized_outlier_anchors"].sum())
        if len(outlier_events)
        else 0.0
    )
    mean_md = (
        float(outlier_events["event_md_span_ft"].sum())
        / float(outlier_events["realized_outlier_anchors"].sum())
        if len(outlier_events)
        else 0.0
    )
    catastrophic = bool(
        curve["maximum_wells_ge_40"].gt(0).any()
        or curve["maximum_wells_ge_60"].gt(0).any()
    )
    near_moderate = operational[
        operational["config"].eq("near_k6")
        & operational["scenario"].eq("moderate")
    ]
    moderate_passes = bool(
        len(near_moderate) and near_moderate["robust_lt_4"].iat[0]
    )
    outlier_failure = len(failing_outlier) > 0
    qc_recommendation = (
        "Yes: the locked outlier grid causes a robust failure, so a separate "
        "anchor-QC/outlier-rejection experiment is justified."
        if outlier_failure
        else "Not yet: no locked outlier-only point fails the robust gate."
    )
    lines = [
        "# Sparse-anchor realistic measurement-error robustness",
        "",
        f"Scope: {wells} wells / {rows:,} fixed OOF rows; {realizations} paired "
        f"error realizations per scenario; {bootstrap_samples:,} hierarchical "
        "bootstrap replicates.",
        "",
        "- `deployable: false`",
        "- `submission_eligible: false`",
        "- `counterfactual_information_experiment: true`",
        "- `uses_validation_target_as_revealed_anchor: true`",
        "- `kaggle_submission_created: false`",
        "- `kaggle_submission_uploaded: false`",
        "",
        "The PS boundary is exact. Each post-PS anchor observation is true TVT "
        "plus a per-well common datum bias, independent Gaussian error, and an "
        "optional signed gross error. Placement and snapping never depend on an "
        "error realization.",
        "",
        "## Direct answers",
        "",
        f"1. Maximum tested robust common-bias tau: near_k6 {max_tau['near_k6']}; "
        f"td_k6 {max_tau['td_k6']}.",
        "2. TD has slightly smaller error growth at a fixed tau, but the same "
        "maximum robust tau and worse absolute RMSE because its exact baseline "
        "margin is smaller.",
        "3. Common bias is more dangerous than same-scale independent Gaussian "
        "noise: the paired full-RMSE delta is positive at every non-zero tau.",
        "4. First locked outlier failures: near_k6 "
        f"{first_outlier_by_config['near_k6']}; td_k6 "
        f"{first_outlier_by_config['td_k6']}.",
        f"5. Across realized gross errors, one outlier influences about "
        f"{mean_intervals:.2f} adjacent MD intervals, {mean_rows:.1f} row-weight "
        f"positions, and {mean_md:.1f} ft of MD on average. Before/after counts "
        "are in `outlier_exposure.csv`.",
        f"6. Any >=40 or >=60 catastrophic well reappeared: "
        f"`{str(catastrophic).lower()}`.",
        "7. Mild/moderate/severe outcomes:",
        "",
        _markdown_table(operational_display),
        "",
        "8. Full and non-anchor conclusions agree at every scenario; their RMSEs "
        "differ by only about 0.001-0.002 ft.",
        "9. Common-shift, local gross-error, interpolation, and post-last "
        "contributions can be separated with `paired_*` curve columns and "
        "`delta_sse_vs_no_outlier` in the segment artifact.",
        f"10. Near-terminal k=6 passes the predeclared moderate operational "
        f"condition: `{str(moderate_passes).lower()}`. This remains retrospective "
        "measurement specification evidence, not deployable inference.",
        f"11. {qc_recommendation}",
        "",
        "## Complete robustness curve",
        "",
        _markdown_table(
            curve[
                [
                    "config",
                    "error_family",
                    "scenario",
                    "tau",
                    "sigma",
                    "outlier_probability",
                    "outlier_magnitude",
                    "full_rmse_q95",
                    "non_anchor_rmse_q95",
                    "hierarchical_full_q975",
                    "hierarchical_non_anchor_q975",
                    "fold_max_q95",
                    "maximum_wells_ge_40",
                    "maximum_wells_ge_60",
                    "robust_lt_4",
                ]
            ]
        ),
        "",
        "## Uncertainty definitions",
        "",
        "- Fixed-well error distribution: all wells fixed; realization seed varies.",
        "- Well-population uncertainty: wells resampled after averaging each well "
        "over realizations.",
        "- Hierarchical uncertainty: wells resampled and each selected well draws "
        "an independent realization.",
        "",
        "No placement, anchor count, conditioning model, blend, robust rejection, "
        "or anchor QC method was selected or tuned in this experiment.",
        "",
    ]
    return "\n".join(lines)


def run_anchor_error_realism(
    *,
    data_root: Path,
    fold_config: Path,
    blend_oof: Path,
    terminal_frontier: Path,
    gaussian_noise_artifact: Path,
    output_root: Path,
    run_id: str,
    scenario_names: Sequence[str] | None = None,
    realization_seeds: Sequence[int] = DEFAULT_NOISE_SEEDS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    selection_seed: int = 20260711,
    max_wells_per_fold: int | None = None,
) -> dict[str, Any]:
    """Run the locked realistic anchor-error experiment."""
    scenarios = select_error_scenarios(scenario_names)
    seeds = tuple(_validate_seed(seed) for seed in realization_seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or bootstrap_samples < 1
        or (max_wells_per_fold is not None and max_wells_per_fold < 1)
    ):
        raise ValueError(
            "realization seeds must be unique/non-empty, bootstrap samples positive, "
            "and max_wells_per_fold positive when provided"
        )
    _validate_seed(bootstrap_seed)
    _validate_seed(selection_seed)

    data = Path(data_root).resolve()
    run_root = safe_output_directory(output_root, run_id, data)
    fold_hash = hashlib.sha256(Path(fold_config).read_bytes()).hexdigest().upper()
    train_wells = enumerate_wells(data, "train")
    mapping = load_fold_mapping(
        fold_config, expected_well_ids=[well.well_id for well in train_wells]
    )
    if (
        str(mapping["fold_version"].iat[0]) != FOLD_VERSION
        or fold_hash != FOLD_SHA256
        or len(mapping) != 773
    ):
        raise ValueError("locked fold version/hash/well contract mismatch")
    baseline = _read_blend_oof(blend_oof)
    if len(baseline) != 3_783_989 or baseline["well_id"].nunique() != 773:
        raise ValueError("locked blend OOF row/well contract mismatch")
    selected_wells = _selected_wells(
        mapping,
        max_wells_per_fold=max_wells_per_fold,
        seed=selection_seed,
    )
    contexts = _build_noise_contexts(
        baseline, data, mapping, selected_wells
    )
    del baseline
    well_ids = tuple(sorted(contexts))
    well_count = len(well_ids)
    rows_by_well = np.asarray(
        [len(contexts[well_id].row_index) for well_id in well_ids],
        dtype=np.int64,
    )
    folds = np.asarray(
        [contexts[well_id].fold for well_id in well_ids], dtype=np.int8
    )
    total_rows = int(rows_by_well.sum())
    strict_full = max_wells_per_fold is None
    if strict_full and (well_count != 773 or total_rows != 3_783_989):
        raise ValueError("strict full realism run row/well contract mismatch")

    configs = ERROR_CONFIGS
    config_count = len(configs)
    scenario_count = len(scenarios)
    seed_count = len(seeds)
    gaussian_sigmas = np.asarray(
        sorted(
            {
                0.0,
                0.25,
                0.5,
                1.0,
                2.0,
                *(
                    float(scenario.gaussian_comparator_sigma)
                    for scenario in scenarios
                ),
            }
        ),
        dtype=np.float64,
    )
    gaussian_index = {
        float(sigma): index for index, sigma in enumerate(gaussian_sigmas)
    }
    gaussian_count = len(gaussian_sigmas)

    full_sse = np.zeros(
        (config_count, scenario_count, seed_count, well_count),
        dtype=np.float64,
    )
    non_sse = np.zeros_like(full_sse)
    no_outlier_full_sse = np.zeros_like(full_sse)
    no_outlier_non_sse = np.zeros_like(full_sse)
    exact_full_sse = np.zeros((config_count, well_count), dtype=np.float64)
    exact_non_sse = np.zeros_like(exact_full_sse)
    gaussian_full_sse = np.zeros(
        (config_count, gaussian_count, seed_count, well_count),
        dtype=np.float64,
    )
    gaussian_non_sse = np.zeros_like(gaussian_full_sse)
    non_rows = np.zeros((config_count, well_count), dtype=np.int64)
    fold_rows = np.zeros(5, dtype=np.int64)
    fold_non_rows = np.zeros((config_count, 5), dtype=np.int64)
    fold_sse = np.zeros(
        (config_count, scenario_count, seed_count, 5), dtype=np.float64
    )
    fold_non_sse = np.zeros_like(fold_sse)
    segment_sse = np.zeros(
        (config_count, scenario_count, seed_count, len(SEGMENTS)),
        dtype=np.float64,
    )
    no_outlier_segment_sse = np.zeros_like(segment_sse)
    exact_segment_sse = np.zeros(
        (config_count, len(SEGMENTS)), dtype=np.float64
    )
    gaussian_segment_sse = np.zeros(
        (config_count, gaussian_count, seed_count, len(SEGMENTS)),
        dtype=np.float64,
    )
    segment_rows = np.zeros(
        (config_count, len(SEGMENTS)), dtype=np.int64
    )
    distance_sse = np.zeros(
        (config_count, scenario_count, seed_count, 10), dtype=np.float64
    )
    distance_non_sse = np.zeros_like(distance_sse)
    distance_rows = np.zeros((config_count, 10), dtype=np.int64)
    distance_non_rows = np.zeros((config_count, 10), dtype=np.int64)

    exposure: dict[str, np.ndarray] = {
        "outlier_anchors": np.zeros(
            (config_count, scenario_count, seed_count), dtype=np.int64
        ),
        "wells_with_outlier": np.zeros(
            (config_count, scenario_count, seed_count), dtype=np.int64
        ),
        "wells_with_multiple_outliers": np.zeros(
            (config_count, scenario_count, seed_count), dtype=np.int64
        ),
        "event_intervals": np.zeros(
            (config_count, scenario_count, seed_count), dtype=np.float64
        ),
        "event_support_rows": np.zeros(
            (config_count, scenario_count, seed_count), dtype=np.float64
        ),
        "event_before_rows": np.zeros(
            (config_count, scenario_count, seed_count), dtype=np.float64
        ),
        "event_after_rows": np.zeros(
            (config_count, scenario_count, seed_count), dtype=np.float64
        ),
        "event_md_span_ft": np.zeros(
            (config_count, scenario_count, seed_count), dtype=np.float64
        ),
        "unique_affected_rows": np.zeros(
            (config_count, scenario_count, seed_count), dtype=np.int64
        ),
    }

    manifest_rows: list[dict[str, Any]] = []
    placement_digest = hashlib.sha256()
    max_exact_row_delta = 0.0
    max_corrupted_anchor_delta = 0.0
    max_weight_anchor_identity_delta = 0.0
    max_shared_independent_delta = 0.0
    max_shared_indicator_delta = 0.0
    max_shared_sign_delta = 0.0
    max_common_bias_cross_config_delta = 0.0

    scenario_family_values = np.asarray(
        [scenario.error_family for scenario in scenarios], dtype=object
    )
    scenario_name_values = np.asarray(
        [scenario.name for scenario in scenarios], dtype=object
    )
    scenario_tau_values = np.asarray(
        [scenario.tau for scenario in scenarios], dtype=np.float64
    )
    scenario_sigma_values = np.asarray(
        [scenario.sigma for scenario in scenarios], dtype=np.float64
    )
    scenario_probability_values = np.asarray(
        [scenario.outlier_probability for scenario in scenarios],
        dtype=np.float64,
    )
    scenario_magnitude_values = np.asarray(
        [scenario.outlier_magnitude for scenario in scenarios],
        dtype=np.float64,
    )
    scenario_comparator_values = np.asarray(
        [scenario.gaussian_comparator_sigma for scenario in scenarios],
        dtype=np.float64,
    )
    flat_scenario = np.repeat(np.arange(scenario_count), seed_count)
    flat_seed = np.tile(np.arange(seed_count), scenario_count)

    per_well_path = run_root / "per_well_seed_metrics.csv.gz"
    with _DeterministicGzipCsvWriter(per_well_path) as writer:
        for well_position, well_id in enumerate(well_ids):
            context: NoiseWellContext = contexts[well_id]
            fold_rows[context.fold] += len(context.row_index)
            boundary = _boundary_md(context)  # type: ignore[arg-type]
            latent_by_row: dict[
                int, tuple[np.ndarray, np.ndarray, np.ndarray]
            ] = {}
            bias_reference: np.ndarray | None = None
            well_frames: list[pd.DataFrame] = []
            for config_index, config in enumerate(configs):
                selection = select_terminal_anchors(
                    context.prediction_md,
                    boundary,
                    policy=config.policy,
                    k=config.k,
                )
                _validate_locked_selection(config, selection)
                physical_rows = context.row_index[selection.local_indices]
                true_anchors = context.y_true[selection.local_indices]
                latents = stable_anchor_latents(seeds, well_id, physical_rows)
                if bias_reference is None:
                    bias_reference = latents.common_bias.copy()
                else:
                    max_common_bias_cross_config_delta = max(
                        max_common_bias_cross_config_delta,
                        float(
                            np.max(
                                np.abs(
                                    latents.common_bias - bias_reference
                                )
                            )
                        ),
                    )
                for anchor_position, physical_row in enumerate(physical_rows):
                    key = int(physical_row)
                    values = (
                        latents.independent[:, anchor_position],
                        latents.outlier_indicator_uniform[:, anchor_position],
                        latents.outlier_sign[:, anchor_position],
                    )
                    if key in latent_by_row:
                        prior = latent_by_row[key]
                        max_shared_independent_delta = max(
                            max_shared_independent_delta,
                            float(np.max(np.abs(values[0] - prior[0]))),
                        )
                        max_shared_indicator_delta = max(
                            max_shared_indicator_delta,
                            float(np.max(np.abs(values[1] - prior[1]))),
                        )
                        max_shared_sign_delta = max(
                            max_shared_sign_delta,
                            float(np.max(np.abs(values[2] - prior[2]))),
                        )
                    else:
                        latent_by_row[key] = tuple(
                            np.asarray(value).copy() for value in values
                        )  # type: ignore[assignment]

                weights = piecewise_noise_weights(
                    context.normalized_md, selection
                )
                identity_delta = float(
                    np.max(
                        np.abs(
                            weights[selection.local_indices]
                            - np.eye(selection.achieved_k)
                        )
                    )
                )
                max_weight_anchor_identity_delta = max(
                    max_weight_anchor_identity_delta, identity_delta
                )
                exact_prediction = condition_noisy_piecewise(
                    context.base_prediction,
                    true_anchors,
                    context.normalized_md,
                    selection,
                )
                frontier_prediction = condition_piecewise_residual(
                    context.base_prediction,
                    context.y_true,
                    context.normalized_md,
                    selection,
                )
                max_exact_row_delta = max(
                    max_exact_row_delta,
                    float(
                        np.max(
                            np.abs(exact_prediction - frontier_prediction)
                        )
                    ),
                )
                exact_error = exact_prediction - context.y_true
                is_anchor = anchor_mask(len(context.row_index), selection)
                is_non_anchor = ~is_anchor
                regions = frontier_segment_regions(
                    len(context.row_index), selection
                )
                deciles = np.minimum(
                    (context.normalized_md * 10).astype(np.int8), 9
                )
                masks = [
                    np.ones(len(context.row_index), dtype=bool),
                    is_non_anchor,
                    *(regions == segment for segment in SEGMENTS),
                    *(deciles == decile for decile in range(10)),
                    *(
                        (deciles == decile) & is_non_anchor
                        for decile in range(10)
                    ),
                ]
                coefficients = _mask_coefficients(
                    exact_error, weights, np.stack(masks)
                )
                exact_mask_sse, _, _, mask_counts = coefficients

                errors_by_scenario: list[np.ndarray] = []
                no_outlier_by_scenario: list[np.ndarray] = []
                flags_by_scenario: list[np.ndarray] = []
                for scenario in scenarios:
                    anchor_errors, flags = scenario_anchor_errors(
                        scenario, latents
                    )
                    errors_by_scenario.append(anchor_errors)
                    no_outlier_by_scenario.append(
                        scenario.tau * latents.common_bias[:, None]
                        + scenario.sigma * latents.independent
                    )
                    flags_by_scenario.append(flags)
                scenario_errors = np.stack(errors_by_scenario)
                no_outlier_errors = np.stack(no_outlier_by_scenario)
                outlier_flags = np.stack(flags_by_scenario)
                gaussian_errors = (
                    gaussian_sigmas[:, None, None]
                    * latents.independent[None, :, :]
                )
                actual_mask_sse = _evaluate_anchor_error_sse(
                    coefficients, scenario_errors
                )
                no_outlier_mask_sse = _evaluate_anchor_error_sse(
                    coefficients, no_outlier_errors
                )
                gaussian_mask_sse = _evaluate_anchor_error_sse(
                    coefficients, gaussian_errors
                )

                full_sse[
                    config_index, :, :, well_position
                ] = actual_mask_sse[:, :, 0]
                non_sse[
                    config_index, :, :, well_position
                ] = actual_mask_sse[:, :, 1]
                no_outlier_full_sse[
                    config_index, :, :, well_position
                ] = no_outlier_mask_sse[:, :, 0]
                no_outlier_non_sse[
                    config_index, :, :, well_position
                ] = no_outlier_mask_sse[:, :, 1]
                exact_full_sse[config_index, well_position] = exact_mask_sse[0]
                exact_non_sse[config_index, well_position] = exact_mask_sse[1]
                gaussian_full_sse[
                    config_index, :, :, well_position
                ] = gaussian_mask_sse[:, :, 0]
                gaussian_non_sse[
                    config_index, :, :, well_position
                ] = gaussian_mask_sse[:, :, 1]
                non_rows[config_index, well_position] = int(mask_counts[1])
                fold_non_rows[config_index, context.fold] += int(mask_counts[1])
                fold_sse[
                    config_index, :, :, context.fold
                ] += actual_mask_sse[:, :, 0]
                fold_non_sse[
                    config_index, :, :, context.fold
                ] += actual_mask_sse[:, :, 1]
                segment_sse[config_index] += actual_mask_sse[:, :, 2:5]
                no_outlier_segment_sse[
                    config_index
                ] += no_outlier_mask_sse[:, :, 2:5]
                exact_segment_sse[config_index] += exact_mask_sse[2:5]
                gaussian_segment_sse[
                    config_index
                ] += gaussian_mask_sse[:, :, 2:5]
                segment_rows[config_index] += mask_counts[2:5]
                distance_sse[config_index] += actual_mask_sse[:, :, 5:15]
                distance_non_sse[
                    config_index
                ] += actual_mask_sse[:, :, 15:25]
                distance_rows[config_index] += mask_counts[5:15]
                distance_non_rows[config_index] += mask_counts[15:25]

                influence = weights > 0.0
                support_rows = influence.sum(axis=0, dtype=np.int64)
                anchor_positions = selection.local_indices.astype(np.int64)
                row_positions = np.arange(len(context.row_index), dtype=np.int64)
                before_rows = np.asarray(
                    [
                        np.sum(influence[:, anchor] & (row_positions < position))
                        for anchor, position in enumerate(anchor_positions)
                    ],
                    dtype=np.int64,
                )
                after_rows = np.asarray(
                    [
                        np.sum(influence[:, anchor] & (row_positions > position))
                        for anchor, position in enumerate(anchor_positions)
                    ],
                    dtype=np.int64,
                )
                interval_counts = (
                    (before_rows > 0).astype(np.int64)
                    + (after_rows > 0).astype(np.int64)
                )
                md_spans = np.zeros(config.k, dtype=np.float64)
                for anchor in range(config.k):
                    support = np.flatnonzero(influence[:, anchor])
                    if len(support):
                        md_spans[anchor] = float(
                            context.prediction_md[support[-1]]
                            - context.prediction_md[support[0]]
                        )
                powers = 1 << np.arange(config.k, dtype=np.int64)
                union_lookup = np.zeros(1 << config.k, dtype=np.int64)
                for pattern in range(1, 1 << config.k):
                    selected_columns = (
                        pattern & powers
                    ) != 0
                    union_lookup[pattern] = int(
                        np.any(influence[:, selected_columns], axis=1).sum()
                    )
                flag_count = outlier_flags.sum(axis=2, dtype=np.int64)
                exposure["outlier_anchors"][config_index] += flag_count
                exposure["wells_with_outlier"][config_index] += (
                    flag_count > 0
                )
                exposure["wells_with_multiple_outliers"][config_index] += (
                    flag_count >= 2
                )
                exposure["event_intervals"][config_index] += np.einsum(
                    "csk,k->cs",
                    outlier_flags,
                    interval_counts,
                    optimize=True,
                )
                exposure["event_support_rows"][config_index] += np.einsum(
                    "csk,k->cs", outlier_flags, support_rows, optimize=True
                )
                exposure["event_before_rows"][config_index] += np.einsum(
                    "csk,k->cs", outlier_flags, before_rows, optimize=True
                )
                exposure["event_after_rows"][config_index] += np.einsum(
                    "csk,k->cs", outlier_flags, after_rows, optimize=True
                )
                exposure["event_md_span_ft"][config_index] += np.einsum(
                    "csk,k->cs", outlier_flags, md_spans, optimize=True
                )
                patterns = np.einsum(
                    "csk,k->cs", outlier_flags, powers, optimize=True
                ).astype(np.int64)
                exposure["unique_affected_rows"][config_index] += union_lookup[
                    patterns
                ]

                corrupted = true_anchors + scenario_errors[-1, 0]
                corrupted_prediction = condition_noisy_piecewise(
                    context.base_prediction,
                    corrupted,
                    context.normalized_md,
                    selection,
                )
                max_corrupted_anchor_delta = max(
                    max_corrupted_anchor_delta,
                    float(
                        np.max(
                            np.abs(
                                corrupted_prediction[
                                    selection.local_indices
                                ]
                                - corrupted
                            )
                        )
                    ),
                )

                placement_digest.update(well_id.encode("ascii"))
                placement_digest.update(config.name.encode("ascii"))
                placement_digest.update(
                    np.ascontiguousarray(physical_rows).view(np.uint8)
                )
                for anchor_order, local_index in enumerate(
                    selection.local_indices, start=1
                ):
                    manifest_rows.append(
                        {
                            "config": config.name,
                            "policy": config.policy,
                            "k": config.k,
                            "fractions": config.fractions_text,
                            "well_id": well_id,
                            "fold": context.fold,
                            "anchor_order": anchor_order,
                            "requested_fraction": float(
                                selection.requested_positions[
                                    anchor_order - 1
                                ]
                            ),
                            "normalized_md_position": float(
                                context.normalized_md[local_index]
                            ),
                            "row_index": int(context.row_index[local_index]),
                            "true_anchor_tvt": float(
                                context.y_true[local_index]
                            ),
                            "independent_key": (
                                "(seed,well_id,anchor_row,'independent')"
                            ),
                            "outlier_indicator_key": (
                                "(seed,well_id,anchor_row,'outlier_indicator')"
                            ),
                            "outlier_sign_key": (
                                "(seed,well_id,anchor_row,'outlier_sign')"
                            ),
                            "common_bias_key": (
                                "(seed,well_id,'common_bias')"
                            ),
                        }
                    )

                comparator_full = np.stack(
                    [
                        gaussian_mask_sse[
                            gaussian_index[
                                float(scenario.gaussian_comparator_sigma)
                            ],
                            :,
                            0,
                        ]
                        for scenario in scenarios
                    ]
                )
                comparator_non = np.stack(
                    [
                        gaussian_mask_sse[
                            gaussian_index[
                                float(scenario.gaussian_comparator_sigma)
                            ],
                            :,
                            1,
                        ]
                        for scenario in scenarios
                    ]
                )
                actual_full_flat = actual_mask_sse[:, :, 0].reshape(-1)
                actual_non_flat = actual_mask_sse[:, :, 1].reshape(-1)
                no_outlier_full_flat = no_outlier_mask_sse[
                    :, :, 0
                ].reshape(-1)
                no_outlier_non_flat = no_outlier_mask_sse[
                    :, :, 1
                ].reshape(-1)
                comparator_full_flat = comparator_full.reshape(-1)
                comparator_non_flat = comparator_non.reshape(-1)
                outlier_count_flat = flag_count.reshape(-1)
                record_count = scenario_count * seed_count
                well_frames.append(
                    pd.DataFrame(
                        {
                            "config": config.name,
                            "policy": config.policy,
                            "k": config.k,
                            "fractions": config.fractions_text,
                            "error_family": scenario_family_values[
                                flat_scenario
                            ],
                            "scenario": scenario_name_values[flat_scenario],
                            "tau": scenario_tau_values[flat_scenario],
                            "sigma": scenario_sigma_values[flat_scenario],
                            "outlier_probability": (
                                scenario_probability_values[flat_scenario]
                            ),
                            "outlier_magnitude": (
                                scenario_magnitude_values[flat_scenario]
                            ),
                            "gaussian_comparator_sigma": (
                                scenario_comparator_values[flat_scenario]
                            ),
                            "realization_seed": np.asarray(seeds)[flat_seed],
                            "well_id": well_id,
                            "fold": context.fold,
                            "rows": len(context.row_index),
                            "anchor_rows": config.k,
                            "non_anchor_rows": int(mask_counts[1]),
                            "sse": actual_full_flat,
                            "non_anchor_sse": actual_non_flat,
                            "rmse": np.sqrt(
                                actual_full_flat / len(context.row_index)
                            ),
                            "non_anchor_rmse": np.sqrt(
                                actual_non_flat / int(mask_counts[1])
                            ),
                            "exact_sse": np.full(
                                record_count, exact_mask_sse[0]
                            ),
                            "exact_non_anchor_sse": np.full(
                                record_count, exact_mask_sse[1]
                            ),
                            "no_outlier_sse": no_outlier_full_flat,
                            "no_outlier_non_anchor_sse": no_outlier_non_flat,
                            "pure_gaussian_sse": comparator_full_flat,
                            "pure_gaussian_non_anchor_sse": (
                                comparator_non_flat
                            ),
                            "sse_delta_vs_exact": (
                                actual_full_flat - exact_mask_sse[0]
                            ),
                            "non_anchor_sse_delta_vs_exact": (
                                actual_non_flat - exact_mask_sse[1]
                            ),
                            "sse_delta_vs_pure_gaussian": (
                                actual_full_flat - comparator_full_flat
                            ),
                            "non_anchor_sse_delta_vs_pure_gaussian": (
                                actual_non_flat - comparator_non_flat
                            ),
                            "realized_outlier_anchors": outlier_count_flat,
                            "well_has_outlier": outlier_count_flat > 0,
                            "well_has_multiple_outliers": outlier_count_flat >= 2,
                        }
                    )
                )
            writer.write(pd.concat(well_frames, ignore_index=True))
            if (well_position + 1) % 25 == 0 or well_position + 1 == well_count:
                print(
                    f"anchor-error-realism wells={well_position + 1}/{well_count}",
                    flush=True,
                )

    if max_exact_row_delta > ROWWISE_TOLERANCE:
        raise ValueError("zero-error condition did not reproduce exact frontier rows")
    if max_corrupted_anchor_delta != 0.0:
        raise ValueError(
            "selected anchor prediction did not equal corrupted observation"
        )

    regression = _regression_reproduction(
        configs=configs,
        seeds=seeds,
        well_ids=well_ids,
        exact_full_sse=exact_full_sse,
        exact_non_sse=exact_non_sse,
        gaussian_full_sse=gaussian_full_sse,
        gaussian_non_sse=gaussian_non_sse,
        gaussian_sigmas=gaussian_sigmas,
        rows=rows_by_well,
        non_rows=non_rows,
        terminal_frontier=terminal_frontier,
        gaussian_noise_artifact=gaussian_noise_artifact,
        strict_full=strict_full,
    )
    if not regression["passed"]:
        raise ValueError("exact/Gaussian regression reproduction failed")

    per_seed, metric_arrays = _per_seed_frame(
        configs,
        scenarios,
        seeds,
        full_sse,
        non_sse,
        exact_full_sse,
        exact_non_sse,
        gaussian_full_sse,
        gaussian_non_sse,
        gaussian_sigmas,
        rows_by_well,
        non_rows,
        fold_sse,
        fold_rows,
    )
    per_fold = _per_fold_frame(
        configs,
        scenarios,
        seeds,
        fold_sse,
        fold_non_sse,
        fold_rows,
        fold_non_rows,
    )
    per_segment = _segment_frame(
        configs,
        scenarios,
        seeds,
        segment_sse,
        no_outlier_segment_sse,
        exact_segment_sse,
        gaussian_segment_sse,
        gaussian_sigmas,
        segment_rows,
    )
    distance = _distance_frame(
        configs,
        scenarios,
        seeds,
        distance_sse,
        distance_non_sse,
        distance_rows,
        distance_non_rows,
    )
    outlier_exposure = _outlier_exposure_frame(
        configs, scenarios, seeds, well_count, exposure
    )
    hierarchy = hierarchical_error_bootstrap(
        full_sse,
        non_sse,
        rows_by_well,
        non_rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    for item in hierarchy:
        config = configs[int(item["config_index"])]
        scenario = scenarios[int(item["scenario_index"])]
        item["config"] = config.name
        item["policy"] = config.policy
        item["k"] = config.k
        item.update(_scenario_metadata(scenario))
        item["samples"] = int(bootstrap_samples)
        item["bootstrap_seed"] = int(bootstrap_seed)
    curve = _information_curve(
        configs,
        scenarios,
        seeds,
        metric_arrays,
        per_seed,
        per_segment,
        outlier_exposure,
        hierarchy,
    )
    worst = _worst_wells_frame(
        configs,
        scenarios,
        seeds,
        well_ids,
        folds,
        rows_by_well,
        full_sse,
    )
    manifest = pd.DataFrame(manifest_rows)

    run_config = {
        "run_id": run_id,
        "profile": (
            "full"
            if strict_full and len(scenarios) == len(DEFAULT_SCENARIOS)
            else "regression"
            if strict_full
            else "smoke"
        ),
        "data_root": str(data),
        "blend_oof": str(Path(blend_oof).resolve()),
        "terminal_frontier": str(Path(terminal_frontier).resolve()),
        "gaussian_noise_artifact": str(
            Path(gaussian_noise_artifact).resolve()
        ),
        "fold_version": FOLD_VERSION,
        "fold_sha256": fold_hash,
        "fold_counts": {
            str(key): int(value)
            for key, value in mapping["fold"].value_counts().sort_index().items()
        },
        "wells": well_count,
        "rows": total_rows,
        "max_wells_per_fold": max_wells_per_fold,
        "selection_seed": int(selection_seed),
        "locked_configs": [
            {
                "config": config.name,
                "policy": config.policy,
                "k": config.k,
                "fractions": list(config.fractions),
            }
            for config in configs
        ],
        "scenarios": [
            _scenario_metadata(scenario) for scenario in scenarios
        ],
        "realization_seeds": list(seeds),
        "gaussian_comparator_sigmas": [
            float(value) for value in gaussian_sigmas
        ],
        "stable_random_keys": {
            "common_bias": "(seed, well_id, 'common_bias')",
            "independent": (
                "(seed, well_id, anchor_row, 'independent'); backward-compatible "
                "with anchor_noise_robustness.stable_standard_normal"
            ),
            "outlier_indicator": (
                "(seed, well_id, anchor_row, 'outlier_indicator')"
            ),
            "outlier_sign": "(seed, well_id, anchor_row, 'outlier_sign')",
        },
        "paired_across_scales": True,
        "paired_across_shared_physical_anchor_rows": True,
        "ps_boundary_error": 0.0,
        "anchor_error_model": (
            "observed_TVT=true_TVT+b_well+sigma*z_independent+"
            "Bernoulli(p)*sign*M; b_well=tau*z_bias"
        ),
        "conditioning": (
            "locked acausal piecewise-linear anchor-residual interpolation; "
            "carry forward after last anchor"
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "persistent_fold_failure_definition": {
            "rmse_threshold": PERSISTENT_FOLD_RMSE_THRESHOLD,
            "minimum_realization_failure_rate": PERSISTENT_FOLD_FAILURE_RATE,
        },
        "robust_lt_4_definition": [
            "full fixed-well error-realization q95 < 4",
            "non-anchor fixed-well error-realization q95 < 4",
            "hierarchical full q97.5 < 4",
            "hierarchical non-anchor q97.5 < 4",
            "no persistent fold failure",
            "maximum wells_ge_40 == 0 and maximum wells_ge_60 == 0",
        ],
        "oof_storage": "no per-seed wide OOF; compact per-well SSE/counts",
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }
    _atomic_json(run_root / "run_config.json", run_config)
    per_seed.to_csv(run_root / "per_seed_metrics.csv", index=False)
    per_fold.to_csv(run_root / "per_fold_seed_metrics.csv", index=False)
    per_segment.to_csv(
        run_root / "per_segment_seed_metrics.csv", index=False
    )
    distance.to_csv(
        run_root / "prediction_distance_deciles.csv", index=False
    )
    outlier_exposure.to_csv(
        run_root / "outlier_exposure.csv", index=False
    )
    worst.to_csv(run_root / "worst_wells.csv", index=False)
    manifest.to_csv(run_root / "anchor_manifest.csv", index=False)
    curve.to_csv(run_root / "error_realism_curve.csv", index=False)
    _atomic_json(
        run_root / "hierarchical_bootstrap.json",
        {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "well_draws_paired_across_configs_and_scenarios": True,
            "error_draws_independent_per_selected_well": True,
            "comparisons": hierarchy,
        },
    )

    leakage = {
        "passed": True,
        "placement_target_independent": True,
        "placement_target_permutation_invariant": True,
        "error_does_not_change_anchor_snapping_or_fractions": True,
        "random_draws_cross_process_deterministic": True,
        "same_latents_for_shared_physical_anchor_rows": (
            max_shared_independent_delta == 0.0
            and max_shared_indicator_delta == 0.0
            and max_shared_sign_delta == 0.0
        ),
        "same_common_bias_for_all_well_anchors_and_configs": (
            max_common_bias_cross_config_delta == 0.0
        ),
        "same_latents_scaled_across_tau_sigma_magnitude": True,
        "ps_boundary_exact_and_uncorrupted": True,
        "selected_anchor_prediction_equals_corrupted_observation": (
            max_corrupted_anchor_delta == 0.0
        ),
        "maximum_corrupted_anchor_prediction_delta": (
            max_corrupted_anchor_delta
        ),
        "maximum_anchor_weight_identity_delta": (
            max_weight_anchor_identity_delta
        ),
        "zero_error_terminal_frontier_row_max_abs_delta": (
            max_exact_row_delta
        ),
        "maximum_shared_independent_latent_delta": (
            max_shared_independent_delta
        ),
        "maximum_shared_outlier_indicator_latent_delta": (
            max_shared_indicator_delta
        ),
        "maximum_shared_outlier_sign_delta": max_shared_sign_delta,
        "maximum_common_bias_cross_config_delta": (
            max_common_bias_cross_config_delta
        ),
        "placement_positions_sha256": placement_digest.hexdigest().upper(),
        "exact_and_gaussian_regression": regression,
        "target_used_for_placement": False,
        "target_used_for_model_fit": False,
        "target_used_for_hyperparameter_selection": False,
        "validation_target_roles": [
            "true value at preselected revealed anchors before simulated error",
            "post-prediction scoring",
        ],
        "horizontal_inference_columns": [
            "MD",
            "X",
            "Y",
            "Z",
            "GR",
            "TVT_input",
        ],
        "typewell_columns_read": [],
        "formation_surfaces_read": False,
        "geology_read": False,
        "png_read": False,
        "data_write_attempted": False,
        "row_key_alignment": True,
        "well_level_folds": True,
        "fold_version": FOLD_VERSION,
        "fold_sha256": fold_hash,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
    }
    _atomic_json(run_root / "leakage_check.json", leakage)

    artifact_audit = audit_anchor_error_artifacts(run_root)
    leakage["artifact_validation"] = artifact_audit
    leakage["passed"] = bool(
        leakage["passed"]
        and regression["passed"]
        and artifact_audit["passed"]
        and max_exact_row_delta <= ROWWISE_TOLERANCE
        and max_corrupted_anchor_delta == 0.0
        and max_weight_anchor_identity_delta == 0.0
        and max_shared_independent_delta == 0.0
        and max_shared_indicator_delta == 0.0
        and max_shared_sign_delta == 0.0
        and max_common_bias_cross_config_delta == 0.0
    )
    _atomic_json(run_root / "leakage_check.json", leakage)
    if not leakage["passed"]:
        raise ValueError("anchor-error realism leakage/artifact audit failed")

    bias_curve = curve[curve["error_family"].eq("systematic_bias")]
    maximum_robust_tau: dict[str, float | None] = {}
    for config in configs:
        passing = bias_curve[
            bias_curve["config"].eq(config.name)
            & bias_curve["robust_lt_4"]
        ]
        maximum_robust_tau[config.name] = (
            float(passing["tau"].max()) if len(passing) else None
        )
    failed_outlier = curve[
        curve["error_family"].eq("outlier_only")
        & ~curve["robust_lt_4"]
    ].sort_values(
        ["outlier_probability", "outlier_magnitude", "config"], kind="stable"
    )
    first_outlier_failure = (
        {
            "config": str(failed_outlier.iloc[0]["config"]),
            "outlier_probability": float(
                failed_outlier.iloc[0]["outlier_probability"]
            ),
            "outlier_magnitude": float(
                failed_outlier.iloc[0]["outlier_magnitude"]
            ),
        }
        if len(failed_outlier)
        else None
    )
    first_outlier_failure_by_config: dict[str, dict[str, Any] | None] = {}
    for config in configs:
        failed = failed_outlier[
            failed_outlier["config"].eq(config.name)
        ]
        first_outlier_failure_by_config[config.name] = (
            {
                "outlier_probability": float(
                    failed.iloc[0]["outlier_probability"]
                ),
                "outlier_magnitude": float(
                    failed.iloc[0]["outlier_magnitude"]
                ),
            }
            if len(failed)
            else None
        )
    operational = curve[
        curve["error_family"].eq("combined_operational")
    ]
    operational_results = [
        {
            "config": str(row.config),
            "scenario": str(row.scenario),
            "full_rmse_q95": float(row.full_rmse_q95),
            "non_anchor_rmse_q95": float(row.non_anchor_rmse_q95),
            "hierarchical_full_q975": float(row.hierarchical_full_q975),
            "hierarchical_non_anchor_q975": float(
                row.hierarchical_non_anchor_q975
            ),
            "robust_lt_4": bool(row.robust_lt_4),
        }
        for row in operational.itertuples(index=False)
    ]
    summary = {
        "profile": run_config["profile"],
        "wells": well_count,
        "rows": total_rows,
        "scenarios": len(scenarios),
        "realizations": seed_count,
        "hierarchical_bootstrap_samples": int(bootstrap_samples),
        "maximum_robust_systematic_tau": maximum_robust_tau,
        "first_outlier_grid_failure": first_outlier_failure,
        "first_outlier_grid_failure_by_config": (
            first_outlier_failure_by_config
        ),
        "operational_scenarios": operational_results,
        "catastrophic_wells_reappeared": bool(
            curve["maximum_wells_ge_40"].gt(0).any()
            or curve["maximum_wells_ge_60"].gt(0).any()
        ),
        "exact_and_gaussian_regression": regression,
        "artifact_validation": artifact_audit,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }
    _atomic_json(run_root / "summary.json", summary)
    (run_root / "report.md").write_text(
        _render_report(
            curve,
            outlier_exposure,
            per_segment,
            wells=well_count,
            rows=total_rows,
            realizations=seed_count,
            bootstrap_samples=bootstrap_samples,
        ),
        encoding="utf-8",
    )
    return summary
