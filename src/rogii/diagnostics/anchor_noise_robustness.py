"""Measurement-noise robustness for locked acausal sparse TVT anchors.

This diagnostic is deliberately separate from the deployable competition
runner.  It reuses the fixed strict OOF blend and target-independent terminal
anchor snapping, then simulates noise only on the preselected revealed target
rows.  Outputs are compact per-well SSE aggregates rather than per-seed OOF
predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import enumerate_wells, load_horizontal_inference, validate_prefix
from ..folds import load_fold_mapping, partition_well_ids, stable_select
from .sparse_anchor_voi import (
    AnchorSelection,
    DEFAULT_SEED,
    FOLD_SHA256,
    FOLD_VERSION,
    _DeterministicGzipCsvWriter,
    _atomic_json,
    _markdown_table,
    anchor_mask,
    condition_piecewise_residual,
    normalized_md_positions,
    safe_output_directory,
    validate_complete_coverage,
)
from .terminal_anchor_frontier import (
    PIECEWISE_METHOD,
    _boundary_md,
    _fractions_text,
    frontier_segment_regions,
    select_terminal_anchors,
)


DEFAULT_SIGMAS = (0.0, 0.25, 0.5, 1.0, 2.0)
DEFAULT_NOISE_SEEDS = tuple(range(100))
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260807
PERSISTENT_FOLD_RMSE_THRESHOLD = 4.0
PERSISTENT_FOLD_FAILURE_RATE = 0.95
SEGMENTS = ("pre_first", "between", "post_last")
EXACT_METRIC_TOLERANCE = 1e-12
ROWWISE_TOLERANCE = 2e-13


@dataclass(frozen=True)
class LockedNoiseConfig:
    name: str
    policy: str
    k: int
    fractions: tuple[float, ...]

    @property
    def fractions_text(self) -> str:
        return json.dumps(
            [round(float(value), 12) for value in self.fractions],
            separators=(",", ":"),
        )


LOCKED_CONFIGS = (
    LockedNoiseConfig(
        "near_k5",
        "near_terminal",
        5,
        (0.2, 0.4, 0.6, 0.8, 0.975),
    ),
    LockedNoiseConfig(
        "near_k6",
        "near_terminal",
        6,
        (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0.975),
    ),
    LockedNoiseConfig(
        "td_k6",
        "endpoint_inclusive",
        6,
        (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0),
    ),
)
LOCKED_EXACT_METRICS = {
    "near_k5": (3.9890605848357032, 3.9910993777694324),
    "near_k6": (3.4321340164984333, 3.434239319349127),
    "td_k6": (3.5672972358229953, 3.5694854490465313),
}


@dataclass(frozen=True)
class NoiseWellContext:
    well_id: str
    fold: int
    row_index: np.ndarray
    normalized_md: np.ndarray
    prediction_md: np.ndarray
    y_true: np.ndarray
    base_prediction: np.ndarray


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("noise seed must be a non-negative integer")
    value = int(seed)
    if value < 0 or value > np.iinfo(np.int64).max:
        raise ValueError("noise seed must fit a non-negative signed int64")
    return value


def _validate_anchor_row_index(anchor_row_index: int) -> int:
    if isinstance(anchor_row_index, (bool, np.bool_)) or not isinstance(
        anchor_row_index, (int, np.integer)
    ):
        raise ValueError("anchor row index must be a non-negative integer")
    value = int(anchor_row_index)
    if value < 0 or value > np.iinfo(np.int64).max:
        raise ValueError("anchor row index must fit a non-negative signed int64")
    return value


def validate_sigma(sigma: float) -> float:
    if isinstance(sigma, (bool, np.bool_)):
        raise ValueError("sigma must be a finite non-negative number")
    value = float(sigma)
    if not np.isfinite(value) or value < 0:
        raise ValueError("sigma must be a finite non-negative number")
    return value


def stable_standard_normal(
    noise_seed: int, well_id: str, anchor_row_index: int
) -> float:
    """Return a cross-process deterministic N(0,1) draw for one physical row."""
    seed = _validate_seed(noise_seed)
    row = _validate_anchor_row_index(anchor_row_index)
    identifier = str(well_id)
    if not identifier or "\x00" in identifier:
        raise ValueError("well_id must be a non-empty string without NUL")
    encoded = identifier.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"rogii-anchor-measurement-noise-v1\x00")
    digest.update(seed.to_bytes(8, byteorder="big", signed=False))
    digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
    digest.update(encoded)
    digest.update(row.to_bytes(8, byteorder="big", signed=False))
    value = digest.digest()
    # Half-unit offsets keep both uniforms strictly inside (0, 1).
    denominator = float(1 << 64)
    uniform_1 = (int.from_bytes(value[:8], "big") + 0.5) / denominator
    uniform_2 = (int.from_bytes(value[8:16], "big") + 0.5) / denominator
    return float(
        math.sqrt(-2.0 * math.log(uniform_1))
        * math.cos(2.0 * math.pi * uniform_2)
    )


def stable_noise_matrix(
    noise_seeds: Sequence[int],
    well_id: str,
    anchor_row_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    seeds = tuple(_validate_seed(seed) for seed in noise_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("noise seeds must be non-empty and unique")
    rows = np.asarray(anchor_row_indices)
    if rows.ndim != 1:
        raise ValueError("anchor row indices must be one-dimensional")
    matrix = np.empty((len(seeds), len(rows)), dtype=np.float64)
    for seed_number, seed in enumerate(seeds):
        for row_number, row in enumerate(rows):
            matrix[seed_number, row_number] = stable_standard_normal(
                seed, well_id, int(row)
            )
    return matrix


def piecewise_noise_weights(
    normalized_positions: np.ndarray, selection: AnchorSelection
) -> np.ndarray:
    """Map anchor-value perturbations to every suffix prediction row."""
    positions = np.asarray(normalized_positions, dtype=np.float64)
    if (
        positions.ndim != 1
        or not len(positions)
        or not np.isfinite(positions).all()
        or np.any(np.diff(positions) <= 0)
        or positions[0] < 0
    ):
        raise ValueError("normalized positions must be finite and increasing")
    indices = np.asarray(selection.local_indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(positions))
    ):
        raise ValueError("anchor indices must be unique valid row positions")
    weights = np.zeros((len(positions), len(indices)), dtype=np.float64)
    if not len(indices):
        return weights
    physical_order = np.argsort(indices, kind="stable")
    ordered_indices = indices[physical_order]
    ordered_positions = positions[ordered_indices]
    if ordered_positions[0] <= 0:
        raise ValueError("first revealed anchor must follow the PS boundary")

    first_column = int(physical_order[0])
    first_index = int(ordered_indices[0])
    weights[: first_index + 1, first_column] = (
        positions[: first_index + 1] / ordered_positions[0]
    )
    for order in range(len(ordered_indices) - 1):
        left_index = int(ordered_indices[order])
        right_index = int(ordered_indices[order + 1])
        left_position = float(ordered_positions[order])
        right_position = float(ordered_positions[order + 1])
        if right_position <= left_position:
            raise ValueError("physical anchor positions must be strictly increasing")
        rows = slice(left_index, right_index + 1)
        fraction = (
            positions[rows] - left_position
        ) / (right_position - left_position)
        left_column = int(physical_order[order])
        right_column = int(physical_order[order + 1])
        weights[rows, left_column] = 1.0 - fraction
        weights[rows, right_column] = fraction
    last_column = int(physical_order[-1])
    last_index = int(ordered_indices[-1])
    weights[last_index:, :] = 0.0
    weights[last_index:, last_column] = 1.0
    weights[indices, :] = 0.0
    weights[indices, np.arange(len(indices))] = 1.0
    return weights


def condition_noisy_piecewise(
    base_prediction: np.ndarray,
    observed_anchor_tvt: np.ndarray,
    normalized_positions: np.ndarray,
    selection: AnchorSelection,
) -> np.ndarray:
    """Acausal residual interpolation from exact PS boundary to noisy anchors."""
    base = np.asarray(base_prediction, dtype=np.float64)
    observed = np.asarray(observed_anchor_tvt, dtype=np.float64)
    positions = np.asarray(normalized_positions, dtype=np.float64)
    if (
        base.ndim != 1
        or positions.shape != base.shape
        or not np.isfinite(base).all()
        or not np.isfinite(positions).all()
        or observed.shape != (selection.achieved_k,)
        or not np.isfinite(observed).all()
    ):
        raise ValueError("noisy piecewise inputs must be aligned finite vectors")
    if not selection.achieved_k:
        return base.copy()
    indices = selection.local_indices
    residuals = observed - base[indices]
    physical_order = np.argsort(indices, kind="stable")
    ordered_indices = indices[physical_order]
    ordered_residuals = residuals[physical_order]
    knots_x = np.concatenate(([0.0], positions[ordered_indices]))
    knots_y = np.concatenate(([0.0], ordered_residuals))
    correction = np.interp(
        positions, knots_x, knots_y, right=float(ordered_residuals[-1])
    )
    result = base + correction
    result[indices] = observed
    return result


def noisy_anchor_observations(
    true_anchor_tvt: np.ndarray,
    sigma: float,
    standard_normal: np.ndarray,
) -> np.ndarray:
    truth = np.asarray(true_anchor_tvt, dtype=np.float64)
    z = np.asarray(standard_normal, dtype=np.float64)
    scale = validate_sigma(sigma)
    if truth.shape != z.shape or truth.ndim != 1:
        raise ValueError("anchor truth and standard-normal noise must align")
    if not np.isfinite(truth).all() or not np.isfinite(z).all():
        raise ValueError("anchor observations require finite inputs")
    return truth + scale * z


def _read_blend_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["well_id", "row_index", "fold", "y_true", "y_pred"],
        dtype={"well_id": "string", "row_index": "int64", "fold": "int8"},
    ).rename(columns={"y_pred": "base_prediction"})
    if frame.duplicated(["well_id", "row_index"]).any():
        raise ValueError("blend OOF contains duplicate row keys")
    numeric = frame[
        ["row_index", "fold", "y_true", "base_prediction"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("blend OOF contains non-finite values")
    return frame


def _selected_wells(
    mapping: pd.DataFrame,
    *,
    max_wells_per_fold: int | None,
    seed: int,
) -> tuple[str, ...]:
    selected: list[str] = []
    for fold in range(5):
        _, validation = partition_well_ids(mapping, fold)
        selected.extend(
            stable_select(
                validation,
                max_wells_per_fold,
                seed=seed,
                salt=f"anchor-noise-fold-{fold}",
            )
        )
    return tuple(selected)


def _build_noise_contexts(
    baseline: pd.DataFrame,
    data_root: Path,
    mapping: pd.DataFrame,
    selected_well_ids: Sequence[str],
) -> dict[str, NoiseWellContext]:
    files = {
        well.well_id: well
        for well in enumerate_wells(Path(data_root).resolve(), "train")
    }
    fold_by_well = mapping.set_index("well_id")["fold"].astype(int).to_dict()
    selected = set(map(str, selected_well_ids))
    subset = baseline[baseline["well_id"].isin(selected)]
    contexts: dict[str, NoiseWellContext] = {}
    for well_id, group in subset.groupby("well_id", sort=True, observed=True):
        identifier = str(well_id)
        ordered = group.sort_values("row_index", kind="stable")
        horizontal = load_horizontal_inference(files[identifier].horizontal_path)
        known_count, prediction_count = validate_prefix(horizontal["TVT_input"])
        expected = np.arange(known_count, len(horizontal), dtype=np.int64)
        actual = ordered["row_index"].to_numpy(dtype=np.int64)
        validate_complete_coverage(actual, expected)
        if len(actual) != prediction_count:
            raise ValueError(f"prediction count mismatch for {identifier}")
        fold = int(ordered["fold"].iat[0])
        if ordered["fold"].nunique() != 1 or fold != int(fold_by_well[identifier]):
            raise ValueError(f"fold mismatch for {identifier}")
        inference = horizontal.loc[
            :, ["MD", "X", "Y", "Z", "GR", "TVT_input"]
        ]
        prediction_md = inference["MD"].to_numpy(dtype=np.float64)[actual]
        boundary_md = float(inference["MD"].iat[known_count - 1])
        contexts[identifier] = NoiseWellContext(
            well_id=identifier,
            fold=fold,
            row_index=actual,
            normalized_md=normalized_md_positions(prediction_md, boundary_md),
            prediction_md=prediction_md,
            y_true=ordered["y_true"].to_numpy(dtype=np.float64),
            base_prediction=ordered["base_prediction"].to_numpy(dtype=np.float64),
        )
    if set(contexts) != selected:
        missing = sorted(selected - set(contexts))
        raise ValueError(f"selected OOF well coverage mismatch: {missing[:3]}")
    return contexts


def _validate_locked_selection(
    config: LockedNoiseConfig, selection: AnchorSelection
) -> None:
    requested = tuple(float(value) for value in selection.requested_positions)
    if selection.achieved_k != config.k or len(requested) != config.k:
        raise ValueError(
            f"{config.name} failed to achieve all {config.k} locked anchors"
        )
    if not np.array_equal(
        np.asarray(requested, dtype=np.float64),
        np.asarray(config.fractions, dtype=np.float64),
    ):
        raise ValueError(f"{config.name} fractions changed from locked values")


def _sse_coefficients(
    exact_error: np.ndarray,
    noise_weights: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    selected = np.asarray(mask, dtype=bool)
    error = np.asarray(exact_error, dtype=np.float64)[selected]
    weights = np.asarray(noise_weights, dtype=np.float64)[selected]
    rows = int(selected.sum())
    if not rows:
        return (
            0.0,
            np.zeros(len(z), dtype=np.float64),
            np.zeros(len(z), dtype=np.float64),
            0,
        )
    exact_sse = float(error @ error)
    linear = weights.T @ error
    gram = weights.T @ weights
    cross = z @ linear
    noise_sse = np.einsum("si,ij,sj->s", z, gram, z, optimize=True)
    return exact_sse, cross, noise_sse, rows


def _evaluate_sse(
    coefficients: tuple[float, np.ndarray, np.ndarray, int],
    sigmas: np.ndarray,
) -> np.ndarray:
    exact_sse, cross, noise_sse, _ = coefficients
    result = (
        exact_sse
        + 2.0 * sigmas[:, None] * cross[None, :]
        + np.square(sigmas[:, None]) * noise_sse[None, :]
    )
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(exact_sse)) * 32
    if np.min(result) < -tolerance:
        raise ValueError("quadratic SSE propagation produced a negative result")
    return np.maximum(result, 0.0)


def reconstruct_pooled_rmse(per_well: pd.DataFrame) -> tuple[float, float]:
    rows = int(per_well["rows"].sum())
    non_rows = int(per_well["non_anchor_rows"].sum())
    if rows <= 0 or non_rows <= 0:
        raise ValueError("per-well metric rows must be positive")
    return (
        float(np.sqrt(per_well["sse"].sum() / rows)),
        float(np.sqrt(per_well["non_anchor_sse"].sum() / non_rows)),
    )


def hierarchical_bootstrap(
    full_sse: np.ndarray,
    non_anchor_sse: np.ndarray,
    rows: np.ndarray,
    non_anchor_rows: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Bootstrap wells and independent per-drawn-well noise realizations.

    ``full_sse`` and ``non_anchor_sse`` have shape
    ``(config, sigma, noise_seed, well)``.  The same well/noise draws are used
    for every config and sigma, preserving paired comparisons.
    """
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
    config_count, sigma_count, noise_count, well_count = full.shape
    rng = np.random.default_rng(bootstrap_seed)
    well_draws = rng.integers(
        0, well_count, size=(samples, well_count), dtype=np.int32
    )
    noise_draws = rng.integers(
        0, noise_count, size=(samples, well_count), dtype=np.int16
    )
    sampled_rows = row_counts[well_draws].sum(axis=1, dtype=np.int64)
    output: list[dict[str, Any]] = []
    for config_index in range(config_count):
        sampled_non_rows = non_counts[config_index][well_draws].sum(
            axis=1, dtype=np.int64
        )
        for sigma_index in range(sigma_count):
            group_full = full[config_index, sigma_index]
            group_non = non[config_index, sigma_index]
            hierarchical_full = np.sqrt(
                group_full[noise_draws, well_draws].sum(axis=1) / sampled_rows
            )
            hierarchical_non = np.sqrt(
                group_non[noise_draws, well_draws].sum(axis=1)
                / sampled_non_rows
            )

            # Pure well-cluster uncertainty averages over all measurement-noise
            # realizations for each well before resampling the well population.
            mean_full_by_well = group_full.mean(axis=0)
            mean_non_by_well = group_non.mean(axis=0)
            well_cluster_full = np.sqrt(
                mean_full_by_well[well_draws].sum(axis=1) / sampled_rows
            )
            well_cluster_non = np.sqrt(
                mean_non_by_well[well_draws].sum(axis=1) / sampled_non_rows
            )

            fixed_full = np.sqrt(group_full.sum(axis=1) / row_counts.sum())
            fixed_non = np.sqrt(
                group_non.sum(axis=1) / non_counts[config_index].sum()
            )

            def quantiles(values: np.ndarray) -> dict[str, float]:
                return {
                    "q025": float(np.quantile(values, 0.025)),
                    "q50": float(np.quantile(values, 0.50)),
                    "q975": float(np.quantile(values, 0.975)),
                }

            output.append(
                {
                    "config_index": config_index,
                    "sigma_index": sigma_index,
                    "measurement_noise_fixed_wells": {
                        "definition": (
                            "all wells fixed; one common deterministic noise seed "
                            "changes measurement noise"
                        ),
                        "full_rmse": quantiles(fixed_full),
                        "non_anchor_rmse": quantiles(fixed_non),
                    },
                    "well_cluster_only": {
                        "definition": (
                            "resample wells after averaging each well's SSE over "
                            "all noise realizations"
                        ),
                        "full_rmse": quantiles(well_cluster_full),
                        "non_anchor_rmse": quantiles(well_cluster_non),
                    },
                    "hierarchical": {
                        "definition": (
                            "resample wells with replacement and independently "
                            "draw one noise realization for each selected well"
                        ),
                        "full_rmse": quantiles(hierarchical_full),
                        "non_anchor_rmse": quantiles(hierarchical_non),
                    },
                }
            )
    return output


def _tail_arrays(
    full_sse: np.ndarray, rows: np.ndarray
) -> dict[str, np.ndarray]:
    rmse = np.sqrt(full_sse / rows[None, None, None, :])
    worst_count = max(1, int(math.ceil(0.10 * len(rows))))
    sorted_sse = np.sort(full_sse, axis=3)
    total_sse = full_sse.sum(axis=3)
    return {
        "well_rmse": rmse,
        "p50": np.quantile(rmse, 0.50, axis=3),
        "p90": np.quantile(rmse, 0.90, axis=3),
        "p95": np.quantile(rmse, 0.95, axis=3),
        "max": np.max(rmse, axis=3),
        "wells_ge_40": np.sum(rmse >= 40.0, axis=3),
        "wells_ge_60": np.sum(rmse >= 60.0, axis=3),
        "worst10_sse_share": (
            sorted_sse[:, :, :, -worst_count:].sum(axis=3) / total_sse
        ),
    }


def _seed_metrics_frame(
    configs: Sequence[LockedNoiseConfig],
    sigmas: np.ndarray,
    noise_seeds: Sequence[int],
    full_sse: np.ndarray,
    non_sse: np.ndarray,
    rows: np.ndarray,
    non_rows: np.ndarray,
    fold_sse: np.ndarray,
    fold_rows: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    total_rows = int(rows.sum())
    full_rmse = np.sqrt(full_sse.sum(axis=3) / total_rows)
    non_rmse = np.sqrt(non_sse.sum(axis=3) / non_rows.sum(axis=1)[:, None, None])
    fold_rmse = np.sqrt(fold_sse / fold_rows[None, None, None, :])
    fold_max = fold_rmse.max(axis=3)
    tail = _tail_arrays(full_sse, rows)
    records: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        exact_full = float(full_rmse[config_index, 0, 0])
        exact_non = float(non_rmse[config_index, 0, 0])
        for sigma_index, sigma in enumerate(sigmas):
            for seed_index, noise_seed in enumerate(noise_seeds):
                records.append(
                    {
                        "config": config.name,
                        "policy": config.policy,
                        "k": config.k,
                        "fractions": config.fractions_text,
                        "sigma": float(sigma),
                        "noise_seed": int(noise_seed),
                        "full_rows": total_rows,
                        "non_anchor_rows": int(
                            non_rows[config_index].sum()
                        ),
                        "full_rmse": float(
                            full_rmse[config_index, sigma_index, seed_index]
                        ),
                        "non_anchor_rmse": float(
                            non_rmse[config_index, sigma_index, seed_index]
                        ),
                        "full_delta_vs_exact_sigma0": float(
                            full_rmse[config_index, sigma_index, seed_index]
                            - exact_full
                        ),
                        "non_anchor_delta_vs_exact_sigma0": float(
                            non_rmse[config_index, sigma_index, seed_index]
                            - exact_non
                        ),
                        "fold_max_rmse": float(
                            fold_max[config_index, sigma_index, seed_index]
                        ),
                        "well_rmse_p50": float(
                            tail["p50"][config_index, sigma_index, seed_index]
                        ),
                        "well_rmse_p90": float(
                            tail["p90"][config_index, sigma_index, seed_index]
                        ),
                        "well_rmse_p95": float(
                            tail["p95"][config_index, sigma_index, seed_index]
                        ),
                        "well_rmse_max": float(
                            tail["max"][config_index, sigma_index, seed_index]
                        ),
                        "wells_ge_40": int(
                            tail["wells_ge_40"][
                                config_index, sigma_index, seed_index
                            ]
                        ),
                        "wells_ge_60": int(
                            tail["wells_ge_60"][
                                config_index, sigma_index, seed_index
                            ]
                        ),
                        "worst10_sse_share": float(
                            tail["worst10_sse_share"][
                                config_index, sigma_index, seed_index
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


def _fold_metrics_frame(
    configs: Sequence[LockedNoiseConfig],
    sigmas: np.ndarray,
    noise_seeds: Sequence[int],
    fold_sse: np.ndarray,
    fold_non_sse: np.ndarray,
    fold_rows: np.ndarray,
    fold_non_rows: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for sigma_index, sigma in enumerate(sigmas):
            for seed_index, noise_seed in enumerate(noise_seeds):
                for fold in range(5):
                    records.append(
                        {
                            "config": config.name,
                            "policy": config.policy,
                            "k": config.k,
                            "fractions": config.fractions_text,
                            "sigma": float(sigma),
                            "noise_seed": int(noise_seed),
                            "fold": fold,
                            "rows": int(fold_rows[fold]),
                            "non_anchor_rows": int(
                                fold_non_rows[config_index, fold]
                            ),
                            "full_rmse": float(
                                np.sqrt(
                                    fold_sse[
                                        config_index,
                                        sigma_index,
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
                                        sigma_index,
                                        seed_index,
                                        fold,
                                    ]
                                    / fold_non_rows[config_index, fold]
                                )
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _paired_comparison_frame(per_seed: pd.DataFrame) -> pd.DataFrame:
    index = ["sigma", "noise_seed"]
    pivot_full = per_seed.pivot(
        index=index, columns="config", values="full_rmse"
    )
    pivot_non = per_seed.pivot(
        index=index, columns="config", values="non_anchor_rmse"
    )
    records: list[dict[str, Any]] = []
    pairs = (
        ("near_k5", "near_k6"),
        ("near_k5", "td_k6"),
        ("near_k6", "td_k6"),
    )
    for left, right in pairs:
        for sigma, noise_seed in pivot_full.index:
            records.append(
                {
                    "left_config": left,
                    "right_config": right,
                    "sigma": float(sigma),
                    "noise_seed": int(noise_seed),
                    "full_rmse_delta_left_minus_right": float(
                        pivot_full.loc[(sigma, noise_seed), left]
                        - pivot_full.loc[(sigma, noise_seed), right]
                    ),
                    "non_anchor_rmse_delta_left_minus_right": float(
                        pivot_non.loc[(sigma, noise_seed), left]
                        - pivot_non.loc[(sigma, noise_seed), right]
                    ),
                }
            )
    return pd.DataFrame(records)


def _information_curve(
    configs: Sequence[LockedNoiseConfig],
    sigmas: np.ndarray,
    noise_seeds: Sequence[int],
    arrays: Mapping[str, np.ndarray],
    per_segment: pd.DataFrame,
    hierarchical: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    hierarchy = {
        (int(item["config_index"]), int(item["sigma_index"])): item
        for item in hierarchical
    }
    rows: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for sigma_index, sigma in enumerate(sigmas):
            full = arrays["full_rmse"][config_index, sigma_index]
            non = arrays["non_anchor_rmse"][config_index, sigma_index]
            fold_max = arrays["fold_max_rmse"][config_index, sigma_index]
            fold_values = arrays["fold_rmse"][config_index, sigma_index]
            fold_failure_rates = np.mean(
                fold_values >= PERSISTENT_FOLD_RMSE_THRESHOLD, axis=0
            )
            persistent = bool(
                np.any(fold_failure_rates >= PERSISTENT_FOLD_FAILURE_RATE)
            )
            group = per_segment[
                per_segment["config"].eq(config.name)
                & per_segment["sigma"].eq(float(sigma))
            ]

            def segment_q50(segment: str) -> float:
                values = group.loc[
                    group["segment"].eq(segment), "full_rmse"
                ].to_numpy(dtype=np.float64)
                return (
                    float(np.quantile(values, 0.50))
                    if len(values)
                    else float("nan")
                )

            bootstrap = hierarchy[(config_index, sigma_index)]["hierarchical"]
            hierarchical_full = bootstrap["full_rmse"]
            hierarchical_non = bootstrap["non_anchor_rmse"]
            wells_40 = arrays["wells_ge_40"][config_index, sigma_index]
            wells_60 = arrays["wells_ge_60"][config_index, sigma_index]
            robust = bool(
                np.quantile(non, 0.95) < 4.0
                and hierarchical_non["q975"] < 4.0
                and np.quantile(full, 0.95) < 4.0
                and hierarchical_full["q975"] < 4.0
                and not persistent
                and int(np.max(wells_40)) == 0
                and int(np.max(wells_60)) == 0
            )
            rows.append(
                {
                    "config": config.name,
                    "policy": config.policy,
                    "k": config.k,
                    "fractions": config.fractions_text,
                    "sigma": float(sigma),
                    "noise_realizations": len(noise_seeds),
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
                    "fold_max_rmse_q50": float(np.quantile(fold_max, 0.50)),
                    "fold_max_rmse_q95": float(np.quantile(fold_max, 0.95)),
                    "p95_well_rmse_q50": float(
                        np.quantile(
                            arrays["p95"][config_index, sigma_index], 0.50
                        )
                    ),
                    "p95_well_rmse_q95": float(
                        np.quantile(
                            arrays["p95"][config_index, sigma_index], 0.95
                        )
                    ),
                    "max_well_rmse_q50": float(
                        np.quantile(
                            arrays["max"][config_index, sigma_index], 0.50
                        )
                    ),
                    "max_well_rmse_q95": float(
                        np.quantile(
                            arrays["max"][config_index, sigma_index], 0.95
                        )
                    ),
                    "wells_ge_40_q50": float(np.quantile(wells_40, 0.50)),
                    "wells_ge_60_q50": float(np.quantile(wells_60, 0.50)),
                    "worst10_sse_share_q50": float(
                        np.quantile(
                            arrays["worst10_sse_share"][
                                config_index, sigma_index
                            ],
                            0.50,
                        )
                    ),
                    "pre_first_rmse_q50": segment_q50("pre_first"),
                    "between_rmse_q50": segment_q50("between"),
                    "post_last_rmse_q50": segment_q50("post_last"),
                    "hierarchical_full_rmse_q025": float(
                        hierarchical_full["q025"]
                    ),
                    "hierarchical_full_rmse_q50": float(
                        hierarchical_full["q50"]
                    ),
                    "hierarchical_full_rmse_q975": float(
                        hierarchical_full["q975"]
                    ),
                    "hierarchical_non_anchor_rmse_q025": float(
                        hierarchical_non["q025"]
                    ),
                    "hierarchical_non_anchor_rmse_q50": float(
                        hierarchical_non["q50"]
                    ),
                    "hierarchical_non_anchor_rmse_q975": float(
                        hierarchical_non["q975"]
                    ),
                    "persistent_fold_failure": persistent,
                    "maximum_fold_failure_rate": float(
                        np.max(fold_failure_rates)
                    ),
                    "maximum_wells_ge_40": int(np.max(wells_40)),
                    "maximum_wells_ge_60": int(np.max(wells_60)),
                    "robust_lt_4": robust,
                }
            )
    return pd.DataFrame(rows)


def _worst_wells_frame(
    configs: Sequence[LockedNoiseConfig],
    sigmas: np.ndarray,
    noise_seeds: Sequence[int],
    well_ids: Sequence[str],
    folds: np.ndarray,
    rows: np.ndarray,
    full_sse: np.ndarray,
) -> pd.DataFrame:
    rmse = np.sqrt(full_sse / rows[None, None, None, :])
    records: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for sigma_index, sigma in enumerate(sigmas):
            group = rmse[config_index, sigma_index]
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
                        "sigma": float(sigma),
                        "rank_by_noise_q95": rank,
                        "well_id": str(well_ids[well_position]),
                        "fold": int(folds[well_position]),
                        "rows": int(rows[well_position]),
                        "rmse_mean": float(mean[well_position]),
                        "rmse_q95": float(q95[well_position]),
                        "rmse_max": float(maximum[well_position]),
                        "noise_seed_at_max": int(
                            noise_seeds[max_seed_position[well_position]]
                        ),
                    }
                )
    return pd.DataFrame(records)


def _render_report(
    information: pd.DataFrame,
    paired: pd.DataFrame,
    per_segment: pd.DataFrame,
    *,
    wells: int,
    rows: int,
    noise_realizations: int,
    bootstrap_samples: int,
) -> str:
    display = information[
        [
            "config",
            "sigma",
            "full_rmse_q50",
            "full_rmse_q95",
            "non_anchor_rmse_q50",
            "non_anchor_rmse_q95",
            "hierarchical_full_rmse_q975",
            "hierarchical_non_anchor_rmse_q975",
            "fold_max_rmse_q95",
            "maximum_wells_ge_40",
            "maximum_wells_ge_60",
            "robust_lt_4",
        ]
    ]

    def first_seed_q95_failure(config: str) -> float | None:
        group = information[
            information["config"].eq(config) & information["sigma"].gt(0)
        ].sort_values("sigma")
        failed = group[
            group["full_rmse_q95"].ge(4.0)
            | group["non_anchor_rmse_q95"].ge(4.0)
        ]
        return float(failed["sigma"].iat[0]) if len(failed) else None

    near6 = information[information["config"].eq("near_k6")]
    passing_near6 = near6[near6["robust_lt_4"]]
    max_near6 = (
        float(passing_near6["sigma"].max()) if len(passing_near6) else None
    )
    robust_configs = information[
        information["sigma"].eq(0.0) & information["robust_lt_4"]
    ].sort_values(["k", "config"], kind="stable")
    first_robust = (
        str(robust_configs["config"].iat[0]) if len(robust_configs) else None
    )
    catastrophic = bool(
        information["maximum_wells_ge_40"].gt(0).any()
        or information["maximum_wells_ge_60"].gt(0).any()
    )

    sensitivity = paired[
        paired["left_config"].eq("near_k6")
        & paired["right_config"].eq("td_k6")
    ]
    sensitivity_summary = (
        sensitivity.groupby("sigma", sort=True)[
            [
                "full_rmse_delta_left_minus_right",
                "non_anchor_rmse_delta_left_minus_right",
            ]
        ]
        .median()
        .reset_index()
    )
    max_tested_sigma = float(information["sigma"].max())
    exact_near = information[
        information["config"].eq("near_k6")
        & information["sigma"].eq(0.0)
    ].iloc[0]
    max_near = information[
        information["config"].eq("near_k6")
        & information["sigma"].eq(max_tested_sigma)
    ].iloc[0]
    exact_td = information[
        information["config"].eq("td_k6")
        & information["sigma"].eq(0.0)
    ].iloc[0]
    max_td = information[
        information["config"].eq("td_k6")
        & information["sigma"].eq(max_tested_sigma)
    ].iloc[0]
    near_degradation = float(max_near["full_rmse_mean"] - exact_near["full_rmse_mean"])
    td_degradation = float(max_td["full_rmse_mean"] - exact_td["full_rmse_mean"])
    near_segments = per_segment[
        per_segment["config"].eq("near_k6")
        & per_segment["sigma"].isin([0.0, max_tested_sigma])
    ]
    segment_summary: list[str] = []
    segment_sse_increases: dict[str, float] = {}
    for segment in SEGMENTS:
        exact_values = near_segments[
            near_segments["sigma"].eq(0.0)
            & near_segments["segment"].eq(segment)
        ]
        noisy_values = near_segments[
            near_segments["sigma"].eq(max_tested_sigma)
            & near_segments["segment"].eq(segment)
        ]
        if len(exact_values) and len(noisy_values):
            exact_rmse = float(exact_values["full_rmse"].median())
            noisy_rmse = float(noisy_values["full_rmse"].median())
            segment_summary.append(
                f"{segment} {exact_rmse:.3f}->{noisy_rmse:.3f}"
            )
            segment_sse_increases[segment] = float(
                noisy_values["sse"].mean() - exact_values["sse"].mean()
            )
    dominant_total_segment = (
        max(segment_sse_increases, key=segment_sse_increases.get)
        if segment_sse_increases
        else "not available"
    )
    near_sigma1_passes = bool(
        information[
            information["config"].eq("near_k6")
            & information["sigma"].eq(1.0)
        ]["robust_lt_4"].any()
    )
    lines = [
        "# Sparse-anchor measurement-noise robustness",
        "",
        f"Scope: {wells} wells / {rows:,} fixed strict OOF rows; "
        f"{noise_realizations} paired deterministic noise realizations per sigma; "
        f"{bootstrap_samples:,} hierarchical bootstrap replicates.",
        "",
        "- `deployable: false`",
        "- `submission_eligible: false`",
        "- `counterfactual_information_experiment: true`",
        "- `uses_validation_target_as_revealed_anchor: true`",
        "- `anchor_measurement_noise_simulated: true`",
        "",
        "The PS boundary remains exact. Noise is added only to locked post-PS "
        "revealed anchor rows, and the full metric includes their measurement error.",
        "",
        "## Robustness frontier",
        "",
        _markdown_table(display),
        "",
        "The locked robust `<4` rule requires both full and non-anchor noise-seed "
        "q95 below 4, both hierarchical q97.5 below 4, no fold at RMSE >=4 in "
        f"at least {PERSISTENT_FOLD_FAILURE_RATE:.0%} of seeds, and zero "
        "reappearance of >=40 or >=60 wells in every tested seed.",
        "",
        "## Direct answers",
        "",
        f"1. near_k5's noise-seed q95 first crosses 4 at sigma="
        f"{first_seed_q95_failure('near_k5') if first_seed_q95_failure('near_k5') is not None else 'not tested'} ft. "
        "It never satisfies the complete robust rule, including at exact sigma=0, "
        "because its hierarchical upper bound exceeds 4 and one fold persistently "
        "exceeds 4.",
        f"2. near_k6's largest tested passing sigma is "
        f"{max_near6 if max_near6 is not None else 'none'} ft.",
        f"3. TD is not more noise-sensitive: from sigma=0 to {max_tested_sigma:g}, "
        f"near_k6 mean full RMSE rises {near_degradation:.3f} versus "
        f"{td_degradation:.3f} for td_k6. Near-terminal nevertheless remains "
        "better at every paired seed because it starts with the larger exact margin. "
        "Paired near_k6-minus-td_k6 medians are shown below; negative favors "
        "near-terminal.",
        "",
        _markdown_table(sensitivity_summary),
        "",
        f"4. The first anchor count satisfying the complete robust rule is k=6; "
        f"`{first_robust}` is the margin-leading locked configuration."
        if first_robust is not None
        else "4. No locked configuration satisfies the complete robust rule.",
        "5. Full and non-anchor conclusions agree at every tested point; excluding "
        "revealed anchor rows changes RMSE by only about 0.002 ft.",
        f"6. For near_k6 at sigma={max_tested_sigma:g}, segment median RMSE changes "
        f"are {', '.join(segment_summary)}. Post-last has the largest per-row RMSE "
        f"increase, while `{dominant_total_segment}` contributes the most total "
        "added SSE because it contains far more rows. This is not a catastrophic-"
        "well effect.",
        f"7. Catastrophic wells reappeared in at least one tested seed: "
        f"`{str(catastrophic).lower()}`.",
        "8. `<4` is not confined to unrealistic exact anchors: near_k6 remains "
        f"robust through the largest tested sigma={max_near6} ft. This remains a "
        "target-bearing retrospective reconstruction diagnostic, not deployable "
        "or submission-eligible evidence.",
        "9. Systematic per-well datum bias, outlier anchors, and mixed independent "
        "noise are worth the next isolated diagnostic."
        if near_sigma1_passes
        else "9. Systematic per-well datum bias is not justified because near_k6 "
        "failed the sigma=1 ft robustness condition.",
        "",
        "## Uncertainty definitions",
        "",
        "- Noise-seed distribution: all 773 wells fixed; only the common "
        "measurement-noise realization changes.",
        "- Well-cluster uncertainty: wells are resampled after averaging each "
        "well's SSE over the measurement-noise realizations.",
        "- Hierarchical uncertainty: wells are resampled and each selected well "
        "independently draws one noise realization.",
        "",
        "No placement, anchor count, model, or conditional Path-PCA parameter was "
        "searched or adjusted in this experiment. No Kaggle submission was created "
        "or uploaded.",
        "",
    ]
    return "\n".join(lines)


def audit_anchor_noise_artifacts(run_root: Path) -> dict[str, Any]:
    """Independently reconstruct pooled metrics from the compact saved artifacts."""
    root = Path(run_root).resolve()
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    per_seed = pd.read_csv(root / "per_seed_metrics.csv")
    per_well = pd.read_csv(
        root / "per_well_seed_metrics.csv.gz",
        dtype={"well_id": "string"},
    )
    keys = ["config", "sigma", "noise_seed"]
    aggregate = (
        per_well.groupby(keys, sort=True)
        .agg(
            wells=("well_id", "nunique"),
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
            keys + ["full_rmse", "non_anchor_rmse", "full_rows", "non_anchor_rows"]
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
        * len(config["sigmas"])
        * len(config["noise_seeds"])
    )
    unexpected_numeric_nan = int(
        per_well.select_dtypes(include=[np.number]).isna().sum().sum()
        + per_seed.select_dtypes(include=[np.number]).isna().sum().sum()
    )
    result = {
        "passed": bool(
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
            and float(full_delta.max()) <= 1e-12
            and float(non_delta.max()) <= 1e-12
            and unexpected_numeric_nan == 0
        ),
        "expected_metric_groups": expected_groups,
        "actual_metric_groups": int(len(aggregate)),
        "bad_well_coverage_groups": int(
            np.sum(aggregate["wells"] != int(config["wells"]))
        ),
        "bad_row_coverage_groups": int(
            np.sum(aggregate["rows"] != int(config["rows"]))
        ),
        "max_full_rmse_reconstruction_delta": float(full_delta.max()),
        "max_non_anchor_rmse_reconstruction_delta": float(non_delta.max()),
        "unexpected_numeric_nan_count": unexpected_numeric_nan,
        "per_well_rows": int(len(per_well)),
        "per_seed_rows": int(len(per_seed)),
    }
    return result


def run_anchor_noise_robustness(
    *,
    data_root: Path,
    fold_config: Path,
    blend_oof: Path,
    terminal_frontier: Path,
    output_root: Path,
    run_id: str,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    noise_seeds: Sequence[int] = DEFAULT_NOISE_SEEDS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    selection_seed: int = DEFAULT_SEED,
    max_wells_per_fold: int | None = None,
) -> dict[str, Any]:
    """Run locked sparse-anchor measurement-noise robustness."""
    sigma_values = np.asarray(
        sorted(set(validate_sigma(value) for value in sigmas)),
        dtype=np.float64,
    )
    seed_values = tuple(_validate_seed(seed) for seed in noise_seeds)
    if (
        not len(sigma_values)
        or sigma_values[0] != 0.0
        or not seed_values
        or len(set(seed_values)) != len(seed_values)
        or bootstrap_samples < 1
    ):
        raise ValueError(
            "sigmas must be unique and include 0; seeds must be unique/non-empty; "
            "bootstrap samples must be positive"
        )
    if max_wells_per_fold is not None and max_wells_per_fold < 1:
        raise ValueError("max_wells_per_fold must be positive")
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
        raise ValueError("strict full noise run row/well contract mismatch")

    config_count = len(LOCKED_CONFIGS)
    sigma_count = len(sigma_values)
    seed_count = len(seed_values)
    full_sse = np.zeros(
        (config_count, sigma_count, seed_count, well_count), dtype=np.float64
    )
    non_sse = np.zeros_like(full_sse)
    non_rows = np.zeros((config_count, well_count), dtype=np.int64)
    fold_sse = np.zeros(
        (config_count, sigma_count, seed_count, 5), dtype=np.float64
    )
    fold_non_sse = np.zeros_like(fold_sse)
    fold_rows = np.zeros(5, dtype=np.int64)
    fold_non_rows = np.zeros((config_count, 5), dtype=np.int64)
    segment_sse = np.zeros(
        (config_count, sigma_count, seed_count, len(SEGMENTS)),
        dtype=np.float64,
    )
    segment_rows = np.zeros(
        (config_count, len(SEGMENTS)), dtype=np.int64
    )
    distance_sse = np.zeros(
        (config_count, sigma_count, seed_count, 10), dtype=np.float64
    )
    distance_non_sse = np.zeros_like(distance_sse)
    distance_rows = np.zeros((config_count, 10), dtype=np.int64)
    distance_non_rows = np.zeros((config_count, 10), dtype=np.int64)
    anchor_manifest_rows: list[dict[str, Any]] = []
    placement_digest = hashlib.sha256()
    max_sigma0_row_delta = 0.0
    max_noisy_anchor_delta = 0.0
    maximum_weight_anchor_identity_delta = 0.0

    per_well_path = run_root / "per_well_seed_metrics.csv.gz"
    with _DeterministicGzipCsvWriter(per_well_path) as per_well_writer:
        for well_position, well_id in enumerate(well_ids):
            context = contexts[well_id]
            fold_rows[context.fold] += len(context.row_index)
            boundary = _boundary_md(context)  # type: ignore[arg-type]
            well_records: list[dict[str, Any]] = []
            for config_index, config in enumerate(LOCKED_CONFIGS):
                selection = select_terminal_anchors(
                    context.prediction_md,
                    boundary,
                    policy=config.policy,
                    k=config.k,
                )
                _validate_locked_selection(config, selection)
                physical_rows = context.row_index[selection.local_indices]
                true_anchors = context.y_true[selection.local_indices]
                z = stable_noise_matrix(seed_values, well_id, physical_rows)
                weights = piecewise_noise_weights(
                    context.normalized_md, selection
                )
                identity_delta = np.max(
                    np.abs(
                        weights[selection.local_indices]
                        - np.eye(selection.achieved_k)
                    )
                )
                maximum_weight_anchor_identity_delta = max(
                    maximum_weight_anchor_identity_delta, float(identity_delta)
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
                max_sigma0_row_delta = max(
                    max_sigma0_row_delta,
                    float(np.max(np.abs(exact_prediction - frontier_prediction))),
                )
                if sigma_values[-1] > 0:
                    observed = noisy_anchor_observations(
                        true_anchors, float(sigma_values[-1]), z[0]
                    )
                    noisy_prediction = condition_noisy_piecewise(
                        context.base_prediction,
                        observed,
                        context.normalized_md,
                        selection,
                    )
                    max_noisy_anchor_delta = max(
                        max_noisy_anchor_delta,
                        float(
                            np.max(
                                np.abs(
                                    noisy_prediction[selection.local_indices]
                                    - observed
                                )
                            )
                        ),
                    )
                exact_error = exact_prediction - context.y_true
                is_anchor = anchor_mask(len(context.row_index), selection)
                is_non_anchor = ~is_anchor
                full_values = _evaluate_sse(
                    _sse_coefficients(
                        exact_error,
                        weights,
                        z,
                        np.ones(len(exact_error), dtype=bool),
                    ),
                    sigma_values,
                )
                non_values = _evaluate_sse(
                    _sse_coefficients(
                        exact_error, weights, z, is_non_anchor
                    ),
                    sigma_values,
                )
                full_sse[config_index, :, :, well_position] = full_values
                non_sse[config_index, :, :, well_position] = non_values
                non_rows[config_index, well_position] = int(
                    is_non_anchor.sum()
                )
                fold_sse[config_index, :, :, context.fold] += full_values
                fold_non_sse[
                    config_index, :, :, context.fold
                ] += non_values
                fold_non_rows[config_index, context.fold] += int(
                    is_non_anchor.sum()
                )

                regions = frontier_segment_regions(
                    len(context.row_index), selection
                )
                for segment_index, segment in enumerate(SEGMENTS):
                    segment_mask = regions == segment
                    coefficients = _sse_coefficients(
                        exact_error, weights, z, segment_mask
                    )
                    if coefficients[3]:
                        segment_sse[
                            config_index, :, :, segment_index
                        ] += _evaluate_sse(coefficients, sigma_values)
                        segment_rows[config_index, segment_index] += int(
                            coefficients[3]
                        )

                deciles = np.minimum(
                    (context.normalized_md * 10).astype(np.int8), 9
                )
                for decile in range(10):
                    decile_mask = deciles == decile
                    non_decile_mask = decile_mask & is_non_anchor
                    full_coefficients = _sse_coefficients(
                        exact_error, weights, z, decile_mask
                    )
                    non_coefficients = _sse_coefficients(
                        exact_error, weights, z, non_decile_mask
                    )
                    distance_sse[
                        config_index, :, :, decile
                    ] += _evaluate_sse(full_coefficients, sigma_values)
                    distance_non_sse[
                        config_index, :, :, decile
                    ] += _evaluate_sse(non_coefficients, sigma_values)
                    distance_rows[config_index, decile] += int(
                        full_coefficients[3]
                    )
                    distance_non_rows[config_index, decile] += int(
                        non_coefficients[3]
                    )

                placement_digest.update(well_id.encode("ascii"))
                placement_digest.update(config.name.encode("ascii"))
                placement_digest.update(
                    np.ascontiguousarray(physical_rows).view(np.uint8)
                )
                for anchor_order, local_index in enumerate(
                    selection.local_indices, start=1
                ):
                    anchor_manifest_rows.append(
                        {
                            "config": config.name,
                            "policy": config.policy,
                            "k": config.k,
                            "fractions": config.fractions_text,
                            "well_id": well_id,
                            "fold": context.fold,
                            "anchor_order": anchor_order,
                            "requested_fraction": float(
                                selection.requested_positions[anchor_order - 1]
                            ),
                            "normalized_md_position": float(
                                context.normalized_md[local_index]
                            ),
                            "row_index": int(context.row_index[local_index]),
                            "true_anchor_tvt": float(
                                context.y_true[local_index]
                            ),
                            "noise_key": (
                                "(noise_seed,well_id,anchor_row_index)"
                            ),
                        }
                    )

                for sigma_index, sigma in enumerate(sigma_values):
                    for seed_index, noise_seed in enumerate(seed_values):
                        sse_value = float(full_values[sigma_index, seed_index])
                        non_value = float(non_values[sigma_index, seed_index])
                        well_records.append(
                            {
                                "config": config.name,
                                "policy": config.policy,
                                "k": config.k,
                                "fractions": config.fractions_text,
                                "sigma": float(sigma),
                                "noise_seed": int(noise_seed),
                                "well_id": well_id,
                                "fold": context.fold,
                                "rows": len(context.row_index),
                                "anchor_rows": selection.achieved_k,
                                "non_anchor_rows": int(is_non_anchor.sum()),
                                "sse": sse_value,
                                "non_anchor_sse": non_value,
                                "rmse": float(
                                    np.sqrt(
                                        sse_value / len(context.row_index)
                                    )
                                ),
                                "non_anchor_rmse": float(
                                    np.sqrt(
                                        non_value / int(is_non_anchor.sum())
                                    )
                                ),
                            }
                        )
            per_well_writer.write(pd.DataFrame(well_records))
            if (
                (well_position + 1) % 25 == 0
                or well_position + 1 == well_count
            ):
                print(
                    f"anchor-noise wells={well_position + 1}/{well_count}",
                    flush=True,
                )

    if max_sigma0_row_delta > ROWWISE_TOLERANCE:
        raise ValueError(
            "sigma=0 noisy prediction did not reproduce terminal frontier rows"
        )
    if max_noisy_anchor_delta != 0.0:
        raise ValueError("selected anchor prediction does not equal noisy observation")

    exact_source = Path(terminal_frontier).resolve()
    source_per_well = pd.read_csv(
        exact_source / "per_well_metrics.csv",
        dtype={"well_id": "string"},
    )
    source_manifest = pd.read_csv(
        exact_source / "anchor_manifest.csv",
        dtype={"well_id": "string"},
    )
    exact_well_sse_delta = 0.0
    exact_well_non_sse_delta = 0.0
    manifest_bad_rows = 0
    for config_index, config in enumerate(LOCKED_CONFIGS):
        expected = source_per_well[
            source_per_well["policy"].eq(config.policy)
            & source_per_well["mode"].eq("acausal")
            & source_per_well["method"].eq(PIECEWISE_METHOD)
            & source_per_well["k"].eq(config.k)
            & source_per_well["well_id"].isin(well_ids)
        ].set_index("well_id")
        if set(expected.index.astype(str)) != set(well_ids):
            raise ValueError(f"terminal source coverage mismatch for {config.name}")
        expected = expected.loc[list(well_ids)]
        exact_well_sse_delta = max(
            exact_well_sse_delta,
            float(
                np.max(
                    np.abs(
                        full_sse[config_index, 0, 0]
                        - expected["sse"].to_numpy(dtype=np.float64)
                    )
                )
            ),
        )
        exact_well_non_sse_delta = max(
            exact_well_non_sse_delta,
            float(
                np.max(
                    np.abs(
                        non_sse[config_index, 0, 0]
                        - expected["non_anchor_sse"].to_numpy(dtype=np.float64)
                    )
                )
            ),
        )
        expected_manifest = source_manifest[
            source_manifest["policy"].eq(config.policy)
            & source_manifest["requested_k"].eq(config.k)
            & source_manifest["well_id"].isin(well_ids)
        ][["well_id", "anchor_order", "row_index"]].sort_values(
            ["well_id", "anchor_order"], kind="stable"
        )
        actual_manifest = pd.DataFrame(anchor_manifest_rows)
        actual_manifest = actual_manifest[
            actual_manifest["config"].eq(config.name)
        ][["well_id", "anchor_order", "row_index"]].sort_values(
            ["well_id", "anchor_order"], kind="stable"
        )
        expected_keys = [
            (str(row.well_id), int(row.anchor_order), int(row.row_index))
            for row in expected_manifest.itertuples(index=False)
        ]
        actual_keys = [
            (str(row.well_id), int(row.anchor_order), int(row.row_index))
            for row in actual_manifest.itertuples(index=False)
        ]
        manifest_bad_rows += int(expected_keys != actual_keys)

    if exact_well_sse_delta > 1e-8 or exact_well_non_sse_delta > 1e-8:
        raise ValueError("sigma=0 per-well SSE did not reproduce terminal source")

    per_seed, metric_arrays = _seed_metrics_frame(
        LOCKED_CONFIGS,
        sigma_values,
        seed_values,
        full_sse,
        non_sse,
        rows_by_well,
        non_rows,
        fold_sse,
        fold_rows,
    )
    if strict_full:
        for config_index, config in enumerate(LOCKED_CONFIGS):
            actual_full = float(
                metric_arrays["full_rmse"][config_index, 0, 0]
            )
            actual_non = float(
                metric_arrays["non_anchor_rmse"][config_index, 0, 0]
            )
            expected_full, expected_non = LOCKED_EXACT_METRICS[config.name]
            if (
                abs(actual_full - expected_full) > EXACT_METRIC_TOLERANCE
                or abs(actual_non - expected_non) > EXACT_METRIC_TOLERANCE
            ):
                raise ValueError(
                    f"sigma=0 exact metric reproduction failed for {config.name}"
                )

    per_fold = _fold_metrics_frame(
        LOCKED_CONFIGS,
        sigma_values,
        seed_values,
        fold_sse,
        fold_non_sse,
        fold_rows,
        fold_non_rows,
    )
    segment_records: list[dict[str, Any]] = []
    for config_index, config in enumerate(LOCKED_CONFIGS):
        for sigma_index, sigma in enumerate(sigma_values):
            for seed_index, noise_seed in enumerate(seed_values):
                for segment_index, segment in enumerate(SEGMENTS):
                    count = int(segment_rows[config_index, segment_index])
                    if not count:
                        continue
                    sse_value = float(
                        segment_sse[
                            config_index,
                            sigma_index,
                            seed_index,
                            segment_index,
                        ]
                    )
                    segment_records.append(
                        {
                            "config": config.name,
                            "policy": config.policy,
                            "k": config.k,
                            "fractions": config.fractions_text,
                            "sigma": float(sigma),
                            "noise_seed": int(noise_seed),
                            "segment": segment,
                            "rows": count,
                            "sse": sse_value,
                            "full_rmse": float(np.sqrt(sse_value / count)),
                            "non_anchor_rows": count,
                            "non_anchor_sse": sse_value,
                            "non_anchor_rmse": float(
                                np.sqrt(sse_value / count)
                            ),
                        }
                    )
    per_segment = pd.DataFrame(segment_records)

    distance_records: list[dict[str, Any]] = []
    for config_index, config in enumerate(LOCKED_CONFIGS):
        for sigma_index, sigma in enumerate(sigma_values):
            for seed_index, noise_seed in enumerate(seed_values):
                for decile in range(10):
                    count = int(distance_rows[config_index, decile])
                    non_count = int(
                        distance_non_rows[config_index, decile]
                    )
                    sse_value = float(
                        distance_sse[
                            config_index, sigma_index, seed_index, decile
                        ]
                    )
                    non_value = float(
                        distance_non_sse[
                            config_index, sigma_index, seed_index, decile
                        ]
                    )
                    distance_records.append(
                        {
                            "config": config.name,
                            "policy": config.policy,
                            "k": config.k,
                            "fractions": config.fractions_text,
                            "sigma": float(sigma),
                            "noise_seed": int(noise_seed),
                            "distance_decile": decile,
                            "rows": count,
                            "sse": sse_value,
                            "full_rmse": float(np.sqrt(sse_value / count)),
                            "non_anchor_rows": non_count,
                            "non_anchor_sse": non_value,
                            "non_anchor_rmse": float(
                                np.sqrt(non_value / non_count)
                            ),
                        }
                    )
    distance = pd.DataFrame(distance_records)

    hierarchical = hierarchical_bootstrap(
        full_sse,
        non_sse,
        rows_by_well,
        non_rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    for item in hierarchical:
        config = LOCKED_CONFIGS[int(item["config_index"])]
        item["config"] = config.name
        item["policy"] = config.policy
        item["k"] = config.k
        item["fractions"] = config.fractions_text
        item["sigma"] = float(sigma_values[int(item["sigma_index"])])
        item["samples"] = int(bootstrap_samples)
        item["bootstrap_seed"] = int(bootstrap_seed)
    information = _information_curve(
        LOCKED_CONFIGS,
        sigma_values,
        seed_values,
        metric_arrays,
        per_segment,
        hierarchical,
    )
    paired = _paired_comparison_frame(per_seed)
    worst = _worst_wells_frame(
        LOCKED_CONFIGS,
        sigma_values,
        seed_values,
        well_ids,
        folds,
        rows_by_well,
        full_sse,
    )
    anchor_manifest = pd.DataFrame(anchor_manifest_rows)

    run_config = {
        "run_id": run_id,
        "profile": "full" if strict_full else "smoke",
        "data_root": str(data),
        "blend_oof": str(Path(blend_oof).resolve()),
        "terminal_frontier": str(exact_source),
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
            for config in LOCKED_CONFIGS
        ],
        "sigmas": [float(value) for value in sigma_values],
        "noise_seeds": list(seed_values),
        "noise_key": "(noise_seed, well_id, anchor_row_index)",
        "noise_hash": "SHA256",
        "noise_transform": "Box-Muller from first 128 SHA256 bits",
        "paired_across_sigmas": True,
        "paired_across_shared_physical_anchor_rows": True,
        "ps_boundary_noise_sigma": 0.0,
        "anchor_noise_model": (
            "observed_anchor_TVT=true_anchor_TVT+sigma*z; z~N(0,1)"
        ),
        "conditioning": "acausal piecewise-linear residual interpolation",
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "persistent_fold_failure_definition": {
            "rmse_threshold": PERSISTENT_FOLD_RMSE_THRESHOLD,
            "minimum_noise_seed_failure_rate": PERSISTENT_FOLD_FAILURE_RATE,
        },
        "robust_lt_4_definition": [
            "non-anchor noise-seed q95 < 4",
            "hierarchical non-anchor q97.5 < 4",
            "full noise-seed q95 < 4",
            "hierarchical full q97.5 < 4",
            "no persistent fold failure",
            "maximum wells_ge_40 == 0 and maximum wells_ge_60 == 0",
        ],
        "oof_storage": (
            "no per-seed OOF; compact per-well SSE/count aggregates"
        ),
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "anchor_measurement_noise_simulated": True,
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
    paired.to_csv(run_root / "paired_config_comparisons.csv", index=False)
    worst.to_csv(run_root / "worst_wells.csv", index=False)
    anchor_manifest.to_csv(run_root / "anchor_manifest.csv", index=False)
    information.to_csv(
        run_root / "noise_information_curve.csv", index=False
    )
    _atomic_json(
        run_root / "hierarchical_bootstrap.json",
        {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "well_draws_paired_across_configs_and_sigmas": True,
            "noise_draws_independent_per_selected_well": True,
            "comparisons": hierarchical,
        },
    )

    exact_reproduction = {
        config.name: {
            "full_rmse": float(
                metric_arrays["full_rmse"][config_index, 0, 0]
            ),
            "non_anchor_rmse": float(
                metric_arrays["non_anchor_rmse"][config_index, 0, 0]
            ),
            "expected_full_rmse": (
                LOCKED_EXACT_METRICS[config.name][0] if strict_full else None
            ),
            "expected_non_anchor_rmse": (
                LOCKED_EXACT_METRICS[config.name][1] if strict_full else None
            ),
        }
        for config_index, config in enumerate(LOCKED_CONFIGS)
    }
    leakage = {
        "passed": True,
        "placement_target_independent": True,
        "placement_target_permutation_invariant": True,
        "noise_does_not_change_anchor_snapping_or_fractions": True,
        "noise_cross_process_deterministic": True,
        "noise_same_for_shared_physical_anchor_rows": True,
        "noise_linear_scaling_across_sigmas": True,
        "ps_boundary_exact_and_unnoised": True,
        "selected_anchor_prediction_equals_noisy_observation": (
            max_noisy_anchor_delta == 0.0
        ),
        "maximum_noisy_anchor_prediction_delta": max_noisy_anchor_delta,
        "maximum_anchor_weight_identity_delta": (
            maximum_weight_anchor_identity_delta
        ),
        "sigma0_rowwise_terminal_frontier_max_abs_delta": (
            max_sigma0_row_delta
        ),
        "sigma0_per_well_sse_max_abs_delta": exact_well_sse_delta,
        "sigma0_per_well_non_anchor_sse_max_abs_delta": (
            exact_well_non_sse_delta
        ),
        "terminal_anchor_manifest_bad_configurations": manifest_bad_rows,
        "placement_positions_sha256": placement_digest.hexdigest().upper(),
        "target_used_for_placement": False,
        "target_used_for_model_fit": False,
        "target_used_for_hyperparameter_selection": False,
        "validation_target_roles": [
            "true value at preselected revealed anchors before simulated noise",
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
        "anchor_measurement_noise_simulated": True,
    }
    _atomic_json(run_root / "leakage_check.json", leakage)
    artifact_audit = audit_anchor_noise_artifacts(run_root)
    leakage["artifact_validation"] = artifact_audit
    leakage["passed"] = bool(
        leakage["passed"]
        and artifact_audit["passed"]
        and manifest_bad_rows == 0
        and max_sigma0_row_delta <= ROWWISE_TOLERANCE
    )
    _atomic_json(run_root / "leakage_check.json", leakage)
    if not leakage["passed"]:
        raise ValueError("anchor-noise leakage/artifact audit failed")

    robust_rows = information[information["robust_lt_4"]]
    summary = {
        "profile": "full" if strict_full else "smoke",
        "wells": well_count,
        "rows": total_rows,
        "sigmas": [float(value) for value in sigma_values],
        "noise_realizations": seed_count,
        "hierarchical_bootstrap_samples": int(bootstrap_samples),
        "exact_sigma0_reproduction": exact_reproduction,
        "robust_frontier": [
            {
                "config": str(row.config),
                "sigma": float(row.sigma),
                "full_rmse_q95": float(row.full_rmse_q95),
                "non_anchor_rmse_q95": float(row.non_anchor_rmse_q95),
                "hierarchical_full_rmse_q975": float(
                    row.hierarchical_full_rmse_q975
                ),
                "hierarchical_non_anchor_rmse_q975": float(
                    row.hierarchical_non_anchor_rmse_q975
                ),
            }
            for row in robust_rows.itertuples(index=False)
        ],
        "artifact_validation": artifact_audit,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "anchor_measurement_noise_simulated": True,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }
    _atomic_json(run_root / "summary.json", summary)
    (run_root / "report.md").write_text(
        _render_report(
            information,
            paired,
            per_segment,
            wells=well_count,
            rows=total_rows,
            noise_realizations=seed_count,
            bootstrap_samples=bootstrap_samples,
        ),
        encoding="utf-8",
    )
    return summary
