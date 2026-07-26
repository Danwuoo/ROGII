"""Fold-safe MVP for learned contextual TVT-state emissions.

This is deliberately a research probe rather than a registered production
model.  It trains an XGBoost hard-negative candidate classifier on outer-fold
training wells, inserts the learned margins into a signed-state lattice, and
scores posterior-mean and marginal-MAP paths on held-out wells.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import uniform_filter1d
from scipy.special import logsumexp
import xgboost
from xgboost import XGBClassifier

from rogii.data import enumerate_wells
from rogii.folds import (
    load_fold_mapping,
    partition_well_ids,
    stable_select,
)


STATE_RADIUS_FT = 128
STATE_COUNT = 2 * STATE_RADIUS_FT + 1
TRAIN_POSITIONS_PER_WELL = 120
INFERENCE_STRIDE = 20


@dataclass(frozen=True)
class Bundle:
    well_id: str
    horizontal: pd.DataFrame
    known_count: int
    anchor: float
    states: np.ndarray
    horizontal_descriptor: np.ndarray
    observed: np.ndarray
    typewell_descriptor: np.ndarray
    inside_typewell: np.ndarray
    self_descriptor: np.ndarray
    self_available: np.ndarray


def _descriptors(
    values: np.ndarray, widths: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(raw)
    filled = (
        pd.Series(raw)
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    if finite.any():
        center = float(np.mean(filled[finite]))
        scale = max(float(np.std(filled[finite])), 1e-6)
    else:
        center, scale = 0.0, 1.0
    standardized = (filled - center) / scale
    output = [standardized]
    for width in widths:
        size = min(width, len(standardized))
        mean = uniform_filter1d(standardized, size=size, mode="nearest")
        square = uniform_filter1d(
            standardized * standardized, size=size, mode="nearest"
        )
        output.extend([mean, np.sqrt(np.maximum(square - mean * mean, 0.0))])
    output.append(np.gradient(standardized))
    return np.column_stack(output), finite.astype(np.float64)


def _load_bundle(train_root: Path, well_id: str) -> Bundle:
    horizontal = pd.read_csv(
        train_root / f"{well_id}__horizontal_well.csv",
        usecols=["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"],
    )
    typewell = pd.read_csv(
        train_root / f"{well_id}__typewell.csv",
        usecols=["TVT", "GR"],
    )
    known_count = int(horizontal["TVT_input"].notna().sum())
    anchor = float(horizontal["TVT_input"].iloc[known_count - 1])
    states = anchor + np.arange(
        -STATE_RADIUS_FT, STATE_RADIUS_FT + 1, dtype=np.float64
    )
    horizontal_descriptor, observed = _descriptors(
        horizontal["GR"].to_numpy(dtype=np.float64), (11, 51, 201)
    )
    typewell_descriptor_raw, _ = _descriptors(
        typewell["GR"].to_numpy(dtype=np.float64), (21, 91, 271)
    )
    typewell_tvt = typewell["TVT"].to_numpy(dtype=np.float64)
    typewell_descriptor = np.column_stack(
        [
            np.interp(states, typewell_tvt, typewell_descriptor_raw[:, column])
            for column in range(typewell_descriptor_raw.shape[1])
        ]
    )
    inside_typewell = (states >= typewell_tvt[0]) & (states <= typewell_tvt[-1])

    self_descriptor = np.zeros_like(typewell_descriptor)
    self_available = np.zeros(len(states), dtype=np.float64)
    prefix_tvt = horizontal["TVT_input"].iloc[:known_count].to_numpy(
        dtype=np.float64
    )
    for state_index, state in enumerate(states):
        matched = np.abs(prefix_tvt - state) <= 0.75
        if int(matched.sum()) >= 3:
            self_descriptor[state_index] = np.median(
                horizontal_descriptor[:known_count][matched], axis=0
            )
            self_available[state_index] = min(1.0, float(matched.sum()) / 10.0)
    return Bundle(
        well_id=well_id,
        horizontal=horizontal,
        known_count=known_count,
        anchor=anchor,
        states=states,
        horizontal_descriptor=horizontal_descriptor,
        observed=observed,
        typewell_descriptor=typewell_descriptor,
        inside_typewell=inside_typewell,
        self_descriptor=self_descriptor,
        self_available=self_available,
    )


def _pair_features(
    bundle: Bundle, row: int, candidate_index: np.ndarray
) -> np.ndarray:
    candidate = np.asarray(candidate_index, dtype=np.int64)
    horizontal = bundle.horizontal_descriptor[row]
    typewell = bundle.typewell_descriptor[candidate]
    recurrence = bundle.self_descriptor[candidate]
    repeated = np.tile(horizontal, (len(candidate), 1))
    fraction = (row - (bundle.known_count - 1)) / max(
        len(bundle.horizontal) - bundle.known_count, 1
    )
    md = bundle.horizontal["MD"].to_numpy(dtype=np.float64)
    md_fraction = (md[row] - md[bundle.known_count - 1]) / max(
        md[-1] - md[bundle.known_count - 1], 1.0
    )
    z = bundle.horizontal["Z"].to_numpy(dtype=np.float64)
    delta_z = z[row] - z[bundle.known_count - 1]
    offset = (bundle.states[candidate] - bundle.anchor) / STATE_RADIUS_FT
    pieces = [
        repeated,
        typewell,
        np.abs(typewell - repeated),
        typewell * repeated,
        recurrence,
        np.abs(recurrence - repeated),
        bundle.self_available[candidate, None],
        offset[:, None],
        np.full((len(candidate), 1), fraction),
        np.full((len(candidate), 1), md_fraction),
        np.full((len(candidate), 1), delta_z / 200.0),
        np.full((len(candidate), 1), bundle.observed[row]),
    ]
    return np.concatenate(pieces, axis=1).astype(np.float32)


def _training_pairs(
    bundles: list[Bundle], *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for bundle in bundles:
        start = bundle.known_count
        positions = np.linspace(
            start,
            len(bundle.horizontal) - 1,
            min(TRAIN_POSITIONS_PER_WELL, len(bundle.horizontal) - start),
            dtype=np.int64,
        )
        truth = bundle.horizontal["TVT"].to_numpy(dtype=np.float64)
        valid = np.flatnonzero(bundle.inside_typewell)
        for row in positions:
            positive = int(np.argmin(np.abs(bundle.states - truth[row])))
            hard_order = np.argsort(
                np.abs(
                    bundle.typewell_descriptor[valid, 0]
                    - bundle.horizontal_descriptor[row, 0]
                )
            )
            hard = valid[hard_order[:30]]
            hard = hard[np.abs(hard - positive) > 3][:8]
            near = np.clip(
                positive + np.asarray([-20, -10, -5, 5, 10, 20]),
                0,
                STATE_COUNT - 1,
            )
            random_negative = rng.choice(valid, size=8, replace=False)
            candidate = np.unique(
                np.concatenate(([positive], hard, near, random_negative))
            )
            blocks.append(_pair_features(bundle, int(row), candidate))
            labels.append(
                (
                    np.abs(bundle.states[candidate] - truth[row]) <= 0.5
                ).astype(np.uint8)
            )
    return np.concatenate(blocks), np.concatenate(labels)


def _decode(
    model: XGBClassifier,
    bundle: Bundle,
    *,
    emission_weight: float,
    transition_sigma: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    start = bundle.known_count
    prediction_rows = np.arange(start, len(bundle.horizontal), dtype=np.int64)
    coarse = np.unique(
        np.concatenate(
            (
                [start - 1],
                prediction_rows[::INFERENCE_STRIDE],
                [prediction_rows[-1]],
            )
        )
    )
    emission = np.zeros((len(coarse), STATE_COUNT), dtype=np.float64)
    top1_square: list[float] = []
    top10_square: list[float] = []
    truth = bundle.horizontal["TVT"].to_numpy(dtype=np.float64)
    all_candidate = np.arange(STATE_COUNT, dtype=np.int64)
    for coarse_index, row in enumerate(coarse):
        margin = model.predict(
            _pair_features(bundle, int(row), all_candidate),
            output_margin=True,
        )
        emission[coarse_index] = emission_weight * margin
        emission[coarse_index, ~bundle.inside_typewell] -= 50.0
        if row >= start:
            order = np.argsort(margin)
            top1_square.append(
                float((bundle.states[order[-1]] - truth[row]) ** 2)
            )
            top10_square.append(
                float(
                    np.min(
                        (bundle.states[order[-10:]] - truth[row]) ** 2
                    )
                )
            )
    emission[0] -= 0.5 * ((bundle.states - bundle.anchor) / 0.25) ** 2
    moves = np.arange(-5, 6, dtype=np.int64)
    transition = -0.5 * (moves / transition_sigma) ** 2
    transition -= logsumexp(transition)

    alpha = np.full_like(emission, -np.inf)
    alpha[0] = emission[0]
    for row in range(1, len(coarse)):
        incoming = np.full((len(moves), STATE_COUNT), -np.inf)
        for move_index, move in enumerate(moves):
            if move >= 0:
                incoming[move_index, move:] = (
                    alpha[row - 1, : STATE_COUNT - move]
                    + transition[move_index]
                )
            else:
                incoming[move_index, : STATE_COUNT + move] = (
                    alpha[row - 1, -move:] + transition[move_index]
                )
        alpha[row] = emission[row] + logsumexp(incoming, axis=0)
        alpha[row] -= logsumexp(alpha[row])

    beta = np.zeros_like(emission)
    for row in range(len(coarse) - 2, -1, -1):
        incoming = np.full((len(moves), STATE_COUNT), -np.inf)
        following = emission[row + 1] + beta[row + 1]
        for move_index, move in enumerate(moves):
            if move >= 0:
                incoming[move_index, : STATE_COUNT - move] = (
                    following[move:] + transition[move_index]
                )
            else:
                incoming[move_index, -move:] = (
                    following[: STATE_COUNT + move] + transition[move_index]
                )
        beta[row] = logsumexp(incoming, axis=0)
        beta[row] -= logsumexp(beta[row])

    log_posterior = alpha + beta
    log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
    posterior = np.exp(log_posterior)
    posterior_mean = posterior @ bundle.states
    marginal_map = bundle.states[np.argmax(posterior, axis=1)]
    md = bundle.horizontal["MD"].to_numpy(dtype=np.float64)
    mean_prediction = np.interp(
        md[prediction_rows], md[coarse], posterior_mean
    )
    map_prediction = np.interp(
        md[prediction_rows], md[coarse], marginal_map
    )
    diagnostics = {
        "coarse_top1_depth_rmse": math.sqrt(float(np.mean(top1_square))),
        "coarse_top10_oracle_depth_rmse": math.sqrt(
            float(np.mean(top10_square))
        ),
        "mean_entropy": float(
            np.mean(
                -np.sum(
                    posterior
                    * np.log(
                        np.maximum(posterior, np.finfo(np.float64).tiny)
                    ),
                    axis=1,
                )
            )
        ),
    }
    return mean_prediction, map_prediction, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-train-wells", type=int, default=100)
    parser.add_argument("--max-validation-wells", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--emission-weight", type=float, default=0.5)
    parser.add_argument("--transition-sigma", type=float, default=0.5)
    args = parser.parse_args()
    started = time.perf_counter()
    data = args.data_root.resolve()
    output = args.output_root.resolve()
    if output == data or data in output.parents:
        raise ValueError("Probe output cannot be inside competition data")
    train_root = data / "train"
    well_ids = [well.well_id for well in enumerate_wells(data, "train")]
    mapping = load_fold_mapping(args.fold_config, expected_well_ids=well_ids)
    train_pool, validation_pool = partition_well_ids(mapping, args.fold)
    training_ids = stable_select(
        train_pool,
        args.max_train_wells,
        seed=args.seed,
        salt=f"fold-{args.fold}-train",
    )
    validation_ids = stable_select(
        validation_pool,
        args.max_validation_wells,
        seed=args.seed,
        salt=f"fold-{args.fold}-validation",
    )
    training_bundle = [_load_bundle(train_root, well) for well in training_ids]
    x, y = _training_pairs(training_bundle, seed=args.seed)
    classifier = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=4,
        random_state=args.seed,
        eval_metric="logloss",
    )
    classifier.fit(x, y, sample_weight=np.where(y == 1, 20.0, 1.0))

    rows: list[dict[str, float | int | str]] = []
    total_rows = 0
    flat_sse = mean_sse = map_sse = 0.0
    diagnostics: list[dict[str, float]] = []
    for well_id in validation_ids:
        bundle = _load_bundle(train_root, well_id)
        mean_prediction, map_prediction, diagnostic = _decode(
            classifier,
            bundle,
            emission_weight=args.emission_weight,
            transition_sigma=args.transition_sigma,
        )
        truth = bundle.horizontal["TVT"].iloc[
            bundle.known_count :
        ].to_numpy(dtype=np.float64)
        flat = np.full(len(truth), bundle.anchor)
        flat_error = truth - flat
        mean_error = truth - mean_prediction
        map_error = truth - map_prediction
        flat_well_sse = float(flat_error @ flat_error)
        mean_well_sse = float(mean_error @ mean_error)
        map_well_sse = float(map_error @ map_error)
        total_rows += len(truth)
        flat_sse += flat_well_sse
        mean_sse += mean_well_sse
        map_sse += map_well_sse
        diagnostics.append(diagnostic)
        rows.append(
            {
                "well_id": well_id,
                "n_rows": len(truth),
                "flat_rmse": math.sqrt(flat_well_sse / len(truth)),
                "posterior_mean_rmse": math.sqrt(
                    mean_well_sse / len(truth)
                ),
                "marginal_map_rmse": math.sqrt(map_well_sse / len(truth)),
            }
        )

    per_well = pd.DataFrame(rows)
    summary = {
        "scope": "fold_safe_contextual_emission_probe",
        "fold": args.fold,
        "training_wells": len(training_ids),
        "validation_wells": len(validation_ids),
        "validation_rows": total_rows,
        "training_pairs": len(y),
        "positive_pair_fraction": float(np.mean(y)),
        "flat_pooled_row_rmse": math.sqrt(flat_sse / total_rows),
        "posterior_mean_pooled_row_rmse": math.sqrt(mean_sse / total_rows),
        "marginal_map_pooled_row_rmse": math.sqrt(map_sse / total_rows),
        "coarse_top1_depth_rmse": float(
            np.mean([item["coarse_top1_depth_rmse"] for item in diagnostics])
        ),
        "coarse_top10_oracle_depth_rmse": float(
            np.mean(
                [
                    item["coarse_top10_oracle_depth_rmse"]
                    for item in diagnostics
                ]
            )
        ),
        "mean_entropy": float(
            np.mean([item["mean_entropy"] for item in diagnostics])
        ),
        "runtime_seconds": time.perf_counter() - started,
        "seed": args.seed,
        "emission_weight": args.emission_weight,
        "transition_sigma": args.transition_sigma,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "xgboost_version": xgboost.__version__,
    }
    output.mkdir(parents=True, exist_ok=True)
    per_well.to_csv(output / "per_well_metrics.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

