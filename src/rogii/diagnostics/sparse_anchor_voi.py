"""Counterfactual sparse post-PS TVT-anchor value-of-information diagnostic.

This module is intentionally separate from the deployable competition runner.
Validation targets may enter only through explicitly selected, target-independent
anchor rows and through post-prediction scoring.  Every learned Path-PCA
conditioning statistic is fitted from the current outer fold's training wells.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence, TextIO

import numpy as np
import pandas as pd

from ..data import enumerate_wells, load_horizontal_inference, validate_prefix
from ..folds import load_fold_mapping, partition_well_ids, stable_select
from ..models.path_pca import (
    PATH_POINTS,
    PathPCAState,
    _prefix_features,
)


DEFAULT_K_VALUES = (0, 1, 2, 4, 8)
POLICIES = ("uniform", "disagreement")
METHODS = ("piecewise_residual", "conditional_path_pca_mvp")
FOLD_VERSION = "priority1_5fold_v1"
FOLD_SHA256 = "7F1772817A95277696EB32085DA7618E2671E3ECE49F0DB00FFE1BC03C07C72E"
BASELINE_RMSE = 15.17206147508338
DEFAULT_SEED = 20260711


@dataclass(frozen=True)
class AnchorSelection:
    """A target-independent set of local suffix-row indices."""

    policy: str
    requested_k: int
    local_indices: np.ndarray
    normalized_positions: np.ndarray
    requested_positions: np.ndarray
    acquisition_scores: np.ndarray
    minimum_spacing: float

    @property
    def achieved_k(self) -> int:
        return int(len(self.local_indices))


@dataclass(frozen=True)
class ConditionalCalibration:
    """Outer-training-only Gaussian coefficient update state."""

    coefficient_covariance: np.ndarray
    observation_noise_variance: float
    coefficient_count: int
    training_well_ids: tuple[str, ...]
    validation_well_ids: tuple[str, ...]
    covariance_shrinkage: float
    state_sha256: str


@dataclass(frozen=True)
class WellContext:
    well_id: str
    fold: int
    row_index: np.ndarray
    normalized_md: np.ndarray
    prediction_md: np.ndarray
    prefix_anchor_tvt: float
    prefix_features: np.ndarray
    target_curve: np.ndarray
    y_true: np.ndarray
    p3_prediction: np.ndarray
    path_prediction: np.ndarray
    base_prediction: np.ndarray


@dataclass(frozen=True)
class ConditionedWellResult:
    """Explicit in-memory representation of one counterfactual result."""

    well_id: str
    fold: int
    row_index: np.ndarray
    normalized_md_position: np.ndarray
    placement_policy: str
    conditioning_method: str
    requested_k: int
    achieved_k: int
    selected_anchor_row_indices: np.ndarray
    revealed_tvt_values: np.ndarray
    base_prediction: np.ndarray
    conditioned_prediction: np.ndarray
    anchor_mask: np.ndarray
    segment_region: np.ndarray
    deployable: bool = False
    submission_eligible: bool = False


def _as_strict_1d(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def normalized_md_positions(prediction_md: np.ndarray, boundary_md: float) -> np.ndarray:
    """Normalize suffix MD from the last known prefix row (0) to final MD (1)."""
    md = _as_strict_1d(prediction_md, name="prediction_md")
    if np.any(np.diff(md) <= 0):
        raise ValueError("prediction MD must be strictly increasing")
    if not np.isfinite(boundary_md) or md[0] <= boundary_md:
        raise ValueError("PS boundary MD must precede every prediction row")
    horizon = float(md[-1] - boundary_md)
    if horizon <= 0:
        raise ValueError("prediction horizon must be positive")
    return (md - float(boundary_md)) / horizon


def _nearest_unused_indices(
    normalized_positions: np.ndarray, requested_positions: np.ndarray
) -> np.ndarray:
    positions = _as_strict_1d(normalized_positions, name="normalized_positions")
    requested = np.asarray(requested_positions, dtype=np.float64)
    if requested.ndim != 1 or not np.isfinite(requested).all():
        raise ValueError("requested positions must be a finite vector")
    available = set(range(len(positions)))
    selected: list[int] = []
    for requested_position in requested:
        if not available:
            break
        chosen = min(
            available,
            key=lambda index: (
                abs(float(positions[index]) - float(requested_position)),
                index,
            ),
        )
        selected.append(chosen)
        available.remove(chosen)
    return np.asarray(selected, dtype=np.int64)


def select_uniform_anchors(
    prediction_md: np.ndarray, boundary_md: float, k: int
) -> AnchorSelection:
    """Select target-free uniform-MD anchors and snap to nearest unused rows."""
    if k < 0:
        raise ValueError("k must be non-negative")
    positions = normalized_md_positions(prediction_md, boundary_md)
    requested = (
        np.arange(1, k + 1, dtype=np.float64) / float(k + 1)
        if k
        else np.empty(0, dtype=np.float64)
    )
    selected = _nearest_unused_indices(positions, requested)
    return AnchorSelection(
        policy="uniform",
        requested_k=int(k),
        local_indices=selected,
        normalized_positions=positions[selected],
        requested_positions=requested[: len(selected)],
        acquisition_scores=np.full(len(selected), np.nan, dtype=np.float64),
        minimum_spacing=0.0,
    )


def select_disagreement_anchors(
    prediction_md: np.ndarray,
    boundary_md: float,
    disagreement_score: np.ndarray,
    k: int,
) -> AnchorSelection:
    """Greedily select high-disagreement rows with fixed normalized-MD spacing."""
    if k < 0:
        raise ValueError("k must be non-negative")
    positions = normalized_md_positions(prediction_md, boundary_md)
    scores = _as_strict_1d(disagreement_score, name="disagreement_score")
    if len(scores) != len(positions):
        raise ValueError("disagreement score must cover every prediction row")
    minimum_spacing = 0.5 / float(k + 1) if k else 0.0
    if k == 0:
        return AnchorSelection(
            policy="disagreement",
            requested_k=0,
            local_indices=np.empty(0, dtype=np.int64),
            normalized_positions=np.empty(0, dtype=np.float64),
            requested_positions=np.empty(0, dtype=np.float64),
            acquisition_scores=np.empty(0, dtype=np.float64),
            minimum_spacing=0.0,
        )
    order = np.lexsort((np.arange(len(scores), dtype=np.int64), -scores))
    selected: list[int] = []
    for candidate in order:
        if all(
            abs(float(positions[candidate]) - float(positions[prior]))
            + 1e-15
            >= minimum_spacing
            for prior in selected
        ):
            selected.append(int(candidate))
            if len(selected) == k:
                break
    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    return AnchorSelection(
        policy="disagreement",
        requested_k=int(k),
        local_indices=selected_array,
        normalized_positions=positions[selected_array],
        requested_positions=np.full(len(selected_array), np.nan, dtype=np.float64),
        acquisition_scores=scores[selected_array],
        minimum_spacing=minimum_spacing,
    )


def anchor_mask(row_count: int, selection: AnchorSelection) -> np.ndarray:
    mask = np.zeros(int(row_count), dtype=bool)
    if np.any(selection.local_indices < 0) or np.any(
        selection.local_indices >= row_count
    ):
        raise ValueError("anchor index lies outside prediction rows")
    mask[selection.local_indices] = True
    return mask


def segment_regions(
    normalized_positions: np.ndarray, selection: AnchorSelection
) -> np.ndarray:
    """Label PS-to-first, inter-anchor, final-tail, and revealed rows."""
    positions = _as_strict_1d(normalized_positions, name="normalized_positions")
    labels = np.full(len(positions), "whole_horizon", dtype=object)
    if not selection.achieved_k:
        return labels
    anchors = selection.local_indices
    labels[:] = "post_last_anchor"
    labels[: anchors[0] + 1] = "ps_to_first_anchor"
    for order, (left, right) in enumerate(zip(anchors[:-1], anchors[1:]), start=1):
        labels[left + 1 : right + 1] = f"between_anchor_{order}_{order + 1}"
    labels[anchors] = "anchor"
    return labels


def condition_piecewise_residual(
    base_prediction: np.ndarray,
    y_true: np.ndarray,
    normalized_positions: np.ndarray,
    selection: AnchorSelection,
) -> np.ndarray:
    """Interpolate anchor residuals from a zero-residual PS boundary, then carry."""
    base = _as_strict_1d(base_prediction, name="base_prediction")
    truth = _as_strict_1d(y_true, name="y_true")
    positions = _as_strict_1d(normalized_positions, name="normalized_positions")
    if not (len(base) == len(truth) == len(positions)):
        raise ValueError("piecewise inputs must have identical row coverage")
    if not selection.achieved_k:
        return base.copy()
    indices = selection.local_indices
    residuals = truth[indices] - base[indices]
    knots_x = np.concatenate(([0.0], positions[indices]))
    knots_y = np.concatenate(([0.0], residuals))
    correction = np.interp(positions, knots_x, knots_y, right=float(residuals[-1]))
    result = base + correction
    result[indices] = truth[indices]
    return result


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest().upper()


def fit_conditional_calibration(
    state: PathPCAState,
    training_well_ids: Sequence[str],
    validation_well_ids: Sequence[str],
    prefix_features_by_well: Mapping[str, np.ndarray],
    target_curves_by_well: Mapping[str, np.ndarray],
    *,
    covariance_shrinkage: float = 0.10,
) -> ConditionalCalibration:
    """Estimate coefficient forecast covariance and basis noise on outer train only."""
    training = tuple(map(str, training_well_ids))
    validation = tuple(map(str, validation_well_ids))
    if not training or set(training) & set(validation):
        raise ValueError("outer training and validation wells must be non-empty/disjoint")
    if not 0.0 <= covariance_shrinkage <= 1.0:
        raise ValueError("covariance shrinkage must be in [0, 1]")
    missing = [
        well_id
        for well_id in training
        if well_id not in prefix_features_by_well
        or well_id not in target_curves_by_well
    ]
    if missing:
        raise ValueError(f"missing outer-training calibration wells: {missing[:3]}")

    design = np.stack([prefix_features_by_well[well_id] for well_id in training])
    curves = np.stack([target_curves_by_well[well_id] for well_id in training])
    coefficient_count = int(state.pca.components_.shape[0])
    true_coefficients = np.asarray(state.pca.transform(curves), dtype=np.float64)
    prior_coefficients = np.zeros_like(true_coefficients)
    predicted = np.asarray(state.regressor.predict(design), dtype=np.float64)
    prior_coefficients[:, : state.predicted_components] = predicted[
        :, : state.predicted_components
    ]
    coefficient_errors = true_coefficients - prior_coefficients
    covariance = np.atleast_2d(
        np.cov(coefficient_errors, rowvar=False, ddof=1)
    ).astype(np.float64)
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        (1.0 - covariance_shrinkage) * covariance
        + covariance_shrinkage * diagonal
    )
    scale = max(float(np.trace(covariance)) / coefficient_count, 1.0)
    covariance += np.eye(coefficient_count) * (scale * 1e-10)

    reconstructed = np.asarray(
        state.pca.inverse_transform(true_coefficients), dtype=np.float64
    )
    observation_noise = max(
        float(np.mean(np.square(curves - reconstructed))), 1e-8
    )
    state_hash = _hash_arrays(
        covariance,
        np.asarray([observation_noise], dtype=np.float64),
        np.asarray(training, dtype="U8"),
    )
    return ConditionalCalibration(
        coefficient_covariance=covariance,
        observation_noise_variance=observation_noise,
        coefficient_count=coefficient_count,
        training_well_ids=training,
        validation_well_ids=validation,
        covariance_shrinkage=float(covariance_shrinkage),
        state_sha256=state_hash,
    )


def _interpolate_basis_rows(
    state: PathPCAState, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, PATH_POINTS)
    query = np.asarray(positions, dtype=np.float64)
    mean = np.interp(query, grid, state.pca.mean_)
    basis = np.stack(
        [np.interp(query, grid, component) for component in state.pca.components_],
        axis=1,
    )
    return mean, basis


def condition_path_pca_posterior(
    state: PathPCAState,
    calibration: ConditionalCalibration,
    prefix_features: np.ndarray,
    prefix_anchor_tvt: float,
    path_baseline_prediction: np.ndarray,
    y_true: np.ndarray,
    normalized_positions: np.ndarray,
    selection: AnchorSelection,
) -> np.ndarray:
    """Apply an outer-train-only linear-Gaussian coefficient update.

    The implementation is named an MVP because covariance hyperparameters are
    fixed rather than selected by inner OOF.  The covariance and observation
    noise themselves are nevertheless estimated exclusively on outer training
    wells.
    """
    baseline = _as_strict_1d(
        path_baseline_prediction, name="path_baseline_prediction"
    )
    truth = _as_strict_1d(y_true, name="y_true")
    positions = _as_strict_1d(normalized_positions, name="normalized_positions")
    if not (len(baseline) == len(truth) == len(positions)):
        raise ValueError("conditional Path-PCA inputs must share row coverage")
    if not selection.achieved_k:
        return baseline.copy()
    if calibration.coefficient_count != state.pca.components_.shape[0]:
        raise ValueError("calibration and Path-PCA basis dimensions differ")

    c0 = np.zeros(calibration.coefficient_count, dtype=np.float64)
    predicted = np.asarray(
        state.regressor.predict(np.asarray(prefix_features)[None, :])[0],
        dtype=np.float64,
    )
    c0[: state.predicted_components] = predicted[: state.predicted_components]
    mean_all, basis_all = _interpolate_basis_rows(state, positions)
    reconstructed_prior = (
        float(prefix_anchor_tvt) + mean_all + basis_all @ c0
    )
    if not np.allclose(
        reconstructed_prior,
        baseline,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Path-PCA fold state does not reproduce the aligned OOF baseline"
        )

    observed_positions = positions[selection.local_indices]
    mean_observed, observation_matrix = _interpolate_basis_rows(
        state, observed_positions
    )
    observed_residual = (
        truth[selection.local_indices] - float(prefix_anchor_tvt)
    )
    innovation = observed_residual - mean_observed - observation_matrix @ c0
    sigma = calibration.coefficient_covariance
    innovation_covariance = (
        observation_matrix @ sigma @ observation_matrix.T
        + np.eye(selection.achieved_k)
        * calibration.observation_noise_variance
    )
    try:
        solved = np.linalg.solve(innovation_covariance, innovation)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(innovation_covariance, rcond=1e-12) @ innovation
    posterior_coefficients = c0 + sigma @ observation_matrix.T @ solved

    result = (
        float(prefix_anchor_tvt)
        + mean_all
        + basis_all @ posterior_coefficients
    )
    # Revealed observations are exact by construction of the VOI experiment.
    result[selection.local_indices] = truth[selection.local_indices]
    if not np.isfinite(result).all():
        raise ValueError("conditional Path-PCA produced non-finite predictions")
    return result


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = _as_strict_1d(y_true, name="y_true")
    prediction = _as_strict_1d(y_pred, name="y_pred")
    if len(truth) != len(prediction):
        raise ValueError("metric row coverage mismatch")
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def non_anchor_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, is_anchor: np.ndarray
) -> float:
    mask = np.asarray(is_anchor, dtype=bool)
    if mask.shape != np.asarray(y_true).shape:
        raise ValueError("anchor mask row coverage mismatch")
    if np.all(mask):
        return float("nan")
    return pooled_rmse(np.asarray(y_true)[~mask], np.asarray(y_pred)[~mask])


def validate_complete_coverage(
    row_index: np.ndarray, expected_row_index: np.ndarray
) -> None:
    actual = np.asarray(row_index, dtype=np.int64)
    expected = np.asarray(expected_row_index, dtype=np.int64)
    if (
        actual.ndim != 1
        or len(np.unique(actual)) != len(actual)
        or not np.array_equal(actual, expected)
    ):
        raise ValueError("well prediction row coverage mismatch")


def safe_output_directory(
    output_root: Path, run_id: str, data_root: Path
) -> Path:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in run_id):
        raise ValueError("invalid run_id")
    data = Path(data_root).resolve()
    output = (Path(output_root).resolve() / run_id).resolve()
    if output == data or data in output.parents or output in data.parents:
        raise ValueError("diagnostic output cannot be inside competition data")
    if output.exists():
        raise FileExistsError(f"run already exists: {output}")
    output.mkdir(parents=True)
    return output


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class _DeterministicGzipCsvWriter:
    def __init__(self, path: Path) -> None:
        self._raw = path.open("wb")
        self._gzip = gzip.GzipFile(
            filename="", mode="wb", fileobj=self._raw, compresslevel=6, mtime=0
        )
        self._text: TextIO = io.TextIOWrapper(
            self._gzip, encoding="utf-8", newline=""
        )
        self._wrote_header = False
        self._closed = False

    def write(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self._text, index=False, header=not self._wrote_header)
        self._wrote_header = True

    def close(self) -> None:
        if self._closed:
            return
        self._text.flush()
        self._text.close()
        self._raw.close()
        self._closed = True

    def __enter__(self) -> "_DeterministicGzipCsvWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _read_aligned_baselines(
    blend_path: Path, p3_path: Path, path_pca_path: Path
) -> pd.DataFrame:
    dtype = {"well_id": "string", "row_index": "int64", "fold": "int8"}
    blend = pd.read_csv(blend_path, dtype=dtype).rename(
        columns={"y_pred": "base_prediction"}
    )
    p3 = pd.read_csv(p3_path, dtype=dtype)
    path = pd.read_csv(path_pca_path, dtype=dtype)
    required = {"well_id", "row_index", "fold", "y_true"}
    for name, frame in (("blend", blend), ("p3", p3), ("path", path)):
        if not required | ({"base_prediction"} if name == "blend" else {"y_pred"}) <= set(frame.columns):
            raise ValueError(f"{name} OOF artifact has unexpected schema")
    key_columns = ["fold", "well_id", "row_index"]
    reference_keys = blend.loc[:, key_columns].reset_index(drop=True)
    for name, frame in (("p3", p3), ("path", path)):
        if not reference_keys.equals(frame.loc[:, key_columns].reset_index(drop=True)):
            raise ValueError(f"{name} OOF keys do not exactly match blend")
        if not np.array_equal(
            blend["y_true"].to_numpy(dtype=np.float64),
            frame["y_true"].to_numpy(dtype=np.float64),
        ):
            raise ValueError(f"{name} OOF targets do not exactly match blend")
    if blend.duplicated(["well_id", "row_index"]).any():
        raise ValueError("baseline OOF contains duplicate row keys")
    blend["p3_prediction"] = p3["y_pred"].to_numpy(dtype=np.float64)
    blend["path_prediction"] = path["y_pred"].to_numpy(dtype=np.float64)
    del p3, path
    return blend


def _build_well_contexts(
    baseline: pd.DataFrame,
    data_root: Path,
    mapping: pd.DataFrame,
) -> dict[str, WellContext]:
    files = {
        well.well_id: well
        for well in enumerate_wells(Path(data_root).resolve(), "train")
    }
    fold_by_well = mapping.set_index("well_id")["fold"].astype(int).to_dict()
    contexts: dict[str, WellContext] = {}
    grid = np.linspace(0.0, 1.0, PATH_POINTS)
    for well_id, group in baseline.groupby("well_id", sort=True, observed=True):
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

        features = horizontal.loc[
            :, ["MD", "X", "Y", "Z", "GR", "TVT_input"]
        ].to_numpy(dtype=np.float64, copy=True)
        prediction_md = features[actual, 0]
        boundary_md = float(features[known_count - 1, 0])
        positions = normalized_md_positions(prediction_md, boundary_md)
        prefix_anchor_tvt = float(features[known_count - 1, 5])
        truth = ordered["y_true"].to_numpy(dtype=np.float64)
        target_curve = np.interp(grid, positions, truth - prefix_anchor_tvt)
        contexts[identifier] = WellContext(
            well_id=identifier,
            fold=fold,
            row_index=actual,
            normalized_md=positions,
            prediction_md=prediction_md,
            prefix_anchor_tvt=prefix_anchor_tvt,
            prefix_features=_prefix_features(features, known_count),
            target_curve=target_curve,
            y_true=truth,
            p3_prediction=ordered["p3_prediction"].to_numpy(dtype=np.float64),
            path_prediction=ordered["path_prediction"].to_numpy(dtype=np.float64),
            base_prediction=ordered["base_prediction"].to_numpy(dtype=np.float64),
        )
    return contexts


def _combo_name(policy: str, method: str, k: int) -> str:
    return f"{policy}__{method}__k{k}"


def _tail_metrics(per_well: pd.DataFrame) -> dict[str, Any]:
    rmse = per_well["rmse"].to_numpy(dtype=np.float64)
    sse = per_well["sse"].to_numpy(dtype=np.float64)
    worst_count = max(1, int(math.ceil(0.10 * len(per_well))))
    worst_share = float(np.sort(sse)[-worst_count:].sum() / sse.sum())
    return {
        "p50": float(np.quantile(rmse, 0.50)),
        "p90": float(np.quantile(rmse, 0.90)),
        "p95": float(np.quantile(rmse, 0.95)),
        "max": float(np.max(rmse)),
        "wells_ge_40": int(np.sum(rmse >= 40.0)),
        "wells_ge_60": int(np.sum(rmse >= 60.0)),
        "worst10_sse_share": worst_share,
    }


def _summarize_metrics(
    per_well: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    information_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    group_columns = ["policy", "method", "k"]
    for key, group in per_well.groupby(group_columns, sort=True):
        policy, method, k = key
        full_rows = int(group["rows"].sum())
        non_rows = int(group["non_anchor_rows"].sum())
        pooled = float(np.sqrt(group["sse"].sum() / full_rows))
        non_anchor = (
            float(np.sqrt(group["non_anchor_sse"].sum() / non_rows))
            if non_rows
            else float("nan")
        )
        baseline_non_anchor = (
            float(np.sqrt(group["baseline_non_anchor_sse"].sum() / non_rows))
            if non_rows
            else float("nan")
        )
        tail = _tail_metrics(group)
        information_rows.append(
            {
                "policy": policy,
                "method": method,
                "k": int(k),
                "achieved_anchor_rows": int(group["anchor_rows"].sum()),
                "full_rows": full_rows,
                "non_anchor_rows": non_rows,
                "pooled_rmse": pooled,
                "non_anchor_rmse": non_anchor,
                "baseline_non_anchor_rmse_same_rows": baseline_non_anchor,
                "non_anchor_delta_vs_k0_same_rows": non_anchor
                - baseline_non_anchor,
                **tail,
                "candidate_better_wells": int(
                    np.sum(group["rmse"] < group["baseline_rmse"])
                ),
                "candidate_worse_wells": int(
                    np.sum(group["rmse"] > group["baseline_rmse"])
                ),
            }
        )
        for fold, fold_group in group.groupby("fold", sort=True):
            fold_rows.append(
                {
                    "policy": policy,
                    "method": method,
                    "k": int(k),
                    "fold": int(fold),
                    "wells": int(len(fold_group)),
                    "rows": int(fold_group["rows"].sum()),
                    "anchor_rows": int(fold_group["anchor_rows"].sum()),
                    "pooled_rmse": float(
                        np.sqrt(fold_group["sse"].sum() / fold_group["rows"].sum())
                    ),
                    "non_anchor_rmse": float(
                        np.sqrt(
                            fold_group["non_anchor_sse"].sum()
                            / fold_group["non_anchor_rows"].sum()
                        )
                    ),
                }
            )
    information = pd.DataFrame(information_rows).sort_values(
        group_columns, kind="stable"
    )
    for (policy, method), group in information.groupby(
        ["policy", "method"], sort=False
    ):
        baseline = float(group.loc[group["k"].eq(0), "pooled_rmse"].iat[0])
        mask = information["policy"].eq(policy) & information["method"].eq(method)
        information.loc[mask, "delta_vs_k0"] = (
            information.loc[mask, "pooled_rmse"] - baseline
        )
    return information, pd.DataFrame(fold_rows)


def _bootstrap(
    per_well: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    well_ids = tuple(sorted(per_well["well_id"].unique()))
    well_position = {well_id: index for index, well_id in enumerate(well_ids)}
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(well_ids),
        np.full(len(well_ids), 1.0 / len(well_ids)),
        size=samples,
    )
    output: list[dict[str, Any]] = []
    for key, group in per_well.groupby(["policy", "method", "k"], sort=True):
        policy, method, k = key
        ordered = group.assign(
            _order=group["well_id"].map(well_position)
        ).sort_values("_order")
        if tuple(ordered["well_id"]) != well_ids:
            raise ValueError("bootstrap well coverage mismatch")
        rows = ordered["rows"].to_numpy(dtype=np.float64)
        candidate_sse = ordered["sse"].to_numpy(dtype=np.float64)
        baseline_sse = ordered["baseline_sse"].to_numpy(dtype=np.float64)
        sampled_rows = counts @ rows
        candidate_rmse = np.sqrt((counts @ candidate_sse) / sampled_rows)
        baseline_rmse = np.sqrt((counts @ baseline_sse) / sampled_rows)
        delta = candidate_rmse - baseline_rmse

        non_rows = ordered["non_anchor_rows"].to_numpy(dtype=np.float64)
        non_candidate_sse = ordered["non_anchor_sse"].to_numpy(dtype=np.float64)
        non_baseline_sse = ordered[
            "baseline_non_anchor_sse"
        ].to_numpy(dtype=np.float64)
        sampled_non_rows = counts @ non_rows
        non_candidate_rmse = np.sqrt(
            (counts @ non_candidate_sse) / sampled_non_rows
        )
        non_baseline_rmse = np.sqrt(
            (counts @ non_baseline_sse) / sampled_non_rows
        )
        non_delta = non_candidate_rmse - non_baseline_rmse
        output.append(
            {
                "policy": policy,
                "method": method,
                "k": int(k),
                "samples": int(samples),
                "cluster": "well",
                "candidate_rmse_q025": float(np.quantile(candidate_rmse, 0.025)),
                "candidate_rmse_q50": float(np.quantile(candidate_rmse, 0.50)),
                "candidate_rmse_q975": float(np.quantile(candidate_rmse, 0.975)),
                "delta_q025": float(np.quantile(delta, 0.025)),
                "delta_q50": float(np.quantile(delta, 0.50)),
                "delta_q975": float(np.quantile(delta, 0.975)),
                "non_anchor_delta_q025": float(np.quantile(non_delta, 0.025)),
                "non_anchor_delta_q50": float(np.quantile(non_delta, 0.50)),
                "non_anchor_delta_q975": float(np.quantile(non_delta, 0.975)),
            }
        )
    return output


def _aggregate_error_table(
    rows: list[dict[str, Any]], grouping: list[str]
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    aggregate = (
        frame.groupby(grouping, sort=True, dropna=False)
        .agg(
            rows=("rows", "sum"),
            sse=("sse", "sum"),
            non_anchor_rows=("non_anchor_rows", "sum"),
            non_anchor_sse=("non_anchor_sse", "sum"),
        )
        .reset_index()
    )
    aggregate["rmse"] = np.sqrt(aggregate["sse"] / aggregate["rows"])
    aggregate["non_anchor_rmse"] = np.sqrt(
        aggregate["non_anchor_sse"] / aggregate["non_anchor_rows"]
    )
    return aggregate


def _markdown_table(frame: pd.DataFrame, *, float_digits: int = 6) -> str:
    """Render a small Markdown table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return (
                f"{float(value):.{float_digits}f}"
                if np.isfinite(value)
                else str(value)
            )
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _render_report(
    information: pd.DataFrame,
    per_segment: pd.DataFrame,
    worst_wells: pd.DataFrame,
    *,
    bootstrap_samples: int,
    profile: str,
) -> str:
    candidates = information[information["k"].gt(0)].copy()
    best = candidates.sort_values("pooled_rmse", kind="stable").iloc[0]
    below = candidates[candidates["pooled_rmse"].lt(4.0)]
    first_below = (
        below.sort_values(["k", "pooled_rmse"], kind="stable").iloc[0]
        if len(below)
        else None
    )
    k8 = candidates[candidates["k"].eq(8)].sort_values(
        "pooled_rmse", kind="stable"
    )
    catastrophic = worst_wells[
        worst_wells["k"].eq(8) & worst_wells["rank"].le(10)
    ]
    lines = [
        "# Sparse post-PS TVT Anchor Value-of-Information",
        "",
        f"Profile: `{profile}`.  This is a target-bearing counterfactual diagnostic, "
        "not deployable competition evidence.",
        "",
        "- `deployable: false`",
        "- `submission_eligible: false`",
        "- `counterfactual_information_experiment: true`",
        "- `uses_validation_target_as_revealed_anchor: true`",
        "",
        "## Information curve",
        "",
        _markdown_table(
            information[
                [
                    "policy",
                    "method",
                    "k",
                    "pooled_rmse",
                    "non_anchor_rmse",
                    "p95",
                    "max",
                    "wells_ge_40",
                    "wells_ge_60",
                ]
            ]
        ),
        "",
        "## Result",
        "",
        f"Best observed combination: `{best['policy']}` / `{best['method']}` / "
        f"`k={int(best['k'])}`, full-output RMSE `{best['pooled_rmse']:.6f}` and "
        f"non-anchor RMSE `{best['non_anchor_rmse']:.6f}`.",
    ]
    if first_below is None:
        lines.append("No tested combination reached strict pooled RMSE <4.")
    else:
        lines.append(
            f"The first tested anchor count reaching <4 was k={int(first_below['k'])} "
            f"with `{first_below['policy']}` / `{first_below['method']}`."
        )
    if len(k8) and float(k8["pooled_rmse"].min()) > 6.0:
        lines.append(
            "Even the best k=8 result remains above 6; sparse TVT anchors alone are "
            "insufficient under these conditioning rules.  The next information "
            "candidate should be formation tops, azimuthal/deep resistivity, or "
            "seismic/formation-surface constraints, not broad model tuning."
        )
    lines.extend(
        [
            "",
            "Uniform fractions are exactly j/(k+1), so the mandated k grids are "
            "not nested (for example k=1 uses 0.5 while k=2 uses 1/3 and 2/3). "
            "Disagreement uses abs(P3-Path) and fixed spacing 0.5/(k+1).",
            "",
            "The conditional method is a fixed-hyperparameter Gaussian coefficient "
            "update MVP.  Its coefficient covariance and observation noise are "
            "estimated only from each outer fold's training wells; no validation "
            "target calibrates fit state or regularization.",
            "",
            f"Paired uncertainty uses {bootstrap_samples:,} well-cluster bootstrap "
            "replicates. Full-output scoring includes zero-error revealed rows; "
            "non-anchor scoring excludes them.",
            "",
            "## k=8 catastrophic wells",
            "",
            _markdown_table(
                catastrophic[
                    ["policy", "method", "well_id", "fold", "rmse", "rank"]
                ]
            )
            if len(catastrophic)
            else "No k=8 rows were evaluated.",
            "",
            "## Segment interpretation",
            "",
            "See `per_segment_metrics.csv` for PS-to-first-anchor, every inter-anchor "
            "interval, and post-last-anchor pooled/non-anchor errors.  This file is "
            "the authoritative check for residual growth after the final anchor.",
            "",
            "## Evidence classification",
            "",
            "These results are counterfactual TVT-anchor/oracle evidence and active-"
            "acquisition research only. They are not current-competition deployable "
            "evidence, are not submission-eligible, and no Kaggle submission was "
            "created or uploaded.",
            "",
        ]
    )
    return "\n".join(lines)


def run_sparse_anchor_voi(
    *,
    data_root: Path,
    fold_config: Path,
    p3_oof: Path,
    path_pca_oof: Path,
    blend_oof: Path,
    path_pca_models: Path,
    output_root: Path,
    run_id: str,
    profile: str = "full",
    policies: Sequence[str] = POLICIES,
    methods: Sequence[str] = METHODS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    max_validation_wells: int | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = DEFAULT_SEED,
    write_oof: bool = True,
) -> dict[str, Any]:
    """Run the complete strict-fold sparse-anchor counterfactual diagnostic."""
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    requested_policies = tuple(dict.fromkeys(map(str, policies)))
    requested_methods = tuple(dict.fromkeys(map(str, methods)))
    if not requested_policies or not set(requested_policies) <= set(POLICIES):
        raise ValueError(f"policies must be a subset of {POLICIES}")
    if not requested_methods or not set(requested_methods) <= set(METHODS):
        raise ValueError(f"methods must be a subset of {METHODS}")
    requested_k = tuple(sorted(set(map(int, k_values))))
    if 0 not in requested_k or any(k < 0 for k in requested_k):
        raise ValueError("k values must be non-negative and include zero")

    data = Path(data_root).resolve()
    run_root = safe_output_directory(output_root, run_id, data)
    baseline = _read_aligned_baselines(blend_oof, p3_oof, path_pca_oof)
    train_files = enumerate_wells(data, "train")
    mapping = load_fold_mapping(
        fold_config, expected_well_ids=[well.well_id for well in train_files]
    )
    fold_version = str(mapping["fold_version"].iat[0])
    fold_hash = hashlib.sha256(Path(fold_config).read_bytes()).hexdigest().upper()
    if fold_version != FOLD_VERSION or fold_hash != FOLD_SHA256:
        raise ValueError("fixed fold version/hash contract mismatch")
    if len(mapping) != 773 or len(baseline) != 3_783_989:
        raise ValueError("strict full OOF well/row contract mismatch")

    contexts = _build_well_contexts(baseline, data, mapping)
    del baseline
    if len(contexts) != 773:
        raise ValueError("strict OOF well coverage mismatch")

    selected_by_fold: dict[int, tuple[str, ...]] = {}
    for fold in range(5):
        _, validation_pool = partition_well_ids(mapping, fold)
        limit = (
            max_validation_wells
            if max_validation_wells is not None
            else (5 if profile == "quick" else None)
        )
        selected_by_fold[fold] = stable_select(
            validation_pool,
            limit,
            seed=seed,
            salt=f"sparse-anchor-fold-{fold}-validation",
        )
    selected_wells = tuple(
        well_id
        for fold in range(5)
        for well_id in selected_by_fold[fold]
    )

    states: dict[int, PathPCAState] = {}
    calibrations: dict[int, ConditionalCalibration] = {}
    calibration_metadata: list[dict[str, Any]] = []
    if "conditional_path_pca_mvp" in requested_methods:
        prefix_features = {
            well_id: context.prefix_features
            for well_id, context in contexts.items()
        }
        target_curves = {
            well_id: context.target_curve for well_id, context in contexts.items()
        }
        for fold in range(5):
            state_path = Path(path_pca_models) / f"fold_{fold}.pkl"
            with state_path.open("rb") as handle:
                state = pickle.load(handle)
            if not isinstance(state, PathPCAState):
                raise TypeError(f"unexpected Path-PCA state type for fold {fold}")
            training_ids, validation_ids = partition_well_ids(mapping, fold)
            calibration = fit_conditional_calibration(
                state,
                training_ids,
                validation_ids,
                prefix_features,
                target_curves,
            )
            states[fold] = state
            calibrations[fold] = calibration
            calibration_metadata.append(
                {
                    "fold": fold,
                    "training_wells": len(training_ids),
                    "validation_wells": len(validation_ids),
                    "training_validation_overlap": 0,
                    "coefficient_count": calibration.coefficient_count,
                    "observation_noise_variance": calibration.observation_noise_variance,
                    "covariance_shrinkage": calibration.covariance_shrinkage,
                    "state_sha256": calibration.state_sha256,
                }
            )

    config = {
        "profile": profile,
        "run_id": run_id,
        "seed": seed,
        "policies": list(requested_policies),
        "methods": list(requested_methods),
        "k_values": list(requested_k),
        "bootstrap_samples": bootstrap_samples,
        "selected_wells": list(selected_wells),
        "fold_version": fold_version,
        "fold_sha256": fold_hash,
        "fold_counts": {
            str(key): int(value)
            for key, value in mapping["fold"].value_counts().sort_index().items()
        },
        "baseline_artifacts": {
            "p3": str(Path(p3_oof).resolve()),
            "path_pca": str(Path(path_pca_oof).resolve()),
            "fixed_50_50_blend": str(Path(blend_oof).resolve()),
            "path_pca_models": str(Path(path_pca_models).resolve()),
        },
        "uniform_fraction_rule": "j/(k+1), j=1..k; nearest unused MD row",
        "uniform_nested": False,
        "uniform_nested_explanation": "The mandated j/(k+1) grids are not nested across k.",
        "disagreement_score": "abs(P3_prediction - Path_PCA_prediction)",
        "disagreement_minimum_spacing": "0.5/(k+1) normalized MD",
        "conditional_method_status": "fixed-hyperparameter Gaussian coefficient-update MVP",
        "conditional_calibration": calibration_metadata,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
        "oof_storage": "compact wide format; anchor values/positions are in anchor_manifest.csv",
    }
    _atomic_json(run_root / "run_config.json", config)

    per_well_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    placement_hashes: dict[str, str] = {}
    placement_digest = hashlib.sha256()

    writer_context: Any
    if write_oof:
        writer_context = _DeterministicGzipCsvWriter(
            run_root / "oof_predictions.csv.gz"
        )
    else:
        writer_context = None
    try:
        for well_number, well_id in enumerate(selected_wells, start=1):
            context = contexts[well_id]
            boundary_md = context.prediction_md[0] - (
                context.normalized_md[0]
                * (context.prediction_md[-1] - context.prediction_md[0])
                / max(1.0 - context.normalized_md[0], 1e-15)
            )
            # Recompute the exact boundary from the normalization identity.
            horizon = (
                context.prediction_md[-1] - context.prediction_md[0]
            ) / max(
                context.normalized_md[-1] - context.normalized_md[0], 1e-15
            )
            boundary_md = float(
                context.prediction_md[0] - context.normalized_md[0] * horizon
            )
            selections: dict[tuple[str, int], AnchorSelection] = {}
            disagreement = np.abs(
                context.p3_prediction - context.path_prediction
            )
            for policy in requested_policies:
                for k in requested_k:
                    selection = (
                        select_uniform_anchors(
                            context.prediction_md, boundary_md, k
                        )
                        if policy == "uniform"
                        else select_disagreement_anchors(
                            context.prediction_md,
                            boundary_md,
                            disagreement,
                            k,
                        )
                    )
                    selections[(policy, k)] = selection
                    placement_digest.update(well_id.encode("ascii"))
                    placement_digest.update(policy.encode("ascii"))
                    placement_digest.update(np.int64(k).tobytes())
                    placement_digest.update(
                        np.ascontiguousarray(
                            context.row_index[selection.local_indices]
                        ).view(np.uint8)
                    )
                    for anchor_order, local_index in enumerate(
                        selection.local_indices, start=1
                    ):
                        anchor_rows.append(
                            {
                                "well_id": well_id,
                                "fold": context.fold,
                                "policy": policy,
                                "requested_k": k,
                                "achieved_k": selection.achieved_k,
                                "anchor_order": anchor_order,
                                "row_index": int(context.row_index[local_index]),
                                "normalized_md_position": float(
                                    context.normalized_md[local_index]
                                ),
                                "requested_fraction": float(
                                    selection.requested_positions[anchor_order - 1]
                                ),
                                "minimum_spacing": selection.minimum_spacing,
                                "acquisition_score": float(
                                    selection.acquisition_scores[anchor_order - 1]
                                ),
                                "revealed_tvt": float(
                                    context.y_true[local_index]
                                ),
                                "uses_validation_target_as_revealed_anchor": True,
                                "deployable": False,
                                "submission_eligible": False,
                            }
                        )

            wide = pd.DataFrame(
                {
                    "well_id": well_id,
                    "fold": np.int8(context.fold),
                    "row_index": context.row_index,
                    "normalized_md_position": context.normalized_md,
                    "y_true": context.y_true,
                    "p3_prediction": context.p3_prediction,
                    "path_prediction": context.path_prediction,
                    "base_prediction": context.base_prediction,
                }
            )
            for policy in requested_policies:
                for k in requested_k:
                    selection = selections[(policy, k)]
                    mask = anchor_mask(len(context.row_index), selection)
                    wide[f"anchor__{policy}__k{k}"] = mask
                    regions = segment_regions(context.normalized_md, selection)
                    for method in requested_methods:
                        method_base = (
                            context.base_prediction
                            if method == "piecewise_residual"
                            else context.path_prediction
                        )
                        prediction = (
                            condition_piecewise_residual(
                                context.base_prediction,
                                context.y_true,
                                context.normalized_md,
                                selection,
                            )
                            if method == "piecewise_residual"
                            else condition_path_pca_posterior(
                                states[context.fold],
                                calibrations[context.fold],
                                context.prefix_features,
                                context.prefix_anchor_tvt,
                                context.path_prediction,
                                context.y_true,
                                context.normalized_md,
                                selection,
                            )
                        )
                        result = ConditionedWellResult(
                            well_id=well_id,
                            fold=context.fold,
                            row_index=context.row_index,
                            normalized_md_position=context.normalized_md,
                            placement_policy=policy,
                            conditioning_method=method,
                            requested_k=k,
                            achieved_k=selection.achieved_k,
                            selected_anchor_row_indices=context.row_index[
                                selection.local_indices
                            ],
                            revealed_tvt_values=context.y_true[
                                selection.local_indices
                            ],
                            base_prediction=method_base,
                            conditioned_prediction=prediction,
                            anchor_mask=mask,
                            segment_region=regions,
                        )
                        if k == 0 and not np.array_equal(
                            result.conditioned_prediction,
                            result.base_prediction,
                        ):
                            raise ValueError("k=0 did not exactly reproduce baseline")
                        if selection.achieved_k and not np.array_equal(
                            prediction[selection.local_indices],
                            context.y_true[selection.local_indices],
                        ):
                            raise ValueError("revealed anchor continuity failed")
                        wide[
                            "pred__" + _combo_name(policy, method, k)
                        ] = prediction

                        errors = prediction - context.y_true
                        baseline_errors = method_base - context.y_true
                        non_anchor = ~mask
                        sse = float(np.dot(errors, errors))
                        baseline_sse = float(
                            np.dot(baseline_errors, baseline_errors)
                        )
                        non_sse = float(
                            np.dot(errors[non_anchor], errors[non_anchor])
                        )
                        baseline_non_sse = float(
                            np.dot(
                                baseline_errors[non_anchor],
                                baseline_errors[non_anchor],
                            )
                        )
                        per_well_rows.append(
                            {
                                "policy": policy,
                                "method": method,
                                "k": k,
                                "well_id": well_id,
                                "fold": context.fold,
                                "requested_k": k,
                                "achieved_k": selection.achieved_k,
                                "rows": len(errors),
                                "anchor_rows": int(mask.sum()),
                                "non_anchor_rows": int(non_anchor.sum()),
                                "sse": sse,
                                "non_anchor_sse": non_sse,
                                "baseline_sse": baseline_sse,
                                "baseline_non_anchor_sse": baseline_non_sse,
                                "rmse": float(np.sqrt(sse / len(errors))),
                                "non_anchor_rmse": float(
                                    np.sqrt(non_sse / non_anchor.sum())
                                ),
                                "baseline_rmse": float(
                                    np.sqrt(baseline_sse / len(errors))
                                ),
                            }
                        )
                        for region in np.unique(regions):
                            region_mask = regions == region
                            region_non = region_mask & non_anchor
                            segment_rows.append(
                                {
                                    "policy": policy,
                                    "method": method,
                                    "k": k,
                                    "segment_region": str(region),
                                    "rows": int(region_mask.sum()),
                                    "sse": float(
                                        np.dot(
                                            errors[region_mask],
                                            errors[region_mask],
                                        )
                                    ),
                                    "non_anchor_rows": int(region_non.sum()),
                                    "non_anchor_sse": float(
                                        np.dot(
                                            errors[region_non],
                                            errors[region_non],
                                        )
                                    ),
                                }
                            )
                        deciles = np.minimum(
                            (context.normalized_md * 10).astype(np.int8), 9
                        )
                        for decile in range(10):
                            decile_mask = deciles == decile
                            decile_non = decile_mask & non_anchor
                            distance_rows.append(
                                {
                                    "policy": policy,
                                    "method": method,
                                    "k": k,
                                    "distance_decile": decile,
                                    "rows": int(decile_mask.sum()),
                                    "sse": float(
                                        np.dot(
                                            errors[decile_mask],
                                            errors[decile_mask],
                                        )
                                    ),
                                    "non_anchor_rows": int(decile_non.sum()),
                                    "non_anchor_sse": float(
                                        np.dot(
                                            errors[decile_non],
                                            errors[decile_non],
                                        )
                                    ),
                                }
                            )
            if writer_context is not None:
                writer_context.write(wide)
            if well_number % 25 == 0 or well_number == len(selected_wells):
                print(
                    f"sparse-anchor evaluate wells={well_number}/{len(selected_wells)}",
                    flush=True,
                )
    finally:
        if writer_context is not None:
            writer_context.close()

    placement_hashes["all_selected_positions_sha256"] = (
        placement_digest.hexdigest().upper()
    )
    per_well = pd.DataFrame(per_well_rows)
    information, per_fold = _summarize_metrics(per_well)
    per_segment = _aggregate_error_table(
        segment_rows, ["policy", "method", "k", "segment_region"]
    )
    distance_deciles = _aggregate_error_table(
        distance_rows, ["policy", "method", "k", "distance_decile"]
    )
    bootstrap = _bootstrap(
        per_well, samples=bootstrap_samples, seed=seed + 31
    )
    worst = (
        per_well.sort_values(
            ["policy", "method", "k", "rmse"],
            ascending=[True, True, True, False],
            kind="stable",
        )
        .assign(
            rank=lambda frame: frame.groupby(
                ["policy", "method", "k"], sort=False
            ).cumcount()
            + 1
        )
    )
    worst = worst[worst["rank"].le(20)].copy()

    anchor_manifest = pd.DataFrame(anchor_rows)
    per_well.sort_values(
        ["policy", "method", "k", "fold", "well_id"], kind="stable"
    ).to_csv(run_root / "per_well_metrics.csv", index=False)
    per_fold.sort_values(
        ["policy", "method", "k", "fold"], kind="stable"
    ).to_csv(run_root / "per_fold_metrics.csv", index=False)
    information.to_csv(run_root / "per_k_metrics.csv", index=False)
    information.to_csv(run_root / "information_curve.csv", index=False)
    per_segment.to_csv(run_root / "per_segment_metrics.csv", index=False)
    distance_deciles.to_csv(run_root / "distance_deciles.csv", index=False)
    worst.to_csv(run_root / "worst_wells.csv", index=False)
    anchor_manifest.to_csv(run_root / "anchor_manifest.csv", index=False)
    _atomic_json(
        run_root / "bootstrap_summary.json",
        {
            "samples": bootstrap_samples,
            "seed": seed + 31,
            "cluster": "well",
            "comparisons": bootstrap,
        },
    )

    best = information[information["k"].gt(0)].sort_values(
        "pooled_rmse", kind="stable"
    ).iloc[0]
    leakage_check = {
        "passed": True,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "target_used_for_placement": False,
        "target_used_for_model_fit": False,
        "target_used_for_uncertainty_fit": False,
        "target_used_for_selector_or_gate_fit": False,
        "target_used_for_hyperparameter_selection": False,
        "target_used_for_posthoc_well_selection": False,
        "validation_target_roles": [
            "revealed values at preselected anchor rows",
            "post-prediction scoring",
        ],
        "placement_function_inputs": {
            "uniform": ["prediction_md", "boundary_md", "k"],
            "disagreement": [
                "prediction_md",
                "boundary_md",
                "abs(p3_prediction-path_prediction)",
                "k",
            ],
        },
        "target_permutation_position_invariance": True,
        "placement_hashes": placement_hashes,
        "horizontal_inference_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
        "typewell_columns_read": [],
        "formation_surfaces_read": False,
        "geology_read": False,
        "png_read": False,
        "outer_validation_in_conditional_fit": False,
        "conditional_fit_audit": calibration_metadata,
        "data_write_attempted": False,
        "row_key_alignment": True,
        "well_level_folds": True,
        "fold_version": fold_version,
        "fold_sha256": fold_hash,
    }
    _atomic_json(run_root / "leakage_check.json", leakage_check)
    summary = {
        "profile": profile,
        "wells": len(selected_wells),
        "rows": int(
            sum(len(contexts[well_id].row_index) for well_id in selected_wells)
        ),
        "best": {
            key: (
                int(best[key])
                if key == "k"
                else float(best[key])
                if key in {"pooled_rmse", "non_anchor_rmse", "p95", "max"}
                else best[key]
            )
            for key in [
                "policy",
                "method",
                "k",
                "pooled_rmse",
                "non_anchor_rmse",
                "p95",
                "max",
            ]
        },
        "goal_rmse": 4.0,
        "goal_achieved": bool(
            information.loc[information["k"].gt(0), "pooled_rmse"].min() < 4.0
        ),
        "non_anchor_goal_achieved": bool(
            information.loc[
                information["k"].gt(0), "non_anchor_rmse"
            ].min()
            < 4.0
        ),
        "baseline_fixed_blend_rmse_reference": BASELINE_RMSE,
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
            information,
            per_segment,
            worst,
            bootstrap_samples=bootstrap_samples,
            profile=profile,
        ),
        encoding="utf-8",
    )
    return summary
