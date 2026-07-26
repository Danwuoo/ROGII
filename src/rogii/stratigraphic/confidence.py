"""Fold-safe confidence calibration and precision-first operating points."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Literal, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


CalibrationMethod = Literal[
    "main", "raw_nn", "point_gr", "no_cycle", "destroy_order"
]
CALIBRATION_METHODS: tuple[CalibrationMethod, ...] = (
    "main",
    "raw_nn",
    "point_gr",
    "no_cycle",
    "destroy_order",
)
MAIN_FEATURE_COLUMNS = (
    "raw_shape_distance",
    "derivative_distance",
    "rank_distance",
    "barcode_distance",
    "pair_distance",
    "absolute_margin",
    "relative_margin",
    "distance_ratio",
    "reciprocal_rank",
    "reciprocal_top_k",
    "scale_vote_count",
    "scale_agreement",
    "shared_neighbor_count",
    "local_cycle_support_count",
    "query_observed_fraction",
    "candidate_observed_fraction",
    "query_low_information",
    "candidate_low_information",
)
NO_CYCLE_COLUMNS = (
    "reciprocal_rank",
    "reciprocal_top_k",
    "scale_vote_count",
    "scale_agreement",
    "shared_neighbor_count",
    "local_cycle_support_count",
)


@dataclass(frozen=True)
class Calibrator:
    """Serializable StandardScaler plus unweighted logistic regression."""

    method: str
    feature_names: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    supported: bool
    unsupported_reason: str = ""
    fit_edges: int = 0
    positives: int = 0
    positive_prevalence: float = 0.0

    def predict(self, edges: pd.DataFrame) -> np.ndarray:
        if not self.supported:
            raise ValueError(f"Unsupported calibrator: {self.unsupported_reason}")
        matrix = feature_matrix(edges, method=self.method)
        standardized = (matrix - self.scaler_mean) / self.scaler_scale
        logits = standardized @ self.coefficients + self.intercept
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "feature_names": list(self.feature_names),
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "fit_edges": self.fit_edges,
            "positives": self.positives,
            "positive_prevalence": self.positive_prevalence,
            "class_weight": None,
        }


def feature_names_for_method(method: str) -> tuple[str, ...]:
    if method == "raw_nn":
        return ("raw_shape_distance",)
    if method == "point_gr":
        return ("point_gr_distance",)
    return MAIN_FEATURE_COLUMNS


def feature_matrix(edges: pd.DataFrame, *, method: str) -> np.ndarray:
    """Build a finite calibration matrix for one explicitly named method."""
    names = feature_names_for_method(method)
    missing = set(names) - set(edges.columns)
    if missing:
        raise ValueError(f"Calibration features missing: {sorted(missing)}")
    matrix = edges.loc[:, list(names)].to_numpy(dtype=np.float64, copy=True)
    if method == "no_cycle":
        for column in NO_CYCLE_COLUMNS:
            matrix[:, names.index(column)] = 0.0
    elif method == "destroy_order":
        if "destroyed_barcode_distance" not in edges:
            raise ValueError("destroy_order requires destroyed_barcode_distance")
        matrix[:, names.index("barcode_distance")] = edges[
            "destroyed_barcode_distance"
        ].to_numpy(dtype=np.float64)
        matrix[:, names.index("pair_distance")] = (
            0.35 * matrix[:, names.index("raw_shape_distance")]
            + 0.20 * matrix[:, names.index("derivative_distance")]
            + 0.15 * matrix[:, names.index("rank_distance")]
            + 0.30 * matrix[:, names.index("barcode_distance")]
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Confidence features must be finite")
    return matrix


def unsupported_calibrator(method: str, reason: str) -> Calibrator:
    names = feature_names_for_method(method)
    return Calibrator(
        method=method,
        feature_names=names,
        scaler_mean=np.zeros(len(names), dtype=np.float64),
        scaler_scale=np.ones(len(names), dtype=np.float64),
        coefficients=np.zeros(len(names), dtype=np.float64),
        intercept=0.0,
        supported=False,
        unsupported_reason=reason,
    )


def fit_calibrator(
    edges: pd.DataFrame,
    labels: Iterable[bool],
    *,
    method: str,
    random_state: int,
) -> Calibrator:
    """Fit an unweighted fold-local logistic probability model."""
    y = np.asarray(tuple(labels), dtype=np.int8)
    if len(edges) != y.size or y.size == 0:
        return unsupported_calibrator(method, "empty_or_misaligned_fit_edges")
    positives = int(y.sum())
    if positives == 0 or positives == y.size:
        return unsupported_calibrator(method, "requires_positive_and_negative_edges")
    matrix = feature_matrix(edges, method=method)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(matrix)
    model = LogisticRegression(
        class_weight=None,
        random_state=random_state,
        solver="lbfgs",
        max_iter=500,
    )
    model.fit(transformed, y)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    return Calibrator(
        method=method,
        feature_names=feature_names_for_method(method),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=scale,
        coefficients=np.asarray(model.coef_[0], dtype=np.float64),
        intercept=float(model.intercept_[0]),
        supported=True,
        fit_edges=int(y.size),
        positives=positives,
        positive_prevalence=float(positives / y.size),
    )


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return max(0.0, (center - radius) / denominator)


def select_operating_point(
    confidence: Iterable[float],
    primary_positive: Iterable[bool],
    catastrophic: Iterable[bool],
    query_ids: Iterable[int],
    well_ids: Iterable[str],
    *,
    target_precision: float,
    total_future_rows: int,
    minimum_edges: int = 1_000,
    minimum_wells: int = 25,
) -> dict[str, Any]:
    """Choose maximum observed row coverage among precision-qualified thresholds."""
    probability = np.asarray(tuple(confidence), dtype=np.float64)
    positive = np.asarray(tuple(primary_positive), dtype=bool)
    catastrophic_values = np.asarray(tuple(catastrophic), dtype=bool)
    queries = np.asarray(tuple(query_ids), dtype=np.int64)
    wells = np.asarray(tuple(well_ids), dtype=object)
    size = probability.size
    if not (
        size
        == positive.size
        == catastrophic_values.size
        == queries.size
        == wells.size
    ):
        raise ValueError("Operating-point arrays must be aligned")
    if total_future_rows <= 0:
        raise ValueError("total_future_rows must be positive")
    if size == 0:
        return {
            "supported": False,
            "insufficient_support": True,
            "threshold": 1.0,
            "target_precision": target_precision,
            "accepted_edges": 0,
            "positive_edges": 0,
            "precision": 0.0,
            "wilson_lower_bound": 0.0,
            "catastrophic_false_edge_rate": 0.0,
            "unique_future_rows": 0,
            "all_future_coverage": 0.0,
            "unique_wells": 0,
            "maximum_empirical_precision": 0.0,
            "maximum_empirical_precision_accepted_edges": 0,
            "maximum_supported_precision": 0.0,
            "maximum_supported_precision_threshold": 1.0,
            "maximum_supported_precision_coverage": 0.0,
        }
    order = np.lexsort((queries, -probability))
    probability = probability[order]
    positive = positive[order]
    catastrophic_values = catastrophic_values[order]
    queries = queries[order]
    wells = wells[order]
    cumulative_positive = np.cumsum(positive, dtype=np.int64)
    cumulative_catastrophic = np.cumsum(catastrophic_values, dtype=np.int64)
    end_of_tie = np.r_[probability[1:] != probability[:-1], True]
    seen_queries: set[int] = set()
    seen_wells: set[str] = set()
    unique_query_count = np.zeros(size, dtype=np.int64)
    unique_well_count = np.zeros(size, dtype=np.int64)
    for index, (query, well) in enumerate(zip(queries, wells, strict=True)):
        seen_queries.add(int(query))
        seen_wells.add(str(well))
        unique_query_count[index] = len(seen_queries)
        unique_well_count[index] = len(seen_wells)
    accepted = np.arange(1, size + 1, dtype=np.int64)
    precision = cumulative_positive / accepted
    coverage = unique_query_count / float(total_future_rows)
    candidate_thresholds = np.flatnonzero(end_of_tie)
    maximum_index = int(
        candidate_thresholds[
            np.lexsort(
                (
                    -coverage[candidate_thresholds],
                    -precision[candidate_thresholds],
                )
            )[0]
        ]
    )
    support_mask = (
        end_of_tie
        & (accepted >= minimum_edges)
        & (unique_well_count >= minimum_wells)
    )
    if support_mask.any():
        support_indices = np.flatnonzero(support_mask)
        supported_best_index = int(
            support_indices[
                np.lexsort(
                    (
                        -coverage[support_indices],
                        -precision[support_indices],
                    )
                )[0]
            ]
        )
        maximum_supported_precision = float(precision[supported_best_index])
        maximum_supported_threshold = float(probability[supported_best_index])
        maximum_supported_coverage = float(coverage[supported_best_index])
    else:
        maximum_supported_precision = 0.0
        maximum_supported_threshold = 1.0
        maximum_supported_coverage = 0.0
    sensitivity = {
        "maximum_empirical_precision": float(precision[maximum_index]),
        "maximum_empirical_precision_accepted_edges": int(maximum_index + 1),
        "maximum_supported_precision": maximum_supported_precision,
        "maximum_supported_precision_threshold": maximum_supported_threshold,
        "maximum_supported_precision_coverage": maximum_supported_coverage,
    }
    qualified = end_of_tie & (precision >= target_precision)
    if not qualified.any():
        return {
            "supported": False,
            "insufficient_support": True,
            "threshold": 1.0,
            "target_precision": target_precision,
            "accepted_edges": 0,
            "positive_edges": 0,
            "precision": 0.0,
            "wilson_lower_bound": 0.0,
            "catastrophic_false_edge_rate": 0.0,
            "unique_future_rows": 0,
            "all_future_coverage": 0.0,
            "unique_wells": 0,
            **sensitivity,
        }
    indices = np.flatnonzero(qualified)
    best_index = int(
        indices[
            np.lexsort(
                (
                    -probability[indices],
                    -precision[indices],
                    -coverage[indices],
                )
            )[0]
        ]
    )
    accepted_edges = best_index + 1
    positive_edges = int(cumulative_positive[best_index])
    distinct_wells = int(unique_well_count[best_index])
    insufficient = (
        accepted_edges < minimum_edges or distinct_wells < minimum_wells
    )
    return {
        "supported": not insufficient,
        "insufficient_support": insufficient,
        "threshold": float(probability[best_index]),
        "target_precision": float(target_precision),
        "accepted_edges": accepted_edges,
        "positive_edges": positive_edges,
        "precision": float(precision[best_index]),
        "wilson_lower_bound": wilson_lower_bound(
            positive_edges, accepted_edges
        ),
        "catastrophic_false_edge_rate": float(
            cumulative_catastrophic[best_index] / accepted_edges
        ),
        "unique_future_rows": int(unique_query_count[best_index]),
        "all_future_coverage": float(coverage[best_index]),
        "unique_wells": distinct_wells,
        **sensitivity,
    }


def edge_truth(error_ft: Iterable[float]) -> dict[str, np.ndarray]:
    """Apply exact SLP-E1 truth boundaries."""
    error = np.asarray(tuple(error_ft), dtype=np.float64)
    if not np.isfinite(error).all() or np.any(error < 0.0):
        raise ValueError("edge errors must be finite and non-negative")
    return {
        "strict_positive": error <= 0.5,
        "primary_positive": error <= 1.0,
        "loose_positive": error <= 2.0,
        "catastrophic": error > 10.0,
    }


def reliability_bins(
    confidence: Iterable[float],
    labels: Iterable[bool],
    *,
    bins: int = 10,
) -> tuple[pd.DataFrame, float, float]:
    probability = np.asarray(tuple(confidence), dtype=np.float64)
    y = np.asarray(tuple(labels), dtype=np.float64)
    if probability.size != y.size:
        raise ValueError("Reliability arrays must be aligned")
    if probability.size == 0:
        rows = [
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": 0,
                "mean_confidence": 0.0,
                "observed_precision": 0.0,
            }
            for index in range(bins)
        ]
        return pd.DataFrame(rows), 0.0, 0.0
    assigned = np.minimum((probability * bins).astype(np.int64), bins - 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        mask = assigned == index
        count = int(mask.sum())
        mean_confidence = float(probability[mask].mean()) if count else 0.0
        observed = float(y[mask].mean()) if count else 0.0
        if count:
            ece += count / probability.size * abs(mean_confidence - observed)
        rows.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "mean_confidence": mean_confidence,
                "observed_precision": observed,
            }
        )
    brier = float(np.mean((probability - y) ** 2))
    return pd.DataFrame(rows), brier, float(ece)


def scored_edge_metrics(
    confidence: Iterable[float],
    error_ft: Iterable[float],
    query_ids: Iterable[int],
    *,
    threshold: float,
    total_future_rows: int,
    gr_observed_future_rows: int | None = None,
    descriptor_eligible_future_rows: int | None = None,
    accepted_mask: Iterable[bool] | None = None,
) -> dict[str, Any]:
    """Compute pooled edge precision, coverage, safety, and calibration."""
    probability = np.asarray(tuple(confidence), dtype=np.float64)
    error = np.asarray(tuple(error_ft), dtype=np.float64)
    queries = np.asarray(tuple(query_ids), dtype=np.int64)
    if not (probability.size == error.size == queries.size):
        raise ValueError("Scored edge arrays must be aligned")
    truth = edge_truth(error)
    if accepted_mask is None:
        accepted = probability >= threshold
    else:
        accepted = np.asarray(tuple(accepted_mask), dtype=bool)
        if accepted.size != probability.size:
            raise ValueError("accepted_mask must align with scored edges")
    accepted_count = int(accepted.sum())
    unique_rows = int(np.unique(queries[accepted]).size) if accepted_count else 0
    primary_count = int(truth["primary_positive"][accepted].sum())
    strict_count = int(truth["strict_positive"][accepted].sum())
    loose_count = int(truth["loose_positive"][accepted].sum())
    catastrophic_count = int(truth["catastrophic"][accepted].sum())
    accepted_error = error[accepted]
    quantiles = {}
    for label, quantile in (
        ("median", 0.50),
        ("q75", 0.75),
        ("q90", 0.90),
        ("q95", 0.95),
        ("q99", 0.99),
    ):
        quantiles[label] = (
            float(np.quantile(accepted_error, quantile))
            if accepted_count
            else 0.0
        )
    bins, brier, ece = reliability_bins(
        probability, truth["primary_positive"]
    )
    del bins
    if probability.size and np.unique(truth["primary_positive"]).size == 2:
        average_precision = float(
            average_precision_score(truth["primary_positive"], probability)
        )
    else:
        average_precision = 0.0
    result = {
        "proposed_edges": int(probability.size),
        "accepted_edges": accepted_count,
        "primary_positive_edges": primary_count,
        "edge_precision": (
            float(primary_count / accepted_count) if accepted_count else 0.0
        ),
        "strict_precision": (
            float(strict_count / accepted_count) if accepted_count else 0.0
        ),
        "loose_precision": (
            float(loose_count / accepted_count) if accepted_count else 0.0
        ),
        "catastrophic_edges": catastrophic_count,
        "catastrophic_false_edge_rate": (
            float(catastrophic_count / accepted_count)
            if accepted_count
            else 0.0
        ),
        "unique_future_rows": unique_rows,
        "all_future_rows": int(total_future_rows),
        "all_future_coverage": (
            float(unique_rows / total_future_rows) if total_future_rows else 0.0
        ),
        "average_precision": average_precision,
        "brier_score": brier,
        "ece_10bin": ece,
        "wilson_precision_lower_bound": wilson_lower_bound(
            primary_count, accepted_count
        ),
        "error_max": float(accepted_error.max()) if accepted_count else 0.0,
        **{f"error_{key}": value for key, value in quantiles.items()},
    }
    if gr_observed_future_rows is not None:
        result["gr_observed_future_rows"] = int(gr_observed_future_rows)
        result["gr_observed_query_coverage"] = (
            float(unique_rows / gr_observed_future_rows)
            if gr_observed_future_rows
            else 0.0
        )
    if descriptor_eligible_future_rows is not None:
        result["descriptor_eligible_future_rows"] = int(
            descriptor_eligible_future_rows
        )
        result["eligible_query_coverage"] = (
            float(unique_rows / descriptor_eligible_future_rows)
            if descriptor_eligible_future_rows
            else 0.0
        )
    return result


def calibration_from_mapping(payload: Mapping[str, Any]) -> Calibrator:
    """Rehydrate a calibrator from the JSON representation."""
    return Calibrator(
        method=str(payload["method"]),
        feature_names=tuple(map(str, payload["feature_names"])),
        scaler_mean=np.asarray(payload["scaler_mean"], dtype=np.float64),
        scaler_scale=np.asarray(payload["scaler_scale"], dtype=np.float64),
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        intercept=float(payload["intercept"]),
        supported=bool(payload["supported"]),
        unsupported_reason=str(payload.get("unsupported_reason", "")),
        fit_edges=int(payload.get("fit_edges", 0)),
        positives=int(payload.get("positives", 0)),
        positive_prevalence=float(payload.get("positive_prevalence", 0.0)),
    )
