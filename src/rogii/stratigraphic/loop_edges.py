"""Inference-safe PF, FF, and HT loop-edge candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..data import (
    HORIZONTAL_INFERENCE_COLUMNS,
    TYPEWELL_INFERENCE_COLUMNS,
    select_inference_view,
    validate_prefix,
)
from .cycle_checks import (
    build_nonlocal_neighbor_graph,
    local_cycle_support_count,
    shared_neighbor_count,
)
from .descriptors import DescriptorBatch, DescriptorConfig, build_descriptor_batch


EdgeType = Literal["PF", "FF", "HT"]
EDGE_TYPES: tuple[EdgeType, ...] = ("PF", "FF", "HT")


@dataclass(frozen=True)
class EdgeGenerationConfig:
    """Deterministic retrieval and descriptor configuration."""

    horizontal_half_spans: tuple[float, ...] = (16.0, 32.0, 64.0)
    typewell_half_spans: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0)
    points: int = 33
    top_k: int = 3
    minimum_ff_separation: float = 128.0
    minimum_observed_fraction: float = 0.80
    flat_epsilon: float = 1e-6
    graph_neighbors: int = 8

    def __post_init__(self) -> None:
        if not self.horizontal_half_spans or any(
            value <= 0.0 for value in self.horizontal_half_spans
        ):
            raise ValueError("horizontal_half_spans must be positive")
        if not self.typewell_half_spans or any(
            value <= 0.0 for value in self.typewell_half_spans
        ):
            raise ValueError("typewell_half_spans must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.minimum_ff_separation <= 0.0:
            raise ValueError("minimum_ff_separation must be positive")
        DescriptorConfig(
            points=self.points,
            minimum_observed_fraction=self.minimum_observed_fraction,
            flat_epsilon=self.flat_epsilon,
        )


@dataclass(frozen=True)
class EdgeGenerationResult:
    """One well/cut's target-free candidates and coverage metadata."""

    edges: pd.DataFrame
    all_future_rows: int
    gr_observed_future_rows: int
    requested_query_rows: int
    descriptor_eligible_future_rows: int
    eligible_query_rows: tuple[int, ...]
    candidate_counts: dict[str, int]
    descriptor_seconds: float
    retrieval_seconds: float
    peak_candidates: int
    peak_index_rows: int


_FLOAT_EDGE_COLUMNS = (
    "query_md",
    "candidate_md",
    "candidate_tvt",
    "query_scale",
    "candidate_scale",
    "raw_shape_distance",
    "derivative_distance",
    "rank_distance",
    "barcode_distance",
    "destroyed_barcode_distance",
    "point_gr_distance",
    "pair_distance",
    "best_distance",
    "second_best_distance",
    "absolute_margin",
    "relative_margin",
    "distance_ratio",
    "scale_agreement",
    "query_observed_fraction",
    "candidate_observed_fraction",
    "raw_score",
)


def _empty_edges() -> pd.DataFrame:
    columns: dict[str, pd.Series] = {
        "method": pd.Series(dtype="string"),
        "edge_type": pd.Series(dtype="string"),
        "query_well_id": pd.Series(dtype="string"),
        "candidate_well_id": pd.Series(dtype="string"),
        "query_row": pd.Series(dtype="int64"),
        "candidate_row": pd.Series(dtype="int64"),
        "edge_row_a": pd.Series(dtype="int64"),
        "edge_row_b": pd.Series(dtype="int64"),
        "reciprocal_rank": pd.Series(dtype="int16"),
        "reciprocal_top_k": pd.Series(dtype="bool"),
        "scale_vote_count": pd.Series(dtype="int16"),
        "shared_neighbor_count": pd.Series(dtype="int16"),
        "local_cycle_support_count": pd.Series(dtype="int16"),
        "query_low_information": pd.Series(dtype="bool"),
        "candidate_low_information": pd.Series(dtype="bool"),
    }
    columns.update(
        {column: pd.Series(dtype="float64") for column in _FLOAT_EDGE_COLUMNS}
    )
    return pd.DataFrame(columns)


def deterministic_shuffle_gr(
    values: Iterable[float], *, well_id: str, seed: int
) -> np.ndarray:
    """Shuffle observed GR values while preserving their missingness positions."""
    gr = np.asarray(values, dtype=np.float64).copy()
    finite = np.flatnonzero(np.isfinite(gr))
    digest = hashlib.sha256(
        f"{seed}:SLP-E1-shuffle:{well_id}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    gr[finite] = gr[finite][rng.permutation(finite.size)]
    return gr


def _descriptor_config(config: EdgeGenerationConfig) -> DescriptorConfig:
    return DescriptorConfig(
        points=config.points,
        minimum_observed_fraction=config.minimum_observed_fraction,
        flat_epsilon=config.flat_epsilon,
    )


def _subset_batch(batch: DescriptorBatch, rows: np.ndarray) -> DescriptorBatch:
    if batch.eligible_rows == 0 or rows.size == 0:
        return batch.take(np.empty((0,), dtype=np.int64))
    positions = np.searchsorted(batch.rows, rows)
    valid = positions < batch.rows.size
    matched = np.zeros(rows.size, dtype=bool)
    matched[valid] = batch.rows[positions[valid]] == rows[valid]
    return batch.take(positions[matched])


def _fit_neighbor_index(values: np.ndarray, neighbors: int) -> NearestNeighbors:
    return NearestNeighbors(
        n_neighbors=min(neighbors, values.shape[0]),
        algorithm="auto",
        metric="euclidean",
        n_jobs=1,
    ).fit(values)


def _neighbor_lists(
    query_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    query_coordinate: np.ndarray,
    candidate_coordinate: np.ndarray,
    top_k: int,
    minimum_separation: float | None,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    if query_values.size == 0 or candidate_values.size == 0:
        return [], [], 0
    if minimum_separation is None:
        requested = min(candidate_values.shape[0], max(top_k + 1, top_k))
    else:
        requested = min(
            candidate_values.shape[0],
            max(top_k + 1, min(384, candidate_values.shape[0])),
        )
    index = _fit_neighbor_index(candidate_values, requested)
    distances, indices = index.kneighbors(query_values, return_distance=True)
    selected_indices: list[np.ndarray] = []
    selected_distances: list[np.ndarray] = []
    for query_index in range(query_values.shape[0]):
        row_indices = indices[query_index]
        row_distances = distances[query_index]
        if minimum_separation is not None:
            keep = (
                np.abs(
                    candidate_coordinate[row_indices]
                    - query_coordinate[query_index]
                )
                >= minimum_separation
            )
            row_indices = row_indices[keep]
            row_distances = row_distances[keep]
        selected_indices.append(row_indices[:top_k].astype(np.int64, copy=False))
        selected_distances.append(row_distances[:top_k])
    return selected_indices, selected_distances, int(candidate_values.shape[0])


def _reverse_rank_lookup(
    query: DescriptorBatch,
    candidate: DescriptorBatch,
    *,
    feature_name: Literal["signature", "raw_shape"],
    top_k: int,
    minimum_separation: float | None,
) -> dict[tuple[int, int], int]:
    query_values = getattr(query, feature_name)
    candidate_values = getattr(candidate, feature_name)
    reverse_indices, _, _ = _neighbor_lists(
        candidate_values,
        query_values,
        query_coordinate=candidate.coordinates,
        candidate_coordinate=query.coordinates,
        top_k=top_k,
        minimum_separation=minimum_separation,
    )
    lookup: dict[tuple[int, int], int] = {}
    for candidate_local, query_locals in enumerate(reverse_indices):
        candidate_row = int(candidate.rows[candidate_local])
        for rank, query_local in enumerate(query_locals, start=1):
            lookup[(candidate_row, int(query.rows[query_local]))] = rank
    return lookup


def _pair_feature_frame(
    query: DescriptorBatch,
    candidate: DescriptorBatch,
    query_local: np.ndarray,
    candidate_local: np.ndarray,
    *,
    method: str,
    edge_type: EdgeType,
    query_well_id: str,
    candidate_well_id: str,
    query_scale: float,
    candidate_scale: float,
    reverse_ranks: dict[tuple[int, int], int],
    top_k: int,
    candidate_coordinate_is_tvt: bool,
) -> pd.DataFrame:
    if query_local.size == 0:
        return _empty_edges()
    q_raw = query.raw_shape[query_local]
    c_raw = candidate.raw_shape[candidate_local]
    raw_distance = np.sqrt(np.mean((q_raw - c_raw) ** 2, axis=1))
    derivative_distance = np.sqrt(
        np.mean(
            (query.derivative[query_local] - candidate.derivative[candidate_local])
            ** 2,
            axis=1,
        )
    )
    rank_distance = np.sqrt(
        np.mean(
            (query.rank[query_local] - candidate.rank[candidate_local]) ** 2,
            axis=1,
        )
    )
    q_barcode = query.barcode[query_local]
    c_barcode = candidate.barcode[candidate_local]
    barcode_distance = np.sqrt(np.mean((q_barcode - c_barcode) ** 2, axis=1))
    destroyed_barcode_distance = np.sqrt(
        np.mean((q_barcode[:, ::-1] - c_barcode) ** 2, axis=1)
    )
    point_distance = np.abs(
        query.center_gr[query_local] - candidate.center_gr[candidate_local]
    )
    pair_distance = (
        0.35 * raw_distance
        + 0.20 * derivative_distance
        + 0.15 * rank_distance
        + 0.30 * barcode_distance
    )
    query_rows = query.rows[query_local].astype(np.int64, copy=False)
    candidate_rows = candidate.rows[candidate_local].astype(np.int64, copy=False)
    reciprocal_rank = np.array(
        [
            reverse_ranks.get((int(candidate_row), int(query_row)), top_k + 1)
            for query_row, candidate_row in zip(
                query_rows, candidate_rows, strict=True
            )
        ],
        dtype=np.int16,
    )
    if edge_type == "FF":
        edge_a = np.minimum(query_rows, candidate_rows)
        edge_b = np.maximum(query_rows, candidate_rows)
    else:
        edge_a = query_rows.copy()
        edge_b = candidate_rows.copy()
    candidate_coord = candidate.coordinates[candidate_local]
    return pd.DataFrame(
        {
            "method": method,
            "edge_type": edge_type,
            "query_well_id": query_well_id,
            "candidate_well_id": candidate_well_id,
            "query_row": query_rows,
            "candidate_row": candidate_rows,
            "edge_row_a": edge_a,
            "edge_row_b": edge_b,
            "query_md": query.coordinates[query_local],
            "candidate_md": (
                np.full(query_local.size, np.nan)
                if candidate_coordinate_is_tvt
                else candidate_coord
            ),
            "candidate_tvt": (
                candidate_coord
                if candidate_coordinate_is_tvt
                else np.full(query_local.size, np.nan)
            ),
            "query_scale": np.full(query_local.size, query_scale),
            "candidate_scale": np.full(query_local.size, candidate_scale),
            "raw_shape_distance": raw_distance,
            "derivative_distance": derivative_distance,
            "rank_distance": rank_distance,
            "barcode_distance": barcode_distance,
            "destroyed_barcode_distance": destroyed_barcode_distance,
            "point_gr_distance": point_distance,
            "pair_distance": pair_distance,
            "best_distance": pair_distance,
            "second_best_distance": np.full(query_local.size, np.inf),
            "absolute_margin": np.zeros(query_local.size),
            "relative_margin": np.zeros(query_local.size),
            "distance_ratio": np.ones(query_local.size),
            "reciprocal_rank": reciprocal_rank,
            "reciprocal_top_k": reciprocal_rank <= top_k,
            "scale_vote_count": np.ones(query_local.size, dtype=np.int16),
            "scale_agreement": np.zeros(query_local.size),
            "shared_neighbor_count": np.zeros(query_local.size, dtype=np.int16),
            "local_cycle_support_count": np.zeros(
                query_local.size, dtype=np.int16
            ),
            "query_observed_fraction": query.observed_fraction[query_local],
            "candidate_observed_fraction": candidate.observed_fraction[
                candidate_local
            ],
            "query_low_information": query.low_information[query_local],
            "candidate_low_information": candidate.low_information[
                candidate_local
            ],
            "raw_score": -pair_distance,
        }
    )


def _propose_scale_pair(
    query: DescriptorBatch,
    candidate: DescriptorBatch,
    *,
    method: Literal["main", "raw_nn"],
    edge_type: EdgeType,
    query_well_id: str,
    candidate_well_id: str,
    query_scale: float,
    candidate_scale: float,
    top_k: int,
    minimum_separation: float | None,
    candidate_coordinate_is_tvt: bool,
) -> tuple[pd.DataFrame, int]:
    feature_name: Literal["signature", "raw_shape"] = (
        "signature" if method == "main" else "raw_shape"
    )
    query_values = getattr(query, feature_name)
    candidate_values = getattr(candidate, feature_name)
    lists, _, index_rows = _neighbor_lists(
        query_values,
        candidate_values,
        query_coordinate=query.coordinates,
        candidate_coordinate=candidate.coordinates,
        top_k=top_k,
        minimum_separation=minimum_separation,
    )
    if method == "raw_nn":
        reverse: dict[tuple[int, int], int] = {}
    elif edge_type == "FF" and np.array_equal(query.rows, candidate.rows):
        reverse = {}
        for source_local, candidate_locals in enumerate(lists):
            source_row = int(query.rows[source_local])
            for rank, candidate_local_value in enumerate(
                candidate_locals, start=1
            ):
                reverse[
                    (
                        source_row,
                        int(candidate.rows[candidate_local_value]),
                    )
                ] = rank
    else:
        reverse = _reverse_rank_lookup(
            query,
            candidate,
            feature_name=feature_name,
            top_k=top_k,
            minimum_separation=minimum_separation,
        )
    query_local = np.repeat(
        np.arange(len(lists), dtype=np.int64),
        np.fromiter((len(values) for values in lists), dtype=np.int64),
    )
    candidate_local = (
        np.concatenate(lists)
        if lists and any(len(values) for values in lists)
        else np.empty((0,), dtype=np.int64)
    )
    frame = _pair_feature_frame(
        query,
        candidate,
        query_local,
        candidate_local,
        method=method,
        edge_type=edge_type,
        query_well_id=query_well_id,
        candidate_well_id=candidate_well_id,
        query_scale=query_scale,
        candidate_scale=candidate_scale,
        reverse_ranks=reverse,
        top_k=top_k,
        candidate_coordinate_is_tvt=candidate_coordinate_is_tvt,
    )
    return frame, index_rows


def _attach_consistency_and_prune(
    frame: pd.DataFrame,
    *,
    edge_type: EdgeType,
    top_k: int,
    scale_pair_count: int,
    graph: dict[int, frozenset[int]],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.sort_values(
        ["query_row", "pair_distance", "candidate_row", "query_scale", "candidate_scale"],
        kind="stable",
    ).reset_index(drop=True)
    if edge_type == "FF":
        exact_keys = ["edge_row_a", "edge_row_b"]
    else:
        exact_keys = ["query_row", "candidate_row"]
    result = result.drop_duplicates(exact_keys, keep="first").reset_index(drop=True)

    neighborhood_width = 16.0
    candidate_coordinate = np.where(
        result["candidate_tvt"].notna(),
        result["candidate_tvt"],
        result["candidate_md"],
    )
    result["_candidate_neighborhood"] = np.rint(
        candidate_coordinate / neighborhood_width
    ).astype(np.int64)
    votes = (
        result.groupby(
            ["query_row", "_candidate_neighborhood"], sort=False
        )["query_scale"]
        .transform("count")
        .clip(upper=np.iinfo(np.int16).max)
    )
    result["scale_vote_count"] = votes.astype(np.int16)
    result["scale_agreement"] = result["scale_vote_count"] / float(
        max(1, scale_pair_count)
    )

    group = result.groupby("query_row", sort=False)["pair_distance"]
    best = group.transform("min")
    second_lookup: dict[int, float] = {}
    for query_row, values in group:
        ordered = np.sort(values.to_numpy(dtype=np.float64), kind="stable")
        second_lookup[int(query_row)] = (
            float(ordered[1]) if ordered.size > 1 else float(ordered[0])
        )
    second = result["query_row"].map(second_lookup).to_numpy(dtype=np.float64)
    result["best_distance"] = best.to_numpy(dtype=np.float64)
    result["second_best_distance"] = second
    result["absolute_margin"] = np.maximum(second - result["best_distance"], 0.0)
    result["relative_margin"] = result["absolute_margin"] / np.maximum(
        second, 1e-12
    )
    result["distance_ratio"] = result["best_distance"] / np.maximum(second, 1e-12)

    result = (
        result.sort_values(
            ["query_row", "pair_distance", "candidate_row"], kind="stable"
        )
        .groupby("query_row", sort=False, as_index=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    if edge_type in {"PF", "FF"}:
        shared = shared_neighbor_count(
            result["query_row"].to_numpy(dtype=np.int64),
            result["candidate_row"].to_numpy(dtype=np.int64),
            graph,
        )
    else:
        shared = np.zeros(len(result), dtype=np.int16)
    result["shared_neighbor_count"] = shared
    result["local_cycle_support_count"] = local_cycle_support_count(
        result["reciprocal_top_k"].to_numpy(dtype=bool),
        result["scale_vote_count"].to_numpy(dtype=np.int16),
        shared,
    )
    result["raw_score"] = (
        -result["pair_distance"]
        + 0.20 * result["relative_margin"]
        + 0.08 * result["reciprocal_top_k"].astype(float)
        + 0.04 * result["scale_vote_count"].astype(float)
        + 0.03 * result["local_cycle_support_count"].astype(float)
    )
    return result.drop(columns=["_candidate_neighborhood"])


def _nearest_point_candidates(
    query: DescriptorBatch,
    candidate: DescriptorBatch,
    *,
    edge_type: EdgeType,
    query_well_id: str,
    candidate_well_id: str,
    minimum_separation: float | None,
    candidate_coordinate_is_tvt: bool,
) -> pd.DataFrame:
    if query.eligible_rows == 0 or candidate.eligible_rows == 0:
        return _empty_edges()
    finite_candidate = np.flatnonzero(np.isfinite(candidate.center_gr))
    if finite_candidate.size == 0:
        return _empty_edges()
    order = finite_candidate[
        np.argsort(candidate.center_gr[finite_candidate], kind="stable")
    ]
    sorted_gr = candidate.center_gr[order]
    query_local: list[int] = []
    candidate_local: list[int] = []
    for q_local, value in enumerate(query.center_gr):
        if not math.isfinite(float(value)):
            continue
        position = int(np.searchsorted(sorted_gr, value, side="left"))
        best: tuple[float, int] | None = None
        maximum_scan = min(order.size, 512)
        for offset in range(maximum_scan):
            choices = (position - 1 - offset, position + offset)
            for choice in choices:
                if choice < 0 or choice >= order.size:
                    continue
                c_local = int(order[choice])
                if minimum_separation is not None and (
                    abs(
                        float(
                            candidate.coordinates[c_local]
                            - query.coordinates[q_local]
                        )
                    )
                    < minimum_separation
                ):
                    continue
                key = (
                    abs(float(candidate.center_gr[c_local] - value)),
                    int(candidate.rows[c_local]),
                )
                if best is None or key < best:
                    best = key
            if best is not None and offset >= 4:
                break
        if best is None:
            continue
        selected_row = best[1]
        c_local = int(np.searchsorted(candidate.rows, selected_row))
        query_local.append(q_local)
        candidate_local.append(c_local)
    frame = _pair_feature_frame(
        query,
        candidate,
        np.asarray(query_local, dtype=np.int64),
        np.asarray(candidate_local, dtype=np.int64),
        method="point_gr",
        edge_type=edge_type,
        query_well_id=query_well_id,
        candidate_well_id=candidate_well_id,
        query_scale=query.half_span,
        candidate_scale=candidate.half_span,
        reverse_ranks={},
        top_k=1,
        candidate_coordinate_is_tvt=candidate_coordinate_is_tvt,
    )
    if not frame.empty:
        frame["raw_score"] = -frame["point_gr_distance"]
        frame["reciprocal_rank"] = np.int16(2)
        frame["reciprocal_top_k"] = False
    return frame


def _requested_rows(
    cut_row: int, total_rows: int, query_rows: Iterable[int] | None
) -> np.ndarray:
    if query_rows is None:
        return np.arange(cut_row, total_rows, dtype=np.int64)
    rows = np.asarray(tuple(query_rows), dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < cut_row) or np.any(rows >= total_rows):
        raise ValueError("query_rows must be unique future-row indices")
    if np.unique(rows).size != rows.size:
        raise ValueError("query_rows must be unique future-row indices")
    return np.sort(rows, kind="stable")


def generate_loop_edges(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    well_id: str,
    cut_row: int | None = None,
    query_rows: Iterable[int] | None = None,
    edge_types: Iterable[EdgeType] = EDGE_TYPES,
    config: EdgeGenerationConfig | None = None,
    include_retrieval_controls: bool = True,
    candidate_typewell_well_id: str | None = None,
) -> EdgeGenerationResult:
    """Generate target-free nonlocal candidates for one well and one cut."""
    generation_config = config or EdgeGenerationConfig()
    requested_types = tuple(dict.fromkeys(edge_types))
    if not requested_types or any(value not in EDGE_TYPES for value in requested_types):
        raise ValueError("edge_types must be a non-empty subset of PF, FF, HT")
    horizontal_view = select_inference_view(horizontal, kind="horizontal")
    typewell_view = select_inference_view(typewell, kind="typewell")
    if tuple(horizontal_view.columns) != HORIZONTAL_INFERENCE_COLUMNS:
        raise ValueError("horizontal edge input must contain only inference columns")
    if tuple(typewell_view.columns) != TYPEWELL_INFERENCE_COLUMNS:
        raise ValueError("typewell edge input must contain only inference columns")
    inferred_cut, _ = validate_prefix(horizontal_view["TVT_input"])
    boundary = inferred_cut if cut_row is None else int(cut_row)
    if boundary != inferred_cut:
        raise ValueError("cut_row must equal the contiguous TVT_input prefix length")
    total_rows = len(horizontal_view)
    requested = _requested_rows(boundary, total_rows, query_rows)
    descriptor_config = _descriptor_config(generation_config)
    md = horizontal_view["MD"].to_numpy(dtype=np.float64, copy=False)
    horizontal_gr = horizontal_view["GR"].to_numpy(dtype=np.float64, copy=False)
    type_coordinate = typewell_view["TVT"].to_numpy(dtype=np.float64, copy=False)
    type_gr = typewell_view["GR"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(md).all() or not np.all(np.diff(md) > 0.0):
        raise ValueError("horizontal MD must be finite and strictly increasing")
    if not np.isfinite(type_coordinate).all() or not np.all(
        np.diff(type_coordinate) > 0.0
    ):
        raise ValueError("typewell coordinate must be finite and strictly increasing")

    descriptor_start = time.perf_counter()
    prefix_batches: dict[float, DescriptorBatch] = {}
    future_all_batches: dict[float, DescriptorBatch] = {}
    future_query_batches: dict[float, DescriptorBatch] = {}
    for scale in generation_config.horizontal_half_spans:
        prefix_batches[scale] = build_descriptor_batch(
            md,
            horizontal_gr,
            center_rows=np.arange(0, boundary, dtype=np.int64),
            half_span=scale,
            region_start=0,
            region_stop=boundary,
            config=descriptor_config,
        )
        future_all = build_descriptor_batch(
            md,
            horizontal_gr,
            center_rows=np.arange(boundary, total_rows, dtype=np.int64),
            half_span=scale,
            region_start=boundary,
            region_stop=total_rows,
            config=descriptor_config,
        )
        future_all_batches[scale] = future_all
        future_query_batches[scale] = _subset_batch(future_all, requested)
    type_batches: dict[float, DescriptorBatch] = {}
    if "HT" in requested_types:
        for scale in generation_config.typewell_half_spans:
            type_batches[scale] = build_descriptor_batch(
                type_coordinate,
                type_gr,
                center_rows=np.arange(type_coordinate.size, dtype=np.int64),
                half_span=scale,
                region_start=0,
                region_stop=type_coordinate.size,
                config=descriptor_config,
            )
    descriptor_seconds = time.perf_counter() - descriptor_start
    eligible_query_rows = tuple(
        sorted(
            set().union(
                *(
                    set(batch.rows.tolist())
                    for batch in future_query_batches.values()
                )
            )
        )
    )
    all_eligible_future_rows = tuple(
        sorted(
            set().union(
                *(
                    set(batch.rows.tolist())
                    for batch in future_all_batches.values()
                )
            )
        )
    )
    gr_observed_future_rows = int(
        np.isfinite(horizontal_gr[boundary:]).sum()
    )

    middle_scale = generation_config.horizontal_half_spans[
        len(generation_config.horizontal_half_spans) // 2
    ]
    graph = build_nonlocal_neighbor_graph(
        (prefix_batches[middle_scale], future_all_batches[middle_scale]),
        minimum_coordinate_separation=generation_config.minimum_ff_separation,
        neighbors=generation_config.graph_neighbors,
    )

    retrieval_start = time.perf_counter()
    output_frames: list[pd.DataFrame] = []
    candidate_counts: dict[str, int] = {}
    peak_candidates = 0
    peak_index_rows = 0
    type_candidate_well = candidate_typewell_well_id or well_id
    for edge_type in requested_types:
        main_parts: list[pd.DataFrame] = []
        raw_parts: list[pd.DataFrame] = []
        if edge_type == "PF":
            scale_pairs = tuple(
                (scale, scale)
                for scale in generation_config.horizontal_half_spans
            )
        elif edge_type == "FF":
            scale_pairs = tuple(
                (scale, scale)
                for scale in generation_config.horizontal_half_spans
            )
        else:
            horizontal_scales = generation_config.horizontal_half_spans
            type_scales = generation_config.typewell_half_spans
            scale_pairs = tuple(
                dict.fromkeys(
                    [
                        *(
                            (
                                horizontal_scale,
                                type_scales[
                                    min(index, len(type_scales) - 1)
                                ],
                            )
                            for index, horizontal_scale in enumerate(
                                horizontal_scales
                            )
                        ),
                        (horizontal_scales[-1], type_scales[-1]),
                    ]
                )
            )
        if edge_type in {"PF", "FF"}:
            raw_pair = (middle_scale, middle_scale)
        else:
            raw_pair = (
                middle_scale,
                generation_config.typewell_half_spans[
                    min(
                        len(generation_config.horizontal_half_spans) // 2,
                        len(generation_config.typewell_half_spans) - 1,
                    )
                ],
            )
        for query_scale, candidate_scale in scale_pairs:
            query_batch = future_query_batches[query_scale]
            if edge_type == "PF":
                candidate_batch = prefix_batches[candidate_scale]
                separation = None
                candidate_well = well_id
                coordinate_is_tvt = False
            elif edge_type == "FF":
                candidate_batch = future_all_batches[candidate_scale]
                separation = generation_config.minimum_ff_separation
                candidate_well = well_id
                coordinate_is_tvt = False
            else:
                candidate_batch = type_batches[candidate_scale]
                separation = None
                candidate_well = type_candidate_well
                coordinate_is_tvt = True
            main, index_rows = _propose_scale_pair(
                query_batch,
                candidate_batch,
                method="main",
                edge_type=edge_type,
                query_well_id=well_id,
                candidate_well_id=candidate_well,
                query_scale=query_scale,
                candidate_scale=candidate_scale,
                top_k=generation_config.top_k,
                minimum_separation=separation,
                candidate_coordinate_is_tvt=coordinate_is_tvt,
            )
            main_parts.append(main)
            peak_index_rows = max(peak_index_rows, index_rows)
            if include_retrieval_controls and (
                query_scale,
                candidate_scale,
            ) == raw_pair:
                raw, _ = _propose_scale_pair(
                    query_batch,
                    candidate_batch,
                    method="raw_nn",
                    edge_type=edge_type,
                    query_well_id=well_id,
                    candidate_well_id=candidate_well,
                    query_scale=query_scale,
                    candidate_scale=candidate_scale,
                    top_k=1,
                    minimum_separation=separation,
                    candidate_coordinate_is_tvt=coordinate_is_tvt,
                )
                raw_parts.append(raw)
        main_all = (
            pd.concat(main_parts, ignore_index=True)
            if main_parts
            else _empty_edges()
        )
        peak_candidates = max(peak_candidates, len(main_all))
        main_final = _attach_consistency_and_prune(
            main_all,
            edge_type=edge_type,
            top_k=generation_config.top_k,
            scale_pair_count=len(scale_pairs),
            graph=graph,
        )
        output_frames.append(main_final)
        candidate_counts[f"{edge_type}:main"] = len(main_final)

        if include_retrieval_controls:
            raw_all = (
                pd.concat(raw_parts, ignore_index=True)
                if raw_parts
                else _empty_edges()
            )
            raw_final = _attach_consistency_and_prune(
                raw_all,
                edge_type=edge_type,
                top_k=1,
                scale_pair_count=len(scale_pairs),
                graph={},
            )
            raw_final["reciprocal_top_k"] = False
            raw_final["shared_neighbor_count"] = np.int16(0)
            raw_final["local_cycle_support_count"] = np.int16(0)
            raw_final["scale_vote_count"] = np.int16(1)
            raw_final["scale_agreement"] = 0.0
            raw_final["raw_score"] = -raw_final["raw_shape_distance"]
            output_frames.append(raw_final)
            candidate_counts[f"{edge_type}:raw_nn"] = len(raw_final)

            smallest_query = future_query_batches[
                generation_config.horizontal_half_spans[0]
            ]
            if edge_type == "PF":
                smallest_candidate = prefix_batches[
                    generation_config.horizontal_half_spans[0]
                ]
                point_separation = None
                point_candidate_well = well_id
                point_is_tvt = False
            elif edge_type == "FF":
                smallest_candidate = future_all_batches[
                    generation_config.horizontal_half_spans[0]
                ]
                point_separation = generation_config.minimum_ff_separation
                point_candidate_well = well_id
                point_is_tvt = False
            else:
                smallest_candidate = type_batches[
                    generation_config.typewell_half_spans[0]
                ]
                point_separation = None
                point_candidate_well = type_candidate_well
                point_is_tvt = True
            point = _nearest_point_candidates(
                smallest_query,
                smallest_candidate,
                edge_type=edge_type,
                query_well_id=well_id,
                candidate_well_id=point_candidate_well,
                minimum_separation=point_separation,
                candidate_coordinate_is_tvt=point_is_tvt,
            )
            output_frames.append(point)
            candidate_counts[f"{edge_type}:point_gr"] = len(point)
    retrieval_seconds = time.perf_counter() - retrieval_start
    edges = (
        pd.concat(output_frames, ignore_index=True)
        if output_frames
        else _empty_edges()
    )
    if not edges.empty:
        edges = edges.sort_values(
            ["method", "edge_type", "query_row", "candidate_row"],
            kind="stable",
        ).reset_index(drop=True)
    return EdgeGenerationResult(
        edges=edges,
        all_future_rows=total_rows - boundary,
        gr_observed_future_rows=gr_observed_future_rows,
        requested_query_rows=int(requested.size),
        descriptor_eligible_future_rows=len(all_eligible_future_rows),
        eligible_query_rows=eligible_query_rows,
        candidate_counts=candidate_counts,
        descriptor_seconds=float(descriptor_seconds),
        retrieval_seconds=float(retrieval_seconds),
        peak_candidates=int(peak_candidates),
        peak_index_rows=int(peak_index_rows),
    )
