"""Target-free reciprocal, scale, and local graph consistency utilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .descriptors import DescriptorBatch


def build_nonlocal_neighbor_graph(
    batches: tuple[DescriptorBatch, ...],
    *,
    minimum_coordinate_separation: float,
    neighbors: int = 8,
) -> dict[int, frozenset[int]]:
    """Build an ephemeral same-well graph from descriptor signatures only."""
    usable = tuple(batch for batch in batches if batch.eligible_rows)
    if not usable:
        return {}
    rows = np.concatenate([batch.rows for batch in usable])
    coordinate = np.concatenate([batch.coordinates for batch in usable])
    signature = np.concatenate([batch.signature for batch in usable])
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    coordinate = coordinate[order]
    signature = signature[order]
    unique, unique_indices = np.unique(rows, return_index=True)
    rows = unique
    coordinate = coordinate[unique_indices]
    signature = signature[unique_indices]
    if rows.size < 3:
        return {int(row): frozenset() for row in rows}
    requested = min(rows.size, max(neighbors + 1, 32))
    index = NearestNeighbors(
        n_neighbors=requested,
        algorithm="auto",
        metric="euclidean",
        n_jobs=1,
    ).fit(signature)
    _, candidate_indices = index.kneighbors(signature, return_distance=True)
    graph: dict[int, frozenset[int]] = {}
    for source in range(rows.size):
        selected: list[int] = []
        for candidate in candidate_indices[source]:
            if candidate == source:
                continue
            if (
                abs(float(coordinate[candidate] - coordinate[source]))
                < minimum_coordinate_separation
            ):
                continue
            selected.append(int(rows[candidate]))
            if len(selected) >= neighbors:
                break
        graph[int(rows[source])] = frozenset(selected)
    return graph


def shared_neighbor_count(
    row_a: np.ndarray,
    row_b: np.ndarray,
    graph: Mapping[int, frozenset[int]],
) -> np.ndarray:
    """Count common nonlocal graph neighbors for aligned endpoint arrays."""
    if row_a.shape != row_b.shape:
        raise ValueError("shared-neighbor endpoint arrays must be aligned")
    result = np.zeros(row_a.size, dtype=np.int16)
    for index, (left, right) in enumerate(zip(row_a, row_b, strict=True)):
        result[index] = len(
            graph.get(int(left), frozenset())
            & graph.get(int(right), frozenset())
        )
    return result


def local_cycle_support_count(
    reciprocal_top_k: np.ndarray,
    scale_vote_count: np.ndarray,
    shared_neighbors: np.ndarray,
) -> np.ndarray:
    """Combine independent target-free closures into a bounded support count."""
    if not (
        reciprocal_top_k.shape
        == scale_vote_count.shape
        == shared_neighbors.shape
    ):
        raise ValueError("cycle-support feature arrays must be aligned")
    support = (
        reciprocal_top_k.astype(np.int16)
        + np.maximum(scale_vote_count.astype(np.int16) - 1, 0)
        + np.minimum(shared_neighbors.astype(np.int16), 3)
    )
    return support.astype(np.int16, copy=False)
