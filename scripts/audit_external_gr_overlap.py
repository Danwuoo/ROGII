from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AUDIT_DATE = "2026-07-28"
DEFAULT_SEED = 20260728
DEFAULT_WINDOW_SIZE = 256
DEFAULT_STRIDE = 8
DEFAULT_EMBEDDING_BINS = 32
DEFAULT_LSH_TABLES = 32
DEFAULT_LSH_BITS = 18
DEFAULT_REFINE_PER_QUERY = 4
DEFAULT_OVERLAP_THRESHOLD = 0.90
CHRONOLOG_COMMIT = "336e3e2a20df029ecccbe63611a78550204d64ae"


@dataclass
class WindowCollection:
    names: list[str]
    values: list[np.ndarray]
    flat_values: np.ndarray
    series_offsets: np.ndarray
    window_series: np.ndarray
    window_starts: np.ndarray
    embeddings: np.ndarray
    sample_count: int
    invalid_window_count: int

    @property
    def window_count(self) -> int:
        return int(self.window_starts.size)


def _read_series(paths: list[Path], column: str) -> tuple[list[str], list[np.ndarray]]:
    if not paths:
        raise FileNotFoundError("No input CSV files matched the requested pattern")
    names: list[str] = []
    values: list[np.ndarray] = []
    for path in paths:
        frame = pd.read_csv(path, usecols=[column])
        series = frame[column].to_numpy(dtype=np.float32, copy=True)
        if series.ndim != 1 or series.size == 0:
            raise ValueError(f"{path} has an empty or non-vector {column!r} column")
        if not np.isfinite(series).all():
            bad = int((~np.isfinite(series)).sum())
            raise ValueError(f"{path} contains {bad} non-finite {column!r} values")
        names.append(path.stem)
        values.append(series)
    return names, values


def _make_windows(
    names: list[str],
    values: list[np.ndarray],
    *,
    window_size: int,
    stride: int,
    embedding_bins: int,
) -> WindowCollection:
    if window_size <= 1:
        raise ValueError("window_size must be greater than one")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if embedding_bins <= 1 or window_size % embedding_bins:
        raise ValueError("embedding_bins must be a divisor of window_size")

    block_size = window_size // embedding_bins
    block_offsets = np.arange(0, window_size, block_size, dtype=np.int64)
    all_series: list[np.ndarray] = []
    all_starts: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    invalid_count = 0

    for series_index, series in enumerate(values):
        if series.size < window_size:
            continue
        starts = np.arange(
            0,
            series.size - window_size + 1,
            stride,
            dtype=np.int32,
        )
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(series, dtype=np.float64))
        )
        left = starts.astype(np.int64)[:, None] + block_offsets[None, :]
        embedding = (cumulative[left + block_size] - cumulative[left]) / block_size
        embedding -= embedding.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(embedding, axis=1)
        valid = norms > 1.0e-12
        invalid_count += int((~valid).sum())
        embedding[valid] /= norms[valid, None]
        embedding[~valid] = 0.0

        all_series.append(np.full(starts.size, series_index, dtype=np.int32))
        all_starts.append(starts)
        all_embeddings.append(embedding.astype(np.float32, copy=False))

    if not all_starts:
        raise ValueError("No input series is long enough for the requested window")

    sizes = np.fromiter((series.size for series in values), dtype=np.int64)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(sizes, dtype=np.int64))
    )
    return WindowCollection(
        names=names,
        values=values,
        flat_values=np.concatenate(values).astype(np.float32, copy=False),
        series_offsets=offsets[:-1],
        window_series=np.concatenate(all_series),
        window_starts=np.concatenate(all_starts),
        embeddings=np.concatenate(all_embeddings),
        sample_count=int(sizes.sum()),
        invalid_window_count=invalid_count,
    )


def _hash_codes(features: np.ndarray, hyperplanes: np.ndarray) -> np.ndarray:
    bits = features @ hyperplanes.T >= 0.0
    weights = np.left_shift(
        np.uint32(1), np.arange(hyperplanes.shape[0], dtype=np.uint32)
    )
    return bits.astype(np.uint32) @ weights


def _metric_embeddings(
    raw_embeddings: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "raw_affine_pearson":
        return raw_embeddings
    if metric == "linear_detrended_pearson":
        transformed = raw_embeddings.astype(np.float64, copy=True)
        positions = np.arange(transformed.shape[1], dtype=np.float64)
        positions -= positions.mean()
        slopes = transformed @ positions / np.dot(positions, positions)
        transformed -= slopes[:, None] * positions[None, :]
    elif metric == "first_difference_pearson":
        transformed = np.diff(raw_embeddings.astype(np.float64, copy=False), axis=1)
        transformed -= transformed.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown overlap metric: {metric}")

    norms = np.linalg.norm(transformed, axis=1)
    valid = norms > 1.0e-12
    transformed[valid] /= norms[valid, None]
    transformed[~valid] = 0.0
    return transformed.astype(np.float32, copy=False)


def _best_same_bucket(
    external: np.ndarray,
    queries: np.ndarray,
    external_codes: np.ndarray,
    query_codes: np.ndarray,
    *,
    score_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(external_codes, kind="stable")
    sorted_codes = external_codes[order]
    left = np.searchsorted(sorted_codes, query_codes, side="left")
    right = np.searchsorted(sorted_codes, query_codes, side="right")
    lengths = (right - left).astype(np.int64, copy=False)
    matched_queries = np.flatnonzero(lengths)
    if matched_queries.size == 0:
        return (
            np.full(queries.shape[0], -np.inf, dtype=np.float32),
            np.full(queries.shape[0], -1, dtype=np.int64),
            0,
        )

    matched_lengths = lengths[matched_queries]
    candidate_count = int(matched_lengths.sum())
    query_index = np.repeat(matched_queries, matched_lengths)
    group_offsets = np.repeat(
        np.cumsum(matched_lengths) - matched_lengths,
        matched_lengths,
    )
    within_group = np.arange(candidate_count, dtype=np.int64) - group_offsets
    external_index = order[
        np.repeat(left[matched_queries], matched_lengths) + within_group
    ]

    scores = np.empty(candidate_count, dtype=np.float32)
    for batch_start in range(0, candidate_count, score_batch_size):
        batch_stop = min(batch_start + score_batch_size, candidate_count)
        q_index = query_index[batch_start:batch_stop]
        e_index = external_index[batch_start:batch_stop]
        scores[batch_start:batch_stop] = np.einsum(
            "ij,ij->i",
            queries[q_index],
            external[e_index],
            optimize=True,
        )

    best_scores = np.full(queries.shape[0], -np.inf, dtype=np.float32)
    np.maximum.at(best_scores, query_index, scores)
    is_best = scores == best_scores[query_index]
    best_indices = np.full(queries.shape[0], -1, dtype=np.int64)
    best_indices[query_index[is_best]] = external_index[is_best]
    return best_scores, best_indices, candidate_count


def _update_top_matches(
    top_scores: np.ndarray,
    top_indices: np.ndarray,
    new_scores: np.ndarray,
    new_indices: np.ndarray,
) -> None:
    duplicate = top_indices == new_indices[:, None]
    duplicate &= new_indices[:, None] >= 0
    duplicate_rows = np.flatnonzero(duplicate.any(axis=1))
    if duplicate_rows.size:
        duplicate_columns = duplicate[duplicate_rows].argmax(axis=1)
        top_scores[duplicate_rows, duplicate_columns] = np.maximum(
            top_scores[duplicate_rows, duplicate_columns],
            new_scores[duplicate_rows],
        )

    novel = (new_indices >= 0) & ~duplicate.any(axis=1)
    replace_columns = top_scores.argmin(axis=1)
    replace_rows = np.flatnonzero(
        novel
        & (new_scores > top_scores[np.arange(top_scores.shape[0]), replace_columns])
    )
    if replace_rows.size:
        columns = replace_columns[replace_rows]
        top_scores[replace_rows, columns] = new_scores[replace_rows]
        top_indices[replace_rows, columns] = new_indices[replace_rows]


def _lsh_candidates(
    external: np.ndarray,
    queries: np.ndarray,
    *,
    seed: int,
    tables: int,
    bits: int,
    refine_per_query: int,
    score_batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if bits <= 0 or bits > 31:
        raise ValueError("lsh_bits must be in [1, 31]")
    if tables <= 0:
        raise ValueError("lsh_tables must be positive")
    if refine_per_query <= 0:
        raise ValueError("refine_per_query must be positive")

    rng = np.random.default_rng(seed)
    top_scores = np.full(
        (queries.shape[0], refine_per_query),
        -np.inf,
        dtype=np.float32,
    )
    top_indices = np.full(
        (queries.shape[0], refine_per_query),
        -1,
        dtype=np.int64,
    )
    candidate_count = 0

    for table_index in range(tables):
        hyperplanes = rng.standard_normal(
            (bits, external.shape[1]),
            dtype=np.float32,
        )
        external_codes = _hash_codes(external, hyperplanes)
        query_codes = _hash_codes(queries, hyperplanes)
        scores, indices, table_candidate_count = _best_same_bucket(
            external,
            queries,
            external_codes,
            query_codes,
            score_batch_size=score_batch_size,
        )
        candidate_count += table_candidate_count
        _update_top_matches(top_scores, top_indices, scores, indices)
        if (table_index + 1) % 4 == 0 or table_index + 1 == tables:
            print(
                f"LSH tables complete: {table_index + 1}/{tables}",
                file=sys.stderr,
                flush=True,
            )

    valid_counts = (top_indices >= 0).sum(axis=1)
    diagnostics = {
        "raw_same_bucket_candidate_visits": candidate_count,
        "query_orientations_with_candidate": int((valid_counts > 0).sum()),
        "query_orientations_without_candidate": int((valid_counts == 0).sum()),
        "retained_candidates_for_exact_refinement": int(valid_counts.sum()),
    }
    return top_indices, diagnostics


def _gather_windows(
    collection: WindowCollection,
    window_indices: np.ndarray,
    sample_offsets: np.ndarray,
) -> np.ndarray:
    series = collection.window_series[window_indices]
    starts = collection.window_starts[window_indices].astype(np.int64)
    absolute = collection.series_offsets[series] + starts
    return collection.flat_values[absolute[:, None] + sample_offsets[None, :]]


def _exact_refinement(
    external: WindowCollection,
    query: WindowCollection,
    candidate_indices: np.ndarray,
    query_is_reversed: np.ndarray,
    query_split: np.ndarray,
    query_split_window: np.ndarray,
    *,
    metric: str,
    window_size: int,
    threshold: float,
    batch_size: int,
) -> dict[str, Any]:
    orientation_indices, slots = np.nonzero(candidate_indices >= 0)
    external_indices = candidate_indices[orientation_indices, slots]
    sample_offsets = np.arange(window_size, dtype=np.int64)
    correlations = np.empty(orientation_indices.size, dtype=np.float32)

    for batch_start in range(0, orientation_indices.size, batch_size):
        batch_stop = min(batch_start + batch_size, orientation_indices.size)
        orientations = orientation_indices[batch_start:batch_stop]
        base_query = query_split_window[orientations]
        external_batch = external_indices[batch_start:batch_stop]
        query_windows = _gather_windows(query, base_query, sample_offsets)
        reverse = query_is_reversed[orientations]
        if reverse.any():
            query_windows = query_windows.copy()
            query_windows[reverse] = query_windows[reverse, ::-1]
        external_windows = _gather_windows(external, external_batch, sample_offsets)

        query_windows = _transform_exact_windows(query_windows, metric)
        external_windows = _transform_exact_windows(external_windows, metric)
        numerator = np.einsum(
            "ij,ij->i", query_windows, external_windows, optimize=True
        )
        denominator = np.linalg.norm(query_windows, axis=1)
        denominator *= np.linalg.norm(external_windows, axis=1)
        correlations[batch_start:batch_stop] = np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, np.nan, dtype=np.float64),
            where=denominator > 1.0e-12,
        ).astype(np.float32)

    finite = np.isfinite(correlations)
    if not finite.any():
        raise RuntimeError("Exact refinement produced no finite correlations")

    def describe_match(candidate_position: int) -> dict[str, Any]:
        orientation = int(orientation_indices[candidate_position])
        query_window = int(query_split_window[orientation])
        external_window = int(external_indices[candidate_position])
        split = str(query_split[orientation])
        query_series = int(query.window_series[query_window])
        external_series = int(external.window_series[external_window])
        return {
            "pearson_correlation": float(correlations[candidate_position]),
            "metric": metric,
            "query_split": split,
            "query_well": query.names[query_series],
            "query_window_start_sample": int(query.window_starts[query_window]),
            "external_log": external.names[external_series],
            "external_window_start_sample": int(
                external.window_starts[external_window]
            ),
            "orientation": (
                "external_reversed"
                if bool(query_is_reversed[orientation])
                else "forward"
            ),
        }

    finite_positions = np.flatnonzero(finite)
    overall_position = int(finite_positions[np.argmax(correlations[finite_positions])])
    by_split: dict[str, Any] = {}
    for split in ("train", "test"):
        positions = np.flatnonzero(finite & (query_split[orientation_indices] == split))
        if positions.size:
            best = int(positions[np.argmax(correlations[positions])])
            by_split[split] = {
                "max_pearson_correlation": float(correlations[best]),
                "best_match": describe_match(best),
                "refined_candidate_count": int(positions.size),
                "screened_candidate_count_at_or_above_threshold": int(
                    (correlations[positions] >= threshold).sum()
                ),
            }

    ranked = finite_positions[
        np.argsort(correlations[finite_positions], kind="stable")[-10:][::-1]
    ]
    return {
        "metric": metric,
        "max_pearson_correlation": float(correlations[overall_position]),
        "best_match": describe_match(overall_position),
        "screened_candidate_count_at_or_above_threshold": int(
            (correlations[finite] >= threshold).sum()
        ),
        "by_split": by_split,
        "top_matches": [describe_match(int(position)) for position in ranked],
    }


def _transform_exact_windows(
    windows: np.ndarray,
    metric: str,
) -> np.ndarray:
    transformed = windows.astype(np.float64, copy=True)
    if metric == "first_difference_pearson":
        transformed = np.diff(transformed, axis=1)
    transformed -= transformed.mean(axis=1, keepdims=True)
    if metric == "linear_detrended_pearson":
        positions = np.arange(transformed.shape[1], dtype=np.float64)
        positions -= positions.mean()
        slopes = transformed @ positions / np.dot(positions, positions)
        transformed -= slopes[:, None] * positions[None, :]
    elif metric not in ("raw_affine_pearson", "first_difference_pearson"):
        raise ValueError(f"Unknown overlap metric: {metric}")
    return transformed


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _manifest_sha256(root: Path, files: list[Path]) -> str:
    manifest = hashlib.sha256()
    for path in sorted(files):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.update(path.relative_to(root).as_posix().encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(digest.encode("ascii"))
        manifest.update(b"\n")
    return manifest.hexdigest()


def _source_inventory(
    external_root: Path,
    *,
    resolved_commit: str | None,
    manifest_sha256: str,
    manifest_file_count: int,
) -> list[dict[str, Any]]:
    common_not_downloaded = {
        "download_status": "not_downloaded; metadata/revision inspected only",
        "download_date": None,
        "local_path": None,
    }
    return [
        {
            "id": "chronolog_data",
            "name": "ChronoLog public Permian Basin gamma-ray logs",
            "asset_type": "external_dataset",
            "source_urls": [
                "https://github.com/zsylvester/chronolog_data",
                "https://zsylvester.github.io/papers/chronolog/",
            ],
            "authors_or_maintainers": ["Zoltán Sylvester"],
            "version": {
                "git_commit": CHRONOLOG_COMMIT,
                "resolved_local_git_commit": resolved_commit,
                "commit_date": "2022-08-08",
            },
            "license": {
                "spdx": "CC-BY-4.0",
                "evidence_url": (
                    "https://github.com/zsylvester/chronolog_data/blob/"
                    f"{CHRONOLOG_COMMIT}/license.txt"
                ),
                "upstream_data_note": (
                    "Derived from the S&P Global Permian Basin Digital Logs "
                    "Subscription and shared publicly with permission."
                ),
            },
            "acquisition": {
                "download_status": "downloaded for provenance/overlap audit",
                "download_date": AUDIT_DATE,
                "local_path": str(external_root.resolve()),
                "method": "public git clone with blob filtering/sparse materialization",
            },
            "integrity": {
                "revision_is_primary_identifier": True,
                "local_data_manifest_sha256": manifest_sha256,
                "manifest_file_count": manifest_file_count,
                "manifest_definition": (
                    "SHA-256 over sorted relative path, NUL, per-file SHA-256, "
                    "newline for 681 logs/*.csv files plus "
                    "example_dataset_coordinates.csv"
                ),
            },
            "fairness_and_offline": {
                "publicly_available": True,
                "free_to_download": True,
                "offline_retraining_possible": True,
                "test_time_network_required": False,
                "assessment": (
                    "Public and license-compatible in isolation, but excluded from "
                    "this research round because the raw overlap screen and "
                    "same-basin provenance do not clear leakage risk."
                ),
            },
            "inference_inputs_if_used": {
                "competition_fields": ["GR"],
                "external_runtime_inputs": [],
                "proposed_use": "GR-only self-supervised pretraining",
            },
            "leakage_risk": {
                "level": "low-but-not-zero",
                "target_or_hidden_test_labels_present": False,
                "risk": (
                    "Same broad Permian region could contain related wells. The "
                    "deterministic GR-window screen below finds raw Pearson matches "
                    "above 0.90. Detrended/differenced motif screens cannot prove "
                    "independent provenance."
                ),
                "mitigation": (
                    "Do not use ChronoLog in this round's model training or "
                    "inference. Any future reconsideration requires a stronger "
                    "well-identity/provenance audit, GR only, no external "
                    "coordinates/depths, and random-init plus competition-only "
                    "outer-fold-safe controls."
                ),
            },
            "used_this_round": {
                "source_audit": True,
                "overlap_screen": True,
                "model_training": False,
                "model_inference": False,
                "excluded_due_to_overlap_risk": True,
            },
        },
        {
            "id": "ibm_granite_ttm_r1",
            "name": "IBM Granite Tiny Time Mixer R1",
            "asset_type": "pretrained_time_series_model",
            "source_urls": [
                "https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1",
                "https://github.com/ibm-granite/granite-tsfm",
                "https://arxiv.org/abs/2401.03955",
            ],
            "authors_or_maintainers": [
                "IBM Granite Time Series team",
                "Vijay Ekambaram et al.",
            ],
            "version": {
                "hugging_face_revision": ("f04bebdd4c13475b006ce72672e74c9dc28871dc"),
                "revision_last_modified": "2025-01-30",
                "parameter_count": 805000,
                "model_file_bytes": 3240592,
            },
            "license": {
                "spdx": "Apache-2.0",
                "evidence_url": (
                    "https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1"
                ),
            },
            "acquisition": dict(common_not_downloaded),
            "integrity": {
                "model_safetensors_sha256": (
                    "30f8d9be619518e18c4c15c6ee504e491b9624b48e251661cd5f967bce097731"
                ),
                "revision_is_primary_identifier": True,
            },
            "fairness_and_offline": {
                "publicly_available": True,
                "ungated": True,
                "free_to_download": True,
                "offline_inference_possible": True,
                "offline_retraining_code_available": True,
                "test_time_network_required": False,
                "assessment": (
                    "Public fixed external checkpoint predating the competition; "
                    "must pin revision and cache for offline Kaggle execution."
                ),
            },
            "inference_inputs_if_used": {
                "competition_fields": [
                    "horizontal MD/GR",
                    "typewell TVT/GR",
                ],
                "external_runtime_inputs": [],
                "proposed_use": (
                    "Frozen encoder only; retain a separate raw-datum/common-mode "
                    "head because per-series scaling removes absolute level."
                ),
            },
            "leakage_risk": {
                "level": "low",
                "target_or_hidden_test_labels_present": False,
                "risk": (
                    "Large public pretraining-corpus provenance must remain pinned; "
                    "no evidence it contains ROGII targets, and the pinned revision "
                    "predates the challenge."
                ),
                "mitigation": (
                    "Frozen checkpoint across folds; any competition-data "
                    "adaptation remains outer-fold safe; compare pretrained against "
                    "identical random-init controls."
                ),
            },
            "used_this_round": {
                "source_audit": True,
                "weights_downloaded": False,
                "model_training": False,
                "model_inference": False,
            },
        },
        {
            "id": "ts2vec",
            "name": "TS2Vec official implementation",
            "asset_type": "self_supervised_representation_code",
            "source_urls": [
                "https://github.com/zhihanyue/ts2vec",
                "https://ojs.aaai.org/index.php/AAAI/article/view/20881",
            ],
            "authors_or_maintainers": [
                "Zhihan Yue",
                "Yujing Wang",
                "Juanyong Duan",
                "Tianmeng Huang",
                "Yunhai Tong",
                "Bixiong Xu",
            ],
            "version": {"git_commit": "b0088e14a99706c05451316dc6db8d3da9351163"},
            "license": {
                "spdx": "MIT",
                "evidence_url": "https://github.com/zhihanyue/ts2vec/blob/main/LICENSE",
            },
            "acquisition": dict(common_not_downloaded),
            "integrity": {"revision_is_primary_identifier": True},
            "fairness_and_offline": {
                "publicly_available": True,
                "free_to_download": True,
                "offline_retraining_possible": True,
                "test_time_network_required": False,
                "assessment": (
                    "Use architecture/code only and train on outer-training wells; "
                    "do not import competition-adapted weights across folds."
                ),
            },
            "inference_inputs_if_used": {
                "competition_fields": [
                    "horizontal MD/GR/TVT_input prefix",
                    "typewell TVT/GR",
                ],
                "external_runtime_inputs": [],
            },
            "leakage_risk": {
                "level": "low-if-fold-safe",
                "target_or_hidden_test_labels_present": False,
                "mitigation": (
                    "All ROGII-data pretraining/fitting occurs inside each outer "
                    "training fold; wrong-typewell and shuffled-prefix controls."
                ),
            },
            "used_this_round": {
                "source_audit": True,
                "code_downloaded": False,
                "model_training": False,
            },
        },
        {
            "id": "tf_c",
            "name": "TF-C official implementation",
            "asset_type": "time_frequency_contrastive_code",
            "source_urls": [
                "https://github.com/mims-harvard/TFC-pretraining",
                "https://proceedings.neurips.cc/paper_files/paper/2022/hash/"
                "a500b6664a1e7b8f113c61e66c35a512-Abstract-Conference.html",
            ],
            "authors_or_maintainers": [
                "Xiang Zhang",
                "Ziyuan Zhao",
                "Theodoros Tsiligkaridis",
                "Marinka Zitnik",
            ],
            "version": {"git_commit": "96675826e9ef234a9b01cc63d484c66cb0441bc0"},
            "license": {
                "code_spdx": "MIT",
                "evidence_url": (
                    "https://github.com/mims-harvard/TFC-pretraining/blob/main/LICENSE"
                ),
                "pretraining_data_license": "mixed; some upstream datasets are NC",
            },
            "acquisition": dict(common_not_downloaded),
            "integrity": {"revision_is_primary_identifier": True},
            "fairness_and_offline": {
                "publicly_available": True,
                "code_offline_retraining_possible": True,
                "test_time_network_required": False,
                "assessment": (
                    "Architecture/code is usable under MIT. Released pretrained "
                    "weights are excluded because upstream training-data licenses "
                    "are not uniformly competition-compatible."
                ),
            },
            "inference_inputs_if_used": {
                "competition_fields": ["horizontal GR", "typewell GR"],
                "external_runtime_inputs": [],
            },
            "leakage_risk": {
                "level": "license-risk-for-released-weights",
                "target_or_hidden_test_labels_present": False,
                "mitigation": (
                    "Do not use released pretrained weights; train the "
                    "time/frequency representation only on outer-training wells or "
                    "clearly licensed external GR."
                ),
            },
            "used_this_round": {
                "source_audit": True,
                "code_downloaded": False,
                "pretrained_weights_used": False,
                "excluded_pretrained_weights": True,
            },
        },
        {
            "id": "rogii_q3d_toolkit",
            "name": "ROGII Geosteering Toolkit Q-3D tortuosity implementation",
            "asset_type": "competition_specific_feature_code",
            "source_urls": [
                "https://github.com/mycarta/rogii-geosteering-toolkit",
                "https://doi.org/10.1016/j.petlm.2021.03.008",
            ],
            "authors_or_maintainers": [
                "Matteo Niccoli (mycarta)",
                "J. Jing et al. (Q-3D method)",
            ],
            "version": {"git_commit": "d83213f97999423908228c85e67a96072c8b2a5a"},
            "license": {
                "spdx": "MIT",
                "evidence_url": (
                    "https://github.com/mycarta/"
                    "rogii-geosteering-toolkit/blob/main/LICENSE"
                ),
            },
            "acquisition": dict(common_not_downloaded),
            "integrity": {"revision_is_primary_identifier": True},
            "fairness_and_offline": {
                "publicly_available": True,
                "free_to_download": True,
                "offline_reimplementation_possible": True,
                "test_time_network_required": False,
                "assessment": (
                    "Public competition-specific code is equally accessible as of "
                    f"{AUDIT_DATE}; use only the inference-safe Q-3D calculation."
                ),
            },
            "inference_inputs_if_used": {
                "competition_fields": ["horizontal MD", "X", "Y", "Z"],
                "external_runtime_inputs": [],
            },
            "leakage_risk": {
                "level": "medium-for-whole-toolkit; low-for-isolated-q3d",
                "risk": (
                    "The wider toolkit contains target-informed and "
                    "competition-specific components that are not automatically "
                    "valid under this project's strict folds."
                ),
                "mitigation": (
                    "Reimplement/audit only Q-3D from MD/X/Y/Z; do not import "
                    "formation labels, target statistics, offset priors, folds, or "
                    "predictions."
                ),
            },
            "used_this_round": {
                "source_audit": True,
                "code_downloaded": False,
                "feature_implemented": False,
                "model_training": False,
            },
        },
        {
            "id": "nbeatsx",
            "name": "N-BEATSx official implementation",
            "asset_type": "interpretable_basis_architecture_code",
            "source_urls": [
                "https://github.com/cchallu/nbeatsx",
                "https://arxiv.org/abs/2104.05522",
            ],
            "authors_or_maintainers": ["Cristian Challu et al."],
            "version": {"git_commit": "fe116d21785fca55670d258756e7c35fcb613eca"},
            "license": {
                "spdx": "MIT",
                "evidence_url": "https://github.com/cchallu/nbeatsx/blob/main/LICENSE",
            },
            "acquisition": dict(common_not_downloaded),
            "integrity": {"revision_is_primary_identifier": True},
            "fairness_and_offline": {
                "publicly_available": True,
                "offline_reimplementation_possible": True,
                "test_time_network_required": False,
                "assessment": "Architecture reference only; no external weights.",
            },
            "inference_inputs_if_used": {
                "competition_fields": [
                    "legal horizontal descriptors",
                    "typewell TVT/GR descriptors",
                ],
                "external_runtime_inputs": [],
                "proposed_use": (
                    "Common-mode, slope, and low-order smooth-basis anchor head."
                ),
            },
            "leakage_risk": {
                "level": "low",
                "target_or_hidden_test_labels_present": False,
                "mitigation": "Fit all basis-head parameters on outer-training wells.",
            },
            "used_this_round": {
                "source_audit": True,
                "code_downloaded": False,
                "model_training": False,
            },
        },
        {
            "id": "s4",
            "name": "Structured State Spaces (S4) official implementation",
            "asset_type": "long_sequence_architecture_code",
            "source_urls": [
                "https://github.com/state-spaces/s4",
                "https://openreview.net/forum?id=uYLFoz1vlAC",
            ],
            "authors_or_maintainers": [
                "Albert Gu",
                "Karan Goel",
                "Christopher Ré",
            ],
            "version": {"git_commit": "e757cef57d89e448c413de7325ed5601aceaac13"},
            "license": {
                "spdx": "Apache-2.0",
                "evidence_url": "https://github.com/state-spaces/s4/blob/main/LICENSE",
            },
            "acquisition": dict(common_not_downloaded),
            "integrity": {"revision_is_primary_identifier": True},
            "fairness_and_offline": {
                "publicly_available": True,
                "offline_retraining_possible": True,
                "test_time_network_required": False,
                "assessment": "Architecture/code only; train outer-fold-safe.",
            },
            "inference_inputs_if_used": {
                "competition_fields": [
                    "horizontal MD/X/Y/Z/GR/TVT_input",
                    "typewell TVT/GR",
                ],
                "external_runtime_inputs": [],
            },
            "leakage_risk": {
                "level": "low-if-fold-safe",
                "target_or_hidden_test_labels_present": False,
                "mitigation": (
                    "No cross-fold competition pretraining; sequence scaling and "
                    "all learned representations fit outer-training wells only."
                ),
            },
            "used_this_round": {
                "source_audit": True,
                "code_downloaded": False,
                "model_training": False,
            },
        },
        {
            "id": "mtp_multimode",
            "name": "Multi-trajectory prediction with MTP loss",
            "asset_type": "multimodal_prediction_code_and_optional_data",
            "source_urls": [
                "https://github.com/alin256/"
                "multi-mode-prediction-with-mtp-loss/tree/v1.2",
                "https://doi.org/10.1029/2021EA002186",
            ],
            "authors_or_maintainers": ["Sergey Alyaev", "Ahmed H. Elsheikh"],
            "version": {
                "tag": "v1.2",
                "git_commit": "09b1ba9a7fb9e3d19528c230d63c8d89f5bb8895",
            },
            "license": {
                "spdx": "NCSA",
                "evidence_url": (
                    "https://github.com/alin256/"
                    "multi-mode-prediction-with-mtp-loss/blob/v1.2/LICENSE"
                ),
            },
            "acquisition": dict(common_not_downloaded),
            "integrity": {"revision_is_primary_identifier": True},
            "fairness_and_offline": {
                "publicly_available": True,
                "offline_reimplementation_possible": True,
                "test_time_network_required": False,
                "assessment": (
                    "Loss/architecture can be reimplemented. The optional 467 MB "
                    "synthetic supporting dataset is excluded due to domain "
                    "mismatch and unnecessary provenance risk."
                ),
            },
            "inference_inputs_if_used": {
                "competition_fields": [
                    "legal horizontal descriptors",
                    "typewell descriptors",
                ],
                "external_runtime_inputs": [],
                "proposed_use": (
                    "Modes in low-dimensional common/slope/shape coefficient space."
                ),
            },
            "leakage_risk": {
                "level": "low-for-code; avoid-supporting-data",
                "target_or_hidden_test_labels_present": False,
                "mitigation": (
                    "Use loss/architecture only; do not use authors' synthetic "
                    "supporting data; fit mode probabilities outer-fold-safe."
                ),
            },
            "used_this_round": {
                "source_audit": True,
                "code_downloaded": False,
                "supporting_data_downloaded": False,
                "model_training": False,
            },
        },
        {
            "id": "moment_1_large",
            "name": "MOMENT-1-large",
            "asset_type": "pretrained_time_series_model",
            "source_urls": [
                "https://huggingface.co/AutonLab/MOMENT-1-large",
                "https://github.com/moment-timeseries-foundation-model/moment",
                "https://proceedings.mlr.press/v235/goswami24a.html",
            ],
            "authors_or_maintainers": ["Mononito Goswami et al."],
            "version": {
                "hugging_face_revision": ("411e288267f82cce86296dbe4d6c8bc533cc162f"),
                "approximate_parameter_count": 38000000,
                "approximate_download_bytes": 151600000,
            },
            "license": {
                "spdx": "MIT",
                "evidence_url": "https://huggingface.co/AutonLab/MOMENT-1-large",
            },
            "acquisition": dict(common_not_downloaded),
            "integrity": {"revision_is_primary_identifier": True},
            "fairness_and_offline": {
                "publicly_available": True,
                "free_to_download": True,
                "offline_inference_possible": True,
                "test_time_network_required": False,
                "assessment": (
                    "Public checkpoint, but larger and lower-priority than TTM; "
                    "requires a separate pinned-weight/provenance audit before use."
                ),
            },
            "inference_inputs_if_used": {
                "competition_fields": ["horizontal GR", "typewell GR"],
                "external_runtime_inputs": [],
            },
            "leakage_risk": {
                "level": "low-but-requires-pretraining-corpus-audit",
                "target_or_hidden_test_labels_present": False,
                "mitigation": (
                    "Pin revision; verify corpus provenance; frozen checkpoint "
                    "across folds; compare pretrained and random-init encoders."
                ),
            },
            "used_this_round": {
                "source_audit": True,
                "weights_downloaded": False,
                "model_training": False,
            },
        },
    ]


def _build_audit(
    *,
    external_root: Path,
    data_root: Path,
    window_size: int,
    stride: int,
    embedding_bins: int,
    lsh_tables: int,
    lsh_bits: int,
    refine_per_query: int,
    seed: int,
    threshold: float,
    score_batch_size: int,
    exact_batch_size: int,
) -> dict[str, Any]:
    log_root = external_root / "logs"
    external_paths = sorted(log_root.glob("*.csv"))
    train_paths = sorted((data_root / "train").glob("*__typewell.csv"))
    test_paths = sorted((data_root / "test").glob("*__typewell.csv"))
    if not external_paths:
        raise FileNotFoundError(f"No ChronoLog CSV files found under {log_root}")
    if not train_paths or not test_paths:
        raise FileNotFoundError(
            f"ROGII train/test typewell CSVs were not found under {data_root}"
        )

    external_names, external_values = _read_series(external_paths, "gamma ray")
    train_names, train_values = _read_series(train_paths, "GR")
    test_names, test_values = _read_series(test_paths, "GR")
    query_names = train_names + test_names
    query_values = train_values + test_values

    external = _make_windows(
        external_names,
        external_values,
        window_size=window_size,
        stride=stride,
        embedding_bins=embedding_bins,
    )
    query = _make_windows(
        query_names,
        query_values,
        window_size=window_size,
        stride=stride,
        embedding_bins=embedding_bins,
    )
    train_window_count = sum(
        max(0, (series.size - window_size) // stride + 1) for series in train_values
    )
    test_window_count = query.window_count - train_window_count

    base_indices = np.arange(query.window_count, dtype=np.int64)
    query_split_window = np.concatenate((base_indices, base_indices))
    query_is_reversed = np.concatenate(
        (
            np.zeros(query.window_count, dtype=bool),
            np.ones(query.window_count, dtype=bool),
        )
    )
    base_split = np.where(base_indices < train_window_count, "train", "test")
    query_split = np.concatenate((base_split, base_split))
    raw_oriented_embeddings = np.concatenate(
        (query.embeddings, query.embeddings[:, ::-1]), axis=0
    )
    metrics = (
        "raw_affine_pearson",
        "linear_detrended_pearson",
        "first_difference_pearson",
    )
    metric_results: dict[str, Any] = {}
    metric_search_diagnostics: dict[str, Any] = {}
    for metric_index, metric in enumerate(metrics):
        print(f"Starting metric: {metric}", file=sys.stderr, flush=True)
        external_metric_embeddings = _metric_embeddings(external.embeddings, metric)
        query_metric_embeddings = _metric_embeddings(raw_oriented_embeddings, metric)
        candidate_indices, search_diagnostics = _lsh_candidates(
            external_metric_embeddings,
            query_metric_embeddings,
            seed=seed + metric_index * 1_000_003,
            tables=lsh_tables,
            bits=lsh_bits,
            refine_per_query=refine_per_query,
            score_batch_size=score_batch_size,
        )
        metric_search_diagnostics[metric] = search_diagnostics
        metric_results[metric] = _exact_refinement(
            external,
            query,
            candidate_indices,
            query_is_reversed,
            query_split,
            query_split_window,
            metric=metric,
            window_size=window_size,
            threshold=threshold,
            batch_size=exact_batch_size,
        )

    manifest_files = external_paths.copy()
    coordinates_path = external_root / "example_dataset_coordinates.csv"
    if coordinates_path.is_file():
        manifest_files.append(coordinates_path)
    manifest = _manifest_sha256(external_root, manifest_files)
    resolved_commit = _git_revision(external_root)
    theoretical_collision_probability = 1.0 - math.acos(threshold) / math.pi
    theoretical_collision_probability = (
        1.0 - (1.0 - theoretical_collision_probability**lsh_bits) ** lsh_tables
    )

    raw_threshold_count = metric_results["raw_affine_pearson"][
        "screened_candidate_count_at_or_above_threshold"
    ]
    raw_high = raw_threshold_count > 0
    overlap = {
        "status": (
            "raw_high_correlation_requires_exclusion_from_model_use"
            if raw_high
            else "no_raw_candidate_at_threshold_but_screen_is_not_exhaustive"
        ),
        "interpretation": (
            "Raw affine Pearson contains high-correlation matches, so this screen "
            "cannot rule out source overlap. Linear detrending and first "
            "differences are stricter motif screens, but cannot negate the raw "
            "signal or establish independent provenance. ChronoLog is therefore "
            "excluded from this research round's model training and inference."
            if raw_high
            else "No raw match reached the configured threshold in retained LSH "
            "candidates, but approximate screening cannot prove independent "
            "provenance. Detrended and first-difference results are motif screens."
        ),
        "withdrawn_prior_claim": {
            "claim": (
                "max correlation 0.897327; visible-test max correlation 0.805860; "
                "no match above 0.90"
            ),
            "status": "withdrawn_due_to_unreproduced_preprocessing",
            "reason": (
                "The exploratory preprocessing that produced those figures was "
                "not preserved and a clean raw-window reproduction finds stronger "
                "matches. The prior figures must not be used as leakage clearance."
            ),
        },
        "roots": {
            "external_chronolog_root": str(external_root.resolve()),
            "rogii_data_root": str(data_root.resolve()),
        },
        "inputs_read": {
            "external_columns": ["gamma ray"],
            "rogii_typewell_columns": ["GR"],
            "rogii_target_tvt_read": False,
            "horizontal_files_read": False,
            "formation_surfaces_or_geology_read": False,
            "png_read": False,
        },
        "dataset_counts": {
            "external_logs": len(external_names),
            "external_samples": external.sample_count,
            "external_windows": external.window_count,
            "train_typewells": len(train_names),
            "train_typewell_samples": int(sum(series.size for series in train_values)),
            "train_windows": int(train_window_count),
            "visible_test_typewells": len(test_names),
            "visible_test_typewell_samples": int(
                sum(series.size for series in test_values)
            ),
            "visible_test_windows": int(test_window_count),
            "query_windows_total": query.window_count,
            "query_orientations_total": int(raw_oriented_embeddings.shape[0]),
            "invalid_external_windows": external.invalid_window_count,
            "invalid_query_windows": query.invalid_window_count,
        },
        "method": {
            "window_size_samples": window_size,
            "nominal_window_span_ft": window_size * 0.5,
            "stride_samples": stride,
            "base_embedding": (
                f"{embedding_bins} non-overlapping block means, per-window "
                "centered and L2-normalized"
            ),
            "metrics": {
                "raw_affine_pearson": (
                    "Pearson after per-window centering/scaling; sensitive to "
                    "broad shared trend and therefore the conservative source-"
                    "overlap screen"
                ),
                "linear_detrended_pearson": (
                    "Pearson after removing the least-squares linear trend from "
                    "each 256-sample window"
                ),
                "first_difference_pearson": (
                    "Pearson between the 255 first differences; a local-motif "
                    "screen that suppresses broad trend"
                ),
            },
            "reversal_aware": True,
            "candidate_generation": (
                "Independent random-hyperplane LSH exact-bucket search for each "
                "metric, followed by exact raw-sample transformation and Pearson"
            ),
            "lsh_base_seed": seed,
            "per_metric_seed_rule": (
                "base_seed + metric_index * 1,000,003 in the listed metric order"
            ),
            "lsh_tables": lsh_tables,
            "lsh_bits_per_table": lsh_bits,
            "refine_per_query_orientation": refine_per_query,
            "exact_refinement": "Metric-specific Pearson on original GR samples",
            "pearson_correlation_threshold": threshold,
            "threshold_count_unit": (
                "Retained oriented query-window/external-window candidate pairs; "
                "overlapping windows and identical GR segments shared by multiple "
                "ROGII wells are not deduplicated."
            ),
            "theoretical_at_least_one_bucket_collision_probability_at_threshold": (
                theoretical_collision_probability
            ),
            "determinism": (
                "Sorted input paths, NumPy PCG64 seed, stable hash ordering, and "
                "exact raw-window refinement"
            ),
            "scope_limit": (
                "The LSH probability is a random-hyperplane approximation and "
                "assumes the block-mean embedding has the threshold similarity; "
                "absence of a candidate is not exhaustive."
            ),
        },
        "search_diagnostics": metric_search_diagnostics,
        "results": metric_results,
        "external_snapshot": {
            "expected_git_commit": CHRONOLOG_COMMIT,
            "resolved_local_git_commit": resolved_commit,
            "revision_matches_expected": resolved_commit == CHRONOLOG_COMMIT,
            "data_manifest_sha256": manifest,
            "manifest_file_count": len(manifest_files),
        },
    }
    return {
        "schema_version": 1,
        "audit_id": "external-anchor-research-scouting-20260728",
        "research_date": AUDIT_DATE,
        "purpose": (
            "License/provenance/offline/fairness inventory plus a reproducible "
            "ChronoLog-vs-ROGII GR overlap screen; no external model is used for "
            "the reported competition predictions."
        ),
        "external_sources": _source_inventory(
            external_root,
            resolved_commit=resolved_commit,
            manifest_sha256=manifest,
            manifest_file_count=len(manifest_files),
        ),
        "chronolog_overlap_audit": overlap,
        "global_conclusion": {
            "external_assets_used_in_model_training_this_round": [],
            "external_assets_used_in_model_inference_this_round": [],
            "external_dataset_screened": ["chronolog_data"],
            "submission_created_or_uploaded": False,
            "fairness_conclusion": (
                "ChronoLog is excluded from this round because raw high-correlation "
                "matches and same-basin provenance cannot clear leakage risk. "
                "TTM/TS2Vec/Q-3D/N-BEATSx/S4/MTP/MOMENT were provenance-scoped "
                "only. TF-C released pretrained weights are excluded due to mixed "
                "upstream data-license risk."
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically screen public ChronoLog GR windows against ROGII "
            "train and visible-test typewell GR. The command reads inputs only, "
            "downloads nothing, and writes one JSON audit outside data/."
        )
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        required=True,
        help="ChronoLog repository root containing logs/*.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="ROGII competition data root containing train/ and test/",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--embedding-bins", type=int, default=DEFAULT_EMBEDDING_BINS)
    parser.add_argument("--lsh-tables", type=int, default=DEFAULT_LSH_TABLES)
    parser.add_argument("--lsh-bits", type=int, default=DEFAULT_LSH_BITS)
    parser.add_argument(
        "--refine-per-query",
        type=int,
        default=DEFAULT_REFINE_PER_QUERY,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=DEFAULT_OVERLAP_THRESHOLD,
    )
    parser.add_argument("--score-batch-size", type=int, default=250_000)
    parser.add_argument("--exact-batch-size", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    external_root = args.external_root.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if data_root == output or data_root in output.parents:
        raise ValueError("Refusing to write audit output inside read-only data/")
    if not (0.0 < args.overlap_threshold <= 1.0):
        raise ValueError("overlap_threshold must be in (0, 1]")

    audit = _build_audit(
        external_root=external_root,
        data_root=data_root,
        window_size=args.window_size,
        stride=args.stride,
        embedding_bins=args.embedding_bins,
        lsh_tables=args.lsh_tables,
        lsh_bits=args.lsh_bits,
        refine_per_query=args.refine_per_query,
        seed=args.seed,
        threshold=args.overlap_threshold,
        score_batch_size=args.score_batch_size,
        exact_batch_size=args.exact_batch_size,
    )
    audit["reproduction"] = {
        "command": (
            "python scripts/audit_external_gr_overlap.py "
            f'--external-root "{external_root}" '
            f'--data-root "{data_root}" '
            f'--output "{output}"'
        ),
        "network_access_required": False,
        "inputs_are_read_only": True,
        "only_written_path": str(output),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["chronolog_overlap_audit"]["results"], indent=2))
    print(f"Wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
