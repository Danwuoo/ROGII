"""Target-bearing SLP-E1 loop-closure feasibility diagnostic.

Only this diagnostic harness reads horizontal training labels. Candidate
generation, descriptors, consistency features, and confidence application are
implemented in :mod:`rogii.stratigraphic` and accept inference-time views.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import (
    HORIZONTAL_INFERENCE_COLUMNS,
    TYPEWELL_INFERENCE_COLUMNS,
    WellFiles,
    enumerate_wells,
    load_horizontal_inference,
    load_typewell_inference,
    validate_prefix,
)
from ..folds import FOLD_COUNT, load_fold_mapping, partition_well_ids, stable_select
from ..stratigraphic.confidence import (
    CALIBRATION_METHODS,
    Calibrator,
    edge_truth,
    fit_calibrator,
    reliability_bins,
    scored_edge_metrics,
    select_operating_point,
    unsupported_calibrator,
)
from ..stratigraphic.loop_edges import (
    EDGE_TYPES,
    EdgeGenerationConfig,
    EdgeGenerationResult,
    deterministic_shuffle_gr,
    generate_loop_edges,
)


SLP_E1_AUDIT_ID = "SLP-E1"
SLP_E1_USES_TARGET_FOR_EVALUATION = True
SLP_E1_USES_TARGET_FOR_ARTIFICIAL_CUT_CALIBRATION = True
SLP_E1_EDGE_GENERATOR_INFERENCE_SAFE = True
SLP_E1_DEPLOYABLE = False
SLP_E1_SUBMISSION_ELIGIBLE = False
DEFAULT_ARTIFICIAL_CUTS = (0.00, 0.25, 0.50)
DEFAULT_CALIBRATION_QUERY_ROWS_PER_CUT = 48
MAX_CALIBRATION_EDGES_PER_WELL_CUT_TYPE = 4096
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_QUERY_ID_FACTOR = 1_000_000
_METRIC_DTYPE = np.dtype(
    [
        ("confidence", "<f4"),
        ("error_ft", "<f4"),
        ("query_id", "<i8"),
        ("query_scale", "<f4"),
        ("well_index", "<i4"),
        ("strict_accept", "u1"),
        ("viable_accept", "u1"),
    ]
)


@dataclass(frozen=True)
class _FoldRoles:
    fold: int
    outer_training: tuple[str, ...]
    confidence_fit: tuple[str, ...]
    threshold_select: tuple[str, ...]
    outer_validation: tuple[str, ...]


@dataclass
class _RuntimeTotals:
    descriptor_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    calibration_seconds: float = 0.0
    scoring_seconds: float = 0.0
    peak_candidates: int = 0
    peak_index_rows: int = 0
    well_cut_runs: int = 0

    def add_generation(self, result: EdgeGenerationResult) -> None:
        self.descriptor_seconds += result.descriptor_seconds
        self.retrieval_seconds += result.retrieval_seconds
        self.peak_candidates = max(self.peak_candidates, result.peak_candidates)
        self.peak_index_rows = max(self.peak_index_rows, result.peak_index_rows)
        self.well_cut_runs += 1


@dataclass(frozen=True)
class _CachedTrainingCut:
    fraction: float
    cut_row: int
    known_count: int
    total_rows: int
    status: str
    skip_reason: str
    edges: pd.DataFrame | None
    all_future_rows: int
    gr_observed_future_rows: int
    descriptor_eligible_future_rows: int


class _DeterministicGzipCsvWriter:
    """Stream one atomically replaced deterministic gzip CSV."""

    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.path = Path(path)
        self.temporary = self.path.with_name(f".{self.path.name}.tmp")
        self.columns = tuple(columns)
        self.binary = self.temporary.open("wb")
        self.compressed = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=self.binary,
            mtime=0,
        )
        self.text = io.TextIOWrapper(
            self.compressed, encoding="utf-8", newline=""
        )
        self.wrote_header = False

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Streamed edge columns missing: {sorted(missing)}")
        frame.loc[:, list(self.columns)].to_csv(
            self.text,
            index=False,
            header=not self.wrote_header,
            lineterminator="\n",
        )
        self.wrote_header = True

    def close(self) -> None:
        if not self.wrote_header:
            pd.DataFrame(columns=self.columns).to_csv(
                self.text, index=False, lineterminator="\n"
            )
        self.text.flush()
        self.text.close()
        self.binary.close()
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        try:
            self.text.close()
        finally:
            self.binary.close()
        self.temporary.unlink(missing_ok=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            _json_ready(dict(payload)),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv_atomic(
    frame: pd.DataFrame, path: Path, *, gzip_compressed: bool = False
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if gzip_compressed:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with io.TextIOWrapper(
                    compressed, encoding="utf-8", newline=""
                ) as text:
                    frame.to_csv(text, index=False, lineterminator="\n")
    else:
        frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _safe_run_root(output_root: Path, run_id: str, data_root: Path) -> Path:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must use only letters, digits, dot, underscore, dash")
    data = Path(data_root).resolve()
    output = Path(output_root).resolve()
    try:
        output.relative_to(data)
    except ValueError:
        pass
    else:
        raise ValueError("SLP-E1 output root cannot be inside competition data")
    run_root = output / run_id
    if run_root.exists():
        raise FileExistsError(f"SLP-E1 run directory already exists: {run_root}")
    output.mkdir(parents=True, exist_ok=True)
    run_root.mkdir()
    (run_root / "cache" / "metrics").mkdir(parents=True)
    return run_root


def _load_target(path: Path) -> np.ndarray:
    """Read the horizontal label only inside the non-deployable harness."""
    header = pd.read_csv(path, nrows=0)
    if "TVT" not in header.columns:
        raise ValueError(f"SLP-E1 training label is missing: {path}")
    target = pd.read_csv(path, usecols=["TVT"])["TVT"].to_numpy(
        dtype=np.float64
    )
    if target.size == 0 or not np.isfinite(target).all():
        raise ValueError(f"SLP-E1 target must be finite: {path}")
    return target


def _stable_order_key(seed: int, salt: str, value: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}:{salt}:{value}".encode("utf-8")
    ).hexdigest()
    return digest, value


def _stable_select_rows(
    rows: Iterable[int], limit: int | None, *, seed: int, salt: str
) -> np.ndarray:
    values = tuple(sorted(set(map(int, rows))))
    if limit is None or limit >= len(values):
        return np.asarray(values, dtype=np.int64)
    if limit < 1:
        raise ValueError("row limit must be positive")
    selected = sorted(
        values,
        key=lambda value: _stable_order_key(seed, salt, str(value)),
    )[:limit]
    return np.asarray(sorted(selected), dtype=np.int64)


def _stable_sample_frame(
    frame: pd.DataFrame, limit: int, *, seed: int, salt: str
) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.reset_index(drop=True)
    keys = [
        _stable_order_key(
            seed,
            salt,
            f"{row.query_row}:{row.candidate_row}:{row.query_scale}:{row.candidate_scale}",
        )
        for row in frame.itertuples(index=False)
    ]
    order = sorted(range(len(frame)), key=lambda index: keys[index])[:limit]
    return frame.iloc[sorted(order)].reset_index(drop=True)


def _parse_folds(values: Iterable[int]) -> tuple[int, ...]:
    folds = tuple(sorted(set(map(int, values))))
    if not folds or any(fold not in range(FOLD_COUNT) for fold in folds):
        raise ValueError("folds must be a non-empty subset of 0..4")
    return folds


def _assign_roles(
    outer_training: tuple[str, ...], *, fold: int, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    confidence_count = int(math.floor(0.70 * len(outer_training)))
    confidence_count = min(max(confidence_count, 1), len(outer_training) - 1)
    confidence = stable_select(
        outer_training,
        confidence_count,
        seed=seed + fold,
        salt="SLP-E1-confidence-role-v1",
    )
    threshold = tuple(sorted(set(outer_training) - set(confidence)))
    if (
        not confidence
        or not threshold
        or set(confidence) & set(threshold)
        or set(confidence) | set(threshold) != set(outer_training)
    ):
        raise ValueError(f"Invalid SLP-E1 role split for fold {fold}")
    return confidence, threshold


def _artificial_cut_rows(
    known_count: int, total_rows: int, fractions: Sequence[float]
) -> tuple[tuple[float, int], ...]:
    suffix_count = total_rows - known_count
    return tuple(
        (
            float(fraction),
            known_count + int(math.floor(float(fraction) * suffix_count)),
        )
        for fraction in fractions
    )


def _cut_is_eligible(
    md: np.ndarray,
    cut: int,
    *,
    maximum_half_span: float,
) -> tuple[bool, str]:
    if cut < 2 or cut >= md.size - 1:
        return False, "cut_leaves_empty_prefix_or_future"
    if float(md[cut - 1] - md[0]) < 2.0 * maximum_half_span:
        return False, "prefix_shorter_than_maximum_window"
    if float(md[-1] - md[cut]) < 2.0 * maximum_half_span:
        return False, "future_shorter_than_maximum_window"
    return True, ""


def _synthetic_inference_view(
    horizontal: pd.DataFrame, target: np.ndarray, cut: int
) -> pd.DataFrame:
    synthetic = horizontal.loc[:, list(HORIZONTAL_INFERENCE_COLUMNS)].copy()
    synthetic["TVT_input"] = np.nan
    synthetic.loc[: cut - 1, "TVT_input"] = target[:cut]
    validate_prefix(synthetic["TVT_input"])
    return synthetic


def _edge_error(frame: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    query = frame["query_row"].to_numpy(dtype=np.int64)
    if np.any(query < 0) or np.any(query >= target.size):
        raise ValueError("SLP-E1 query rows lie outside target")
    edge_type = str(frame["edge_type"].iat[0]) if len(frame) else ""
    if edge_type in {"PF", "FF"}:
        candidate = frame["candidate_row"].to_numpy(dtype=np.int64)
        if np.any(candidate < 0) or np.any(candidate >= target.size):
            raise ValueError("SLP-E1 horizontal candidate rows lie outside target")
        error = np.abs(target[query] - target[candidate])
    elif edge_type == "HT":
        coordinate = frame["candidate_tvt"].to_numpy(dtype=np.float64)
        error = np.abs(target[query] - coordinate)
    else:
        raise ValueError(f"Unknown edge type for scoring: {edge_type}")
    if not np.isfinite(error).all():
        raise ValueError("SLP-E1 edge scorer produced non-finite errors")
    return error


def _label_edges(frame: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    labeled = frame.copy()
    error = _edge_error(labeled, target)
    labels = edge_truth(error)
    labeled["edge_error_ft"] = error
    for name, values in labels.items():
        labeled[name] = values
    return labeled


def _compact_cache_edges(frame: pd.DataFrame) -> pd.DataFrame:
    compact = frame.copy()
    for column in compact.select_dtypes(include=["float64"]).columns:
        compact[column] = compact[column].astype(np.float32)
    for column in (
        "query_row",
        "candidate_row",
        "edge_row_a",
        "edge_row_b",
    ):
        if column in compact:
            compact[column] = compact[column].astype(np.int32)
    for column in (
        "reciprocal_rank",
        "scale_vote_count",
        "shared_neighbor_count",
        "local_cycle_support_count",
    ):
        if column in compact:
            compact[column] = compact[column].astype(np.int16)
    for column in (
        "method",
        "edge_type",
        "query_well_id",
        "candidate_well_id",
    ):
        if column in compact:
            compact[column] = compact[column].astype("category")
    return compact


def _training_query_id(
    fold: int, role: str, well_id: str, fraction: float, row: int
) -> int:
    digest = hashlib.sha256(
        f"{fold}:{role}:{well_id}:{fraction:.6f}:{row}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _method_source(method: str) -> str:
    return method if method in {"raw_nn", "point_gr"} else "main"


def _build_training_cache_entry(
    well_id: str,
    *,
    well: WellFiles,
    fractions: Sequence[float],
    generation_config: EdgeGenerationConfig,
    seed: int,
    runtime: _RuntimeTotals,
    query_cap: int,
    workers: int,
) -> tuple[_CachedTrainingCut, ...]:
    horizontal = load_horizontal_inference(well.horizontal_path)
    typewell = load_typewell_inference(well.typewell_path)
    target = _load_target(well.horizontal_path)
    known_count, _ = validate_prefix(
        horizontal["TVT_input"],
        target=pd.Series(target),
    )
    md = horizontal["MD"].to_numpy(dtype=np.float64)
    maximum_span = max(generation_config.horizontal_half_spans)
    cut_specs = _artificial_cut_rows(known_count, len(horizontal), fractions)

    def build_one(
        specification: tuple[float, int],
    ) -> tuple[_CachedTrainingCut, EdgeGenerationResult | None]:
        fraction, cut = specification
        eligible, reason = _cut_is_eligible(
            md, cut, maximum_half_span=maximum_span
        )
        if not eligible:
            return (
                _CachedTrainingCut(
                    fraction=fraction,
                    cut_row=cut,
                    known_count=known_count,
                    total_rows=len(horizontal),
                    status="skipped",
                    skip_reason=reason,
                    edges=None,
                    all_future_rows=len(horizontal) - cut,
                    gr_observed_future_rows=0,
                    descriptor_eligible_future_rows=0,
                ),
                None,
            )
        synthetic = _synthetic_inference_view(horizontal, target, cut)
        query_rows = _stable_select_rows(
            range(cut, len(horizontal)),
            query_cap,
            seed=seed,
            salt=f"SLP-E1-calibration-query:{well_id}:{fraction:.2f}",
        )
        generated = generate_loop_edges(
            synthetic,
            typewell,
            well_id=well_id,
            cut_row=cut,
            query_rows=query_rows,
            config=generation_config,
            include_retrieval_controls=True,
        )
        labeled_parts = [
            _label_edges(
                generated.edges.loc[
                    generated.edges["edge_type"].eq(edge_type)
                ],
                target,
            )
            for edge_type in EDGE_TYPES
            if generated.edges["edge_type"].eq(edge_type).any()
        ]
        labeled = (
            pd.concat(labeled_parts, ignore_index=True)
            if labeled_parts
            else generated.edges.copy()
        )
        labeled = _compact_cache_edges(labeled)
        return (
            _CachedTrainingCut(
                fraction=fraction,
                cut_row=cut,
                known_count=known_count,
                total_rows=len(horizontal),
                status="used",
                skip_reason="",
                edges=labeled,
                all_future_rows=generated.all_future_rows,
                gr_observed_future_rows=generated.gr_observed_future_rows,
                descriptor_eligible_future_rows=(
                    generated.descriptor_eligible_future_rows
                ),
            ),
            generated,
        )
    if workers > 1 and len(cut_specs) > 1:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(cut_specs))
        ) as executor:
            built = list(executor.map(build_one, cut_specs))
    else:
        built = [build_one(specification) for specification in cut_specs]
    entries: list[_CachedTrainingCut] = []
    for entry, generated in built:
        entries.append(entry)
        if generated is not None:
            runtime.add_generation(generated)
    return tuple(entries)


def _collect_training_role(
    well_ids: Sequence[str],
    *,
    role: str,
    fold: int,
    wells_by_id: Mapping[str, WellFiles],
    fractions: Sequence[float],
    generation_config: EdgeGenerationConfig,
    seed: int,
    runtime: _RuntimeTotals,
    query_cap: int,
    candidate_cache: dict[str, tuple[_CachedTrainingCut, ...]],
    workers: int,
) -> tuple[
    dict[tuple[str, str], list[pd.DataFrame]],
    dict[str, int],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    collected: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    denominators = {
        "all_future_rows": 0,
        "gr_observed_future_rows": 0,
        "descriptor_eligible_future_rows": 0,
    }
    cut_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    for well_number, well_id in enumerate(well_ids, start=1):
        if well_id not in candidate_cache:
            candidate_cache[well_id] = _build_training_cache_entry(
                well_id,
                well=wells_by_id[well_id],
                fractions=fractions,
                generation_config=generation_config,
                seed=seed,
                runtime=runtime,
                query_cap=query_cap,
                workers=workers,
            )
        for entry in candidate_cache[well_id]:
            fraction = entry.fraction
            cut = entry.cut_row
            record = {
                "fold": fold,
                "role": role,
                "well_id": well_id,
                "fraction": fraction,
                "cut_row": cut,
                "known_count": entry.known_count,
                "total_rows": entry.total_rows,
                "future_rows": entry.total_rows - cut,
                "status": entry.status,
                "skip_reason": entry.skip_reason,
            }
            cut_records.append(record)
            if entry.status != "used" or entry.edges is None:
                continue
            denominators["all_future_rows"] += entry.all_future_rows
            denominators[
                "gr_observed_future_rows"
            ] += entry.gr_observed_future_rows
            denominators[
                "descriptor_eligible_future_rows"
            ] += entry.descriptor_eligible_future_rows
            for edge_type in EDGE_TYPES:
                for method in ("main", "raw_nn", "point_gr"):
                    source = method
                    subset = entry.edges.loc[
                        entry.edges["edge_type"].eq(edge_type)
                        & entry.edges["method"].eq(source)
                    ].copy()
                    before = len(subset)
                    if subset.empty:
                        sample_records.append(
                            {
                                "fold": fold,
                                "role": role,
                                "well_id": well_id,
                                "fraction": fraction,
                                "edge_type": edge_type,
                                "method": method,
                                "edges_before_sampling": 0,
                                "edges_after_sampling": 0,
                                "positive_prevalence_before": 0.0,
                                "positive_prevalence_after": 0.0,
                            }
                        )
                        continue
                    prevalence_before = float(subset["primary_positive"].mean())
                    subset = _stable_sample_frame(
                        subset,
                        MAX_CALIBRATION_EDGES_PER_WELL_CUT_TYPE,
                        seed=seed,
                        salt=(
                            f"SLP-E1-edge-sample:{fold}:{role}:{well_id}:"
                            f"{fraction:.2f}:{edge_type}:{method}"
                        ),
                    )
                    subset["query_id"] = [
                        _training_query_id(
                            fold, role, well_id, fraction, int(row)
                        )
                        for row in subset["query_row"]
                    ]
                    subset["calibration_method"] = method
                    collected[(method, edge_type)].append(subset)
                    sample_records.append(
                        {
                            "fold": fold,
                            "role": role,
                            "well_id": well_id,
                            "fraction": fraction,
                            "edge_type": edge_type,
                            "method": method,
                            "edges_before_sampling": before,
                            "edges_after_sampling": len(subset),
                            "positive_prevalence_before": prevalence_before,
                            "positive_prevalence_after": float(
                                subset["primary_positive"].mean()
                            ),
                        }
                    )
        if well_number % 25 == 0 or well_number == len(well_ids):
            print(
                f"SLP-E1 fold={fold} role={role} "
                f"processed={well_number}/{len(well_ids)} wells",
                file=sys.stderr,
                flush=True,
            )
    return collected, denominators, cut_records, sample_records


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


def _fit_fold_calibration(
    confidence_edges: Mapping[tuple[str, str], list[pd.DataFrame]],
    threshold_edges: Mapping[tuple[str, str], list[pd.DataFrame]],
    *,
    threshold_denominator: int,
    fold: int,
    seed: int,
) -> tuple[
    dict[tuple[str, str], Calibrator],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, Any],
]:
    calibrators: dict[tuple[str, str], Calibrator] = {}
    operating_points: dict[tuple[str, str, str], dict[str, Any]] = {}
    serialized: dict[str, Any] = {}
    for method in CALIBRATION_METHODS:
        serialized[method] = {}
        for edge_type in EDGE_TYPES:
            source = _method_source(method)
            fit_frame = _concat_frames(
                confidence_edges.get((source, edge_type), [])
            )
            if fit_frame.empty:
                calibrator = unsupported_calibrator(
                    method, "no_confidence_fit_candidates"
                )
            else:
                calibrator = fit_calibrator(
                    fit_frame,
                    fit_frame["primary_positive"].to_numpy(dtype=bool),
                    method=method,
                    random_state=seed + 1009 * fold,
                )
            calibrators[(method, edge_type)] = calibrator
            selection = _concat_frames(
                threshold_edges.get((source, edge_type), [])
            )
            serialized[method][edge_type] = {
                "calibrator": calibrator.to_dict(),
                "operating_points": {},
            }
            if calibrator.supported and not selection.empty:
                probability = calibrator.predict(selection)
                for name, target_precision in (
                    ("strict99", 0.99),
                    ("viable97", 0.97),
                ):
                    point = select_operating_point(
                        probability,
                        selection["primary_positive"].to_numpy(dtype=bool),
                        selection["catastrophic"].to_numpy(dtype=bool),
                        selection["query_id"].to_numpy(dtype=np.int64),
                        selection["query_well_id"].astype(str).to_numpy(),
                        target_precision=target_precision,
                        total_future_rows=threshold_denominator,
                    )
                    operating_points[(method, edge_type, name)] = point
                    serialized[method][edge_type]["operating_points"][name] = point
            else:
                for name, target_precision in (
                    ("strict99", 0.99),
                    ("viable97", 0.97),
                ):
                    point = select_operating_point(
                        (),
                        (),
                        (),
                        (),
                        (),
                        target_precision=target_precision,
                        total_future_rows=max(1, threshold_denominator),
                    )
                    operating_points[(method, edge_type, name)] = point
                    serialized[method][edge_type]["operating_points"][name] = point
    return calibrators, operating_points, serialized


def _threshold_for(
    points: Mapping[tuple[str, str, str], Mapping[str, Any]],
    method: str,
    edge_type: str,
    operating_point: str,
) -> float:
    point = points.get((method, edge_type, operating_point))
    return float(point["threshold"]) if point else 1.0


def _metric_path(
    run_root: Path, fold: int, method: str, edge_type: str
) -> Path:
    safe_method = method.replace("/", "_")
    return (
        run_root
        / "cache"
        / "metrics"
        / f"fold_{fold}__{safe_method}__{edge_type}.bin"
    )


def _append_metric_records(
    path: Path,
    *,
    confidence: np.ndarray,
    error_ft: np.ndarray,
    query_id: np.ndarray,
    query_scale: np.ndarray,
    well_index: int,
    strict_threshold: float,
    viable_threshold: float,
) -> None:
    records = np.empty(confidence.size, dtype=_METRIC_DTYPE)
    records["confidence"] = confidence.astype(np.float32)
    records["error_ft"] = error_ft.astype(np.float32)
    records["query_id"] = query_id.astype(np.int64)
    records["query_scale"] = query_scale.astype(np.float32)
    records["well_index"] = np.int32(well_index)
    records["strict_accept"] = (confidence >= strict_threshold).astype(np.uint8)
    records["viable_accept"] = (confidence >= viable_threshold).astype(np.uint8)
    with path.open("ab") as output:
        records.tofile(output)


def _read_metric_records(paths: Iterable[Path]) -> np.ndarray:
    arrays = [
        np.fromfile(path, dtype=_METRIC_DTYPE)
        for path in paths
        if path.is_file() and path.stat().st_size
    ]
    if not arrays:
        return np.empty((0,), dtype=_METRIC_DTYPE)
    return np.concatenate(arrays)


def _records_metrics(
    records: np.ndarray,
    *,
    operating_point: str,
    all_future_rows: int,
    gr_observed_future_rows: int,
    descriptor_eligible_future_rows: int,
) -> dict[str, Any]:
    accept_field = (
        "strict_accept" if operating_point == "strict99" else "viable_accept"
    )
    return scored_edge_metrics(
        records["confidence"],
        records["error_ft"],
        records["query_id"],
        threshold=0.5,
        total_future_rows=all_future_rows,
        gr_observed_future_rows=gr_observed_future_rows,
        descriptor_eligible_future_rows=descriptor_eligible_future_rows,
        accepted_mask=records[accept_field].astype(bool),
    )


def _select_persisted_edges(
    main_frames: Sequence[pd.DataFrame],
    *,
    fold: int,
    well_id: str,
    seed: int,
) -> pd.DataFrame:
    if not main_frames:
        return pd.DataFrame()
    combined = pd.concat(main_frames, ignore_index=True)
    if combined.empty:
        return combined
    combined["_best"] = False
    best_indices = (
        combined.sort_values(
            ["query_row", "calibrated_confidence", "edge_type", "candidate_row"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        .groupby("query_row", sort=False)
        .head(1)
        .index
    )
    combined.loc[best_indices, "_best"] = True
    accepted = combined["accepted_strict99"] | combined["accepted_viable97"]
    rejected_pool = combined.loc[~accepted & ~combined["_best"]]
    rejected_keys = {
        int(index): _stable_order_key(
            seed,
            f"SLP-E1-rejected-sample:{fold}:{well_id}",
            (
                f"{row.query_row}:{row.candidate_row}:"
                f"{row.query_scale}:{row.candidate_scale}"
            ),
        )
        for index, row in zip(
            rejected_pool.index,
            rejected_pool.itertuples(index=False),
            strict=True,
        )
    }
    rejected_indices = sorted(
        rejected_pool.index,
        key=lambda index: rejected_keys[int(index)],
    )[: min(64, len(rejected_pool))]
    selected_index = set(combined.index[accepted | combined["_best"]])
    selected_index.update(map(int, rejected_indices))
    selected = combined.loc[sorted(selected_index)].copy()
    selected["record_reason"] = np.where(
        selected["_best"],
        "best_query",
        np.where(
            selected["accepted_viable97"] | selected["accepted_strict99"],
            "accepted",
            "rejected_sample",
        ),
    )
    selected.loc[
        selected["_best"]
        & (selected["accepted_viable97"] | selected["accepted_strict99"]),
        "record_reason",
    ] = "best_query+accepted"
    selected["fold"] = fold
    return selected.drop(columns=["_best"]).sort_values(
        ["fold", "query_well_id", "query_row", "edge_type", "candidate_row"],
        kind="stable",
    )


def _score_validation_fold(
    roles: _FoldRoles,
    *,
    wells_by_id: Mapping[str, WellFiles],
    all_well_order: Mapping[str, int],
    calibrators: Mapping[tuple[str, str], Calibrator],
    operating_points: Mapping[tuple[str, str, str], Mapping[str, Any]],
    generation_config: EdgeGenerationConfig,
    query_cap: int | None,
    seed: int,
    run_root: Path,
    edge_writer: _DeterministicGzipCsvWriter,
    runtime: _RuntimeTotals,
    workers: int,
) -> tuple[
    dict[str, int],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    fold_denominators = {
        "all_future_rows": 0,
        "gr_observed_future_rows": 0,
        "descriptor_eligible_future_rows": 0,
    }
    per_well_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    validation_ids = roles.outer_validation
    wrong_typewell = {
        well_id: validation_ids[(index + 1) % len(validation_ids)]
        for index, well_id in enumerate(validation_ids)
    }
    fold_method_records: dict[tuple[str, str], list[Path]] = defaultdict(list)
    del fold_method_records
    for well_number, well_id in enumerate(validation_ids, start=1):
        scoring_started = time.perf_counter()
        well = wells_by_id[well_id]
        horizontal = load_horizontal_inference(well.horizontal_path)
        typewell = load_typewell_inference(well.typewell_path)
        cut, _ = validate_prefix(horizontal["TVT_input"])
        rows = _stable_select_rows(
            range(cut, len(horizontal)),
            query_cap,
            seed=seed,
            salt=f"SLP-E1-validation-query:{roles.fold}:{well_id}",
        )
        if query_cap is None:
            query_argument: np.ndarray | None = None
        else:
            query_argument = rows
        shuffled_horizontal = horizontal.copy()
        shuffled_horizontal["GR"] = deterministic_shuffle_gr(
            horizontal["GR"], well_id=well_id, seed=seed
        )
        wrong_id = wrong_typewell[well_id]
        wrong_type = load_typewell_inference(
            wells_by_id[wrong_id].typewell_path
        )
        generation_calls = (
            lambda: generate_loop_edges(
                horizontal,
                typewell,
                well_id=well_id,
                cut_row=cut,
                query_rows=query_argument,
                config=generation_config,
                include_retrieval_controls=True,
            ),
            lambda: generate_loop_edges(
                shuffled_horizontal,
                typewell,
                well_id=well_id,
                cut_row=cut,
                query_rows=query_argument,
                config=generation_config,
                include_retrieval_controls=False,
            ),
            lambda: generate_loop_edges(
                horizontal,
                wrong_type,
                well_id=well_id,
                cut_row=cut,
                query_rows=query_argument,
                edge_types=("HT",),
                config=generation_config,
                include_retrieval_controls=False,
                candidate_typewell_well_id=wrong_id,
            ),
        )
        if workers > 1:
            with ThreadPoolExecutor(
                max_workers=min(workers, len(generation_calls))
            ) as executor:
                generated, shuffled, wrong = tuple(
                    executor.map(lambda call: call(), generation_calls)
                )
        else:
            generated, shuffled, wrong = tuple(
                call() for call in generation_calls
            )
        runtime.add_generation(generated)
        runtime.add_generation(shuffled)
        runtime.add_generation(wrong)

        fold_denominators["all_future_rows"] += generated.all_future_rows
        fold_denominators[
            "gr_observed_future_rows"
        ] += generated.gr_observed_future_rows
        fold_denominators[
            "descriptor_eligible_future_rows"
        ] += generated.descriptor_eligible_future_rows

        # Freeze confidence and accept/reject decisions before loading labels.
        frozen: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray, float, float]] = {}
        for edge_type in EDGE_TYPES:
            for method in CALIBRATION_METHODS:
                source = _method_source(method)
                frame = generated.edges.loc[
                    generated.edges["edge_type"].eq(edge_type)
                    & generated.edges["method"].eq(source)
                ].copy()
                calibrator = calibrators[(method, edge_type)]
                if not calibrator.supported:
                    frame = frame.iloc[0:0].copy()
                    probability = np.empty((0,), dtype=np.float64)
                else:
                    probability = (
                        calibrator.predict(frame)
                        if not frame.empty
                        else np.empty((0,), dtype=np.float64)
                    )
                frozen[(method, edge_type)] = (
                    frame,
                    probability,
                    _threshold_for(
                        operating_points, method, edge_type, "strict99"
                    ),
                    _threshold_for(
                        operating_points, method, edge_type, "viable97"
                    ),
                )
            shuffle_frame = shuffled.edges.loc[
                shuffled.edges["edge_type"].eq(edge_type)
                & shuffled.edges["method"].eq("main")
            ].copy()
            main_calibrator = calibrators[("main", edge_type)]
            if not main_calibrator.supported:
                shuffle_frame = shuffle_frame.iloc[0:0].copy()
                shuffle_probability = np.empty((0,), dtype=np.float64)
            else:
                shuffle_probability = (
                    main_calibrator.predict(shuffle_frame)
                    if not shuffle_frame.empty
                    else np.empty((0,), dtype=np.float64)
                )
            frozen[("shuffle_gr", edge_type)] = (
                shuffle_frame,
                shuffle_probability,
                _threshold_for(
                    operating_points, "main", edge_type, "strict99"
                ),
                _threshold_for(
                    operating_points, "main", edge_type, "viable97"
                ),
            )
        wrong_frame = wrong.edges.loc[
            wrong.edges["edge_type"].eq("HT")
            & wrong.edges["method"].eq("main")
        ].copy()
        ht_calibrator = calibrators[("main", "HT")]
        if not ht_calibrator.supported:
            wrong_frame = wrong_frame.iloc[0:0].copy()
            wrong_probability = np.empty((0,), dtype=np.float64)
        else:
            wrong_probability = (
                ht_calibrator.predict(wrong_frame)
                if not wrong_frame.empty
                else np.empty((0,), dtype=np.float64)
            )
        frozen[("wrong_typewell", "HT")] = (
            wrong_frame,
            wrong_probability,
            _threshold_for(operating_points, "main", "HT", "strict99"),
            _threshold_for(operating_points, "main", "HT", "viable97"),
        )

        # This is the first validation-label read for this well.
        target = _load_target(well.horizontal_path)
        if target.size != len(horizontal):
            raise ValueError(f"SLP-E1 target length mismatch: {well_id}")
        main_for_persistence: list[pd.DataFrame] = []
        for (method, edge_type), (
            frame,
            probability,
            strict_threshold,
            viable_threshold,
        ) in frozen.items():
            if frame.empty:
                for operating_point in ("strict99", "viable97"):
                    per_well_rows.append(
                        {
                            "fold": roles.fold,
                            "well_id": well_id,
                            "method": method,
                            "edge_type": edge_type,
                            "operating_point": operating_point,
                            "proposed_edges": 0,
                            "accepted_edges": 0,
                            "edge_precision": 0.0,
                            "all_future_rows": generated.all_future_rows,
                            "all_future_coverage": 0.0,
                            "catastrophic_false_edge_rate": 0.0,
                            "zero_support": True,
                        }
                    )
                continue
            labeled = _label_edges(frame, target)
            query_id = (
                np.int64(all_well_order[well_id]) * _QUERY_ID_FACTOR
                + labeled["query_row"].to_numpy(dtype=np.int64)
            )
            path = _metric_path(run_root, roles.fold, method, edge_type)
            _append_metric_records(
                path,
                confidence=probability,
                error_ft=labeled["edge_error_ft"].to_numpy(dtype=np.float64),
                query_id=query_id,
                query_scale=labeled["query_scale"].to_numpy(dtype=np.float64),
                well_index=all_well_order[well_id],
                strict_threshold=strict_threshold,
                viable_threshold=viable_threshold,
            )
            for operating_point, threshold in (
                ("strict99", strict_threshold),
                ("viable97", viable_threshold),
            ):
                metric = scored_edge_metrics(
                    probability,
                    labeled["edge_error_ft"].to_numpy(dtype=np.float64),
                    query_id,
                    threshold=threshold,
                    total_future_rows=generated.all_future_rows,
                    gr_observed_future_rows=generated.gr_observed_future_rows,
                    descriptor_eligible_future_rows=(
                        generated.descriptor_eligible_future_rows
                    ),
                )
                per_well_rows.append(
                    {
                        "fold": roles.fold,
                        "well_id": well_id,
                        "method": method,
                        "edge_type": edge_type,
                        "operating_point": operating_point,
                        **metric,
                        "zero_support": metric["accepted_edges"] == 0,
                    }
                )
            if method == "main":
                labeled["calibrated_confidence"] = probability
                labeled["accepted_strict99"] = probability >= strict_threshold
                labeled["accepted_viable97"] = probability >= viable_threshold
                main_for_persistence.append(labeled)
        persisted = _select_persisted_edges(
            main_for_persistence,
            fold=roles.fold,
            well_id=well_id,
            seed=seed,
        )
        edge_writer.write(persisted)
        runtime.scoring_seconds += time.perf_counter() - scoring_started
        if well_number % 25 == 0 or well_number == len(validation_ids):
            print(
                f"SLP-E1 fold={roles.fold} validation "
                f"processed={well_number}/{len(validation_ids)} wells",
                file=sys.stderr,
                flush=True,
            )

    all_methods = (
        "main",
        "raw_nn",
        "point_gr",
        "no_cycle",
        "destroy_order",
        "shuffle_gr",
        "wrong_typewell",
    )
    for method in all_methods:
        for edge_type in EDGE_TYPES:
            path = _metric_path(run_root, roles.fold, method, edge_type)
            records = _read_metric_records((path,))
            if records.size == 0 and method == "wrong_typewell" and edge_type != "HT":
                continue
            if records.size == 0 and method not in {"wrong_typewell"}:
                # Keep explicit zero rows for unsupported types.
                pass
            for operating_point in ("strict99", "viable97"):
                metric = _records_metrics(
                    records,
                    operating_point=operating_point,
                    all_future_rows=fold_denominators["all_future_rows"],
                    gr_observed_future_rows=fold_denominators[
                        "gr_observed_future_rows"
                    ],
                    descriptor_eligible_future_rows=fold_denominators[
                        "descriptor_eligible_future_rows"
                    ],
                )
                fold_metric_rows.append(
                    {
                        "scope": "fold",
                        "fold": roles.fold,
                        "method": method,
                        "edge_type": edge_type,
                        "operating_point": operating_point,
                        **metric,
                    }
                )
            labels = edge_truth(records["error_ft"])["primary_positive"]
            bins, brier, ece = reliability_bins(records["confidence"], labels)
            for row in bins.to_dict("records"):
                calibration_rows.append(
                    {
                        "scope": "fold",
                        "fold": roles.fold,
                        "method": method,
                        "edge_type": edge_type,
                        **row,
                        "brier_score": brier,
                        "ece_10bin": ece,
                    }
                )
    return (
        fold_denominators,
        per_well_rows,
        fold_metric_rows,
        calibration_rows,
    )


def _recurrence_fraction(
    target: np.ndarray, known_count: int
) -> float:
    prefix = np.sort(target[:known_count], kind="stable")
    future = target[known_count:]
    positions = np.searchsorted(prefix, future, side="left")
    left = prefix[np.maximum(positions - 1, 0)]
    right = prefix[np.minimum(positions, prefix.size - 1)]
    nearest = np.minimum(np.abs(future - left), np.abs(future - right))
    return float(np.mean(nearest <= 1.0))


def _low_recurrence_cohort(
    validation_wells: Sequence[str],
    *,
    wells_by_id: Mapping[str, WellFiles],
) -> tuple[tuple[str, ...], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for well_id in validation_wells:
        well = wells_by_id[well_id]
        horizontal = load_horizontal_inference(well.horizontal_path)
        known_count, future_count = validate_prefix(horizontal["TVT_input"])
        target = _load_target(well.horizontal_path)
        rows.append(
            {
                "well_id": well_id,
                "future_rows": future_count,
                "prefix_recurrence_le_1ft_fraction": _recurrence_fraction(
                    target, known_count
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["prefix_recurrence_le_1ft_fraction", "well_id"], kind="stable"
    )
    count = max(1, int(math.ceil(0.10 * len(frame))))
    cohort = tuple(sorted(frame.head(count)["well_id"].astype(str)))
    return cohort, frame.sort_values("well_id", kind="stable")


def _decision(
    strict_metrics: Mapping[str, Any],
    viable_metrics: Mapping[str, Any],
    *,
    thresholds_supported: bool,
) -> str:
    catastrophic = max(
        float(strict_metrics["catastrophic_false_edge_rate"]),
        float(viable_metrics["catastrophic_false_edge_rate"]),
    )
    if catastrophic > 0.005:
        return "unsafe"
    if (
        viable_metrics["accepted_edges"] > 0
        and viable_metrics["edge_precision"] < 0.97
    ):
        return "unsafe"
    if not thresholds_supported:
        return "unsupported"
    if (
        strict_metrics["edge_precision"] >= 0.99
        and strict_metrics["all_future_coverage"] >= 0.10
    ):
        return "strong"
    if (
        viable_metrics["edge_precision"] >= 0.97
        and viable_metrics["all_future_coverage"] >= 0.30
    ):
        return "viable"
    return "unsupported"


def _aggregate_scale_metrics(
    records: np.ndarray,
    *,
    method: str,
    edge_type: str,
    operating_point: str,
) -> list[dict[str, Any]]:
    accepted_field = (
        "strict_accept" if operating_point == "strict99" else "viable_accept"
    )
    rows: list[dict[str, Any]] = []
    for scale in sorted(set(records["query_scale"].tolist())):
        subset = records[records["query_scale"] == scale]
        accepted = subset[accepted_field].astype(bool)
        error = subset["error_ft"][accepted]
        rows.append(
            {
                "scope": "overall_scale",
                "fold": "",
                "method": method,
                "edge_type": edge_type,
                "operating_point": operating_point,
                "query_scale": float(scale),
                "proposed_edges": int(subset.size),
                "accepted_edges": int(accepted.sum()),
                "error_median": (
                    float(np.median(error)) if error.size else 0.0
                ),
                "error_q90": (
                    float(np.quantile(error, 0.90)) if error.size else 0.0
                ),
                "error_q99": (
                    float(np.quantile(error, 0.99)) if error.size else 0.0
                ),
            }
        )
    return rows


def run_slp_e1_diagnostic(
    *,
    profile: str,
    data_root: Path,
    fold_config: Path,
    output_root: Path,
    run_id: str,
    folds: Iterable[int] | None = None,
    max_train_wells: int | None = None,
    max_validation_wells: int | None = None,
    seed: int = 20260725,
    workers: int = 4,
    max_query_rows_per_well: int | None = None,
    top_k: int = 3,
    artificial_cuts: Sequence[float] = DEFAULT_ARTIFICIAL_CUTS,
    descriptor_set: str = "handcrafted_v1",
) -> dict[str, Any]:
    """Run fold-safe SLP-E1 calibration, OOF scoring, controls, and reporting."""
    started = time.perf_counter()
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    if workers < 1:
        raise ValueError("workers must be positive")
    if descriptor_set != "handcrafted_v1":
        raise ValueError("SLP-E1 V1 supports descriptor_set=handcrafted_v1")
    fractions = tuple(map(float, artificial_cuts))
    if not fractions or any(value < 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("artificial cuts must lie in [0, 1)")
    if tuple(sorted(set(fractions))) != fractions:
        raise ValueError("artificial cuts must be unique and sorted")
    requested_folds = _parse_folds(
        folds
        if folds is not None
        else ((0,) if profile == "quick" else range(FOLD_COUNT))
    )
    train_limit = (
        max_train_wells
        if max_train_wells is not None
        else (100 if profile == "quick" else None)
    )
    validation_limit = (
        max_validation_wells
        if max_validation_wells is not None
        else (20 if profile == "quick" else None)
    )
    validation_query_cap = (
        max_query_rows_per_well
        if max_query_rows_per_well is not None
        else (512 if profile == "quick" else None)
    )
    formal_full = (
        profile == "full"
        and requested_folds == tuple(range(FOLD_COUNT))
        and train_limit is None
        and validation_limit is None
        and validation_query_cap is None
    )
    scope = "full" if formal_full else "quick_or_custom"

    data = Path(data_root).resolve()
    train_wells = enumerate_wells(data, "train")
    wells_by_id = {well.well_id: well for well in train_wells}
    mapping = load_fold_mapping(
        fold_config, expected_well_ids=wells_by_id.keys()
    )
    fold_version = str(mapping["fold_version"].iat[0])
    roles_by_fold: dict[int, _FoldRoles] = {}
    for fold in requested_folds:
        training_pool, validation_pool = partition_well_ids(mapping, fold)
        outer_training = stable_select(
            training_pool,
            train_limit,
            seed=seed,
            salt=f"SLP-E1-fold-{fold}-outer-training",
        )
        outer_validation = stable_select(
            validation_pool,
            validation_limit,
            seed=seed,
            salt=f"SLP-E1-fold-{fold}-outer-validation",
        )
        confidence, threshold = _assign_roles(
            outer_training, fold=fold, seed=seed
        )
        roles_by_fold[fold] = _FoldRoles(
            fold=fold,
            outer_training=outer_training,
            confidence_fit=confidence,
            threshold_select=threshold,
            outer_validation=outer_validation,
        )
    all_validation = tuple(
        well_id
        for fold in requested_folds
        for well_id in roles_by_fold[fold].outer_validation
    )
    if len(all_validation) != len(set(all_validation)):
        raise ValueError("Outer-validation wells are duplicated across requested folds")
    all_well_order = {
        well_id: index for index, well_id in enumerate(sorted(all_validation))
    }

    run_root = _safe_run_root(output_root, run_id, data)
    generation_config = EdgeGenerationConfig(top_k=top_k)
    runtime = _RuntimeTotals()
    fold_manifest: dict[str, Any] = {}
    thresholds_payload: dict[str, Any] = {}
    threshold_sample_rows: list[dict[str, Any]] = []
    operating_point_rows: list[dict[str, Any]] = []
    per_well_rows: list[dict[str, Any]] = []
    edge_metric_rows: list[dict[str, Any]] = []
    calibration_bin_rows: list[dict[str, Any]] = []
    fold_denominators: dict[int, dict[str, int]] = {}
    all_cut_records: list[dict[str, Any]] = []
    training_candidate_cache: dict[
        str, tuple[_CachedTrainingCut, ...]
    ] = {}
    edge_columns: tuple[str, ...] | None = None
    edge_writer: _DeterministicGzipCsvWriter | None = None
    try:
        for fold in requested_folds:
            roles = roles_by_fold[fold]
            calibration_started = time.perf_counter()
            confidence_edges, confidence_denominator, confidence_cuts, confidence_samples = (
                _collect_training_role(
                    roles.confidence_fit,
                    role="confidence_fit",
                    fold=fold,
                    wells_by_id=wells_by_id,
                    fractions=fractions,
                    generation_config=generation_config,
                    seed=seed,
                    runtime=runtime,
                    query_cap=DEFAULT_CALIBRATION_QUERY_ROWS_PER_CUT,
                    candidate_cache=training_candidate_cache,
                    workers=workers,
                )
            )
            threshold_edges, threshold_denominator, threshold_cuts, threshold_samples = (
                _collect_training_role(
                    roles.threshold_select,
                    role="threshold_select",
                    fold=fold,
                    wells_by_id=wells_by_id,
                    fractions=fractions,
                    generation_config=generation_config,
                    seed=seed,
                    runtime=runtime,
                    query_cap=DEFAULT_CALIBRATION_QUERY_ROWS_PER_CUT,
                    candidate_cache=training_candidate_cache,
                    workers=workers,
                )
            )
            calibrators, operating_points, fold_thresholds = _fit_fold_calibration(
                confidence_edges,
                threshold_edges,
                threshold_denominator=threshold_denominator["all_future_rows"],
                fold=fold,
                seed=seed,
            )
            runtime.calibration_seconds += time.perf_counter() - calibration_started
            thresholds_payload[str(fold)] = fold_thresholds
            all_cut_records.extend(confidence_cuts)
            all_cut_records.extend(threshold_cuts)
            threshold_sample_rows.extend(confidence_samples)
            threshold_sample_rows.extend(threshold_samples)
            for (method, edge_type, name), point in operating_points.items():
                operating_point_rows.append(
                    {
                        "scope": "threshold_select",
                        "fold": fold,
                        "method": method,
                        "edge_type": edge_type,
                        "operating_point": name,
                        **point,
                    }
                )

            fold_manifest[str(fold)] = {
                "outer_training_wells": list(roles.outer_training),
                "confidence_fit_wells": list(roles.confidence_fit),
                "threshold_select_wells": list(roles.threshold_select),
                "outer_validation_wells": list(roles.outer_validation),
                "outer_training_count": len(roles.outer_training),
                "confidence_fit_count": len(roles.confidence_fit),
                "threshold_select_count": len(roles.threshold_select),
                "outer_validation_count": len(roles.outer_validation),
                "artificial_cuts_used": sum(
                    row["fold"] == fold and row["status"] == "used"
                    for row in confidence_cuts + threshold_cuts
                ),
                "artificial_cuts_skipped": sum(
                    row["fold"] == fold and row["status"] == "skipped"
                    for row in confidence_cuts + threshold_cuts
                ),
                "confidence_fit_denominators": confidence_denominator,
                "threshold_select_denominators": threshold_denominator,
                "candidate_edges": {
                    f"{method}:{edge_type}": int(
                        sum(
                            len(frame)
                            for frame in confidence_edges.get(
                                (_method_source(method), edge_type), []
                            )
                        )
                        + sum(
                            len(frame)
                            for frame in threshold_edges.get(
                                (_method_source(method), edge_type), []
                            )
                        )
                    )
                    for method in CALIBRATION_METHODS
                    for edge_type in EDGE_TYPES
                },
            }

            # Learn the persisted edge schema from an inference-safe empty run result.
            if edge_writer is None:
                sample_columns: list[str] | None = None
                for frames in confidence_edges.values():
                    if frames:
                        sample_columns = list(frames[0].columns)
                        break
                if sample_columns is None:
                    raise ValueError("SLP-E1 produced no training candidates")
                evaluation_columns = [
                    "edge_error_ft",
                    "strict_positive",
                    "primary_positive",
                    "loose_positive",
                    "catastrophic",
                    "calibrated_confidence",
                    "accepted_strict99",
                    "accepted_viable97",
                    "record_reason",
                    "fold",
                ]
                excluded_training_only = {
                    "query_id",
                    "calibration_method",
                }
                edge_columns = tuple(
                    column
                    for column in sample_columns
                    if column not in excluded_training_only
                    and column
                    not in {
                        "edge_error_ft",
                        "strict_positive",
                        "primary_positive",
                        "loose_positive",
                        "catastrophic",
                    }
                ) + tuple(evaluation_columns)
                edge_writer = _DeterministicGzipCsvWriter(
                    run_root / "edges_oof.csv.gz", edge_columns
                )

            (
                validation_denominator,
                fold_per_well,
                fold_metrics,
                fold_bins,
            ) = _score_validation_fold(
                roles,
                wells_by_id=wells_by_id,
                all_well_order=all_well_order,
                calibrators=calibrators,
                operating_points=operating_points,
                generation_config=generation_config,
                query_cap=validation_query_cap,
                seed=seed,
                run_root=run_root,
                edge_writer=edge_writer,
                runtime=runtime,
                workers=workers,
            )
            fold_denominators[fold] = validation_denominator
            per_well_rows.extend(fold_per_well)
            edge_metric_rows.extend(fold_metrics)
            calibration_bin_rows.extend(fold_bins)
        if edge_writer is None:
            raise ValueError("SLP-E1 could not initialize OOF edge writer")
        edge_writer.close()
        edge_writer = None
    except Exception:
        if edge_writer is not None:
            edge_writer.abort()
        raise

    overall_denominator = {
        key: int(sum(values[key] for values in fold_denominators.values()))
        for key in (
            "all_future_rows",
            "gr_observed_future_rows",
            "descriptor_eligible_future_rows",
        )
    }
    all_methods = (
        "main",
        "raw_nn",
        "point_gr",
        "no_cycle",
        "destroy_order",
        "shuffle_gr",
        "wrong_typewell",
    )
    overall_records: dict[tuple[str, str], np.ndarray] = {}
    for method in all_methods:
        for edge_type in EDGE_TYPES:
            if method == "wrong_typewell" and edge_type != "HT":
                continue
            paths = [
                _metric_path(run_root, fold, method, edge_type)
                for fold in requested_folds
            ]
            records = _read_metric_records(paths)
            if method == "main":
                overall_records[(method, edge_type)] = records
            for operating_point in ("strict99", "viable97"):
                metric = _records_metrics(
                    records,
                    operating_point=operating_point,
                    all_future_rows=overall_denominator["all_future_rows"],
                    gr_observed_future_rows=overall_denominator[
                        "gr_observed_future_rows"
                    ],
                    descriptor_eligible_future_rows=overall_denominator[
                        "descriptor_eligible_future_rows"
                    ],
                )
                edge_metric_rows.append(
                    {
                        "scope": "overall",
                        "fold": "",
                        "method": method,
                        "edge_type": edge_type,
                        "operating_point": operating_point,
                        **metric,
                    }
                )
                edge_metric_rows.extend(
                    _aggregate_scale_metrics(
                        records,
                        method=method,
                        edge_type=edge_type,
                        operating_point=operating_point,
                    )
                )
            labels = edge_truth(records["error_ft"])["primary_positive"]
            bins, brier, ece = reliability_bins(records["confidence"], labels)
            for row in bins.to_dict("records"):
                calibration_bin_rows.append(
                    {
                        "scope": "overall",
                        "fold": "",
                        "method": method,
                        "edge_type": edge_type,
                        **row,
                        "brier_score": brier,
                        "ece_10bin": ece,
                    }
                )

    combined_records: dict[str, np.ndarray] = {}
    combined_records["same_well"] = np.concatenate(
        [
            overall_records.get(
                ("main", edge_type), np.empty((0,), dtype=_METRIC_DTYPE)
            )
            for edge_type in ("PF", "FF")
        ]
    )
    combined_records["overall"] = np.concatenate(
        [
            overall_records.get(
                ("main", edge_type), np.empty((0,), dtype=_METRIC_DTYPE)
            )
            for edge_type in EDGE_TYPES
        ]
    )
    combined_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for name, records in combined_records.items():
        combined_metrics[name] = {}
        for operating_point in ("strict99", "viable97"):
            combined_metrics[name][operating_point] = _records_metrics(
                records,
                operating_point=operating_point,
                all_future_rows=overall_denominator["all_future_rows"],
                gr_observed_future_rows=overall_denominator[
                    "gr_observed_future_rows"
                ],
                descriptor_eligible_future_rows=overall_denominator[
                    "descriptor_eligible_future_rows"
                ],
            )
            edge_metric_rows.append(
                {
                    "scope": "overall",
                    "fold": "",
                    "method": "main",
                    "edge_type": name,
                    "operating_point": operating_point,
                    **combined_metrics[name][operating_point],
                }
            )

    # Truth-based recurrence is deliberately computed only after all OOF scoring.
    cohort, recurrence_frame = _low_recurrence_cohort(
        all_validation, wells_by_id=wells_by_id
    )
    cohort_indices = np.asarray(
        [all_well_order[well_id] for well_id in cohort], dtype=np.int32
    )
    cohort_rows = int(
        recurrence_frame.loc[
            recurrence_frame["well_id"].isin(cohort), "future_rows"
        ].sum()
    )
    low_recurrence: dict[str, Any] = {
        "definition": "bottom_10_percent_oof_wells_by_true_prefix_recurrence_le_1ft",
        "wells": list(cohort),
        "well_count": len(cohort),
        "rows": cohort_rows,
        "edge_types": {},
    }
    for edge_type in EDGE_TYPES:
        records = overall_records[("main", edge_type)]
        subset = records[np.isin(records["well_index"], cohort_indices)]
        per_type: dict[str, Any] = {}
        for operating_point in ("strict99", "viable97"):
            per_type[operating_point] = _records_metrics(
                subset,
                operating_point=operating_point,
                all_future_rows=cohort_rows,
                gr_observed_future_rows=cohort_rows,
                descriptor_eligible_future_rows=cohort_rows,
            )
            accepted_field = (
                "strict_accept"
                if operating_point == "strict99"
                else "viable_accept"
            )
            supported_wells = set(
                subset["well_index"][subset[accepted_field].astype(bool)].tolist()
            )
            per_type[operating_point]["zero_support_wells"] = (
                len(cohort) - len(supported_wells)
            )
        low_recurrence["edge_types"][edge_type] = per_type

    overall_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in edge_metric_rows:
        if row.get("scope") == "overall" and "query_scale" not in row:
            overall_lookup[
                (
                    str(row["method"]),
                    str(row["edge_type"]),
                    str(row["operating_point"]),
                )
            ] = row
    edge_type_summary: dict[str, Any] = {}
    decisions: dict[str, str] = {}
    for edge_type in EDGE_TYPES:
        strict = overall_lookup[("main", edge_type, "strict99")]
        viable = overall_lookup[("main", edge_type, "viable97")]
        supported = all(
            thresholds_payload[str(fold)]["main"][edge_type][
                "operating_points"
            ][name]["supported"]
            for fold in requested_folds
            for name in ("strict99", "viable97")
        )
        decision = _decision(strict, viable, thresholds_supported=supported)
        decisions[edge_type] = decision
        edge_type_summary[edge_type] = {
            "strict99": strict,
            "viable97": viable,
            "decision": decision,
            "zero_support_wells_strict99": int(
                sum(
                    row["method"] == "main"
                    and row["edge_type"] == edge_type
                    and row["operating_point"] == "strict99"
                    and row["zero_support"]
                    for row in per_well_rows
                )
            ),
            "zero_support_wells_viable97": int(
                sum(
                    row["method"] == "main"
                    and row["edge_type"] == edge_type
                    and row["operating_point"] == "viable97"
                    and row["zero_support"]
                    for row in per_well_rows
                )
            ),
        }
    same_supported = decisions["PF"] != "unsupported" or decisions["FF"] != "unsupported"
    same_decision = _decision(
        combined_metrics["same_well"]["strict99"],
        combined_metrics["same_well"]["viable97"],
        thresholds_supported=same_supported,
    )
    overall_supported = any(
        decisions[edge_type] != "unsupported" for edge_type in EDGE_TYPES
    )
    overall_decision = _decision(
        combined_metrics["overall"]["strict99"],
        combined_metrics["overall"]["viable97"],
        thresholds_supported=overall_supported,
    )
    h2_status = (
        "partial_failure"
        if decisions["HT"] in {"strong", "viable"}
        and decisions["PF"] not in {"strong", "viable"}
        and decisions["FF"] not in {"strong", "viable"}
        else "evaluated"
    )

    control_summary: dict[str, Any] = {}
    for method in (
        "raw_nn",
        "point_gr",
        "no_cycle",
        "destroy_order",
        "shuffle_gr",
        "wrong_typewell",
    ):
        control_summary[method] = {}
        for edge_type in EDGE_TYPES:
            if (method, edge_type, "strict99") not in overall_lookup:
                continue
            control_summary[method][edge_type] = {
                operating_point: overall_lookup[
                    (method, edge_type, operating_point)
                ]
                for operating_point in ("strict99", "viable97")
            }

    threshold_stability: dict[str, Any] = {}
    for edge_type in EDGE_TYPES:
        threshold_stability[edge_type] = {}
        for operating_point in ("strict99", "viable97"):
            values = [
                float(
                    thresholds_payload[str(fold)]["main"][edge_type][
                        "operating_points"
                    ][operating_point]["threshold"]
                )
                for fold in requested_folds
            ]
            threshold_stability[edge_type][operating_point] = {
                "values": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    runtime_payload = {
        "wall_seconds": time.perf_counter() - started,
        "descriptor_build_seconds": runtime.descriptor_seconds,
        "retrieval_seconds": runtime.retrieval_seconds,
        "calibration_seconds": runtime.calibration_seconds,
        "scoring_seconds": runtime.scoring_seconds,
        "peak_candidate_count": runtime.peak_candidates,
        "peak_index_rows": runtime.peak_index_rows,
        "well_cut_generation_runs": runtime.well_cut_runs,
        "workers_requested": workers,
        "execution": (
            "deterministic_single_process_thread_parallel_per_cut_and_control"
        ),
    }
    summary = {
        "audit_id": SLP_E1_AUDIT_ID,
        "run_id": run_id,
        "profile": profile,
        "scope": scope,
        "uses_target_for_evaluation": SLP_E1_USES_TARGET_FOR_EVALUATION,
        "uses_target_for_artificial_cut_calibration": (
            SLP_E1_USES_TARGET_FOR_ARTIFICIAL_CUT_CALIBRATION
        ),
        "edge_generator_inference_safe": SLP_E1_EDGE_GENERATOR_INFERENCE_SAFE,
        "deployable": SLP_E1_DEPLOYABLE,
        "submission_eligible": SLP_E1_SUBMISSION_ELIGIBLE,
        "folds": list(requested_folds),
        "fold_version": fold_version,
        "seed": seed,
        "oof_wells": len(all_validation),
        "oof_future_rows": overall_denominator["all_future_rows"],
        "gr_observed_future_rows": overall_denominator[
            "gr_observed_future_rows"
        ],
        "descriptor_eligible_rows": overall_denominator[
            "descriptor_eligible_future_rows"
        ],
        "edge_types": edge_type_summary,
        "same_well_combined": {
            **combined_metrics["same_well"],
            "decision": same_decision,
        },
        "all_edge_types_combined": {
            **combined_metrics["overall"],
            "decision": overall_decision,
        },
        "controls": control_summary,
        "low_recurrence": low_recurrence,
        "threshold_stability": threshold_stability,
        "decision": {
            **decisions,
            "same_well": same_decision,
            "overall": overall_decision,
            "SLP-H2_status": h2_status,
        },
        "runtime": runtime_payload,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_config = {
        "audit_id": SLP_E1_AUDIT_ID,
        "run_id": run_id,
        "profile": profile,
        "scope": scope,
        "data_root": str(data),
        "fold_config": str(Path(fold_config).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "folds": list(requested_folds),
        "max_train_wells": train_limit,
        "max_validation_wells": validation_limit,
        "max_query_rows_per_well": validation_query_cap,
        "calibration_query_rows_per_cut": (
            DEFAULT_CALIBRATION_QUERY_ROWS_PER_CUT
        ),
        "maximum_calibration_edges_per_well_cut_type": (
            MAX_CALIBRATION_EDGES_PER_WELL_CUT_TYPE
        ),
        "artificial_cuts": list(fractions),
        "descriptor_set": descriptor_set,
        "top_k": top_k,
        "seed": seed,
        "workers": workers,
    }
    descriptor_payload = {
        "descriptor_set": descriptor_set,
        "horizontal_half_spans_md_ft": list(
            generation_config.horizontal_half_spans
        ),
        "typewell_half_spans_tvt_ft": list(
            generation_config.typewell_half_spans
        ),
        "points": generation_config.points,
        "minimum_observed_fraction": (
            generation_config.minimum_observed_fraction
        ),
        "flat_epsilon": generation_config.flat_epsilon,
        "minimum_ff_separation_md_ft": (
            generation_config.minimum_ff_separation
        ),
        "families": ["raw_shape", "derivative_rank", "multiscale_barcode"],
        "deterministic": True,
        "learned_descriptor": False,
    }
    leakage_check = {
        "audit_id": SLP_E1_AUDIT_ID,
        "horizontal_edge_generator_columns": list(
            HORIZONTAL_INFERENCE_COLUMNS
        ),
        "typewell_edge_generator_columns": list(TYPEWELL_INFERENCE_COLUMNS),
        "validation_target_loaded_after_candidates_confidence_and_acceptance_frozen": True,
        "outer_training_role_disjointness_checked": True,
        "outer_validation_disjoint_from_training_roles": True,
        "confidence_fit_only_from_confidence_fit_wells": True,
        "thresholds_only_from_threshold_select_wells": True,
        "cycle_features_target_free": True,
        "formation_surfaces_used": False,
        "geology_used": False,
        "png_used": False,
        "e0_oracle_artifacts_used": False,
        "submission_created": False,
        "model_registry_modified": False,
    }
    thresholds_payload = {
        "fold_version": fold_version,
        "folds": thresholds_payload,
        "threshold_stability": threshold_stability,
        "training_sample_aggregate": (
            pd.DataFrame(threshold_sample_rows)
            .groupby(["fold", "role", "method", "edge_type"], as_index=False)
            .agg(
                edges_before_sampling=("edges_before_sampling", "sum"),
                edges_after_sampling=("edges_after_sampling", "sum"),
            )
            .to_dict("records")
        ),
    }

    metric_frame = pd.DataFrame(edge_metric_rows).sort_values(
        ["scope", "method", "edge_type", "operating_point", "fold"],
        kind="stable",
    )
    operating_frame = pd.DataFrame(operating_point_rows).sort_values(
        ["fold", "method", "edge_type", "operating_point"], kind="stable"
    )
    calibration_frame = pd.DataFrame(calibration_bin_rows).sort_values(
        ["scope", "method", "edge_type", "fold", "bin"], kind="stable"
    )
    per_well_frame = pd.DataFrame(per_well_rows).sort_values(
        ["fold", "well_id", "method", "edge_type", "operating_point"],
        kind="stable",
    )
    negative_frame = metric_frame.loc[
        metric_frame["scope"].eq("overall")
        & metric_frame["method"].ne("main")
        & ~metric_frame.get(
            "query_scale", pd.Series(index=metric_frame.index, dtype=float)
        ).notna()
    ].copy()
    fold_manifest_payload = {
        "fold_version": fold_version,
        "folds": fold_manifest,
        "artificial_cuts": sorted(
            all_cut_records,
            key=lambda row: (
                row["fold"],
                row["role"],
                row["well_id"],
                row["fraction"],
            ),
        ),
    }
    recurrence_frame = recurrence_frame.sort_values("well_id", kind="stable")
    _write_json_atomic(run_root / "summary.json", summary)
    _write_json_atomic(run_root / "run_config.json", run_config)
    _write_json_atomic(run_root / "fold_manifest.json", fold_manifest_payload)
    _write_json_atomic(
        run_root / "descriptor_config.json", descriptor_payload
    )
    _write_json_atomic(run_root / "thresholds.json", thresholds_payload)
    _write_csv_atomic(metric_frame, run_root / "edge_metrics.csv")
    _write_csv_atomic(
        operating_frame, run_root / "operating_points.csv"
    )
    _write_csv_atomic(
        calibration_frame, run_root / "calibration_bins.csv"
    )
    _write_csv_atomic(
        per_well_frame,
        run_root / "per_well_metrics.csv.gz",
        gzip_compressed=True,
    )
    _write_csv_atomic(
        negative_frame, run_root / "negative_controls.csv"
    )
    _write_csv_atomic(
        recurrence_frame,
        run_root / "low_recurrence_per_well.csv.gz",
        gzip_compressed=True,
    )
    _write_json_atomic(run_root / "runtime.json", runtime_payload)
    _write_json_atomic(run_root / "leakage_check.json", leakage_check)
    return {
        "run_root": run_root,
        "run_id": run_id,
        "audit_id": SLP_E1_AUDIT_ID,
        "profile": profile,
        "scope": scope,
        "folds": list(requested_folds),
        "oof_wells": len(all_validation),
        "oof_future_rows": overall_denominator["all_future_rows"],
        "decision": summary["decision"],
    }
