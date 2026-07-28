"""Evidence harness for inference-safe heel-registered Anchor Student research.

The production algorithm lives in :mod:`rogii.heel_registration` and
:mod:`rogii.models.heel_registered_anchor_student`.  This module owns every
operation that may read training targets: strict OOF evaluation, controls,
multi-seed refits, uncertainty summaries, and artifact audits.

Two profiles are intentionally separate:

``quick``
    Reuses the exact 100-well Sparse-Anchor quick OOF cohort and evaluates
    three mechanistically different families plus negative controls.

``full``
    Reuses only the fixed fold-fit P3/Path-PCA states, refits the Anchor
    Student on five seeds within each outer fold, applies the locked
    registration, and evaluates all 773 wells / 3,783,989 prediction rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from ..data import (
    WellFiles,
    enumerate_wells,
    load_horizontal_inference,
    load_typewell_inference,
    validate_prefix,
)
from ..folds import FOLD_COUNT, load_fold_mapping, partition_well_ids
from ..heel_registration import (
    HeelRegistrationResult,
    clipped_quantile_loss,
    estimate_heel_registration,
    heel_registration_manifest,
)
from ..models.base import RunContext, TrainingWell
from ..models.heel_registered_anchor_student import (
    HeelRegisteredAnchorStudentModel,
    heel_registered_anchor_student_manifest,
    heel_registered_state_provenance,
    wrap_sparse_anchor_student_state,
)
from ..models.sparse_anchor_student import (
    SparseAnchorInferenceWell,
    SparseAnchorStudentState,
    TrainingRangeClipper,
    fit_student_estimator,
    predict_anchor_delta,
)
from ..resources import configure_cpu
from ..sparse_anchor import (
    ANCHOR_FRACTIONS,
    DESCRIPTOR_VERSION,
    AnchorPlacement,
    build_anchor_delta_targets,
    reconstruct_from_predicted_anchors,
)
from ..submission import (
    align_predictions_to_sample,
    validate_sample_submission_contract,
)


PRIMARY_SEED = 20260711
FULL_SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)
BASELINE_COLUMN = "y_pred_extra_trees_s20260711"
PRIMARY_COLUMN = "y_pred_heel_registered_ensemble"
EXPECTED_FULL_WELLS = 773
EXPECTED_FULL_ROWS = 3_783_989
BOOTSTRAP_SEED = 20260728


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_run_root(
    output_root: Path,
    run_id: str,
    data_root: Path,
    *,
    resume: bool,
) -> Path:
    root = (Path(output_root).resolve() / str(run_id)).resolve()
    data = Path(data_root).resolve()
    if root == data or data in root.parents:
        raise ValueError("research output cannot be placed inside competition data")
    if root.exists():
        if not resume:
            raise FileExistsError(f"run already exists: {root}")
        if (root / "summary.json").exists() and (root / "artifact_audit.json").exists():
            raise ValueError("cannot resume a completed run")
    else:
        root.mkdir(parents=True)
    (root / "models").mkdir(exist_ok=True)
    return root


def _data_manifest(root: Path) -> dict[str, Any]:
    """Hash names/sizes/mtimes without rereading the large read-only inputs."""
    rows: list[str] = []
    total_bytes = 0
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        rows.append(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}")
        total_bytes += int(stat.st_size)
    payload = "\n".join(rows).encode("utf-8")
    return {
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "name_size_mtime_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_baseline_oof(path: Path) -> pd.DataFrame:
    required = [
        "well_id",
        "row_index",
        "fold",
        "y_true",
        "y_pred_blend",
        BASELINE_COLUMN,
    ]
    frame = pd.read_csv(
        path,
        usecols=required,
        dtype={"well_id": "string", "row_index": "int64", "fold": "int8"},
    )
    if frame.duplicated(["well_id", "row_index"]).any():
        raise ValueError("baseline OOF contains duplicate row keys")
    if not np.isfinite(frame.drop(columns="well_id").to_numpy(dtype=np.float64)).all():
        raise ValueError("baseline OOF contains non-finite values")
    return frame.sort_values(
        ["fold", "well_id", "row_index"], kind="stable"
    ).reset_index(drop=True)


def _target_from_train_file(well: WellFiles, row_index: np.ndarray) -> np.ndarray:
    """Training-only evaluation loader; never called by production inference."""
    header = pd.read_csv(well.horizontal_path, nrows=0)
    if "TVT" not in header.columns:
        raise ValueError(f"training target missing for {well.well_id}")
    frame = pd.read_csv(well.horizontal_path, usecols=["TVT", "TVT_input"])
    known_count, prediction_count = validate_prefix(
        frame["TVT_input"], target=frame["TVT"]
    )
    expected = np.arange(known_count, known_count + prediction_count, dtype=np.int64)
    if not np.array_equal(expected, row_index):
        raise ValueError(f"training target row alignment differs: {well.well_id}")
    target = frame["TVT"].iloc[known_count:].to_numpy(dtype=np.float64)
    if target.shape != row_index.shape or not np.isfinite(target).all():
        raise ValueError(f"invalid evaluation target: {well.well_id}")
    return target


def _descriptor_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()


def _load_lightweight_payload(
    well: WellFiles,
    *,
    row_index: np.ndarray,
    cache_root: Path,
) -> SparseAnchorInferenceWell:
    """Load the small descriptor cache without retaining row-level P3 tensors."""
    horizontal = load_horizontal_inference(well.horizontal_path)
    known_count, prediction_count = validate_prefix(horizontal["TVT_input"])
    expected_rows = np.arange(
        known_count, known_count + prediction_count, dtype=np.int64
    )
    if not np.array_equal(row_index, expected_rows):
        raise ValueError(f"baseline row alignment differs: {well.well_id}")
    path = Path(cache_root) / DESCRIPTOR_VERSION / "train" / f"{well.well_id}.npz"
    if not path.is_file():
        # Rebuilding descriptors is safe, but intentionally delegated to the
        # production prepare path so there is only one feature implementation.
        from ..models.sparse_anchor_student import SparseAnchorStudentModel

        prepared = SparseAnchorStudentModel().prepare(
            well, split="train", cache_root=Path(cache_root)
        )
        return SparseAnchorInferenceWell(
            well_id=prepared.well_id,
            row_index=prepared.row_index,
            anchor=prepared.anchor,
            features=prepared.features,
            descriptor_names=prepared.descriptor_names,
            p3=None,  # type: ignore[arg-type]
            path_pca=None,  # type: ignore[arg-type]
            placement=prepared.placement,
        )
    with np.load(path, allow_pickle=False) as cached:
        features = cached["descriptor"].astype(np.float64, copy=True)
        names = tuple(str(value) for value in cached["descriptor_names"].tolist())
        local = cached["anchor_local_indices"].astype(np.int64, copy=True)
        rows = cached["anchor_row_indices"].astype(np.int64, copy=True)
        normalized = cached["normalized_positions"].astype(np.float64, copy=True)
        selected = cached["selected_positions"].astype(np.float64, copy=True)
        requested = cached["requested_positions"].astype(np.float64, copy=True)
        cached_rows = cached["prediction_row_index"].astype(np.int64, copy=True)
        digest = str(cached["descriptor_name_digest"][0])
    valid = (
        features.ndim == 1
        and len(features) == len(names)
        and not np.isinf(features).any()
        and len(set(names)) == len(names)
        and digest == _descriptor_digest(names)
        and np.array_equal(cached_rows, row_index)
        and local.shape == (len(ANCHOR_FRACTIONS),)
        and np.all(np.diff(local) > 0)
        and np.array_equal(rows, row_index[local])
        and normalized.shape == row_index.shape
        and np.array_equal(selected, normalized[local])
        and np.array_equal(requested, np.asarray(ANCHOR_FRACTIONS))
    )
    if not valid:
        raise ValueError(f"descriptor cache contract differs: {well.well_id}")
    placement = AnchorPlacement(
        local_indices=local,
        row_indices=rows,
        normalized_positions=normalized,
        selected_positions=selected,
        requested_positions=requested,
    )
    return SparseAnchorInferenceWell(
        well_id=well.well_id,
        row_index=row_index.copy(),
        anchor=float(horizontal["TVT_input"].iloc[known_count - 1]),
        features=features,
        descriptor_names=names,
        p3=None,  # type: ignore[arg-type]
        path_pca=None,  # type: ignore[arg-type]
        placement=placement,
    )


def _prepare_payloads_and_targets(
    well_ids: Iterable[str],
    *,
    files: dict[str, WellFiles],
    row_indices: dict[str, np.ndarray],
    cache_root: Path,
    known_targets: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, SparseAnchorInferenceWell], dict[str, np.ndarray], dict[str, Any]]:
    payloads: dict[str, SparseAnchorInferenceWell] = {}
    targets: dict[str, np.ndarray] = {}
    target_maximum_delta = 0.0
    for offset, well_id in enumerate(sorted(set(map(str, well_ids)))):
        payload = _load_lightweight_payload(
            files[well_id], row_index=row_indices[well_id], cache_root=cache_root
        )
        raw_target = _target_from_train_file(files[well_id], payload.row_index)
        if known_targets is not None and well_id in known_targets:
            reference = known_targets[well_id]
            if reference.shape != raw_target.shape:
                raise ValueError(f"baseline truth shape differs: {well_id}")
            target_maximum_delta = max(
                target_maximum_delta,
                float(np.max(np.abs(reference - raw_target))),
            )
        payloads[well_id] = payload
        targets[well_id] = raw_target
        if (offset + 1) % 100 == 0:
            print(f"prepared lightweight wells={offset + 1}", flush=True)
    names = next(iter(payloads.values())).descriptor_names
    if any(payload.descriptor_names != names for payload in payloads.values()):
        raise ValueError("descriptor schema differs across wells")
    return (
        payloads,
        targets,
        {
            "wells": len(payloads),
            "descriptor_width": len(names),
            "raw_target_vs_baseline_maximum_absolute_delta": target_maximum_delta,
            "descriptor_cache_version": DESCRIPTOR_VERSION,
        },
    )


def _rmse(sse: float, rows: int) -> float:
    return float(math.sqrt(float(sse) / int(rows)))


def _metric_tables(
    oof: pd.DataFrame,
    prediction_columns: dict[str, str],
    *,
    baseline: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    fold_counts = oof.groupby("well_id", sort=False)["fold"].nunique()
    if (fold_counts != 1).any():
        invalid = fold_counts.loc[fold_counts.ne(1)].index.astype(str).tolist()
        raise ValueError(
            f"each OOF well must belong to exactly one fold: {invalid[:5]}"
        )
    truth = oof["y_true"].to_numpy(dtype=np.float64)
    well_groups = oof.groupby(["fold", "well_id"], sort=True).indices
    fold_groups = oof.groupby("fold", sort=True).indices
    well_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    overall: dict[str, float] = {}
    for candidate, column in prediction_columns.items():
        prediction = oof[column].to_numpy(dtype=np.float64)
        squared = (prediction - truth) ** 2
        overall[candidate] = _rmse(float(np.sum(squared)), len(oof))
        for (fold, well_id), indices in well_groups.items():
            selected = np.asarray(indices, dtype=np.int64)
            sse = float(np.sum(squared[selected]))
            rows = len(selected)
            well_rows.append(
                {
                    "candidate": candidate,
                    "fold": int(fold),
                    "well_id": str(well_id),
                    "n_rows": rows,
                    "sse": sse,
                    "rmse": _rmse(sse, rows),
                }
            )
        for fold, indices in fold_groups.items():
            selected = np.asarray(indices, dtype=np.int64)
            sse = float(np.sum(squared[selected]))
            fold_rows.append(
                {
                    "candidate": candidate,
                    "fold": int(fold),
                    "wells": int(oof.iloc[selected]["well_id"].nunique()),
                    "n_rows": len(selected),
                    "sse": sse,
                    "pooled_row_rmse": _rmse(sse, len(selected)),
                }
            )
    per_well = pd.DataFrame(well_rows)
    reference = per_well.loc[
        per_well["candidate"].eq(baseline), ["well_id", "rmse", "sse"]
    ].rename(columns={"rmse": "baseline_rmse", "sse": "baseline_sse"})
    per_well = per_well.merge(
        reference, on="well_id", how="left", validate="many_to_one"
    )
    per_well["rmse_delta_vs_baseline"] = per_well["rmse"] - per_well["baseline_rmse"]
    return per_well, pd.DataFrame(fold_rows), overall


def _tail_summary(per_well: pd.DataFrame, candidate: str) -> dict[str, Any]:
    frame = per_well.loc[per_well["candidate"].eq(candidate)]
    rmse = frame["rmse"].to_numpy(dtype=np.float64)
    sse = frame["sse"].to_numpy(dtype=np.float64)
    worst = max(1, math.ceil(0.1 * len(frame)))
    total_sse = float(np.sum(sse))
    return {
        "p50_rmse": float(np.quantile(rmse, 0.50)),
        "p90_rmse": float(np.quantile(rmse, 0.90)),
        "p95_rmse": float(np.quantile(rmse, 0.95)),
        "max_rmse": float(np.max(rmse)),
        "wells_rmse_ge_40": int(np.sum(rmse >= 40.0)),
        "wells_rmse_ge_60": int(np.sum(rmse >= 60.0)),
        "worst_10pct_sse_share": (
            float(np.sort(sse)[-worst:].sum() / total_sse) if total_sse > 0.0 else 0.0
        ),
    }


def _paired_bootstrap(
    per_well: pd.DataFrame,
    candidates: Sequence[str],
    *,
    baseline: str,
    samples: int = 10_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    required = {baseline, *candidates}
    selected = per_well.loc[per_well["candidate"].isin(required)].copy()
    if selected.duplicated(["well_id", "candidate"]).any():
        raise ValueError("paired bootstrap has duplicate well/candidate rows")
    if set(selected["candidate"].astype(str)) != required:
        raise ValueError("paired bootstrap candidate coverage is incomplete")
    if (
        not np.isfinite(selected[["sse", "n_rows"]].to_numpy(dtype=np.float64)).all()
        or (selected["sse"] < 0.0).any()
        or (selected["n_rows"] <= 0).any()
    ):
        raise ValueError("paired bootstrap SSE/row counts are invalid")
    sse = per_well.pivot(index="well_id", columns="candidate", values="sse")
    rows = per_well.pivot(index="well_id", columns="candidate", values="n_rows")
    if sse.isna().any().any() or rows.isna().any().any():
        raise ValueError("paired bootstrap requires complete well coverage")
    counts = rows[baseline].to_numpy(dtype=np.int64)
    for candidate in required:
        if not np.array_equal(rows[candidate].to_numpy(dtype=np.int64), counts):
            raise ValueError("paired bootstrap candidate row counts differ")
    baseline_sse = sse[baseline].to_numpy(dtype=np.float64)
    random = np.random.default_rng(seed)
    values = {
        candidate: np.empty(samples, dtype=np.float64) for candidate in candidates
    }
    offset = 0
    while offset < samples:
        block = min(256, samples - offset)
        indices = random.integers(0, len(counts), size=(block, len(counts)))
        denominator = np.sum(counts[indices], axis=1)
        base = np.sqrt(np.sum(baseline_sse[indices], axis=1) / denominator)
        for candidate in candidates:
            candidate_sse = sse[candidate].to_numpy(dtype=np.float64)
            metric = np.sqrt(np.sum(candidate_sse[indices], axis=1) / denominator)
            values[candidate][offset : offset + block] = metric - base
        offset += block
    return {
        "method": (
            "paired well-cluster bootstrap; wells sampled with replacement; "
            "all rows of each selected well retained"
        ),
        "baseline": baseline,
        "samples": samples,
        "seed": seed,
        "candidates": {
            candidate: {
                "delta_rmse_q2.5": float(np.quantile(delta, 0.025)),
                "delta_rmse_q50": float(np.quantile(delta, 0.50)),
                "delta_rmse_q97.5": float(np.quantile(delta, 0.975)),
            }
            for candidate, delta in values.items()
        },
    }


def _anchor_decomposition(anchor_oof: pd.DataFrame, candidate: str) -> dict[str, Any]:
    frame = anchor_oof.loc[anchor_oof["candidate"].eq(candidate)].sort_values(
        ["well_id", "anchor_order"], kind="stable"
    )
    error = (
        frame["predicted_anchor_tvt"].to_numpy(dtype=np.float64)
        - frame["true_anchor_tvt"].to_numpy(dtype=np.float64)
    ).reshape(-1, len(ANCHOR_FRACTIONS))
    common = np.mean(error, axis=1)
    centered = error - common[:, None]
    position = np.sqrt(np.mean(error**2, axis=0))
    return {
        "wells": len(error),
        "anchor_rmse": float(np.sqrt(np.mean(error**2))),
        "common_mode_rmse": float(np.sqrt(np.mean(common**2))),
        "common_mode_bias": float(np.mean(common)),
        "centered_shape_rmse": float(np.sqrt(np.mean(centered**2))),
        "common_mode_mse_share": float(np.mean(common**2) / np.mean(error**2)),
        "anchor_position_rmse": position.tolist(),
        "error_covariance": np.cov(error, rowvar=False, ddof=1).tolist(),
        "error_correlation": np.corrcoef(error, rowvar=False).tolist(),
    }


def _registration_diagnostics(
    result: HeelRegistrationResult,
    *,
    well_id: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    weights = result.posterior_weight
    shifts = result.shift_grid
    posterior_std = float(
        np.sqrt(weights @ (shifts - result.posterior_mean_shift) ** 2)
    )
    ordered_cost = np.sort(result.candidate_cost)
    return {
        "well_id": well_id,
        "fold": fold,
        "seed": seed,
        "posterior_mean_shift": result.posterior_mean_shift,
        "map_shift": result.map_shift,
        "posterior_shift_std": posterior_std,
        "map_mean_gap": abs(result.map_shift - result.posterior_mean_shift),
        "posterior_entropy": result.posterior_entropy,
        "effective_support": result.effective_support,
        "maximum_posterior": result.maximum_posterior,
        "edge_mass": result.edge_mass,
        "best_coverage": result.best_coverage,
        "posterior_mean_coverage": result.posterior_mean_coverage,
        "minimum_cost": float(ordered_cost[0]),
        "second_best_cost_gap": float(ordered_cost[1] - ordered_cost[0]),
        "prefix_pair_count": result.calibration.pair_count,
        "prefix_coverage": result.calibration.prefix_coverage,
        "prefix_mse": result.calibration.prefix_mse,
        "calibration_intercept": result.calibration.intercept,
        "calibration_slope": result.calibration.slope,
    }


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = pd.Series(left).rank(method="average").to_numpy(dtype=np.float64)
    b = pd.Series(right).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _risk_signal_analysis(
    diagnostic_rows: pd.DataFrame,
    per_well: pd.DataFrame,
    *,
    primary: str,
    baseline: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    diagnostics = diagnostic_rows.groupby("well_id", as_index=False).mean(
        numeric_only=True
    )
    candidate = per_well.loc[
        per_well["candidate"].eq(primary), ["well_id", "rmse", "sse", "n_rows"]
    ].rename(columns={"rmse": "candidate_rmse", "sse": "candidate_sse"})
    reference = per_well.loc[
        per_well["candidate"].eq(baseline), ["well_id", "rmse", "sse"]
    ].rename(columns={"rmse": "baseline_rmse", "sse": "baseline_sse"})
    frame = diagnostics.merge(candidate, on="well_id", validate="one_to_one").merge(
        reference, on="well_id", validate="one_to_one"
    )
    frame["rmse_delta"] = frame["candidate_rmse"] - frame["baseline_rmse"]
    frame["catastrophic_ge_40"] = frame["candidate_rmse"] >= 40.0
    frame["worsened"] = frame["rmse_delta"] > 0.0
    ignored = {
        "fold",
        "seed",
        "prefix_pair_count",
        "candidate_rmse",
        "candidate_sse",
        "baseline_rmse",
        "baseline_sse",
        "n_rows",
    }
    signals = [
        column
        for column in diagnostics.columns
        if column != "well_id" and column not in ignored
    ]
    analysis: dict[str, Any] = {}
    labels = frame["catastrophic_ge_40"].to_numpy(dtype=bool)
    for signal in signals:
        values = frame[signal].to_numpy(dtype=np.float64)
        auc = None
        if labels.any() and (~labels).any() and np.std(values) > 0.0:
            raw_auc = float(roc_auc_score(labels, values))
            auc = {
                "raw": raw_auc,
                "two_sided_posthoc": max(raw_auc, 1.0 - raw_auc),
            }
        analysis[signal] = {
            "spearman_vs_candidate_rmse": _rank_correlation(
                values, frame["candidate_rmse"].to_numpy(dtype=np.float64)
            ),
            "spearman_vs_rmse_delta": _rank_correlation(
                values, frame["rmse_delta"].to_numpy(dtype=np.float64)
            ),
            "catastrophic_auc": auc,
        }
    worst = frame.nlargest(20, "candidate_rmse")[
        [
            "well_id",
            "fold",
            "candidate_rmse",
            "baseline_rmse",
            "rmse_delta",
            "posterior_mean_shift",
            "posterior_shift_std",
            "maximum_posterior",
            "edge_mass",
            "best_coverage",
            "prefix_mse",
        ]
    ]
    return frame, {
        "all_signals_are_pre_prediction_computable": True,
        "uses_target_to_choose_or_gate_prediction": False,
        "signals": analysis,
        "catastrophic_wells": int(labels.sum()),
        "worst_20": worst.to_dict(orient="records"),
    }


def _full_bootstrap_bundle(
    per_well: pd.DataFrame, seeds: Sequence[int]
) -> dict[str, Any]:
    incumbent_candidates = sorted(
        set(per_well["candidate"].astype(str)) - {"anchor_student"}
    )
    incumbent = _paired_bootstrap(
        per_well,
        incumbent_candidates,
        baseline="anchor_student",
        samples=10_000,
    )
    ensemble_matched = _paired_bootstrap(
        per_well,
        ["heel_registered_ensemble"],
        baseline="anchor_seed_ensemble",
        samples=10_000,
    )
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        per_seed[str(seed)] = _paired_bootstrap(
            per_well,
            [f"heel_registered_seed_{seed}"],
            baseline=f"anchor_seed_{seed}",
            samples=10_000,
        )
    return {
        "vs_historical_anchor_student": incumbent,
        "matched_registration_ensemble": ensemble_matched,
        "matched_registration_per_seed": per_seed,
    }


def _write_compact_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        f"# {summary['profile'].title()} external-anchor research run",
        "",
        f"Created UTC: {summary['created_utc']}",
        "",
        f"Primary: `{summary['primary_candidate']}`",
        f"Baseline RMSE: {metrics[summary['baseline_candidate']]:.12f}",
        f"Primary RMSE: {metrics[summary['primary_candidate']]:.12f}",
        f"Delta: {metrics[summary['primary_candidate']] - metrics[summary['baseline_candidate']]:+.12f}",
        "",
        "This is a machine-generated artifact summary. See "
        "`results/external_anchor_research_20260728.md` for the formal report.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _robust_h_space_line(
    horizontal: pd.DataFrame, *, window_ft: float = 250.0
) -> np.ndarray:
    """Extrapolate ``H = TVT_input + Z`` from the last visible heel window."""
    known_count, prediction_count = validate_prefix(horizontal["TVT_input"])
    md = horizontal["MD"].to_numpy(dtype=np.float64)
    z = horizontal["Z"].to_numpy(dtype=np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(dtype=np.float64)
    if not np.isfinite(md).all() or not np.isfinite(z).all():
        raise ValueError("H-space extrapolation requires finite MD and Z")
    visible_md = md[:known_count]
    selected = visible_md >= visible_md[-1] - float(window_ft)
    x = visible_md[selected]
    y = (tvt_input[:known_count] + z[:known_count])[selected]
    if len(x) < 3 or not np.isfinite(y).all():
        raise ValueError("H-space heel window has insufficient finite rows")
    center = float(x[-1])
    design = np.column_stack([np.ones(len(x)), x - center])
    weights = np.ones(len(x), dtype=np.float64)
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    # Three deterministic Huber IRLS updates make tool spikes unable to set
    # the extrapolated slope by themselves.
    for _ in range(3):
        residual = y - design @ coefficient
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if scale <= np.finfo(np.float64).eps:
            break
        threshold = 1.345 * scale
        absolute = np.abs(residual)
        weights = np.ones(len(x), dtype=np.float64)
        weights[absolute > threshold] = threshold / absolute[absolute > threshold]
        root = np.sqrt(weights)
        coefficient = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    suffix_h = coefficient[0] + coefficient[1] * (md[known_count:] - center)
    prediction = suffix_h - z[known_count:]
    if prediction.shape != (prediction_count,) or not np.isfinite(prediction).all():
        raise ValueError("H-space extrapolation produced invalid values")
    return prediction


def _anchor_basis(components: int = 6) -> np.ndarray:
    if components < 1 or components > len(ANCHOR_FRACTIONS):
        raise ValueError("anchor basis component count is invalid")
    x = 2.0 * np.asarray(ANCHOR_FRACTIONS, dtype=np.float64) - 1.0
    vandermonde = np.column_stack([x**degree for degree in range(components)])
    basis, _ = np.linalg.qr(vandermonde, mode="reduced")
    return basis


def _fit_low_rank_anchor_estimator(
    training: Sequence[TrainingWell],
    *,
    seed: int,
    workers: int,
    shuffled_targets: bool,
) -> tuple[Pipeline, np.ndarray]:
    if not training:
        raise ValueError("low-rank candidate requires training wells")
    x = np.stack([item.inference.features for item in training])
    anchor_targets = np.stack(
        [
            build_anchor_delta_targets(
                item.target,
                last_known_tvt=item.inference.anchor,
                placement=item.inference.placement,
            )
            for item in training
        ]
    )
    if shuffled_targets:
        order = np.random.default_rng(seed + 91_337).permutation(len(anchor_targets))
        anchor_targets = anchor_targets[order]
    basis = _anchor_basis(6)
    coefficients = anchor_targets @ basis
    estimator = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("clip", TrainingRangeClipper()),
            (
                "trees",
                ExtraTreesRegressor(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=5,
                    max_features=0.5,
                    n_jobs=max(1, int(workers)),
                    random_state=int(seed),
                ),
            ),
        ]
    )
    estimator.fit(x, coefficients)
    return estimator, basis


def _predict_low_rank_anchor(
    state: tuple[Pipeline, np.ndarray], payload: SparseAnchorInferenceWell
) -> np.ndarray:
    estimator, basis = state
    coefficient = np.asarray(estimator.predict(payload.features[None, :]))[0]
    prediction = np.asarray(coefficient @ basis.T, dtype=np.float64)
    if (
        prediction.shape != (len(ANCHOR_FRACTIONS),)
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("low-rank anchor estimator produced invalid values")
    return prediction


def run_external_anchor_quick_experiment(
    *,
    data_root: Path,
    fold_config: Path,
    output_root: Path,
    run_id: str,
    cache_root: Path,
    baseline_root: Path = Path(".runs/sparse-anchor-student-quick"),
    workers: int = 4,
    seed: int = PRIMARY_SEED,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the three-family quick portfolio on the exact historical cohort."""
    configure_cpu(workers=workers)
    started = time.perf_counter()
    data = Path(data_root).resolve()
    before_data = _data_manifest(data)
    run_root = _safe_run_root(output_root, run_id, data, resume=resume)
    baseline_path = Path(baseline_root).resolve()
    oof = _read_baseline_oof(baseline_path / "oof_predictions.csv.gz")
    historical_config = json.loads(
        (baseline_path / "run_config.json").read_text(encoding="utf-8")
    )
    if len(oof) != 489_284 or oof["well_id"].nunique() != 100:
        raise ValueError("quick baseline does not have the locked 100-well scope")

    files = {well.well_id: well for well in enumerate_wells(data, "train")}
    mapping = load_fold_mapping(fold_config, expected_well_ids=files)
    groups = {
        str(well_id): np.asarray(indices, dtype=np.int64)
        for well_id, indices in oof.groupby("well_id", sort=True).indices.items()
    }
    row_indices = {
        well_id: oof.iloc[indices]["row_index"].to_numpy(dtype=np.int64)
        for well_id, indices in groups.items()
    }
    target_reference = {
        well_id: oof.iloc[indices]["y_true"].to_numpy(dtype=np.float64)
        for well_id, indices in groups.items()
    }
    fold_provenance = historical_config["fold_provenance"]
    training_ids = {
        int(fold): tuple(
            map(
                str,
                value["student_fit_provenance"][
                    BASELINE_COLUMN.removeprefix("y_pred_")
                ],
            )
        )
        for fold, value in fold_provenance.items()
    }
    validation_ids = {
        int(fold): tuple(map(str, value["validation_wells"]))
        for fold, value in fold_provenance.items()
    }
    for fold in range(FOLD_COUNT):
        expected_training, expected_validation = partition_well_ids(mapping, fold)
        if not set(training_ids[fold]).issubset(expected_training):
            raise ValueError(f"quick training subset violates fixed fold {fold}")
        if not set(validation_ids[fold]).issubset(expected_validation):
            raise ValueError(f"quick validation subset violates fixed fold {fold}")
    required = set(groups)
    for values in training_ids.values():
        required.update(values)
    missing_rows = required - set(row_indices)
    for well_id in sorted(missing_rows):
        horizontal = load_horizontal_inference(files[well_id].horizontal_path)
        known_count, prediction_count = validate_prefix(horizontal["TVT_input"])
        row_indices[well_id] = np.arange(
            known_count, known_count + prediction_count, dtype=np.int64
        )
    payloads, targets, preparation = _prepare_payloads_and_targets(
        required,
        files=files,
        row_indices=row_indices,
        cache_root=cache_root,
        known_targets=target_reference,
    )

    low_rank: dict[int, tuple[Pipeline, np.ndarray]] = {}
    shuffled: dict[int, tuple[Pipeline, np.ndarray]] = {}
    with threadpool_limits(limits=workers):
        for fold in range(FOLD_COUNT):
            if set(training_ids[fold]) & set(validation_ids[fold]):
                raise ValueError(f"quick fold {fold} has training/validation overlap")
            training = [
                TrainingWell(payloads[well_id], targets[well_id])
                for well_id in training_ids[fold]
            ]
            low_rank[fold] = _fit_low_rank_anchor_estimator(
                training,
                seed=seed + fold,
                workers=workers,
                shuffled_targets=False,
            )
            shuffled[fold] = _fit_low_rank_anchor_estimator(
                training,
                seed=seed + fold,
                workers=workers,
                shuffled_targets=True,
            )
            _atomic_pickle(
                run_root / "models" / f"low_rank_fold_{fold}.pkl", low_rank[fold]
            )
            _atomic_pickle(
                run_root / "models" / f"low_rank_shuffled_fold_{fold}.pkl",
                shuffled[fold],
            )

    candidate_arrays = {
        "anchor_student": oof[BASELINE_COLUMN].to_numpy(dtype=np.float64).copy(),
        "heel_quantile_posterior": np.empty(len(oof), dtype=np.float64),
        "heel_hard_map": np.empty(len(oof), dtype=np.float64),
        "heel_wrong_typewell": np.empty(len(oof), dtype=np.float64),
        "random_common_shift": np.empty(len(oof), dtype=np.float64),
        "h_space_line": np.empty(len(oof), dtype=np.float64),
        "h_space_blend_005": np.empty(len(oof), dtype=np.float64),
        "low_rank_anchor_6": np.empty(len(oof), dtype=np.float64),
        "low_rank_shuffled_targets": np.empty(len(oof), dtype=np.float64),
    }
    diagnostics: list[dict[str, Any]] = []
    invariant_maximum = 0.0
    wrong_pairing: dict[str, str] = {}
    ordered_wells = sorted(groups)
    random = np.random.default_rng(seed + 7_919)
    for position, well_id in enumerate(ordered_wells):
        indices = groups[well_id]
        fold = int(oof.iloc[indices[0]]["fold"])
        if well_id not in validation_ids[fold]:
            raise ValueError(f"quick validation provenance differs: {well_id}")
        file = files[well_id]
        horizontal = load_horizontal_inference(file.horizontal_path)
        typewell = load_typewell_inference(file.typewell_path)
        base = candidate_arrays["anchor_student"][indices]
        result = estimate_heel_registration(horizontal, typewell, base)
        candidate_arrays["heel_quantile_posterior"][indices] = (
            base + result.posterior_mean_shift
        )
        candidate_arrays["heel_hard_map"][indices] = base + result.map_shift
        diagnostics.append(
            _registration_diagnostics(result, well_id=well_id, fold=fold, seed=seed)
        )
        # Deterministic wrong-well control. Unsupported pairings are skipped
        # using only safe support/calibration checks, never target quality.
        wrong_result = None
        wrong_id = ""
        for offset in range(1, len(ordered_wells)):
            candidate_id = ordered_wells[(position + offset) % len(ordered_wells)]
            try:
                wrong_result = estimate_heel_registration(
                    horizontal,
                    load_typewell_inference(files[candidate_id].typewell_path),
                    base,
                )
                wrong_id = candidate_id
                break
            except ValueError:
                continue
        if wrong_result is None:
            raise ValueError(f"no supported wrong-typewell control for {well_id}")
        wrong_pairing[well_id] = wrong_id
        candidate_arrays["heel_wrong_typewell"][indices] = (
            base + wrong_result.posterior_mean_shift
        )
        candidate_arrays["random_common_shift"][indices] = base + random.uniform(
            -40.0, 40.0
        )
        line = _robust_h_space_line(horizontal)
        candidate_arrays["h_space_line"][indices] = line
        candidate_arrays["h_space_blend_005"][indices] = 0.95 * base + 0.05 * line
        payload = payloads[well_id]
        blend = oof.iloc[indices]["y_pred_blend"].to_numpy(dtype=np.float64)
        low_delta = _predict_low_rank_anchor(low_rank[fold], payload)
        shuffled_delta = _predict_low_rank_anchor(shuffled[fold], payload)
        candidate_arrays["low_rank_anchor_6"][indices] = (
            reconstruct_from_predicted_anchors(
                blend, payload.anchor + low_delta, payload.placement
            )
        )
        candidate_arrays["low_rank_shuffled_targets"][indices] = (
            reconstruct_from_predicted_anchors(
                blend, payload.anchor + shuffled_delta, payload.placement
            )
        )
        # Explicitly audit the locked objective's order invariance.
        suffix_gr = horizontal["GR"].iloc[payload.row_index].to_numpy(dtype=np.float64)
        suffix_gr = suffix_gr[np.isfinite(suffix_gr)]
        if len(suffix_gr) > 1:
            loss = clipped_quantile_loss(suffix_gr, typewell["GR"].dropna())
            reversed_loss = clipped_quantile_loss(
                suffix_gr[::-1], typewell["GR"].dropna().to_numpy()[::-1]
            )
            invariant_maximum = max(invariant_maximum, abs(loss - reversed_loss))

    for name, values in candidate_arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"quick candidate contains non-finite values: {name}")
        oof[f"y_pred_{name}"] = values
    prediction_columns = {name: f"y_pred_{name}" for name in candidate_arrays}
    per_well, per_fold, overall = _metric_tables(
        oof, prediction_columns, baseline="anchor_student"
    )
    bootstrap = _paired_bootstrap(
        per_well,
        [name for name in candidate_arrays if name != "anchor_student"],
        baseline="anchor_student",
    )
    tail = {name: _tail_summary(per_well, name) for name in candidate_arrays}
    fold_direction = {
        name: int(
            (
                per_fold.loc[per_fold["candidate"].eq(name)].set_index("fold")[
                    "pooled_row_rmse"
                ]
                < per_fold.loc[per_fold["candidate"].eq("anchor_student")].set_index(
                    "fold"
                )["pooled_row_rmse"]
            ).sum()
        )
        for name in candidate_arrays
    }
    primary = "heel_quantile_posterior"
    summary = {
        "profile": "quick",
        "scope": "exact historical 100-well five-fold quick OOF cohort",
        "primary_candidate": primary,
        "baseline_candidate": "anchor_student",
        "oof_wells": int(oof["well_id"].nunique()),
        "oof_rows": len(oof),
        "metrics": overall,
        "delta_vs_anchor_student": overall[primary] - overall["anchor_student"],
        "tail": tail,
        "folds_improved": fold_direction,
        "three_distinct_families": {
            "heel_calibrated_distribution_registration": [
                "heel_quantile_posterior",
                "heel_hard_map",
                "heel_wrong_typewell",
            ],
            "physics_h_space_extrapolation": ["h_space_line", "h_space_blend_005"],
            "structured_low_rank_anchor_shape": [
                "low_rank_anchor_6",
                "low_rank_shuffled_targets",
            ],
        },
        "negative_controls": [
            "heel_wrong_typewell",
            "random_common_shift",
            "low_rank_shuffled_targets",
            "heel_hard_map",
        ],
        "distribution_order_invariance_maximum_absolute_loss_delta": invariant_maximum,
        "uses_validation_target_for_prediction": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    oof.to_csv(
        run_root / "oof_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    per_well.to_csv(run_root / "per_well_metrics.csv", index=False)
    per_fold.to_csv(run_root / "per_fold_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        run_root / "registration_diagnostics.csv", index=False
    )
    _atomic_json(run_root / "summary.json", summary)
    _atomic_json(run_root / "paired_bootstrap.json", bootstrap)
    _atomic_json(run_root / "wrong_typewell_pairing.json", wrong_pairing)
    _atomic_json(
        run_root / "run_config.json",
        {
            "data_root": str(data),
            "baseline_root": str(baseline_path),
            "fold_config": str(Path(fold_config).resolve()),
            "fold_config_sha256": _sha256(Path(fold_config)),
            "fold_version": str(mapping["fold_version"].iat[0]),
            "cache_root": str(Path(cache_root).resolve()),
            "seed": seed,
            "workers": workers,
            "registration": heel_registration_manifest(),
            "low_rank": {
                "basis": "QR-orthonormalized polynomial degree 0..5 at fixed anchors",
                "estimator": "ExtraTrees 400/depth8/min_leaf5/max_features0.5",
            },
            "h_space": "Huber-IRLS line on final 250 ft of visible TVT_input+Z",
            "fold_provenance": fold_provenance,
        },
    )
    after_data = _data_manifest(data)
    artifact_audit = {
        "passed": bool(
            before_data == after_data
            and len(oof) == 489_284
            and oof["well_id"].nunique() == 100
            and not oof.duplicated(["well_id", "row_index"]).any()
            and np.isfinite(oof.select_dtypes(include=[np.number])).all().all()
            and preparation["raw_target_vs_baseline_maximum_absolute_delta"] == 0.0
            and invariant_maximum == 0.0
        ),
        "data_read_only_manifest_before": before_data,
        "data_read_only_manifest_after": after_data,
        "preparation": preparation,
        "duplicate_keys": int(oof.duplicated(["well_id", "row_index"]).sum()),
        "unexpected_nonfinite": int(
            np.size(oof.select_dtypes(include=[np.number]).to_numpy())
            - np.isfinite(oof.select_dtypes(include=[np.number]).to_numpy()).sum()
        ),
        "submission_csv_created": False,
    }
    if not artifact_audit["passed"]:
        raise ValueError("quick external-anchor artifact audit failed")
    _atomic_json(run_root / "artifact_audit.json", artifact_audit)
    _atomic_json(
        run_root / "timings.json",
        {"total_seconds": time.perf_counter() - started},
    )
    _write_compact_report(run_root / "report.md", summary)
    return {"run_root": run_root, "summary": summary, "artifact_audit": artifact_audit}


def _load_canonical_state(path: Path) -> SparseAnchorStudentState:
    with Path(path).open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, SparseAnchorStudentState):
        raise TypeError(f"canonical state has wrong type: {path}")
    return state


def _fit_full_states(
    *,
    mapping: pd.DataFrame,
    payloads: dict[str, SparseAnchorInferenceWell],
    targets: dict[str, np.ndarray],
    baseline_root: Path,
    run_root: Path,
    seeds: Sequence[int],
    workers: int,
    resume: bool,
) -> tuple[
    dict[tuple[int, int], SparseAnchorStudentState],
    dict[str, Any],
    dict[str, str],
]:
    states: dict[tuple[int, int], SparseAnchorStudentState] = {}
    provenance: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for fold in range(FOLD_COUNT):
        training_ids, validation_ids = partition_well_ids(mapping, fold)
        canonical_path = Path(baseline_root) / "models" / f"fold_{fold}.pkl"
        canonical = _load_canonical_state(canonical_path)
        if canonical.fold != fold or canonical.training_well_ids != training_ids:
            raise ValueError(f"canonical fold provenance differs: fold {fold}")
        if canonical.student_state.training_well_ids != training_ids:
            raise ValueError(f"canonical student provenance differs: fold {fold}")
        hashes[f"canonical_fold_{fold}"] = _sha256(canonical_path)
        training = [
            TrainingWell(payloads[well_id], targets[well_id])
            for well_id in training_ids
        ]
        for seed in seeds:
            output = run_root / "models" / f"seed_{seed}" / f"fold_{fold}.pkl"
            if resume and output.is_file():
                with output.open("rb") as handle:
                    wrapped = pickle.load(handle)
                state = wrapped.base_state
                if (
                    state.fold != fold
                    or state.training_well_ids != training_ids
                    or state.student_state.random_seed != seed
                ):
                    raise ValueError(f"resumed state provenance differs: {seed}/{fold}")
            elif seed == PRIMARY_SEED:
                state = canonical
                wrapped = wrap_sparse_anchor_student_state(state)
                _atomic_pickle(output, wrapped)
            else:
                context = RunContext(
                    fold=fold,
                    seed=int(seed),
                    workers=workers,
                    work_dir=output.parent,
                )
                print(
                    f"fit heel-anchor student seed={seed} fold={fold} "
                    f"train_wells={len(training_ids)}",
                    flush=True,
                )
                student = fit_student_estimator(
                    training,
                    kind="extra_trees",
                    context=context,
                    extra_trees_estimators=160,
                )
                state = SparseAnchorStudentState(
                    p3_state=canonical.p3_state,
                    path_pca_state=canonical.path_pca_state,
                    student_state=student,
                    training_well_ids=training_ids,
                    fold=fold,
                )
                wrapped = wrap_sparse_anchor_student_state(state)
                _atomic_pickle(output, wrapped)
            states[(int(seed), fold)] = state
            hashes[f"seed_{seed}_fold_{fold}"] = _sha256(output)
            provenance[f"seed_{seed}_fold_{fold}"] = {
                **heel_registered_state_provenance(
                    wrap_sparse_anchor_student_state(state)
                ),
                "validation_well_count": len(validation_ids),
                "validation_well_ids": list(validation_ids),
                "training_validation_intersection": sorted(
                    set(training_ids) & set(validation_ids)
                ),
                "p3_path_states_reused_from_canonical_fold": True,
                "student_refit_on_outer_training_wells": seed != PRIMARY_SEED,
            }
        del training, canonical
        gc.collect()
    return states, provenance, hashes


def _determinism_refit_audit(
    *,
    mapping: pd.DataFrame,
    payloads: dict[str, SparseAnchorInferenceWell],
    targets: dict[str, np.ndarray],
    reference: SparseAnchorStudentState,
    workers: int,
    work_dir: Path,
) -> dict[str, Any]:
    fold = 0
    seed = 20260712
    training_ids, validation_ids = partition_well_ids(mapping, fold)
    training = [
        TrainingWell(payloads[well_id], targets[well_id]) for well_id in training_ids
    ]
    context = RunContext(fold=fold, seed=seed, workers=workers, work_dir=work_dir)
    repeated = fit_student_estimator(
        training,
        kind="extra_trees",
        context=context,
        extra_trees_estimators=160,
    )
    maximum = 0.0
    for well_id in validation_ids:
        first = predict_anchor_delta(reference.student_state, payloads[well_id])
        second = predict_anchor_delta(repeated, payloads[well_id])
        maximum = max(maximum, float(np.max(np.abs(first - second))))
    tolerance = 1e-12
    return {
        "seed": seed,
        "fold": fold,
        "validation_wells": len(validation_ids),
        "maximum_absolute_anchor_delta_between_repeated_fits": maximum,
        "bitwise_deterministic": maximum == 0.0,
        "numerically_deterministic_tolerance": tolerance,
        "numerically_deterministic": maximum <= tolerance,
    }


def _visible_test_audit(
    *,
    data_root: Path,
    cache_root: Path,
    states: dict[tuple[int, int], SparseAnchorStudentState],
    seeds: Sequence[int],
) -> dict[str, Any]:
    model = HeelRegisteredAnchorStudentModel()
    files = list(enumerate_wells(data_root, "test"))
    blocks: list[pd.DataFrame] = []
    well_summaries: list[dict[str, Any]] = []
    for file in files:
        prepared = model.prepare(file, split="test", cache_root=cache_root)
        predictions: list[np.ndarray] = []
        shifts: list[float] = []
        for seed in seeds:
            for fold in range(FOLD_COUNT):
                wrapped = model.from_base_state(states[(int(seed), fold)])
                components = model.predict_components(wrapped, prepared)
                predictions.append(components.final)
                shifts.append(components.posterior_mean_shift)
        matrix = np.stack(predictions)
        ensemble = np.mean(matrix, axis=0)
        if not np.isfinite(matrix).all() or not np.isfinite(ensemble).all():
            raise ValueError(f"visible-test inference is non-finite: {file.well_id}")
        blocks.append(
            pd.DataFrame(
                {
                    "id": [f"{file.well_id}_{int(row)}" for row in prepared.row_index],
                    "tvt": ensemble,
                }
            )
        )
        well_summaries.append(
            {
                "well_id": file.well_id,
                "rows": len(ensemble),
                "minimum_prediction": float(np.min(ensemble)),
                "maximum_prediction": float(np.max(ensemble)),
                "mean_registration_shift": float(np.mean(shifts)),
                "maximum_absolute_registration_shift": float(np.max(np.abs(shifts))),
            }
        )
    predictions = pd.concat(blocks, ignore_index=True)
    sample = pd.read_csv(
        Path(data_root) / "sample_submission.csv", dtype={"id": "string"}
    )
    validate_sample_submission_contract(sample, Path(data_root) / "test")
    aligned = align_predictions_to_sample(sample, predictions)
    return {
        "performed": True,
        "dynamic_test_wells": [file.well_id for file in files],
        "test_rows": len(aligned),
        "finite": bool(np.isfinite(aligned["tvt"]).all()),
        "sample_id_order_aligned": aligned["id"].tolist() == sample["id"].tolist(),
        "prediction_sha256": hashlib.sha256(
            aligned["tvt"].to_numpy(dtype=np.float64).tobytes()
        ).hexdigest(),
        "ensemble_states": len(seeds) * FOLD_COUNT,
        "per_well": well_summaries,
        "submission_csv_created": False,
    }


def _finalize_existing_full_run(
    *,
    run_root: Path,
    data_root: Path,
    baseline_root: Path,
) -> dict[str, Any]:
    """Finish a run whose expensive OOF artifacts survived a final audit failure."""
    required = (
        "summary.json",
        "oof_predictions.csv.gz",
        "per_well_metrics.csv",
        "per_fold_metrics.csv",
        "paired_bootstrap.json",
        "leakage_check.json",
        "model_manifest.json",
        "run_config.json",
        "test_inference_audit.json",
    )
    if not all((run_root / name).is_file() for name in required):
        raise ValueError("existing run does not contain all finalization inputs")
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    seeds = tuple(map(int, summary["seeds"]))
    if seeds != FULL_SEEDS:
        raise ValueError("existing full run seed contract differs")
    per_well = pd.read_csv(run_root / "per_well_metrics.csv")
    per_fold = pd.read_csv(run_root / "per_fold_metrics.csv")
    bootstrap = _full_bootstrap_bundle(per_well, seeds)
    incumbent_ci = bootstrap["vs_historical_anchor_student"]["candidates"][
        "heel_registered_ensemble"
    ]
    matched_ci = bootstrap["matched_registration_ensemble"]["candidates"][
        "heel_registered_ensemble"
    ]
    metrics = summary["metrics"]
    matched_base = per_fold.loc[
        per_fold["candidate"].eq("anchor_seed_ensemble")
    ].set_index("fold")["pooled_row_rmse"]
    matched_candidate = per_fold.loc[
        per_fold["candidate"].eq("heel_registered_ensemble")
    ].set_index("fold")["pooled_row_rmse"]
    summary["primary_bootstrap_vs_historical_anchor_student"] = incumbent_ci
    summary.pop("primary_bootstrap", None)
    summary["matched_registration_delta_vs_anchor_seed_ensemble"] = float(
        metrics["heel_registered_ensemble"] - metrics["anchor_seed_ensemble"]
    )
    summary["matched_registration_bootstrap"] = matched_ci
    summary["matched_registration_fold_direction"] = {
        "improved_folds": int((matched_candidate < matched_base).sum()),
        "deltas": {
            str(fold): float(matched_candidate.loc[fold] - matched_base.loc[fold])
            for fold in range(FOLD_COUNT)
        },
    }
    tail = summary["tail"]
    summary["research_success"] = bool(
        metrics["heel_registered_ensemble"] < metrics["anchor_student"]
        and incumbent_ci["delta_rmse_q97.5"] < 0.0
        and metrics["heel_registered_ensemble"] < metrics["anchor_seed_ensemble"]
        and matched_ci["delta_rmse_q97.5"] < 0.0
        and tail["heel_registered_ensemble"]["wells_rmse_ge_40"]
        <= tail["anchor_student"]["wells_rmse_ge_40"]
        and tail["heel_registered_ensemble"]["wells_rmse_ge_60"]
        <= tail["anchor_student"]["wells_rmse_ge_60"]
        and summary["test_inference_capable"]
    )
    summary["finalized_after_tolerance_audit"] = True

    manifest = json.loads(
        (run_root / "model_manifest.json").read_text(encoding="utf-8")
    )
    determinism = manifest["determinism_refit"]
    tolerance = 1e-12
    determinism["numerically_deterministic_tolerance"] = tolerance
    determinism["numerically_deterministic"] = bool(
        determinism["maximum_absolute_anchor_delta_between_repeated_fits"] <= tolerance
    )
    leakage = json.loads((run_root / "leakage_check.json").read_text(encoding="utf-8"))
    test_audit = json.loads(
        (run_root / "test_inference_audit.json").read_text(encoding="utf-8")
    )
    config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    current_data = _data_manifest(data_root)
    baseline_audit = json.loads(
        (baseline_root / "artifact_audit.json").read_text(encoding="utf-8")
    )
    artifact_audit = {
        "passed": bool(
            summary["oof_rows"] == EXPECTED_FULL_ROWS
            and summary["oof_wells"] == EXPECTED_FULL_WELLS
            and summary["reference_seed_maximum_absolute_prediction_delta"] <= 1e-9
            and summary["anchor_row_exactness_maximum_absolute_delta"] <= 1e-9
            and determinism["numerically_deterministic"]
            and leakage["passed"]
            and test_audit["finite"]
            and test_audit["sample_id_order_aligned"]
            and baseline_audit["passed"]
            and current_data == config["data_manifest"]
            and not list(run_root.glob("*submission*.csv"))
        ),
        "rows": summary["oof_rows"],
        "wells": summary["oof_wells"],
        "data_read_only_manifest_before": config["data_manifest"],
        "data_read_only_manifest_after": current_data,
        "baseline_artifact_audit_passed": baseline_audit["passed"],
        "reference_seed_maximum_absolute_prediction_delta": summary[
            "reference_seed_maximum_absolute_prediction_delta"
        ],
        "anchor_row_exactness_maximum_absolute_delta": summary[
            "anchor_row_exactness_maximum_absolute_delta"
        ],
        "determinism_refit": determinism,
        "leakage_check_passed": leakage["passed"],
        "test_inference_audit": test_audit,
        "full_disk_reaudit_required": True,
        "submission_csv_created": False,
    }
    if not artifact_audit["passed"]:
        raise ValueError("existing full run failed corrected finalization audit")
    _atomic_json(run_root / "summary.json", summary)
    _atomic_json(run_root / "paired_bootstrap.json", bootstrap)
    _atomic_json(run_root / "model_manifest.json", manifest)
    _atomic_json(run_root / "artifact_audit.json", artifact_audit)
    _atomic_json(
        run_root / "timings.json",
        {
            "resumed_artifact_finalization": True,
            "expensive_oof_recomputed": False,
        },
    )
    _write_compact_report(run_root / "report.md", summary)
    return {"run_root": run_root, "summary": summary, "artifact_audit": artifact_audit}


def run_external_anchor_full_experiment(
    *,
    data_root: Path,
    fold_config: Path,
    output_root: Path,
    run_id: str,
    cache_root: Path,
    baseline_root: Path = Path(".runs/sparse-anchor-student-full"),
    workers: int = 4,
    seeds: Sequence[int] = FULL_SEEDS,
    resume: bool = False,
) -> dict[str, Any]:
    """Run strict full five-seed OOF, bootstrap, tail, and inference audits."""
    seeds = tuple(map(int, seeds))
    if seeds != FULL_SEEDS:
        raise ValueError(f"formal full seeds must be exactly {FULL_SEEDS}")
    configure_cpu(workers=workers)
    started = time.perf_counter()
    data = Path(data_root).resolve()
    before_data = _data_manifest(data)
    run_root = _safe_run_root(output_root, run_id, data, resume=resume)
    baseline_path = Path(baseline_root).resolve()
    if (
        resume
        and (run_root / "oof_predictions.csv.gz").is_file()
        and (run_root / "summary.json").is_file()
    ):
        return _finalize_existing_full_run(
            run_root=run_root,
            data_root=data,
            baseline_root=baseline_path,
        )
    baseline_oof_path = baseline_path / "oof_predictions.csv.gz"
    oof = _read_baseline_oof(baseline_oof_path)
    if (
        len(oof) != EXPECTED_FULL_ROWS
        or oof["well_id"].nunique() != EXPECTED_FULL_WELLS
    ):
        raise ValueError("formal full baseline scope is not 773 wells / 3,783,989 rows")
    files = {well.well_id: well for well in enumerate_wells(data, "train")}
    if set(files) != set(oof["well_id"].astype(str)):
        raise ValueError("dynamic train wells differ from baseline OOF")
    mapping = load_fold_mapping(fold_config, expected_well_ids=files)
    assigned_fold = mapping.set_index("well_id")["fold"].astype(int).to_dict()
    for well_id, fold_values in oof.groupby("well_id", sort=False)["fold"]:
        values = fold_values.unique()
        if len(values) != 1 or int(values[0]) != assigned_fold[str(well_id)]:
            raise ValueError(f"OOF fold assignment differs: {well_id}")

    groups = {
        str(well_id): np.asarray(indices, dtype=np.int64)
        for well_id, indices in oof.groupby("well_id", sort=True).indices.items()
    }
    row_indices = {
        well_id: oof.iloc[indices]["row_index"].to_numpy(dtype=np.int64)
        for well_id, indices in groups.items()
    }
    target_reference = {
        well_id: oof.iloc[indices]["y_true"].to_numpy(dtype=np.float64)
        for well_id, indices in groups.items()
    }
    payloads, targets, preparation = _prepare_payloads_and_targets(
        files,
        files=files,
        row_indices=row_indices,
        cache_root=cache_root,
        known_targets=target_reference,
    )
    states, provenance, state_hashes = _fit_full_states(
        mapping=mapping,
        payloads=payloads,
        targets=targets,
        baseline_root=baseline_path,
        run_root=run_root,
        seeds=seeds,
        workers=workers,
        resume=resume,
    )
    determinism = _determinism_refit_audit(
        mapping=mapping,
        payloads=payloads,
        targets=targets,
        reference=states[(20260712, 0)],
        workers=workers,
        work_dir=run_root / "models" / "determinism_refit",
    )

    seed_base = np.empty((len(seeds), len(oof)), dtype=np.float64)
    seed_registered = np.empty((len(seeds), len(oof)), dtype=np.float64)
    diagnostic_rows: list[dict[str, Any]] = []
    anchor_rows: list[pd.DataFrame] = []
    reference_maximum_delta = 0.0
    anchor_exactness_maximum = 0.0
    for well_number, well_id in enumerate(sorted(groups)):
        indices = groups[well_id]
        fold = assigned_fold[well_id]
        payload = payloads[well_id]
        truth = targets[well_id]
        blend = oof.iloc[indices]["y_pred_blend"].to_numpy(dtype=np.float64)
        horizontal = load_horizontal_inference(files[well_id].horizontal_path)
        typewell = load_typewell_inference(files[well_id].typewell_path)
        true_delta = build_anchor_delta_targets(
            truth,
            last_known_tvt=payload.anchor,
            placement=payload.placement,
        )
        true_anchor = payload.anchor + true_delta
        for seed_index, seed in enumerate(seeds):
            state = states[(seed, fold)]
            delta = predict_anchor_delta(state.student_state, payload)
            predicted_anchor = payload.anchor + delta
            base = reconstruct_from_predicted_anchors(
                blend, predicted_anchor, payload.placement
            )
            if seed == PRIMARY_SEED:
                reference = oof.iloc[indices][BASELINE_COLUMN].to_numpy(
                    dtype=np.float64
                )
                reference_maximum_delta = max(
                    reference_maximum_delta,
                    float(np.max(np.abs(base - reference))),
                )
            result = estimate_heel_registration(horizontal, typewell, base)
            final = base + result.posterior_mean_shift
            shifted_anchor = predicted_anchor + result.posterior_mean_shift
            seed_base[seed_index, indices] = base
            seed_registered[seed_index, indices] = final
            anchor_exactness_maximum = max(
                anchor_exactness_maximum,
                float(
                    np.max(
                        np.abs(final[payload.placement.local_indices] - shifted_anchor)
                    )
                ),
            )
            diagnostic_rows.append(
                _registration_diagnostics(result, well_id=well_id, fold=fold, seed=seed)
            )
            anchor_rows.append(
                pd.DataFrame(
                    {
                        "well_id": well_id,
                        "fold": fold,
                        "seed": seed,
                        "candidate": f"heel_registered_seed_{seed}",
                        "anchor_order": np.arange(1, len(ANCHOR_FRACTIONS) + 1),
                        "fraction": ANCHOR_FRACTIONS,
                        "local_index": payload.placement.local_indices,
                        "row_index": payload.placement.row_indices,
                        "true_anchor_tvt": true_anchor,
                        "predicted_anchor_tvt": shifted_anchor,
                    }
                )
            )
        # Baseline-anchor records are needed for direct decomposition.
        reference_delta = predict_anchor_delta(
            states[(PRIMARY_SEED, fold)].student_state, payload
        )
        anchor_rows.append(
            pd.DataFrame(
                {
                    "well_id": well_id,
                    "fold": fold,
                    "seed": PRIMARY_SEED,
                    "candidate": "anchor_student",
                    "anchor_order": np.arange(1, len(ANCHOR_FRACTIONS) + 1),
                    "fraction": ANCHOR_FRACTIONS,
                    "local_index": payload.placement.local_indices,
                    "row_index": payload.placement.row_indices,
                    "true_anchor_tvt": true_anchor,
                    "predicted_anchor_tvt": payload.anchor + reference_delta,
                }
            )
        )
        if (well_number + 1) % 100 == 0:
            running = np.arange(len(oof)) < 0  # avoids retaining partial indices
            del running
            print(f"full registered inference wells={well_number + 1}", flush=True)

    if not np.isfinite(seed_base).all() or not np.isfinite(seed_registered).all():
        raise ValueError("full seed predictions contain non-finite values")
    base_ensemble = np.mean(seed_base, axis=0)
    registered_ensemble = np.mean(seed_registered, axis=0)
    oof["y_pred_anchor_student"] = oof[BASELINE_COLUMN].to_numpy(dtype=np.float64)
    for index, seed in enumerate(seeds):
        oof[f"y_pred_anchor_seed_{seed}"] = seed_base[index]
        oof[f"y_pred_heel_registered_seed_{seed}"] = seed_registered[index]
    oof["y_pred_anchor_seed_ensemble"] = base_ensemble
    oof[PRIMARY_COLUMN] = registered_ensemble

    anchor_oof = pd.concat(anchor_rows, ignore_index=True)
    ensemble_anchor = (
        anchor_oof.loc[anchor_oof["candidate"].str.startswith("heel_registered_seed_")]
        .groupby(
            ["well_id", "fold", "anchor_order", "fraction", "local_index", "row_index"],
            as_index=False,
        )
        .agg(
            true_anchor_tvt=("true_anchor_tvt", "first"),
            predicted_anchor_tvt=("predicted_anchor_tvt", "mean"),
        )
    )
    ensemble_anchor["seed"] = -1
    ensemble_anchor["candidate"] = "heel_registered_ensemble"
    anchor_oof = pd.concat(
        [anchor_oof, ensemble_anchor], ignore_index=True
    ).sort_values(["candidate", "fold", "well_id", "anchor_order"], kind="stable")

    prediction_columns = {
        "anchor_student": "y_pred_anchor_student",
        "anchor_seed_ensemble": "y_pred_anchor_seed_ensemble",
        "heel_registered_ensemble": PRIMARY_COLUMN,
    }
    for seed in seeds:
        prediction_columns[f"anchor_seed_{seed}"] = f"y_pred_anchor_seed_{seed}"
        prediction_columns[f"heel_registered_seed_{seed}"] = (
            f"y_pred_heel_registered_seed_{seed}"
        )
    if reference_maximum_delta > 1e-9:
        raise ValueError(
            "canonical seed reconstruction differs from historical Anchor Student: "
            f"{reference_maximum_delta}"
        )
    per_well, per_fold, overall = _metric_tables(
        oof, prediction_columns, baseline="anchor_student"
    )
    bootstrap = _full_bootstrap_bundle(per_well, seeds)
    tail = {name: _tail_summary(per_well, name) for name in prediction_columns}
    decomposition = {
        "anchor_student": _anchor_decomposition(anchor_oof, "anchor_student"),
        "heel_registered_ensemble": _anchor_decomposition(
            anchor_oof, "heel_registered_ensemble"
        ),
    }
    per_anchor_rows: list[dict[str, Any]] = []
    for (candidate, order, fraction), group in anchor_oof.groupby(
        ["candidate", "anchor_order", "fraction"], sort=True
    ):
        error = group["predicted_anchor_tvt"].to_numpy(dtype=np.float64) - group[
            "true_anchor_tvt"
        ].to_numpy(dtype=np.float64)
        per_anchor_rows.append(
            {
                "candidate": candidate,
                "anchor_order": int(order),
                "fraction": float(fraction),
                "wells": len(group),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "mae": float(np.mean(np.abs(error))),
            }
        )
    per_anchor = pd.DataFrame(per_anchor_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    risk_frame, risk_analysis = _risk_signal_analysis(
        diagnostics,
        per_well,
        primary="heel_registered_ensemble",
        baseline="anchor_student",
    )
    test_audit = _visible_test_audit(
        data_root=data,
        cache_root=Path(cache_root),
        states=states,
        seeds=seeds,
    )
    after_data = _data_manifest(data)
    fold_directions = {}
    baseline_folds = per_fold.loc[per_fold["candidate"].eq("anchor_student")].set_index(
        "fold"
    )["pooled_row_rmse"]
    for candidate in prediction_columns:
        candidate_folds = per_fold.loc[per_fold["candidate"].eq(candidate)].set_index(
            "fold"
        )["pooled_row_rmse"]
        fold_directions[candidate] = {
            "improved_folds": int((candidate_folds < baseline_folds).sum()),
            "deltas": {
                str(fold): float(candidate_folds.loc[fold] - baseline_folds.loc[fold])
                for fold in range(FOLD_COUNT)
            },
        }
    matched_base_folds = per_fold.loc[
        per_fold["candidate"].eq("anchor_seed_ensemble")
    ].set_index("fold")["pooled_row_rmse"]
    matched_registered_folds = per_fold.loc[
        per_fold["candidate"].eq("heel_registered_ensemble")
    ].set_index("fold")["pooled_row_rmse"]
    matched_fold_direction = {
        "improved_folds": int((matched_registered_folds < matched_base_folds).sum()),
        "deltas": {
            str(fold): float(
                matched_registered_folds.loc[fold] - matched_base_folds.loc[fold]
            )
            for fold in range(FOLD_COUNT)
        },
    }
    seed_metrics = [overall[f"heel_registered_seed_{seed}"] for seed in seeds]
    primary = "heel_registered_ensemble"
    incumbent_ci = bootstrap["vs_historical_anchor_student"]["candidates"][primary]
    matched_ci = bootstrap["matched_registration_ensemble"]["candidates"][primary]
    success = bool(
        overall[primary] < overall["anchor_student"]
        and incumbent_ci["delta_rmse_q97.5"] < 0.0
        and overall[primary] < overall["anchor_seed_ensemble"]
        and matched_ci["delta_rmse_q97.5"] < 0.0
        and tail[primary]["wells_rmse_ge_40"]
        <= tail["anchor_student"]["wells_rmse_ge_40"]
        and tail[primary]["wells_rmse_ge_60"]
        <= tail["anchor_student"]["wells_rmse_ge_60"]
        and test_audit["finite"]
        and test_audit["sample_id_order_aligned"]
    )
    summary = {
        "profile": "full",
        "scope": "full strict five-fold well-level OOF",
        "research_success": success,
        "primary_candidate": primary,
        "baseline_candidate": "anchor_student",
        "oof_wells": int(oof["well_id"].nunique()),
        "oof_rows": len(oof),
        "folds": list(range(FOLD_COUNT)),
        "seeds": list(seeds),
        "metrics": overall,
        "primary_delta_vs_anchor_student": overall[primary] - overall["anchor_student"],
        "primary_bootstrap_vs_historical_anchor_student": incumbent_ci,
        "matched_registration_delta_vs_anchor_seed_ensemble": overall[primary]
        - overall["anchor_seed_ensemble"],
        "matched_registration_bootstrap": matched_ci,
        "seed_sensitivity": {
            "registered_seed_metrics": {
                str(seed): overall[f"heel_registered_seed_{seed}"] for seed in seeds
            },
            "minimum": float(np.min(seed_metrics)),
            "maximum": float(np.max(seed_metrics)),
            "mean": float(np.mean(seed_metrics)),
            "standard_deviation": float(np.std(seed_metrics, ddof=1)),
            "ensemble": overall[primary],
        },
        "fold_direction": fold_directions,
        "matched_registration_fold_direction": matched_fold_direction,
        "tail": tail,
        "anchor_error_decomposition": decomposition,
        "reference_seed_maximum_absolute_prediction_delta": reference_maximum_delta,
        "anchor_row_exactness_maximum_absolute_delta": anchor_exactness_maximum,
        "uses_true_validation_target_for_prediction": False,
        "registration_has_learned_cross_well_state": False,
        "test_inference_capable": test_audit["finite"],
        "submission_created": False,
        "submission_uploaded": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    oof.to_csv(
        run_root / "oof_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    anchor_oof.to_csv(
        run_root / "anchor_oof_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    per_well.to_csv(run_root / "per_well_metrics.csv", index=False)
    per_fold.to_csv(run_root / "per_fold_metrics.csv", index=False)
    per_anchor.to_csv(run_root / "per_anchor_metrics.csv", index=False)
    diagnostics.to_csv(run_root / "registration_diagnostics.csv.gz", index=False)
    risk_frame.to_csv(run_root / "pre_prediction_risk_by_well.csv", index=False)
    _atomic_json(run_root / "summary.json", summary)
    _atomic_json(run_root / "paired_bootstrap.json", bootstrap)
    _atomic_json(run_root / "pre_prediction_risk_analysis.json", risk_analysis)
    _atomic_json(
        run_root / "leakage_check.json",
        {
            "passed": all(
                not item["training_validation_intersection"]
                for item in provenance.values()
            ),
            "horizontal_inference_columns": [
                "MD",
                "X",
                "Y",
                "Z",
                "GR",
                "TVT_input",
            ],
            "typewell_inference_columns": ["TVT", "GR"],
            "validation_target_used_for_prepare": False,
            "validation_target_used_for_registration": False,
            "validation_target_used_for_prediction_or_gating": False,
            "student_fit_scope": "outer-training wells only",
            "registration_fit_scope": "none; deterministic within-well inference",
            "fold_provenance": provenance,
        },
    )
    _atomic_json(
        run_root / "model_manifest.json",
        {
            **heel_registered_anchor_student_manifest(),
            "seeds": list(seeds),
            "test_ensemble": "mean of five seeds x five outer-fold states",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "state_sha256": state_hashes,
            "determinism_refit": determinism,
        },
    )
    _atomic_json(
        run_root / "run_config.json",
        {
            "data_root": str(data),
            "fold_config": str(Path(fold_config).resolve()),
            "cache_root": str(Path(cache_root).resolve()),
            "baseline_root": str(baseline_path),
            "baseline_oof_sha256": _sha256(baseline_oof_path),
            "seeds": list(seeds),
            "workers": workers,
            "registration": heel_registration_manifest(),
            "p3_path_reuse": (
                "fixed canonical fold states; fitted only on each outer-training fold"
            ),
            "student_refit": "ExtraTrees 160/depth8/min_leaf5/max_features0.5",
            "data_manifest": before_data,
        },
    )
    _atomic_json(run_root / "test_inference_audit.json", test_audit)
    artifact_audit = {
        "passed": bool(
            before_data == after_data
            and len(oof) == EXPECTED_FULL_ROWS
            and oof["well_id"].nunique() == EXPECTED_FULL_WELLS
            and not oof.duplicated(["well_id", "row_index"]).any()
            and np.isfinite(oof.select_dtypes(include=[np.number])).all().all()
            and preparation["raw_target_vs_baseline_maximum_absolute_delta"] == 0.0
            and reference_maximum_delta <= 1e-9
            and anchor_exactness_maximum <= 1e-9
            and determinism["numerically_deterministic"]
            and all(
                not item["training_validation_intersection"]
                for item in provenance.values()
            )
            and test_audit["finite"]
            and test_audit["sample_id_order_aligned"]
            and not list(run_root.glob("*submission*.csv"))
        ),
        "rows": len(oof),
        "wells": int(oof["well_id"].nunique()),
        "duplicate_row_keys": int(oof.duplicated(["well_id", "row_index"]).sum()),
        "unexpected_nonfinite": int(
            np.size(oof.select_dtypes(include=[np.number]).to_numpy())
            - np.isfinite(oof.select_dtypes(include=[np.number]).to_numpy()).sum()
        ),
        "data_read_only_manifest_before": before_data,
        "data_read_only_manifest_after": after_data,
        "preparation": preparation,
        "reference_seed_maximum_absolute_prediction_delta": reference_maximum_delta,
        "anchor_row_exactness_maximum_absolute_delta": anchor_exactness_maximum,
        "determinism_refit": determinism,
        "test_inference_audit": test_audit,
        "submission_csv_created": False,
    }
    if not artifact_audit["passed"]:
        raise ValueError("full external-anchor artifact audit failed")
    _atomic_json(run_root / "artifact_audit.json", artifact_audit)
    _atomic_json(
        run_root / "timings.json",
        {"total_seconds": time.perf_counter() - started},
    )
    _write_compact_report(run_root / "report.md", summary)
    return {"run_root": run_root, "summary": summary, "artifact_audit": artifact_audit}


__all__ = [
    "BOOTSTRAP_SEED",
    "FULL_SEEDS",
    "PRIMARY_SEED",
    "run_external_anchor_full_experiment",
    "run_external_anchor_quick_experiment",
]
