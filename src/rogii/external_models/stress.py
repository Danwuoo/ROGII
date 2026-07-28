"""Inference-only well descriptors for target-free stability stress tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import (
    enumerate_wells,
    load_horizontal_inference,
    load_typewell_inference,
    validate_prefix,
)
from ..portfolio_metrics import (
    make_target_free_strata,
    target_free_stratified_metrics,
)
from .artifacts import (
    INCUMBENT_COLUMN,
    OofCandidateSchema,
    read_external_artifact_bundle,
)


TARGET_FREE_FEATURE_COLUMNS = (
    "horizontal_md_length",
    "prediction_md_length",
    "prediction_rows",
    "sampling_density_rows_per_md",
    "known_prefix_fraction",
    "trajectory_path_length",
    "trajectory_tortuosity",
    "inclination_mean_rad",
    "inclination_p90_rad",
    "horizontal_gr_mean",
    "horizontal_gr_std",
    "horizontal_gr_iqr",
    "horizontal_gr_spectral_centroid",
    "horizontal_gr_high_frequency_share",
    "horizontal_gr_observed_fraction",
    "typewell_tvt_span",
    "typewell_rows",
    "typewell_gr_mean",
    "typewell_gr_std",
    "typewell_gr_iqr",
    "typewell_gr_spectral_centroid",
    "typewell_gr_high_frequency_share",
    "registration_posterior_shift_std_mean",
    "registration_posterior_entropy_mean",
    "registration_maximum_posterior_mean",
    "registration_edge_mass_mean",
    "registration_posterior_coverage_mean",
    "registration_prefix_coverage_mean",
    "registration_prefix_mse_mean",
    "registration_shift_across_seed_std",
)
EXTERNAL_STRESS_SCHEMA_VERSION = "rogii_external_target_free_stress_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def _finite_increasing(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 1
        or len(result) < 2
        or not np.isfinite(result).all()
        or np.any(np.diff(result) <= 0.0)
    ):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return result


def _filled(
    values: Any, coordinate: np.ndarray, *, name: str
) -> tuple[np.ndarray, float]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != coordinate.shape or np.isinf(result).any():
        raise ValueError(
            f"{name} must align with its coordinate and contain no infinity"
        )
    observed = np.isfinite(result)
    if not observed.any():
        raise ValueError(f"{name} has no finite observations")
    return (
        np.interp(coordinate, coordinate[observed], result[observed]),
        float(np.mean(observed)),
    )


def _spectral(values: np.ndarray) -> tuple[float, float]:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    power = np.square(np.abs(np.fft.rfft(centered)))
    frequency = np.fft.rfftfreq(len(centered))
    if len(power):
        power[0] = 0.0
    total = float(np.sum(power))
    if total <= np.finfo(np.float64).eps:
        return 0.0, 0.0
    centroid = float(np.sum(frequency * power) / total)
    high_frequency_share = float(np.sum(power[frequency >= 0.25]) / total)
    return centroid, high_frequency_share


def _trajectory(
    md: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[float, float, float, float]:
    xyz = np.column_stack([x, y, z])
    segment = np.diff(xyz, axis=0)
    path_length = float(np.sum(np.linalg.norm(segment, axis=1)))
    displacement = float(np.linalg.norm(xyz[-1] - xyz[0]))
    tortuosity = path_length / max(displacement, np.finfo(np.float64).eps)
    dx = np.gradient(x, md, edge_order=2 if len(md) >= 3 else 1)
    dy = np.gradient(y, md, edge_order=2 if len(md) >= 3 else 1)
    dz = np.gradient(z, md, edge_order=2 if len(md) >= 3 else 1)
    inclination = np.arctan2(
        np.hypot(dx, dy),
        np.maximum(np.abs(dz), np.finfo(np.float64).eps),
    )
    values = (
        path_length,
        tortuosity,
        float(np.mean(inclination)),
        float(np.quantile(inclination, 0.90)),
    )
    if not np.isfinite(values).all():
        raise ValueError("trajectory summaries are non-finite")
    return values


def _registration_features(
    path: Path,
    *,
    well_ids: Sequence[str],
) -> pd.DataFrame:
    columns = [
        "well_id",
        "fold",
        "seed",
        "posterior_mean_shift",
        "posterior_shift_std",
        "posterior_entropy",
        "maximum_posterior",
        "edge_mass",
        "posterior_mean_coverage",
        "prefix_coverage",
        "prefix_mse",
    ]
    frame = pd.read_csv(path, usecols=columns, dtype={"well_id": "string"})
    frame["well_id"] = frame["well_id"].astype(str)
    selected = frame.loc[frame["well_id"].isin(set(map(str, well_ids)))].copy()
    if (
        selected.empty
        or selected.duplicated(["well_id", "fold", "seed"]).any()
        or set(selected["well_id"]) != set(map(str, well_ids))
        or not np.isfinite(
            selected.drop(columns="well_id").to_numpy(dtype=np.float64)
        ).all()
    ):
        raise ValueError("registration diagnostics have invalid coverage or values")
    fold_count = selected.groupby("well_id")["fold"].nunique()
    seed_count = selected.groupby("well_id")["seed"].nunique()
    if not (fold_count == 1).all() or not (seed_count >= 1).all():
        raise ValueError(
            "registration diagnostics must have one fold and seeds per well"
        )
    grouped = selected.groupby("well_id", sort=True)
    return grouped.agg(
        registration_posterior_shift_std_mean=("posterior_shift_std", "mean"),
        registration_posterior_entropy_mean=("posterior_entropy", "mean"),
        registration_maximum_posterior_mean=("maximum_posterior", "mean"),
        registration_edge_mass_mean=("edge_mass", "mean"),
        registration_posterior_coverage_mean=("posterior_mean_coverage", "mean"),
        registration_prefix_coverage_mean=("prefix_coverage", "mean"),
        registration_prefix_mse_mean=("prefix_mse", "mean"),
        registration_shift_across_seed_std=(
            "posterior_mean_shift",
            lambda values: float(np.std(values.to_numpy(dtype=np.float64), ddof=0)),
        ),
    ).reset_index()


def build_target_free_well_features(
    data_root: Path,
    well_ids: Sequence[str],
    *,
    registration_diagnostics: Path,
) -> pd.DataFrame:
    """Build one inference-safe descriptor row per requested training well."""

    requested = tuple(sorted(map(str, well_ids)))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("well_ids must be non-empty and unique")
    train = {well.well_id: well for well in enumerate_wells(Path(data_root), "train")}
    missing = set(requested) - set(train)
    if missing:
        raise ValueError(f"unknown training wells: {sorted(missing)[:5]}")
    rows: list[dict[str, Any]] = []
    for well_id in requested:
        well = train[well_id]
        horizontal = load_horizontal_inference(well.horizontal_path)
        typewell = load_typewell_inference(well.typewell_path)
        known_count, prediction_rows = validate_prefix(horizontal["TVT_input"])
        md = _finite_increasing(horizontal["MD"], name=f"{well_id} horizontal MD")
        x, _ = _filled(horizontal["X"], md, name=f"{well_id} X")
        y, _ = _filled(horizontal["Y"], md, name=f"{well_id} Y")
        z, _ = _filled(horizontal["Z"], md, name=f"{well_id} Z")
        horizontal_gr, horizontal_observed = _filled(
            horizontal["GR"],
            md,
            name=f"{well_id} horizontal GR",
        )
        typewell_tvt = _finite_increasing(
            typewell["TVT"],
            name=f"{well_id} typewell TVT",
        )
        typewell_gr, _ = _filled(
            typewell["GR"],
            typewell_tvt,
            name=f"{well_id} typewell GR",
        )
        trajectory = _trajectory(md, x, y, z)
        horizontal_spectral = _spectral(horizontal_gr)
        typewell_spectral = _spectral(typewell_gr)
        prediction_md_length = float(md[-1] - md[known_count - 1])
        if prediction_md_length <= 0.0:
            raise ValueError(f"{well_id} prediction MD length is not positive")
        rows.append(
            {
                "well_id": well_id,
                "horizontal_md_length": float(md[-1] - md[0]),
                "prediction_md_length": prediction_md_length,
                "prediction_rows": int(prediction_rows),
                "sampling_density_rows_per_md": prediction_rows / prediction_md_length,
                "known_prefix_fraction": known_count / len(horizontal),
                "trajectory_path_length": trajectory[0],
                "trajectory_tortuosity": trajectory[1],
                "inclination_mean_rad": trajectory[2],
                "inclination_p90_rad": trajectory[3],
                "horizontal_gr_mean": float(np.mean(horizontal_gr)),
                "horizontal_gr_std": float(np.std(horizontal_gr)),
                "horizontal_gr_iqr": float(
                    np.quantile(horizontal_gr, 0.75) - np.quantile(horizontal_gr, 0.25)
                ),
                "horizontal_gr_spectral_centroid": horizontal_spectral[0],
                "horizontal_gr_high_frequency_share": horizontal_spectral[1],
                "horizontal_gr_observed_fraction": horizontal_observed,
                "typewell_tvt_span": float(typewell_tvt[-1] - typewell_tvt[0]),
                "typewell_rows": int(len(typewell)),
                "typewell_gr_mean": float(np.mean(typewell_gr)),
                "typewell_gr_std": float(np.std(typewell_gr)),
                "typewell_gr_iqr": float(
                    np.quantile(typewell_gr, 0.75) - np.quantile(typewell_gr, 0.25)
                ),
                "typewell_gr_spectral_centroid": typewell_spectral[0],
                "typewell_gr_high_frequency_share": typewell_spectral[1],
            }
        )
    base = pd.DataFrame(rows)
    registration = _registration_features(
        registration_diagnostics,
        well_ids=requested,
    )
    result = (
        base.merge(
            registration,
            on="well_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("well_id", kind="stable")
        .reset_index(drop=True)
    )
    if (
        tuple(column for column in result.columns if column != "well_id")
        != TARGET_FREE_FEATURE_COLUMNS
        or not np.isfinite(
            result.loc[:, list(TARGET_FREE_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        ).all()
    ):
        raise ValueError("target-free well feature schema or finiteness differs")
    return result


def analyze_external_bundle_stress(
    run_root: Path,
    *,
    bins: int = 5,
    candidate_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Re-read a canonical bundle and persist target-free quintile diagnostics."""

    root = Path(run_root).resolve()
    bundle = read_external_artifact_bundle(root)
    candidate_records = tuple(
        OofCandidateSchema.from_dict(value)
        for value in bundle.manifest["oof"]["candidates"]
    )
    if candidate_ids is None:
        selected = tuple(
            candidate
            for candidate in candidate_records
            if candidate.eligible_for_model_selection
        )
    else:
        requested = set(map(str, candidate_ids))
        selected = tuple(
            candidate
            for candidate in candidate_records
            if candidate.candidate_id in requested
        )
        if {candidate.candidate_id for candidate in selected} != requested:
            raise ValueError("requested candidate IDs differ from the bundle")
    if not selected:
        raise ValueError("no stress-test candidates selected")
    well_ids = tuple(sorted(bundle.oof["well_id"].astype(str).unique()))
    incumbent_root = Path(
        bundle.manifest["incumbent"]["state_manifest"]["artifact_root"]
    )
    registration = incumbent_root / "registration_diagnostics.csv.gz"
    features = build_target_free_well_features(
        Path(bundle.manifest["data_root"]),
        well_ids,
        registration_diagnostics=registration,
    )
    strata = make_target_free_strata(
        features,
        TARGET_FREE_FEATURE_COLUMNS,
        bins=bins,
    )
    prediction_columns = (
        INCUMBENT_COLUMN,
        *(candidate.prediction_column for candidate in selected),
    )
    metrics = target_free_stratified_metrics(
        bundle.oof,
        strata,
        prediction_columns,
        baseline_column=INCUMBENT_COLUMN,
    )
    output_root = root / "stress"
    _atomic_csv(output_root / "target_free_well_features.csv", features)
    _atomic_csv(output_root / "target_free_strata.csv", strata)
    _atomic_csv(output_root / "target_free_stratified_metrics.csv", metrics)
    candidate_metrics = metrics.loc[~metrics["candidate"].eq(INCUMBENT_COLUMN)]
    summary = {
        "schema_version": EXTERNAL_STRESS_SCHEMA_VERSION,
        "completed": True,
        "run_root": str(root),
        "rows": int(len(bundle.oof)),
        "wells": int(len(well_ids)),
        "bins": int(bins),
        "feature_columns": list(TARGET_FREE_FEATURE_COLUMNS),
        "candidate_ids": [candidate.candidate_id for candidate in selected],
        "registration_diagnostics": str(registration.resolve()),
        "registration_diagnostics_sha256": _sha256_file(registration),
        "data_root": str(Path(bundle.manifest["data_root"]).resolve()),
        "feature_sha256": _sha256_file(output_root / "target_free_well_features.csv"),
        "strata_sha256": _sha256_file(output_root / "target_free_strata.csv"),
        "metrics_sha256": _sha256_file(
            output_root / "target_free_stratified_metrics.csv"
        ),
        "strata_rows": int(len(strata)),
        "metric_rows": int(len(metrics)),
        "candidate_worst_delta": {
            str(candidate): float(group["delta_rmse_vs_baseline"].max())
            for candidate, group in candidate_metrics.groupby("candidate", sort=True)
        },
        "candidate_best_delta": {
            str(candidate): float(group["delta_rmse_vs_baseline"].min())
            for candidate, group in candidate_metrics.groupby("candidate", sort=True)
        },
        "target_columns_used_for_strata": [],
        "prediction_columns_used_for_strata": [],
    }
    _atomic_json(output_root / "summary.json", summary)
    return summary


__all__ = [
    "EXTERNAL_STRESS_SCHEMA_VERSION",
    "TARGET_FREE_FEATURE_COLUMNS",
    "analyze_external_bundle_stress",
    "build_target_free_well_features",
]
