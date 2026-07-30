"""Run target-free external geometry and strict matched safe-physics OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rogii.data import (
    enumerate_wells,
    load_horizontal_inference,
    load_typewell_inference,
)
from rogii.external_models.artifacts import capture_data_manifest
from rogii.external_models.incumbent import HeelIncumbentProvider
from rogii.external_predictions import (
    audit_external_prediction_files,
    external_prediction_geometry,
)
from rogii.folds import FOLD_COUNT, load_fold_mapping
from rogii.nested_blend import cross_fit_component_shrinkage
from rogii.physics_alignment import (
    CONTROL_GR_ORDER_SHUFFLE,
    CONTROL_NONE,
    CONTROL_NO_CROSS_INTERACTION,
    CONTROL_NO_PREFIX_ANCHOR,
    CONTROL_PAIR_SHUFFLE,
    CONTROL_PHASE_DESTROY,
    CONTROL_WRONG_TYPEWELL,
    METHOD_BEAM,
    METHOD_NCC,
    METHOD_PARTICLE_FILTER,
    SafeAlignmentPrediction,
    SafePhysicsAlignmentConfig,
    SafePhysicsAlignmentFamily,
)
from rogii.portfolio_metrics import (
    common_slope_shape_metrics,
    paired_well_cluster_bootstrap,
    per_fold_metrics,
    per_well_metrics,
    per_well_win_summary,
    pooled_metrics,
    residual_correlation,
    shadow_private_subset_simulation,
    tail_metrics,
    validate_oof_frame,
)


INCUMBENT_COLUMN = "incumbent"
FULL_INCUMBENT_SOURCE_COLUMN = "y_pred_heel_registered_ensemble"
EXPECTED_INCUMBENT_RMSE = 14.170847892501936
DEFAULT_SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)
PRIMARY_CORRECT_CANDIDATE = "pf_multiscale"
NEGATIVE_CONTROL_CANDIDATES = (
    "pf_wrong_typewell",
    "pf_pair_shuffle",
    "pf_gr_order_shuffle",
    "pf_phase_destroy",
    "pf_no_prefix_anchor",
    "pf_no_cross_interaction",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quick_validation_ids(
    source_path: Path, mapping: pd.DataFrame
) -> dict[int, tuple[str, ...]]:
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    provenance = payload.get("fold_provenance")
    if payload.get("profile") != "quick" or not isinstance(provenance, dict):
        raise ValueError("quick cohort source is invalid")
    expected_fold = mapping.set_index("well_id")["fold"].astype(int).to_dict()
    selected: dict[int, tuple[str, ...]] = {}
    for fold in range(FOLD_COUNT):
        record = provenance.get(str(fold))
        if not isinstance(record, dict):
            raise ValueError(f"quick cohort lacks fold {fold}")
        well_ids = tuple(sorted(map(str, record.get("validation_wells", ()))))
        if (
            len(well_ids) != 20
            or len(set(well_ids)) != len(well_ids)
            or any(expected_fold.get(well_id) != fold for well_id in well_ids)
        ):
            raise ValueError(f"quick validation cohort mismatch for fold {fold}")
        selected[fold] = well_ids
    return selected


def _full_validation_ids(mapping: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    return {
        fold: tuple(
            sorted(
                mapping.loc[mapping["fold"].astype(int).eq(fold), "well_id"].astype(str)
            )
        )
        for fold in range(FOLD_COUNT)
    }


def _derangement(values: Sequence[str], *, seed: int) -> dict[str, str]:
    identifiers = tuple(sorted(map(str, values)))
    if len(identifiers) < 2:
        raise ValueError("a pairing control requires at least two wells")
    random = np.random.default_rng(int(seed))
    for _ in range(10_000):
        permuted = list(identifiers)
        random.shuffle(permuted)
        if all(
            left != right for left, right in zip(identifiers, permuted, strict=True)
        ):
            return dict(zip(identifiers, permuted, strict=True))
    raise RuntimeError("failed to construct deterministic derangement")


def _baseline_audit(
    *,
    oof_path: Path,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(
        oof_path,
        usecols=[
            "well_id",
            "row_index",
            "fold",
            "y_true",
            FULL_INCUMBENT_SOURCE_COLUMN,
        ],
        dtype={"well_id": "string"},
    ).rename(columns={FULL_INCUMBENT_SOURCE_COLUMN: INCUMBENT_COLUMN})
    expected_ids = tuple(sorted(mapping["well_id"].astype(str)))
    validation = validate_oof_frame(
        frame,
        [INCUMBENT_COLUMN],
        expected_well_ids=expected_ids,
        expected_folds=range(FOLD_COUNT),
    )
    pooled = pooled_metrics(frame, [INCUMBENT_COLUMN]).iloc[0].to_dict()
    rmse = float(pooled["pooled_row_rmse"])
    if abs(rmse - EXPECTED_INCUMBENT_RMSE) > 1e-10:
        raise ValueError(
            f"incumbent RMSE mismatch: {rmse} != {EXPECTED_INCUMBENT_RMSE}"
        )
    return frame, {
        "source_path": str(Path(oof_path).resolve()),
        "source_sha256": _sha256(oof_path),
        "validation": validation,
        "pooled": pooled,
        "expected_rmse": EXPECTED_INCUMBENT_RMSE,
        "absolute_rmse_delta": abs(rmse - EXPECTED_INCUMBENT_RMSE),
    }


def _prediction_record(
    result: SafeAlignmentPrediction,
    *,
    candidate: str,
    fold: int,
) -> dict[str, Any]:
    uncertainty = np.asarray(result.uncertainty, dtype=np.float64)
    ess = np.asarray(result.effective_particle_count, dtype=np.float64)
    return {
        "candidate": candidate,
        "well_id": result.well_id,
        "fold": int(fold),
        "paired_typewell_id": result.paired_typewell_id,
        "method": result.method,
        "control_id": result.control_id,
        "status": result.status,
        "reason": result.reason,
        "confidence": result.confidence,
        "prediction_rows": int(len(result.prediction)),
        "prediction_min": float(np.min(result.prediction)),
        "prediction_max": float(np.max(result.prediction)),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "maximum_uncertainty": float(np.max(uncertainty)),
        "mean_effective_particle_count": float(np.mean(ess)),
        "minimum_effective_particle_count": float(np.min(ess)),
        "metadata_json": json.dumps(_jsonable(result.metadata), sort_keys=True),
    }


def _candidate_result(
    *,
    family: SafePhysicsAlignmentFamily,
    candidate: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    incumbent: np.ndarray,
    well_id: str,
    paired_typewell_id: str,
) -> SafeAlignmentPrediction:
    settings: dict[str, dict[str, Any]] = {
        "ncc_multiscale": {
            "method": METHOD_NCC,
            "control_id": CONTROL_NONE,
            "multiscale": True,
        },
        "pf_single_scale": {
            "method": METHOD_PARTICLE_FILTER,
            "control_id": CONTROL_NONE,
            "multiscale": False,
        },
        "pf_multiscale": {
            "method": METHOD_PARTICLE_FILTER,
            "control_id": CONTROL_NONE,
            "multiscale": True,
        },
        "beam_single_path": {
            "method": METHOD_BEAM,
            "control_id": CONTROL_NONE,
            "multiscale": True,
            "beam_ensemble": False,
        },
        "beam_ensemble": {
            "method": METHOD_BEAM,
            "control_id": CONTROL_NONE,
            "multiscale": True,
            "beam_ensemble": True,
        },
        "pf_wrong_typewell": {
            "method": METHOD_PARTICLE_FILTER,
            "control_id": CONTROL_WRONG_TYPEWELL,
            "multiscale": True,
        },
        "pf_pair_shuffle": {
            "method": METHOD_PARTICLE_FILTER,
            "control_id": CONTROL_PAIR_SHUFFLE,
            "multiscale": True,
        },
        "pf_gr_order_shuffle": {
            "method": METHOD_PARTICLE_FILTER,
            "control_id": CONTROL_GR_ORDER_SHUFFLE,
            "multiscale": True,
        },
        "pf_phase_destroy": {
            "method": METHOD_PARTICLE_FILTER,
            "control_id": CONTROL_PHASE_DESTROY,
            "multiscale": True,
        },
        "pf_no_prefix_anchor": {
            "method": METHOD_PARTICLE_FILTER,
            "control_id": CONTROL_NO_PREFIX_ANCHOR,
            "multiscale": True,
        },
        "pf_no_cross_interaction": {
            "method": METHOD_PARTICLE_FILTER,
            "control_id": CONTROL_NO_CROSS_INTERACTION,
            "multiscale": True,
        },
    }
    if candidate not in settings:
        raise KeyError(f"unknown safe-physics candidate {candidate!r}")
    return family.predict(
        horizontal_frame=horizontal,
        typewell_frame=typewell,
        incumbent_prediction=incumbent,
        well_id=well_id,
        paired_typewell_id=paired_typewell_id,
        seeds=DEFAULT_SEEDS,
        **settings[candidate],
    )


def _metrics_bundle(
    frame: pd.DataFrame,
    *,
    candidate_columns: Sequence[str],
    bootstrap_samples: int,
    shadow_samples: int,
) -> dict[str, Any]:
    candidates = tuple(map(str, candidate_columns))
    columns = (INCUMBENT_COLUMN, *candidates)
    validation = validate_oof_frame(frame, columns)
    pooled = pooled_metrics(frame, columns, baseline_column=INCUMBENT_COLUMN)
    folds = per_fold_metrics(frame, columns, baseline_column=INCUMBENT_COLUMN)
    tails = tail_metrics(frame, columns)
    components = common_slope_shape_metrics(frame, columns)
    wells = per_well_metrics(frame, columns, baseline_column=INCUMBENT_COLUMN)
    wins = per_well_win_summary(
        frame,
        baseline_column=INCUMBENT_COLUMN,
        candidate_columns=candidates,
    )
    bootstrap = paired_well_cluster_bootstrap(
        frame,
        baseline_column=INCUMBENT_COLUMN,
        candidate_columns=candidates,
        samples=bootstrap_samples,
        seed=20260729,
    )
    residual = residual_correlation(
        frame,
        columns,
        aggregation="row",
    )
    shadow: dict[str, Any]
    available_wells = int(frame["well_id"].nunique())
    if available_wells >= 200 and shadow_samples > 0:
        shadow = shadow_private_subset_simulation(
            frame,
            baseline_column=INCUMBENT_COLUMN,
            candidate_columns=candidates,
            samples=shadow_samples,
            subset_wells=200,
            seed=20260729,
        )
    else:
        shadow = {
            "status": "not_run",
            "reason": (
                f"requires at least 200 wells; available={available_wells}"
                if shadow_samples > 0
                else "disabled"
            ),
        }
    return {
        "validation": validation,
        "pooled": pooled,
        "folds": folds,
        "tails": tails,
        "components": components,
        "per_well": wells,
        "wins": wins,
        "bootstrap": bootstrap,
        "residual_correlation": residual,
        "shadow_private": shadow,
    }


def _save_metrics(output_root: Path, bundle: Mapping[str, Any]) -> None:
    metric_root = output_root / "metrics"
    metric_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "pooled",
        "folds",
        "tails",
        "components",
        "per_well",
        "wins",
        "residual_correlation",
    ):
        value = bundle[name]
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"metric {name!r} is not a DataFrame")
        value.to_csv(metric_root / f"{name}.csv", index=name == "residual_correlation")
    _write_json(metric_root / "validation.json", bundle["validation"])
    _write_json(metric_root / "paired_bootstrap.json", bundle["bootstrap"])
    _write_json(metric_root / "shadow_private.json", bundle["shadow_private"])


def materialize_seed_diagnostics(
    frame: pd.DataFrame,
    *,
    stochastic_candidates: Sequence[str],
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Make every stochastic seed explicit, finite, and safe on fallback rows."""

    result = frame.copy()
    seed_columns: list[str] = []
    for candidate in map(str, stochastic_candidates):
        if candidate not in result:
            raise KeyError(f"missing stochastic candidate {candidate!r}")
        fallback = result[candidate].to_numpy(dtype=np.float64)
        for seed in map(int, seeds):
            column = f"{candidate}_seed_{seed}"
            if column not in result:
                result[column] = fallback
            else:
                values = result[column].to_numpy(dtype=np.float64).copy()
                missing = ~np.isfinite(values)
                if missing.any():
                    values[missing] = fallback[missing]
                    result[column] = values
            if not np.isfinite(result[column].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"seed prediction remains non-finite: {column}")
            seed_columns.append(column)
    return result, tuple(seed_columns)


def save_seed_metrics(
    output_root: Path,
    frame: pd.DataFrame,
    *,
    seed_columns: Sequence[str],
) -> None:
    """Persist deterministic seed-by-seed pooled and fold deltas."""

    columns = tuple(map(str, seed_columns))
    if not columns:
        return
    metric_root = Path(output_root) / "metrics"
    metric_root.mkdir(parents=True, exist_ok=True)
    pooled_metrics(
        frame,
        (INCUMBENT_COLUMN, *columns),
        baseline_column=INCUMBENT_COLUMN,
    ).to_csv(metric_root / "seed_pooled.csv", index=False)
    per_fold_metrics(
        frame,
        (INCUMBENT_COLUMN, *columns),
        baseline_column=INCUMBENT_COLUMN,
    ).to_csv(metric_root / "seed_folds.csv", index=False)


def _quick_gate(bundle: Mapping[str, Any]) -> dict[str, Any]:
    candidate = f"nested_{PRIMARY_CORRECT_CANDIDATE}"
    control_names = [f"nested_{name}" for name in NEGATIVE_CONTROL_CANDIDATES]
    pooled = bundle["pooled"].set_index("candidate")
    folds = bundle["folds"]
    components = bundle["components"].set_index("candidate")
    bootstrap = bundle["bootstrap"]["candidates"][candidate]
    correct_delta = float(pooled.loc[candidate, "delta_rmse_vs_baseline"])
    control_deltas = {
        name: float(pooled.loc[name, "delta_rmse_vs_baseline"])
        for name in control_names
    }
    improved_folds = int(
        np.sum(
            folds.loc[
                folds["candidate"].eq(candidate), "delta_rmse_vs_baseline"
            ].to_numpy(dtype=np.float64)
            < 0.0
        )
    )
    slope_improvement = float(
        components.loc[INCUMBENT_COLUMN, "slope_component_rmse"]
        - components.loc[candidate, "slope_component_rmse"]
    )
    shape_improvement = float(
        components.loc[INCUMBENT_COLUMN, "zero_mean_shape_rmse"]
        - components.loc[candidate, "zero_mean_shape_rmse"]
    )
    pairing_margin = (
        min(control_deltas.values()) - correct_delta if control_deltas else -np.inf
    )
    checks = {
        "negative_point_delta": correct_delta < 0.0,
        "bootstrap_upper_below_zero": float(bootstrap["delta_rmse_q97.5"]) < 0.0,
        "improves_at_least_four_folds": improved_folds >= 4,
        "slope_or_shape_improves": max(slope_improvement, shape_improvement) > 0.0,
        "correct_pairing_beats_every_control_by_0.01ft": pairing_margin > 0.01,
    }
    return {
        "candidate": candidate,
        "checks": checks,
        "passed": all(checks.values()),
        "pooled_delta": correct_delta,
        "bootstrap": bootstrap,
        "improved_folds": improved_folds,
        "slope_rmse_improvement": slope_improvement,
        "shape_rmse_improvement": shape_improvement,
        "negative_control_deltas": control_deltas,
        "pairing_margin_vs_best_control": float(pairing_margin),
        "full_oof_authorized_by_gate": all(checks.values()),
    }


def _visible_incumbent_geometry(
    *,
    data_root: Path,
    fold_config: Path,
    incumbent_root: Path,
    cache_root: Path,
    sample_submission_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    provider = HeelIncumbentProvider(
        data_root=data_root,
        fold_config=fold_config,
        artifact_root=incumbent_root,
        cache_root=cache_root,
    )
    rows: list[pd.DataFrame] = []
    prediction_vectors: list[np.ndarray] = []
    sample = pd.read_csv(sample_submission_path, dtype={"id": "string"})
    for well in enumerate_wells(data_root, "test"):
        result = provider.predict_visible_test(well.well_id)
        ids = np.asarray(
            [f"{well.well_id}_{row}" for row in result.row_index], dtype=object
        )
        rows.append(pd.DataFrame({"id": ids, "tvt_incumbent": result.prediction}))
        prediction_vectors.append(np.asarray(result.prediction, dtype=np.float64))
    frame = sample.loc[:, ["id"]].merge(
        pd.concat(rows, ignore_index=True),
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    values = frame["tvt_incumbent"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("visible incumbent reconstruction is incomplete")
    digest = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    return frame, {
        "dynamic_test_wells": [
            well.well_id for well in enumerate_wells(data_root, "test")
        ],
        "rows": int(len(frame)),
        "finite": True,
        "sample_id_order_aligned": True,
        "float64_prediction_sha256": digest,
        "submission_created": False,
    }


def run_experiment(
    *,
    profile: str,
    data_root: Path,
    fold_config: Path,
    output_root: Path,
    incumbent_oof: Path,
    incumbent_root: Path,
    cache_root: Path,
    quick_source: Path,
    external_dataset_dir: Path,
    bootstrap_samples: int,
    shadow_samples: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    profile = str(profile)
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == data_root or data_root in output_root.parents:
        raise ValueError("output_root cannot be inside data/")
    output_root.mkdir(parents=True, exist_ok=True)
    before_manifest = capture_data_manifest(data_root)

    train_files = {item.well_id: item for item in enumerate_wells(data_root, "train")}
    mapping = load_fold_mapping(fold_config, expected_well_ids=train_files)
    full_oof, baseline_audit = _baseline_audit(oof_path=incumbent_oof, mapping=mapping)
    selected = (
        _quick_validation_ids(quick_source, mapping)
        if profile == "quick"
        else _full_validation_ids(mapping)
    )
    selected_ids = tuple(
        sorted({well_id for values in selected.values() for well_id in values})
    )
    working = full_oof.loc[full_oof["well_id"].astype(str).isin(selected_ids)].copy()
    expected_rows = int(
        full_oof.groupby("well_id", sort=False).size().reindex(selected_ids).sum()
    )
    if len(working) != expected_rows:
        raise ValueError("selected OOF rows are incomplete")

    family_config = SafePhysicsAlignmentConfig(
        maximum_steps=128 if profile == "quick" else 160,
        rows_per_step=36 if profile == "quick" else 30,
        particles=192 if profile == "quick" else 256,
    )
    family = SafePhysicsAlignmentFamily(family_config)
    wrong_pair = {
        well_id: selected_ids[(index + 1) % len(selected_ids)]
        for index, well_id in enumerate(selected_ids)
    }
    shuffled_pair = _derangement(selected_ids, seed=20260729)
    typewell_cache = {
        well_id: load_typewell_inference(train_files[well_id].typewell_path)
        for well_id in selected_ids
    }
    candidates = (
        "ncc_multiscale",
        "pf_single_scale",
        "pf_multiscale",
        "beam_single_path",
        "beam_ensemble",
        *NEGATIVE_CONTROL_CANDIDATES,
    )
    outputs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    progress_path = output_root / "progress.json"
    for completed, well_id in enumerate(selected_ids, start=1):
        group = working.loc[working["well_id"].astype(str).eq(well_id)].sort_values(
            "row_index", kind="stable"
        )
        fold_values = group["fold"].astype(int).unique()
        if len(fold_values) != 1:
            raise ValueError(f"well spans multiple folds: {well_id}")
        fold = int(fold_values[0])
        if well_id not in selected[fold]:
            raise ValueError(f"selected fold mismatch: {well_id}")
        horizontal = load_horizontal_inference(train_files[well_id].horizontal_path)
        incumbent = group[INCUMBENT_COLUMN].to_numpy(dtype=np.float64)
        current = group.loc[
            :, ["well_id", "row_index", "fold", "y_true", INCUMBENT_COLUMN]
        ].copy()
        for candidate in candidates:
            if candidate == "pf_wrong_typewell":
                paired_id = wrong_pair[well_id]
            elif candidate == "pf_pair_shuffle":
                paired_id = shuffled_pair[well_id]
            else:
                paired_id = well_id
            result = _candidate_result(
                family=family,
                candidate=candidate,
                horizontal=horizontal,
                typewell=typewell_cache[paired_id],
                incumbent=incumbent,
                well_id=well_id,
                paired_typewell_id=paired_id,
            )
            if not np.array_equal(
                result.row_index, current["row_index"].to_numpy(dtype=np.int64)
            ):
                raise ValueError(f"alignment keys changed for {well_id}/{candidate}")
            current[candidate] = result.prediction
            if candidate.startswith("pf_"):
                for seed, seed_prediction in result.seed_predictions.items():
                    current[f"{candidate}_seed_{seed}"] = seed_prediction
            if candidate == PRIMARY_CORRECT_CANDIDATE:
                current[f"{candidate}_uncertainty"] = result.uncertainty
                current[f"{candidate}_effective_particles"] = (
                    result.effective_particle_count
                )
            diagnostics.append(
                _prediction_record(result, candidate=candidate, fold=fold)
            )
        outputs.append(current)
        _write_json(
            progress_path,
            {
                "profile": profile,
                "completed_wells": completed,
                "total_wells": len(selected_ids),
                "last_well_id": well_id,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )

    oof = (
        pd.concat(outputs, ignore_index=True)
        .sort_values(["fold", "well_id", "row_index"], kind="stable")
        .reset_index(drop=True)
    )
    stochastic_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.startswith("pf_") and candidate != "pf_no_cross_interaction"
    )
    oof, seed_columns = materialize_seed_diagnostics(
        oof,
        stochastic_candidates=stochastic_candidates,
    )
    base_candidate_columns = tuple(candidates)
    nested_oof, nested_provenance = cross_fit_component_shrinkage(
        oof,
        incumbent_column=INCUMBENT_COLUMN,
        candidate_columns=base_candidate_columns,
    )
    formal_candidates = (
        *base_candidate_columns,
        *(f"nested_{candidate}" for candidate in base_candidate_columns),
    )
    metric_frame = nested_oof.loc[
        :,
        [
            "well_id",
            "row_index",
            "fold",
            "y_true",
            INCUMBENT_COLUMN,
            *formal_candidates,
        ],
    ].copy()
    bundle = _metrics_bundle(
        metric_frame,
        candidate_columns=formal_candidates,
        bootstrap_samples=bootstrap_samples,
        shadow_samples=shadow_samples,
    )
    _save_metrics(output_root, bundle)
    save_seed_metrics(output_root, nested_oof, seed_columns=seed_columns)
    nested_oof.to_csv(
        output_root / "oof_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    pd.DataFrame(diagnostics).to_csv(
        output_root / "alignment_diagnostics.csv", index=False
    )
    _write_json(output_root / "nested_component_weights.json", nested_provenance)

    external_family, external_inventory = audit_external_prediction_files(
        dataset_dir=external_dataset_dir,
        sample_submission_path=data_root / "sample_submission.csv",
    )
    visible_incumbent, visible_audit = _visible_incumbent_geometry(
        data_root=data_root,
        fold_config=fold_config,
        incumbent_root=incumbent_root,
        cache_root=cache_root,
        sample_submission_path=data_root / "sample_submission.csv",
    )
    geometry = external_prediction_geometry(
        family=external_family,
        sample_submission_path=data_root / "sample_submission.csv",
        incumbent=visible_incumbent,
    )
    provenance_root = output_root.parent / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    external_inventory.to_csv(
        provenance_root / "external_prediction_inventory.csv", index=False
    )
    _write_json(
        provenance_root / "external_prediction_registry.json",
        external_family.manifest(),
    )
    for name, frame in geometry.items():
        frame.to_csv(provenance_root / f"geometry_{name}.csv", index=False)
    visible_incumbent.to_csv(
        provenance_root / "visible_incumbent_diagnostic.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    persisted_visible = pd.read_csv(
        provenance_root / "visible_incumbent_diagnostic.csv.gz",
        dtype={"id": "string"},
    )
    persisted_values = persisted_visible["tvt_incumbent"].to_numpy(dtype=np.float64)
    visible_audit["float64_prediction_sha256"] = hashlib.sha256(
        persisted_values.tobytes(order="C")
    ).hexdigest()
    _write_json(provenance_root / "visible_incumbent_audit.json", visible_audit)

    gate = _quick_gate(bundle) if profile == "quick" else {"status": "not_applicable"}
    after_manifest = capture_data_manifest(data_root)
    if after_manifest != before_manifest:
        raise RuntimeError("data manifest changed during safe-physics experiment")
    summary = {
        "schema_version": "rogii03_safe_physics_experiment_v1",
        "profile": profile,
        "population": {
            "wells": int(metric_frame["well_id"].nunique()),
            "rows": int(len(metric_frame)),
            "folds": sorted(map(int, metric_frame["fold"].unique())),
        },
        "baseline_audit": baseline_audit,
        "family_manifest": family.manifest(),
        "candidate_columns": list(formal_candidates),
        "seed_prediction_columns": list(seed_columns),
        "nested_component_weights_path": str(
            output_root / "nested_component_weights.json"
        ),
        "quick_gate": gate,
        "data_manifest_before": before_manifest,
        "data_manifest_after": after_manifest,
        "data_manifest_equal": True,
        "external_prediction_registry": external_family.manifest(),
        "visible_incumbent_audit": visible_audit,
        "submission_created": False,
        "submission_uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--incumbent-oof", type=Path, required=True)
    parser.add_argument("--incumbent-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--quick-source", type=Path, required=True)
    parser.add_argument("--external-dataset-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--shadow-samples", type=int, default=100_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_experiment(
        profile=args.profile,
        data_root=args.data_root,
        fold_config=args.fold_config,
        output_root=args.output_root,
        incumbent_oof=args.incumbent_oof,
        incumbent_root=args.incumbent_root,
        cache_root=args.cache_root,
        quick_source=args.quick_source,
        external_dataset_dir=args.external_dataset_dir,
        bootstrap_samples=args.bootstrap_samples,
        shadow_samples=args.shadow_samples,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
