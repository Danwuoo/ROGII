from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import rogii.external_models.weight_stability as weight_stability_module
from rogii.diagnostics.nested_external_ensemble import (
    FAMILY_MAX_WEIGHT,
    INCUMBENT_COLUMN,
    REGULARIZATION_STRENGTH,
    REGULARIZATION_TARGET,
    SCHEMA_VERSION as NESTED_RUN_SCHEMA_VERSION,
    TCN_COLUMN,
)
from rogii.ensemble import fit_constrained_blend
from rogii.external_models.artifacts import sha256_file, well_ids_sha256
from rogii.external_models.nested_reaudit import NESTED_REAUDIT_SCHEMA_VERSION
from rogii.external_models.weight_stability import (
    BOOTSTRAP_SAMPLES,
    STATISTICS_FILENAME,
    SUMMARY_FILENAME,
    WEIGHT_STABILITY_SCHEMA_VERSION,
    WeightStabilityError,
    analyze_nested_blend_weight_stability,
    bootstrap_two_family_weights,
    fit_two_family_sufficient_statistics,
    two_family_well_sufficient_statistics,
)


TCN_SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)
INCUMBENT_SEEDS = TCN_SEEDS
REQUIRED_TRUE_CHECKS = (
    "checkpoint_and_sidecar_hashes",
    "data_manifest_equal_before_after_current",
    "disk_only_source_reference",
    "exact_key_and_truth_coverage",
    "final_predictions_equal_outer_artifacts",
    "fold_and_training_provenance",
    "inner_family_first_seed_average",
    "inner_residual_component_algebra",
    "metrics_recomputed_from_final_oof",
    "outer_predictions_recomputed_from_weights",
    "reciprocal_reuse_equivalence_and_alias_counts",
    "runner_artifact_audit_hash_anchored_by_summary",
    "runner_summary_recomputed",
    "source_independent_lock_required_for_production",
    "source_outer_predictions_untouched",
    "weights_recomputed_from_inner_oof",
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _blend_policy() -> dict[str, Any]:
    return {
        "families": {
            "incumbent": [INCUMBENT_COLUMN],
            "tcn": [TCN_COLUMN],
        },
        "family_first_seed_average": True,
        "regularization_strength": REGULARIZATION_STRENGTH,
        "regularization_target": REGULARIZATION_TARGET,
        "family_max_weight": FAMILY_MAX_WEIGHT,
        "fixed_controls": {
            "y_pred_fixed_075_025": {"incumbent": 0.75, "tcn": 0.25},
            "y_pred_simple_050_050": {"incumbent": 0.5, "tcn": 0.5},
        },
        "weight_selection": "inner OOF only",
        "outer_validation_used_for_weight_fit": False,
    }


def _prediction_columns() -> tuple[str, ...]:
    return (
        INCUMBENT_COLUMN,
        TCN_COLUMN,
        *(f"{TCN_COLUMN}__seed_{seed}" for seed in TCN_SEEDS),
        *(f"{TCN_COLUMN}__{component}" for component in ("common", "slope", "shape")),
        *(
            f"{TCN_COLUMN}__seed_{seed}__{component}"
            for seed in TCN_SEEDS
            for component in ("common", "slope", "shape")
        ),
    )


def _inner_frame(well_ids: tuple[str, ...], inner_fold: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well_index, well_id in enumerate(well_ids):
        well_number = int(well_id.split("-")[-1])
        for row_index in range(1 + (well_index + inner_fold) % 3):
            truth = 100.0 + 1.7 * well_number + 0.4 * row_index
            incumbent = truth + 2.4 - 0.18 * well_number + 0.1 * row_index
            seed_values = [
                truth + 0.8 - 0.06 * well_number + 0.03 * row_index + 0.01 * seed_index
                for seed_index in range(len(TCN_SEEDS))
            ]
            tcn = float(np.mean(seed_values))
            row: dict[str, Any] = {
                "well_id": well_id,
                "row_index": row_index,
                "fold": inner_fold,
                "y_true": truth,
                INCUMBENT_COLUMN: incumbent,
                TCN_COLUMN: tcn,
                f"{TCN_COLUMN}__common": tcn - incumbent,
                f"{TCN_COLUMN}__slope": 0.0,
                f"{TCN_COLUMN}__shape": 0.0,
            }
            for seed, seed_value in zip(TCN_SEEDS, seed_values, strict=True):
                row[f"{TCN_COLUMN}__seed_{seed}"] = seed_value
                row[f"{TCN_COLUMN}__seed_{seed}__common"] = seed_value - incumbent
                row[f"{TCN_COLUMN}__seed_{seed}__slope"] = 0.0
                row[f"{TCN_COLUMN}__seed_{seed}__shape"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def _fixture(root: Path) -> Path:
    run_root = root / "nested-run"
    run_root.mkdir()
    folds = tuple(range(5))
    validation_by_fold = {
        fold: tuple(f"well-{2 * fold + offset:02d}" for offset in range(2))
        for fold in folds
    }
    universe = tuple(
        sorted(well_id for values in validation_by_fold.values() for well_id in values)
    )
    cohort: dict[str, Any] = {}
    for fold in folds:
        validation = validation_by_fold[fold]
        training = tuple(sorted(set(universe) - set(validation)))
        cohort[str(fold)] = {
            "training_well_ids": list(training),
            "training_ids_sha256": well_ids_sha256(training),
            "validation_well_ids": list(validation),
            "validation_ids_sha256": well_ids_sha256(validation),
            "intersection": [],
        }
    run_config: dict[str, Any] = {
        "schema_version": NESTED_RUN_SCHEMA_VERSION,
        "run_id": "synthetic-nested",
        "evidence_mode": "production",
        "evidence_eligible": True,
        "submission_created": False,
        "blend_policy": _blend_policy(),
        "policy": {
            "outer_folds": list(folds),
            "tcn_seeds": list(TCN_SEEDS),
            "incumbent_seeds": list(INCUMBENT_SEEDS),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "source": {
            "oof_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        },
        "cohort": cohort,
        "implementation_hashes": {},
    }
    run_config_path = run_root / "run_config.json"
    _write_json(run_config_path, run_config)
    run_config_digest = _canonical_sha256(run_config)
    policy_digest = _canonical_sha256(run_config["policy"])

    inner_records: list[dict[str, Any]] = []
    statistics_by_outer: dict[int, list[pd.DataFrame]] = {fold: [] for fold in folds}
    for outer_fold in folds:
        for inner_fold in folds:
            if inner_fold == outer_fold:
                continue
            validation = validation_by_fold[inner_fold]
            excluded = validation_by_fold[outer_fold]
            training = tuple(sorted(set(universe) - set(validation) - set(excluded)))
            frame = _inner_frame(validation, inner_fold)
            directory = (
                run_root / "inner_oof" / f"outer_{outer_fold}" / f"inner_{inner_fold}"
            )
            directory.mkdir(parents=True)
            oof_path = directory / "oof_predictions.csv.gz"
            frame.to_csv(
                oof_path,
                index=False,
                float_format="%.17g",
                compression={"method": "gzip", "mtime": 0},
            )
            oof_sha = sha256_file(oof_path)
            predictions = _prediction_columns()
            manifest = {
                "schema_version": NESTED_RUN_SCHEMA_VERSION,
                "artifact_kind": "inner_oof",
                "completed": True,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "training_well_ids": list(training),
                "training_ids_sha256": well_ids_sha256(training),
                "validation_well_ids": list(validation),
                "validation_ids_sha256": well_ids_sha256(validation),
                "excluded_outer_well_ids": list(excluded),
                "excluded_outer_ids_sha256": well_ids_sha256(excluded),
                "training_validation_intersection": [],
                "training_outer_intersection": [],
                "validation_outer_intersection": [],
                "policy_digest": policy_digest,
                "run_config_digest": run_config_digest,
                "source_oof_sha256": run_config["source"]["oof_sha256"],
                "truth_attached_only_after_all_seed_predictions": True,
                "outer_validation_labels_used_for_fit": False,
                "inner_validation_labels_used_for_fit": False,
                "prediction_columns": list(predictions),
                "incumbent_checkpoints": [{} for _ in INCUMBENT_SEEDS],
                "tcn_checkpoints": [{} for _ in TCN_SEEDS],
                "rows": len(frame),
                "wells": len(validation),
                "oof_path": oof_path.relative_to(run_root).as_posix(),
                "oof_sha256": oof_sha,
                "validation": {
                    "coverage_checked": True,
                    "truth_checked": True,
                    "rows": len(frame),
                    "wells": len(validation),
                    "folds": [inner_fold],
                    "duplicate_well_row_keys": 0,
                    "nonfinite_numeric_values": 0,
                    "prediction_columns": list(predictions),
                },
            }
            manifest_path = directory / "manifest.json"
            _write_json(manifest_path, manifest)
            inner_records.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "rows": len(frame),
                    "wells": len(validation),
                    "training_wells": len(training),
                    "validation_wells": len(validation),
                    "incumbent_checkpoints": len(INCUMBENT_SEEDS),
                    "tcn_checkpoints": len(TCN_SEEDS),
                    "oof_sha256": oof_sha,
                    "manifest_sha256": sha256_file(manifest_path),
                }
            )
            statistics_by_outer[outer_fold].append(
                two_family_well_sufficient_statistics(
                    frame,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
            )

    weights: list[dict[str, Any]] = []
    for outer_fold in folds:
        point = fit_two_family_sufficient_statistics(
            pd.concat(statistics_by_outer[outer_fold], ignore_index=True)
        )
        for family in ("incumbent", "tcn"):
            weights.append(
                {
                    "outer_fold": outer_fold,
                    "family": family,
                    "weight": point[f"{family}_weight"],
                }
            )

    final_oof_path = run_root / "oof_predictions.csv.gz"
    final_oof_path.write_bytes(b"opaque-final-oof-bytes")
    audit = {
        "schema_version": NESTED_REAUDIT_SCHEMA_VERSION,
        "passed": True,
        "run_root": str(run_root.resolve()),
        "evidence_mode": "production",
        "evidence_eligible": True,
        "submission_created": False,
        "checks": {
            **{name: True for name in REQUIRED_TRUE_CHECKS},
            "submission_created": False,
        },
        "source": {
            "independently_locked": True,
            "oof_sha256": run_config["source"]["oof_sha256"],
            "manifest_sha256": run_config["source"]["manifest_sha256"],
        },
        "artifacts": {
            "run_config": {
                "path": str(run_config_path.resolve()),
                "sha256": sha256_file(run_config_path),
                "digest": run_config_digest,
            },
            "final_oof": {
                "path": str(final_oof_path.resolve()),
                "sha256": sha256_file(final_oof_path),
            },
        },
        "inner_splits": inner_records,
        "weights": weights,
    }
    _write_json(run_root / "independent_nested_reaudit.json", audit)
    return run_root


@pytest.mark.parametrize("cap_case", (False, True))
def test_sufficient_statistics_matches_row_level_constrained_refit(
    cap_case: bool,
) -> None:
    rows: list[dict[str, Any]] = []
    for well_index, row_count in enumerate((2, 4, 3)):
        for row_index in range(row_count):
            truth = 20.0 + well_index + 0.2 * row_index
            if cap_case:
                incumbent = truth + 10.0 + 0.5 * well_index
                tcn = truth - 1.0 - 0.1 * well_index
            else:
                incumbent = truth + 2.0 + 0.5 * well_index
                tcn = truth + 0.7 - 0.1 * well_index
            rows.append(
                {
                    "well_id": f"w{well_index}",
                    "y_true": truth,
                    INCUMBENT_COLUMN: incumbent,
                    TCN_COLUMN: tcn,
                }
            )
    frame = pd.DataFrame(rows)
    statistics = two_family_well_sufficient_statistics(frame)
    multiplicities = np.asarray([2, 0, 3], dtype=np.int64)
    expanded = pd.concat(
        [
            frame.loc[frame["well_id"].eq(well_id)].copy()
            for well_id, count in zip(
                statistics["well_id"],
                multiplicities,
                strict=True,
            )
            for _ in range(int(count))
        ],
        ignore_index=True,
    )
    row_fit = fit_constrained_blend(
        expanded["y_true"].to_numpy(dtype=np.float64),
        expanded.loc[:, [INCUMBENT_COLUMN, TCN_COLUMN]].rename(
            columns={
                INCUMBENT_COLUMN: "incumbent",
                TCN_COLUMN: "tcn",
            }
        ),
        regularization_strength=REGULARIZATION_STRENGTH,
        regularization_target=REGULARIZATION_TARGET,
        family_max_weight=FAMILY_MAX_WEIGHT,
    )
    sufficient_fit = fit_two_family_sufficient_statistics(
        statistics,
        multiplicities=multiplicities,
    )
    assert sufficient_fit["incumbent_weight"] == pytest.approx(
        row_fit.weight_by_family["incumbent"],
        abs=1.0e-10,
    )
    assert sufficient_fit["tcn_weight"] == pytest.approx(
        row_fit.weight_by_family["tcn"],
        abs=1.0e-10,
    )
    assert sufficient_fit["training_rmse"] == pytest.approx(
        row_fit.training_rmse,
        abs=1.0e-10,
    )
    assert sufficient_fit["objective_value"] == pytest.approx(
        row_fit.objective_value,
        abs=1.0e-10,
    )
    if cap_case:
        assert sufficient_fit["tcn_weight"] == pytest.approx(0.5, abs=1.0e-12)


def test_bootstrap_is_clustered_row_weighted_and_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["short", "long", "long", "long", "long"],
            "y_true": [0.0, 0.0, 0.0, 0.0, 0.0],
            INCUMBENT_COLUMN: [1.0, 4.0, 4.0, 4.0, 4.0],
            TCN_COLUMN: [0.0, 3.0, 3.0, 3.0, 3.0],
        }
    )
    statistics = two_family_well_sufficient_statistics(frame)
    first = bootstrap_two_family_weights(
        statistics,
        samples=100,
        rng=np.random.default_rng(123),
        batch_size=17,
    )
    second = bootstrap_two_family_weights(
        statistics,
        samples=100,
        rng=np.random.default_rng(123),
        batch_size=17,
    )
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.all((first >= 0.0) & (first <= 0.5))
    assert np.unique(first).size > 1


def test_formal_analyzer_never_parses_final_oof_and_writes_atomic_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _fixture(tmp_path)
    final_path = (run_root / "oof_predictions.csv.gz").resolve()
    original_read_csv = weight_stability_module.pd.read_csv
    parsed_paths: list[Path] = []

    def guarded_read_csv(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        current = Path(path).resolve()
        assert current != final_path
        parsed_paths.append(current)
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(weight_stability_module.pd, "read_csv", guarded_read_csv)
    final_before = final_path.read_bytes()
    summary = analyze_nested_blend_weight_stability(run_root)
    output = run_root / "weight_stability"

    assert summary["schema_version"] == WEIGHT_STABILITY_SCHEMA_VERSION
    assert summary["completed"] is True
    assert summary["evidence_eligible"] is False
    assert summary["evidence_mode"] == "conditional_weight_crossfit"
    assert summary["evidence_contract"]["model_fit_safe"] is True
    assert summary["evidence_contract"]["weight_fit_safe"] is True
    assert summary["evidence_contract"]["family_selection_safe"] is False
    assert summary["checks"]["only_inner_oof_labels_parsed"] is True
    assert summary["checks"]["outer_validation_labels_used_for_fit"] is False
    assert summary["checks"]["final_oof_csv_parsed"] is False
    assert summary["scope"]["bootstrap_samples_per_outer_fold"] == 10_000
    assert summary["scope"]["pooled_all_fold_draws"] == 50_000
    assert summary["scope"]["pooled_fold_mean_samples"] == 10_000
    assert len(summary["locks"]["inner_artifacts"]) == 20
    assert len(parsed_paths) == 20
    assert all("inner_oof" in str(path) for path in parsed_paths)
    assert final_path.read_bytes() == final_before
    assert summary["point_fit_equivalence"]["passed"] is True
    assert summary["point_fit_equivalence"]["maximum_absolute_weight_delta"] <= 1e-8
    assert (
        json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8")) == summary
    )
    compact = pd.read_csv(output / STATISTICS_FILENAME)
    assert len(compact) == 14
    assert set(compact["family"]) == {"incumbent", "tcn"}
    assert set(compact["scope"]) == {
        *(f"outer_fold_{fold}" for fold in range(5)),
        "pooled_all_fold_draws",
        "pooled_fold_mean",
    }
    assert not list(run_root.glob(".weight_stability.tmp-*"))


@pytest.mark.parametrize(
    "corruption",
    (
        "audit_failed",
        "final_oof_bytes",
        "manifest_bytes",
        "outer_validation_leak",
        "tcn_family_mean",
        "duplicate_key",
    ),
)
def test_corruption_fails_without_atomic_output(
    tmp_path: Path,
    corruption: str,
) -> None:
    run_root = _fixture(tmp_path)
    audit_path = run_root / "independent_nested_reaudit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest_path = run_root / "inner_oof" / "outer_0" / "inner_1" / "manifest.json"
    oof_path = manifest_path.with_name("oof_predictions.csv.gz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if corruption == "audit_failed":
        audit["passed"] = False
        _write_json(audit_path, audit)
    elif corruption == "final_oof_bytes":
        (run_root / "oof_predictions.csv.gz").write_bytes(b"changed")
    elif corruption == "manifest_bytes":
        manifest["completed"] = False
        _write_json(manifest_path, manifest)
    elif corruption == "outer_validation_leak":
        manifest["validation_well_ids"][0] = manifest["excluded_outer_well_ids"][0]
        manifest["validation_ids_sha256"] = well_ids_sha256(
            tuple(manifest["validation_well_ids"])
        )
        _write_json(manifest_path, manifest)
        audit["inner_splits"][0]["manifest_sha256"] = sha256_file(manifest_path)
        _write_json(audit_path, audit)
    else:
        frame = pd.read_csv(oof_path, float_precision="round_trip")
        if corruption == "tcn_family_mean":
            frame.loc[0, TCN_COLUMN] += 0.5
        else:
            frame.loc[1, ["well_id", "row_index", "fold"]] = frame.loc[
                0,
                ["well_id", "row_index", "fold"],
            ].to_numpy()
        frame.to_csv(
            oof_path,
            index=False,
            float_format="%.17g",
            compression={"method": "gzip", "mtime": 0},
        )
        new_oof_sha = sha256_file(oof_path)
        manifest["oof_sha256"] = new_oof_sha
        _write_json(manifest_path, manifest)
        audit["inner_splits"][0]["oof_sha256"] = new_oof_sha
        audit["inner_splits"][0]["manifest_sha256"] = sha256_file(manifest_path)
        _write_json(audit_path, audit)

    with pytest.raises((WeightStabilityError, ValueError)):
        analyze_nested_blend_weight_stability(run_root)
    assert not (run_root / "weight_stability").exists()
    assert not list(run_root.glob(".weight_stability.tmp-*"))
