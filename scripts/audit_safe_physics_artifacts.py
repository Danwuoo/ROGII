"""Independent disk re-audit for the safe-physics experiment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rogii.external_models.artifacts import capture_data_manifest
from rogii.external_predictions import split_submission_id
from rogii.nested_blend import (
    COMPONENT_NAMES,
    apply_component_shrinkage,
    fit_component_shrinkage,
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
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _assert_frame_close(
    actual: pd.DataFrame,
    expected_path: Path,
    *,
    index_column: str | None = None,
) -> float:
    expected = pd.read_csv(expected_path, index_col=0 if index_column else None)
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=not bool(index_column)),
        expected.reset_index(drop=not bool(index_column)),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-8,
    )
    numeric = actual.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    reference = expected.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    return (
        float(np.max(np.abs(numeric - reference)))
        if numeric.size and reference.size
        else 0.0
    )


def audit(
    *,
    run_root: Path,
    data_root: Path,
    incumbent_oof: Path,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(
        run_root / "oof_predictions.csv.gz",
        dtype={"well_id": "string"},
    )
    candidates = tuple(map(str, summary["candidate_columns"]))
    seed_columns = tuple(map(str, summary.get("seed_prediction_columns", ())))
    validation = validate_oof_frame(
        frame,
        [INCUMBENT_COLUMN, *candidates],
        expected_folds=range(5),
    )
    if seed_columns:
        missing_seed_columns = sorted(set(seed_columns) - set(frame.columns))
        if missing_seed_columns:
            raise ValueError(f"missing seed columns: {missing_seed_columns}")
        if not np.isfinite(frame.loc[:, seed_columns].to_numpy(dtype=np.float64)).all():
            raise ValueError("seed predictions contain non-finite values")

    reference = pd.read_csv(
        incumbent_oof,
        usecols=[
            "well_id",
            "row_index",
            "fold",
            "y_true",
            "y_pred_heel_registered_ensemble",
        ],
        dtype={"well_id": "string"},
    ).rename(columns={"y_pred_heel_registered_ensemble": INCUMBENT_COLUMN})
    joined = frame.loc[
        :, ["well_id", "row_index", "fold", "y_true", INCUMBENT_COLUMN]
    ].merge(
        reference,
        on=["well_id", "row_index", "fold"],
        how="left",
        validate="one_to_one",
        suffixes=("_artifact", "_reference"),
        sort=False,
    )
    truth_delta = float(
        np.max(
            np.abs(
                joined["y_true_artifact"].to_numpy(dtype=np.float64)
                - joined["y_true_reference"].to_numpy(dtype=np.float64)
            )
        )
    )
    incumbent_delta = float(
        np.max(
            np.abs(
                joined[f"{INCUMBENT_COLUMN}_artifact"].to_numpy(dtype=np.float64)
                - joined[f"{INCUMBENT_COLUMN}_reference"].to_numpy(dtype=np.float64)
            )
        )
    )
    if truth_delta != 0.0 or incumbent_delta != 0.0:
        raise ValueError("artifact truth/incumbent differs from canonical OOF")

    provenance = json.loads(
        (run_root / "nested_component_weights.json").read_text(encoding="utf-8")
    )
    nested_maximum = 0.0
    raw_candidates = tuple(
        candidate for candidate in candidates if not candidate.startswith("nested_")
    )
    for candidate in raw_candidates:
        records = provenance.get(candidate)
        if not isinstance(records, list) or len(records) != 5:
            raise ValueError(f"nested provenance incomplete for {candidate!r}")
        for record in records:
            fold = int(record["outer_fold"])
            state = fit_component_shrinkage(
                frame,
                outer_fold=fold,
                incumbent_column=INCUMBENT_COLUMN,
                candidate_column=candidate,
            )
            recorded_weights = tuple(
                float(record["weights"][name]) for name in COMPONENT_NAMES
            )
            weight_delta = float(
                np.max(np.abs(np.asarray(state.weights) - np.asarray(recorded_weights)))
            )
            if weight_delta > 1e-12:
                raise ValueError(f"nested weight mismatch for {candidate}/fold={fold}")
            validation_frame = frame.loc[frame["fold"].astype(int).eq(fold)]
            recomputed = apply_component_shrinkage(validation_frame, state).to_numpy(
                dtype=np.float64
            )
            recorded = validation_frame[f"nested_{candidate}"].to_numpy(
                dtype=np.float64
            )
            nested_maximum = max(
                nested_maximum, float(np.max(np.abs(recomputed - recorded)))
            )
    if nested_maximum > 1e-9:
        raise ValueError("nested predictions do not reconstruct from disk provenance")

    columns = (INCUMBENT_COLUMN, *candidates)
    metric_root = run_root / "metrics"
    recomputed = {
        "pooled": pooled_metrics(frame, columns, baseline_column=INCUMBENT_COLUMN),
        "folds": per_fold_metrics(frame, columns, baseline_column=INCUMBENT_COLUMN),
        "tails": tail_metrics(frame, columns),
        "components": common_slope_shape_metrics(frame, columns),
        "per_well": per_well_metrics(frame, columns, baseline_column=INCUMBENT_COLUMN),
        "wins": per_well_win_summary(
            frame,
            baseline_column=INCUMBENT_COLUMN,
            candidate_columns=candidates,
        ),
        "residual_correlation": residual_correlation(frame, columns, aggregation="row"),
    }
    metric_deltas = {
        name: _assert_frame_close(
            value,
            metric_root / f"{name}.csv",
            index_column="candidate" if name == "residual_correlation" else None,
        )
        for name, value in recomputed.items()
    }
    seed_metric_deltas: dict[str, float] = {}
    if seed_columns:
        for name, value in {
            "seed_pooled": pooled_metrics(
                frame,
                (INCUMBENT_COLUMN, *seed_columns),
                baseline_column=INCUMBENT_COLUMN,
            ),
            "seed_folds": per_fold_metrics(
                frame,
                (INCUMBENT_COLUMN, *seed_columns),
                baseline_column=INCUMBENT_COLUMN,
            ),
        }.items():
            seed_metric_deltas[name] = _assert_frame_close(
                value,
                metric_root / f"{name}.csv",
            )

    recorded_bootstrap = json.loads(
        (metric_root / "paired_bootstrap.json").read_text(encoding="utf-8")
    )
    bootstrap = paired_well_cluster_bootstrap(
        frame,
        baseline_column=INCUMBENT_COLUMN,
        candidate_columns=candidates,
        samples=int(recorded_bootstrap["samples"]),
        seed=int(recorded_bootstrap["seed"]),
    )
    bootstrap_delta = 0.0
    for candidate in candidates:
        for key, value in bootstrap["candidates"][candidate].items():
            bootstrap_delta = max(
                bootstrap_delta,
                abs(
                    float(value)
                    - float(recorded_bootstrap["candidates"][candidate][key])
                ),
            )
    if bootstrap_delta > 1e-12:
        raise ValueError("paired bootstrap differs from recorded artifact")

    recorded_shadow = json.loads(
        (metric_root / "shadow_private.json").read_text(encoding="utf-8")
    )
    shadow_delta = 0.0
    if recorded_shadow.get("status") != "not_run":
        shadow = shadow_private_subset_simulation(
            frame,
            baseline_column=INCUMBENT_COLUMN,
            candidate_columns=candidates,
            samples=int(recorded_shadow["samples"]),
            subset_wells=int(recorded_shadow["subset_wells"]),
            seed=int(recorded_shadow["seed"]),
        )
        for candidate in candidates:
            for key, value in shadow["candidates"][candidate].items():
                shadow_delta = max(
                    shadow_delta,
                    abs(
                        float(value)
                        - float(recorded_shadow["candidates"][candidate][key])
                    ),
                )
        if shadow_delta > 1e-12:
            raise ValueError("shadow-private simulation differs from disk artifact")

    registry_path = run_root.parent / "provenance" / "external_prediction_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (
        registry.get("eligibility") != "diagnostic_only"
        or registry.get("fit_allowed") is not False
        or registry.get("oof_scoring_allowed") is not False
    ):
        raise ValueError("external registry eligibility changed")
    external_hashes = {}
    for record in registry["files"]:
        path = Path(record["path"])
        actual = _sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"external prediction hash mismatch: {path}")
        external_hashes[path.name] = actual
    source_registry_path = (
        run_root.parent / "provenance" / "source_lineage_registry.json"
    )
    source_registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    source_files = {
        str(record["name"]): str(record["sha256"])
        for record in source_registry["files"]
    }
    if source_registry["dataset"].get("eligibility") != "diagnostic_only":
        raise ValueError("source registry eligibility changed")
    if source_files != external_hashes:
        raise ValueError("source registry file hashes differ from audited externals")

    visible_path = (
        run_root.parent / "provenance" / "visible_incumbent_diagnostic.csv.gz"
    )
    visible = pd.read_csv(visible_path, dtype={"id": "string"})
    split_submission_id(visible["id"])
    visible_values = visible["tvt_incumbent"].to_numpy(dtype=np.float64)
    visible_digest = hashlib.sha256(visible_values.tobytes(order="C")).hexdigest()
    visible_audit = json.loads(
        (run_root.parent / "provenance" / "visible_incumbent_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if visible_digest != visible_audit["float64_prediction_sha256"]:
        raise ValueError("visible incumbent diagnostic digest mismatch")

    current_manifest = capture_data_manifest(data_root)
    if (
        current_manifest != summary["data_manifest_before"]
        or current_manifest != summary["data_manifest_after"]
    ):
        raise ValueError("current data manifest differs from experiment")
    submission_candidates = [
        str(path)
        for root in (run_root, run_root.parent / "provenance")
        for path in root.glob("submission*.csv*")
    ]
    if submission_candidates:
        raise ValueError(f"unexpected submission artifacts: {submission_candidates}")

    result = {
        "schema_version": "rogii03_safe_physics_independent_audit_v1",
        "passed": True,
        "run_root": str(run_root),
        "oof_sha256": _sha256(run_root / "oof_predictions.csv.gz"),
        "validation": validation,
        "canonical_truth_maximum_absolute_delta": truth_delta,
        "canonical_incumbent_maximum_absolute_delta": incumbent_delta,
        "nested_prediction_maximum_absolute_delta": nested_maximum,
        "metric_maximum_absolute_deltas": metric_deltas,
        "seed_metric_maximum_absolute_deltas": seed_metric_deltas,
        "bootstrap_maximum_absolute_delta": bootstrap_delta,
        "shadow_private_maximum_absolute_delta": shadow_delta,
        "external_prediction_hashes": external_hashes,
        "source_lineage_registry_sha256": _sha256(source_registry_path),
        "visible_incumbent_prediction_sha256": visible_digest,
        "data_manifest_before_after_current_equal": True,
        "submission_artifacts": [],
    }
    _write_json(run_root / "independent_audit.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--incumbent-oof", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit(
        run_root=args.run_root,
        data_root=args.data_root,
        incumbent_oof=args.incumbent_oof,
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
