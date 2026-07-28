from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import rogii.diagnostics.nested_external_ensemble as nested_module
import rogii.external_models.nested_reaudit as nested_reaudit_module
from rogii.diagnostics.nested_external_ensemble import (
    FIXED_050_COLUMN,
    FIXED_075_COLUMN,
    INCUMBENT_COLUMN,
    NESTED_COLUMN,
    REGULARIZATION_STRENGTH,
    TCN_COLUMN,
    NestedRunPolicy,
    estimate_nested_tcn_runtime,
    make_smoke_outer_tcn_source,
    run_nested_external_ensemble,
)
from rogii.external_models.data import (
    MODEL_INPUT_CHANNEL_NAMES,
    FixedGridConfig,
)
from rogii.external_models.artifacts import sha256_file
from rogii.external_models.nested_reaudit import (
    DEFAULT_OUTPUT_NAME as DEFAULT_NESTED_REAUDIT_OUTPUT,
    NESTED_REAUDIT_SCHEMA_VERSION,
    NestedReauditError,
    audit_nested_external_ensemble,
)
from rogii.external_models.targets import SmoothTargetConfig
from rogii.models.heel_registered_anchor_student import (
    HeelRegisteredAnchorStudentModel,
)
from rogii.portfolio_metrics import validate_oof_frame


def _write_dataset(
    root: Path,
) -> tuple[Path, Path, pd.DataFrame]:
    data_root = root / "competition"
    train_root = data_root / "train"
    train_root.mkdir(parents=True)
    fold_records: list[dict[str, object]] = []
    source_records: list[pd.DataFrame] = []
    rows = 40
    known = 16
    for fold in range(5):
        for member in range(2):
            well_id = f"{fold + 1:02d}{member + 1:06d}"
            md = np.cumsum(np.linspace(4.0, 6.0, rows))
            offset = 2.5 * fold + 0.4 * member
            truth = 1_000.0 + offset + 0.16 * md + 0.0006 * md**2
            tvt_input = truth.copy()
            tvt_input[known:] = np.nan
            horizontal = pd.DataFrame(
                {
                    "MD": md,
                    "X": 500.0 + offset + 0.82 * md,
                    "Y": 1_500.0 - 0.23 * md,
                    "Z": -8_000.0 + 0.11 * md,
                    "GR": 68.0 + 8.0 * np.sin(truth / 14.0 + 0.15 * member),
                    "TVT_input": tvt_input,
                    "TVT": truth,
                }
            )
            horizontal.to_csv(
                train_root / f"{well_id}__horizontal_well.csv",
                index=False,
            )
            typewell_tvt = np.linspace(900.0, 1_200.0, 101)
            pd.DataFrame(
                {
                    "TVT": typewell_tvt,
                    "GR": 68.0 + 8.0 * np.sin(typewell_tvt / 14.0 + 0.15 * member),
                }
            ).to_csv(
                train_root / f"{well_id}__typewell.csv",
                index=False,
            )
            fold_records.append(
                {
                    "well_id": well_id,
                    "fold": fold,
                    "fold_version": "nested-ensemble-smoke-v1",
                }
            )
            suffix = np.arange(known, rows, dtype=np.int64)
            source_records.append(
                pd.DataFrame(
                    {
                        "well_id": well_id,
                        "row_index": suffix,
                        "fold": fold,
                        "y_true": truth[known:],
                        INCUMBENT_COLUMN: truth[known:]
                        + 2.0
                        + 0.04 * np.arange(rows - known),
                        TCN_COLUMN: truth[known:]
                        - 1.0
                        + 0.02 * np.arange(rows - known),
                    }
                )
            )
    fold_config = root / "folds.csv"
    pd.DataFrame(fold_records).to_csv(fold_config, index=False)
    return data_root, fold_config, pd.concat(source_records, ignore_index=True)


def _smoke_policy() -> NestedRunPolicy:
    return NestedRunPolicy(
        evidence_mode="smoke",
        outer_folds=(0,),
        tcn_seeds=(41,),
        incumbent_seeds=(43,),
        grid_config=FixedGridConfig(horizontal_points=16, typewell_points=16),
        target_config=SmoothTargetConfig(
            retained_dct_modes=4,
            maximum_absolute_correction=80.0,
        ),
        epochs=1,
        hidden_dim=4,
        layers=1,
        bootstrap_samples=20,
        shadow_samples=20,
        shadow_subset_wells=2,
    )


def test_nested_incumbent_lock_delegates_to_shared_audited_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "synthetic-incumbent.csv.gz"
    expected_record = {"sha256": "a" * 64}
    calls: list[Path] = []

    def fake_shared_loader(path: Path):
        calls.append(path)
        return (
            pd.DataFrame(
                {
                    "well_id": pd.Series([101], dtype="Int64"),
                    "row_index": [0],
                    "fold": [0],
                    "y_true": [1.0],
                    INCUMBENT_COLUMN: [1.5],
                }
            ),
            expected_record,
        )

    monkeypatch.setattr(
        nested_module,
        "load_audited_incumbent_reference",
        fake_shared_loader,
    )
    frame, record = nested_module._locked_incumbent_reference(source)
    assert calls == [source]
    assert frame["well_id"].tolist() == ["101"]
    assert record is expected_record


def test_production_policy_and_runtime_estimate_are_predeclared() -> None:
    policy = NestedRunPolicy()
    assert policy.outer_folds == (0, 1, 2, 3, 4)
    assert len(policy.tcn_seeds) == 5
    assert len(policy.incumbent_seeds) == 5
    assert policy.epochs == 60
    assert policy.model_fit_safe
    assert policy.weight_fit_safe
    assert not policy.family_selection_safe
    assert not policy.evidence_eligible
    assert REGULARIZATION_STRENGTH == 25.0
    estimate = estimate_nested_tcn_runtime()
    assert estimate["tcn_checkpoints"] == 100
    assert estimate["tcn_physical_fits"] == 50
    assert estimate["incumbent_checkpoints"] == 100
    assert estimate["incumbent_physical_fits"] == 50
    assert estimate["tcn_reciprocal_aliases"] == 50
    assert estimate["estimated_hours"] > 0.0

    with pytest.raises(ValueError, match="production policy is locked"):
        NestedRunPolicy(outer_folds=(0,))
    with pytest.raises(ValueError, match="production policy is locked"):
        NestedRunPolicy(epochs=59)
    with pytest.raises(ValueError, match="production policy is locked"):
        NestedRunPolicy(bootstrap_samples=100)


def test_smoke_source_rejects_duplicate_or_nonfinite_predictions() -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["a", "a"],
            "row_index": [1, 2],
            "fold": [0, 0],
            "y_true": [1.0, 2.0],
            INCUMBENT_COLUMN: [1.5, 2.5],
            TCN_COLUMN: [1.25, 2.25],
        }
    )
    source = make_smoke_outer_tcn_source(frame)
    assert not source.evidence_eligible
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        make_smoke_outer_tcn_source(duplicate)
    nonfinite = frame.copy()
    nonfinite.loc[0, TCN_COLUMN] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        make_smoke_outer_tcn_source(nonfinite)


def test_tcn_reciprocal_checkpoint_reuse_is_exact_and_resumable(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    seed = 41
    rng = np.random.default_rng(20260729)
    x = rng.normal(
        size=(4, len(MODEL_INPUT_CHANNEL_NAMES), 16),
    ).astype(np.float32)
    targets = rng.normal(size=(4, 16)).astype(np.float32)
    row_counts = np.asarray([7, 9, 11, 13], dtype=np.int64)
    training_ids = tuple(f"training-{index}" for index in range(len(x)))
    config = nested_module._tcn_config(
        seed=seed,
        device="cpu",
        epochs=1,
        hidden_dim=4,
        layers=1,
    )
    grid_config = FixedGridConfig(horizontal_points=16, typewell_points=16)
    target_config = SmoothTargetConfig(
        retained_dct_modes=4,
        maximum_absolute_correction=80.0,
    )
    common = {
        "seed": seed,
        "training_ids": training_ids,
        "x": x,
        "targets": targets,
        "row_counts": row_counts,
        "config": config,
        "data_manifest": {"digest": "toy-data"},
        "implementation_hashes": {"runner": "a" * 64},
        "grid_config": grid_config,
        "target_config": target_config,
    }
    source_provenance = nested_module._checkpoint_provenance(
        outer_fold=0,
        inner_fold=1,
        validation_ids=("fold-1",),
        excluded_outer_ids=("fold-0",),
        incumbent_checkpoint_sha256=("b" * 64,),
        **common,
    )
    reciprocal_provenance = nested_module._checkpoint_provenance(
        outer_fold=1,
        inner_fold=0,
        validation_ids=("fold-0",),
        excluded_outer_ids=("fold-1",),
        # Oriented incumbent checkpoint bytes differ because fold provenance
        # differs; exact model inputs/targets above are the numerical contract.
        incumbent_checkpoint_sha256=("c" * 64,),
        **common,
    )
    source_path, source_sidecar = nested_module._checkpoint_paths(
        run_root,
        outer_fold=0,
        inner_fold=1,
        seed=seed,
    )
    reciprocal_path, reciprocal_sidecar = nested_module._checkpoint_paths(
        run_root,
        outer_fold=1,
        inner_fold=0,
        seed=seed,
    )
    source_state, source_record = nested_module._fit_or_load_tcn(
        source_path,
        source_sidecar,
        run_root=run_root,
        x=x,
        targets=targets,
        channel_names=MODEL_INPUT_CHANNEL_NAMES,
        training_ids=training_ids,
        row_counts=row_counts,
        config=config,
        provenance=source_provenance,
        resume=False,
        reciprocal_source=(reciprocal_path, reciprocal_sidecar),
    )
    reciprocal_state, reciprocal_record = nested_module._fit_or_load_tcn(
        reciprocal_path,
        reciprocal_sidecar,
        run_root=run_root,
        x=x,
        targets=targets,
        channel_names=MODEL_INPUT_CHANNEL_NAMES,
        training_ids=training_ids,
        row_counts=row_counts,
        config=config,
        provenance=reciprocal_provenance,
        resume=False,
        reciprocal_source=(source_path, source_sidecar),
    )
    assert source_record["reciprocal_state_reuse"] is None
    assert reciprocal_record["reciprocal_state_reuse"] is not None
    assert source_record["checkpoint_sha256"] == reciprocal_record["checkpoint_sha256"]
    validated_records, validated_paths = (
        nested_reaudit_module._validate_tcn_checkpoints(
            run_root,
            [reciprocal_record],
            outer_fold=1,
            inner_fold=0,
            training_ids=training_ids,
            validation_ids=("fold-0",),
            outer_ids=("fold-1",),
            seeds=(seed,),
            run_config={
                "implementation_hashes": common["implementation_hashes"],
                "policy": {"epochs": 1, "hidden_dim": 4, "layers": 1},
            },
            data_manifest=common["data_manifest"],
        )
    )
    assert validated_records[0]["reciprocal_state_reuse"] is not None
    assert reciprocal_path.resolve() in validated_paths
    assert reciprocal_sidecar.resolve() in validated_paths
    np.testing.assert_array_equal(
        source_state.predict(
            x,
            channel_names=MODEL_INPUT_CHANNEL_NAMES,
            ids=training_ids,
        ),
        reciprocal_state.predict(
            x,
            channel_names=MODEL_INPUT_CHANNEL_NAMES,
            ids=training_ids,
        ),
    )

    resumed_state, resumed_record = nested_module._fit_or_load_tcn(
        reciprocal_path,
        reciprocal_sidecar,
        run_root=run_root,
        x=x,
        targets=targets,
        channel_names=MODEL_INPUT_CHANNEL_NAMES,
        training_ids=training_ids,
        row_counts=row_counts,
        config=config,
        provenance=reciprocal_provenance,
        resume=True,
        reciprocal_source=(source_path, source_sidecar),
    )
    assert resumed_record["resumed"]
    np.testing.assert_array_equal(
        reciprocal_state.predict(
            x,
            channel_names=MODEL_INPUT_CHANNEL_NAMES,
            ids=training_ids,
        ),
        resumed_state.predict(
            x,
            channel_names=MODEL_INPUT_CHANNEL_NAMES,
            ids=training_ids,
        ),
    )

    with source_sidecar.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(
        nested_module.NestedExternalEnsembleError,
        match="reciprocal source checkpoint/sidecar is stale",
    ):
        nested_module._fit_or_load_tcn(
            reciprocal_path,
            reciprocal_sidecar,
            run_root=run_root,
            x=x,
            targets=targets,
            channel_names=MODEL_INPUT_CHANNEL_NAMES,
            training_ids=training_ids,
            row_counts=row_counts,
            config=config,
            provenance=reciprocal_provenance,
            resume=True,
            reciprocal_source=(source_path, source_sidecar),
        )


def test_end_to_end_nested_tcn_smoke_is_fold_safe_and_resumable(
    tmp_path: Path,
) -> None:
    data_root, fold_config, source_frame = _write_dataset(tmp_path)
    source = make_smoke_outer_tcn_source(source_frame)
    before = {
        path.relative_to(data_root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in data_root.rglob("*")
        if path.is_file()
    }
    result = run_nested_external_ensemble(
        data_root=data_root,
        fold_config=fold_config,
        output_root=tmp_path / "runs",
        run_id="nested-smoke",
        cache_root=tmp_path / "cache",
        source=source,
        policy=_smoke_policy(),
        workers=1,
        incumbent_model=HeelRegisteredAnchorStudentModel(student_kind="mean"),
    )
    assert result["completed"]
    assert result["evidence_mode"] == "smoke"
    assert not result["evidence_eligible"]
    assert result["outer_folds"] == [0]
    assert result["inner_splits"] == 4
    assert result["tcn_checkpoints"] == 4
    assert result["tcn_physical_fits"] == 4
    assert result["tcn_reciprocal_aliases"] == 0
    assert result["incumbent_checkpoints"] == 4
    assert result["incumbent_refits"] == 4
    assert result["incumbent_reciprocal_aliases"] == 0
    assert not result["submission_created"]

    run_root = Path(result["run_root"])
    oof = pd.read_csv(
        run_root / "oof_predictions.csv.gz",
        dtype={"well_id": "string"},
        float_precision="round_trip",
    )
    validate_oof_frame(
        oof,
        (
            INCUMBENT_COLUMN,
            TCN_COLUMN,
            NESTED_COLUMN,
            FIXED_075_COLUMN,
            FIXED_050_COLUMN,
        ),
        expected_keys=source_frame.loc[source_frame["fold"].eq(0)],
        expected_folds=(0,),
    )
    expected_outer = source.frame.loc[source.frame["fold"].eq(0)].sort_values(
        ["well_id", "row_index", "fold"],
        kind="stable",
    )
    np.testing.assert_array_equal(
        oof[INCUMBENT_COLUMN].to_numpy(dtype=np.float64),
        expected_outer[INCUMBENT_COLUMN].to_numpy(dtype=np.float64),
    )
    np.testing.assert_array_equal(
        oof[TCN_COLUMN].to_numpy(dtype=np.float64),
        expected_outer[TCN_COLUMN].to_numpy(dtype=np.float64),
    )
    np.testing.assert_allclose(
        oof[FIXED_075_COLUMN],
        0.75 * oof[INCUMBENT_COLUMN] + 0.25 * oof[TCN_COLUMN],
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        oof[FIXED_050_COLUMN],
        0.5 * oof[INCUMBENT_COLUMN] + 0.5 * oof[TCN_COLUMN],
        rtol=0.0,
        atol=1.0e-10,
    )
    weights = pd.read_csv(run_root / "nested_weights.csv")
    assert set(weights["family"]) == {"incumbent", "tcn"}
    assert weights["weight"].sum() == pytest.approx(1.0)
    assert weights.loc[weights["family"].eq("tcn"), "weight"].iat[0] <= 0.5
    progress = json.loads((run_root / "progress.json").read_text(encoding="utf-8"))
    assert len(progress["completed_inner_splits"]) == 4
    assert progress["completed_outer_folds"] == [0]
    assert list((tmp_path / "cache" / "external_sequence_v1").rglob("*.npz"))
    assert not (tmp_path / "cache" / "nested_incumbent").exists()
    model_manifest = json.loads(
        (run_root / "model_manifest.json").read_text(encoding="utf-8")
    )
    assert model_manifest["physical_fit_count"] == 4
    assert model_manifest["reciprocal_alias_count"] == 0
    assert model_manifest["incumbent_physical_fit_count"] == 4
    assert model_manifest["incumbent_reciprocal_alias_count"] == 0

    fold_map = pd.read_csv(fold_config, dtype={"well_id": "string"})
    assigned = fold_map.set_index("well_id")["fold"].astype(int).to_dict()
    for manifest_path in sorted((run_root / "inner_oof").rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outer = int(manifest["outer_fold"])
        inner = int(manifest["inner_fold"])
        assert outer == 0
        assert all(
            assigned[well_id] not in {outer, inner}
            for well_id in manifest["training_well_ids"]
        )
        assert all(
            assigned[well_id] == inner for well_id in manifest["validation_well_ids"]
        )
        assert not manifest["training_validation_intersection"]
        assert not manifest["training_outer_intersection"]
        assert manifest["truth_attached_only_after_all_seed_predictions"]
        for checkpoint in manifest["tcn_checkpoints"]:
            provenance = checkpoint["provenance"]
            assert provenance["training_well_ids"] == manifest["training_well_ids"]
            assert provenance["outer_validation_labels_used_for_fit"] is False
            assert provenance["inner_validation_labels_used_for_fit"] is False

    after = {
        path.relative_to(data_root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in data_root.rglob("*")
        if path.is_file()
    }
    assert after == before
    with pytest.raises(ValueError, match="completed nested ensemble run"):
        run_nested_external_ensemble(
            data_root=data_root,
            fold_config=fold_config,
            output_root=tmp_path / "runs",
            run_id="nested-smoke",
            cache_root=tmp_path / "cache",
            source=source,
            policy=_smoke_policy(),
            workers=1,
            resume=True,
            incumbent_model=HeelRegisteredAnchorStudentModel(student_kind="mean"),
        )


def test_independent_nested_reaudit_recomputes_disk_evidence(
    tmp_path: Path,
) -> None:
    data_root, fold_config, source_frame = _write_dataset(tmp_path)
    source = make_smoke_outer_tcn_source(source_frame)
    source_path = tmp_path / "locked-smoke-source.csv.gz"
    source.frame.to_csv(
        source_path,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "mtime": 0},
    )
    result = run_nested_external_ensemble(
        data_root=data_root,
        fold_config=fold_config,
        output_root=tmp_path / "runs",
        run_id="nested-reaudit-smoke",
        cache_root=tmp_path / "cache",
        source=source,
        policy=_smoke_policy(),
        workers=1,
        incumbent_model=HeelRegisteredAnchorStudentModel(student_kind="mean"),
    )
    run_root = Path(result["run_root"])

    report = audit_nested_external_ensemble(
        run_root,
        source_oof=source_path,
    )

    assert report["schema_version"] == NESTED_REAUDIT_SCHEMA_VERSION
    assert report["passed"] is True
    assert report["evidence_mode"] == "smoke"
    assert report["evidence_eligible"] is False
    assert report["scope"]["outer_folds"] == [0]
    assert report["scope"]["inner_splits"] == 4
    assert report["scope"]["rows"] == 48
    assert report["checks"]["weights_recomputed_from_inner_oof"] is True
    assert report["checks"]["metrics_recomputed_from_final_oof"] is True
    assert report["checks"]["runner_artifact_audit_hash_anchored_by_summary"] is True
    assert report["checks"]["submission_created"] is False
    assert (
        report["numerical_comparison"]["outer_prediction_maximum_absolute_delta"]
        <= 1.0e-9
    )
    assert (run_root / DEFAULT_NESTED_REAUDIT_OUTPUT).is_file()

    def write_json(path: Path, payload: dict[str, object]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    metrics_path = run_root / "metrics" / "summary.json"
    audit_path = run_root / "artifact_audit.json"
    summary_path = run_root / "summary.json"
    metrics_backup = metrics_path.read_bytes()
    audit_backup = audit_path.read_bytes()
    summary_backup = summary_path.read_bytes()
    try:
        metrics = json.loads(metrics_backup)
        nested_metric = next(
            row for row in metrics["pooled"] if row["candidate"] == NESTED_COLUMN
        )
        nested_metric["pooled_row_rmse"] += 0.25
        write_json(metrics_path, metrics)
        artifact_audit = json.loads(audit_backup)
        artifact_audit["metrics_sha256"] = sha256_file(metrics_path)
        write_json(audit_path, artifact_audit)
        summary = json.loads(summary_backup)
        summary["metrics"] = metrics
        summary["artifact_audit_sha256"] = sha256_file(audit_path)
        write_json(summary_path, summary)
        with pytest.raises(NestedReauditError, match="metrics.summary"):
            audit_nested_external_ensemble(
                run_root,
                source_oof=source_path,
                output_name="tampered-metrics-audit.json",
            )
    finally:
        metrics_path.write_bytes(metrics_backup)
        audit_path.write_bytes(audit_backup)
        summary_path.write_bytes(summary_backup)

    weights_path = run_root / "nested_weights.csv"
    weights_backup = weights_path.read_bytes()
    audit_backup = audit_path.read_bytes()
    summary_backup = summary_path.read_bytes()
    try:
        weights = pd.read_csv(weights_path, float_precision="round_trip")
        weights.loc[weights["family"].eq("incumbent"), "weight"] += 0.01
        weights.loc[weights["family"].eq("tcn"), "weight"] -= 0.01
        weights.to_csv(weights_path, index=False, float_format="%.17g")
        artifact_audit = json.loads(audit_backup)
        artifact_audit["weights_sha256"] = sha256_file(weights_path)
        write_json(audit_path, artifact_audit)
        summary = json.loads(summary_backup)
        summary["weights"] = weights.to_dict("records")
        summary["artifact_audit_sha256"] = sha256_file(audit_path)
        write_json(summary_path, summary)
        with pytest.raises(NestedReauditError, match="nested_weights"):
            audit_nested_external_ensemble(
                run_root,
                source_oof=source_path,
                output_name="tampered-weights-audit.json",
            )
    finally:
        weights_path.write_bytes(weights_backup)
        audit_path.write_bytes(audit_backup)
        summary_path.write_bytes(summary_backup)

    final_path = run_root / "oof_predictions.csv.gz"
    final_backup = final_path.read_bytes()
    audit_backup = audit_path.read_bytes()
    summary_backup = summary_path.read_bytes()
    try:
        final = pd.read_csv(
            final_path,
            dtype={"well_id": "string", "row_index": "int64", "fold": "int64"},
            float_precision="round_trip",
        )
        for column in (
            INCUMBENT_COLUMN,
            TCN_COLUMN,
            NESTED_COLUMN,
            FIXED_075_COLUMN,
            FIXED_050_COLUMN,
        ):
            final[column] += 1.0
        final.to_csv(
            final_path,
            index=False,
            float_format="%.17g",
            compression={"method": "gzip", "mtime": 0},
        )
        artifact_audit = json.loads(audit_backup)
        artifact_audit["oof_sha256"] = sha256_file(final_path)
        write_json(audit_path, artifact_audit)
        summary = json.loads(summary_backup)
        summary["oof_sha256"] = sha256_file(final_path)
        summary["artifact_audit_sha256"] = sha256_file(audit_path)
        write_json(summary_path, summary)
        with pytest.raises(
            NestedReauditError,
            match="final versus locked outer source",
        ):
            audit_nested_external_ensemble(
                run_root,
                source_oof=source_path,
                output_name="tampered-prediction-audit.json",
            )
    finally:
        final_path.write_bytes(final_backup)
        audit_path.write_bytes(audit_backup)
        summary_path.write_bytes(summary_backup)

    audit_backup = audit_path.read_bytes()
    try:
        artifact_audit = json.loads(audit_backup)
        artifact_audit["untrusted_note"] = "changes runner-audit bytes only"
        write_json(audit_path, artifact_audit)
        with pytest.raises(
            NestedReauditError,
            match="artifact_audit_sha256 differs",
        ):
            audit_nested_external_ensemble(
                run_root,
                source_oof=source_path,
                output_name="tampered-runner-audit.json",
            )
    finally:
        audit_path.write_bytes(audit_backup)

    checkpoint = next((run_root / "models" / "tcn").rglob("*.pt"))
    checkpoint_backup = checkpoint.read_bytes()
    try:
        checkpoint.write_bytes(checkpoint_backup + b"tamper")
        with pytest.raises(NestedReauditError, match="SHA-256"):
            audit_nested_external_ensemble(
                run_root,
                source_oof=source_path,
                output_name="tampered-checkpoint-audit.json",
            )
    finally:
        checkpoint.write_bytes(checkpoint_backup)

    submission = run_root / "submission.csv"
    submission.write_text("id,TVD\n1,2\n", encoding="utf-8")
    try:
        with pytest.raises(NestedReauditError, match="submission CSV"):
            audit_nested_external_ensemble(
                run_root,
                source_oof=source_path,
                output_name="submission-present-audit.json",
            )
    finally:
        submission.unlink()


def test_nested_reaudit_cli_help_is_importable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_nested_external_ensemble.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-root" in result.stdout
    assert "--source-oof" in result.stdout
    assert "--incumbent-oof" in result.stdout


def test_nested_runner_cli_exposes_read_only_preflight() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_nested_external_ensemble.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--estimate-only" in result.stdout
    assert "--preflight-only" in result.stdout
