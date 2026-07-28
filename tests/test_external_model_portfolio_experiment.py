from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import rogii.diagnostics.external_model_portfolio_experiment as portfolio_module
from rogii.diagnostics.external_model_portfolio_experiment import (
    DEFAULT_QUICK_VARIANTS,
    HORIZONTAL_MODEL_CHANNELS,
    INCUMBENT_COLUMN,
    PortfolioVariant,
    TYPEWELL_MODEL_CHANNELS,
    _assemble_oof,
    _cached_incumbent,
    _load_fold_training_targets,
    _load_or_fit_state,
    _safe_run_root,
)
from rogii.external_models.controls import CONTROL_NO_CROSS
from rogii.external_models.data import MODEL_INPUT_CHANNEL_NAMES
from rogii.external_models.incumbent import IncumbentPrediction
from rogii.external_models.torch_families import (
    FAMILY_PAIRED_PATCH,
    FAMILY_TCN,
    TorchFamilyConfig,
)


def _baseline() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        well_id = f"well_{fold}"
        for row_index in range(2):
            rows.append(
                {
                    "well_id": well_id,
                    "row_index": row_index,
                    "fold": fold,
                    "y_true": float(fold + row_index),
                    INCUMBENT_COLUMN: float(fold + row_index + 1),
                }
            )
    return pd.DataFrame(rows)


def test_default_portfolio_has_four_families_and_required_controls() -> None:
    assert {variant.family for variant in DEFAULT_QUICK_VARIANTS} >= {
        "paired_patch_cross",
        "s4d",
        "fno1d",
        "tcn",
    }
    assert {variant.control_id for variant in DEFAULT_QUICK_VARIANTS} >= {
        "wrong_typewell",
        "pair_shuffle",
        "gr_order_shuffle",
        "phase_destroy",
        "no_cross_interaction",
    }
    assert any(variant.frozen_random_encoder for variant in DEFAULT_QUICK_VARIANTS)
    tcn_controls = {
        variant.control_id
        for variant in DEFAULT_QUICK_VARIANTS
        if variant.family == "tcn"
    }
    assert tcn_controls >= {
        "wrong_typewell",
        "pair_shuffle",
        "gr_order_shuffle",
        "phase_destroy",
    }
    assert set(HORIZONTAL_MODEL_CHANNELS) | set(TYPEWELL_MODEL_CHANNELS) == set(
        MODEL_INPUT_CHANNEL_NAMES
    )
    assert set(HORIZONTAL_MODEL_CHANNELS).isdisjoint(TYPEWELL_MODEL_CHANNELS)


def test_no_cross_variant_requires_disabled_cross_path() -> None:
    with pytest.raises(ValueError, match="disable"):
        PortfolioVariant(
            "bad_no_cross",
            FAMILY_PAIRED_PATCH,
            control_id=CONTROL_NO_CROSS,
            cross_series=True,
        )


def test_assemble_oof_averages_seeds_before_family_scoring(
    tmp_path: Path,
) -> None:
    baseline = _baseline()
    variant = PortfolioVariant("paired_patch_cross", FAMILY_PAIRED_PATCH)
    blocks = {}
    completed = {}
    for seed, correction in ((11, 2.0), (12, 4.0)):
        key = (variant.variant_id, seed)
        blocks[key] = []
        completed[key] = set(range(5))
        for fold in range(5):
            group = baseline.loc[baseline["fold"].eq(fold)]
            blocks[key].append(
                pd.DataFrame(
                    {
                        "well_id": group["well_id"].to_numpy(),
                        "row_index": group["row_index"].to_numpy(),
                        "fold": group["fold"].to_numpy(),
                        "prediction": group[INCUMBENT_COLUMN].to_numpy() + correction,
                        "correction_common": np.full(len(group), correction),
                        "correction_slope": np.zeros(len(group)),
                        "correction_shape": np.zeros(len(group)),
                    }
                )
            )
        if seed == 12:
            paths = []
            for fold, block in enumerate(blocks[key]):
                path = tmp_path / f"seed_{seed}_fold_{fold}.csv.gz"
                block.to_csv(path, index=False)
                paths.append(path)
            blocks[key] = paths
    output, predictions, families, incomplete = _assemble_oof(
        baseline,
        variants=(variant,),
        seeds=(11, 12),
        blocks=blocks,
        completed=completed,
        shrinkage=0.25,
    )
    assert incomplete == []
    assert "y_pred__paired_patch_cross" in families
    np.testing.assert_allclose(
        output["y_pred__paired_patch_cross"],
        baseline[INCUMBENT_COLUMN] + 3.0,
    )
    np.testing.assert_allclose(
        output["y_pred__paired_patch_cross__shrink_025"],
        baseline[INCUMBENT_COLUMN] + 0.75,
    )
    assert set(predictions).issubset(output)


def test_incomplete_seed_is_not_silently_scored() -> None:
    baseline = _baseline()
    variant = PortfolioVariant("paired_patch_cross", FAMILY_PAIRED_PATCH)
    output, predictions, families, incomplete = _assemble_oof(
        baseline,
        variants=(variant,),
        seeds=(11,),
        blocks={(variant.variant_id, 11): []},
        completed={(variant.variant_id, 11): {0, 1, 2, 3}},
        shrinkage=0.25,
    )
    assert predictions == [INCUMBENT_COLUMN]
    assert families == []
    assert incomplete == ["paired_patch_cross/seed_11"]
    assert list(output.columns) == list(baseline.columns)


def test_resume_requires_exact_immutable_run_config(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "runs"
    config = {
        "schema_version": "test",
        "run_id": "immutable",
        "data_manifest": {"digest": "a"},
        "implementation_hashes": {"runner": "b"},
    }
    root = _safe_run_root(
        output,
        "immutable",
        data,
        resume=False,
        run_config=config,
    )
    assert json.loads((root / "run_config.json").read_text(encoding="utf-8")) == config
    assert (
        _safe_run_root(
            output,
            "immutable",
            data,
            resume=True,
            run_config=config,
        )
        == root
    )
    changed = {**config, "data_manifest": {"digest": "different"}}
    with pytest.raises(ValueError, match="run_config differs"):
        _safe_run_root(
            output,
            "immutable",
            data,
            resume=True,
            run_config=changed,
        )
    (root / "run_config.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable JSON"):
        _safe_run_root(
            output,
            "immutable",
            data,
            resume=True,
            run_config=config,
        )


def test_fold_target_loader_never_opens_outer_validation_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_files = {
        "train_a": SimpleNamespace(well_id="train_a"),
        "train_b": SimpleNamespace(well_id="train_b"),
        "validation": SimpleNamespace(well_id="validation"),
    }
    prepared = {
        well_id: SimpleNamespace(row_index=np.arange(3, dtype=np.int64))
        for well_id in train_files
    }
    opened: list[str] = []

    def fake_loader(well: SimpleNamespace, row_index: np.ndarray) -> np.ndarray:
        opened.append(well.well_id)
        return np.asarray(row_index, dtype=np.float64)

    monkeypatch.setattr(portfolio_module, "_target_from_train_file", fake_loader)
    loaded = _load_fold_training_targets(
        train_files,
        prepared,
        training_ids=("train_a", "train_b"),
        validation_ids=("validation",),
    )
    assert list(loaded) == ["train_a", "train_b"]
    assert opened == ["train_a", "train_b"]
    with pytest.raises(ValueError, match="overlap"):
        _load_fold_training_targets(
            train_files,
            prepared,
            training_ids=("train_a", "validation"),
            validation_ids=("validation",),
        )


def _checkpoint_context() -> dict[str, object]:
    return {
        "variant_id": "tcn",
        "family": FAMILY_TCN,
        "control_id": "none",
        "fold": 0,
        "seed": 7,
        "validation_well_ids": ["validation"],
        "data_manifest": {"name_size_mtime_sha256": "data-a"},
        "implementation_hashes": {"runner": "implementation-a"},
        "incumbent": {
            "state_digest": "state-a",
            "manifest_sha256": "manifest-a",
            "training_predictions_sha256": "prediction-a",
        },
        "run_config_sha256": "run-a",
        "fixed_grid": {"config": {"horizontal_points": 8}},
        "residual_target": {"retained_dct_modes": 2},
    }


def test_checkpoint_sidecar_binds_arrays_data_implementation_and_incumbent(
    tmp_path: Path,
) -> None:
    random = np.random.default_rng(17)
    x = random.normal(size=(2, 3, 8)).astype(np.float32)
    targets = random.normal(size=(2, 8)).astype(np.float32)
    row_counts = np.asarray([11, 17], dtype=np.int64)
    names = ("a", "b", "c")
    ids = ("train_a", "train_b")
    config = TorchFamilyConfig(
        family=FAMILY_TCN,
        seed=7,
        epochs=1,
        hidden_dim=4,
        layers=1,
        tcn_kernel_size=3,
    )
    path = tmp_path / "models" / "fold_0.pt"
    context = _checkpoint_context()
    state, resumed, record = _load_or_fit_state(
        path,
        x=x,
        targets=targets,
        channel_names=names,
        training_ids=ids,
        row_counts=row_counts,
        config=config,
        provenance=context,
        resume=False,
    )
    assert not resumed
    assert state.manifest["row_count_sum"] == 28
    sidecar = path.with_suffix(".json")
    assert sidecar.is_file()
    assert record["checkpoint_sidecar_sha256"]

    _, resumed, _ = _load_or_fit_state(
        path,
        x=x,
        targets=targets,
        channel_names=names,
        training_ids=ids,
        row_counts=row_counts,
        config=config,
        provenance=context,
        resume=True,
    )
    assert resumed

    stale_cases: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = []
    changed_x = x.copy()
    changed_x[0, 0, 0] += 1.0
    stale_cases.append((changed_x, targets, row_counts, context))
    changed_targets = targets.copy()
    changed_targets[0, 0] += 1.0
    stale_cases.append((x, changed_targets, row_counts, context))
    changed_counts = row_counts.copy()
    changed_counts[0] += 1
    stale_cases.append((x, targets, changed_counts, context))
    for field, replacement in (
        ("data_manifest", {"name_size_mtime_sha256": "data-b"}),
        ("implementation_hashes", {"runner": "implementation-b"}),
        (
            "incumbent",
            {
                "state_digest": "state-b",
                "manifest_sha256": "manifest-a",
                "training_predictions_sha256": "prediction-a",
            },
        ),
    ):
        stale_context = {**context, field: replacement}
        stale_cases.append((x, targets, row_counts, stale_context))

    for stale_x, stale_targets, stale_counts, stale_context in stale_cases:
        with pytest.raises(ValueError, match="provenance differs"):
            _load_or_fit_state(
                path,
                x=stale_x,
                targets=stale_targets,
                channel_names=names,
                training_ids=ids,
                row_counts=stale_counts,
                config=config,
                provenance=stale_context,
                resume=True,
            )

    sidecar_bytes = sidecar.read_bytes()
    sidecar.write_bytes(b"{corrupt")
    with pytest.raises(ValueError, match="unreadable JSON"):
        _load_or_fit_state(
            path,
            x=x,
            targets=targets,
            channel_names=names,
            training_ids=ids,
            row_counts=row_counts,
            config=config,
            provenance=context,
            resume=True,
        )
    sidecar.write_bytes(sidecar_bytes)
    checkpoint_bytes = path.read_bytes()
    path.write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError, match="provenance differs"):
        _load_or_fit_state(
            path,
            x=x,
            targets=targets,
            channel_names=names,
            training_ids=ids,
            row_counts=row_counts,
            config=config,
            provenance=context,
            resume=True,
        )
    path.write_bytes(checkpoint_bytes)


def test_incumbent_cache_rejects_stale_provenance_and_corrupt_bytes(
    tmp_path: Path,
) -> None:
    rows = np.arange(4, dtype=np.int64)

    class Provider:
        seeds = (1, 2)

        def __init__(self) -> None:
            self.calls = 0

        def predict_outer_fold(
            self,
            well_id: str,
            *,
            outer_fold: int,
        ) -> IncumbentPrediction:
            self.calls += 1
            prediction = np.arange(4, dtype=np.float64) + 10.0 + self.calls
            return IncumbentPrediction(
                well_id=well_id,
                row_index=rows,
                prediction=prediction,
                fold_states=(outer_fold,),
                seeds=self.seeds,
                state_sha256=("state",),
            )

    provider = Provider()
    run_root = tmp_path / "run"
    provenance = {
        "raw_source_provenance_sha256": "raw-a",
        "data_manifest": {"digest": "data-a"},
        "implementation_hashes": {"incumbent": "implementation-a"},
        "incumbent_manifest_sha256": "manifest-a",
    }
    first = _cached_incumbent(
        provider,
        run_root=run_root,
        well_id="well",
        fold=0,
        expected_rows=rows,
        state_digest="state-a",
        cache_provenance=provenance,
        resume=False,
    )
    second = _cached_incumbent(
        provider,
        run_root=run_root,
        well_id="well",
        fold=0,
        expected_rows=rows,
        state_digest="state-a",
        cache_provenance=provenance,
        resume=True,
    )
    assert provider.calls == 1
    np.testing.assert_array_equal(first.prediction, second.prediction)

    stale = {**provenance, "data_manifest": {"digest": "data-b"}}
    refreshed = _cached_incumbent(
        provider,
        run_root=run_root,
        well_id="well",
        fold=0,
        expected_rows=rows,
        state_digest="state-a",
        cache_provenance=stale,
        resume=True,
    )
    assert provider.calls == 2
    assert not np.array_equal(second.prediction, refreshed.prediction)

    cache_path = run_root / "incumbent_cache" / "fold_0" / "well.npz"
    cache_path.write_bytes(b"corrupt-cache")
    _cached_incumbent(
        provider,
        run_root=run_root,
        well_id="well",
        fold=0,
        expected_rows=rows,
        state_digest="state-a",
        cache_provenance=stale,
        resume=True,
    )
    assert provider.calls == 3
