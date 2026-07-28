from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from rogii.external_models import visible_smoke
from rogii.external_models.visible_smoke import (
    resolve_visible_checkpoints,
    run_visible_test_smoke,
)


TRAINING_DIGEST = "a" * 64


def _write_visible_well(data_root: Path, well_id: str, offset: float) -> None:
    test = data_root / "test"
    test.mkdir(parents=True, exist_ok=True)
    md = np.arange(12, dtype=np.float64)
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": md + offset,
            "Y": np.sin(md / 4.0),
            "Z": -md,
            "GR": 60.0 + offset + np.cos(md / 3.0),
            "TVT_input": [
                10.0 + offset,
                11.0 + offset,
                12.0 + offset,
                13.0 + offset,
                *([np.nan] * 8),
            ],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(0.0, 160.0, 20),
            "GR": np.linspace(50.0 + offset, 80.0 + offset, 20),
        }
    )
    horizontal.to_csv(test / f"{well_id}__horizontal_well.csv", index=False)
    typewell.to_csv(test / f"{well_id}__typewell.csv", index=False)


def _make_smoke_tree(tmp_path: Path) -> tuple[Path, Path]:
    data_root = (tmp_path / "data").resolve()
    _write_visible_well(data_root, "aaaaaaaa", 0.0)
    _write_visible_well(data_root, "bbbbbbbb", 2.0)
    ids = [
        f"{well_id}_{row_index}"
        for well_id in ("aaaaaaaa", "bbbbbbbb")
        for row_index in range(4, 12)
    ]
    ids.reverse()
    pd.DataFrame({"id": ids, "tvt": np.zeros(len(ids))}).to_csv(
        data_root / "sample_submission.csv",
        index=False,
    )

    run_root = (tmp_path / "run").resolve()
    checkpoint = run_root / "models" / "tcn" / "seed_7" / "fold_0.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"test checkpoint")
    fold_config = (tmp_path / "folds.csv").resolve()
    fold_config.write_text("well_id,fold\n", encoding="utf-8")
    incumbent_root = (tmp_path / "incumbent").resolve()
    incumbent_root.mkdir()
    cache_root = (tmp_path / "cache").resolve()
    config = {
        "run_id": "unit-smoke",
        "profile": "quick",
        "data_root": str(data_root),
        "fold_config": str(fold_config),
        "incumbent_root": str(incumbent_root),
        "cache_root": str(cache_root),
        "variants": [
            {
                "variant_id": "tcn",
                "family": "tcn",
                "control_id": "none",
                "cross_series": True,
                "frozen_random_encoder": False,
            }
        ],
        "seeds": [7],
        "cohort": {
            "0": {
                "training_ids_sha256": TRAINING_DIGEST,
                "training_well_ids": ["train_a"],
                "validation_well_ids": ["valid_a"],
            }
        },
        "fixed_grid": {"config": {"horizontal_points": 8, "typewell_points": 8}},
        "residual_target": {
            "retained_dct_modes": 4,
            "maximum_absolute_correction": 80.0,
            "validation_labels_used_at_prediction": False,
        },
        "shrinkage": 0.25,
        "architecture_only": True,
        "external_pretrained_weights_used": [],
        "submission_created": False,
    }
    config["data_manifest"] = visible_smoke._data_manifest(data_root)
    (run_root / "run_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    return run_root, data_root


class _FakeIncumbentProvider:
    calls: list[tuple[str, tuple[int, ...], str]] = []

    def __init__(self, **_: object) -> None:
        self.seeds = (1, 2, 3, 4, 5)

    def predict_with_folds(
        self,
        well_id: str,
        *,
        folds: tuple[int, ...],
        split: str,
    ) -> SimpleNamespace:
        fold_tuple = tuple(folds)
        if len(fold_tuple) != 1 or split != "test":
            raise AssertionError("smoke must request one test fold at a time")
        self.calls.append((well_id, fold_tuple, split))
        fold = fold_tuple[0]
        row_index = np.arange(4, 12, dtype=np.int64)
        offset = 0.0 if well_id == "aaaaaaaa" else 5.0
        return SimpleNamespace(
            row_index=row_index,
            prediction=100.0 + 50.0 * fold + offset + row_index.astype(np.float64),
            fold_states=fold_tuple,
            seeds=self.seeds,
            state_sha256=tuple(
                f"{fold * len(self.seeds) + index:064x}"
                for index in range(len(self.seeds))
            ),
        )


class _FakeState:
    def __init__(self, *, nonfinite: bool = False) -> None:
        self.config = SimpleNamespace(
            family="tcn",
            seed=7,
            frozen_random_encoder=False,
        )
        self.manifest = {
            "family": "tcn",
            "seed": 7,
            "architecture_only": True,
            "fit_ids_sha256": TRAINING_DIGEST,
            "parameter_sha256": "b" * 64,
        }
        self.nonfinite = nonfinite

    def predict(
        self,
        x: np.ndarray,
        *,
        channel_names: tuple[str, ...],
        ids: tuple[str, ...],
    ) -> np.ndarray:
        assert len(channel_names) == x.shape[1]
        assert len(ids) == x.shape[0]
        values = np.full((len(x), x.shape[2]), 2.0, dtype=np.float64)
        if self.nonfinite:
            values[0, 0] = np.nan
        return values


def test_resolve_visible_checkpoints_requires_every_selected_file(
    tmp_path: Path,
) -> None:
    run_root, _ = _make_smoke_tree(tmp_path)
    config = json.loads((run_root / "run_config.json").read_text())
    with pytest.raises(ValueError, match="cohort is missing fold 4"):
        resolve_visible_checkpoints(
            run_root,
            config,
            variants=["tcn"],
            seeds=[7],
            folds=[4],
        )
    config["cohort"]["1"] = {
        "training_ids_sha256": TRAINING_DIGEST,
        "training_well_ids": ["train_b"],
        "validation_well_ids": ["valid_b"],
    }
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_visible_checkpoints(
            run_root,
            config,
            variants=["tcn"],
            seeds=[7],
            folds=[0, 1],
        )


def test_visible_smoke_preserves_sample_order_and_writes_json_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _ = _make_smoke_tree(tmp_path)
    _FakeIncumbentProvider.calls.clear()
    monkeypatch.setattr(
        visible_smoke,
        "HeelIncumbentProvider",
        _FakeIncumbentProvider,
    )
    monkeypatch.setattr(
        visible_smoke,
        "load_torch_family",
        lambda _: _FakeState(),
    )
    audit = run_visible_test_smoke(
        run_root=run_root,
        variants=["tcn"],
        seeds=[7],
        folds=[0],
        output_json=Path("audits/visible.json"),
    )
    output = run_root / "audits" / "visible.json"
    assert output.is_file()
    assert audit == json.loads(output.read_text(encoding="utf-8"))
    assert audit["test_wells"] == ["aaaaaaaa", "bbbbbbbb"]
    assert audit["prediction_rows"] == 16
    assert audit["sample_order_first_id"] == "bbbbbbbb_11"
    assert audit["sample_order_last_id"] == "aaaaaaaa_4"
    assert audit["checkpoint_count"] == 1
    assert audit["key_checks"] == {
        "dynamic_test_enumeration": True,
        "sample_contract_exact": True,
        "duplicate_ids": 0,
        "coverage_rows": 16,
        "sample_order_preserved_by_explicit_id_join": True,
        "finite_incumbent": True,
        "finite_checkpoint_outputs": True,
        "finite_smoke_average": True,
        "fold_specific_incumbent_conditioning": True,
    }
    assert _FakeIncumbentProvider.calls == [
        ("aaaaaaaa", (0,), "test"),
        ("bbbbbbbb", (0,), "test"),
    ]
    assert audit["labels_read"] is False
    assert audit["submission_csv_created"] is False
    assert audit["data_manifest_equal"] is True
    assert not list(run_root.rglob("*.csv"))


def test_visible_smoke_conditions_each_checkpoint_on_its_own_fold_incumbent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _ = _make_smoke_tree(tmp_path)
    config_path = run_root / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["cohort"]["1"] = {
        "training_ids_sha256": TRAINING_DIGEST,
        "training_well_ids": ["train_b"],
        "validation_well_ids": ["valid_b"],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    second = run_root / "models" / "tcn" / "seed_7" / "fold_1.pt"
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(b"test checkpoint fold one")

    _FakeIncumbentProvider.calls.clear()
    monkeypatch.setattr(
        visible_smoke,
        "HeelIncumbentProvider",
        _FakeIncumbentProvider,
    )
    monkeypatch.setattr(
        visible_smoke,
        "load_torch_family",
        lambda _: _FakeState(),
    )
    audit = run_visible_test_smoke(
        run_root=run_root,
        variants=["tcn"],
        seeds=[7],
        folds=[0, 1],
    )

    assert _FakeIncumbentProvider.calls == [
        ("aaaaaaaa", (0,), "test"),
        ("bbbbbbbb", (0,), "test"),
        ("aaaaaaaa", (1,), "test"),
        ("bbbbbbbb", (1,), "test"),
    ]
    assert audit["incumbent"]["folds"] == [0, 1]
    assert audit["incumbent"]["state_count"] == 10
    assert {
        record["incumbent_conditioning_fold"] for record in audit["checkpoints"]
    } == {0, 1}
    checkpoint_incumbent_hashes = {
        record["fold"]: record["ordered_incumbent_sha256"]
        for record in audit["checkpoints"]
    }
    assert checkpoint_incumbent_hashes[0] != checkpoint_incumbent_hashes[1]
    assert audit["key_checks"]["fold_specific_incumbent_conditioning"] is True


def test_visible_smoke_fails_closed_before_writing_on_nonfinite_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _ = _make_smoke_tree(tmp_path)
    _FakeIncumbentProvider.calls.clear()
    monkeypatch.setattr(
        visible_smoke,
        "HeelIncumbentProvider",
        _FakeIncumbentProvider,
    )
    monkeypatch.setattr(
        visible_smoke,
        "load_torch_family",
        lambda _: _FakeState(nonfinite=True),
    )
    output = run_root / "should_not_exist.json"
    with pytest.raises(ValueError, match="invalid grid predictions"):
        run_visible_test_smoke(
            run_root=run_root,
            variants=["tcn"],
            seeds=[7],
            folds=[0],
            output_json=output,
        )
    assert not output.exists()


def test_visible_smoke_rejects_output_outside_run_root(tmp_path: Path) -> None:
    run_root, _ = _make_smoke_tree(tmp_path)
    with pytest.raises(ValueError, match="inside run root"):
        run_visible_test_smoke(
            run_root=run_root,
            variants=["tcn"],
            seeds=[7],
            folds=[0],
            output_json=tmp_path / "outside.json",
        )
