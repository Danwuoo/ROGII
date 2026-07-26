from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.cli import main
from rogii.diagnostics.slp_e1 import run_slp_e1_diagnostic
from rogii.stratigraphic.loop_edges import generate_loop_edges


EXPECTED_ARTIFACTS = {
    "summary.json",
    "run_config.json",
    "fold_manifest.json",
    "descriptor_config.json",
    "thresholds.json",
    "edge_metrics.csv",
    "operating_points.csv",
    "calibration_bins.csv",
    "per_well_metrics.csv.gz",
    "edges_oof.csv.gz",
    "negative_controls.csv",
    "runtime.json",
    "leakage_check.json",
}


def _synthetic_e1_data(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "competition-data"
    train = data_root / "train"
    train.mkdir(parents=True)
    mapping_rows = []
    for index in range(10):
        well_id = f"well-{index:02d}"
        rows = 460 + index
        known = 160 + index
        md = np.arange(rows, dtype=np.float64)
        latent = 100.0 + np.mod(md + 3 * index, 140.0) * 0.25
        gr = (
            np.sin(latent / 2.7)
            + 0.35 * np.sin(latent / 0.91)
            + 0.01 * np.mod(md, 140.0)
        )
        gr[(np.arange(rows) + index) % 41 == 0] = np.nan
        tvt_input = latent.copy()
        tvt_input[known:] = np.nan
        horizontal = pd.DataFrame(
            {
                "MD": md,
                "X": 10_000.0 + md,
                "Y": 20_000.0 + md,
                "Z": -500.0 - 0.5 * md,
                "GR": gr,
                "TVT_input": tvt_input,
                "TVT": latent,
            }
        )
        horizontal.to_csv(
            train / f"{well_id}__horizontal_well.csv", index=False
        )
        type_coordinate = np.arange(95.0, 145.0, 0.25)
        type_gr = (
            np.sin(type_coordinate / 2.7)
            + 0.35 * np.sin(type_coordinate / 0.91)
        )
        pd.DataFrame({"TVT": type_coordinate, "GR": type_gr}).to_csv(
            train / f"{well_id}__typewell.csv", index=False
        )
        mapping_rows.append(
            {
                "well_id": well_id,
                "fold": index % 5,
                "fold_version": "synthetic_5fold_v1",
            }
        )
    fold_config = tmp_path / "folds.csv"
    pd.DataFrame(mapping_rows).to_csv(fold_config, index=False)
    return data_root, fold_config


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def _run_small(
    data_root: Path, fold_config: Path, output_root: Path, run_id: str, workers: int
) -> Path:
    result = run_slp_e1_diagnostic(
        profile="quick",
        data_root=data_root,
        fold_config=fold_config,
        output_root=output_root,
        run_id=run_id,
        max_train_wells=6,
        max_validation_wells=2,
        max_query_rows_per_well=48,
        top_k=2,
        seed=1234,
        workers=workers,
    )
    return Path(result["run_root"])


def test_quick_cli_writes_artifacts_preserves_data_and_has_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root, fold_config = _synthetic_e1_data(tmp_path)
    before = _tree_hashes(data_root)
    exit_code = main(
        [
            "slp-e1",
            "--profile",
            "quick",
            "--data-root",
            str(data_root),
            "--fold-config",
            str(fold_config),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "synthetic-e1",
            "--max-train-wells",
            "6",
            "--max-validation-wells",
            "2",
            "--max-query-rows-per-well",
            "48",
            "--top-k",
            "2",
            "--workers",
            "2",
        ]
    )
    captured = capsys.readouterr()
    completion = json.loads(captured.out)
    run_root = Path(completion["run_root"])
    assert exit_code == 0
    assert EXPECTED_ARTIFACTS <= {
        path.name for path in run_root.iterdir() if path.is_file()
    }
    assert _tree_hashes(data_root) == before
    summary = json.loads((run_root / "summary.json").read_text("utf-8"))
    assert summary["audit_id"] == "SLP-E1"
    assert summary["uses_target_for_evaluation"] is True
    assert summary["uses_target_for_artificial_cut_calibration"] is True
    assert summary["edge_generator_inference_safe"] is True
    assert summary["deployable"] is False
    assert summary["submission_eligible"] is False
    assert summary["scope"] == "quick_or_custom"
    assert set(summary["edge_types"]) == {"PF", "FF", "HT"}
    assert {"shuffle_gr", "wrong_typewell"} <= set(summary["controls"])
    assert not any("submission" in path.name for path in run_root.iterdir())
    assert not any(path.suffix == ".pkl" for path in run_root.rglob("*"))


def test_existing_run_and_output_inside_data_fail_safely(tmp_path: Path) -> None:
    data_root, fold_config = _synthetic_e1_data(tmp_path)
    run_root = _run_small(
        data_root, fold_config, tmp_path / "runs", "duplicate", workers=1
    )
    assert run_root.is_dir()
    with pytest.raises(FileExistsError):
        _run_small(
            data_root, fold_config, tmp_path / "runs", "duplicate", workers=1
        )
    with pytest.raises(ValueError, match="cannot be inside competition data"):
        run_slp_e1_diagnostic(
            profile="quick",
            data_root=data_root,
            fold_config=fold_config,
            output_root=data_root / "unsafe",
            run_id="unsafe",
            max_train_wells=6,
            max_validation_wells=2,
            max_query_rows_per_well=16,
            top_k=1,
        )
    assert not (data_root / "unsafe").exists()


def test_workers_produce_same_numeric_artifacts(tmp_path: Path) -> None:
    data_root, fold_config = _synthetic_e1_data(tmp_path)
    one = _run_small(
        data_root, fold_config, tmp_path / "runs", "worker-1", workers=1
    )
    four = _run_small(
        data_root, fold_config, tmp_path / "runs", "worker-4", workers=4
    )
    for filename in (
        "edge_metrics.csv",
        "operating_points.csv",
        "calibration_bins.csv",
        "per_well_metrics.csv.gz",
        "edges_oof.csv.gz",
        "negative_controls.csv",
        "fold_manifest.json",
        "thresholds.json",
        "descriptor_config.json",
        "leakage_check.json",
    ):
        assert (one / filename).read_bytes() == (four / filename).read_bytes()


def test_role_sets_are_disjoint_and_complete(tmp_path: Path) -> None:
    data_root, fold_config = _synthetic_e1_data(tmp_path)
    run_root = _run_small(
        data_root, fold_config, tmp_path / "runs", "roles", workers=1
    )
    manifest = json.loads((run_root / "fold_manifest.json").read_text("utf-8"))
    fold = manifest["folds"]["0"]
    outer = set(fold["outer_training_wells"])
    confidence = set(fold["confidence_fit_wells"])
    threshold = set(fold["threshold_select_wells"])
    validation = set(fold["outer_validation_wells"])
    assert confidence.isdisjoint(threshold)
    assert outer == confidence | threshold
    assert outer.isdisjoint(validation)


def test_validation_label_is_not_an_edge_generator_argument() -> None:
    parameters = set(generate_loop_edges.__annotations__)
    assert "target" not in parameters
    assert "labels" not in parameters


def test_static_leakage_boundaries() -> None:
    stratigraphic = Path("src/rogii/stratigraphic")
    for path in stratigraphic.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any("diagnostics" in item for item in imports)
        assert "slp_e0" not in source
        assert "slp_e1" not in source
        assert not any(
            name in source for name in ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
        )

    production = [
        Path("src/rogii/runner.py"),
        Path("src/rogii/submission.py"),
        *Path("src/rogii/models").glob("*.py"),
    ]
    for path in production:
        source = path.read_text(encoding="utf-8")
        assert "diagnostics.slp_e1" not in source
        assert "slp_e1" not in source
