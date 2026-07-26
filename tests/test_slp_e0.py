from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.cli import main
from rogii.diagnostics.slp_e0 import (
    UnsupportedOracleComplexity,
    accumulate_pooled_metric,
    accumulate_recurrence_diagnostic,
    fit_continuous_piecewise,
    fit_discontinuous_piecewise,
    fit_polynomial_oracle,
    nearest_prefix_distance,
    run_slp_e0_diagnostic,
    surface_identity_stats,
)


FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
EXPECTED_ARTIFACTS = {
    "summary.json",
    "run_config.json",
    "oracle_metrics.csv",
    "per_well_oracles.csv.gz",
    "recurrence_summary.json",
    "recurrence_per_well.csv.gz",
    "surface_identity.csv",
}


def _write_synthetic_well(
    data_root: Path,
    *,
    well_id: str,
    rows: int = 52,
    known_count: int = 8,
) -> Path:
    train = data_root / "train"
    train.mkdir(parents=True, exist_ok=True)
    md = np.arange(rows, dtype=np.float64)
    tvt = 1000.0 + 0.025 * md**2 + 0.2 * np.sin(md / 3.0)
    h = 250.0 + 1.75 * md
    z = h - tvt
    tvt_input = tvt.copy()
    tvt_input[known_count:] = np.nan
    frame = pd.DataFrame(
        {
            "MD": md,
            "X": 10_000.0 + md,
            "Y": 20_000.0 + md,
            "Z": z,
            "TVT": tvt,
            "GR": np.where(md % 7 == 0, np.nan, 80.0 + np.cos(md)),
            "TVT_input": tvt_input,
        }
    )
    for index, formation in enumerate(FORMATION_COLUMNS):
        frame[formation] = h - (100.0 + index * 10.0)
    horizontal = train / f"{well_id}__horizontal_well.csv"
    typewell = train / f"{well_id}__typewell.csv"
    frame.to_csv(horizontal, index=False)
    pd.DataFrame({"TVT": tvt, "GR": 80.0 + np.cos(md)}).to_csv(
        typewell, index=False
    )
    return horizontal


def _synthetic_data_root(tmp_path: Path, *, wells: int = 2) -> Path:
    data_root = tmp_path / "competition-data"
    for index in range(wells):
        _write_synthetic_well(
            data_root,
            well_id=f"well-{index:02d}",
            rows=52 + index * 4,
            known_count=8 + index,
        )
    return data_root


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def test_linear_tvt_polynomial_degree_one_is_exact() -> None:
    t = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    fit = fit_polynomial_oracle(t, 4.25 - 7.5 * t, degree=1)
    assert fit.rmse < 1e-12
    assert fit.predictions.dtype == np.float64


def test_known_two_segment_continuous_hinge_path_is_exact() -> None:
    t = np.linspace(0.0, 1.0, 41, dtype=np.float64)
    target = 2.0 + 3.0 * t - 8.0 * np.maximum(t - 0.5, 0.0)
    fit = fit_continuous_piecewise(t, target, segments=2)
    assert fit.rmse < 1e-12
    assert fit.segment_row_counts == (20, 21)


def test_jump_requires_discontinuous_piecewise_representation() -> None:
    t = np.linspace(0.0, 1.0, 40, dtype=np.float64)
    target = np.where(t < 0.5, 1.0 + 2.0 * t, 10.0 - 3.0 * t)
    discontinuous = fit_discontinuous_piecewise(t, target, segments=2)
    continuous = fit_continuous_piecewise(t, target, segments=2)
    assert discontinuous.rmse < 1e-12
    assert continuous.rmse > 0.5


def test_h_representation_is_tvt_plus_z_in_diagnostic_run(tmp_path: Path) -> None:
    data_root = _synthetic_data_root(tmp_path, wells=1)
    result = run_slp_e0_diagnostic(
        data_root=data_root,
        output_root=tmp_path / "runs",
        run_id="h-definition",
        workers=1,
    )
    metrics = pd.read_csv(Path(result["run_root"]) / "oracle_metrics.csv")
    h_k1 = metrics.loc[
        (metrics["representation"] == "H")
        & (metrics["method"] == "piecewise_continuous_oracle")
        & (metrics["complexity"] == 1),
        "pooled_rmse",
    ].iat[0]
    tvt_k1 = metrics.loc[
        (metrics["representation"] == "TVT")
        & (metrics["method"] == "piecewise_continuous_oracle")
        & (metrics["complexity"] == 1),
        "pooled_rmse",
    ].iat[0]
    assert h_k1 < 1e-12
    assert tvt_k1 > 0.1


def test_nearest_prefix_search_handles_boundaries_ties_sorting_and_duplicates() -> None:
    prefix = np.array([2.0, 0.0, 4.0, 2.0])
    future = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    distance = nearest_prefix_distance(prefix, future)
    np.testing.assert_array_equal(distance, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


def test_recurrence_aggregate_uses_pooled_rows_not_mean_wells() -> None:
    per_well = pd.DataFrame(
        {
            "well_id": ["short", "long"],
            "rows": [1, 9],
            "inside_prefix_range_rows": [1, 0],
            "inside_prefix_range_fraction": [1.0, 0.0],
        }
    )
    summary = accumulate_recurrence_diagnostic(per_well)
    assert summary["inside_prefix_range_fraction"] == pytest.approx(0.1)
    assert summary["inside_prefix_range_fraction"] != pytest.approx(0.5)


def test_surface_identity_constant_has_zero_within_well_std() -> None:
    tvt = np.array([100.0, 101.0, 102.0])
    z = np.array([-50.0, -50.5, -51.0])
    surface = tvt + z - 12.5
    stats = surface_identity_stats(tvt, z, surface)
    assert stats.mean_constant == pytest.approx(12.5)
    assert stats.population_std == pytest.approx(0.0, abs=1e-15)


def test_all_missing_surface_pair_is_skipped_by_diagnostic_run(
    tmp_path: Path,
) -> None:
    data_root = _synthetic_data_root(tmp_path, wells=1)
    horizontal = next((data_root / "train").glob("*__horizontal_well.csv"))
    frame = pd.read_csv(horizontal)
    frame["ANCC"] = np.nan
    frame.to_csv(horizontal, index=False)
    result = run_slp_e0_diagnostic(
        data_root=data_root,
        output_root=tmp_path / "runs",
        run_id="missing-surface",
        workers=1,
    )
    surfaces = pd.read_csv(Path(result["run_root"]) / "surface_identity.csv")
    assert "ANCC" not in set(surfaces["formation"])
    assert set(surfaces["formation"]) == set(FORMATION_COLUMNS) - {"ANCC"}


def test_incomplete_prefix_fails_safely(tmp_path: Path) -> None:
    data_root = _synthetic_data_root(tmp_path, wells=1)
    horizontal = next((data_root / "train").glob("*__horizontal_well.csv"))
    frame = pd.read_csv(horizontal)
    frame.loc[2, "TVT_input"] = np.nan
    frame.loc[3, "TVT_input"] = frame.loc[3, "TVT"]
    frame.to_csv(horizontal, index=False)
    with pytest.raises(ValueError, match="contiguous known prefix"):
        run_slp_e0_diagnostic(
            data_root=data_root,
            output_root=tmp_path / "runs",
            run_id="invalid-prefix",
            workers=1,
        )


def test_non_monotonic_md_fails_safely(tmp_path: Path) -> None:
    data_root = _synthetic_data_root(tmp_path, wells=1)
    horizontal = next((data_root / "train").glob("*__horizontal_well.csv"))
    frame = pd.read_csv(horizontal)
    frame.loc[12, "MD"] = frame.loc[11, "MD"]
    frame.to_csv(horizontal, index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        run_slp_e0_diagnostic(
            data_root=data_root,
            output_root=tmp_path / "runs",
            run_id="invalid-md",
            workers=1,
        )


def test_unsupported_k_raises_instead_of_producing_nan() -> None:
    t = np.linspace(0.0, 1.0, 5)
    target = t**2
    with pytest.raises(UnsupportedOracleComplexity):
        fit_continuous_piecewise(t, target, segments=8)
    with pytest.raises(UnsupportedOracleComplexity):
        fit_discontinuous_piecewise(t, target, segments=4)


def test_pooled_metric_accumulates_sse_and_rows() -> None:
    pooled = accumulate_pooled_metric([(4.0, 1), (9.0, 3)])
    assert pooled.wells == 2
    assert pooled.rows == 4
    assert pooled.sse == 13.0
    assert pooled.pooled_rmse == pytest.approx(np.sqrt(13.0 / 4.0))


def test_output_inside_data_root_is_rejected(tmp_path: Path) -> None:
    data_root = _synthetic_data_root(tmp_path, wells=1)
    with pytest.raises(ValueError, match="cannot be inside competition data"):
        run_slp_e0_diagnostic(
            data_root=data_root,
            output_root=data_root / "diagnostics",
            run_id="unsafe-output",
            workers=1,
        )
    assert not (data_root / "diagnostics").exists()


def test_synthetic_cli_writes_all_artifacts_without_modifying_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = _synthetic_data_root(tmp_path, wells=2)
    before = _tree_hashes(data_root)
    exit_code = main(
        [
            "slp-e0",
            "--data-root",
            str(data_root),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "synthetic-cli",
            "--workers",
            "2",
        ]
    )
    captured = capsys.readouterr()
    completion = json.loads(captured.out)
    run_root = Path(completion["run_root"])
    assert exit_code == 0
    assert set(path.name for path in run_root.iterdir()) == EXPECTED_ARTIFACTS
    assert _tree_hashes(data_root) == before
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["uses_target"] is True
    assert summary["deployable"] is False
    assert summary["submission_eligible"] is False
    assert summary["scope"] == "full_train_diagnostic"
    assert summary["train_wells"] == 2
    assert captured.err
    with pytest.raises(FileExistsError):
        run_slp_e0_diagnostic(
            data_root=data_root,
            output_root=tmp_path / "runs",
            run_id="synthetic-cli",
            workers=1,
        )


def test_worker_counts_produce_deterministic_numeric_artifacts(tmp_path: Path) -> None:
    data_root = _synthetic_data_root(tmp_path, wells=2)
    run_one = run_slp_e0_diagnostic(
        data_root=data_root,
        output_root=tmp_path / "runs",
        run_id="workers-1",
        workers=1,
    )["run_root"]
    run_four = run_slp_e0_diagnostic(
        data_root=data_root,
        output_root=tmp_path / "runs",
        run_id="workers-4",
        workers=4,
    )["run_root"]
    for filename in (
        "oracle_metrics.csv",
        "per_well_oracles.csv.gz",
        "recurrence_summary.json",
        "recurrence_per_well.csv.gz",
        "surface_identity.csv",
    ):
        assert (Path(run_one) / filename).read_bytes() == (
            Path(run_four) / filename
        ).read_bytes()


def test_production_prediction_modules_do_not_import_diagnostics() -> None:
    source_root = Path("src/rogii")
    production_files = [
        source_root / "runner.py",
        source_root / "submission.py",
        *sorted((source_root / "models").glob("*.py")),
    ]
    for path in production_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(
            "diagnostics" in imported or "slp_e0" in imported
            for imported in imports
        ), f"Production module imports target-bearing diagnostics: {path}"
