from __future__ import annotations

import json
from pathlib import Path

import pytest

from rogii.diagnostics.slp_e0 import (
    SLP_E0_FORMATION_COLUMNS,
    run_slp_e0_diagnostic,
)


POLYNOMIAL_REFERENCE = {
    "0": 9.035410,
    "1": 6.697195,
    "2": 5.342276,
    "3": 4.424078,
    "4": 3.819453,
    "5": 3.281881,
    "6": 2.900173,
}
TVT_CONTINUOUS_REFERENCE = {
    "1": 6.6972,
    "2": 5.3244,
    "3": 4.3030,
    "4": 3.6941,
    "6": 2.7290,
}
TVT_DISCONTINUOUS_REFERENCE = {
    "1": 6.6972,
    "2": 4.6474,
    "3": 3.5287,
    "4": 2.8064,
    "6": 1.9249,
}
H_CONTINUOUS_REFERENCE = {
    "1": 5.9639,
    "2": 3.6891,
    "3": 2.7301,
    "4": 2.0942,
    "6": 1.4988,
}
H_DISCONTINUOUS_REFERENCE = {
    "1": 5.9639,
    "2": 3.1010,
    "3": 2.0931,
    "4": 1.5854,
    "6": 1.0808,
}


def _load_or_run_full_summary(data_root: Path) -> tuple[Path, dict]:
    requested = Path(".runs/slp-e0-full").resolve()
    fallback = Path(".runs/slp-e0-full-pytest").resolve()
    if (requested / "summary.json").is_file():
        run_root = requested
    elif (fallback / "summary.json").is_file():
        run_root = fallback
    else:
        result = run_slp_e0_diagnostic(
            data_root=data_root,
            output_root=Path(".runs"),
            run_id="slp-e0-full-pytest",
            workers=4,
        )
        run_root = Path(result["run_root"])
    return run_root, json.loads(
        (run_root / "summary.json").read_text(encoding="utf-8")
    )


def _assert_piecewise(
    actual: dict, reference: dict[str, float]
) -> None:
    for complexity, expected in reference.items():
        metric = actual[complexity]
        assert metric["wells"] == 773
        assert metric["rows"] == 3_783_989
        assert metric["pooled_rmse"] == pytest.approx(expected, abs=5e-4)


@pytest.mark.slow
def test_slp_e0_full_train_regression(data_root: Path) -> None:
    run_root, summary = _load_or_run_full_summary(data_root)
    assert summary["audit_id"] == "SLP-E0"
    assert summary["uses_target"] is True
    assert summary["deployable"] is False
    assert summary["submission_eligible"] is False
    assert summary["scope"] == "full_train_diagnostic"
    assert summary["train_wells"] == 773
    assert summary["prediction_rows"] == 3_783_989
    assert summary["flat_last_known_rmse"] == pytest.approx(
        15.909852870734552, abs=1e-12
    )

    for degree, expected in POLYNOMIAL_REFERENCE.items():
        metric = summary["polynomial_tvt"][degree]
        assert metric["wells"] == 773
        assert metric["rows"] == 3_783_989
        assert metric["pooled_rmse"] == pytest.approx(expected, abs=5e-4)

    _assert_piecewise(
        summary["piecewise_tvt"]["continuous"], TVT_CONTINUOUS_REFERENCE
    )
    _assert_piecewise(
        summary["piecewise_tvt"]["discontinuous"],
        TVT_DISCONTINUOUS_REFERENCE,
    )
    _assert_piecewise(
        summary["piecewise_h_tvt_plus_z"]["continuous"],
        H_CONTINUOUS_REFERENCE,
    )
    _assert_piecewise(
        summary["piecewise_h_tvt_plus_z"]["discontinuous"],
        H_DISCONTINUOUS_REFERENCE,
    )

    recurrence = summary["recurrence"]
    assert recurrence["inside_prefix_range_fraction"] == pytest.approx(
        0.6143, abs=6e-5
    )
    assert recurrence["nearest_distance_le_0_25_fraction"] == pytest.approx(
        0.6212, abs=6e-5
    )
    assert recurrence["nearest_distance_le_0_50_fraction"] == pytest.approx(
        0.6297, abs=6e-5
    )
    assert recurrence["nearest_distance_le_1_00_fraction"] == pytest.approx(
        0.6442, abs=6e-5
    )
    assert recurrence["nearest_distance_le_2_00_fraction"] == pytest.approx(
        0.6729, abs=6e-5
    )
    assert recurrence["nearest_distance_le_4_00_fraction"] == pytest.approx(
        0.7233, abs=6e-5
    )
    assert recurrence[
        "nearest_distance_le_1_00_and_gr_observed_fraction"
    ] == pytest.approx(0.4431, abs=6e-5)
    per_well = recurrence["per_well_fraction_quantiles"][
        "nearest_distance_le_1_00"
    ]
    assert per_well["q10"] == pytest.approx(0.0879, abs=6e-5)
    assert per_well["q50"] == pytest.approx(0.7445, abs=6e-5)
    assert per_well["q90"] == pytest.approx(1.0, abs=1e-12)

    assert set(summary["surface_identity"]) == set(SLP_E0_FORMATION_COLUMNS)
    for formation in SLP_E0_FORMATION_COLUMNS:
        metric = summary["surface_identity"][formation]
        assert 0 < metric["wells"] <= 773
        assert metric["pooled_within_well_std"] < 0.01
        assert metric["pooled_within_well_std"] == pytest.approx(
            0.0053, abs=5e-4
        )
        assert metric["diagnostic_type"] == "structural_identity_only"

    expected_artifacts = {
        "summary.json",
        "run_config.json",
        "oracle_metrics.csv",
        "per_well_oracles.csv.gz",
        "recurrence_summary.json",
        "recurrence_per_well.csv.gz",
        "surface_identity.csv",
    }
    assert expected_artifacts <= {
        path.name for path in run_root.iterdir() if path.is_file()
    }
