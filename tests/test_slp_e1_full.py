from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.diagnostics.slp_e1 import run_slp_e1_diagnostic


def _load_or_run_full(data_root: Path) -> tuple[Path, dict]:
    requested = Path(".runs/slp-e1-full").resolve()
    fallback = Path(".runs/slp-e1-full-pytest").resolve()
    if (requested / "summary.json").is_file():
        run_root = requested
    elif (fallback / "summary.json").is_file():
        run_root = fallback
    else:
        result = run_slp_e1_diagnostic(
            profile="full",
            data_root=data_root,
            fold_config=Path("configs/folds/main_5fold.csv"),
            output_root=Path(".runs"),
            run_id="slp-e1-full-pytest",
            seed=20260725,
            workers=4,
        )
        run_root = Path(result["run_root"])
    return run_root, json.loads(
        (run_root / "summary.json").read_text(encoding="utf-8")
    )


@pytest.mark.slow
def test_slp_e1_full_oof_contract(data_root: Path) -> None:
    run_root, summary = _load_or_run_full(data_root)
    assert summary["audit_id"] == "SLP-E1"
    assert summary["scope"] == "full"
    assert summary["folds"] == [0, 1, 2, 3, 4]
    assert summary["oof_wells"] == 773
    assert summary["oof_future_rows"] == 3_783_989
    assert 0 < summary["descriptor_eligible_rows"] <= 3_783_989
    assert set(summary["edge_types"]) == {"PF", "FF", "HT"}
    assert {"shuffle_gr", "wrong_typewell"} <= set(summary["controls"])

    manifest = json.loads(
        (run_root / "fold_manifest.json").read_text(encoding="utf-8")
    )
    validation: list[str] = []
    for fold in range(5):
        item = manifest["folds"][str(fold)]
        outer = set(item["outer_training_wells"])
        confidence = set(item["confidence_fit_wells"])
        threshold = set(item["threshold_select_wells"])
        held_out = set(item["outer_validation_wells"])
        assert confidence.isdisjoint(threshold)
        assert confidence | threshold == outer
        assert outer.isdisjoint(held_out)
        validation.extend(held_out)
    assert len(validation) == len(set(validation)) == 773

    metrics = pd.read_csv(run_root / "edge_metrics.csv")
    numeric = metrics.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.fillna(0.0)).all().all()
    for edge_type in ("PF", "FF", "HT"):
        assert (
            (metrics["scope"] == "overall")
            & (metrics["method"] == "main")
            & (metrics["edge_type"] == edge_type)
        ).any()
    bins = pd.read_csv(run_root / "calibration_bins.csv")
    overall_main = bins.query("scope == 'overall' and method == 'main'")
    assert set(overall_main["bin"]) == set(range(10))
    assert all(
        (overall_main["edge_type"] == edge_type).sum() == 10
        for edge_type in ("PF", "FF", "HT")
    )
    assert (run_root / "edges_oof.csv.gz").stat().st_size > 0
    assert (run_root / "negative_controls.csv").stat().st_size > 0
