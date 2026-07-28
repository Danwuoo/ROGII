from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.external_models.stress import (
    TARGET_FREE_FEATURE_COLUMNS,
    build_target_free_well_features,
)
from rogii.portfolio_metrics import make_target_free_strata


def _write_well(root: Path, well_id: str, offset: float) -> None:
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)
    rows = 12
    known = 4
    md = np.linspace(100.0, 210.0, rows)
    tvt = 1000.0 + offset + 0.8 * np.arange(rows)
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 10.0 + 0.5 * np.arange(rows),
            "Y": 20.0 + 0.2 * np.arange(rows),
            "Z": -md,
            "GR": 50.0 + offset + np.sin(np.arange(rows)),
            "TVT_input": np.r_[tvt[:known], np.full(rows - known, np.nan)],
            "TVT": tvt,
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(995.0, 1020.0, 20),
            "GR": 48.0 + offset + np.cos(np.arange(20)),
        }
    )
    horizontal.to_csv(train / f"{well_id}__horizontal_well.csv", index=False)
    typewell.to_csv(train / f"{well_id}__typewell.csv", index=False)


def test_target_free_features_use_inference_schema_and_make_strata(
    tmp_path: Path,
) -> None:
    _write_well(tmp_path, "well_a", 0.0)
    _write_well(tmp_path, "well_b", 2.0)
    diagnostics = pd.DataFrame(
        [
            {
                "well_id": well_id,
                "fold": fold,
                "seed": seed,
                "posterior_mean_shift": 0.1 * seed_index,
                "posterior_shift_std": 2.0 + seed_index,
                "posterior_entropy": 1.5,
                "maximum_posterior": 0.4,
                "edge_mass": 0.05,
                "posterior_mean_coverage": 0.9,
                "prefix_coverage": 0.8,
                "prefix_mse": 3.0,
            }
            for well_id, fold in (("well_a", 0), ("well_b", 1))
            for seed_index, seed in enumerate((11, 12))
        ]
    )
    path = tmp_path / "registration.csv.gz"
    diagnostics.to_csv(path, index=False)
    features = build_target_free_well_features(
        tmp_path,
        ("well_b", "well_a"),
        registration_diagnostics=path,
    )
    assert features["well_id"].tolist() == ["well_a", "well_b"]
    assert tuple(features.columns[1:]) == TARGET_FREE_FEATURE_COLUMNS
    assert np.isfinite(
        features.loc[:, list(TARGET_FREE_FEATURE_COLUMNS)].to_numpy()
    ).all()
    strata = make_target_free_strata(
        features,
        TARGET_FREE_FEATURE_COLUMNS,
        bins=2,
    )
    assert len(strata) == 2 * len(TARGET_FREE_FEATURE_COLUMNS)
    assert not {
        "y_true",
        "target",
        "prediction",
        "residual",
    }.intersection(features.columns)
