from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.diagnostics.external_anchor_research_experiment import (
    _anchor_basis,
    _full_bootstrap_bundle,
    _metric_tables,
    _paired_bootstrap,
    _robust_h_space_line,
    _tail_summary,
)


def _toy_oof() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["a", "b", "b", "b"],
            "row_index": [2, 5, 6, 7],
            "fold": [0, 1, 1, 1],
            "y_true": [0.0, 0.0, 0.0, 0.0],
            "base": [2.0, 1.0, 1.0, 1.0],
            "candidate": [1.0, 0.0, 0.0, 0.0],
        },
        index=[19, 3, 11, 7],
    )


def test_metric_tables_use_pooled_rows_and_ignore_dataframe_index() -> None:
    oof = _toy_oof()
    per_well, per_fold, overall = _metric_tables(
        oof,
        {"base": "base", "candidate": "candidate"},
        baseline="base",
    )
    assert overall["base"] == pytest.approx(np.sqrt(7.0 / 4.0))
    assert overall["candidate"] == pytest.approx(0.5)
    shuffled = oof.sample(frac=1.0, random_state=8)
    shuffled_well, shuffled_fold, shuffled_overall = _metric_tables(
        shuffled,
        {"base": "base", "candidate": "candidate"},
        baseline="base",
    )
    assert shuffled_overall == pytest.approx(overall)
    pd.testing.assert_frame_equal(
        per_well.sort_values(["candidate", "well_id"]).reset_index(drop=True),
        shuffled_well.sort_values(["candidate", "well_id"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        per_fold.sort_values(["candidate", "fold"]).reset_index(drop=True),
        shuffled_fold.sort_values(["candidate", "fold"]).reset_index(drop=True),
    )


def test_metric_tables_reject_one_well_in_multiple_folds() -> None:
    oof = _toy_oof()
    oof.loc[oof["well_id"].eq("b"), "fold"] = [1, 1, 2]
    with pytest.raises(ValueError, match="exactly one fold"):
        _metric_tables(
            oof,
            {"base": "base", "candidate": "candidate"},
            baseline="base",
        )


def test_tail_summary_is_finite_for_perfect_predictions() -> None:
    frame = pd.DataFrame(
        {
            "candidate": ["perfect", "perfect"],
            "well_id": ["a", "b"],
            "n_rows": [3, 5],
            "sse": [0.0, 0.0],
            "rmse": [0.0, 0.0],
        }
    )
    summary = _tail_summary(frame, "perfect")
    assert summary["worst_10pct_sse_share"] == 0.0
    assert all(np.isfinite(value) for value in summary.values())


def _bootstrap_frame() -> pd.DataFrame:
    rows = []
    for well_id, count, base_sse, candidate_sse in (
        ("a", 2, 8.0, 2.0),
        ("b", 5, 5.0, 20.0),
        ("c", 9, 36.0, 9.0),
    ):
        for candidate, sse in (("base", base_sse), ("candidate", candidate_sse)):
            rows.append(
                {
                    "well_id": well_id,
                    "candidate": candidate,
                    "n_rows": count,
                    "sse": sse,
                }
            )
    return pd.DataFrame(rows)


def test_paired_bootstrap_is_order_invariant_and_validates_counts() -> None:
    frame = _bootstrap_frame()
    first = _paired_bootstrap(
        frame, ["candidate"], baseline="base", samples=1_000, seed=77
    )
    second = _paired_bootstrap(
        frame.sample(frac=1.0, random_state=4),
        ["candidate"],
        baseline="base",
        samples=1_000,
        seed=77,
    )
    assert first == second
    mismatched = frame.copy()
    mismatched.loc[
        (mismatched["well_id"].eq("a")) & (mismatched["candidate"].eq("candidate")),
        "n_rows",
    ] = 3
    with pytest.raises(ValueError, match="row counts differ"):
        _paired_bootstrap(mismatched, ["candidate"], baseline="base", samples=10)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _paired_bootstrap(duplicated, ["candidate"], baseline="base", samples=10)


def test_full_bootstrap_separates_ensemble_gain_from_registration_gain() -> None:
    rows = []
    seeds = (20260711, 20260712, 20260713, 20260714, 20260715)
    candidates = {
        "anchor_student": 100.0,
        "anchor_seed_ensemble": 0.0,
        "heel_registered_ensemble": 0.16,
    }
    for seed in seeds:
        candidates[f"anchor_seed_{seed}"] = 1.0
        candidates[f"heel_registered_seed_{seed}"] = 4.0
    for candidate, sse in candidates.items():
        rows.append(
            {
                "well_id": "only",
                "candidate": candidate,
                "n_rows": 1,
                "sse": sse,
            }
        )
    bundle = _full_bootstrap_bundle(pd.DataFrame(rows), seeds)
    vs_incumbent = bundle["vs_historical_anchor_student"]["candidates"][
        "heel_registered_ensemble"
    ]
    matched = bundle["matched_registration_ensemble"]["candidates"][
        "heel_registered_ensemble"
    ]
    assert vs_incumbent["delta_rmse_q50"] < 0.0
    assert matched["delta_rmse_q50"] > 0.0
    for seed in seeds:
        seed_delta = bundle["matched_registration_per_seed"][str(seed)]["candidates"][
            f"heel_registered_seed_{seed}"
        ]
        assert seed_delta["delta_rmse_q2.5"] > 0.0


def test_anchor_basis_is_orthonormal_and_h_space_line_is_exact() -> None:
    basis = _anchor_basis(6)
    assert basis.shape == (8, 6)
    np.testing.assert_allclose(basis.T @ basis, np.eye(6), rtol=0.0, atol=1e-12)

    rows, known = 30, 12
    md = np.arange(rows, dtype=np.float64) * 10.0
    z = -8_000.0 + 0.25 * md
    h = 200.0 + 0.1 * md
    tvt = h - z
    tvt_input = tvt.copy()
    tvt_input[known:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 1_000.0 + md,
            "Y": 2_000.0 - md,
            "Z": z,
            "GR": 50.0 + np.sin(md),
            "TVT_input": tvt_input,
        }
    )
    prediction = _robust_h_space_line(horizontal, window_ft=250.0)
    np.testing.assert_allclose(prediction, tvt[known:], rtol=0.0, atol=1e-10)
