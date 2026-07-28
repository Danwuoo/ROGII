from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.portfolio_metrics import (
    common_slope_shape_metrics,
    leave_one_family_out,
    make_target_free_strata,
    paired_well_cluster_bootstrap,
    per_fold_metrics,
    per_well_metrics,
    per_well_win_summary,
    pooled_metrics,
    residual_correlation,
    shadow_private_subset_simulation,
    tail_metrics,
    target_free_stratified_metrics,
    validate_oof_frame,
)


def _oof() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for well_number in range(4):
        fold = well_number // 2
        for row_index, truth in enumerate((0.0, 1.0, 2.0)):
            signed = truth - 1.0
            rows.append(
                {
                    "well_id": f"w{well_number}",
                    "row_index": row_index,
                    "fold": fold,
                    "y_true": truth,
                    "y_pred_base": truth + 1.0,
                    "y_pred_exact": truth,
                    "y_pred_scaled": truth + 2.0 * signed,
                    "y_pred_opposite": truth - signed,
                }
            )
    return pd.DataFrame(rows)


def test_validate_oof_frame_checks_exact_keys_truth_finite_and_coverage() -> None:
    frame = _oof()
    expected = frame[["well_id", "row_index", "fold", "y_true"]].sample(
        frac=1.0, random_state=3
    )
    report = validate_oof_frame(
        frame,
        ["y_pred_base", "y_pred_exact"],
        expected_keys=expected,
        expected_well_ids=["w0", "w1", "w2", "w3"],
        expected_folds=[0, 1],
    )
    assert report["rows"] == 12
    assert report["wells"] == 4
    assert report["folds"] == [0, 1]
    assert report["coverage_checked"]
    assert report["truth_checked"]
    assert len(report["key_sha256"]) == 64

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate well/row"):
        validate_oof_frame(duplicate, ["y_pred_base"])

    nonfinite = frame.copy()
    nonfinite.loc[0, "y_pred_base"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_oof_frame(nonfinite, ["y_pred_base"])

    incomplete = frame.iloc[1:].copy()
    with pytest.raises(ValueError, match="coverage differs"):
        validate_oof_frame(incomplete, ["y_pred_base"], expected_keys=expected)

    wrong_truth = expected.copy()
    wrong_truth.loc[wrong_truth.index[0], "y_true"] += 0.25
    with pytest.raises(ValueError, match="truth differs"):
        validate_oof_frame(frame, ["y_pred_base"], expected_keys=wrong_truth)


def test_pooled_well_fold_and_tail_metrics_use_pooled_rows() -> None:
    frame = _oof()
    pooled = pooled_metrics(
        frame,
        ["y_pred_base", "y_pred_exact"],
        baseline_column="y_pred_base",
    ).set_index("candidate")
    assert pooled.loc["y_pred_base", "pooled_row_rmse"] == pytest.approx(1.0)
    assert pooled.loc["y_pred_exact", "pooled_row_rmse"] == pytest.approx(0.0)
    assert pooled.loc["y_pred_exact", "delta_rmse_vs_baseline"] == pytest.approx(-1.0)

    wells = per_well_metrics(
        frame,
        ["y_pred_base", "y_pred_exact"],
        baseline_column="y_pred_base",
    )
    assert len(wells) == 8
    assert set(wells["n_rows"]) == {3}
    assert set(wells.loc[wells["candidate"].eq("y_pred_base"), "rmse"]) == {1.0}

    folds = per_fold_metrics(
        frame,
        ["y_pred_base", "y_pred_exact"],
        baseline_column="y_pred_base",
    )
    assert len(folds) == 4
    assert set(folds["n_wells"]) == {2}
    assert set(folds.loc[folds["candidate"].eq("y_pred_exact"), "pooled_row_rmse"]) == {
        0.0
    }

    tails = tail_metrics(frame, ["y_pred_base", "y_pred_exact"]).set_index("candidate")
    assert tails.loc["y_pred_base", "p95_rmse"] == pytest.approx(1.0)
    assert tails.loc["y_pred_base", "worst_10pct_sse_share"] == pytest.approx(0.25)
    assert tails.loc["y_pred_exact", "worst_10pct_sse_share"] == 0.0


def test_common_slope_shape_is_an_orthogonal_row_decomposition() -> None:
    frame = _oof()
    normalized_position = np.tile(np.array([-1.0, 0.0, 1.0]), 4)
    frame["y_pred_linear_error"] = (
        frame["y_true"].to_numpy(dtype=float) + 2.0 + 3.0 * normalized_position
    )
    summary = common_slope_shape_metrics(frame, ["y_pred_linear_error"]).iloc[0]
    assert summary["total_rmse"] == pytest.approx(np.sqrt(10.0))
    assert summary["common_mode_rmse"] == pytest.approx(2.0)
    assert summary["common_mode_bias"] == pytest.approx(2.0)
    assert summary["centered_rmse"] == pytest.approx(np.sqrt(6.0))
    assert summary["slope_component_rmse"] == pytest.approx(np.sqrt(6.0))
    assert summary["zero_mean_shape_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert summary["terminal_rmse"] == pytest.approx(5.0)
    assert summary["relative_mse_reconstruction_error"] < 1e-12


def test_bootstrap_correlation_wins_and_shadow_simulation_are_deterministic() -> None:
    frame = _oof()
    frame.index = np.arange(100, 100 + len(frame))
    first = paired_well_cluster_bootstrap(
        frame,
        baseline_column="y_pred_base",
        candidate_columns=["y_pred_exact"],
        samples=200,
        seed=19,
        block_size=17,
    )
    second = paired_well_cluster_bootstrap(
        frame,
        baseline_column="y_pred_base",
        candidate_columns=["y_pred_exact"],
        samples=200,
        seed=19,
        block_size=31,
    )
    assert first == second
    interval = first["candidates"]["y_pred_exact"]
    assert interval["delta_rmse_q2.5"] == pytest.approx(-1.0)
    assert interval["delta_rmse_q97.5"] == pytest.approx(-1.0)

    correlation = residual_correlation(
        frame,
        ["y_pred_scaled", "y_pred_opposite"],
    )
    assert correlation.loc["y_pred_scaled", "y_pred_opposite"] == pytest.approx(-1.0)

    wins = per_well_win_summary(
        frame,
        baseline_column="y_pred_base",
        candidate_columns=["y_pred_exact"],
    ).iloc[0]
    assert wins["wins"] == 4
    assert wins["losses"] == 0
    assert wins["win_fraction"] == 1.0

    simulation_one = shadow_private_subset_simulation(
        frame,
        baseline_column="y_pred_base",
        candidate_columns=["y_pred_exact"],
        samples=1_000,
        subset_wells=2,
        seed=23,
        block_size=37,
    )
    simulation_two = shadow_private_subset_simulation(
        frame,
        baseline_column="y_pred_base",
        candidate_columns=["y_pred_exact"],
        samples=1_000,
        subset_wells=2,
        seed=23,
        block_size=101,
    )
    assert simulation_one == simulation_two
    candidate = simulation_one["candidates"]["y_pred_exact"]
    assert candidate["mean_delta"] == pytest.approx(-1.0)
    assert candidate["p_worse"] == 0.0


def test_lofo_and_target_free_strata() -> None:
    frame = _oof()
    lofo = leave_one_family_out(
        frame,
        family_columns={"exact": "y_pred_exact", "offset": "y_pred_base"},
        weights={"exact": 0.75, "offset": 0.25},
    ).set_index("family")
    assert lofo.loc["exact", "full_rmse"] == pytest.approx(0.25)
    assert lofo.loc["exact", "rmse_without_family"] == pytest.approx(1.0)
    assert lofo.loc["exact", "marginal_rmse_increase"] == pytest.approx(0.75)
    assert lofo.loc["offset", "rmse_without_family"] == pytest.approx(0.0)

    features = pd.DataFrame(
        {
            "well_id": ["w0", "w1", "w2", "w3"],
            "well_length": [100.0, 200.0, 300.0, 400.0],
            "typewell_group": ["a", "a", "b", "b"],
        }
    )
    strata = make_target_free_strata(
        features,
        ["well_length", "typewell_group"],
        bins=2,
        categorical_columns=["typewell_group"],
    )
    assert len(strata) == 8
    assert set(strata["stratum_feature"]) == {"well_length", "typewell_group"}
    metrics = target_free_stratified_metrics(
        frame,
        strata,
        ["y_pred_base", "y_pred_exact"],
        baseline_column="y_pred_base",
    )
    assert set(metrics["n_wells"]) == {2}
    exact = metrics.loc[metrics["candidate"].eq("y_pred_exact")]
    assert np.allclose(exact["delta_rmse_vs_baseline"], -1.0)

    with pytest.raises(ValueError, match="Target-derived"):
        make_target_free_strata(features.assign(y_true=0.0), ["y_true"])
