from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.diagnostics.slp_e1 import _decision
from rogii.stratigraphic.confidence import (
    edge_truth,
    fit_calibrator,
    reliability_bins,
    scored_edge_metrics,
    select_operating_point,
)


def _feature_frame(rows: int = 100) -> pd.DataFrame:
    x = np.linspace(0.0, 2.0, rows)
    return pd.DataFrame(
        {
            "raw_shape_distance": x,
            "derivative_distance": x * 0.8,
            "rank_distance": x * 0.7,
            "barcode_distance": x * 0.9,
            "destroyed_barcode_distance": x[::-1],
            "point_gr_distance": 10.0 * x,
            "pair_distance": x * 0.85,
            "absolute_margin": 2.0 - x,
            "relative_margin": 1.0 - x / 2.0,
            "distance_ratio": x / 2.0,
            "reciprocal_rank": np.where(x < 1.0, 1, 4),
            "reciprocal_top_k": x < 1.0,
            "scale_vote_count": np.where(x < 1.0, 3, 1),
            "scale_agreement": np.where(x < 1.0, 1.0, 1 / 3),
            "shared_neighbor_count": np.where(x < 1.0, 2, 0),
            "local_cycle_support_count": np.where(x < 1.0, 4, 0),
            "query_observed_fraction": np.ones(rows),
            "candidate_observed_fraction": np.ones(rows),
            "query_low_information": np.zeros(rows, dtype=bool),
            "candidate_low_information": np.zeros(rows, dtype=bool),
        }
    )


def test_truth_label_boundaries_are_inclusive_and_catastrophic_is_strict() -> None:
    truth = edge_truth([0.5, 1.0, 2.0, 10.0, 10.000001])
    np.testing.assert_array_equal(
        truth["strict_positive"], [True, False, False, False, False]
    )
    np.testing.assert_array_equal(
        truth["primary_positive"], [True, True, False, False, False]
    )
    np.testing.assert_array_equal(
        truth["loose_positive"], [True, True, True, False, False]
    )
    np.testing.assert_array_equal(
        truth["catastrophic"], [False, False, False, False, True]
    )


def test_calibrator_is_unweighted_and_finite() -> None:
    features = _feature_frame()
    labels = np.arange(100) < 30
    model = fit_calibrator(features, labels, method="main", random_state=7)
    probability = model.predict(features)
    assert model.supported
    assert model.positive_prevalence == pytest.approx(0.30)
    assert np.isfinite(probability).all()
    assert (probability >= 0.0).all() and (probability <= 1.0).all()


def test_operating_point_uses_unique_query_coverage_and_support() -> None:
    confidence = [0.99, 0.98, 0.97, 0.96]
    positive = [True, True, True, False]
    catastrophic = [False, False, False, True]
    point = select_operating_point(
        confidence,
        positive,
        catastrophic,
        query_ids=[1, 1, 2, 3],
        well_ids=["a", "a", "b", "c"],
        target_precision=0.97,
        total_future_rows=10,
        minimum_edges=1,
        minimum_wells=1,
    )
    assert point["accepted_edges"] == 3
    assert point["unique_future_rows"] == 2
    assert point["all_future_coverage"] == pytest.approx(0.2)
    assert point["precision"] == 1.0


def test_metrics_use_accepted_edges_unique_rows_and_all_future_denominator() -> None:
    metric = scored_edge_metrics(
        confidence=[0.9, 0.8, 0.7, 0.1],
        error_ft=[0.1, 2.1, 11.0, 0.0],
        query_ids=[1, 1, 2, 3],
        threshold=0.7,
        total_future_rows=10,
        gr_observed_future_rows=8,
        descriptor_eligible_future_rows=5,
    )
    assert metric["accepted_edges"] == 3
    assert metric["edge_precision"] == pytest.approx(1 / 3)
    assert metric["catastrophic_false_edge_rate"] == pytest.approx(1 / 3)
    assert metric["unique_future_rows"] == 2
    assert metric["all_future_coverage"] == pytest.approx(0.2)
    assert metric["eligible_query_coverage"] == pytest.approx(0.4)


def test_reliability_ap_brier_and_ece_are_finite() -> None:
    bins, brier, ece = reliability_bins(
        [0.05, 0.25, 0.75, 0.95], [False, False, True, True]
    )
    assert len(bins) == 10
    assert np.isfinite(bins[["mean_confidence", "observed_precision"]]).all().all()
    assert np.isfinite(brier)
    assert np.isfinite(ece)


def test_empty_calibration_type_is_explicitly_unsupported() -> None:
    model = fit_calibrator(
        _feature_frame(5), [True] * 5, method="main", random_state=1
    )
    assert not model.supported
    assert "positive_and_negative" in model.unsupported_reason


def test_catastrophic_oof_edges_are_unsafe_even_with_insufficient_support() -> None:
    strict = {
        "accepted_edges": 10,
        "edge_precision": 0.1,
        "all_future_coverage": 0.001,
        "catastrophic_false_edge_rate": 0.5,
    }
    viable = dict(strict)
    assert _decision(strict, viable, thresholds_supported=False) == "unsafe"
