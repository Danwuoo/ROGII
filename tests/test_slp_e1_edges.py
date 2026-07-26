from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.stratigraphic.loop_edges import (
    EdgeGenerationConfig,
    generate_loop_edges,
)


def _repeated_motif_views() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    md = np.arange(520, dtype=np.float64)
    latent = 100.0 + np.mod(md, 140.0) * 0.25
    gr = (
        np.sin(latent / 2.7)
        + 0.35 * np.sin(latent / 0.91)
        + 0.02 * np.mod(md, 140.0)
    )
    known = 190
    tvt_input = latent.copy()
    tvt_input[known:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 1000.0 + md,
            "Y": 2000.0 + md,
            "Z": -500.0 - md,
            "GR": gr,
            "TVT_input": tvt_input,
        }
    )
    type_coordinate = np.arange(95.0, 145.0, 0.25)
    type_gr = (
        np.sin(type_coordinate / 2.7)
        + 0.35 * np.sin(type_coordinate / 0.91)
    )
    typewell = pd.DataFrame({"TVT": type_coordinate, "GR": type_gr})
    return horizontal, typewell, latent


def _result():
    horizontal, typewell, latent = _repeated_motif_views()
    result = generate_loop_edges(
        horizontal,
        typewell,
        well_id="synthetic",
        config=EdgeGenerationConfig(
            horizontal_half_spans=(8.0, 16.0, 32.0),
            typewell_half_spans=(2.0, 4.0, 8.0),
            top_k=3,
            minimum_ff_separation=100.0,
        ),
    )
    return result, latent


def test_pf_recovers_known_prefix_recurrence() -> None:
    result, latent = _result()
    pf = result.edges.query("method == 'main' and edge_type == 'PF'")
    error = np.abs(
        latent[pf["query_row"].to_numpy()]
        - latent[pf["candidate_row"].to_numpy()]
    )
    assert len(pf) > 0
    assert np.mean(error <= 1.0) > 0.25


def test_ff_is_nonlocal_unique_and_recovers_recurrence() -> None:
    result, latent = _result()
    ff = result.edges.query("method == 'main' and edge_type == 'FF'")
    assert len(ff) > 0
    assert (
        np.abs(ff["query_md"] - ff["candidate_md"]) >= 100.0
    ).all()
    assert not ff[["edge_row_a", "edge_row_b"]].duplicated().any()
    error = np.abs(
        latent[ff["query_row"].to_numpy()]
        - latent[ff["candidate_row"].to_numpy()]
    )
    assert np.mean(error <= 1.0) > 0.25


def test_ht_finds_correct_typewell_coordinate() -> None:
    result, latent = _result()
    ht = result.edges.query("method == 'main' and edge_type == 'HT'")
    error = np.abs(
        latent[ht["query_row"].to_numpy()]
        - ht["candidate_tvt"].to_numpy()
    )
    assert len(ht) > 0
    assert np.mean(error <= 1.0) > 0.10


def test_windows_do_not_cross_prefix_future_boundary() -> None:
    result, _ = _result()
    pf = result.edges.query("method == 'main' and edge_type == 'PF'")
    assert (pf["query_row"] >= 190 + 8).all()
    assert (pf["candidate_row"] <= 190 - 1 - 8).all()


def test_top_k_margin_reciprocal_scale_and_cycle_features_exist() -> None:
    result, _ = _result()
    main = result.edges.query("method == 'main'")
    assert main.groupby(["edge_type", "query_row"]).size().max() <= 3
    assert np.isfinite(main["best_distance"]).all()
    assert np.isfinite(main["second_best_distance"]).all()
    assert (main["absolute_margin"] >= 0.0).all()
    assert (main["reciprocal_rank"] >= 1).all()
    assert main["scale_vote_count"].min() >= 1
    assert (main["local_cycle_support_count"] >= 0).all()


def test_cycle_support_has_no_label_input_and_consistency_changes_score() -> None:
    result, _ = _result()
    main = result.edges.query("method == 'main'")
    supported = main.loc[main["local_cycle_support_count"] > 0]
    unsupported = main.loc[main["local_cycle_support_count"] == 0]
    assert len(supported) > 0
    assert len(unsupported) > 0
    adjusted_supported = (
        supported["raw_score"] + supported["pair_distance"]
    ).median()
    adjusted_unsupported = (
        unsupported["raw_score"] + unsupported["pair_distance"]
    ).median()
    assert adjusted_supported > adjusted_unsupported


def test_negative_retrieval_controls_are_emitted() -> None:
    result, _ = _result()
    assert {"main", "raw_nn", "point_gr"} <= set(result.edges["method"])
    for edge_type in ("PF", "FF", "HT"):
        assert (
            result.edges.query(
                "method == 'raw_nn' and edge_type == @edge_type"
            ).groupby("query_row").size().max()
            <= 1
        )
