from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.ensemble import (
    NestedBlendProvenance,
    apply_blend,
    average_family_predictions,
    combine_nested_fold_predictions,
    fit_constrained_blend,
    fit_family_blend,
    fit_nested_fold_blend,
    nested_weight_table,
    simple_blend_target,
)


FAMILIES = {
    "incumbent": ("y_pred_inc_seed1", "y_pred_inc_seed2"),
    "sequence": ("y_pred_sequence",),
}


def _prediction_rows(
    well_ids: list[str], *, fold_by_well: dict[str, int]
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for well_number, well_id in enumerate(well_ids):
        for row_index in range(3):
            truth = float(10 * well_number + row_index)
            rows.append(
                {
                    "well_id": well_id,
                    "row_index": row_index,
                    "fold": fold_by_well[well_id],
                    "y_true": truth,
                    "y_pred_inc_seed1": truth + 1.0,
                    "y_pred_inc_seed2": truth - 1.0,
                    "y_pred_sequence": truth + 2.0,
                }
            )
    return pd.DataFrame(rows)


def _nested_inputs() -> tuple[pd.DataFrame, pd.DataFrame, NestedBlendProvenance]:
    training = ["t0", "t1", "t2", "t3"]
    validation = ["v0", "v1"]
    inner = _prediction_rows(
        training,
        fold_by_well={"t0": 0, "t1": 0, "t2": 1, "t3": 1},
    )
    outer = _prediction_rows(
        validation,
        fold_by_well={"v0": 2, "v1": 2},
    )
    provenance = NestedBlendProvenance(
        outer_fold=2,
        outer_training_well_ids=tuple(training),
        outer_validation_well_ids=tuple(validation),
        source_artifacts=(
            "incumbent-inner-oof:sha256-a",
            "sequence-inner-oof:sha256-b",
        ),
    )
    return inner, outer, provenance


def test_family_predictions_are_averaged_before_weight_fit() -> None:
    inner, _, _ = _nested_inputs()
    averaged = average_family_predictions(inner, FAMILIES)
    assert list(averaged.columns) == ["incumbent", "sequence"]
    assert np.array_equal(averaged["incumbent"].to_numpy(), inner["y_true"].to_numpy())
    assert np.array_equal(
        averaged["sequence"].to_numpy(), inner["y_true"].to_numpy() + 2.0
    )

    fit = fit_family_blend(inner, FAMILIES)
    assert fit.weight_by_family["incumbent"] == pytest.approx(1.0, abs=1e-8)
    assert fit.weight_by_family["sequence"] == pytest.approx(0.0, abs=1e-8)
    assert fit.training_rmse == pytest.approx(0.0, abs=1e-8)

    # Adding another seed without changing the family mean must not give the
    # family another cross-family vote.
    expanded = inner.assign(y_pred_inc_seed3=inner["y_true"])
    expanded_families = {
        "incumbent": (
            "y_pred_inc_seed1",
            "y_pred_inc_seed2",
            "y_pred_inc_seed3",
        ),
        "sequence": ("y_pred_sequence",),
    }
    expanded_fit = fit_family_blend(expanded, expanded_families)
    assert expanded_fit.weight_by_family == pytest.approx(fit.weight_by_family)


def test_constrained_blend_supports_regularization_target_and_family_caps() -> None:
    inner, _, _ = _nested_inputs()
    averaged = average_family_predictions(inner, FAMILIES)
    truth = inner["y_true"].to_numpy(dtype=float)

    capped = fit_constrained_blend(
        truth,
        averaged,
        family_max_weight={"incumbent": 0.60},
    )
    assert sum(capped.weights) == pytest.approx(1.0)
    assert capped.weight_by_family["incumbent"] == pytest.approx(0.60, abs=1e-7)
    assert capped.weight_by_family["sequence"] == pytest.approx(0.40, abs=1e-7)
    prediction = apply_blend(averaged, capped)
    assert np.all(np.isfinite(prediction))
    assert np.sqrt(np.mean((prediction - truth) ** 2)) == pytest.approx(0.8)

    target = simple_blend_target(
        averaged.columns,
        incumbent_family="incumbent",
        incumbent_weight=0.75,
    )
    regularized = fit_constrained_blend(
        truth,
        averaged,
        regularization_strength=1_000_000.0,
        regularization_target=target,
    )
    assert regularized.weight_by_family["incumbent"] == pytest.approx(0.75, abs=1e-5)
    assert regularized.target_by_family == target

    with pytest.raises(ValueError, match="infeasible"):
        fit_constrained_blend(
            truth,
            averaged,
            family_max_weight={"incumbent": 0.40, "sequence": 0.40},
        )


def test_nested_fold_fit_uses_only_inner_oof_and_emits_auditable_output() -> None:
    inner, outer, provenance = _nested_inputs()
    result = fit_nested_fold_blend(
        inner,
        outer,
        families=FAMILIES,
        provenance=provenance,
        incumbent_family="incumbent",
        incumbent_target_weight=0.75,
        regularization_strength=0.0,
    )
    assert result.outer_fold == 2
    assert result.inner_fold_count == 2
    assert result.inner_wells == 4
    assert result.outer_wells == 2
    assert result.weights["incumbent"] == pytest.approx(1.0, abs=1e-8)
    assert result.inner_rmse == pytest.approx(0.0, abs=1e-8)
    assert result.outer_rmse == pytest.approx(0.0, abs=1e-8)
    assert result.provenance_digest == provenance.digest
    assert set(result.predictions.columns) == {
        "well_id",
        "row_index",
        "fold",
        "y_true",
        "y_pred_family__incumbent",
        "y_pred_family__sequence",
        "y_pred_nested_blend",
    }
    assert np.allclose(
        result.predictions["y_pred_nested_blend"],
        result.predictions["y_true"],
        atol=1e-12,
        rtol=0.0,
    )

    combined = combine_nested_fold_predictions(
        [result],
        expected_keys=outer[["well_id", "row_index", "fold", "y_true"]],
    )
    assert len(combined) == len(outer)
    weights = nested_weight_table([result])
    assert set(weights["family"]) == {"incumbent", "sequence"}
    assert set(weights["outer_fold"]) == {2}


def test_nested_fit_rejects_outer_validation_leak_and_incomplete_inner_oof() -> None:
    inner, outer, provenance = _nested_inputs()
    leaked = inner.copy()
    leaked.loc[leaked["well_id"].eq("t0"), "well_id"] = "v0"
    with pytest.raises(ValueError, match="Outer-validation IDs"):
        fit_nested_fold_blend(
            leaked,
            outer,
            families=FAMILIES,
            provenance=provenance,
        )

    incomplete = inner.loc[~inner["well_id"].eq("t0")].copy()
    with pytest.raises(ValueError, match="well coverage differs"):
        fit_nested_fold_blend(
            incomplete,
            outer,
            families=FAMILIES,
            provenance=provenance,
        )

    wrong_outer = outer.copy()
    wrong_outer["fold"] = 1
    with pytest.raises(ValueError, match="fold coverage differs"):
        fit_nested_fold_blend(
            inner,
            wrong_outer,
            families=FAMILIES,
            provenance=provenance,
        )

    with pytest.raises(TypeError, match="Provenance"):
        fit_nested_fold_blend(
            inner,
            outer,
            families=FAMILIES,
            provenance=None,  # type: ignore[arg-type]
        )


def test_provenance_itself_rejects_overlap_and_requires_source_ids() -> None:
    with pytest.raises(ValueError, match="overlap"):
        NestedBlendProvenance(
            outer_fold=0,
            outer_training_well_ids=("a", "b"),
            outer_validation_well_ids=("b", "c"),
            source_artifacts=("artifact",),
        )
    with pytest.raises(ValueError, match="source artifact"):
        NestedBlendProvenance(
            outer_fold=0,
            outer_training_well_ids=("a", "b"),
            outer_validation_well_ids=("c",),
            source_artifacts=(),
        )
