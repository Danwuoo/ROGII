from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.inverse_solver.contracts import PrefixCalibrationConfig
from rogii.inverse_solver.controls import (
    deterministic_matched_wrong_typewell,
    phase_destroy_preserve_marginal,
    shuffle_gr_preserve_missingness,
    summarize_typewell,
)
from rogii.inverse_solver.stage0 import audit_stage0_frame, paired_mean_bootstrap


def test_phase_destroy_and_gr_shuffle_are_deterministic_and_preserve_marginals() -> None:
    values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 6.0])
    for transform in (phase_destroy_preserve_marginal, shuffle_gr_preserve_missingness):
        first = transform(values, seed=19)
        second = transform(values, seed=19)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.isfinite(first), np.isfinite(values))
        np.testing.assert_array_equal(
            np.sort(first[np.isfinite(first)]),
            np.sort(values[np.isfinite(values)]),
        )


def test_wrong_typewell_matching_is_nonself_and_deterministic() -> None:
    summaries = {}
    for index, identifier in enumerate(("00000001", "00000002", "00000003")):
        tvt = np.linspace(1000.0 + index, 1100.0 + index, 100)
        frame = pd.DataFrame({"TVT": tvt, "GR": np.sin(tvt / 5.0) + index})
        summaries[identifier] = summarize_typewell(identifier, frame)
    prefix = np.linspace(1010.0, 1070.0, 60)
    first = deterministic_matched_wrong_typewell(
        "00000001", summaries, prefix_tvt=prefix
    )
    second = deterministic_matched_wrong_typewell(
        "00000001", summaries, prefix_tvt=prefix
    )
    assert first == second
    assert first[0] != "00000001"


def _audit_frame(correct_delta: float) -> pd.DataFrame:
    wells = 100
    control = np.linspace(0.8, 1.2, wells)
    return pd.DataFrame(
        {
            "well_id": [f"{index:08x}" for index in range(wells)],
            "fold": np.arange(wells) % 5,
            "correct_cost": control + correct_delta,
            "wrong_cost": control,
            "phase_cost": control + 0.1,
            "shuffle_cost": control + 0.2,
            "correct_coherence": 0.5,
            "correct_effective_rows": 64,
            "correct_reliability": 0.8,
            "correct_nuisance_cost": control + 0.02,
            "wrong_nuisance_cost": control,
        }
    )


def test_observability_gate_is_locked_and_fail_closed() -> None:
    config = PrefixCalibrationConfig(bootstrap_replicates=200)
    failed = audit_stage0_frame(
        _audit_frame(0.05), config=config, expected_wells=100
    )
    assert failed["passed"] is False
    assert failed["decision"] == "stop_rogii_modeling_no_competitive_signal"
    incomplete = audit_stage0_frame(
        _audit_frame(-0.2).iloc[:-1], config=config, expected_wells=100
    )
    assert incomplete["checks"]["all_training_wells_audited"] is False
    assert incomplete["passed"] is False


def test_paired_bootstrap_is_deterministic() -> None:
    delta = np.linspace(-0.4, 0.1, 100)
    first = paired_mean_bootstrap(delta, replicates=500, seed=20260729)
    second = paired_mean_bootstrap(delta, replicates=500, seed=20260729)
    assert first == second
