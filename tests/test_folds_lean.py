from __future__ import annotations

import pandas as pd

from rogii.folds import partition_well_ids, stable_select


def test_well_partition_is_disjoint() -> None:
    mapping = pd.DataFrame(
        {
            "well_id": [f"w{index}" for index in range(10)],
            "fold": [index % 5 for index in range(10)],
        }
    )
    training, validation = partition_well_ids(mapping, 0)
    assert set(training).isdisjoint(validation)
    assert set(training) | set(validation) == set(mapping["well_id"])


def test_quick_selection_is_stable_and_bounded() -> None:
    wells = [f"w{index}" for index in range(100)]
    first = stable_select(wells, 20, seed=7, salt="quick")
    second = stable_select(reversed(wells), 20, seed=7, salt="quick")
    assert first == second
    assert len(first) == 20
