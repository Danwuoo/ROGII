"""Fixed well-level fold loading and deterministic development sampling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Collection, Iterable

import pandas as pd


FOLD_COUNT = 5


def load_fold_mapping(path: Path, *, expected_well_ids: Collection[str]) -> pd.DataFrame:
    mapping = pd.read_csv(path, dtype={"well_id": "string"})
    required = {"well_id", "fold", "fold_version"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Fold mapping missing columns: {sorted(missing)}")
    if mapping["well_id"].isna().any() or mapping["well_id"].duplicated().any():
        raise ValueError("Fold mapping must contain each well exactly once")
    if set(mapping["well_id"].astype(str)) != set(map(str, expected_well_ids)):
        raise ValueError("Fold mapping well coverage mismatch")
    if set(mapping["fold"]) != set(range(FOLD_COUNT)):
        raise ValueError("Fold mapping must contain exactly folds 0..4")
    if mapping["fold_version"].astype(str).nunique() != 1:
        raise ValueError("Fold mapping must contain exactly one fold version")
    return mapping.sort_values("well_id", kind="stable").reset_index(drop=True)


def partition_well_ids(
    mapping: pd.DataFrame, validation_fold: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if validation_fold not in range(FOLD_COUNT):
        raise ValueError("validation_fold must be in 0..4")
    validation = tuple(
        sorted(mapping.loc[mapping["fold"].eq(validation_fold), "well_id"].astype(str))
    )
    training = tuple(
        sorted(mapping.loc[~mapping["fold"].eq(validation_fold), "well_id"].astype(str))
    )
    if not validation or not training or set(training) & set(validation):
        raise ValueError(f"Invalid well partition for fold {validation_fold}")
    return training, validation


def stable_select(
    well_ids: Iterable[str], limit: int | None, *, seed: int, salt: str
) -> tuple[str, ...]:
    values = tuple(map(str, well_ids))
    if limit is None or limit >= len(values):
        return tuple(sorted(values))
    if limit < 1:
        raise ValueError("Well limits must be positive")

    def key(well_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{salt}:{well_id}".encode("utf-8")).hexdigest()
        return digest, well_id

    return tuple(sorted(sorted(values, key=key)[:limit]))
