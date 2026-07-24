"""Competition file discovery and inference-schema contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


HORIZONTAL_INFERENCE_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TYPEWELL_INFERENCE_COLUMNS = ("TVT", "GR")
FORMATION_COLUMNS = frozenset({"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"})
FORBIDDEN_FEATURE_COLUMNS = frozenset({"TVT", "Geology", *FORMATION_COLUMNS})


@dataclass(frozen=True)
class WellFiles:
    """Paths for one dynamically discovered horizontal/typewell pair."""

    well_id: str
    horizontal_path: Path
    typewell_path: Path


def enumerate_wells(data_root: Path, split: Literal["train", "test"]) -> tuple[WellFiles, ...]:
    """Return sorted paired well files for a split, rejecting incomplete pairs."""
    split_dir = Path(data_root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    horizontal = {
        path.name.removesuffix("__horizontal_well.csv"): path
        for path in split_dir.glob("*__horizontal_well.csv")
    }
    typewell = {
        path.name.removesuffix("__typewell.csv"): path
        for path in split_dir.glob("*__typewell.csv")
    }
    if not horizontal:
        raise ValueError(f"No horizontal well CSV files found in {split_dir}")
    missing_typewell = sorted(horizontal.keys() - typewell.keys())
    missing_horizontal = sorted(typewell.keys() - horizontal.keys())
    if missing_typewell or missing_horizontal:
        raise ValueError(
            "Unpaired well files: "
            f"missing typewell={missing_typewell}, missing horizontal={missing_horizontal}"
        )
    return tuple(
        WellFiles(well_id, horizontal[well_id], typewell[well_id])
        for well_id in sorted(horizontal)
    )


def assert_feature_columns_allowed(
    columns: Iterable[str], *, kind: Literal["horizontal", "typewell"] = "horizontal"
) -> None:
    """Reject any requested feature column unavailable at inference time."""
    requested = set(columns)
    if kind == "horizontal":
        forbidden = requested & FORBIDDEN_FEATURE_COLUMNS
        allowed = set(HORIZONTAL_INFERENCE_COLUMNS)
    elif kind == "typewell":
        forbidden = requested & ({"Geology"} | FORMATION_COLUMNS)
        allowed = set(TYPEWELL_INFERENCE_COLUMNS)
    else:
        raise ValueError(f"Unknown feature-view kind: {kind!r}")
    unknown = requested - allowed
    if forbidden:
        raise ValueError(f"Requested forbidden {kind} feature columns: {sorted(forbidden)}")
    if unknown:
        raise ValueError(f"Requested non-inference {kind} feature columns: {sorted(unknown)}")


def select_inference_view(
    frame: pd.DataFrame, *, kind: Literal["horizontal", "typewell"]
) -> pd.DataFrame:
    """Return a defensive copy containing exactly the test-time schema."""
    if frame.columns.has_duplicates:
        raise ValueError(f"Duplicate columns are not allowed in {kind} data")
    required = (
        HORIZONTAL_INFERENCE_COLUMNS if kind == "horizontal" else TYPEWELL_INFERENCE_COLUMNS
    )
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required {kind} inference columns: {sorted(missing)}")
    assert_feature_columns_allowed(required, kind=kind)
    return frame.loc[:, list(required)].copy()


def _read_inference_columns(path: Path, *, kind: Literal["horizontal", "typewell"]) -> pd.DataFrame:
    """Read only inference-safe columns, even if target columns exist in the CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Well CSV does not exist: {path}")
    header = pd.read_csv(path, nrows=0)
    required = (
        HORIZONTAL_INFERENCE_COLUMNS if kind == "horizontal" else TYPEWELL_INFERENCE_COLUMNS
    )
    missing = set(required) - set(header.columns)
    if missing:
        raise ValueError(f"Missing required {kind} inference columns in {path}: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=list(required))
    return select_inference_view(frame, kind=kind)


def load_horizontal_inference(path: Path) -> pd.DataFrame:
    """Load a horizontal well without ever reading its target or formation columns."""
    return _read_inference_columns(path, kind="horizontal")


def load_typewell_inference(path: Path) -> pd.DataFrame:
    """Load a typewell without ever reading its train-only Geology column."""
    return _read_inference_columns(path, kind="typewell")


def validate_prefix(
    tvt_input: pd.Series, *, target: pd.Series | None = None
) -> tuple[int, int]:
    """Validate a non-empty known prefix followed by a non-empty prediction suffix.

    If ``target`` is supplied for training evaluation, every known prefix value
    must equal the corresponding target exactly.
    """
    values = pd.Series(tvt_input, copy=False).reset_index(drop=True)
    if values.empty:
        raise ValueError("TVT_input is empty")
    known = values.notna().to_numpy()
    known_count = int(known.sum())
    if known_count == 0:
        raise ValueError("TVT_input has no known prefix values")
    if known_count == len(values):
        raise ValueError("TVT_input has no prediction rows")
    expected = np.arange(len(values)) < known_count
    if not np.array_equal(known, expected):
        raise ValueError("TVT_input must be one contiguous known prefix followed by NaN")

    if target is not None:
        labels = pd.Series(target, copy=False).reset_index(drop=True)
        if len(labels) != len(values):
            raise ValueError("TVT and TVT_input lengths differ")
        if labels.iloc[:known_count].isna().any():
            raise ValueError("TVT contains NaN inside the known prefix")
        if not np.array_equal(
            values.iloc[:known_count].to_numpy(), labels.iloc[:known_count].to_numpy()
        ):
            raise ValueError("Known TVT_input values must exactly equal TVT")
    return known_count, len(values) - known_count


def validate_visible_test_copies(data_root: Path) -> tuple[str, ...]:
    """Verify authoring examples equal train's inference-schema subsets.

    This is a data-contract check only. It does not read train targets or compute
    any validation metric.
    """
    train = {well.well_id: well for well in enumerate_wells(data_root, "train")}
    test = enumerate_wells(data_root, "test")
    matched: list[str] = []
    for test_well in test:
        if test_well.well_id not in train:
            raise ValueError(f"Visible test well has no train authoring copy: {test_well.well_id}")
        train_well = train[test_well.well_id]
        pd.testing.assert_frame_equal(
            load_horizontal_inference(test_well.horizontal_path),
            load_horizontal_inference(train_well.horizontal_path),
            check_dtype=False,
            obj=f"horizontal authoring copy {test_well.well_id}",
        )
        pd.testing.assert_frame_equal(
            load_typewell_inference(test_well.typewell_path),
            load_typewell_inference(train_well.typewell_path),
            check_dtype=False,
            obj=f"typewell authoring copy {test_well.well_id}",
        )
        matched.append(test_well.well_id)
    return tuple(matched)
