"""Sample-ID parsing, validation, and prediction alignment without file output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import enumerate_wells, load_horizontal_inference, validate_prefix


ID_PATTERN = r"^(?P<well_id>[0-9a-f]{8})_(?P<row_index>0|[1-9][0-9]*)$"


def parse_sample_ids(ids: pd.Series) -> pd.DataFrame:
    """Parse ``{well_hash}_{zero_based_row_index}`` IDs without reordering them."""
    values = pd.Series(ids, dtype="string", copy=False).reset_index(drop=True)
    parsed = values.str.extract(ID_PATTERN)
    malformed = parsed.isna().any(axis=1)
    if malformed.any():
        examples = values[malformed].head(5).tolist()
        raise ValueError(f"Malformed sample id values: {examples}")
    parsed["row_index"] = pd.to_numeric(parsed["row_index"], errors="raise").astype("int64")
    parsed.insert(0, "id", values)
    return parsed


def validate_sample_submission_contract(sample: pd.DataFrame, test_dir: Path) -> pd.DataFrame:
    """Prove sample IDs exactly equal all test ``TVT_input.isna()`` rows."""
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError("Sample submission columns must be exactly ['id', 'tvt']")
    if sample["id"].duplicated().any():
        raise ValueError("Sample submission contains duplicate ids")
    parsed = parse_sample_ids(sample["id"])

    test_dir = Path(test_dir)
    expected: set[str] = set()
    for well in enumerate_wells(test_dir.parent, split=test_dir.name):
        horizontal = load_horizontal_inference(well.horizontal_path)
        validate_prefix(horizontal["TVT_input"])
        indices = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
        expected.update(f"{well.well_id}_{int(index)}" for index in indices)
    actual = set(parsed["id"].astype(str))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra or len(parsed) != len(expected):
        raise ValueError(
            "Sample IDs do not match test prediction rows: "
            f"missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})"
        )
    return parsed


def align_predictions_to_sample(sample: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return ``id,tvt`` predictions in sample order using an explicit ID join."""
    if "id" not in sample or "id" not in predictions or "tvt" not in predictions:
        raise ValueError("Sample and predictions must provide id; predictions must provide tvt")
    if sample["id"].duplicated().any():
        raise ValueError("Sample contains duplicate ids")
    if predictions["id"].duplicated().any():
        raise ValueError("Predictions contain duplicate ids")

    sample_ids = set(sample["id"].astype(str))
    prediction_ids = set(predictions["id"].astype(str))
    missing = sample_ids - prediction_ids
    extra = prediction_ids - sample_ids
    if missing or extra:
        raise ValueError(
            f"Prediction IDs mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    aligned = sample.loc[:, ["id"]].merge(
        predictions.loc[:, ["id", "tvt"]], on="id", how="left", validate="one_to_one", sort=False
    )
    values = pd.to_numeric(aligned["tvt"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions must all be finite numeric values")
    aligned["tvt"] = values
    return aligned
