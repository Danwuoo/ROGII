"""Create a fixed-weight blend from two already strict OOF artifacts.

This utility never estimates its weight from targets.  It verifies exact row-key
and target identity, then averages predictions with a caller-supplied constant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["well_id", "row_index", "fold"]
VALUE_COLUMNS = ["y_true", "y_pred"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    weight = float(args.first_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("first-weight must be in [0, 1]")
    first = pd.read_csv(
        args.first,
        usecols=KEY_COLUMNS + VALUE_COLUMNS,
        dtype={"well_id": "string"},
    )
    second = pd.read_csv(
        args.second,
        usecols=KEY_COLUMNS + VALUE_COLUMNS,
        dtype={"well_id": "string"},
    )
    if len(first) == 0 or len(first) != len(second):
        raise ValueError("OOF artifacts must have equal non-zero row counts")
    for column in KEY_COLUMNS:
        if not np.array_equal(first[column].to_numpy(), second[column].to_numpy()):
            raise ValueError(f"OOF key mismatch in {column}")
    if not np.array_equal(
        first["y_true"].to_numpy(dtype=np.float64),
        second["y_true"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("OOF target mismatch")

    first_prediction = first["y_pred"].to_numpy(dtype=np.float64)
    second_prediction = second["y_pred"].to_numpy(dtype=np.float64)
    if not np.isfinite(first_prediction).all() or not np.isfinite(
        second_prediction
    ).all():
        raise ValueError("OOF predictions must be finite")
    output = first.loc[:, KEY_COLUMNS + ["y_true"]].copy()
    output["y_pred"] = (
        weight * first_prediction + (1.0 - weight) * second_prediction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    truth = output["y_true"].to_numpy(dtype=np.float64)
    rmse = float(
        np.sqrt(np.mean((truth - output["y_pred"].to_numpy(dtype=np.float64)) ** 2))
    )
    metadata = {
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "first_weight": weight,
        "second_weight": 1.0 - weight,
        "rows": len(output),
        "pooled_row_rmse": rmse,
        "weight_selection": "caller_supplied_constant; no target fitting",
    }
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
