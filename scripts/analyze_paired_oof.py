"""Strict paired analysis for two well-level OOF prediction artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["fold", "well_id", "row_index"]
VALUE_COLUMNS = ["y_true", "y_pred"]


def _rmse(sse: float, rows: int) -> float:
    if rows <= 0 or not math.isfinite(sse) or sse < 0:
        raise ValueError("RMSE requires finite non-negative SSE and positive rows")
    return math.sqrt(sse / rows)


def _tail(per_well: pd.DataFrame) -> dict[str, float]:
    rmse = per_well["candidate_rmse"].to_numpy(dtype=np.float64)
    sse = per_well["candidate_sse"].to_numpy(dtype=np.float64)
    count = max(1, math.ceil(0.10 * len(per_well)))
    return {
        "p50_rmse": float(np.quantile(rmse, 0.50)),
        "p90_rmse": float(np.quantile(rmse, 0.90)),
        "p95_rmse": float(np.quantile(rmse, 0.95)),
        "max_rmse": float(np.max(rmse)),
        "worst_10pct_sse_share": float(np.sort(sse)[-count:].sum() / sse.sum()),
        "wells_rmse_ge_40": int(np.sum(rmse >= 40.0)),
        "wells_rmse_ge_60": int(np.sum(rmse >= 60.0)),
    }


def _cluster_bootstrap(
    per_well: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    base = per_well["baseline_sse"].to_numpy(dtype=np.float64)
    candidate = per_well["candidate_sse"].to_numpy(dtype=np.float64)
    rows = per_well["n_rows"].to_numpy(dtype=np.int64)
    wells = len(per_well)
    deltas = np.empty(samples, dtype=np.float64)
    chunk = 500
    for begin in range(0, samples, chunk):
        end = min(samples, begin + chunk)
        index = rng.integers(0, wells, size=(end - begin, wells))
        sampled_rows = rows[index].sum(axis=1)
        base_rmse = np.sqrt(base[index].sum(axis=1) / sampled_rows)
        candidate_rmse = np.sqrt(candidate[index].sum(axis=1) / sampled_rows)
        deltas[begin:end] = candidate_rmse - base_rmse
    return {
        "samples": samples,
        "seed": seed,
        "delta_q025": float(np.quantile(deltas, 0.025)),
        "delta_q50": float(np.quantile(deltas, 0.50)),
        "delta_q975": float(np.quantile(deltas, 0.975)),
        "probability_candidate_better": float(np.mean(deltas < 0.0)),
    }


def analyze(
    baseline_path: Path,
    candidate_path: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(
        baseline_path,
        usecols=KEY_COLUMNS + VALUE_COLUMNS,
        dtype={"well_id": "string"},
    )
    candidate = pd.read_csv(
        candidate_path,
        usecols=KEY_COLUMNS + VALUE_COLUMNS,
        dtype={"well_id": "string"},
    )
    if len(baseline) != len(candidate) or len(baseline) == 0:
        raise ValueError("Paired OOF artifacts must have equal non-zero row counts")
    for column in KEY_COLUMNS:
        if not np.array_equal(
            baseline[column].to_numpy(), candidate[column].to_numpy()
        ):
            raise ValueError(f"Paired OOF key mismatch in {column}")
    if not np.allclose(
        baseline["y_true"].to_numpy(dtype=np.float64),
        candidate["y_true"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Paired OOF target mismatch")

    truth = baseline["y_true"].to_numpy(dtype=np.float64)
    baseline_prediction = baseline["y_pred"].to_numpy(dtype=np.float64)
    candidate_prediction = candidate["y_pred"].to_numpy(dtype=np.float64)
    if not all(
        np.isfinite(values).all()
        for values in (truth, baseline_prediction, candidate_prediction)
    ):
        raise ValueError("OOF values must be finite")
    baseline_square = (truth - baseline_prediction) ** 2
    candidate_square = (truth - candidate_prediction) ** 2
    frame = baseline.loc[:, KEY_COLUMNS].copy()
    frame["baseline_square"] = baseline_square
    frame["candidate_square"] = candidate_square

    grouped = frame.groupby("well_id", sort=True, observed=True)
    per_well = grouped.agg(
        fold=("fold", "first"),
        n_rows=("row_index", "size"),
        baseline_sse=("baseline_square", "sum"),
        candidate_sse=("candidate_square", "sum"),
    ).reset_index()
    per_well["baseline_rmse"] = np.sqrt(
        per_well["baseline_sse"] / per_well["n_rows"]
    )
    per_well["candidate_rmse"] = np.sqrt(
        per_well["candidate_sse"] / per_well["n_rows"]
    )
    per_well["rmse_delta"] = (
        per_well["candidate_rmse"] - per_well["baseline_rmse"]
    )
    per_well["sse_delta"] = (
        per_well["candidate_sse"] - per_well["baseline_sse"]
    )

    ordinal = grouped.cumcount().to_numpy(dtype=np.int64)
    group_size = grouped["row_index"].transform("size").to_numpy(dtype=np.int64)
    distance_decile = np.minimum(9, (10 * ordinal) // group_size)
    frame["distance_decile"] = distance_decile
    decile = (
        frame.groupby("distance_decile", sort=True)
        .agg(
            rows=("row_index", "size"),
            baseline_sse=("baseline_square", "sum"),
            candidate_sse=("candidate_square", "sum"),
        )
        .reset_index()
    )
    decile["baseline_rmse"] = np.sqrt(
        decile["baseline_sse"] / decile["rows"]
    )
    decile["candidate_rmse"] = np.sqrt(
        decile["candidate_sse"] / decile["rows"]
    )
    decile["rmse_delta"] = decile["candidate_rmse"] - decile["baseline_rmse"]

    rows = len(frame)
    baseline_sse = float(np.sum(baseline_square))
    candidate_sse = float(np.sum(candidate_square))
    baseline_rmse = _rmse(baseline_sse, rows)
    candidate_rmse = _rmse(candidate_sse, rows)
    worst = per_well.sort_values(
        ["candidate_sse", "well_id"], ascending=[False, True]
    ).head(20)
    summary: dict[str, Any] = {
        "baseline_path": str(Path(baseline_path).resolve()),
        "candidate_path": str(Path(candidate_path).resolve()),
        "rows": rows,
        "wells": int(per_well["well_id"].nunique()),
        "baseline_pooled_row_rmse": baseline_rmse,
        "candidate_pooled_row_rmse": candidate_rmse,
        "paired_rmse_delta": candidate_rmse - baseline_rmse,
        "candidate_better_wells": int(np.sum(per_well["rmse_delta"] < 0.0)),
        "candidate_worse_wells": int(np.sum(per_well["rmse_delta"] > 0.0)),
        "tail": _tail(per_well),
        "well_cluster_bootstrap": _cluster_bootstrap(
            per_well, samples=bootstrap_samples, seed=seed
        ),
        "worst_candidate_wells": worst[
            [
                "well_id",
                "fold",
                "n_rows",
                "baseline_rmse",
                "candidate_rmse",
                "rmse_delta",
                "candidate_sse",
            ]
        ].to_dict(orient="records"),
    }
    return summary, per_well, decile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    summary, per_well, decile = analyze(
        args.baseline,
        args.candidate,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    per_well.to_csv(output / "paired_per_well.csv", index=False)
    decile.to_csv(output / "distance_deciles.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

