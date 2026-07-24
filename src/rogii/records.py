"""Keep only small, explicit summaries from local runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def record_run(
    *,
    run_root: Path,
    results_root: Path = Path("results"),
    note: str = "",
    replace: bool = False,
) -> dict[str, Any]:
    run = Path(run_root).resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    root = Path(results_root).resolve()
    data_root = Path(config["data_root"]).resolve()
    if root == data_root or data_root in root.parents:
        raise ValueError("Result records cannot be written inside competition data")
    record_dir = root / "recorded"
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f"{summary['run_id']}.json"
    if path.exists() and not replace:
        raise FileExistsError(f"Result record already exists: {path}")
    recorded_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "recorded_utc": recorded_utc,
        "note": note,
        "summary": summary,
        "run_config": config,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)

    index_path = root / "recorded_runs.csv"
    columns = [
        "run_id",
        "model_id",
        "scope",
        "pooled_row_rmse",
        "oof_wells",
        "oof_rows",
        "recorded_utc",
        "note",
    ]
    row = {
        "run_id": summary["run_id"],
        "model_id": summary["model_id"],
        "scope": summary["scope"],
        "pooled_row_rmse": summary["pooled_row_rmse"],
        "oof_wells": summary["oof_wells"],
        "oof_rows": summary["oof_rows"],
        "recorded_utc": recorded_utc,
        "note": note,
    }
    if index_path.exists():
        index = pd.read_csv(index_path)
        index = index.loc[index["run_id"].astype(str) != str(summary["run_id"])]
        index = pd.concat([index, pd.DataFrame([row])], ignore_index=True)
    else:
        index = pd.DataFrame([row], columns=columns)
    index = index.loc[:, columns].sort_values("run_id", kind="stable")
    index_tmp = index_path.with_suffix(".csv.tmp")
    index.to_csv(index_tmp, index=False, lineterminator="\n")
    os.replace(index_tmp, index_path)
    return {"record_path": str(path), "index_path": str(index_path)}
