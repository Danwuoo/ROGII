"""Print a concise view of an existing sparse-anchor VOI run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run_root / "summary.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(args.run_root / "information_curve.csv")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        curve[
            [
                "policy",
                "method",
                "k",
                "pooled_rmse",
                "non_anchor_rmse",
                "p95",
                "max",
                "wells_ge_40",
                "wells_ge_60",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

