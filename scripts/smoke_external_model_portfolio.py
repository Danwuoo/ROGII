from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.external_models.visible_smoke import run_visible_test_smoke


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected external-family checkpoints on visible test wells and "
            "write a JSON-only offline smoke audit."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        help=(
            "Repeat to select variants. The default is every non-control, "
            "non-random variant declared in run_config.json."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Repeat to select configured seeds; default uses all configured seeds.",
    )
    parser.add_argument(
        "--fold",
        action="append",
        type=int,
        help="Repeat to select folds; default requires all five fold checkpoints.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help=(
            "JSON path relative to --run-root (default visible_test_smoke.json). "
            "Paths outside the run root and non-JSON outputs are rejected."
        ),
    )
    args = parser.parse_args()
    result = run_visible_test_smoke(
        run_root=args.run_root,
        variants=args.variant,
        seeds=args.seed,
        folds=args.fold,
        output_json=args.output_json,
    )
    print(
        json.dumps(
            {
                "output_json": result["output_json"],
                "test_well_count": result["test_well_count"],
                "prediction_rows": result["prediction_rows"],
                "checkpoint_count": result["checkpoint_count"],
                "submission_csv_created": result["submission_csv_created"],
                "data_manifest_equal": result["data_manifest_equal"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
