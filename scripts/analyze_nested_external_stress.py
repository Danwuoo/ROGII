from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.external_models.nested_stress import (
    DEFAULT_OUTPUT_NAME,
    analyze_nested_target_free_stress,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the full external run's frozen target-free strata to an "
            "independently re-audited evidence-eligible nested final OOF."
        )
    )
    parser.add_argument("--nested-run-root", type=Path, required=True)
    parser.add_argument("--external-run-root", type=Path, required=True)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = parser.parse_args()
    summary = analyze_nested_target_free_stress(
        args.nested_run_root,
        external_run_root=args.external_run_root,
        output_name=args.output_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
