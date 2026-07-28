from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.external_models.weight_stability import (
    DEFAULT_OUTPUT_NAME,
    analyze_nested_blend_weight_stability,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed 10,000-draw well-cluster bootstrap over each outer "
            "fold's persisted inner OOF, without parsing outer/final OOF labels."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = parser.parse_args()
    summary = analyze_nested_blend_weight_stability(
        args.run_root,
        output_name=args.output_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
