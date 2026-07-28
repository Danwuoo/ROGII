from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.external_models.stress import analyze_external_bundle_stress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--candidate", action="append")
    args = parser.parse_args()
    result = analyze_external_bundle_stress(
        args.run_root,
        bins=args.bins,
        candidate_ids=args.candidate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
