from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.diagnostics.external_anchor_research_experiment import (
    FULL_SEEDS,
    run_external_anchor_full_experiment,
    run_external_anchor_quick_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--fold-config", type=Path, default=Path("configs/folds/main_5fold.csv")
    )
    parser.add_argument("--output-root", type=Path, default=Path(".runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-root", type=Path, default=Path(".runs/cache"))
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    common = {
        "data_root": args.data_root,
        "fold_config": args.fold_config,
        "output_root": args.output_root,
        "run_id": args.run_id,
        "cache_root": args.cache_root,
        "workers": args.workers,
        "resume": args.resume,
    }
    if args.profile == "quick":
        if args.baseline_root is not None:
            common["baseline_root"] = args.baseline_root
        result = run_external_anchor_quick_experiment(seed=args.seed, **common)
    else:
        if args.seed != FULL_SEEDS[0]:
            parser.error(
                f"formal full profile uses locked seeds {FULL_SEEDS}; "
                f"--seed must remain {FULL_SEEDS[0]}"
            )
        if args.baseline_root is not None:
            common["baseline_root"] = args.baseline_root
        result = run_external_anchor_full_experiment(seeds=FULL_SEEDS, **common)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
