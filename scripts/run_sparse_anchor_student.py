from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.diagnostics.sparse_anchor_student_experiment import (
    run_sparse_anchor_student_experiment,
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--max-train-wells", type=int)
    parser.add_argument("--max-validation-wells", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_sparse_anchor_student_experiment(
        profile=args.profile,
        data_root=args.data_root,
        fold_config=args.fold_config,
        output_root=args.output_root,
        run_id=args.run_id,
        cache_root=args.cache_root,
        workers=args.workers,
        seed=args.seed,
        max_train_wells=args.max_train_wells,
        max_validation_wells=args.max_validation_wells,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
