"""CLI for locked sparse-anchor measurement-noise robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.diagnostics.anchor_noise_robustness import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_NOISE_SEEDS,
    DEFAULT_SIGMAS,
    audit_anchor_noise_artifacts,
    run_anchor_noise_robustness,
)


def _float_list(value: str) -> tuple[float, ...]:
    try:
        return tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "sigmas must be comma-separated numbers"
        ) from error


def _seed_list(value: str) -> tuple[int, ...]:
    try:
        return tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "noise seeds must be comma-separated integers"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/rogii-wellbore-geology-prediction"),
    )
    parser.add_argument(
        "--fold-config",
        type=Path,
        default=Path("configs/folds/main_5fold.csv"),
    )
    parser.add_argument(
        "--blend-oof",
        type=Path,
        default=Path(
            ".runs/final-p3-path-blend-full/oof_predictions.csv.gz"
        ),
    )
    parser.add_argument(
        "--terminal-frontier",
        type=Path,
        default=Path(".runs/sparse-anchor-terminal-frontier-full"),
    )
    parser.add_argument("--output-root", type=Path, default=Path(".runs"))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--sigmas",
        type=_float_list,
        default=DEFAULT_SIGMAS,
    )
    parser.add_argument(
        "--noise-seeds",
        type=_seed_list,
        default=DEFAULT_NOISE_SEEDS,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument("--selection-seed", type=int, default=20260711)
    parser.add_argument("--max-wells-per-fold", type=int)
    parser.add_argument(
        "--audit-only",
        type=Path,
        help="recompute metrics from an existing compact artifact directory",
    )
    args = parser.parse_args()
    if args.audit_only is not None:
        print(
            json.dumps(
                audit_anchor_noise_artifacts(args.audit_only),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not args.run_id:
        parser.error("--run-id is required unless --audit-only is used")
    summary = run_anchor_noise_robustness(
        data_root=args.data_root,
        fold_config=args.fold_config,
        blend_oof=args.blend_oof,
        terminal_frontier=args.terminal_frontier,
        output_root=args.output_root,
        run_id=args.run_id,
        sigmas=args.sigmas,
        noise_seeds=args.noise_seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        selection_seed=args.selection_seed,
        max_wells_per_fold=args.max_wells_per_fold,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
