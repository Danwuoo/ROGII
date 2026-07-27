"""CLI for the locked sparse-anchor outlier-QC diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.diagnostics.anchor_outlier_qc import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    audit_anchor_outlier_qc_artifacts,
    run_anchor_outlier_qc,
)
from rogii.diagnostics.anchor_noise_robustness import DEFAULT_NOISE_SEEDS


def _seed_list(value: str) -> tuple[int, ...]:
    try:
        return tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "realization seeds must be comma-separated integers"
        ) from error


def _scenario_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


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
    parser.add_argument(
        "--prior-realism-artifact",
        type=Path,
        default=Path(".runs/sparse-anchor-error-realism-full"),
    )
    parser.add_argument("--output-root", type=Path, default=Path(".runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--scenarios", type=_scenario_list)
    parser.add_argument(
        "--realization-seeds",
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
    parser.add_argument("--audit-only", type=Path)
    args = parser.parse_args()
    if args.audit_only is not None:
        print(
            json.dumps(
                audit_anchor_outlier_qc_artifacts(args.audit_only),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not args.run_id:
        parser.error("--run-id is required unless --audit-only is used")
    summary = run_anchor_outlier_qc(
        data_root=args.data_root,
        fold_config=args.fold_config,
        blend_oof=args.blend_oof,
        terminal_frontier=args.terminal_frontier,
        prior_realism_artifact=args.prior_realism_artifact,
        output_root=args.output_root,
        run_id=args.run_id,
        scenario_names=args.scenarios,
        realization_seeds=args.realization_seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        selection_seed=args.selection_seed,
        max_wells_per_fold=args.max_wells_per_fold,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
