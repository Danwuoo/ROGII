"""CLI for the sparse post-PS TVT-anchor VOI diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.diagnostics.sparse_anchor_voi import (
    DEFAULT_K_VALUES,
    METHODS,
    POLICIES,
    run_sparse_anchor_voi,
)


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


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
        "--p3-oof",
        type=Path,
        default=Path(".runs/final-p3-full/oof_predictions.csv.gz"),
    )
    parser.add_argument(
        "--path-pca-oof",
        type=Path,
        default=Path(".runs/final-path-pca-full/oof_predictions.csv.gz"),
    )
    parser.add_argument(
        "--blend-oof",
        type=Path,
        default=Path(".runs/final-p3-path-blend-full/oof_predictions.csv.gz"),
    )
    parser.add_argument(
        "--path-pca-models",
        type=Path,
        default=Path(".runs/final-path-pca-full/models"),
    )
    parser.add_argument("--output-root", type=Path, default=Path(".runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument(
        "--policies",
        type=_csv_strings,
        default=POLICIES,
        help="comma-separated subset: uniform,disagreement",
    )
    parser.add_argument(
        "--methods",
        type=_csv_strings,
        default=METHODS,
        help="comma-separated subset: piecewise_residual,conditional_path_pca_mvp",
    )
    parser.add_argument(
        "--k-values",
        type=_csv_ints,
        default=DEFAULT_K_VALUES,
        help="comma-separated non-negative values including 0",
    )
    parser.add_argument("--max-validation-wells", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--no-oof", action="store_true")
    args = parser.parse_args()
    summary = run_sparse_anchor_voi(
        data_root=args.data_root,
        fold_config=args.fold_config,
        p3_oof=args.p3_oof,
        path_pca_oof=args.path_pca_oof,
        blend_oof=args.blend_oof,
        path_pca_models=args.path_pca_models,
        output_root=args.output_root,
        run_id=args.run_id,
        profile=args.profile,
        policies=args.policies,
        methods=args.methods,
        k_values=args.k_values,
        max_validation_wells=args.max_validation_wells,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        write_oof=not args.no_oof,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

