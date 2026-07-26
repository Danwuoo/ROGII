"""CLI for the terminal-aware sparse-anchor frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.diagnostics.terminal_anchor_frontier import run_terminal_frontier


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
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--piecewise-only", action="store_true")
    parser.add_argument(
        "--piecewise-source",
        type=Path,
        help="reuse a completed locked piecewise-only terminal frontier",
    )
    args = parser.parse_args()
    summary = run_terminal_frontier(
        data_root=args.data_root,
        fold_config=args.fold_config,
        p3_oof=args.p3_oof,
        path_pca_oof=args.path_pca_oof,
        blend_oof=args.blend_oof,
        path_pca_models=args.path_pca_models,
        output_root=args.output_root,
        run_id=args.run_id,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        piecewise_only=args.piecewise_only,
        piecewise_source=args.piecewise_source,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
