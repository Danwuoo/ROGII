from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.diagnostics.external_model_portfolio_experiment import (
    DEFAULT_QUICK_VARIANTS,
    PortfolioVariant,
    run_external_model_portfolio_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--fold-config",
        type=Path,
        default=Path("configs/folds/main_5fold.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path(".runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-root", type=Path, default=Path(".runs/cache"))
    parser.add_argument(
        "--incumbent-root",
        type=Path,
        default=Path(".runs/heel-registered-anchor-full-20260728"),
    )
    parser.add_argument(
        "--incumbent-oof",
        type=Path,
        default=Path(
            ".runs/heel-registered-anchor-full-20260728/oof_predictions.csv.gz"
        ),
    )
    parser.add_argument(
        "--quick-source",
        type=Path,
        default=Path(".runs/sparse-anchor-student-quick/run_config.json"),
    )
    parser.add_argument(
        "--incumbent-cache-source",
        type=Path,
        help="Validated read-only fold/well cache from an earlier matched run.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(variant.variant_id for variant in DEFAULT_QUICK_VARIANTS),
        help="Repeat to run a subset; default runs the complete registered portfolio.",
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shrinkage", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--shadow-samples", type=int, default=100_000)
    parser.add_argument("--shadow-subset-wells", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    by_id: dict[str, PortfolioVariant] = {
        variant.variant_id: variant for variant in DEFAULT_QUICK_VARIANTS
    }
    variants = (
        None
        if args.variant is None
        else tuple(by_id[variant_id] for variant_id in args.variant)
    )
    result = run_external_model_portfolio_experiment(
        profile=args.profile,
        data_root=args.data_root,
        fold_config=args.fold_config,
        output_root=args.output_root,
        run_id=args.run_id,
        cache_root=args.cache_root,
        incumbent_root=args.incumbent_root,
        incumbent_oof=args.incumbent_oof,
        quick_source=args.quick_source,
        incumbent_cache_source=args.incumbent_cache_source,
        variants=variants,
        seeds=args.seed,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        device=args.device,
        workers=args.workers,
        shrinkage=args.shrinkage,
        bootstrap_samples=args.bootstrap_samples,
        shadow_samples=args.shadow_samples,
        shadow_subset_wells=args.shadow_subset_wells,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
