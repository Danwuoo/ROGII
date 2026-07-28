from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.diagnostics.nested_external_ensemble import (
    NestedRunPolicy,
    estimate_nested_tcn_runtime,
    load_production_outer_tcn_source,
    run_nested_external_ensemble,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit incumbent/TCN blend weights and evaluate untouched outer "
            "OOF predictions. TCN family selection predates this runner, so "
            "outputs are conditional and not promotion-eligible nested evidence."
        )
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--fold-config",
        type=Path,
        default=Path("configs/folds/main_5fold.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path(".runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--cache-root", type=Path, default=Path(".runs/cache"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(".runs/external-model-tcn-full-20260728"),
        help=(
            "Completed full TCN run containing a published canonical "
            "external_model_artifacts bundle."
        ),
    )
    parser.add_argument(
        "--incumbent-oof",
        type=Path,
        default=Path(
            ".runs/heel-registered-anchor-full-20260728/oof_predictions.csv.gz"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print the predeclared sequential-CPU planning estimate and exit.",
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Independently validate the locked production source, data/folds, "
            "and all 25 outer TCN checkpoints without creating a run."
        ),
    )
    args = parser.parse_args()

    if args.estimate_only:
        print(json.dumps(estimate_nested_tcn_runtime(), indent=2, sort_keys=True))
        return 0
    if args.data_root is None:
        parser.error("--data-root is required unless --estimate-only")
    if args.preflight_only:
        source = load_production_outer_tcn_source(
            args.source_root,
            incumbent_oof=args.incumbent_oof,
            data_root=args.data_root,
            fold_config=args.fold_config,
        )
        print(
            json.dumps(
                {
                    "passed": True,
                    "source_model_fit_evidence_eligible": (source.evidence_eligible),
                    "promotion_evidence_eligible": False,
                    "family_selection_safe": False,
                    "rows": len(source.frame),
                    "wells": int(source.frame["well_id"].nunique()),
                    "source_root": source.source_root,
                    "oof_sha256": source.oof_sha256,
                    "manifest_sha256": source.manifest_sha256,
                    "runtime_estimate": estimate_nested_tcn_runtime(),
                    "run_created": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.run_id is None:
        parser.error("--run-id is required for a full conditional run")

    result = run_nested_external_ensemble(
        data_root=args.data_root,
        fold_config=args.fold_config,
        output_root=args.output_root,
        run_id=args.run_id,
        cache_root=args.cache_root,
        source_root=args.source_root,
        incumbent_oof=args.incumbent_oof,
        policy=NestedRunPolicy(),
        device=args.device,
        workers=args.workers,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
