"""Single command-line entrypoint for normal ROGII development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import MODEL_IDS
from .records import record_run
from .runner import check_data, run_experiment, submit_run


DEFAULT_FOLDS = Path("configs/folds/main_5fold.csv")


def _folds(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("folds must be comma-separated integers") from error


def _fractions(value: str) -> tuple[float, ...]:
    try:
        return tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "artificial cuts must be comma-separated numbers"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rogii")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="validate competition data and folds")
    check.add_argument("--data-root", type=Path, required=True)
    check.add_argument("--fold-config", type=Path, default=DEFAULT_FOLDS)

    run = commands.add_parser("run", help="run quick or full well-level CV")
    run.add_argument("--model", choices=MODEL_IDS, required=True)
    run.add_argument("--profile", choices=("quick", "full"), default="quick")
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--fold-config", type=Path, default=DEFAULT_FOLDS)
    run.add_argument("--output-root", type=Path, default=Path(".runs"))
    run.add_argument("--run-id")
    run.add_argument("--folds", type=_folds)
    run.add_argument("--max-train-wells", type=int)
    run.add_argument("--max-validation-wells", type=int)
    run.add_argument("--seed", type=int, default=20260711)
    run.add_argument("--workers", type=int, default=4)

    record = commands.add_parser("record", help="record only a small run summary")
    record.add_argument("--run", type=Path, required=True)
    record.add_argument("--results-root", type=Path, default=Path("results"))
    record.add_argument("--note", default="")
    record.add_argument("--replace", action="store_true")

    submit = commands.add_parser("submit", help="generate a local submission CSV")
    submit.add_argument("--run", type=Path, required=True)
    submit.add_argument("--data-root", type=Path, required=True)
    submit.add_argument("--output", type=Path, required=True)
    submit.add_argument("--workers", type=int, default=4)

    slp_e0 = commands.add_parser(
        "slp-e0",
        help="run the target-bearing, non-deployable SLP-E0 representation audit",
    )
    slp_e0.add_argument("--data-root", type=Path, required=True)
    slp_e0.add_argument("--output-root", type=Path, default=Path(".runs"))
    slp_e0.add_argument("--run-id", required=True)
    slp_e0.add_argument("--workers", type=int, default=4)

    slp_e1 = commands.add_parser(
        "slp-e1",
        help="run fold-safe, target-bearing SLP-E1 loop-edge diagnostics",
    )
    slp_e1.add_argument("--profile", choices=("quick", "full"), default="quick")
    slp_e1.add_argument("--data-root", type=Path, required=True)
    slp_e1.add_argument("--fold-config", type=Path, default=DEFAULT_FOLDS)
    slp_e1.add_argument("--output-root", type=Path, default=Path(".runs"))
    slp_e1.add_argument("--run-id", required=True)
    slp_e1.add_argument("--folds", type=_folds)
    slp_e1.add_argument("--max-train-wells", type=int)
    slp_e1.add_argument("--max-validation-wells", type=int)
    slp_e1.add_argument("--seed", type=int, default=20260725)
    slp_e1.add_argument("--workers", type=int, default=4)
    slp_e1.add_argument("--max-query-rows-per-well", type=int)
    slp_e1.add_argument("--top-k", type=int, default=3)
    slp_e1.add_argument(
        "--artificial-cuts", type=_fractions, default=(0.0, 0.25, 0.5)
    )
    slp_e1.add_argument(
        "--descriptor-set", choices=("handcrafted_v1",), default="handcrafted_v1"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check":
        result = check_data(data_root=args.data_root, fold_config=args.fold_config)
    elif args.command == "run":
        result = run_experiment(
            model_id=args.model,
            profile=args.profile,
            data_root=args.data_root,
            fold_config=args.fold_config,
            output_root=args.output_root,
            run_id=args.run_id,
            folds=args.folds,
            max_train_wells=args.max_train_wells,
            max_validation_wells=args.max_validation_wells,
            seed=args.seed,
            workers=args.workers,
        )
        result = {**result, "run_root": str(result["run_root"])}
    elif args.command == "record":
        result = record_run(
            run_root=args.run,
            results_root=args.results_root,
            note=args.note,
            replace=args.replace,
        )
    elif args.command == "submit":
        result = submit_run(
            run_root=args.run,
            data_root=args.data_root,
            output=args.output,
            workers=args.workers,
        )
    elif args.command == "slp-e0":
        # Keep the target-bearing diagnostic module out of production command
        # imports and prediction code paths.
        from .diagnostics.slp_e0 import run_slp_e0_diagnostic

        result = run_slp_e0_diagnostic(
            data_root=args.data_root,
            output_root=args.output_root,
            run_id=args.run_id,
            workers=args.workers,
        )
        result = {**result, "run_root": str(result["run_root"])}
    else:
        # SLP-E1 labels and artificial cuts remain isolated from production
        # models, runners, and submission code.
        from .diagnostics.slp_e1 import run_slp_e1_diagnostic

        result = run_slp_e1_diagnostic(
            profile=args.profile,
            data_root=args.data_root,
            fold_config=args.fold_config,
            output_root=args.output_root,
            run_id=args.run_id,
            folds=args.folds,
            max_train_wells=args.max_train_wells,
            max_validation_wells=args.max_validation_wells,
            seed=args.seed,
            workers=args.workers,
            max_query_rows_per_well=args.max_query_rows_per_well,
            top_k=args.top_k,
            artificial_cuts=args.artificial_cuts,
            descriptor_set=args.descriptor_set,
        )
        result = {**result, "run_root": str(result["run_root"])}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0
