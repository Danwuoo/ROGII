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
    else:
        result = submit_run(
            run_root=args.run,
            data_root=args.data_root,
            output=args.output,
            workers=args.workers,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0
