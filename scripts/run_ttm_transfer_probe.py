"""CLI for synthetic smoke or strict quick/full TTM transfer experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.diagnostics.ttm_transfer_probe import run_ttm_transfer_experiment
from rogii.external_models.ttm import default_weight_dir


DEFAULT_DATA = Path("data/rogii-wellbore-geology-prediction")
DEFAULT_FOLDS = Path("configs/folds/main_5fold.csv")
DEFAULT_CACHE = Path(".runs/ttm-r2-transfer-probe-20260729/cache")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the matched Granite TTM R2 transfer probe.  --profile is "
            "required; full is never selected by default."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "quick", "full"),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--weight-dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fold-config", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--locked-config",
        type=Path,
        default=Path("configs/external/ttm_r2_transfer_probe_20260729.json"),
    )
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
        "--variant",
        action="append",
        dest="variants",
        help="Repeat to override the profile's locked variant list.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Repeat for every locked head seed.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-adapter", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.profile == "full":
        quick_gate_path = args.run_dir.resolve().parent / "quick" / "quick_gate.json"
        try:
            quick_gate = json.loads(quick_gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                "full is fail-closed: a passed sibling quick/quick_gate.json "
                f"is required ({quick_gate_path})"
            ) from exc
        if quick_gate.get("passed") is not True:
            raise SystemExit(
                "full is prohibited because the locked quick transfer gate did "
                f"not pass ({quick_gate_path})"
            )
    weight_dir = (
        default_weight_dir(args.cache_dir)
        if args.weight_dir is None
        else args.weight_dir
    )
    summary = run_ttm_transfer_experiment(
        profile=args.profile,
        run_dir=args.run_dir,
        cache_dir=args.cache_dir,
        weight_dir=weight_dir,
        data_root=args.data_root,
        fold_config=args.fold_config,
        locked_config_path=args.locked_config,
        incumbent_root=args.incumbent_root,
        incumbent_oof=args.incumbent_oof,
        quick_source=args.quick_source,
        variants=args.variants,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
        resume=args.resume,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        offline=args.offline,
        bootstrap_samples=args.bootstrap_samples,
        allow_adapter=args.allow_adapter,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
