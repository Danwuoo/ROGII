from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.external_models.artifacts import (
    DEFAULT_AUDITED_INCUMBENT_OOF,
    DEFAULT_BUNDLE_NAME,
)
from rogii.external_models.portfolio_reaudit import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_RUNNER_SUMMARY_NAME,
    DEFAULT_SHADOW_SAMPLES,
    SUMMARY_ABSOLUTE_TOLERANCE,
    SUMMARY_RELATIVE_TOLERANCE,
    audit_external_model_portfolio,
)
from rogii.portfolio_metrics import DEFAULT_BOOTSTRAP_SEED


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Independently re-read and audit one completed canonical external-model "
            "portfolio. This command does not read runner-owned prediction frames."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--incumbent-oof",
        type=Path,
        default=DEFAULT_AUDITED_INCUMBENT_OOF,
        help=(
            "Independently re-audited Heel full OOF used to lock exact keys, "
            "truth, and incumbent predictions."
        ),
    )
    parser.add_argument("--bundle-name", default=DEFAULT_BUNDLE_NAME)
    parser.add_argument("--runner-summary-name", default=DEFAULT_RUNNER_SUMMARY_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--shadow-samples", type=int, default=DEFAULT_SHADOW_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--summary-absolute-tolerance",
        type=float,
        default=SUMMARY_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--summary-relative-tolerance",
        type=float,
        default=SUMMARY_RELATIVE_TOLERANCE,
    )
    args = parser.parse_args()

    report = audit_external_model_portfolio(
        args.run_root,
        incumbent_oof=args.incumbent_oof,
        bundle_name=args.bundle_name,
        runner_summary_name=args.runner_summary_name,
        output_name=args.output_name,
        bootstrap_samples=args.bootstrap_samples,
        shadow_samples=args.shadow_samples,
        seed=args.seed,
        summary_absolute_tolerance=args.summary_absolute_tolerance,
        summary_relative_tolerance=args.summary_relative_tolerance,
    )
    pooled = {
        value["candidate"]: value["pooled_row_rmse"]
        for value in report["metrics"]["pooled"]
    }
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "run_id": report["run_id"],
                "profile": report["profile"],
                "rows": report["scope"]["rows"],
                "wells": report["scope"]["wells"],
                "candidates": report["scope"]["candidate_count"],
                "incumbent_pooled_row_rmse": pooled["y_pred_incumbent"],
                "summary_comparison_passed": report["runner_summary_comparison"][
                    "passed"
                ],
                "shadow_private_performed": report["metrics"]["shadow_private"][
                    "performed"
                ],
                "output": str(Path(args.run_root).resolve() / args.output_name),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
