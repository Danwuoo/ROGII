from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.external_models.artifacts import DEFAULT_AUDITED_INCUMBENT_OOF
from rogii.external_models.nested_reaudit import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_SUMMARY_ATOL,
    DEFAULT_SUMMARY_RTOL,
    DEFAULT_VALUE_ATOL,
    audit_nested_external_ensemble,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Independently re-read a completed nested TCN ensemble and "
            "recompute its keys, truth, predictions, fold weights, metrics, "
            "checkpoint provenance, data manifest, and no-submission checks."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--source-oof",
        type=Path,
        help=(
            "External disk OOF required only for evidence-ineligible smoke runs. "
            "Production always reloads the canonical source bundle."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Optional relocated canonical external-model source root for "
            "production; its recorded hashes must still match."
        ),
    )
    parser.add_argument(
        "--incumbent-oof",
        type=Path,
        default=DEFAULT_AUDITED_INCUMBENT_OOF,
        help=(
            "Independently re-audited Heel OOF used to lock production keys, "
            "truth, and incumbent predictions."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional relocated read-only competition data root.",
    )
    parser.add_argument(
        "--fold-config",
        type=Path,
        help="Optional relocated fold mapping with the recorded SHA-256.",
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--value-atol", type=float, default=DEFAULT_VALUE_ATOL)
    parser.add_argument(
        "--summary-atol",
        type=float,
        default=DEFAULT_SUMMARY_ATOL,
    )
    parser.add_argument(
        "--summary-rtol",
        type=float,
        default=DEFAULT_SUMMARY_RTOL,
    )
    args = parser.parse_args()

    report = audit_nested_external_ensemble(
        args.run_root,
        source_oof=args.source_oof,
        source_root=args.source_root,
        incumbent_oof=args.incumbent_oof,
        data_root=args.data_root,
        fold_config=args.fold_config,
        output_name=args.output_name,
        value_atol=args.value_atol,
        summary_atol=args.summary_atol,
        summary_rtol=args.summary_rtol,
    )
    pooled = {
        row["candidate"]: row["pooled_row_rmse"] for row in report["metrics"]["pooled"]
    }
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "run_id": report["run_id"],
                "evidence_mode": report["evidence_mode"],
                "evidence_eligible": report["evidence_eligible"],
                "rows": report["scope"]["rows"],
                "wells": report["scope"]["wells"],
                "nested_pooled_row_rmse": pooled["y_pred_nested_tcn"],
                "weights_recomputed": report["checks"][
                    "weights_recomputed_from_inner_oof"
                ],
                "submission_created": report["submission_created"],
                "output": str(Path(args.run_root).resolve() / args.output_name),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
