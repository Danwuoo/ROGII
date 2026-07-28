from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.external_models.artifacts import DEFAULT_BUNDLE_NAME
from rogii.external_models.portfolio_publisher import (
    publish_external_portfolio_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Translate a completed wide external-portfolio OOF run into the "
            "canonical, independently re-audited artifact bundle."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--row-coordinate",
        type=Path,
        help=(
            "Optional CSV/CSV.GZ with well_id,row_index[,fold],row_coordinate. "
            "Without it, coordinates are read from the wide OOF when available "
            "or rebuilt from inference-safe raw MD/TVT_input."
        ),
    )
    parser.add_argument(
        "--incumbent-oof",
        type=Path,
        help=(
            "Audited Heel full OOF used as the external truth/incumbent lock. "
            "Defaults to run_config.incumbent_oof, then the canonical 20260728 path."
        ),
    )
    parser.add_argument("--bundle-name", default=DEFAULT_BUNDLE_NAME)
    args = parser.parse_args()

    bundle = publish_external_portfolio_artifacts(
        args.run_root,
        row_coordinate_path=args.row_coordinate,
        incumbent_oof_path=args.incumbent_oof,
        bundle_name=args.bundle_name,
    )
    candidates = bundle.manifest["oof"]["candidates"]
    print(
        json.dumps(
            {
                "bundle_root": str(bundle.bundle_root),
                "passed": bundle.audit["passed"],
                "rows": bundle.audit["oof"]["rows"],
                "wells": bundle.audit["oof"]["wells"],
                "candidate_count": len(candidates),
                "candidates": [
                    {
                        "candidate_id": value["candidate_id"],
                        "control_id": value["control_id"],
                        "eligible_for_model_selection": value[
                            "eligible_for_model_selection"
                        ],
                        "source_model_count": len(value["source_model_ids"]),
                    }
                    for value in candidates
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
