"""CLI for the fixed sparse-anchor practical acquisition protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.diagnostics.anchor_acquisition_protocol import (
    audit_anchor_acquisition_protocol_artifacts,
    run_anchor_acquisition_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("quick", "full"), required=True
    )
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
        "--blend-oof",
        type=Path,
        default=Path(
            ".runs/final-p3-path-blend-full/oof_predictions.csv.gz"
        ),
    )
    parser.add_argument(
        "--prior-redundancy",
        type=Path,
        default=Path(
            ".runs/sparse-anchor-acquisition-redundancy-full"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "artifact directory; defaults to "
            ".runs/sparse-anchor-acquisition-protocol-{profile}"
        ),
    )
    parser.add_argument(
        "--audit-only",
        type=Path,
        help="validate an existing protocol artifact directory",
    )
    args = parser.parse_args()
    if args.audit_only is not None:
        print(
            json.dumps(
                audit_anchor_acquisition_protocol_artifacts(
                    args.audit_only
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    output_dir = args.output_dir or Path(
        f".runs/sparse-anchor-acquisition-protocol-{args.profile}"
    )
    summary = run_anchor_acquisition_protocol(
        data_root=args.data_root,
        fold_config=args.fold_config,
        blend_oof=args.blend_oof,
        prior_redundancy=args.prior_redundancy,
        output_dir=output_dir,
        profile=args.profile,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
