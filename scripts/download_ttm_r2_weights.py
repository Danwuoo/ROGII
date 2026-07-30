"""Download the single pinned Granite TTM R2 checkpoint and verify its hash."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii.external_models.ttm import (
    default_weight_dir,
    download_ttm_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download only the pinned Granite TTM R2 checkpoint files."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    runs_root = (project_root / ".runs").resolve()
    cache_dir = args.cache_dir.resolve()
    destination = (
        default_weight_dir(cache_dir)
        if args.weight_dir is None
        else args.weight_dir.resolve()
    )
    for label, path in (("cache", cache_dir), ("weight", destination)):
        if path == runs_root or runs_root not in path.parents:
            raise ValueError(
                f"TTM {label} directory must stay below project .runs: {path}"
            )
    identity = download_ttm_checkpoint(
        cache_dir,
        weight_dir=destination,
    )
    print(json.dumps(identity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
