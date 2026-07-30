"""Dynamic visible-test smoke for the inference-safe alignment family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.data import (
    enumerate_wells,
    load_horizontal_inference,
    load_typewell_inference,
)
from rogii.external_models.artifacts import capture_data_manifest
from rogii.external_models.incumbent import HeelIncumbentProvider
from rogii.physics_alignment import (
    ALIGNMENT_STATUS_IDENTITY,
    CONTROL_NONE,
    METHOD_BEAM,
    METHOD_NCC,
    METHOD_PARTICLE_FILTER,
    SafePhysicsAlignmentConfig,
    SafePhysicsAlignmentFamily,
)


SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=np.float64).tobytes(order="C")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_smoke(
    *,
    data_root: Path,
    fold_config: Path,
    incumbent_root: Path,
    cache_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    before = capture_data_manifest(data_root)
    tests = enumerate_wells(data_root, "test")
    sample = pd.read_csv(data_root / "sample_submission.csv", dtype={"id": "string"})
    if sample["id"].duplicated().any():
        raise ValueError("sample submission contains duplicate IDs")
    sample_ids = sample["id"].astype(str).to_numpy()
    provider = HeelIncumbentProvider(
        data_root=data_root,
        fold_config=fold_config,
        artifact_root=incumbent_root,
        cache_root=cache_root,
    )
    family = SafePhysicsAlignmentFamily(
        SafePhysicsAlignmentConfig(maximum_steps=128, rows_per_step=36, particles=192)
    )
    records: list[dict[str, Any]] = []
    observed_ids: list[str] = []
    for well in tests:
        horizontal = load_horizontal_inference(well.horizontal_path)
        typewell = load_typewell_inference(well.typewell_path)
        incumbent = provider.predict_visible_test(well.well_id)
        observed_ids.extend(f"{well.well_id}_{int(row)}" for row in incumbent.row_index)
        settings = (
            ("ncc_multiscale", METHOD_NCC, False),
            ("pf_multiscale", METHOD_PARTICLE_FILTER, False),
            ("beam_ensemble", METHOD_BEAM, True),
        )
        for candidate, method, beam_ensemble in settings:
            result = family.predict(
                horizontal_frame=horizontal,
                typewell_frame=typewell,
                incumbent_prediction=incumbent.prediction,
                well_id=well.well_id,
                paired_typewell_id=well.well_id,
                method=method,
                control_id=CONTROL_NONE,
                multiscale=True,
                beam_ensemble=beam_ensemble,
                seeds=SEEDS,
            )
            if not np.array_equal(result.row_index, incumbent.row_index):
                raise ValueError(f"visible-test row keys changed for {well.well_id}")
            if not (
                np.isfinite(result.prediction).all()
                and np.isfinite(result.uncertainty).all()
                and np.isfinite(result.effective_particle_count).all()
            ):
                raise ValueError(f"non-finite visible-test output for {well.well_id}")
            seed_finite = all(
                np.isfinite(values).all() for values in result.seed_predictions.values()
            )
            records.append(
                {
                    "well_id": well.well_id,
                    "candidate": candidate,
                    "rows": int(len(result.row_index)),
                    "status": result.status,
                    "fallback": result.status == ALIGNMENT_STATUS_IDENTITY,
                    "confidence": float(result.confidence),
                    "finite": True,
                    "seed_count": int(len(result.seed_predictions)),
                    "seed_finite": bool(seed_finite),
                    "prediction_sha256": _digest(result.prediction),
                    "uncertainty_mean": float(np.mean(result.uncertainty)),
                    "effective_particles_mean": float(
                        np.mean(result.effective_particle_count)
                    ),
                }
            )
    if observed_ids != list(sample_ids):
        raise ValueError(
            "dynamic visible-test rows do not match sample submission order"
        )
    after = capture_data_manifest(data_root)
    if before != after:
        raise RuntimeError("data manifest changed during visible-test smoke")
    payload = {
        "schema_version": "safe_physics_visible_test_smoke_v1",
        "passed": True,
        "dynamic_test_wells": [well.well_id for well in tests],
        "rows": int(len(observed_ids)),
        "sample_id_order_match": True,
        "records": records,
        "data_manifest_before_after_equal": True,
        "submission_created": False,
        "submission_uploaded": False,
    }
    _write_json(Path(output_path), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold-config", type=Path, required=True)
    parser.add_argument("--incumbent-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_smoke(
        data_root=args.data_root,
        fold_config=args.fold_config,
        incumbent_root=args.incumbent_root,
        cache_root=args.cache_root,
        output_path=args.output_path,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
