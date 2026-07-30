"""Offline dynamic visible-test inference smoke for trained TTM quick heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rogii.data import enumerate_wells
from rogii.external_models.artifacts import capture_data_manifest
from rogii.external_models.components import validate_residual_components
from rogii.external_models.data import (
    FixedGridConfig,
    build_sequence_model_input,
    prepare_fixed_grid_well,
)
from rogii.external_models.incumbent import HeelIncumbentProvider
from rogii.external_models.targets import (
    SmoothTargetConfig,
    grid_prediction_to_row_components,
)
from rogii.external_models.ttm import (
    forbid_network,
    freeze_backbone,
    load_ttm_forecaster,
    resolve_device,
    sha256_file,
    verify_ttm_checkpoint,
)
from rogii.external_models.ttm_embeddings import (
    encode_ttm_batch,
    load_head_checkpoint,
    make_ttm_batch,
    predict_ttm_head,
)
from rogii.submission import (
    align_predictions_to_sample,
    validate_sample_submission_contract,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ordered_hash(ids: tuple[str, ...], values: np.ndarray) -> str:
    vector = np.asarray(values, dtype="<f8")
    if vector.shape != (len(ids),) or not np.isfinite(vector).all():
        raise ValueError("visible-test ordered prediction is invalid")
    digest = hashlib.sha256()
    for identifier, value in zip(ids, vector, strict=True):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray([value], dtype="<f8").tobytes())
    return digest.hexdigest()


def run_visible_smoke(
    *,
    run_dir: Path,
    output_path: Path | None,
    device: str,
    offline: bool,
    num_workers: int,
    batch_size: int,
    embedding_batch_size: int,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    config = _json(root / "run_config.json")
    if config.get("profile") != "quick":
        raise ValueError("visible TTM smoke requires a completed quick run")
    if "ttm_pretrained_frozen" not in config["variants"]:
        raise ValueError("quick run lacks ttm_pretrained_frozen")
    data_root = Path(config["data_root"])
    before = capture_data_manifest(data_root)
    if before != config["data_manifest"]:
        raise ValueError("competition data differs from quick run")
    identity = verify_ttm_checkpoint(config["weight_dir"])
    resolved_device = resolve_device(device)
    sample_path = data_root / "sample_submission.csv"
    sample = pd.read_csv(sample_path, dtype={"id": "string"})
    parsed = validate_sample_submission_contract(sample, data_root / "test")
    sample_ids = tuple(parsed["id"].astype(str))
    test_wells = enumerate_wells(data_root, "test")
    test_ids = tuple(well.well_id for well in test_wells)
    grid = FixedGridConfig(**dict(config["fixed_grid"]["config"]))
    target = SmoothTargetConfig(
        retained_dct_modes=int(
            config["residual_target"]["retained_dct_modes"]
        ),
        maximum_absolute_correction=float(
            config["residual_target"]["maximum_absolute_correction"]
        ),
    )
    cache = Path(config["cache_dir"])
    prepared = {
        well.well_id: prepare_fixed_grid_well(
            well,
            split="test",
            cache_root=cache / "external-sequence-ttm",
            config=grid,
        )
        for well in test_wells
    }
    provider = HeelIncumbentProvider(
        data_root=data_root,
        fold_config=Path(config["fold_config"]),
        artifact_root=Path(config["incumbent_root"]),
        cache_root=cache / "incumbent-provider",
    )
    with forbid_network(offline):
        model = load_ttm_forecaster(
            config["weight_dir"],
            pretrained=True,
            device=resolved_device,
        )
    backbone = freeze_backbone(model)
    checkpoint_predictions: list[np.ndarray] = []
    checkpoint_records: list[dict[str, Any]] = []
    for fold in range(5):
        inputs = {}
        for well_id in test_ids:
            incumbent = provider.predict_with_folds(
                well_id,
                folds=(fold,),
                split="test",
            )
            if not np.array_equal(
                incumbent.row_index,
                prepared[well_id].row_index,
            ):
                raise ValueError(f"visible TTM incumbent keys differ: {well_id}")
            inputs[well_id] = build_sequence_model_input(
                prepared[well_id],
                incumbent.prediction,
                grid_points=512,
            )
        batch = make_ttm_batch(inputs, ids=test_ids)
        horizontal, typewell = encode_ttm_batch(
            backbone,
            batch,
            device=resolved_device,
            batch_size=embedding_batch_size,
        )
        for seed in config["seeds"]:
            checkpoint_path = (
                root
                / "checkpoints"
                / "ttm_pretrained_frozen"
                / f"seed_{seed}"
                / f"fold_{fold}.pt"
            )
            checkpoint = load_head_checkpoint(checkpoint_path)
            grids = predict_ttm_head(
                horizontal=horizontal,
                typewell=typewell,
                batch=batch,
                checkpoint=checkpoint,
                device=resolved_device,
                batch_size=batch_size,
            )
            blocks = []
            for index, well_id in enumerate(test_ids):
                inference = inputs[well_id]
                components = grid_prediction_to_row_components(
                    grids[index],
                    inference,
                    config=target,
                )
                validate_residual_components(
                    components,
                    inference.incumbent_prediction,
                    inference.row_coordinate,
                )
                blocks.append(
                    pd.DataFrame(
                        {
                            "id": [
                                f"{well_id}_{int(row)}"
                                for row in inference.row_index
                            ],
                            "tvt": components.final,
                        }
                    )
                )
            aligned = align_predictions_to_sample(
                sample,
                pd.concat(blocks, ignore_index=True),
            )
            values = aligned["tvt"].to_numpy(dtype=np.float64)
            checkpoint_predictions.append(values)
            checkpoint_records.append(
                {
                    "fold": fold,
                    "seed": int(seed),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "rows": int(len(values)),
                    "finite": True,
                    "ordered_prediction_sha256": _ordered_hash(
                        sample_ids,
                        values,
                    ),
                }
            )
    average = np.mean(np.stack(checkpoint_predictions), axis=0)
    if not np.isfinite(average).all():
        raise ValueError("visible TTM ensemble is non-finite")
    after = capture_data_manifest(data_root)
    if after != before:
        raise RuntimeError("competition data changed during visible TTM smoke")
    submissions = [
        str(path)
        for path in root.rglob("*.csv")
        if "submission" in path.name.lower()
    ]
    if submissions:
        raise ValueError(f"TTM run contains submission CSVs: {submissions}")
    payload = {
        "schema_version": "ttm_visible_test_smoke_v1",
        "passed": True,
        "run_dir": str(root),
        "offline": bool(offline),
        "network_forbidden": bool(offline),
        "device": str(resolved_device),
        "dynamic_test_wells": list(test_ids),
        "test_well_count": len(test_ids),
        "sample_rows": len(sample_ids),
        "sample_id_order_match": True,
        "sample_submission_sha256": sha256_file(sample_path),
        "weight_identity": identity,
        "checkpoint_count": len(checkpoint_records),
        "checkpoints": checkpoint_records,
        "ensemble_ordered_prediction_sha256": _ordered_hash(
            sample_ids,
            average,
        ),
        "finite": True,
        "labels_read": False,
        "data_manifest_before_after_equal": True,
        "submission_created": False,
        "submission_uploaded": False,
    }
    destination = (
        root / "visible_test_smoke.json"
        if output_path is None
        else Path(output_path).resolve()
    )
    if root not in destination.parents:
        raise ValueError("visible smoke output must stay inside the quick run")
    _write_json(destination, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dynamic label-free visible-test TTM inference smoke."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_visible_smoke(
        run_dir=args.run_dir,
        output_path=args.output_path,
        device=args.device,
        offline=args.offline,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
