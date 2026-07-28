"""Offline visible-test smoke audit for external sequence-family checkpoints.

The smoke path is intentionally not a submission path.  It dynamically
discovers visible test wells, proves their prediction-row keys against
``sample_submission.csv``, runs the immutable incumbent and selected external
checkpoints, and writes only a compact JSON audit.  No prediction CSV is
materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import enumerate_wells
from ..submission import (
    align_predictions_to_sample,
    validate_sample_submission_contract,
)
from .components import validate_residual_components
from .contracts import SequenceModelInput
from .controls import (
    CONTROL_GR_ORDER_SHUFFLE,
    CONTROL_NONE,
    CONTROL_NO_CROSS,
    CONTROL_PAIR_SHUFFLE,
    CONTROL_PHASE_DESTROY,
    CONTROL_WRONG_TYPEWELL,
    ControlSpec,
    apply_individual_control,
    paired_control_inputs,
)
from .data import (
    FixedGridConfig,
    build_sequence_model_input,
    prepare_fixed_grid_well,
)
from .incumbent import DEFAULT_INCUMBENT_SEEDS, HeelIncumbentProvider
from .targets import SmoothTargetConfig, grid_prediction_to_row_components
from .torch_families import load_torch_family


SMOKE_SCHEMA_VERSION = "external_visible_test_smoke_v2"
PAIRING_CONTROLS = {
    CONTROL_NONE,
    CONTROL_WRONG_TYPEWELL,
    CONTROL_PAIR_SHUFFLE,
}
INDIVIDUAL_CONTROLS = {
    CONTROL_GR_ORDER_SHUFFLE,
    CONTROL_PHASE_DESTROY,
    CONTROL_NO_CROSS,
}


@dataclass(frozen=True)
class VisibleCheckpoint:
    """One requested and provenance-checked fold checkpoint."""

    variant_id: str
    family: str
    control_id: str
    frozen_random_encoder: bool
    seed: int
    fold: int
    path: Path
    expected_training_ids_sha256: str

    def __post_init__(self) -> None:
        variant = str(self.variant_id)
        family = str(self.family)
        control = str(self.control_id)
        seed = int(self.seed)
        fold = int(self.fold)
        path = Path(self.path).resolve()
        digest = str(self.expected_training_ids_sha256)
        if not variant or not family:
            raise ValueError("checkpoint variant and family must be non-empty")
        if control not in PAIRING_CONTROLS | INDIVIDUAL_CONTROLS:
            raise ValueError(f"unsupported checkpoint control: {control}")
        if isinstance(self.seed, bool) or seed != self.seed or seed < 0:
            raise ValueError("checkpoint seed must be a non-negative integer")
        if isinstance(self.fold, bool) or fold != self.fold or fold not in range(5):
            raise ValueError("checkpoint fold must be in 0..4")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("expected training-ID digest is not SHA-256")
        if not isinstance(self.frozen_random_encoder, bool):
            raise TypeError("frozen_random_encoder must be bool")
        object.__setattr__(self, "variant_id", variant)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "control_id", control)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "expected_training_ids_sha256", digest)


def _json_mapping(path: Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read valid JSON: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return payload


def _absolute_config_path(config: Mapping[str, Any], key: str) -> Path:
    try:
        path = Path(str(config[key]))
    except KeyError as error:
        raise ValueError(f"run config is missing {key!r}") from error
    if not path.is_absolute():
        raise ValueError(f"run config path {key!r} must be absolute")
    return path.resolve()


def _unique_selection(
    values: Iterable[Any] | None,
    *,
    default: Sequence[Any],
    name: str,
) -> tuple[Any, ...]:
    selected = tuple(default if values is None else values)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"{name} selection must be non-empty and unique")
    return selected


def resolve_visible_checkpoints(
    run_root: Path,
    run_config: Mapping[str, Any],
    *,
    variants: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    folds: Iterable[int] | None = None,
) -> tuple[VisibleCheckpoint, ...]:
    """Resolve a complete Cartesian checkpoint selection or fail closed."""

    root = Path(run_root).resolve()
    raw_variants = run_config.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("run config must contain a non-empty variants list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in raw_variants:
        if not isinstance(record, dict):
            raise ValueError("run config variant records must be objects")
        variant_id = str(record.get("variant_id", ""))
        if not variant_id or variant_id in by_id:
            raise ValueError("run config variant IDs must be non-empty and unique")
        by_id[variant_id] = record

    default_variants = tuple(
        variant_id
        for variant_id, record in by_id.items()
        if str(record.get("control_id", CONTROL_NONE)) == CONTROL_NONE
        and not bool(record.get("frozen_random_encoder", False))
    )
    selected_variants = tuple(
        map(
            str,
            _unique_selection(
                variants,
                default=default_variants,
                name="variant",
            ),
        )
    )
    unknown_variants = sorted(set(selected_variants) - set(by_id))
    if unknown_variants:
        raise ValueError(f"unknown variants: {unknown_variants}")

    raw_seeds = run_config.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("run config must contain a non-empty seeds list")
    configured_seeds = tuple(map(int, raw_seeds))
    selected_seeds = tuple(
        map(
            int,
            _unique_selection(seeds, default=configured_seeds, name="seed"),
        )
    )
    unknown_seeds = sorted(set(selected_seeds) - set(configured_seeds))
    if unknown_seeds:
        raise ValueError(f"seeds are not declared in run config: {unknown_seeds}")

    selected_folds = tuple(
        map(
            int,
            _unique_selection(folds, default=tuple(range(5)), name="fold"),
        )
    )
    if any(fold not in range(5) for fold in selected_folds):
        raise ValueError("fold selection must be within 0..4")
    cohort = run_config.get("cohort")
    if not isinstance(cohort, dict):
        raise ValueError("run config must contain a cohort object")

    result: list[VisibleCheckpoint] = []
    for variant_id in selected_variants:
        variant = by_id[variant_id]
        family = str(variant.get("family", ""))
        control_id = str(variant.get("control_id", CONTROL_NONE))
        frozen = variant.get("frozen_random_encoder", False)
        if not isinstance(frozen, bool):
            raise ValueError(f"variant {variant_id} has invalid frozen flag")
        for seed in selected_seeds:
            for fold in selected_folds:
                fold_record = cohort.get(str(fold))
                if not isinstance(fold_record, dict):
                    raise ValueError(f"run config cohort is missing fold {fold}")
                training_digest = str(fold_record.get("training_ids_sha256", ""))
                path = (
                    root / "models" / variant_id / f"seed_{seed}" / f"fold_{fold}.pt"
                ).resolve()
                if root not in path.parents:
                    raise ValueError("resolved checkpoint escaped run root")
                if not path.is_file():
                    raise FileNotFoundError(f"selected checkpoint is missing: {path}")
                result.append(
                    VisibleCheckpoint(
                        variant_id=variant_id,
                        family=family,
                        control_id=control_id,
                        frozen_random_encoder=frozen,
                        seed=seed,
                        fold=fold,
                        path=path,
                        expected_training_ids_sha256=training_digest,
                    )
                )
    return tuple(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for identifier in ids:
        digest.update(str(identifier).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _ordered_prediction_sha256(ids: Sequence[str], values: np.ndarray) -> str:
    vector = np.asarray(values, dtype="<f8")
    if vector.shape != (len(ids),) or not np.isfinite(vector).all():
        raise ValueError("ordered prediction vector is invalid")
    digest = hashlib.sha256()
    for identifier, value in zip(ids, vector, strict=True):
        digest.update(str(identifier).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray([value], dtype="<f8").tobytes())
    return digest.hexdigest()


def _data_manifest(root: Path) -> dict[str, Any]:
    data = Path(root).resolve()
    if not data.is_dir():
        raise FileNotFoundError(f"competition data root does not exist: {data}")
    rows: list[str] = []
    count = 0
    total = 0
    for path in sorted(item for item in data.rglob("*") if item.is_file()):
        relative = path.relative_to(data).as_posix()
        stat = path.stat()
        rows.append(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}")
        count += 1
        total += stat.st_size
    return {
        "file_count": count,
        "total_bytes": total,
        "name_size_mtime_sha256": hashlib.sha256(
            "\n".join(rows).encode("utf-8")
        ).hexdigest(),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _variant_inputs(
    base_inputs: Sequence[SequenceModelInput],
    *,
    control_id: str,
    seed: int,
) -> dict[str, SequenceModelInput]:
    if control_id in PAIRING_CONTROLS:
        return paired_control_inputs(
            base_inputs,
            control_id=control_id,
            seed=seed,
        )
    if control_id in INDIVIDUAL_CONTROLS:
        spec = ControlSpec(control_id=control_id, seed=seed)
        return {
            input_well.well_id: apply_individual_control(
                input_well,
                spec=spec,
            )
            for input_well in base_inputs
        }
    raise ValueError(f"unsupported control: {control_id}")


def _prediction_rows(
    inputs: Mapping[str, SequenceModelInput],
    corrections: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    if set(inputs) != set(corrections):
        raise ValueError("prediction correction wells do not match model inputs")
    for well_id in sorted(inputs):
        inference = inputs[well_id]
        correction = np.asarray(corrections[well_id], dtype=np.float64)
        if (
            correction.shape != inference.row_index.shape
            or not np.isfinite(correction).all()
        ):
            raise ValueError(f"invalid correction rows for {well_id}")
        blocks.append(
            pd.DataFrame(
                {
                    "id": [
                        f"{well_id}_{int(row_index)}"
                        for row_index in inference.row_index
                    ],
                    "tvt": correction,
                }
            )
        )
    result = pd.concat(blocks, ignore_index=True)
    if result["id"].duplicated().any():
        raise ValueError("visible-test prediction rows contain duplicate IDs")
    return result


def _align_vector(
    sample: pd.DataFrame,
    inputs: Mapping[str, SequenceModelInput],
    values: Mapping[str, np.ndarray],
) -> np.ndarray:
    frame = _prediction_rows(inputs, values)
    aligned = align_predictions_to_sample(sample, frame)
    result = aligned["tvt"].to_numpy(dtype=np.float64)
    if result.shape != (len(sample),) or not np.isfinite(result).all():
        raise ValueError("sample-aligned vector is invalid")
    return result


def _finite_stats(values: np.ndarray) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("finite statistics require a non-empty finite vector")
    return {
        "count": int(len(vector)),
        "minimum": float(np.min(vector)),
        "maximum": float(np.max(vector)),
        "mean": float(np.mean(vector)),
        "standard_deviation": float(np.std(vector)),
    }


def _validate_checkpoint_state(
    state: Any,
    checkpoint: VisibleCheckpoint,
) -> dict[str, Any]:
    manifest = state.manifest
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    mismatches: list[str] = []
    if str(state.config.family) != checkpoint.family:
        mismatches.append("family")
    if int(state.config.seed) != checkpoint.seed:
        mismatches.append("seed")
    if bool(state.config.frozen_random_encoder) != checkpoint.frozen_random_encoder:
        mismatches.append("frozen_random_encoder")
    if (
        str(manifest.get("fit_ids_sha256", ""))
        != checkpoint.expected_training_ids_sha256
    ):
        mismatches.append("fit_ids_sha256")
    if str(manifest.get("family", "")) != checkpoint.family:
        mismatches.append("manifest_family")
    if int(manifest.get("seed", -1)) != checkpoint.seed:
        mismatches.append("manifest_seed")
    if bool(manifest.get("architecture_only")) is not True:
        mismatches.append("architecture_only")
    if mismatches:
        raise ValueError(
            f"checkpoint provenance mismatch ({checkpoint.path}): {mismatches}"
        )
    return manifest


def _output_path(run_root: Path, output_json: Path | None) -> Path:
    root = Path(run_root).resolve()
    destination = (
        root / "visible_test_smoke.json" if output_json is None else Path(output_json)
    )
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if root not in destination.parents:
        raise ValueError("smoke JSON output must remain inside run root")
    if destination.suffix.lower() != ".json":
        raise ValueError("smoke output must use a .json suffix")
    return destination


def run_visible_test_smoke(
    *,
    run_root: Path,
    variants: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    folds: Iterable[int] | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Run selected checkpoints on the visible test set and emit JSON only."""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"portfolio run root does not exist: {root}")
    config_path = root / "run_config.json"
    config = _json_mapping(config_path)
    if bool(config.get("submission_created")) is not False:
        raise ValueError("run config must explicitly state submission_created=false")
    data_root = _absolute_config_path(config, "data_root")
    fold_config = _absolute_config_path(config, "fold_config")
    incumbent_root = _absolute_config_path(config, "incumbent_root")
    cache_root = _absolute_config_path(config, "cache_root")
    destination = _output_path(root, output_json)
    if (
        data_root == root
        or data_root in root.parents
        or data_root in destination.parents
    ):
        raise ValueError("smoke output/run root must not be inside competition data")
    existing_submission_csvs = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.csv")
        if "submission" in path.name.lower()
    )
    if existing_submission_csvs:
        raise ValueError(
            f"run root already contains submission CSVs: {existing_submission_csvs}"
        )

    checkpoints = resolve_visible_checkpoints(
        root,
        config,
        variants=variants,
        seeds=seeds,
        folds=folds,
    )
    grid_record = config.get("fixed_grid")
    if not isinstance(grid_record, dict) or not isinstance(
        grid_record.get("config"), dict
    ):
        raise ValueError("run config fixed_grid.config is missing")
    grid_config = FixedGridConfig(**dict(grid_record["config"]))
    target_record = config.get("residual_target")
    if not isinstance(target_record, dict):
        raise ValueError("run config residual_target is missing")
    target_config = SmoothTargetConfig(
        retained_dct_modes=int(target_record["retained_dct_modes"]),
        maximum_absolute_correction=float(target_record["maximum_absolute_correction"]),
    )
    shrinkage = float(config.get("shrinkage", 1.0))
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("run config shrinkage must be finite within [0, 1]")

    before_data = _data_manifest(data_root)
    declared_data = config.get("data_manifest")
    if not isinstance(declared_data, dict) or declared_data != before_data:
        raise ValueError(
            "current competition data manifest differs from the portfolio run config"
        )
    sample_path = data_root / "sample_submission.csv"
    sample = pd.read_csv(sample_path, dtype={"id": "string"})
    parsed = validate_sample_submission_contract(sample, data_root / "test")
    sample_ids = tuple(parsed["id"].astype(str))
    test_wells = enumerate_wells(data_root, "test")
    test_ids = tuple(well.well_id for well in test_wells)
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("dynamic test-well enumeration returned duplicate IDs")

    sequence_cache = cache_root / "external_sequence_v1"
    prepared = {
        well.well_id: prepare_fixed_grid_well(
            well,
            split="test",
            cache_root=sequence_cache,
            config=grid_config,
        )
        for well in test_wells
    }
    provider = HeelIncumbentProvider(
        data_root=data_root,
        fold_config=fold_config,
        artifact_root=incumbent_root,
        cache_root=cache_root,
        seeds=tuple(map(int, config.get("incumbent_seeds", DEFAULT_INCUMBENT_SEEDS))),
    )
    selected_folds = tuple(sorted({checkpoint.fold for checkpoint in checkpoints}))
    inputs_by_fold: dict[int, dict[str, SequenceModelInput]] = {}
    incumbent_by_fold: dict[int, np.ndarray] = {}
    incumbent_hashes_by_fold: dict[int, tuple[str, ...]] = {}
    incumbent_hashes: set[str] = set()
    for fold in selected_folds:
        fold_inputs: dict[str, SequenceModelInput] = {}
        fold_hashes: set[str] = set()
        for well_id in test_ids:
            incumbent = provider.predict_with_folds(
                well_id,
                folds=(fold,),
                split="test",
            )
            sequence = prepared[well_id]
            if not np.array_equal(incumbent.row_index, sequence.row_index):
                raise ValueError(
                    f"incumbent row keys differ from sequence keys: {well_id}"
                )
            if tuple(incumbent.fold_states) != (fold,):
                raise ValueError(
                    "visible-test checkpoint conditioning did not use exactly "
                    f"fold {fold}"
                )
            if tuple(incumbent.seeds) != tuple(provider.seeds):
                raise ValueError("visible-test incumbent seed provenance changed")
            fold_hashes.update(incumbent.state_sha256)
            incumbent_hashes.update(incumbent.state_sha256)
            fold_inputs[well_id] = build_sequence_model_input(
                sequence,
                incumbent.prediction,
                grid_points=grid_config.horizontal_points,
            )
        inputs_by_fold[fold] = fold_inputs
        incumbent_hashes_by_fold[fold] = tuple(sorted(fold_hashes))
        incumbent_by_well = {
            well_id: fold_inputs[well_id].incumbent_prediction for well_id in test_ids
        }
        incumbent_by_fold[fold] = _align_vector(
            sample,
            fold_inputs,
            incumbent_by_well,
        )

    incumbent_ordered = np.mean(
        np.stack([incumbent_by_fold[fold] for fold in selected_folds]),
        axis=0,
    )
    if not np.isfinite(incumbent_ordered).all():
        raise ValueError("fold-averaged visible-test incumbent is non-finite")
    checkpoint_finals: list[np.ndarray] = []
    checkpoint_audits: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        base_inputs = inputs_by_fold[checkpoint.fold]
        state = load_torch_family(checkpoint.path)
        manifest = _validate_checkpoint_state(state, checkpoint)
        controlled = _variant_inputs(
            tuple(base_inputs[well_id] for well_id in test_ids),
            control_id=checkpoint.control_id,
            seed=20260728 + checkpoint.fold,
        )
        channel_names = controlled[test_ids[0]].channel_names
        if any(
            controlled[well_id].channel_names != channel_names for well_id in test_ids
        ):
            raise ValueError("visible-test model-input channels are inconsistent")
        x = np.stack([controlled[well_id].x for well_id in test_ids]).astype(np.float32)
        grid_predictions = state.predict(
            x,
            channel_names=channel_names,
            ids=test_ids,
        )
        if (
            grid_predictions.shape != (len(test_ids), grid_config.horizontal_points)
            or not np.isfinite(grid_predictions).all()
        ):
            raise ValueError(
                f"checkpoint returned invalid grid predictions: {checkpoint.path}"
            )
        corrections: dict[str, np.ndarray] = {}
        for index, well_id in enumerate(test_ids):
            inference = controlled[well_id]
            components = grid_prediction_to_row_components(
                grid_predictions[index],
                inference,
                config=target_config,
            )
            validate_residual_components(
                components,
                inference.incumbent_prediction,
                inference.row_coordinate,
            )
            if not components.eligible:
                raise ValueError("visible-test smoke rejects identity fallbacks")
            corrections[well_id] = components.correction
        ordered_correction = _align_vector(sample, controlled, corrections)
        checkpoint_incumbent = incumbent_by_fold[checkpoint.fold]
        ordered_final = checkpoint_incumbent + shrinkage * ordered_correction
        if not np.isfinite(ordered_final).all():
            raise ValueError("checkpoint final prediction is non-finite")
        checkpoint_finals.append(ordered_final)
        checkpoint_audits.append(
            {
                "variant_id": checkpoint.variant_id,
                "family": checkpoint.family,
                "control_id": checkpoint.control_id,
                "frozen_random_encoder": checkpoint.frozen_random_encoder,
                "seed": checkpoint.seed,
                "fold": checkpoint.fold,
                "checkpoint": str(checkpoint.path),
                "checkpoint_sha256": _sha256_file(checkpoint.path),
                "fit_ids_sha256": checkpoint.expected_training_ids_sha256,
                "parameter_sha256": manifest.get("parameter_sha256"),
                "incumbent_conditioning_fold": checkpoint.fold,
                "incumbent_state_sha256": list(
                    incumbent_hashes_by_fold[checkpoint.fold]
                ),
                "ordered_incumbent_sha256": _ordered_prediction_sha256(
                    sample_ids,
                    checkpoint_incumbent,
                ),
                "input_channels": int(x.shape[1]),
                "grid_points": int(x.shape[2]),
                "correction": _finite_stats(ordered_correction),
                "final": _finite_stats(ordered_final),
                "ordered_final_sha256": _ordered_prediction_sha256(
                    sample_ids,
                    ordered_final,
                ),
            }
        )

    smoke_average = np.mean(np.stack(checkpoint_finals), axis=0)
    if not np.isfinite(smoke_average).all():
        raise ValueError("smoke-average prediction is non-finite")
    after_data = _data_manifest(data_root)
    if before_data != after_data:
        raise RuntimeError("competition data manifest changed during visible smoke")
    if any(
        path.suffix.lower() == ".csv" and "submission" in path.name.lower()
        for path in root.rglob("*")
        if path.is_file()
    ):
        raise RuntimeError("a submission CSV appeared during visible smoke")

    audit: dict[str, Any] = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "run_root": str(root),
        "run_config": str(config_path),
        "run_config_sha256": _sha256_file(config_path),
        "profile": config.get("profile"),
        "architecture_only": bool(config.get("architecture_only")),
        "external_pretrained_weights_used": config.get(
            "external_pretrained_weights_used",
            [],
        ),
        "labels_read": False,
        "submission_csv_created": False,
        "sample_submission": str(sample_path),
        "sample_submission_sha256": _sha256_file(sample_path),
        "test_wells": list(test_ids),
        "test_well_count": len(test_ids),
        "prediction_rows": len(sample_ids),
        "sample_order_first_id": sample_ids[0],
        "sample_order_last_id": sample_ids[-1],
        "ordered_sample_ids_sha256": _ordered_ids_sha256(sample_ids),
        "incumbent": {
            "folds": list(selected_folds),
            "seeds": list(provider.seeds),
            "state_count": len(incumbent_hashes),
            "state_sha256": sorted(incumbent_hashes),
            "state_sha256_by_fold": {
                str(fold): list(incumbent_hashes_by_fold[fold])
                for fold in selected_folds
            },
            "prediction": _finite_stats(incumbent_ordered),
            "ordered_prediction_sha256": _ordered_prediction_sha256(
                sample_ids,
                incumbent_ordered,
            ),
        },
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoint_audits,
        "smoke_average_policy": (
            "unweighted mean of selected checkpoint final predictions; every "
            "checkpoint is conditioned on the matching fold-specific incumbent; "
            f"fixed run-config shrinkage={shrinkage}"
        ),
        "smoke_average": {
            **_finite_stats(smoke_average),
            "ordered_prediction_sha256": _ordered_prediction_sha256(
                sample_ids,
                smoke_average,
            ),
        },
        "key_checks": {
            "dynamic_test_enumeration": True,
            "sample_contract_exact": True,
            "duplicate_ids": 0,
            "coverage_rows": len(sample_ids),
            "sample_order_preserved_by_explicit_id_join": True,
            "finite_incumbent": True,
            "finite_checkpoint_outputs": True,
            "finite_smoke_average": True,
            "fold_specific_incumbent_conditioning": True,
        },
        "data_read_only_manifest_before": before_data,
        "data_read_only_manifest_after": after_data,
        "data_manifest_matches_run_config": True,
        "data_manifest_equal": True,
        "output_json": str(destination),
    }
    _atomic_json(destination, audit)
    return audit


__all__ = [
    "SMOKE_SCHEMA_VERSION",
    "VisibleCheckpoint",
    "resolve_visible_checkpoints",
    "run_visible_test_smoke",
]
