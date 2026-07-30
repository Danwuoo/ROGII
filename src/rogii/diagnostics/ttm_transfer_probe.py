"""Strict matched Granite TTM R2 pretrained-transfer experiment."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..data import enumerate_wells
from ..external_models.artifacts import capture_data_manifest
from ..external_models.contracts import ResidualTrainingWell, SequenceModelInput
from ..external_models.data import (
    FixedGridConfig,
    build_sequence_model_input,
    fixed_grid_manifest,
    raw_source_provenance_digest,
    raw_well_source_provenance,
)
from ..external_models.incumbent import (
    DEFAULT_INCUMBENT_SEEDS,
    HeelIncumbentProvider,
)
from ..external_models.targets import (
    SmoothTargetConfig,
    make_grid_residual_target,
    target_manifest,
)
from ..external_models.ttm import (
    TTM_BACKBONE_PARAMETER_COUNT,
    TTM_D_MODEL,
    TTM_PARAMETER_COUNT,
    TTM_PATCH_COUNT,
    TTM_RANDOM_BACKBONE_SEED,
    environment_manifest,
    forbid_network,
    freeze_backbone,
    load_ttm_forecaster,
    resolve_device,
    sha256_file,
    verify_ttm_checkpoint,
)
from ..external_models.ttm_audit import (
    INCUMBENT_COLUMN,
    evaluate_ttm_oof,
    locked_quick_gate,
    prediction_column,
    seed_prediction_column,
)
from ..external_models.ttm_embeddings import (
    ELIGIBLE_FOR_MODEL_SELECTION,
    PRETRAINED_VARIANTS,
    TTMHeadConfig,
    TTMInputBatch,
    TTM_VARIANTS,
    TTMPairedInteractionHead,
    VARIANT_CONTROL,
    apply_ttm_control,
    encode_ttm_batch,
    fit_ttm_head,
    fit_ttm_scratch_model,
    head_parameter_count,
    load_head_checkpoint,
    make_ttm_batch,
    predict_ttm_head,
    predict_ttm_scratch_model,
)
from ..folds import FOLD_COUNT, load_fold_mapping
from ..portfolio_metrics import validate_oof_frame
from .external_model_portfolio_experiment import (
    DEFAULT_FULL_OOF,
    DEFAULT_INCUMBENT_ROOT,
    DEFAULT_QUICK_SOURCE,
    _array_sha256,
    _atomic_csv,
    _atomic_json,
    _cached_incumbent,
    _canonical_sha256,
    _fold_state_digest,
    _full_cohort,
    _load_fold_training_targets,
    _prediction_block,
    _prepare_sequences,
    _quick_cohort,
    _read_incumbent_oof,
    _read_incumbent_reference,
    _same_keys,
)


LOCKED_CONFIG = Path("configs/external/ttm_r2_transfer_probe_20260729.json")
DEFAULT_GRID = FixedGridConfig(horizontal_points=512, typewell_points=512)
DEFAULT_TARGET = SmoothTargetConfig(
    retained_dct_modes=16,
    maximum_absolute_correction=80.0,
)
RUN_SCHEMA = "ttm_transfer_probe_run_v1"
QUICK_SEEDS = (20260729, 20260730, 20260731)
FULL_SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)
QUICK_VARIANTS = (
    "ttm_random_frozen",
    "ttm_pretrained_frozen",
    "ttm_pretrained_wrong_typewell",
    "ttm_pretrained_pair_shuffle",
    "ttm_pretrained_gr_order_shuffle",
    "ttm_pretrained_phase_destroy",
)
FULL_VARIANTS = ("ttm_random_frozen", "ttm_pretrained_frozen")


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _source_hashes() -> dict[str, str]:
    classes = (
        run_ttm_transfer_experiment,
        apply_ttm_control,
        TTMPairedInteractionHead,
        locked_quick_gate,
    )
    paths = sorted({Path(inspect.getsourcefile(value) or "").resolve() for value in classes})
    return {str(path): sha256_file(path) for path in paths}


def _safe_run_dir(
    run_dir: Path,
    *,
    data_root: Path,
    resume: bool,
    run_config: Mapping[str, Any],
) -> Path:
    root = Path(run_dir).resolve()
    data = Path(data_root).resolve()
    if root == data or data in root.parents:
        raise ValueError("TTM run directory cannot be inside competition data")
    config_path = root / "run_config.json"
    if resume:
        if not config_path.is_file():
            raise FileNotFoundError("--resume requires an existing run_config.json")
        if _json(config_path) != json.loads(
            json.dumps(dict(run_config), sort_keys=True, allow_nan=False)
        ):
            raise ValueError("resume run config differs from disk")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                f"new TTM run directory must be empty or absent: {root}"
            )
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_path, run_config)
    for name in (
        "checkpoints",
        "prediction_blocks",
        "embedding_cache",
        "incumbent_cache",
        "metrics",
        "provenance",
    ):
        (root / name).mkdir(exist_ok=True)
    return root


def _embedding_digest(batch: TTMInputBatch) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(batch.ids).encode("utf-8"))
    digest.update(b"\0")
    digest.update("\n".join(batch.paired_typewell_ids).encode("utf-8"))
    for array in (
        batch.horizontal,
        batch.horizontal_mask,
        batch.typewell,
        batch.typewell_mask,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _embedding_cache(
    path: Path,
    *,
    backbone: torch.nn.Module,
    batch: TTMInputBatch,
    device: torch.device,
    batch_size: int,
    backbone_identity: str,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    expected = {
        "schema_version": "ttm_embedding_cache_v1",
        "ids": list(batch.ids),
        "paired_typewell_ids": list(batch.paired_typewell_ids),
        "control_id": batch.control_id,
        "input_sha256": _embedding_digest(batch),
        "backbone_identity": backbone_identity,
    }
    if resume and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as payload:
                provenance = json.loads(str(payload["provenance_json"].item()))
                horizontal = np.asarray(payload["horizontal"], dtype=np.float32)
                typewell = np.asarray(payload["typewell"], dtype=np.float32)
            if (
                provenance == expected
                and horizontal.shape
                == (len(batch.ids), TTM_PATCH_COUNT, TTM_D_MODEL)
                and typewell.shape == horizontal.shape
                and np.isfinite(horizontal).all()
                and np.isfinite(typewell).all()
            ):
                return horizontal, typewell, True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        raise ValueError(f"resume embedding cache differs or is corrupt: {path}")
    horizontal, typewell = encode_ttm_batch(
        backbone,
        batch,
        device=device,
        batch_size=batch_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        provenance_json=np.asarray(
            json.dumps(expected, sort_keys=True, separators=(",", ":"))
        ),
        horizontal=horizontal,
        typewell=typewell,
    )
    os.replace(temporary, path)
    return horizontal, typewell, False


def _assemble_oof(
    baseline: pd.DataFrame,
    *,
    run_root: Path,
    variants: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame:
    output = baseline.sort_values(
        ["fold", "well_id", "row_index"], kind="stable"
    ).reset_index(drop=True)
    prediction_columns = [INCUMBENT_COLUMN]
    for variant in variants:
        seed_predictions: list[np.ndarray] = []
        component_values: dict[str, list[np.ndarray]] = {
            "common": [],
            "slope": [],
            "shape": [],
        }
        for seed in seeds:
            blocks = []
            for fold in range(FOLD_COUNT):
                path = (
                    run_root
                    / "prediction_blocks"
                    / variant
                    / f"seed_{seed}"
                    / f"fold_{fold}.csv.gz"
                )
                if not path.is_file():
                    raise FileNotFoundError(f"prediction block missing: {path}")
                blocks.append(
                    pd.read_csv(
                        path,
                        dtype={
                            "well_id": "string",
                            "row_index": "int64",
                            "fold": "int8",
                        },
                        float_precision="round_trip",
                    )
                )
            current = (
                pd.concat(blocks, ignore_index=True)
                .sort_values(["fold", "well_id", "row_index"], kind="stable")
                .reset_index(drop=True)
            )
            if len(current) != len(output) or not _same_keys(output, current):
                raise ValueError(f"TTM OOF keys differ: {variant}/seed_{seed}")
            column = seed_prediction_column(variant, seed)
            values = current["prediction"].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"TTM seed prediction is non-finite: {column}")
            output[column] = values
            prediction_columns.append(column)
            seed_predictions.append(values)
            for component in component_values:
                component_values[component].append(
                    current[f"correction_{component}"].to_numpy(dtype=np.float64)
                )
        family_column = prediction_column(variant)
        output[family_column] = np.mean(np.stack(seed_predictions), axis=0)
        prediction_columns.append(family_column)
        for component, values in component_values.items():
            output[f"correction__{variant}__{component}"] = np.mean(
                np.stack(values),
                axis=0,
            )
    validate_oof_frame(
        output,
        prediction_columns,
        expected_keys=baseline,
        expected_well_ids=sorted(baseline["well_id"].astype(str).unique()),
        expected_folds=range(FOLD_COUNT),
    )
    return output


def _write_metrics(run_root: Path, metrics: Mapping[str, Any]) -> None:
    frames = {
        "pooled.csv": metrics["pooled_frame"],
        "per_fold.csv": metrics["per_fold_frame"],
        "per_well.csv.gz": metrics["per_well_frame"],
        "tail.csv": metrics["tail_frame"],
        "decomposition.csv": metrics["decomposition_frame"],
        "per_well_wins.csv": metrics["wins_frame"],
        "residual_correlation.csv": metrics["correlation_frame"].reset_index(
            names="candidate"
        ),
    }
    for name, frame in frames.items():
        _atomic_csv(run_root / "metrics" / name, frame)
    _atomic_json(run_root / "metrics" / "summary.json", metrics["summary"])


def _smoke_batch(seed: int = 20260729) -> TTMInputBatch:
    random = np.random.default_rng(seed)
    ids = tuple(f"synthetic-{index}" for index in range(4))
    horizontal = random.normal(size=(4, 512, 1)).astype(np.float32)
    typewell = random.normal(size=(4, 512, 1)).astype(np.float32)
    horizontal_mask = np.ones_like(horizontal)
    typewell_mask = np.ones_like(typewell)
    horizontal_mask[0, 450:] = 0.0
    typewell_mask[1, :32] = 0.0
    bypass = random.normal(size=(4, 16, 512)).astype(np.float32)
    prefix = np.zeros((4, 1, 512), dtype=np.float32)
    prefix[:, :, :160] = 1.0
    return TTMInputBatch(
        ids=ids,
        horizontal=horizontal,
        horizontal_mask=horizontal_mask,
        typewell=typewell,
        typewell_mask=typewell_mask,
        bypass=bypass,
        prefix_mask=prefix,
        paired_typewell_ids=ids,
        control_id="none",
    )


def run_ttm_smoke(
    *,
    run_dir: Path,
    cache_dir: Path,
    weight_dir: Path,
    data_root: Path,
    device: str,
    offline: bool,
    resume: bool,
    batch_size: int,
    embedding_batch_size: int,
) -> dict[str, Any]:
    """Synthetic forward/backward, frozen gradients, serialization, and resume."""

    started = time.perf_counter()
    before = capture_data_manifest(data_root)
    identity = verify_ttm_checkpoint(weight_dir)
    resolved_device = resolve_device(device)
    config = {
        "schema_version": RUN_SCHEMA,
        "profile": "smoke",
        "offline": bool(offline),
        "device": str(resolved_device),
        "weight_identity": identity,
        "data_root": str(Path(data_root).resolve()),
        "data_manifest": before,
        "submission_created": False,
    }
    root = _safe_run_dir(
        run_dir,
        data_root=data_root,
        resume=resume,
        run_config=config,
    )
    with forbid_network(offline):
        pretrained_model = load_ttm_forecaster(
            weight_dir,
            pretrained=True,
            device=resolved_device,
        )
        random_model = load_ttm_forecaster(
            weight_dir,
            pretrained=False,
            device=resolved_device,
        )
    pretrained_backbone = freeze_backbone(pretrained_model)
    random_backbone = freeze_backbone(random_model)
    batch = _smoke_batch()
    pretrained_h, pretrained_t = encode_ttm_batch(
        pretrained_backbone,
        batch,
        device=resolved_device,
        batch_size=embedding_batch_size,
    )
    random_h, random_t = encode_ttm_batch(
        random_backbone,
        batch,
        device=resolved_device,
        batch_size=embedding_batch_size,
    )
    if pretrained_h.shape != random_h.shape:
        raise ValueError("random/pretrained TTM embedding shapes differ")
    targets = np.random.default_rng(20260730).normal(
        scale=0.1, size=(4, 512)
    ).astype(np.float32)
    row_counts = np.asarray([64, 96, 128, 160], dtype=np.int64)
    head_config = TTMHeadConfig(
        variant_id="ttm_pretrained_frozen",
        seed=20260729,
        epochs=2,
        batch_size=batch_size,
        hidden_dim=32,
    )
    provenance = {
        "profile": "smoke",
        "fold": 0,
        "backbone": "pretrained",
        "weight_sha256": identity["weight_sha256"],
        "input_sha256": _embedding_digest(batch),
    }
    checkpoint_path = root / "checkpoints" / "resume.pt"
    first = fit_ttm_head(
        train_horizontal=pretrained_h,
        train_typewell=pretrained_t,
        train_batch=batch,
        targets=targets,
        row_counts=row_counts,
        validation_ids=("held-out-smoke",),
        checkpoint_path=checkpoint_path,
        config=head_config,
        provenance=provenance,
        device=resolved_device,
        resume=resume,
        max_epochs_this_call=None if resume else 1,
    )
    if not resume and int(first["epoch"]) != 1:
        raise ValueError("synthetic interrupted checkpoint did not stop at epoch 1")
    resumed = fit_ttm_head(
        train_horizontal=pretrained_h,
        train_typewell=pretrained_t,
        train_batch=batch,
        targets=targets,
        row_counts=row_counts,
        validation_ids=("held-out-smoke",),
        checkpoint_path=checkpoint_path,
        config=head_config,
        provenance=provenance,
        device=resolved_device,
        resume=True,
    )
    prediction = predict_ttm_head(
        horizontal=pretrained_h,
        typewell=pretrained_t,
        batch=batch,
        checkpoint=resumed,
        device=resolved_device,
        batch_size=batch_size,
    )
    fresh_path = root / "checkpoints" / "fresh.pt"
    if fresh_path.exists() and resume:
        fresh = load_head_checkpoint(fresh_path)
    else:
        fresh = fit_ttm_head(
            train_horizontal=pretrained_h,
            train_typewell=pretrained_t,
            train_batch=batch,
            targets=targets,
            row_counts=row_counts,
            validation_ids=("held-out-smoke",),
            checkpoint_path=fresh_path,
            config=head_config,
            provenance=provenance,
            device=resolved_device,
            resume=False,
        )
    fresh_prediction = predict_ttm_head(
        horizontal=pretrained_h,
        typewell=pretrained_t,
        batch=batch,
        checkpoint=fresh,
        device=resolved_device,
        batch_size=batch_size,
    )
    resume_max_delta = float(np.max(np.abs(prediction - fresh_prediction)))
    if resume_max_delta != 0.0:
        raise ValueError(f"TTM resume differs from fresh fit: {resume_max_delta}")

    head = TTMPairedInteractionHead(bypass_channels=16, hidden_dim=32).to(
        resolved_device
    )
    head_output = head(
        torch.from_numpy(pretrained_h).to(resolved_device),
        torch.from_numpy(pretrained_t).to(resolved_device),
        torch.from_numpy(batch.bypass).to(resolved_device),
        torch.from_numpy(batch.prefix_mask).to(resolved_device),
    )
    head_output.square().mean().backward()
    head_gradient = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )
    backbone_gradient = any(
        parameter.grad is not None for parameter in pretrained_backbone.parameters()
    )
    if not head_gradient or backbone_gradient:
        raise ValueError("TTM freeze/head gradient contract failed")
    random_config = TTMHeadConfig(
        variant_id="ttm_random_frozen",
        seed=20260729,
        epochs=2,
        batch_size=batch_size,
    )
    pretrained_count = head_parameter_count(head_config, bypass_channels=16)
    random_count = head_parameter_count(random_config, bypass_channels=16)
    if pretrained_count != random_count:
        raise ValueError("random/pretrained trainable parameter counts differ")
    toy_oof = pd.DataFrame(
        {
            "well_id": batch.ids,
            "row_index": np.zeros(4, dtype=np.int64),
            "fold": np.arange(4, dtype=np.int64),
            "y_true": np.zeros(4),
            "prediction": prediction[:, 0],
        }
    )
    oof_validation = validate_oof_frame(toy_oof, ("prediction",))
    after = capture_data_manifest(data_root)
    if before != after:
        raise RuntimeError("competition data manifest changed during TTM smoke")
    cuda_smoke = {
        "available": torch.cuda.is_available(),
        "performed": resolved_device.type == "cuda",
        "reason": (
            None
            if torch.cuda.is_available()
            else "installed PyTorch build exposes no CUDA device"
        ),
    }
    summary = {
        "schema_version": "ttm_smoke_v1",
        "passed": True,
        "profile": "smoke",
        "offline": bool(offline),
        "network_forbidden": bool(offline),
        "device": str(resolved_device),
        "cpu_fallback": resolved_device.type == "cpu",
        "cuda_smoke": cuda_smoke,
        "weight_identity": identity,
        "total_parameters": TTM_PARAMETER_COUNT,
        "backbone_parameters": TTM_BACKBONE_PARAMETER_COUNT,
        "embedding_shape": list(pretrained_h.shape),
        "architecture_shape_match": True,
        "frozen_backbone_has_gradient": False,
        "head_has_gradient": True,
        "matched_trainable_parameter_count": pretrained_count,
        "random_pretrained_trainable_counts_equal": True,
        "prediction_finite": True,
        "prediction_sha256": hashlib.sha256(
            np.ascontiguousarray(prediction).tobytes()
        ).hexdigest(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_serialization": True,
        "resume_exact_maximum_absolute_delta": resume_max_delta,
        "oof_validation": oof_validation,
        "data_manifest_before_after_equal": True,
        "submission_created": False,
        "submission_uploaded": False,
        "environment": environment_manifest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(root / "smoke_summary.json", summary)
    return summary


def run_ttm_transfer_experiment(
    *,
    profile: str,
    run_dir: Path,
    cache_dir: Path,
    weight_dir: Path,
    data_root: Path,
    fold_config: Path,
    locked_config_path: Path = LOCKED_CONFIG,
    incumbent_root: Path = DEFAULT_INCUMBENT_ROOT,
    incumbent_oof: Path = DEFAULT_FULL_OOF,
    quick_source: Path = DEFAULT_QUICK_SOURCE,
    variants: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    epochs: int | None = None,
    device: str = "auto",
    resume: bool = False,
    num_workers: int = 4,
    batch_size: int = 16,
    embedding_batch_size: int = 32,
    offline: bool = False,
    bootstrap_samples: int = 10_000,
    allow_adapter: bool = False,
) -> dict[str, Any]:
    if profile == "smoke":
        return run_ttm_smoke(
            run_dir=run_dir,
            cache_dir=cache_dir,
            weight_dir=weight_dir,
            data_root=data_root,
            device=device,
            offline=offline,
            resume=resume,
            batch_size=batch_size,
            embedding_batch_size=embedding_batch_size,
        )
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be explicitly smoke, quick, or full")
    selected_variants = tuple(
        (QUICK_VARIANTS if profile == "quick" else FULL_VARIANTS)
        if variants is None
        else map(str, variants)
    )
    if not selected_variants or len(set(selected_variants)) != len(selected_variants):
        raise ValueError("TTM variants must be non-empty and unique")
    if any(variant not in TTM_VARIANTS for variant in selected_variants):
        raise ValueError("unknown TTM variant requested")
    if "ttm_pretrained_adapter" in selected_variants:
        if not allow_adapter:
            raise ValueError("adapter requires --allow-adapter")
        prior_gate = Path(run_dir).resolve().parent / "quick" / "quick_gate.json"
        if not prior_gate.is_file() or _json(prior_gate).get("passed") is not True:
            raise ValueError("adapter requires a passed, frozen quick transfer gate")
    selected_seeds = tuple(
        (QUICK_SEEDS if profile == "quick" else FULL_SEEDS)
        if seeds is None
        else map(int, seeds)
    )
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("TTM seeds must be non-empty and unique")
    locked = _json(locked_config_path)
    expected_seeds = tuple(
        locked["optimization"][
            "quick_seeds" if profile == "quick" else "full_seeds"
        ]
    )
    if selected_seeds != expected_seeds:
        raise ValueError(
            f"{profile} seeds are locked to {expected_seeds}; got {selected_seeds}"
        )
    fit_epochs = int(
        epochs
        if epochs is not None
        else locked["optimization"][
            "quick_epochs" if profile == "quick" else "full_epochs"
        ]
    )
    locked_epochs = int(
        locked["optimization"][
            "quick_epochs" if profile == "quick" else "full_epochs"
        ]
    )
    if fit_epochs != locked_epochs:
        raise ValueError(f"{profile} epochs are locked to {locked_epochs}")
    locked_quick_variants = tuple(locked["quick_variants"])
    locked_full_variants = tuple(locked["full_default_variants"])
    if profile == "quick" and selected_variants not in {
        locked_quick_variants,
        (*locked_quick_variants, "ttm_scratch_trainable"),
    }:
        raise ValueError(
            "quick requires the six locked variants, optionally followed by "
            "ttm_scratch_trainable"
        )
    if profile == "full" and selected_variants != locked_full_variants:
        adapter_full = (*locked_full_variants, "ttm_pretrained_adapter")
        if selected_variants != adapter_full or not allow_adapter:
            raise ValueError(
                "full defaults to the two locked variants; an adapter may only "
                "be appended explicitly after a passed quick gate"
            )

    started = time.perf_counter()
    data = Path(data_root).resolve()
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    before_data = capture_data_manifest(data)
    identity = verify_ttm_checkpoint(weight_dir)
    resolved_device = resolve_device(device)
    implementation_hashes = _source_hashes()
    config_hash = sha256_file(locked_config_path)
    train_wells = enumerate_wells(data, "train")
    train_by_id = {well.well_id: well for well in train_wells}
    mapping = load_fold_mapping(fold_config, expected_well_ids=train_by_id)
    selected = (
        _quick_cohort(quick_source, mapping)
        if profile == "quick"
        else _full_cohort(mapping)
    )
    validation_ids = tuple(
        sorted(
            {
                well_id
                for _, validation in selected.values()
                for well_id in validation
            }
        )
    )
    if (profile == "quick" and len(validation_ids) != 100) or (
        profile == "full" and len(validation_ids) != 773
    ):
        raise ValueError(f"{profile} validation cohort size differs")
    required_ids = tuple(
        sorted(
            {
                well_id
                for training, validation in selected.values()
                for well_id in (*training, *validation)
            }
        )
    )
    provider = HeelIncumbentProvider(
        data_root=data,
        fold_config=fold_config,
        artifact_root=incumbent_root,
        cache_root=cache / "incumbent-provider",
    )
    incumbent_manifest = provider.state_manifest()
    state_digests = _fold_state_digest(incumbent_manifest)
    run_config = {
        "schema_version": RUN_SCHEMA,
        "profile": profile,
        "data_root": str(data),
        "fold_config": str(Path(fold_config).resolve()),
        "quick_source": (
            str(Path(quick_source).resolve()) if profile == "quick" else None
        ),
        "incumbent_root": str(Path(incumbent_root).resolve()),
        "incumbent_oof": str(Path(incumbent_oof).resolve()),
        "cache_dir": str(cache),
        "weight_dir": str(Path(weight_dir).resolve()),
        "weight_identity": identity,
        "locked_config": str(Path(locked_config_path).resolve()),
        "locked_config_sha256": config_hash,
        "variants": list(selected_variants),
        "seeds": list(selected_seeds),
        "epochs": fit_epochs,
        "batch_size": int(batch_size),
        "embedding_batch_size": int(embedding_batch_size),
        "num_workers": int(num_workers),
        "device_requested": device,
        "device_resolved": str(resolved_device),
        "offline": bool(offline),
        "bootstrap_samples": int(bootstrap_samples),
        "fixed_grid": fixed_grid_manifest(DEFAULT_GRID),
        "residual_target": target_manifest(DEFAULT_TARGET),
        "cohort": {
            str(fold): {
                "training_well_ids": list(training),
                "validation_well_ids": list(validation),
                "training_validation_intersection": [],
            }
            for fold, (training, validation) in selected.items()
        },
        "data_manifest": before_data,
        "implementation_hashes": implementation_hashes,
        "incumbent_manifest_sha256": _canonical_sha256(incumbent_manifest),
        "environment": environment_manifest(),
        "external_pretrained_weights_used": [identity],
        "architecture_only": False,
        "rogii03_used": False,
        "submission_created": False,
        "submission_uploaded": False,
    }
    root = _safe_run_dir(
        run_dir,
        data_root=data,
        resume=resume,
        run_config=run_config,
    )
    _atomic_json(root / "incumbent_manifest.json", incumbent_manifest)
    prepared = _prepare_sequences(
        train_by_id,
        required_ids,
        cache_root=cache / "external-sequence-ttm",
        grid_config=DEFAULT_GRID,
        workers=num_workers,
    )
    raw_provenance = {
        well_id: raw_well_source_provenance(train_by_id[well_id])
        for well_id in required_ids
    }
    baseline_reference_full = _read_incumbent_reference(incumbent_oof)
    baseline_reference = baseline_reference_full.loc[
        baseline_reference_full["well_id"].astype(str).isin(validation_ids)
    ].copy()
    del baseline_reference_full
    with forbid_network(offline):
        pretrained_model = load_ttm_forecaster(
            weight_dir,
            pretrained=True,
            device=resolved_device,
        )
        random_model = load_ttm_forecaster(
            weight_dir,
            pretrained=False,
            device=resolved_device,
            random_seed=TTM_RANDOM_BACKBONE_SEED,
        )
    pretrained_backbone = freeze_backbone(pretrained_model)
    random_backbone = freeze_backbone(random_model)
    model_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    maximum_incumbent_delta = 0.0
    baseline_by_well = {
        str(well_id): group.sort_values("row_index", kind="stable")
        for well_id, group in baseline_reference.groupby("well_id", sort=False)
    }
    for fold in range(FOLD_COUNT):
        fold_started = time.perf_counter()
        training_ids, fold_validation_ids = selected[fold]
        print(
            f"TTM fold={fold} train={len(training_ids)} "
            f"validation={len(fold_validation_ids)}",
            flush=True,
        )
        fold_inputs: dict[str, SequenceModelInput] = {}
        for index, well_id in enumerate(
            (*training_ids, *fold_validation_ids),
            start=1,
        ):
            sequence = prepared[well_id]
            incumbent = _cached_incumbent(
                provider,
                run_root=root,
                well_id=well_id,
                fold=fold,
                expected_rows=sequence.row_index,
                state_digest=state_digests[fold],
                cache_provenance={
                    "raw_source_provenance": raw_provenance[well_id],
                    "raw_source_provenance_sha256": raw_source_provenance_digest(
                        raw_provenance[well_id]
                    ),
                    "data_manifest": before_data,
                    "implementation_hashes": implementation_hashes,
                    "incumbent_manifest_sha256": run_config[
                        "incumbent_manifest_sha256"
                    ],
                    "fixed_grid": run_config["fixed_grid"],
                },
                resume=resume,
            )
            fold_inputs[well_id] = build_sequence_model_input(
                sequence,
                incumbent.prediction,
                grid_points=512,
            )
            if well_id in fold_validation_ids:
                reference = baseline_by_well[well_id]
                if not np.array_equal(
                    sequence.row_index,
                    reference["row_index"].to_numpy(dtype=np.int64),
                ):
                    raise ValueError(f"TTM baseline keys differ: {well_id}")
                maximum_incumbent_delta = max(
                    maximum_incumbent_delta,
                    float(
                        np.max(
                            np.abs(
                                incumbent.prediction
                                - reference[INCUMBENT_COLUMN].to_numpy(
                                    dtype=np.float64
                                )
                            )
                        )
                    ),
                )
            if index % 25 == 0:
                print(
                    f"TTM fold={fold} incumbent={index}/"
                    f"{len(training_ids) + len(fold_validation_ids)}",
                    flush=True,
                )
        if maximum_incumbent_delta > 1.0e-9:
            raise ValueError(
                f"TTM incumbent differs from audited OOF: {maximum_incumbent_delta}"
            )
        training_truth = _load_fold_training_targets(
            train_by_id,
            prepared,
            training_ids=training_ids,
            validation_ids=fold_validation_ids,
        )
        targets = np.stack(
            [
                make_grid_residual_target(
                    ResidualTrainingWell(
                        inference=fold_inputs[well_id],
                        incumbent_prediction=fold_inputs[
                            well_id
                        ].incumbent_prediction,
                        target=training_truth[well_id],
                    ),
                    config=DEFAULT_TARGET,
                ).correction
                for well_id in training_ids
            ]
        ).astype(np.float32)
        del training_truth
        row_counts = np.asarray(
            [prepared[well_id].prediction_rows for well_id in training_ids],
            dtype=np.int64,
        )
        control_seed = 20260729 + fold
        for variant in selected_variants:
            variant_started = time.perf_counter()
            control = VARIANT_CONTROL[variant]
            controlled_training = apply_ttm_control(
                [fold_inputs[well_id] for well_id in training_ids],
                control_id=control,
                seed=control_seed,
            )
            controlled_validation = apply_ttm_control(
                [fold_inputs[well_id] for well_id in fold_validation_ids],
                control_id=control,
                seed=control_seed,
            )
            train_batch = make_ttm_batch(
                controlled_training,
                ids=training_ids,
            )
            validation_batch = make_ttm_batch(
                controlled_validation,
                ids=fold_validation_ids,
            )
            if variant == "ttm_scratch_trainable":
                for seed in selected_seeds:
                    checkpoint_path = (
                        root
                        / "checkpoints"
                        / variant
                        / f"seed_{seed}"
                        / f"fold_{fold}.pt"
                    )
                    head_config = TTMHeadConfig(
                        variant_id=variant,
                        seed=seed,
                        epochs=fit_epochs,
                        batch_size=batch_size,
                        hidden_dim=int(
                            locked["architecture"]["head_hidden_dim"]
                        ),
                        learning_rate=float(
                            locked["optimization"]["learning_rate"]
                        ),
                        weight_decay=float(
                            locked["optimization"]["weight_decay"]
                        ),
                        gradient_clip=float(
                            locked["optimization"]["gradient_clip"]
                        ),
                    )
                    with forbid_network(offline):
                        scratch_forecaster = load_ttm_forecaster(
                            weight_dir,
                            pretrained=False,
                            device=resolved_device,
                            random_seed=TTM_RANDOM_BACKBONE_SEED,
                        )
                    provenance = {
                        "profile": profile,
                        "variant_id": variant,
                        "control_id": control,
                        "eligible_for_model_selection": False,
                        "fold": fold,
                        "seed": seed,
                        "backbone_identity": (
                            f"official_random_trainable:{TTM_RANDOM_BACKBONE_SEED}"
                        ),
                        "locked_config_sha256": config_hash,
                        "data_manifest": before_data,
                        "implementation_hashes": implementation_hashes,
                        "targets_sha256": _array_sha256(targets),
                        "row_counts_sha256": _array_sha256(row_counts),
                        "labels_loaded_only_for_outer_training_ids": True,
                    }
                    model_started = time.perf_counter()
                    checkpoint = fit_ttm_scratch_model(
                        forecaster=scratch_forecaster,
                        train_batch=train_batch,
                        targets=targets,
                        row_counts=row_counts,
                        validation_ids=fold_validation_ids,
                        checkpoint_path=checkpoint_path,
                        config=head_config,
                        provenance=provenance,
                        device=resolved_device,
                        resume=resume,
                        embedding_batch_size=embedding_batch_size,
                    )
                    with forbid_network(offline):
                        prediction_forecaster = load_ttm_forecaster(
                            weight_dir,
                            pretrained=False,
                            device=resolved_device,
                            random_seed=TTM_RANDOM_BACKBONE_SEED,
                        )
                    grid_predictions = predict_ttm_scratch_model(
                        forecaster=prediction_forecaster,
                        batch=validation_batch,
                        checkpoint=checkpoint,
                        device=resolved_device,
                        batch_size=batch_size,
                    )
                    block = pd.concat(
                        [
                            _prediction_block(
                                inference=controlled_validation[well_id],
                                fold=fold,
                                grid_prediction=grid_predictions[index],
                                target_config=DEFAULT_TARGET,
                            )
                            for index, well_id in enumerate(fold_validation_ids)
                        ],
                        ignore_index=True,
                    )
                    block_path = (
                        root
                        / "prediction_blocks"
                        / variant
                        / f"seed_{seed}"
                        / f"fold_{fold}.csv.gz"
                    )
                    _atomic_csv(block_path, block)
                    model_records.append(
                        {
                            "variant_id": variant,
                            "control_id": control,
                            "eligible_for_model_selection": False,
                            "fold": fold,
                            "seed": seed,
                            "training_wells": len(training_ids),
                            "validation_wells": len(fold_validation_ids),
                            "training_validation_intersection": [],
                            "checkpoint": str(checkpoint_path),
                            "checkpoint_sha256": sha256_file(checkpoint_path),
                            "checkpoint_epoch": int(checkpoint["epoch"]),
                            "checkpoint_completed": bool(
                                checkpoint["completed"]
                            ),
                            "prediction_block": str(block_path),
                            "prediction_block_sha256": sha256_file(block_path),
                            "trainable_parameter_count": int(
                                checkpoint["trainable_parameter_count"]
                            ),
                            "backbone_parameters": TTM_BACKBONE_PARAMETER_COUNT,
                            "backbone_frozen": False,
                            "external_pretrained": False,
                            "embedding_cache_resumed": False,
                            "elapsed_seconds": (
                                time.perf_counter() - model_started
                            ),
                        }
                    )
                continue
            pretrained = variant in PRETRAINED_VARIANTS
            backbone = pretrained_backbone if pretrained else random_backbone
            backbone_identity = (
                f"pretrained:{identity['weight_sha256']}"
                if pretrained
                else f"official_random:{TTM_RANDOM_BACKBONE_SEED}"
            )
            train_h, train_t, train_cache_resumed = _embedding_cache(
                root
                / "embedding_cache"
                / f"fold_{fold}"
                / variant
                / "train.npz",
                backbone=backbone,
                batch=train_batch,
                device=resolved_device,
                batch_size=embedding_batch_size,
                backbone_identity=backbone_identity,
                resume=resume,
            )
            validation_h, validation_t, validation_cache_resumed = _embedding_cache(
                root
                / "embedding_cache"
                / f"fold_{fold}"
                / variant
                / "validation.npz",
                backbone=backbone,
                batch=validation_batch,
                device=resolved_device,
                batch_size=embedding_batch_size,
                backbone_identity=backbone_identity,
                resume=resume,
            )
            for seed in selected_seeds:
                checkpoint_path = (
                    root
                    / "checkpoints"
                    / variant
                    / f"seed_{seed}"
                    / f"fold_{fold}.pt"
                )
                head_config = TTMHeadConfig(
                    variant_id=variant,
                    seed=seed,
                    epochs=fit_epochs,
                    batch_size=batch_size,
                    hidden_dim=int(locked["architecture"]["head_hidden_dim"]),
                    learning_rate=float(
                        locked["optimization"]["learning_rate"]
                    ),
                    weight_decay=float(locked["optimization"]["weight_decay"]),
                    gradient_clip=float(
                        locked["optimization"]["gradient_clip"]
                    ),
                    adapter_rank=(
                        int(locked["architecture"]["adapter_rank"])
                        if variant == "ttm_pretrained_adapter"
                        else 0
                    ),
                )
                provenance = {
                    "profile": profile,
                    "variant_id": variant,
                    "control_id": control,
                    "eligible_for_model_selection": ELIGIBLE_FOR_MODEL_SELECTION[
                        variant
                    ],
                    "fold": fold,
                    "seed": seed,
                    "backbone_identity": backbone_identity,
                    "weight_identity": identity if pretrained else None,
                    "random_backbone_seed": (
                        None if pretrained else TTM_RANDOM_BACKBONE_SEED
                    ),
                    "locked_config_sha256": config_hash,
                    "data_manifest": before_data,
                    "implementation_hashes": implementation_hashes,
                    "targets_sha256": _array_sha256(targets),
                    "row_counts_sha256": _array_sha256(row_counts),
                    "train_embeddings_sha256": _canonical_sha256(
                        {
                            "horizontal": _array_sha256(train_h),
                            "typewell": _array_sha256(train_t),
                        }
                    ),
                    "validation_embeddings_sha256": _canonical_sha256(
                        {
                            "horizontal": _array_sha256(validation_h),
                            "typewell": _array_sha256(validation_t),
                        }
                    ),
                    "paired_typewell_ids_train": list(
                        train_batch.paired_typewell_ids
                    ),
                    "paired_typewell_ids_validation": list(
                        validation_batch.paired_typewell_ids
                    ),
                    "labels_loaded_only_for_outer_training_ids": True,
                }
                model_started = time.perf_counter()
                checkpoint = fit_ttm_head(
                    train_horizontal=train_h,
                    train_typewell=train_t,
                    train_batch=train_batch,
                    targets=targets,
                    row_counts=row_counts,
                    validation_ids=fold_validation_ids,
                    checkpoint_path=checkpoint_path,
                    config=head_config,
                    provenance=provenance,
                    device=resolved_device,
                    resume=resume,
                )
                if checkpoint.get("completed") is not True:
                    raise ValueError(f"TTM checkpoint is incomplete: {checkpoint_path}")
                grid_predictions = predict_ttm_head(
                    horizontal=validation_h,
                    typewell=validation_t,
                    batch=validation_batch,
                    checkpoint=checkpoint,
                    device=resolved_device,
                    batch_size=batch_size,
                )
                block = pd.concat(
                    [
                        _prediction_block(
                            inference=controlled_validation[well_id],
                            fold=fold,
                            grid_prediction=grid_predictions[index],
                            target_config=DEFAULT_TARGET,
                        )
                        for index, well_id in enumerate(fold_validation_ids)
                    ],
                    ignore_index=True,
                )
                block_path = (
                    root
                    / "prediction_blocks"
                    / variant
                    / f"seed_{seed}"
                    / f"fold_{fold}.csv.gz"
                )
                _atomic_csv(block_path, block)
                model_records.append(
                    {
                        "variant_id": variant,
                        "control_id": control,
                        "eligible_for_model_selection": ELIGIBLE_FOR_MODEL_SELECTION[
                            variant
                        ],
                        "fold": fold,
                        "seed": seed,
                        "training_wells": len(training_ids),
                        "validation_wells": len(fold_validation_ids),
                        "training_validation_intersection": [],
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "checkpoint_epoch": int(checkpoint["epoch"]),
                        "checkpoint_completed": bool(checkpoint["completed"]),
                        "prediction_block": str(block_path),
                        "prediction_block_sha256": sha256_file(block_path),
                        "trainable_parameter_count": int(
                            checkpoint["trainable_parameter_count"]
                        ),
                        "backbone_parameters": TTM_BACKBONE_PARAMETER_COUNT,
                        "backbone_frozen": True,
                        "external_pretrained": pretrained,
                        "embedding_cache_resumed": bool(
                            train_cache_resumed and validation_cache_resumed
                        ),
                        "elapsed_seconds": time.perf_counter() - model_started,
                    }
                )
                print(
                    f"TTM scored variant={variant} fold={fold} seed={seed} "
                    f"seconds={time.perf_counter() - model_started:.2f}",
                    flush=True,
                )
            random_counts = {
                record["trainable_parameter_count"]
                for record in model_records
                if record["fold"] == fold
                and record["variant_id"] == "ttm_random_frozen"
            }
            pretrained_counts = {
                record["trainable_parameter_count"]
                for record in model_records
                if record["fold"] == fold
                and record["variant_id"] == "ttm_pretrained_frozen"
            }
            if pretrained_counts and random_counts != pretrained_counts:
                raise ValueError("matched random/pretrained head counts differ")
            print(
                f"TTM completed variant={variant} fold={fold} "
                f"seconds={time.perf_counter() - variant_started:.2f}",
                flush=True,
            )
        fold_records.append(
            {
                "fold": fold,
                "training_wells": len(training_ids),
                "validation_wells": len(fold_validation_ids),
                "validation_rows": int(
                    sum(prepared[well_id].prediction_rows for well_id in fold_validation_ids)
                ),
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
        )
        _atomic_json(
            root / "progress.json",
            {
                "completed_models": len(model_records),
                "models": model_records,
                "folds": fold_records,
            },
        )
    del baseline_by_well, baseline_reference
    baseline_full = _read_incumbent_oof(incumbent_oof)
    baseline = baseline_full.loc[
        baseline_full["well_id"].astype(str).isin(validation_ids)
    ].copy()
    del baseline_full
    oof = _assemble_oof(
        baseline,
        run_root=root,
        variants=selected_variants,
        seeds=selected_seeds,
    )
    oof_path = root / "oof_predictions.csv.gz"
    _atomic_csv(oof_path, oof)
    metrics = evaluate_ttm_oof(
        oof,
        variants=selected_variants,
        seeds=selected_seeds,
        bootstrap_samples=bootstrap_samples,
    )
    _write_metrics(root, metrics)
    after_data = capture_data_manifest(data)
    if after_data != before_data:
        raise RuntimeError("competition data manifest changed during TTM experiment")
    online_smoke = root.parent / "smoke-online" / "smoke_summary.json"
    offline_smoke = root.parent / "smoke-offline" / "smoke_summary.json"
    safety_checks = {
        "finite_exact_oof_keys": metrics["validation"]["nonfinite_numeric_values"]
        == 0
        and metrics["validation"]["duplicate_well_row_keys"] == 0,
        "fold_safety": all(
            not record["training_validation_intersection"]
            for record in model_records
        ),
        "checkpoint_complete": len(model_records)
        == FOLD_COUNT * len(selected_variants) * len(selected_seeds)
        and all(record["checkpoint_completed"] for record in model_records),
        "matched_trainable_parameter_count": len(
            {
                record["trainable_parameter_count"]
                for record in model_records
                if record["variant_id"]
                in {"ttm_random_frozen", "ttm_pretrained_frozen"}
            }
        )
        == 1,
        "weight_identity": identity["weight_sha256"]
        == locked["identity"]["weight_sha256"],
        "offline_execution": bool(offline),
        "online_smoke": online_smoke.is_file()
        and _json(online_smoke).get("passed") is True,
        "offline_smoke": offline_smoke.is_file()
        and _json(offline_smoke).get("passed") is True
        and _json(offline_smoke).get("network_forbidden") is True,
        "data_manifest": before_data == after_data,
        "no_submission": not any(
            "submission" in path.name.lower() and path.suffix.lower() == ".csv"
            for path in root.rglob("*")
            if path.is_file()
        ),
    }
    gate = None
    if profile == "quick":
        gate = locked_quick_gate(
            metrics,
            gate_config=locked["quick_gate"],
            safety_checks=safety_checks,
        )
        _atomic_json(root / "quick_gate.json", gate)
        if gate["passed"]:
            frozen = {
                "locked_config": locked,
                "locked_config_sha256": config_hash,
                "run_config_sha256": sha256_file(root / "run_config.json"),
                "decision": gate["decision"],
            }
            _atomic_json(root / "frozen_full_candidate.json", frozen)
    artifact_audit = {
        "schema_version": "ttm_artifact_audit_v1",
        "passed": all(safety_checks.values()),
        "oof_path": str(oof_path),
        "oof_sha256": sha256_file(oof_path),
        "oof_validation": metrics["validation"],
        "model_count": len(model_records),
        "models": model_records,
        "maximum_incumbent_oof_delta": maximum_incumbent_delta,
        "weight_identity": identity,
        "data_manifest_before": before_data,
        "data_manifest_after": after_data,
        "safety_checks": safety_checks,
        "submission_created": False,
        "submission_uploaded": False,
    }
    _atomic_json(root / "artifact_audit.json", artifact_audit)
    _atomic_json(
        root / "model_manifest.json",
        {
            "models": model_records,
            "matched_random_pretrained_trainable_parameters_equal": safety_checks[
                "matched_trainable_parameter_count"
            ],
            "identity_fallbacks_scored": False,
        },
    )
    summary = {
        "schema_version": "ttm_transfer_summary_v1",
        "profile": profile,
        "scope": {
            "wells": int(oof["well_id"].nunique()),
            "rows": int(len(oof)),
            "folds": list(range(FOLD_COUNT)),
            "seeds": list(selected_seeds),
            "variants": list(selected_variants),
        },
        "metrics": metrics["summary"],
        "quick_gate": gate,
        "artifact_audit_passed": artifact_audit["passed"],
        "data_manifest_equal": True,
        "submission_created": False,
        "submission_uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


__all__ = [
    "FULL_SEEDS",
    "FULL_VARIANTS",
    "LOCKED_CONFIG",
    "QUICK_SEEDS",
    "QUICK_VARIANTS",
    "run_ttm_smoke",
    "run_ttm_transfer_experiment",
]
