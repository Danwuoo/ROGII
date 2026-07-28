"""Strict matched screening for external sequence-model residual families.

The experiment deliberately separates four scopes:

* raw fixed-grid preparation reads inference-safe columns only;
* every outer fold receives the exact incumbent state trained without that fold;
* labels are loaded only for that outer fold's training wells;
* row-level OOF predictions are scored only after exact key/coverage validation.

The small PyTorch implementations are architecture-only probes.  A failed family
or incomplete checkpoint is recorded as ineligible and never silently replaced
by a scored identity prediction.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..data import WellFiles, enumerate_wells, validate_prefix
from ..external_models.components import validate_residual_components
from ..external_models.contracts import ResidualTrainingWell, SequenceModelInput
from ..external_models.controls import (
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
from ..external_models.data import (
    MODEL_INPUT_CHANNEL_NAMES,
    FixedGridConfig,
    FixedGridSequenceWell,
    build_sequence_model_input,
    fixed_grid_manifest,
    prepare_fixed_grid_well,
    raw_source_provenance_digest,
    raw_well_source_provenance,
)
from ..external_models.incumbent import (
    DEFAULT_INCUMBENT_SEEDS,
    HeelIncumbentProvider,
    IncumbentPrediction,
)
from ..external_models.targets import (
    SmoothTargetConfig,
    grid_prediction_to_row_components,
    make_grid_residual_target,
    target_manifest,
)
from ..external_models.torch_families import (
    FAMILY_FNO1D,
    FAMILY_PAIRED_PATCH,
    FAMILY_S4D,
    FAMILY_TCN,
    TorchFamilyConfig,
    TorchFamilyState,
    fit_torch_family,
    load_torch_family,
)
from ..folds import FOLD_COUNT, load_fold_mapping, partition_well_ids
from ..portfolio_metrics import (
    common_slope_shape_metrics,
    paired_well_cluster_bootstrap,
    per_fold_metrics,
    per_well_metrics,
    per_well_win_summary,
    pooled_metrics,
    residual_correlation,
    shadow_private_subset_simulation,
    tail_metrics,
    validate_oof_frame,
)


INCUMBENT_COLUMN = "y_pred_incumbent"
SOURCE_INCUMBENT_COLUMN = "y_pred_heel_registered_ensemble"
DEFAULT_QUICK_SOURCE = Path(".runs/sparse-anchor-student-quick/run_config.json")
DEFAULT_INCUMBENT_ROOT = Path(".runs/heel-registered-anchor-full-20260728")
DEFAULT_FULL_OOF = DEFAULT_INCUMBENT_ROOT / "oof_predictions.csv.gz"
DEFAULT_GRID = FixedGridConfig(horizontal_points=256, typewell_points=256)
DEFAULT_TARGET = SmoothTargetConfig(
    retained_dct_modes=16,
    maximum_absolute_correction=80.0,
)
DEFAULT_QUICK_SEEDS = (20260728,)
DEFAULT_FULL_SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)
DEFAULT_SHRINKAGE = 0.25
RUN_CONFIG_SCHEMA_VERSION = "external_model_portfolio_run_v2"
CHECKPOINT_PROVENANCE_SCHEMA_VERSION = "external_model_checkpoint_v2"
INCUMBENT_CACHE_SCHEMA_VERSION = "external_incumbent_prediction_cache_v2"

TYPEWELL_MODEL_CHANNELS = (
    "typewell_gr_at_incumbent",
    "typewell_support",
)
HORIZONTAL_MODEL_CHANNELS = tuple(
    name for name in MODEL_INPUT_CHANNEL_NAMES if name not in TYPEWELL_MODEL_CHANNELS
)


@dataclass(frozen=True)
class PortfolioVariant:
    """One architecture/control combination with a stable artifact identity."""

    variant_id: str
    family: str
    control_id: str = CONTROL_NONE
    cross_series: bool = True
    frozen_random_encoder: bool = False

    def __post_init__(self) -> None:
        variant = str(self.variant_id)
        family = str(self.family)
        control = str(self.control_id)
        if not variant or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in variant
        ):
            raise ValueError("variant_id must be non-empty lowercase snake_case")
        if family not in {
            FAMILY_PAIRED_PATCH,
            FAMILY_S4D,
            FAMILY_FNO1D,
            FAMILY_TCN,
        }:
            raise ValueError(f"unsupported family: {family}")
        if control not in {
            CONTROL_NONE,
            CONTROL_WRONG_TYPEWELL,
            CONTROL_PAIR_SHUFFLE,
            CONTROL_GR_ORDER_SHUFFLE,
            CONTROL_PHASE_DESTROY,
            CONTROL_NO_CROSS,
        }:
            raise ValueError(f"unsupported control: {control}")
        if control == CONTROL_NO_CROSS and self.cross_series:
            raise ValueError("no-cross control must disable cross_series")
        if control != CONTROL_NO_CROSS and not self.cross_series:
            raise ValueError("only the no-cross control may disable cross_series")
        if self.frozen_random_encoder and control != CONTROL_NONE:
            raise ValueError("random frozen encoder is a separate matched control")
        object.__setattr__(self, "variant_id", variant)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "control_id", control)


DEFAULT_QUICK_VARIANTS = (
    PortfolioVariant("paired_patch_cross", FAMILY_PAIRED_PATCH),
    PortfolioVariant("s4d", FAMILY_S4D),
    PortfolioVariant("fno1d", FAMILY_FNO1D),
    PortfolioVariant("tcn", FAMILY_TCN),
    PortfolioVariant(
        "paired_patch_no_cross",
        FAMILY_PAIRED_PATCH,
        control_id=CONTROL_NO_CROSS,
        cross_series=False,
    ),
    PortfolioVariant(
        "paired_patch_wrong_typewell",
        FAMILY_PAIRED_PATCH,
        control_id=CONTROL_WRONG_TYPEWELL,
    ),
    PortfolioVariant(
        "paired_patch_pair_shuffle",
        FAMILY_PAIRED_PATCH,
        control_id=CONTROL_PAIR_SHUFFLE,
    ),
    PortfolioVariant(
        "paired_patch_gr_order_shuffle",
        FAMILY_PAIRED_PATCH,
        control_id=CONTROL_GR_ORDER_SHUFFLE,
    ),
    PortfolioVariant(
        "paired_patch_phase_destroy",
        FAMILY_PAIRED_PATCH,
        control_id=CONTROL_PHASE_DESTROY,
    ),
    PortfolioVariant(
        "paired_patch_frozen_random",
        FAMILY_PAIRED_PATCH,
        frozen_random_encoder=True,
    ),
    PortfolioVariant(
        "tcn_wrong_typewell",
        FAMILY_TCN,
        control_id=CONTROL_WRONG_TYPEWELL,
    ),
    PortfolioVariant(
        "tcn_pair_shuffle",
        FAMILY_TCN,
        control_id=CONTROL_PAIR_SHUFFLE,
    ),
    PortfolioVariant(
        "tcn_gr_order_shuffle",
        FAMILY_TCN,
        control_id=CONTROL_GR_ORDER_SHUFFLE,
    ),
    PortfolioVariant(
        "tcn_phase_destroy",
        FAMILY_TCN,
        control_id=CONTROL_PHASE_DESTROY,
    ),
    PortfolioVariant(
        "tcn_frozen_random",
        FAMILY_TCN,
        frozen_random_encoder=True,
    ),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if destination.suffix == ".gz":
        frame.to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(ids: Iterable[str]) -> str:
    values = tuple(map(str, ids))
    encoded = json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _implementation_hashes() -> dict[str, str]:
    """Bind resume/checkpoint reuse to every implementation layer it executes."""

    objects = {
        "portfolio_runner": run_external_model_portfolio_experiment,
        "fixed_grid": prepare_fixed_grid_well,
        "model_input": build_sequence_model_input,
        "controls": apply_individual_control,
        "residual_target": make_grid_residual_target,
        "torch_family": fit_torch_family,
        "incumbent_provider": HeelIncumbentProvider,
    }
    result: dict[str, str] = {}
    for name, value in objects.items():
        source = inspect.getsourcefile(value)
        if source is None or not Path(source).is_file():
            raise ValueError(f"cannot resolve implementation source for {name}")
        result[name] = _sha256(Path(source))
    return result


def _data_manifest(root: Path) -> dict[str, Any]:
    """Hash names, sizes, and mtimes without rereading 1.3 GB of inputs."""

    rows: list[str] = []
    total_bytes = 0
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append(
            f"{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}"
        )
        total_bytes += int(stat.st_size)
    return {
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "name_size_mtime_sha256": hashlib.sha256(
            "\n".join(rows).encode("utf-8")
        ).hexdigest(),
    }


def _safe_run_root(
    output_root: Path,
    run_id: str,
    data_root: Path,
    *,
    resume: bool,
    run_config: Mapping[str, Any],
) -> Path:
    run = (Path(output_root).resolve() / str(run_id)).resolve()
    data = Path(data_root).resolve()
    if run == data or data in run.parents:
        raise ValueError("portfolio output cannot be inside competition data")
    config_path = run / "run_config.json"
    if run.exists():
        if not resume:
            raise FileExistsError(f"run already exists: {run}")
        if (run / "summary.json").is_file():
            raise ValueError("completed run cannot be resumed")
        if not config_path.is_file():
            raise ValueError("resume run is missing immutable run_config.json")
        if _read_json(config_path) != _jsonable(run_config):
            raise ValueError("resume run_config differs; use a new run ID")
    else:
        run.mkdir(parents=True)
        _atomic_json(config_path, run_config)
    (run / "models").mkdir(exist_ok=True)
    (run / "incumbent_cache").mkdir(exist_ok=True)
    (run / "metrics").mkdir(exist_ok=True)
    return run


def _target_from_train_file(
    well: WellFiles,
    row_index: np.ndarray,
) -> np.ndarray:
    """Training-only label loader; inference code never calls this function."""

    header = pd.read_csv(well.horizontal_path, nrows=0)
    if "TVT" not in header.columns:
        raise ValueError(f"training target missing for {well.well_id}")
    frame = pd.read_csv(well.horizontal_path, usecols=["TVT", "TVT_input"])
    known_count, prediction_count = validate_prefix(
        frame["TVT_input"],
        target=frame["TVT"],
    )
    expected = np.arange(
        known_count,
        known_count + prediction_count,
        dtype=np.int64,
    )
    if not np.array_equal(expected, np.asarray(row_index, dtype=np.int64)):
        raise ValueError(f"training target row alignment differs: {well.well_id}")
    target = frame["TVT"].iloc[known_count:].to_numpy(dtype=np.float64)
    if target.shape != expected.shape or not np.isfinite(target).all():
        raise ValueError(f"invalid target: {well.well_id}")
    return target


def _read_incumbent_oof(path: Path) -> pd.DataFrame:
    required = [
        "well_id",
        "row_index",
        "fold",
        "y_true",
        SOURCE_INCUMBENT_COLUMN,
    ]
    frame = pd.read_csv(
        path,
        usecols=required,
        dtype={"well_id": "string", "row_index": "int64", "fold": "int8"},
    ).rename(columns={SOURCE_INCUMBENT_COLUMN: INCUMBENT_COLUMN})
    validate_oof_frame(frame, (INCUMBENT_COLUMN,))
    return frame.sort_values(
        ["fold", "well_id", "row_index"],
        kind="stable",
    ).reset_index(drop=True)


def _read_incumbent_reference(path: Path) -> pd.DataFrame:
    """Read target-free incumbent keys/predictions for pre-fit consistency checks."""

    frame = pd.read_csv(
        path,
        usecols=["well_id", "row_index", "fold", SOURCE_INCUMBENT_COLUMN],
        dtype={"well_id": "string", "row_index": "int64", "fold": "int8"},
    ).rename(columns={SOURCE_INCUMBENT_COLUMN: INCUMBENT_COLUMN})
    if (
        frame.empty
        or frame[["well_id", "row_index", "fold"]].isna().any().any()
        or frame.duplicated(["well_id", "row_index"]).any()
        or not np.isfinite(frame[INCUMBENT_COLUMN].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("target-free incumbent reference is invalid")
    return frame.sort_values(
        ["fold", "well_id", "row_index"],
        kind="stable",
    ).reset_index(drop=True)


def _load_fold_training_targets(
    train_by_id: Mapping[str, WellFiles],
    prepared: Mapping[str, FixedGridSequenceWell],
    *,
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Load labels only for the current outer fold's authorized training wells."""

    training = tuple(map(str, training_ids))
    validation = tuple(map(str, validation_ids))
    overlap = sorted(set(training) & set(validation))
    if overlap:
        raise ValueError(f"outer-fold train/validation overlap: {overlap}")
    missing = sorted(set(training) - set(train_by_id) | (set(training) - set(prepared)))
    if missing:
        raise KeyError(f"training wells are unavailable: {missing}")
    return {
        well_id: _target_from_train_file(
            train_by_id[well_id],
            prepared[well_id].row_index,
        )
        for well_id in training
    }


def _quick_cohort(
    path: Path,
    mapping: pd.DataFrame,
) -> dict[int, tuple[tuple[str, ...], tuple[str, ...]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("profile") != "quick":
        raise ValueError("quick cohort source is not a quick run")
    provenance = payload.get("fold_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("quick cohort source lacks fold_provenance")
    selected: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for fold in range(FOLD_COUNT):
        record = provenance.get(str(fold))
        if not isinstance(record, dict):
            raise ValueError(f"quick cohort lacks fold {fold}")
        fit = record.get("student_fit_provenance", {})
        training = fit.get("extra_trees_s20260711")
        if training is None:
            raise ValueError(f"quick cohort lacks primary fit IDs for fold {fold}")
        training_ids = tuple(sorted(map(str, training)))
        validation_ids = tuple(sorted(map(str, record.get("validation_wells", ()))))
        outer_training, outer_validation = partition_well_ids(mapping, fold)
        if (
            len(training_ids) != 100
            or len(validation_ids) != 20
            or not set(training_ids).issubset(outer_training)
            or not set(validation_ids).issubset(outer_validation)
            or set(training_ids) & set(validation_ids)
        ):
            raise ValueError(f"quick cohort provenance mismatch for fold {fold}")
        selected[fold] = training_ids, validation_ids
    return selected


def _full_cohort(
    mapping: pd.DataFrame,
) -> dict[int, tuple[tuple[str, ...], tuple[str, ...]]]:
    return {fold: partition_well_ids(mapping, fold) for fold in range(FOLD_COUNT)}


def _prepare_sequences(
    train_by_id: Mapping[str, WellFiles],
    required_ids: Sequence[str],
    *,
    cache_root: Path,
    grid_config: FixedGridConfig,
    workers: int,
) -> dict[str, FixedGridSequenceWell]:
    def prepare(well_id: str) -> FixedGridSequenceWell:
        return prepare_fixed_grid_well(
            train_by_id[well_id],
            split="train",
            cache_root=cache_root,
            config=grid_config,
        )

    values: dict[str, FixedGridSequenceWell] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(prepare, well_id): well_id for well_id in required_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            well_id = futures[future]
            values[well_id] = future.result()
            if completed % 50 == 0 or completed == len(required_ids):
                print(
                    f"prepared sequence wells={completed}/{len(required_ids)}",
                    flush=True,
                )
    return values


def _fold_state_digest(
    incumbent_manifest: Mapping[str, Any],
) -> dict[int, str]:
    result: dict[int, str] = {}
    records = incumbent_manifest.get("states")
    if not isinstance(records, list):
        raise ValueError("incumbent manifest lacks state records")
    for fold in range(FOLD_COUNT):
        hashes = tuple(
            sorted(
                str(record["sha256"])
                for record in records
                if int(record["fold"]) == fold
            )
        )
        if len(hashes) != len(DEFAULT_INCUMBENT_SEEDS):
            raise ValueError(f"incumbent state count differs for fold {fold}")
        result[fold] = hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
    return result


def _cached_incumbent(
    provider: HeelIncumbentProvider,
    *,
    run_root: Path,
    well_id: str,
    fold: int,
    expected_rows: np.ndarray,
    state_digest: str,
    cache_provenance: Mapping[str, Any],
    resume: bool,
    source_root: Path | None = None,
) -> IncumbentPrediction:
    expected_provenance = _jsonable(
        {
            **dict(cache_provenance),
            "schema_version": INCUMBENT_CACHE_SCHEMA_VERSION,
            "well_id": str(well_id),
            "fold": int(fold),
            "state_digest": str(state_digest),
            "provider_seeds": list(map(int, provider.seeds)),
            "expected_rows_sha256": _array_sha256(
                np.asarray(expected_rows, dtype=np.int64)
            ),
        }
    )
    expected_provenance_digest = _canonical_sha256(expected_provenance)
    path = run_root / "incumbent_cache" / f"fold_{fold}" / f"{well_id}.npz"
    candidates: list[Path] = []
    if resume:
        candidates.append(path)
    if source_root is not None:
        source = Path(source_root).resolve() / f"fold_{fold}" / f"{well_id}.npz"
        if source != path:
            candidates.append(source)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with np.load(candidate, allow_pickle=False) as cached:
                rows = np.asarray(cached["row_index"], dtype=np.int64)
                prediction = np.asarray(cached["prediction"], dtype=np.float64)
                digest = str(cached["state_digest"].item())
                cached_well = str(cached["well_id"].item())
                cached_fold = int(cached["fold"].item())
                cached_schema = str(cached["cache_schema_version"].item())
                provenance_json = str(cached["cache_provenance_json"].item())
                provenance_digest = str(cached["cache_provenance_digest"].item())
                rows_digest = str(cached["row_index_sha256"].item())
                prediction_digest = str(cached["prediction_sha256"].item())
                cached_provenance = json.loads(provenance_json)
            if (
                cached_well == well_id
                and cached_fold == fold
                and digest == state_digest
                and cached_schema == INCUMBENT_CACHE_SCHEMA_VERSION
                and cached_provenance == expected_provenance
                and provenance_digest == expected_provenance_digest
                and np.array_equal(rows, expected_rows)
                and rows_digest == _array_sha256(rows)
                and prediction.shape == rows.shape
                and np.isfinite(prediction).all()
                and prediction_digest == _array_sha256(prediction)
            ):
                return IncumbentPrediction(
                    well_id=well_id,
                    row_index=rows,
                    prediction=prediction,
                    fold_states=(fold,),
                    seeds=provider.seeds,
                    state_sha256=(state_digest,),
                )
        except (OSError, ValueError, KeyError, EOFError):
            pass
    result = provider.predict_outer_fold(well_id, outer_fold=fold)
    if not np.array_equal(result.row_index, expected_rows):
        raise ValueError(f"incumbent row keys differ for {well_id}/fold {fold}")
    prediction = np.asarray(result.prediction, dtype=np.float64)
    if prediction.shape != result.row_index.shape or not np.isfinite(prediction).all():
        raise ValueError(f"incumbent prediction is invalid for {well_id}/fold {fold}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            well_id=np.asarray(well_id),
            fold=np.asarray(fold, dtype=np.int64),
            state_digest=np.asarray(state_digest),
            cache_schema_version=np.asarray(INCUMBENT_CACHE_SCHEMA_VERSION),
            cache_provenance_json=np.asarray(
                json.dumps(
                    expected_provenance,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            ),
            cache_provenance_digest=np.asarray(expected_provenance_digest),
            row_index=result.row_index,
            row_index_sha256=np.asarray(_array_sha256(result.row_index)),
            prediction=prediction,
            prediction_sha256=np.asarray(_array_sha256(prediction)),
        )
    os.replace(temporary, path)
    return result


def _apply_control(
    inputs: Sequence[SequenceModelInput],
    variant: PortfolioVariant,
    *,
    seed: int,
) -> dict[str, SequenceModelInput]:
    if variant.control_id in {CONTROL_WRONG_TYPEWELL, CONTROL_PAIR_SHUFFLE}:
        return paired_control_inputs(
            inputs,
            control_id=variant.control_id,
            seed=seed,
        )
    spec = ControlSpec(variant.control_id, seed)
    return {item.well_id: apply_individual_control(item, spec=spec) for item in inputs}


def _torch_config(
    variant: PortfolioVariant,
    *,
    seed: int,
    device: str,
    epochs: int,
    hidden_dim: int,
    layers: int,
) -> TorchFamilyConfig:
    kwargs: dict[str, Any] = {
        "family": variant.family,
        "seed": int(seed),
        "device": str(device),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "layers": int(layers),
        "learning_rate": 3.0e-3,
        "weight_decay": 1.0e-4,
        "gradient_clip": 5.0,
        "frozen_random_encoder": variant.frozen_random_encoder,
    }
    if variant.family == FAMILY_PAIRED_PATCH:
        kwargs.update(
            {
                "patch_sizes": (8, 16, 32),
                "attention_heads": 2,
                "cross_series": variant.cross_series,
                "horizontal_channels": HORIZONTAL_MODEL_CHANNELS,
                "typewell_channels": TYPEWELL_MODEL_CHANNELS,
            }
        )
    elif variant.family == FAMILY_FNO1D:
        kwargs["fno_modes"] = 12
    elif variant.family == FAMILY_TCN:
        kwargs["tcn_kernel_size"] = 3
    return TorchFamilyConfig(**kwargs)


def _load_or_fit_state(
    path: Path,
    *,
    x: np.ndarray,
    targets: np.ndarray,
    channel_names: Sequence[str],
    training_ids: Sequence[str],
    row_counts: np.ndarray,
    config: TorchFamilyConfig,
    provenance: Mapping[str, Any],
    resume: bool,
) -> tuple[TorchFamilyState, bool, dict[str, Any]]:
    required_context = {
        "variant_id",
        "family",
        "control_id",
        "fold",
        "validation_well_ids",
        "data_manifest",
        "implementation_hashes",
        "incumbent",
        "run_config_sha256",
        "fixed_grid",
        "residual_target",
    }
    if not required_context.issubset(provenance):
        missing = sorted(required_context - set(provenance))
        raise ValueError(f"checkpoint provenance context is incomplete: {missing}")
    if (
        str(provenance["family"]) != config.family
        or int(provenance.get("seed", config.seed)) != config.seed
        or int(provenance["fold"]) not in range(FOLD_COUNT)
        or not isinstance(provenance["data_manifest"], Mapping)
        or not isinstance(provenance["implementation_hashes"], Mapping)
        or not isinstance(provenance["incumbent"], Mapping)
    ):
        raise ValueError("checkpoint provenance context conflicts with model config")
    identifiers = tuple(map(str, training_ids))
    names = tuple(map(str, channel_names))
    validation_ids = tuple(map(str, provenance["validation_well_ids"]))
    expected_provenance = _jsonable(
        {
            "schema_version": CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
            "fit_scope": "outer_training",
            "training_well_ids": list(identifiers),
            "training_ids_sha256": _ids_sha256(identifiers),
            "training_validation_intersection": sorted(
                set(identifiers) & set(validation_ids)
            ),
            "config": asdict(config),
            "channel_names": list(names),
            "x_sha256": _array_sha256(x),
            "targets_sha256": _array_sha256(targets),
            "row_counts_sha256": _array_sha256(row_counts),
            "row_count_sum": int(np.sum(row_counts)),
            "sample_count": len(identifiers),
            "sequence_length": int(np.asarray(x).shape[2]),
            "target_dim": int(np.asarray(targets).shape[1]),
            "context": dict(provenance),
            "labels_loaded_only_for_training_ids": True,
            "outer_validation_labels_used_for_fit": False,
        }
    )
    if expected_provenance["training_validation_intersection"]:
        raise ValueError("checkpoint training/validation IDs overlap")

    sidecar = path.with_suffix(".json")
    existing = (path.exists(), sidecar.exists())
    if any(existing):
        if not all(existing):
            raise ValueError(
                "checkpoint is partial; checkpoint and sidecar are both required"
            )
        if not resume:
            raise FileExistsError(f"checkpoint already exists: {path}")
        payload = _read_json(sidecar)
        actual_checkpoint_sha = _sha256(path)
        if (
            payload.get("schema_version") != CHECKPOINT_PROVENANCE_SCHEMA_VERSION
            or payload.get("checkpoint_sha256") != actual_checkpoint_sha
            or payload.get("provenance") != expected_provenance
        ):
            raise ValueError(f"resume checkpoint provenance differs: {path}")
        state = load_torch_family(path)
        if (
            state.config != config
            or state.channel_names != names
            or state.manifest != payload.get("state_manifest")
            or state.manifest.get("fit_ids_sha256")
            != expected_provenance["training_ids_sha256"]
            or int(state.manifest.get("sample_count", -1)) != len(identifiers)
            or int(state.manifest.get("row_count_sum", -1))
            != expected_provenance["row_count_sum"]
            or int(state.manifest.get("sequence_length", -1))
            != expected_provenance["sequence_length"]
            or int(state.manifest.get("target_dim", -1))
            != expected_provenance["target_dim"]
        ):
            raise ValueError(f"resume checkpoint state differs: {path}")
        return (
            state,
            True,
            {
                "checkpoint_sidecar": str(sidecar),
                "checkpoint_sidecar_sha256": _sha256(sidecar),
                "checkpoint_provenance": expected_provenance,
            },
        )

    state = fit_torch_family(
        x,
        targets,
        channel_names=channel_names,
        ids=training_ids,
        row_counts=row_counts,
        config=config,
    )
    state.save(path)
    restored = load_torch_family(path)
    if (
        restored.config != state.config
        or restored.channel_names != state.channel_names
        or restored.manifest != state.manifest
    ):
        raise ValueError(f"checkpoint changed during round trip: {path}")
    sidecar_payload = {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
        "checkpoint_sha256": _sha256(path),
        "provenance": expected_provenance,
        "state_manifest": restored.manifest,
    }
    _atomic_json(sidecar, sidecar_payload)
    if _read_json(sidecar) != _jsonable(sidecar_payload):
        raise ValueError(f"checkpoint sidecar changed during write: {sidecar}")
    return (
        restored,
        False,
        {
            "checkpoint_sidecar": str(sidecar),
            "checkpoint_sidecar_sha256": _sha256(sidecar),
            "checkpoint_provenance": expected_provenance,
        },
    )


def _prediction_block(
    *,
    inference: SequenceModelInput,
    fold: int,
    grid_prediction: np.ndarray,
    target_config: SmoothTargetConfig,
) -> pd.DataFrame:
    components = grid_prediction_to_row_components(
        grid_prediction,
        inference,
        config=target_config,
    )
    validate_residual_components(
        components,
        inference.incumbent_prediction,
        inference.row_coordinate,
    )
    return pd.DataFrame(
        {
            "well_id": inference.well_id,
            "row_index": inference.row_index,
            "fold": int(fold),
            "prediction": components.final,
            "correction_common": components.common,
            "correction_slope": components.slope,
            "correction_shape": components.shape,
        }
    )


def _same_keys(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return (
        np.array_equal(
            left["well_id"].astype(str).to_numpy(),
            right["well_id"].astype(str).to_numpy(),
        )
        and np.array_equal(
            left["row_index"].to_numpy(dtype=np.int64),
            right["row_index"].to_numpy(dtype=np.int64),
        )
        and np.array_equal(
            left["fold"].to_numpy(dtype=np.int64),
            right["fold"].to_numpy(dtype=np.int64),
        )
    )


def _assemble_oof(
    baseline: pd.DataFrame,
    *,
    variants: Sequence[PortfolioVariant],
    seeds: Sequence[int],
    blocks: Mapping[tuple[str, int], Sequence[pd.DataFrame | Path]],
    completed: Mapping[tuple[str, int], set[int]],
    shrinkage: float,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    output = baseline.copy()
    all_prediction_columns = [INCUMBENT_COLUMN]
    family_columns: list[str] = []
    incomplete: list[str] = []
    for variant in variants:
        seed_predictions: list[np.ndarray] = []
        seed_components: dict[str, list[np.ndarray]] = {
            "common": [],
            "slope": [],
            "shape": [],
        }
        eligible = True
        for seed in seeds:
            key = (variant.variant_id, int(seed))
            if completed.get(key, set()) != set(range(FOLD_COUNT)):
                eligible = False
                incomplete.append(f"{variant.variant_id}/seed_{seed}")
                continue
            pieces = [
                pd.read_csv(
                    piece,
                    dtype={
                        "well_id": "string",
                        "row_index": "int64",
                        "fold": "int8",
                    },
                )
                if isinstance(piece, Path)
                else piece
                for piece in blocks[key]
            ]
            current = (
                pd.concat(pieces, ignore_index=True)
                .sort_values(
                    ["fold", "well_id", "row_index"],
                    kind="stable",
                )
                .reset_index(drop=True)
            )
            if len(current) != len(output) or not _same_keys(output, current):
                raise ValueError(
                    f"OOF prediction keys differ: {variant.variant_id}/seed_{seed}"
                )
            prediction_column = f"y_pred__{variant.variant_id}__seed_{seed}"
            output[prediction_column] = current["prediction"].to_numpy(dtype=np.float64)
            all_prediction_columns.append(prediction_column)
            seed_predictions.append(current["prediction"].to_numpy(dtype=np.float64))
            for component in seed_components:
                seed_components[component].append(
                    current[f"correction_{component}"].to_numpy(dtype=np.float64)
                )
        if not eligible or len(seed_predictions) != len(seeds):
            continue
        family_column = f"y_pred__{variant.variant_id}"
        family_prediction = np.mean(np.stack(seed_predictions), axis=0)
        output[family_column] = family_prediction
        shrink_column = f"{family_column}__shrink_{int(round(100 * shrinkage)):03d}"
        incumbent = output[INCUMBENT_COLUMN].to_numpy(dtype=np.float64)
        output[shrink_column] = incumbent + shrinkage * (family_prediction - incumbent)
        for component, values in seed_components.items():
            output[f"correction__{variant.variant_id}__{component}"] = np.mean(
                np.stack(values), axis=0
            )
        all_prediction_columns.extend((family_column, shrink_column))
        family_columns.extend((family_column, shrink_column))
    validate_oof_frame(
        output,
        all_prediction_columns,
        expected_keys=baseline,
        expected_well_ids=sorted(baseline["well_id"].astype(str).unique()),
        expected_folds=range(FOLD_COUNT),
    )
    return output, all_prediction_columns, family_columns, incomplete


def _metric_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _evaluate_oof(
    oof: pd.DataFrame,
    *,
    prediction_columns: Sequence[str],
    family_columns: Sequence[str],
    metrics_root: Path,
    bootstrap_samples: int,
    shadow_samples: int,
    shadow_subset_wells: int,
) -> dict[str, Any]:
    pooled = pooled_metrics(
        oof,
        prediction_columns,
        baseline_column=INCUMBENT_COLUMN,
    )
    per_fold = per_fold_metrics(
        oof,
        prediction_columns,
        baseline_column=INCUMBENT_COLUMN,
    )
    per_well = per_well_metrics(
        oof,
        prediction_columns,
        baseline_column=INCUMBENT_COLUMN,
    )
    tails = tail_metrics(oof, prediction_columns)
    decomposition = common_slope_shape_metrics(oof, prediction_columns)
    _atomic_csv(metrics_root / "pooled.csv", pooled)
    _atomic_csv(metrics_root / "per_fold.csv", per_fold)
    _atomic_csv(metrics_root / "per_well.csv.gz", per_well)
    _atomic_csv(metrics_root / "tail.csv", tails)
    _atomic_csv(metrics_root / "decomposition.csv", decomposition)

    summary: dict[str, Any] = {
        "pooled": _metric_records(pooled),
        "per_fold": _metric_records(per_fold),
        "tail": _metric_records(tails),
        "decomposition": _metric_records(decomposition),
    }
    if family_columns:
        bootstrap = paired_well_cluster_bootstrap(
            oof,
            baseline_column=INCUMBENT_COLUMN,
            candidate_columns=family_columns,
            samples=bootstrap_samples,
            seed=20260728,
        )
        wins = per_well_win_summary(
            oof,
            baseline_column=INCUMBENT_COLUMN,
            candidate_columns=family_columns,
        )
        correlation = residual_correlation(
            oof,
            (INCUMBENT_COLUMN, *family_columns),
            aggregation="row",
        )
        _atomic_json(metrics_root / "paired_bootstrap.json", bootstrap)
        _atomic_csv(metrics_root / "per_well_wins.csv", wins)
        _atomic_csv(
            metrics_root / "residual_correlation.csv",
            correlation.reset_index(names="candidate"),
        )
        summary.update(
            {
                "paired_bootstrap": bootstrap,
                "per_well_wins": _metric_records(wins),
                "residual_correlation": correlation.to_dict(),
            }
        )
        available_wells = int(oof["well_id"].nunique())
        if shadow_samples > 0 and shadow_subset_wells <= available_wells:
            shadow = shadow_private_subset_simulation(
                oof,
                baseline_column=INCUMBENT_COLUMN,
                candidate_columns=family_columns,
                samples=shadow_samples,
                subset_wells=shadow_subset_wells,
                seed=20260728,
            )
            _atomic_json(metrics_root / "shadow_private.json", shadow)
            summary["shadow_private"] = shadow
        else:
            summary["shadow_private"] = {
                "performed": False,
                "reason": (
                    f"subset_wells={shadow_subset_wells} exceeds "
                    f"available_wells={available_wells}"
                    if shadow_samples > 0
                    else "disabled"
                ),
            }
    return summary


def run_external_model_portfolio_experiment(
    *,
    profile: str,
    data_root: Path,
    fold_config: Path,
    output_root: Path,
    run_id: str,
    cache_root: Path,
    incumbent_root: Path = DEFAULT_INCUMBENT_ROOT,
    incumbent_oof: Path = DEFAULT_FULL_OOF,
    quick_source: Path = DEFAULT_QUICK_SOURCE,
    incumbent_cache_source: Path | None = None,
    variants: Sequence[PortfolioVariant] | None = None,
    seeds: Sequence[int] | None = None,
    epochs: int | None = None,
    hidden_dim: int = 16,
    layers: int = 2,
    device: str = "cpu",
    workers: int = 4,
    shrinkage: float = DEFAULT_SHRINKAGE,
    bootstrap_samples: int = 10_000,
    shadow_samples: int = 100_000,
    shadow_subset_wells: int = 200,
    resume: bool = False,
) -> dict[str, Any]:
    """Run matched quick or strict full outer-fold OOF sequence probes."""

    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    data = Path(data_root).resolve()
    shared_cache = Path(cache_root).resolve()
    if shared_cache == data or data in shared_cache.parents:
        raise ValueError("cache_root cannot be inside competition data")
    if workers < 1:
        raise ValueError("workers must be positive")
    if not math.isfinite(shrinkage) or not 0.0 < shrinkage <= 1.0:
        raise ValueError("shrinkage must be in (0, 1]")
    selected_variants = tuple(DEFAULT_QUICK_VARIANTS if variants is None else variants)
    if not selected_variants or len(
        {variant.variant_id for variant in selected_variants}
    ) != len(selected_variants):
        raise ValueError("variants must be non-empty with unique IDs")
    selected_seeds = tuple(
        (DEFAULT_QUICK_SEEDS if profile == "quick" else DEFAULT_FULL_SEEDS)
        if seeds is None
        else map(int, seeds)
    )
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("seeds must be non-empty and unique")
    fit_epochs = int(
        epochs if epochs is not None else (30 if profile == "quick" else 60)
    )
    if fit_epochs < 1:
        raise ValueError("epochs must be positive")

    started = time.perf_counter()
    before_data = _data_manifest(data)
    torch.set_num_threads(int(workers))

    train_files = enumerate_wells(data, "train")
    train_by_id = {well.well_id: well for well in train_files}
    mapping = load_fold_mapping(fold_config, expected_well_ids=train_by_id)
    selected = (
        _quick_cohort(quick_source, mapping)
        if profile == "quick"
        else _full_cohort(mapping)
    )
    required_ids = tuple(
        sorted(
            {
                well_id
                for training_ids, validation_ids in selected.values()
                for well_id in (*training_ids, *validation_ids)
            }
        )
    )
    validation_ids = tuple(
        sorted(
            {
                well_id
                for _, fold_validation in selected.values()
                for well_id in fold_validation
            }
        )
    )
    grid_config = DEFAULT_GRID
    target_config = DEFAULT_TARGET
    sequence_cache = shared_cache / "external_sequence_v1"
    provider = HeelIncumbentProvider(
        data_root=data,
        fold_config=fold_config,
        artifact_root=incumbent_root,
        cache_root=shared_cache,
    )
    incumbent_manifest = provider.state_manifest()
    state_digests = _fold_state_digest(incumbent_manifest)
    incumbent_manifest_sha256 = _canonical_sha256(incumbent_manifest)
    implementation_hashes = _implementation_hashes()

    run_config = {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "run_id": str(run_id),
        "profile": profile,
        "data_root": str(data),
        "fold_config": str(Path(fold_config).resolve()),
        "fold_config_sha256": _sha256(fold_config),
        "cache_root": str(shared_cache),
        "incumbent_root": str(Path(incumbent_root).resolve()),
        "incumbent_manifest_sha256": incumbent_manifest_sha256,
        "incumbent_oof": str(Path(incumbent_oof).resolve()),
        "incumbent_oof_sha256": _sha256(incumbent_oof),
        "quick_source": (
            str(Path(quick_source).resolve()) if profile == "quick" else None
        ),
        "quick_source_sha256": (_sha256(quick_source) if profile == "quick" else None),
        "incumbent_cache_source": (
            None
            if incumbent_cache_source is None
            else str(Path(incumbent_cache_source).resolve())
        ),
        "variants": [asdict(variant) for variant in selected_variants],
        "seeds": list(selected_seeds),
        "epochs": fit_epochs,
        "hidden_dim": int(hidden_dim),
        "layers": int(layers),
        "device": str(device),
        "workers": int(workers),
        "shrinkage": float(shrinkage),
        "bootstrap_samples": int(bootstrap_samples),
        "shadow_samples": int(shadow_samples),
        "shadow_subset_wells": int(shadow_subset_wells),
        "fixed_grid": fixed_grid_manifest(grid_config),
        "residual_target": target_manifest(target_config),
        "cohort": {
            str(fold): {
                "training_well_ids": list(training_ids),
                "validation_well_ids": list(fold_validation),
                "intersection": sorted(set(training_ids) & set(fold_validation)),
                "training_ids_sha256": _ids_sha256(training_ids),
                "validation_ids_sha256": _ids_sha256(fold_validation),
            }
            for fold, (training_ids, fold_validation) in selected.items()
        },
        "data_manifest": before_data,
        "implementation_hashes": implementation_hashes,
        "label_access_policy": {
            "training_csv_targets": "loaded fold-locally for outer-training IDs only",
            "oof_truth": "loaded only after all outer-fold fits complete",
            "outer_validation_labels_used_for_fit": False,
        },
        "external_pretrained_weights_used": [],
        "architecture_only": True,
        "submission_created": False,
    }
    run_root = _safe_run_root(
        output_root,
        run_id,
        data,
        resume=resume,
        run_config=run_config,
    )
    run_config_sha256 = _canonical_sha256(run_config)
    incumbent_manifest_path = run_root / "incumbent_manifest.json"
    if resume:
        if not incumbent_manifest_path.is_file() or _read_json(
            incumbent_manifest_path
        ) != _jsonable(incumbent_manifest):
            raise ValueError("resume incumbent manifest differs or is missing")
    else:
        _atomic_json(incumbent_manifest_path, incumbent_manifest)

    # Before fitting, read only target-free incumbent keys/predictions.  The
    # OOF truth column is opened only after every outer-fold model is fitted.
    baseline_reference_full = _read_incumbent_reference(incumbent_oof)
    baseline_reference = baseline_reference_full.loc[
        baseline_reference_full["well_id"].astype(str).isin(validation_ids)
    ].copy()
    del baseline_reference_full
    if set(baseline_reference["well_id"].astype(str)) != set(validation_ids) or set(
        baseline_reference["fold"].astype(int)
    ) != set(range(FOLD_COUNT)):
        raise ValueError("target-free incumbent reference coverage differs")
    for fold, (_, fold_validation) in selected.items():
        actual = set(
            baseline_reference.loc[
                baseline_reference["fold"].astype(int).eq(fold),
                "well_id",
            ].astype(str)
        )
        if actual != set(fold_validation):
            raise ValueError(f"target-free incumbent reference differs for fold {fold}")

    prepared = _prepare_sequences(
        train_by_id,
        required_ids,
        cache_root=sequence_cache,
        grid_config=grid_config,
        workers=workers,
    )
    raw_source_provenance = {
        well_id: raw_well_source_provenance(train_by_id[well_id])
        for well_id in required_ids
    }

    prediction_blocks: dict[tuple[str, int], list[pd.DataFrame | Path]] = {
        (variant.variant_id, seed): []
        for variant in selected_variants
        for seed in selected_seeds
    }
    completed: dict[tuple[str, int], set[int]] = {
        key: set() for key in prediction_blocks
    }
    failures: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    maximum_incumbent_oof_delta = 0.0

    baseline_by_well = {
        str(well_id): group.sort_values("row_index", kind="stable")
        for well_id, group in baseline_reference.groupby("well_id", sort=False)
    }
    for fold in range(FOLD_COUNT):
        fold_started = time.perf_counter()
        training_ids, fold_validation_ids = selected[fold]
        print(
            f"external portfolio fold={fold} train={len(training_ids)} "
            f"validation={len(fold_validation_ids)}",
            flush=True,
        )
        fold_inputs: dict[str, SequenceModelInput] = {}
        for index, well_id in enumerate((*training_ids, *fold_validation_ids), start=1):
            raw = prepared[well_id]
            incumbent = _cached_incumbent(
                provider,
                run_root=run_root,
                well_id=well_id,
                fold=fold,
                expected_rows=raw.row_index,
                state_digest=state_digests[fold],
                cache_provenance={
                    "raw_source_provenance": raw_source_provenance[well_id],
                    "raw_source_provenance_sha256": (
                        raw_source_provenance_digest(raw_source_provenance[well_id])
                    ),
                    "data_manifest": before_data,
                    "implementation_hashes": implementation_hashes,
                    "incumbent_manifest_sha256": incumbent_manifest_sha256,
                    "fixed_grid": fixed_grid_manifest(grid_config),
                },
                resume=resume,
                source_root=incumbent_cache_source,
            )
            fold_inputs[well_id] = build_sequence_model_input(
                raw,
                incumbent.prediction,
                grid_points=grid_config.horizontal_points,
            )
            if well_id in fold_validation_ids:
                reference = baseline_by_well[well_id]
                if not np.array_equal(
                    raw.row_index,
                    reference["row_index"].to_numpy(dtype=np.int64),
                ):
                    raise ValueError(f"baseline row keys differ for {well_id}")
                delta = float(
                    np.max(
                        np.abs(
                            incumbent.prediction
                            - reference[INCUMBENT_COLUMN].to_numpy(dtype=np.float64)
                        )
                    )
                )
                maximum_incumbent_oof_delta = max(
                    maximum_incumbent_oof_delta,
                    delta,
                )
            if index % 25 == 0 or index == len(training_ids) + len(fold_validation_ids):
                print(
                    f"fold={fold} incumbent inputs={index}/"
                    f"{len(training_ids) + len(fold_validation_ids)}",
                    flush=True,
                )
        if maximum_incumbent_oof_delta > 1.0e-9:
            raise ValueError(
                "recomputed incumbent differs from audited OOF: "
                f"{maximum_incumbent_oof_delta}"
            )

        fold_training_truth = _load_fold_training_targets(
            train_by_id,
            prepared,
            training_ids=training_ids,
            validation_ids=fold_validation_ids,
        )
        training_targets = np.stack(
            [
                make_grid_residual_target(
                    ResidualTrainingWell(
                        inference=fold_inputs[well_id],
                        incumbent_prediction=fold_inputs[well_id].incumbent_prediction,
                        target=fold_training_truth[well_id],
                    ),
                    config=target_config,
                ).correction
                for well_id in training_ids
            ]
        ).astype(np.float32)
        del fold_training_truth
        row_counts = np.asarray(
            [prepared[well_id].prediction_rows for well_id in training_ids],
            dtype=np.int64,
        )
        incumbent_training_sha256 = _canonical_sha256(
            [
                {
                    "well_id": well_id,
                    "row_index_sha256": _array_sha256(fold_inputs[well_id].row_index),
                    "prediction_sha256": _array_sha256(
                        fold_inputs[well_id].incumbent_prediction
                    ),
                }
                for well_id in training_ids
            ]
        )
        training_raw_source_sha256 = _canonical_sha256(
            [
                {
                    "well_id": well_id,
                    "provenance_sha256": raw_source_provenance_digest(
                        raw_source_provenance[well_id]
                    ),
                }
                for well_id in training_ids
            ]
        )
        for variant in selected_variants:
            control_seed = 20260728 + fold
            try:
                controlled_train = _apply_control(
                    [fold_inputs[well_id] for well_id in training_ids],
                    variant,
                    seed=control_seed,
                )
                controlled_validation = _apply_control(
                    [fold_inputs[well_id] for well_id in fold_validation_ids],
                    variant,
                    seed=control_seed,
                )
                train_x = np.stack(
                    [controlled_train[well_id].x for well_id in training_ids]
                ).astype(np.float32)
                validation_x = np.stack(
                    [
                        controlled_validation[well_id].x
                        for well_id in fold_validation_ids
                    ]
                ).astype(np.float32)
                channel_names = controlled_train[training_ids[0]].channel_names
                if any(
                    controlled_train[well_id].channel_names != channel_names
                    for well_id in training_ids
                ) or any(
                    controlled_validation[well_id].channel_names != channel_names
                    for well_id in fold_validation_ids
                ):
                    raise ValueError("channel schema differs within fold")
            except Exception as error:
                failures.append(
                    {
                        "variant_id": variant.variant_id,
                        "fold": fold,
                        "seed": None,
                        "stage": "control_or_input",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(
                    f"ineligible variant={variant.variant_id} fold={fold}: {error}",
                    flush=True,
                )
                continue

            for seed in selected_seeds:
                model_path = (
                    run_root
                    / "models"
                    / variant.variant_id
                    / f"seed_{seed}"
                    / f"fold_{fold}.pt"
                )
                config = _torch_config(
                    variant,
                    seed=seed,
                    device=device,
                    epochs=fit_epochs,
                    hidden_dim=hidden_dim,
                    layers=layers,
                )
                model_started = time.perf_counter()
                try:
                    state, resumed, checkpoint_record = _load_or_fit_state(
                        model_path,
                        x=train_x,
                        targets=training_targets,
                        channel_names=channel_names,
                        training_ids=training_ids,
                        row_counts=row_counts,
                        config=config,
                        provenance={
                            "variant_id": variant.variant_id,
                            "family": variant.family,
                            "control_id": variant.control_id,
                            "fold": int(fold),
                            "seed": int(seed),
                            "validation_well_ids": list(fold_validation_ids),
                            "validation_ids_sha256": _ids_sha256(fold_validation_ids),
                            "data_manifest": before_data,
                            "implementation_hashes": implementation_hashes,
                            "incumbent": {
                                "state_digest": state_digests[fold],
                                "manifest_sha256": incumbent_manifest_sha256,
                                "training_predictions_sha256": (
                                    incumbent_training_sha256
                                ),
                            },
                            "training_raw_source_provenance_sha256": (
                                training_raw_source_sha256
                            ),
                            "run_config_sha256": run_config_sha256,
                            "fixed_grid": fixed_grid_manifest(grid_config),
                            "residual_target": target_manifest(target_config),
                        },
                        resume=resume,
                    )
                    grid_predictions = state.predict(
                        validation_x,
                        channel_names=channel_names,
                        ids=fold_validation_ids,
                    )
                    block_frame = pd.concat(
                        [
                            _prediction_block(
                                inference=controlled_validation[well_id],
                                fold=fold,
                                grid_prediction=grid_predictions[index],
                                target_config=target_config,
                            )
                            for index, well_id in enumerate(fold_validation_ids)
                        ],
                        ignore_index=True,
                    )
                    block_path = (
                        run_root
                        / "prediction_blocks"
                        / variant.variant_id
                        / f"seed_{seed}"
                        / f"fold_{fold}.csv.gz"
                    )
                    _atomic_csv(block_path, block_frame)
                    prediction_blocks[(variant.variant_id, seed)].append(block_path)
                    del block_frame
                    completed[(variant.variant_id, seed)].add(fold)
                    model_records.append(
                        {
                            "variant_id": variant.variant_id,
                            "family": variant.family,
                            "control_id": variant.control_id,
                            "fold": fold,
                            "seed": seed,
                            "training_wells": len(training_ids),
                            "validation_wells": len(fold_validation_ids),
                            "training_validation_intersection": 0,
                            "checkpoint": str(model_path),
                            "checkpoint_sha256": _sha256(model_path),
                            **checkpoint_record,
                            "resumed": resumed,
                            "elapsed_seconds": time.perf_counter() - model_started,
                            "manifest": state.manifest,
                        }
                    )
                    print(
                        f"scored variant={variant.variant_id} fold={fold} seed={seed} "
                        f"seconds={time.perf_counter() - model_started:.2f}",
                        flush=True,
                    )
                except Exception as error:
                    failures.append(
                        {
                            "variant_id": variant.variant_id,
                            "fold": fold,
                            "seed": seed,
                            "stage": "fit_predict_or_reconstruct",
                            "checkpoint": str(model_path),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(
                        f"ineligible variant={variant.variant_id} fold={fold} "
                        f"seed={seed}: {error}",
                        flush=True,
                    )
        fold_records.append(
            {
                "fold": fold,
                "training_wells": len(training_ids),
                "validation_wells": len(fold_validation_ids),
                "validation_rows": int(
                    sum(
                        prepared[well_id].prediction_rows
                        for well_id in fold_validation_ids
                    )
                ),
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
        )
        _atomic_json(
            run_root / "progress.json",
            {
                "completed": {
                    f"{variant_id}/seed_{seed}": sorted(folds)
                    for (variant_id, seed), folds in completed.items()
                },
                "failures": failures,
                "models": model_records,
                "folds": fold_records,
            },
        )

    # Scoring labels are deliberately opened only after every outer-fold
    # checkpoint/prediction block has been produced.
    del baseline_by_well, baseline_reference
    baseline_full = _read_incumbent_oof(incumbent_oof)
    baseline = baseline_full.loc[
        baseline_full["well_id"].astype(str).isin(validation_ids)
    ].copy()
    del baseline_full
    validate_oof_frame(
        baseline,
        (INCUMBENT_COLUMN,),
        expected_well_ids=validation_ids,
        expected_folds=range(FOLD_COUNT),
    )

    oof, prediction_columns, family_columns, incomplete = _assemble_oof(
        baseline,
        variants=selected_variants,
        seeds=selected_seeds,
        blocks=prediction_blocks,
        completed=completed,
        shrinkage=shrinkage,
    )
    oof_path = run_root / "oof_predictions.csv.gz"
    _atomic_csv(oof_path, oof)
    metric_summary = _evaluate_oof(
        oof,
        prediction_columns=prediction_columns,
        family_columns=family_columns,
        metrics_root=run_root / "metrics",
        bootstrap_samples=bootstrap_samples,
        shadow_samples=shadow_samples,
        shadow_subset_wells=shadow_subset_wells,
    )
    after_data = _data_manifest(data)
    if before_data != after_data:
        raise RuntimeError("competition data manifest changed during portfolio run")

    artifact_audit = {
        "oof_path": str(oof_path),
        "oof_sha256": _sha256(oof_path),
        "oof_validation": validate_oof_frame(
            oof,
            prediction_columns,
            expected_keys=baseline,
            expected_well_ids=validation_ids,
            expected_folds=range(FOLD_COUNT),
        ),
        "incumbent_recompute_maximum_absolute_delta": maximum_incumbent_oof_delta,
        "incumbent_recompute_tolerance": 1.0e-9,
        "eligible_prediction_columns": prediction_columns,
        "eligible_family_columns": family_columns,
        "incomplete_family_seeds": incomplete,
        "failures": failures,
        "data_read_only_manifest_before": before_data,
        "data_read_only_manifest_after": after_data,
        "data_manifest_equal": before_data == after_data,
        "submission_created": False,
    }
    _atomic_json(run_root / "artifact_audit.json", artifact_audit)
    _atomic_json(
        run_root / "model_manifest.json",
        {
            "models": model_records,
            "failures": failures,
            "incomplete_family_seeds": incomplete,
            "identity_fallbacks_scored": False,
        },
    )
    summary = {
        "run_id": str(run_id),
        "profile": profile,
        "scope": {
            "wells": int(oof["well_id"].nunique()),
            "rows": int(len(oof)),
            "folds": list(range(FOLD_COUNT)),
            "seeds": list(selected_seeds),
        },
        "metrics": metric_summary,
        "eligible_family_columns": family_columns,
        "incomplete_family_seeds": incomplete,
        "failure_count": len(failures),
        "fold_records": fold_records,
        "elapsed_seconds": time.perf_counter() - started,
        "external_pretrained_weights_used": [],
        "architecture_only": True,
        "data_manifest_equal": True,
        "submission_created": False,
    }
    _atomic_json(run_root / "summary.json", summary)
    return summary


__all__ = [
    "DEFAULT_QUICK_VARIANTS",
    "PortfolioVariant",
    "run_external_model_portfolio_experiment",
]
