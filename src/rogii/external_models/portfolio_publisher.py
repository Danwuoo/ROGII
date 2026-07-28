"""Translate the portfolio runner's wide OOF into canonical evidence artifacts.

The portfolio runner intentionally writes a convenient wide analysis table.
This adapter is the separate evidence boundary that:

* accepts only completed family means;
* requires every declared outer-fold/seed checkpoint and loads it fail-closed;
* reconstructs or accepts the exact normalized row coordinate;
* publishes common/slope/shape/correction/final columns;
* marks every negative control and random-frozen control ineligible; and
* delegates final atomic writing and independent validation to
  :mod:`rogii.external_models.artifacts`.

It never writes to the competition data directory and does not alter runner
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import enumerate_wells, load_horizontal_inference, validate_prefix
from ..portfolio_metrics import (
    DEFAULT_TRUTH_COLUMN,
    OOF_KEY_COLUMNS,
    validate_oof_frame,
)
from .artifacts import (
    DEFAULT_AUDITED_INCUMBENT_OOF,
    DEFAULT_BUNDLE_NAME,
    INCUMBENT_COLUMN,
    ROW_COORDINATE_COLUMN,
    ExternalArtifactBundle,
    FoldProvenance,
    ModelArtifact,
    OofCandidateSchema,
    canonical_oof_columns,
    make_model_artifact,
    load_audited_incumbent_reference,
    select_audited_incumbent_reference,
    sha256_file,
    validate_data_manifest,
    well_ids_sha256,
    write_external_artifact_bundle,
)
from .torch_families import TorchFamilyState, load_torch_family


PUBLISHER_SCHEMA_VERSION = "rogii_external_portfolio_publisher_v1"
RUN_CONFIG_FILENAME = "run_config.json"
MODEL_MANIFEST_FILENAME = "model_manifest.json"
WIDE_OOF_FILENAME = "oof_predictions.csv.gz"
INCUMBENT_MANIFEST_FILENAME = "incumbent_manifest.json"
_ATOL = 1.0e-8
_COORDINATE_ATOL = 1.0e-12


@dataclass(frozen=True)
class _Variant:
    variant_id: str
    family: str
    control_id: str
    cross_series: bool
    frozen_random_encoder: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_Variant":
        if not isinstance(value, Mapping):
            raise TypeError("run_config variants must be mappings")
        result = cls(
            variant_id=str(value["variant_id"]),
            family=str(value["family"]),
            control_id=str(value.get("control_id", "none")),
            cross_series=value.get("cross_series", True),
            frozen_random_encoder=value.get("frozen_random_encoder", False),
        )
        # OofCandidateSchema owns the canonical identifier grammar.  Constructing
        # a temporary schema here gives configuration errors a single definition.
        if not isinstance(result.cross_series, bool):
            raise TypeError("variant cross_series must be bool")
        if not isinstance(result.frozen_random_encoder, bool):
            raise TypeError("variant frozen_random_encoder must be bool")
        if not result.variant_id or not result.family or not result.control_id:
            raise ValueError("variant identifiers must not be empty")
        return result

    @property
    def transfer_mode(self) -> str:
        if self.control_id != "none":
            return "negative_control"
        if self.frozen_random_encoder:
            return "random_frozen"
        return "architecture_only"

    @property
    def eligible(self) -> bool:
        return self.control_id == "none" and not self.frozen_random_encoder


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid finite JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _json_identity(value: Any, *, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and JSON serializable") from exc


def _read_wide_oof(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"wide OOF does not exist: {path}")
    try:
        frame = pd.read_csv(
            path,
            dtype={"well_id": "string", "row_index": "int64", "fold": "int64"},
            float_precision="round_trip",
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"wide OOF could not be read: {path}") from exc
    if frame.columns.has_duplicates:
        raise ValueError("wide OOF contains duplicate columns")
    required = {
        *OOF_KEY_COLUMNS,
        DEFAULT_TRUTH_COLUMN,
        INCUMBENT_COLUMN,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"wide OOF is missing columns: {sorted(missing)}")
    validate_oof_frame(frame, (INCUMBENT_COLUMN,))
    return frame


def _finite_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"wide OOF is missing column {column!r}")
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"wide OOF column {column!r} must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"wide OOF column {column!r} must be finite")
    return values


def _aligned_coordinate_frame(
    wide: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(coordinates, pd.DataFrame):
        raise TypeError("row coordinates must be a pandas DataFrame")
    if coordinates.columns.has_duplicates:
        raise ValueError("row-coordinate artifact has duplicate columns")
    required = {"well_id", "row_index", ROW_COORDINATE_COLUMN}
    missing = required - set(coordinates.columns)
    if missing:
        raise ValueError(
            f"row-coordinate artifact is missing columns: {sorted(missing)}"
        )
    allowed = required | {"fold"}
    extra = set(coordinates.columns) - allowed
    if extra:
        raise ValueError(
            f"row-coordinate artifact has unexpected columns: {sorted(extra)}"
        )
    current = coordinates.copy()
    current["well_id"] = current["well_id"].astype(str)
    current["row_index"] = pd.to_numeric(current["row_index"], errors="raise").astype(
        np.int64
    )
    merge_keys = ["well_id", "row_index"]
    if "fold" in current:
        current["fold"] = pd.to_numeric(current["fold"], errors="raise").astype(
            np.int64
        )
        merge_keys.append("fold")
    if current.duplicated(merge_keys).any():
        raise ValueError("row-coordinate artifact contains duplicate keys")

    reference = wide.loc[:, list(OOF_KEY_COLUMNS)].copy()
    reference["well_id"] = reference["well_id"].astype(str)
    reference["row_index"] = pd.to_numeric(
        reference["row_index"], errors="raise"
    ).astype(np.int64)
    reference["fold"] = pd.to_numeric(reference["fold"], errors="raise").astype(
        np.int64
    )
    merged = reference.merge(
        current,
        on=merge_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all() or len(merged) != len(reference):
        raise ValueError("row-coordinate artifact does not cover exact wide OOF keys")
    if "fold" not in merge_keys:
        # Folding is still locked by the wide reference; a well may not appear in
        # two folds because validate_oof_frame has already rejected that schema.
        merged["fold"] = reference["fold"].to_numpy(dtype=np.int64)
    result = merged.loc[:, [*OOF_KEY_COLUMNS, ROW_COORDINATE_COLUMN]].copy()
    result[ROW_COORDINATE_COLUMN] = pd.to_numeric(
        result[ROW_COORDINATE_COLUMN], errors="raise"
    ).astype(np.float64)
    _validate_row_coordinates(result)
    return result


def _validate_row_coordinates(coordinates: pd.DataFrame) -> None:
    values = coordinates[ROW_COORDINATE_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("row_coordinate must be finite")
    for well_id, group in coordinates.groupby("well_id", sort=True):
        ordered = group.sort_values("row_index", kind="stable")
        rows = ordered["row_index"].to_numpy(dtype=np.int64)
        current = ordered[ROW_COORDINATE_COLUMN].to_numpy(dtype=np.float64)
        if len(current) < 1 or np.any(np.diff(rows) <= 0):
            raise ValueError(
                f"row indices must be strictly increasing for well {well_id}"
            )
        if len(current) > 1 and np.any(np.diff(current) <= 0.0):
            raise ValueError(
                f"row_coordinate must be strictly increasing for well {well_id}"
            )
        if current[0] < -_COORDINATE_ATOL or current[-1] > 1.0 + _COORDINATE_ATOL:
            raise ValueError(
                f"row_coordinate must lie within [0, 1] for well {well_id}"
            )
        if abs(float(current[-1]) - 1.0) > _COORDINATE_ATOL:
            raise ValueError(
                f"normalized row_coordinate must terminate at 1 for well {well_id}"
            )


def reconstruct_row_coordinates(
    wide: pd.DataFrame,
    *,
    data_root: str | Path,
) -> pd.DataFrame:
    """Rebuild the runner's exact MD-normalized suffix coordinate.

    Only inference-safe horizontal columns are read.  The formula is identical
    to ``build_fixed_grid_sequence``:

    ``(MD[row] - MD[last_visible]) / (MD[last] - MD[last_visible])``.
    """

    data = Path(data_root).resolve()
    wells = {well.well_id: well for well in enumerate_wells(data, "train")}
    requested = tuple(sorted(wide["well_id"].astype(str).unique()))
    missing = sorted(set(requested) - set(wells))
    if missing:
        raise ValueError(
            f"wide OOF wells are absent from the train split: {missing[:5]}"
        )
    records: list[pd.DataFrame] = []
    for well_id in requested:
        horizontal = load_horizontal_inference(wells[well_id].horizontal_path)
        known_count, prediction_count = validate_prefix(horizontal["TVT_input"])
        md = pd.to_numeric(horizontal["MD"], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(md).all() or np.any(np.diff(md) <= 0.0):
            raise ValueError(f"horizontal MD must be finite/increasing: {well_id}")
        row_index = np.arange(
            known_count,
            known_count + prediction_count,
            dtype=np.int64,
        )
        anchor_md = float(md[known_count - 1])
        horizon = float(md[-1] - anchor_md)
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValueError(f"prediction MD horizon must be positive: {well_id}")
        coordinate = (md[row_index] - anchor_md) / horizon
        fold_values = (
            wide.loc[wide["well_id"].astype(str).eq(well_id), "fold"]
            .astype(int)
            .unique()
        )
        if len(fold_values) != 1:
            raise ValueError(f"well appears in multiple OOF folds: {well_id}")
        records.append(
            pd.DataFrame(
                {
                    "well_id": well_id,
                    "row_index": row_index,
                    "fold": int(fold_values[0]),
                    ROW_COORDINATE_COLUMN: coordinate,
                }
            )
        )
    rebuilt = pd.concat(records, ignore_index=True)
    return _aligned_coordinate_frame(wide, rebuilt)


def resolve_row_coordinates(
    wide: pd.DataFrame,
    *,
    data_root: str | Path,
    row_coordinate_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve an explicit artifact, an existing wide column, or raw MD."""

    wide_coordinates: pd.DataFrame | None = None
    if ROW_COORDINATE_COLUMN in wide:
        wide_coordinates = _aligned_coordinate_frame(
            wide,
            wide.loc[:, [*OOF_KEY_COLUMNS, ROW_COORDINATE_COLUMN]],
        )

    explicit: pd.DataFrame | None = None
    explicit_path: Path | None = None
    if row_coordinate_path is not None:
        explicit_path = Path(row_coordinate_path).resolve()
        if not explicit_path.is_file():
            raise FileNotFoundError(
                f"row-coordinate artifact does not exist: {explicit_path}"
            )
        try:
            raw = pd.read_csv(
                explicit_path,
                dtype={"well_id": "string", "row_index": "int64"},
                float_precision="round_trip",
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(
                f"row-coordinate artifact could not be read: {explicit_path}"
            ) from exc
        explicit = _aligned_coordinate_frame(wide, raw)

    if explicit is not None and wide_coordinates is not None:
        left = explicit[ROW_COORDINATE_COLUMN].to_numpy(dtype=np.float64)
        right = wide_coordinates[ROW_COORDINATE_COLUMN].to_numpy(dtype=np.float64)
        if not np.array_equal(left, right):
            raise ValueError(
                "explicit row-coordinate artifact differs from the wide OOF"
            )
        selected = explicit
        source = "explicit_and_wide"
    elif explicit is not None:
        selected = explicit
        source = "explicit_artifact"
    elif wide_coordinates is not None:
        selected = wide_coordinates
        source = "wide_oof"
    else:
        selected = reconstruct_row_coordinates(wide, data_root=data_root)
        source = "raw_inference_md"

    metadata: dict[str, Any] = {
        "source": source,
        "formula": (
            "(MD[row]-MD[last_visible])/(MD[last]-MD[last_visible])"
            if source == "raw_inference_md"
            else "provided key-aligned normalized coordinate"
        ),
        "strictly_increasing_per_well": True,
        "bounded_0_1": True,
        "terminal_equals_1": True,
        "rows": int(len(selected)),
        "wells": int(selected["well_id"].nunique()),
    }
    if explicit_path is not None:
        metadata.update(
            {
                "artifact_path": str(explicit_path),
                "artifact_sha256": sha256_file(explicit_path),
                "artifact_bytes": explicit_path.stat().st_size,
            }
        )
    return selected, metadata


def _parse_variants(run_config: Mapping[str, Any]) -> tuple[_Variant, ...]:
    raw = run_config.get("variants")
    if not isinstance(raw, list) or not raw:
        raise ValueError("run_config must contain non-empty variants")
    variants = tuple(_Variant.from_mapping(value) for value in raw)
    identifiers = [value.variant_id for value in variants]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("run_config variant IDs must be unique")
    return variants


def _parse_seeds(run_config: Mapping[str, Any]) -> tuple[int, ...]:
    raw = run_config.get("seeds")
    if not isinstance(raw, list) or not raw:
        raise ValueError("run_config must contain non-empty seeds")
    seeds = tuple(int(value) for value in raw)
    if (
        any(value < 0 for value in seeds)
        or len(set(seeds)) != len(seeds)
        or any(isinstance(value, bool) for value in raw)
    ):
        raise ValueError("run_config seeds must be unique non-negative integers")
    return seeds


def _fold_provenance(
    run_config: Mapping[str, Any],
    wide: pd.DataFrame,
) -> tuple[
    tuple[FoldProvenance, ...],
    dict[int, tuple[str, ...]],
    dict[int, tuple[str, ...]],
]:
    cohort = run_config.get("cohort")
    if not isinstance(cohort, Mapping) or not cohort:
        raise ValueError("run_config must contain fold cohort provenance")
    try:
        records = {int(key): value for key, value in cohort.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("cohort fold keys must be integers") from exc
    actual_folds = set(pd.to_numeric(wide["fold"], errors="raise").astype(int).unique())
    if set(records) != actual_folds:
        raise ValueError(
            "run_config cohort folds differ from wide OOF folds; "
            f"config={sorted(records)}, wide={sorted(actual_folds)}"
        )

    folds: list[FoldProvenance] = []
    training_by_fold: dict[int, tuple[str, ...]] = {}
    validation_by_fold: dict[int, tuple[str, ...]] = {}
    for fold in sorted(records):
        value = records[fold]
        if not isinstance(value, Mapping):
            raise TypeError(f"cohort fold {fold} must be a mapping")
        training = tuple(map(str, value["training_well_ids"]))
        validation = tuple(map(str, value["validation_well_ids"]))
        if value.get("training_ids_sha256") != well_ids_sha256(training):
            raise ValueError(f"cohort fold {fold} training ID digest differs")
        if value.get("validation_ids_sha256") != well_ids_sha256(validation):
            raise ValueError(f"cohort fold {fold} validation ID digest differs")
        if value.get("intersection") not in ([], ()):
            raise ValueError(f"cohort fold {fold} declares train/validation overlap")
        actual_validation = set(
            wide.loc[wide["fold"].astype(int).eq(fold), "well_id"].astype(str)
        )
        if actual_validation != set(validation):
            raise ValueError(f"cohort fold {fold} validation IDs differ from wide OOF")
        record = FoldProvenance(
            outer_fold=fold,
            training_well_ids=training,
            validation_well_ids=validation,
        )
        folds.append(record)
        training_by_fold[fold] = training
        validation_by_fold[fold] = validation
    return tuple(folds), training_by_fold, validation_by_fold


def _model_record_index(
    model_manifest: Mapping[str, Any],
    *,
    variants: Sequence[_Variant],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    if model_manifest.get("identity_fallbacks_scored") is not False:
        raise ValueError("model_manifest must declare identity_fallbacks_scored=false")
    raw = model_manifest.get("models")
    if not isinstance(raw, list):
        raise ValueError("model_manifest models must be a list")
    allowed_variants = {value.variant_id for value in variants}
    allowed_folds = set(map(int, folds))
    allowed_seeds = set(map(int, seeds))
    result: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for value in raw:
        if not isinstance(value, Mapping):
            raise TypeError("model_manifest model records must be mappings")
        key = (
            str(value["variant_id"]),
            int(value["fold"]),
            int(value["seed"]),
        )
        if key in result:
            raise ValueError(f"duplicate model record: {key}")
        if (
            key[0] not in allowed_variants
            or key[1] not in allowed_folds
            or key[2] not in allowed_seeds
        ):
            raise ValueError(f"model record escapes run_config domain: {key}")
        result[key] = value
    return result


def _resolve_checkpoint(
    run_root: Path,
    record: Mapping[str, Any],
    *,
    variant_id: str,
    fold: int,
    seed: int,
) -> Path:
    declared = Path(str(record["checkpoint"]))
    if not declared.is_absolute():
        declared = run_root / declared
    checkpoint = declared.resolve()
    expected = (
        run_root / "models" / variant_id / f"seed_{seed}" / f"fold_{fold}.pt"
    ).resolve()
    try:
        checkpoint.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("checkpoint path escapes run root") from exc
    if checkpoint != expected:
        raise ValueError(
            f"checkpoint path differs from runner convention: {checkpoint}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint is missing for {variant_id}/fold_{fold}/seed_{seed}: "
            f"{checkpoint}"
        )
    return checkpoint


def _validate_checkpoint_state(
    state: TorchFamilyState,
    record: Mapping[str, Any],
    *,
    variant: _Variant,
    fold: int,
    seed: int,
    training_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
) -> None:
    if state.config.family != variant.family:
        raise ValueError("checkpoint family differs from run_config")
    if state.config.seed != seed:
        raise ValueError("checkpoint seed differs from run_config/model record")
    if state.config.frozen_random_encoder != variant.frozen_random_encoder:
        raise ValueError("checkpoint frozen-random mode differs from run_config")
    if variant.family == "paired_patch_cross":
        if state.config.cross_series != variant.cross_series:
            raise ValueError("checkpoint cross-series mode differs from run_config")
    expected_fit_digest = well_ids_sha256(training_ids)
    if state.manifest.get("fit_ids_sha256") != expected_fit_digest:
        raise ValueError("checkpoint fit IDs differ from exact fold training IDs")
    if int(state.manifest.get("sample_count", -1)) != len(training_ids):
        raise ValueError("checkpoint sample count differs from fold training IDs")
    record_manifest = record.get("manifest")
    if not isinstance(record_manifest, Mapping):
        raise ValueError("model record lacks checkpoint state manifest")
    if _json_identity(record_manifest, label="record manifest") != _json_identity(
        state.manifest, label="checkpoint manifest"
    ):
        raise ValueError("model record manifest differs from checkpoint manifest")
    checks = {
        "family": variant.family,
        "variant_id": variant.variant_id,
        "control_id": variant.control_id,
        "fold": fold,
        "seed": seed,
        "training_wells": len(training_ids),
        "validation_wells": len(validation_ids),
        "training_validation_intersection": 0,
    }
    for field, expected in checks.items():
        if record.get(field) != expected:
            raise ValueError(
                f"model record field {field!r} differs; "
                f"expected={expected!r}, actual={record.get(field)!r}"
            )


def _source_models_for_variant(
    run_root: Path,
    variant: _Variant,
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    training_by_fold: Mapping[int, tuple[str, ...]],
    validation_by_fold: Mapping[int, tuple[str, ...]],
    record_index: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> tuple[ModelArtifact, ...]:
    artifacts: list[ModelArtifact] = []
    for fold in sorted(map(int, folds)):
        for seed in sorted(map(int, seeds)):
            key = (variant.variant_id, fold, seed)
            if key not in record_index:
                raise ValueError(
                    "incomplete checkpoint coverage for complete candidate: "
                    f"{variant.variant_id}/fold_{fold}/seed_{seed}"
                )
            record = record_index[key]
            checkpoint = _resolve_checkpoint(
                run_root,
                record,
                variant_id=variant.variant_id,
                fold=fold,
                seed=seed,
            )
            actual_sha = sha256_file(checkpoint)
            if record.get("checkpoint_sha256") != actual_sha:
                raise ValueError(
                    f"checkpoint SHA-256 differs for {variant.variant_id}/"
                    f"fold_{fold}/seed_{seed}"
                )
            state = load_torch_family(checkpoint)
            _validate_checkpoint_state(
                state,
                record,
                variant=variant,
                fold=fold,
                seed=seed,
                training_ids=training_by_fold[fold],
                validation_ids=validation_by_fold[fold],
            )
            source_variant = str(state.manifest.get("variant", "default"))
            model_id = f"{variant.variant_id}__fold_{fold}__seed_{seed}"
            artifacts.append(
                make_model_artifact(
                    run_root,
                    checkpoint,
                    model_id=model_id,
                    family_id=variant.family,
                    variant_id=source_variant,
                    fit_scope="outer_training",
                    fit_well_ids=training_by_fold[fold],
                    outer_fold=fold,
                    seed=seed,
                    state_manifest=state.manifest,
                )
            )
    return tuple(artifacts)


def _component_source_columns(variant_id: str) -> tuple[str, str, str]:
    return (
        f"correction__{variant_id}__common",
        f"correction__{variant_id}__slope",
        f"correction__{variant_id}__shape",
    )


def _family_is_complete(
    wide: pd.DataFrame,
    variant: _Variant,
    *,
    seeds: Sequence[int],
    model_manifest: Mapping[str, Any],
) -> bool:
    final = f"y_pred__{variant.variant_id}"
    components = _component_source_columns(variant.variant_id)
    present = [column in wide for column in (final, *components)]
    if any(present) and not all(present):
        raise ValueError(
            f"complete family columns are partial for {variant.variant_id}"
        )
    if not all(present):
        return False
    incomplete = set(map(str, model_manifest.get("incomplete_family_seeds", ())))
    declared = {f"{variant.variant_id}/seed_{int(seed)}" for seed in seeds}
    if incomplete & declared:
        raise ValueError(
            f"wide family mean exists despite incomplete seeds: "
            f"{sorted(incomplete & declared)}"
        )
    failures = model_manifest.get("failures", ())
    if not isinstance(failures, list):
        raise ValueError("model_manifest failures must be a list")
    related = [
        value
        for value in failures
        if isinstance(value, Mapping)
        and str(value.get("variant_id")) == variant.variant_id
    ]
    if related:
        raise ValueError(
            f"wide family mean exists despite recorded failures: {variant.variant_id}"
        )
    seed_columns = [f"y_pred__{variant.variant_id}__seed_{int(seed)}" for seed in seeds]
    missing = [column for column in seed_columns if column not in wide]
    if missing:
        raise ValueError(f"family mean lacks complete seed predictions: {missing}")
    expected = np.mean(
        np.stack([_finite_column(wide, column) for column in seed_columns]),
        axis=0,
    )
    actual = _finite_column(wide, final)
    maximum = float(np.max(np.abs(actual - expected)))
    if maximum > _ATOL:
        raise ValueError(
            f"family mean differs from seed mean for {variant.variant_id}; "
            f"maximum_absolute_delta={maximum}"
        )
    return True


def _append_candidate(
    output: pd.DataFrame,
    wide: pd.DataFrame,
    *,
    variant: _Variant,
    candidate_id: str,
    source_model_ids: Sequence[str],
    scale: float,
    wide_prediction_column: str,
) -> OofCandidateSchema:
    schema = OofCandidateSchema.canonical(
        candidate_id,
        family_id=variant.family,
        variant_id=variant.variant_id,
        transfer_mode=variant.transfer_mode,
        control_id=variant.control_id,
        eligible_for_model_selection=variant.eligible,
        source_model_ids=source_model_ids,
    )
    source_common, source_slope, source_shape = _component_source_columns(
        variant.variant_id
    )
    common = scale * _finite_column(wide, source_common)
    slope = scale * _finite_column(wide, source_slope)
    shape = scale * _finite_column(wide, source_shape)
    correction = common + slope + shape
    prediction = _finite_column(wide, wide_prediction_column)
    incumbent = _finite_column(wide, INCUMBENT_COLUMN)
    maximum = float(np.max(np.abs(prediction - (incumbent + correction))))
    if maximum > _ATOL:
        raise ValueError(
            f"wide prediction/component algebra failed for {candidate_id}; "
            f"maximum_absolute_delta={maximum}"
        )
    output[schema.common_column] = common
    output[schema.slope_column] = slope
    output[schema.shape_column] = shape
    output[schema.correction_column] = correction
    output[schema.prediction_column] = prediction
    return schema


def _build_canonical_oof(
    wide: pd.DataFrame,
    coordinates: pd.DataFrame,
    *,
    variants: Sequence[_Variant],
    seeds: Sequence[int],
    shrinkage: float,
    model_manifest: Mapping[str, Any],
    source_models: Mapping[str, tuple[ModelArtifact, ...]],
) -> tuple[pd.DataFrame, tuple[OofCandidateSchema, ...], tuple[str, ...]]:
    coordinate_values = coordinates[ROW_COORDINATE_COLUMN].to_numpy(dtype=np.float64)
    output = wide.loc[
        :, [*OOF_KEY_COLUMNS, DEFAULT_TRUTH_COLUMN, INCUMBENT_COLUMN]
    ].copy()
    output.insert(len(OOF_KEY_COLUMNS), ROW_COORDINATE_COLUMN, coordinate_values)
    candidates: list[OofCandidateSchema] = []
    skipped: list[str] = []
    shrink_code = int(round(100 * shrinkage))
    for variant in variants:
        if not _family_is_complete(
            wide,
            variant,
            seeds=seeds,
            model_manifest=model_manifest,
        ):
            skipped.append(variant.variant_id)
            continue
        models = source_models[variant.variant_id]
        source_ids = tuple(model.model_id for model in models)
        candidates.append(
            _append_candidate(
                output,
                wide,
                variant=variant,
                candidate_id=variant.variant_id,
                source_model_ids=source_ids,
                scale=1.0,
                wide_prediction_column=f"y_pred__{variant.variant_id}",
            )
        )
        shrink_column = f"y_pred__{variant.variant_id}__shrink_{shrink_code:03d}"
        shrink_prefix = f"y_pred__{variant.variant_id}__shrink_"
        declared_shrink_columns = [
            str(column)
            for column in wide.columns
            if str(column).startswith(shrink_prefix)
        ]
        unexpected = [
            column for column in declared_shrink_columns if column != shrink_column
        ]
        if unexpected:
            raise ValueError(
                f"unexpected shrink columns for {variant.variant_id}: {unexpected}"
            )
        if shrink_column in wide:
            candidates.append(
                _append_candidate(
                    output,
                    wide,
                    variant=variant,
                    candidate_id=f"{variant.variant_id}__shrink_{shrink_code:03d}",
                    source_model_ids=source_ids,
                    scale=shrinkage,
                    wide_prediction_column=shrink_column,
                )
            )
    if not candidates:
        raise ValueError("wide OOF has no complete family mean candidates")
    output = output.loc[:, canonical_oof_columns(candidates)]
    return output, tuple(candidates), tuple(skipped)


def publish_external_portfolio_artifacts(
    run_root: str | Path,
    *,
    row_coordinate_path: str | Path | None = None,
    incumbent_oof_path: str | Path | None = None,
    bundle_name: str = DEFAULT_BUNDLE_NAME,
) -> ExternalArtifactBundle:
    """Publish and independently re-read one completed runner output."""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root}")
    run_config_path = root / RUN_CONFIG_FILENAME
    model_manifest_path = root / MODEL_MANIFEST_FILENAME
    wide_path = root / WIDE_OOF_FILENAME
    incumbent_manifest_path = root / INCUMBENT_MANIFEST_FILENAME
    run_config = _read_json(run_config_path, label="run_config")
    model_manifest = _read_json(model_manifest_path, label="model_manifest")
    incumbent_manifest = _read_json(incumbent_manifest_path, label="incumbent_manifest")
    wide = _read_wide_oof(wide_path)

    run_id = str(run_config.get("run_id", ""))
    profile = str(run_config.get("profile", ""))
    if not run_id or profile not in {"quick", "full"}:
        raise ValueError("run_config run_id/profile are invalid")
    if run_config.get("architecture_only") is not True:
        raise ValueError("current publisher accepts architecture-only runs")
    if run_config.get("external_pretrained_weights_used") != []:
        raise ValueError("architecture-only run declares external pretrained weights")
    if run_config.get("submission_created") is not False:
        raise ValueError("run_config must declare submission_created=false")

    variants = _parse_variants(run_config)
    seeds = _parse_seeds(run_config)
    folds, training_by_fold, validation_by_fold = _fold_provenance(run_config, wide)
    validate_oof_frame(
        wide,
        (INCUMBENT_COLUMN,),
        expected_well_ids=sorted(
            well_id for values in validation_by_fold.values() for well_id in values
        ),
        expected_folds=sorted(training_by_fold),
    )
    data_root = Path(str(run_config["data_root"])).resolve()
    if root == data_root or data_root in root.parents:
        raise ValueError("run root cannot be inside competition data")
    before_data = validate_data_manifest(run_config["data_manifest"])
    configured_incumbent = run_config.get("incumbent_oof")
    selected_incumbent = (
        Path(incumbent_oof_path)
        if incumbent_oof_path is not None
        else Path(str(configured_incumbent))
        if configured_incumbent
        else DEFAULT_AUDITED_INCUMBENT_OOF
    )
    locked_incumbent, locked_incumbent_record = load_audited_incumbent_reference(
        selected_incumbent,
        expected_sha256=run_config.get("incumbent_oof_sha256"),
    )
    coordinates, coordinate_metadata = resolve_row_coordinates(
        wide,
        data_root=data_root,
        row_coordinate_path=row_coordinate_path,
    )
    fold_ids = tuple(record.outer_fold for record in folds)
    record_index = _model_record_index(
        model_manifest,
        variants=variants,
        folds=fold_ids,
        seeds=seeds,
    )

    complete_variants = tuple(
        variant
        for variant in variants
        if _family_is_complete(
            wide,
            variant,
            seeds=seeds,
            model_manifest=model_manifest,
        )
    )
    source_models: dict[str, tuple[ModelArtifact, ...]] = {}
    all_models: list[ModelArtifact] = []
    for variant in complete_variants:
        values = _source_models_for_variant(
            root,
            variant,
            folds=fold_ids,
            seeds=seeds,
            training_by_fold=training_by_fold,
            validation_by_fold=validation_by_fold,
            record_index=record_index,
        )
        source_models[variant.variant_id] = values
        all_models.extend(values)
    canonical, candidates, skipped = _build_canonical_oof(
        wide,
        coordinates,
        variants=variants,
        seeds=seeds,
        shrinkage=float(run_config["shrinkage"]),
        model_manifest=model_manifest,
        source_models=source_models,
    )
    reference = select_audited_incumbent_reference(
        canonical,
        locked_incumbent,
    )
    reference = reference.sort_values(
        list(OOF_KEY_COLUMNS),
        kind="stable",
    ).reset_index(drop=True)
    canonical = canonical.sort_values(
        list(OOF_KEY_COLUMNS),
        kind="stable",
    ).reset_index(drop=True)
    if not np.array_equal(
        canonical.loc[:, list(OOF_KEY_COLUMNS)].to_numpy(),
        reference.loc[:, list(OOF_KEY_COLUMNS)].to_numpy(),
    ):
        raise ValueError("audited incumbent reference key order differs")
    canonical[DEFAULT_TRUTH_COLUMN] = reference[DEFAULT_TRUTH_COLUMN].to_numpy(
        dtype=np.float64
    )
    canonical[INCUMBENT_COLUMN] = reference[INCUMBENT_COLUMN].to_numpy(dtype=np.float64)
    fold_config_path = Path(str(run_config["fold_config"])).resolve()
    metadata = {
        "publisher_schema_version": PUBLISHER_SCHEMA_VERSION,
        "source_files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in (
                (RUN_CONFIG_FILENAME, run_config_path),
                (MODEL_MANIFEST_FILENAME, model_manifest_path),
                (WIDE_OOF_FILENAME, wide_path),
                (INCUMBENT_MANIFEST_FILENAME, incumbent_manifest_path),
            )
        },
        "row_coordinate": coordinate_metadata,
        "locked_incumbent_reference": locked_incumbent_record,
        "architecture_only": True,
        "external_pretrained_weights_used": [],
        "source_wide_column_order": list(map(str, wide.columns)),
        "published_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "family_id": candidate.family_id,
                "variant_id": candidate.variant_id,
                "transfer_mode": candidate.transfer_mode,
                "control_id": candidate.control_id,
                "eligible_for_model_selection": (
                    candidate.eligible_for_model_selection
                ),
                "source_model_count": len(candidate.source_model_ids),
            }
            for candidate in candidates
        ],
        "skipped_incomplete_variants": list(skipped),
        "seeds": list(seeds),
        "folds": list(fold_ids),
        "fixed_grid": run_config.get("fixed_grid", {}),
        "residual_target": run_config.get("residual_target", {}),
        "wide_oof_reference_fields": [
            *OOF_KEY_COLUMNS,
            ROW_COORDINATE_COLUMN,
            DEFAULT_TRUTH_COLUMN,
            INCUMBENT_COLUMN,
        ],
    }
    return write_external_artifact_bundle(
        root,
        canonical,
        run_id=run_id,
        profile=profile,
        candidates=candidates,
        fold_provenance=folds,
        model_artifacts=all_models,
        data_root=data_root,
        data_manifest_before=before_data,
        fold_config_path=fold_config_path,
        incumbent_manifest=incumbent_manifest,
        expected_reference=reference,
        metadata=metadata,
        bundle_name=bundle_name,
    )


__all__ = [
    "INCUMBENT_MANIFEST_FILENAME",
    "MODEL_MANIFEST_FILENAME",
    "PUBLISHER_SCHEMA_VERSION",
    "RUN_CONFIG_FILENAME",
    "WIDE_OOF_FILENAME",
    "publish_external_portfolio_artifacts",
    "reconstruct_row_coordinates",
    "resolve_row_coordinates",
]
