"""Provenance and target-free geometry for test-only external predictions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


EXTERNAL_PREDICTION_SCHEMA_VERSION = "external_prediction_registry_v1"
ELIGIBILITY_DIAGNOSTIC_ONLY = "diagnostic_only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_submission_id(values: pd.Series) -> pd.DataFrame:
    """Parse ``well_id_row_index`` without assuming visible well counts."""

    text = values.astype(str)
    extracted = text.str.extract(r"^(?P<well_id>.+)_(?P<row_index>[0-9]+)$")
    if extracted.isna().any(axis=None):
        raise ValueError("submission IDs must end in an integer row index")
    extracted["row_index"] = extracted["row_index"].astype(np.int64)
    if extracted.duplicated(["well_id", "row_index"]).any():
        raise ValueError("submission IDs contain duplicate well/row keys")
    return extracted


@dataclass(frozen=True)
class ExternalPredictionFileRecord:
    """One exact external file and its reuse eligibility."""

    name: str
    path: str
    bytes: int
    sha256: str
    rows: int
    columns: tuple[str, ...]
    unique_ids: int
    duplicate_ids: int
    nonfinite_values: int
    sample_id_set_match: bool
    sample_id_order_match: bool
    eligibility: str = ELIGIBILITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        if self.eligibility != ELIGIBILITY_DIAGNOSTIC_ONLY:
            raise ValueError("test-only external files must remain diagnostic_only")
        if (
            self.bytes < 1
            or self.rows < 1
            or self.unique_ids != self.rows
            or self.duplicate_ids != 0
            or self.nonfinite_values != 0
            or not self.sample_id_set_match
        ):
            raise ValueError("external prediction record failed integrity checks")


class DiagnosticExternalPredictionFamily:
    """Read-only family that explicitly refuses supervised or OOF use."""

    family_id = "rogii03_test_predictions"
    eligibility = ELIGIBILITY_DIAGNOSTIC_ONLY

    def __init__(self, records: Sequence[ExternalPredictionFileRecord]) -> None:
        values = tuple(records)
        if not values or len({record.sha256 for record in values}) != len(values):
            raise ValueError("external prediction records must be non-empty and unique")
        self.records = values

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": EXTERNAL_PREDICTION_SCHEMA_VERSION,
            "family_id": self.family_id,
            "eligibility": self.eligibility,
            "files": [record.__dict__ for record in self.records],
            "train_predictions_available": False,
            "strict_well_level_oof_available": False,
            "fold_definitions_available": False,
            "generation_code_available": False,
            "checkpoints_available": False,
            "fit_allowed": False,
            "oof_scoring_allowed": False,
            "blend_weight_fit_allowed": False,
            "submission_allowed_by_this_registry": False,
        }

    def fit(self, *_: Any, **__: Any) -> None:
        raise PermissionError(
            "diagnostic_only predictions cannot be fitted as a teacher"
        )

    def predict_oof(self, *_: Any, **__: Any) -> None:
        raise PermissionError("test-only predictions cannot produce strict train OOF")

    def load_test_predictions(self, name: str) -> pd.DataFrame:
        matches = [record for record in self.records if record.name == str(name)]
        if len(matches) != 1:
            raise KeyError(f"unknown external prediction file: {name!r}")
        frame = pd.read_csv(matches[0].path, dtype={"id": "string"})
        return frame.loc[:, ["id", "tvt"]].copy()


def audit_external_prediction_files(
    *,
    dataset_dir: Path,
    sample_submission_path: Path,
) -> tuple[DiagnosticExternalPredictionFamily, pd.DataFrame]:
    """Audit all CSVs against the dynamically loaded competition sample IDs."""

    sample = pd.read_csv(sample_submission_path, dtype={"id": "string"})
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError("sample submission schema must be exactly id,tvt")
    if sample["id"].isna().any() or sample["id"].duplicated().any():
        raise ValueError("sample submission IDs must be finite and unique")
    split_submission_id(sample["id"])
    sample_ids = sample["id"].astype(str).to_numpy()
    sample_set = set(sample_ids)

    records: list[ExternalPredictionFileRecord] = []
    rows: list[dict[str, Any]] = []
    paths = tuple(sorted(Path(dataset_dir).glob("*.csv")))
    if not paths:
        raise ValueError("external dataset directory contains no CSV files")
    for path in paths:
        frame = pd.read_csv(path, dtype={"id": "string"})
        columns = tuple(map(str, frame.columns))
        if columns != ("id", "tvt"):
            raise ValueError(f"external prediction schema mismatch: {path}")
        ids = frame["id"].astype(str).to_numpy()
        values = pd.to_numeric(frame["tvt"], errors="raise").to_numpy(dtype=np.float64)
        duplicate_ids = int(pd.Series(ids).duplicated().sum())
        nonfinite = int(np.sum(~np.isfinite(values)))
        record = ExternalPredictionFileRecord(
            name=path.name,
            path=str(path.resolve()),
            bytes=int(path.stat().st_size),
            sha256=_sha256(path),
            rows=int(len(frame)),
            columns=columns,
            unique_ids=int(pd.Series(ids).nunique()),
            duplicate_ids=duplicate_ids,
            nonfinite_values=nonfinite,
            sample_id_set_match=set(ids) == sample_set,
            sample_id_order_match=np.array_equal(ids, sample_ids),
        )
        records.append(record)
        rows.append(record.__dict__.copy())
    return DiagnosticExternalPredictionFamily(records), pd.DataFrame(rows)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = math.sqrt(
        float(left_centered @ left_centered) * float(right_centered @ right_centered)
    )
    return (
        float(left_centered @ right_centered / denominator)
        if denominator > 0.0
        else 0.0
    )


def _well_center(values: np.ndarray, well: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for current in np.unique(well):
        mask = well == current
        result[mask] = values[mask] - float(np.mean(values[mask]))
    return result


def _difference_geometry(
    *,
    ids: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    source: str,
    reference_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parsed = split_submission_id(pd.Series(ids))
    well = parsed["well_id"].astype(str).to_numpy()
    row_index = parsed["row_index"].to_numpy(dtype=np.int64)
    difference = prediction - reference
    per_well: list[dict[str, Any]] = []
    common_sse = 0.0
    slope_sse = 0.0
    shape_sse = 0.0
    high_frequency_energy = 0.0
    total_shape_energy = 0.0
    terminal: list[float] = []
    for well_id in sorted(np.unique(well)):
        mask = well == well_id
        order = np.argsort(row_index[mask], kind="stable")
        current = difference[mask][order]
        current_rows = row_index[mask][order].astype(np.float64)
        common = float(np.mean(current))
        centered = current - common
        x = current_rows - float(np.mean(current_rows))
        scale = float(np.max(np.abs(x))) if len(x) else 0.0
        normalized_x = x / scale if scale > 0.0 else np.zeros_like(x)
        denominator = float(normalized_x @ normalized_x)
        coefficient = (
            float(normalized_x @ centered) / denominator if denominator > 0.0 else 0.0
        )
        slope = coefficient * normalized_x
        shape = centered - slope
        spectrum = np.fft.rfft(shape)
        power = np.square(np.abs(spectrum))
        cutoff = max(1, int(math.ceil(0.20 * len(power))))
        current_total = float(np.sum(power[1:]))
        current_high = float(np.sum(power[cutoff:]))
        roughness = (
            float(np.sqrt(np.mean(np.square(np.diff(shape)))))
            if len(shape) > 1
            else 0.0
        )
        common_sse += len(current) * common * common
        slope_sse += float(slope @ slope)
        shape_sse += float(shape @ shape)
        high_frequency_energy += current_high
        total_shape_energy += current_total
        terminal.append(float(current[-1]))
        per_well.append(
            {
                "source": source,
                "reference": reference_name,
                "well_id": well_id,
                "rows": int(len(current)),
                "common_difference": common,
                "linear_slope_endpoint_amplitude": coefficient,
                "zero_mean_shape_rmse": float(np.sqrt(np.mean(np.square(shape)))),
                "terminal_difference": float(current[-1]),
                "shape_first_difference_rmse": roughness,
                "shape_high_frequency_energy_fraction": (
                    current_high / current_total if current_total > 0.0 else 0.0
                ),
            }
        )
    summary = {
        "source": source,
        "reference": reference_name,
        "rows": int(len(ids)),
        "wells": int(len(np.unique(well))),
        "raw_correlation": _safe_correlation(prediction, reference),
        "within_well_centered_correlation": _safe_correlation(
            _well_center(prediction, well), _well_center(reference, well)
        ),
        "prediction_difference_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "common_difference_rmse": math.sqrt(common_sse / len(ids)),
        "linear_slope_difference_rmse": math.sqrt(slope_sse / len(ids)),
        "zero_mean_shape_difference_rmse": math.sqrt(shape_sse / len(ids)),
        "terminal_difference_rmse": float(np.sqrt(np.mean(np.square(terminal)))),
        "shape_high_frequency_energy_fraction": (
            high_frequency_energy / total_shape_energy
            if total_shape_energy > 0.0
            else 0.0
        ),
    }
    return summary, per_well


def external_prediction_geometry(
    *,
    family: DiagnosticExternalPredictionFamily,
    sample_submission_path: Path,
    incumbent: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute target-free pairwise and incumbent-relative prediction geometry."""

    sample = pd.read_csv(sample_submission_path, dtype={"id": "string"})
    ids = sample["id"].astype(str).to_numpy()
    parsed = split_submission_id(sample["id"])
    well = parsed["well_id"].astype(str).to_numpy()
    predictions: dict[str, np.ndarray] = {}
    for record in family.records:
        frame = family.load_test_predictions(record.name)
        if not np.array_equal(frame["id"].astype(str).to_numpy(), ids):
            raise ValueError(f"external prediction order mismatch: {record.name}")
        predictions[record.name] = frame["tvt"].to_numpy(dtype=np.float64)

    pairwise: list[dict[str, Any]] = []
    names = tuple(sorted(predictions))
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = predictions[left_name]
            right = predictions[right_name]
            pairwise.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "raw_correlation": _safe_correlation(left, right),
                    "within_well_centered_correlation": _safe_correlation(
                        _well_center(left, well), _well_center(right, well)
                    ),
                    "prediction_rmse": float(np.sqrt(np.mean(np.square(left - right)))),
                }
            )

    summaries: list[dict[str, Any]] = []
    per_well: list[dict[str, Any]] = []
    if incumbent is not None:
        required = {"id", "tvt_incumbent"}
        if required - set(incumbent.columns):
            raise ValueError("incumbent geometry frame requires id,tvt_incumbent")
        joined = sample.loc[:, ["id"]].merge(
            incumbent.loc[:, ["id", "tvt_incumbent"]],
            on="id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        incumbent_values = joined["tvt_incumbent"].to_numpy(dtype=np.float64)
        if not np.isfinite(incumbent_values).all():
            raise ValueError("incumbent geometry prediction is incomplete")
        for name in names:
            summary, well_rows = _difference_geometry(
                ids=ids,
                prediction=predictions[name],
                reference=incumbent_values,
                source=name,
                reference_name="heel_registered_incumbent",
            )
            summaries.append(summary)
            per_well.extend(well_rows)
    return {
        "pairwise": pd.DataFrame(pairwise),
        "vs_incumbent_summary": pd.DataFrame(summaries),
        "vs_incumbent_per_well": pd.DataFrame(per_well),
    }


__all__ = [
    "ELIGIBILITY_DIAGNOSTIC_ONLY",
    "EXTERNAL_PREDICTION_SCHEMA_VERSION",
    "DiagnosticExternalPredictionFamily",
    "ExternalPredictionFileRecord",
    "audit_external_prediction_files",
    "external_prediction_geometry",
    "split_submission_id",
]
