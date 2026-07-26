"""Inference-safe model wrapper for the signed probabilistic TVT lattice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data import (
    WellFiles,
    load_horizontal_inference,
    load_typewell_inference,
    validate_prefix,
)
from ..probabilistic_lattice import (
    LatticeConfig,
    LatticeResult,
    LatticeStatistics,
    PrefixCalibration,
    derive_prefix_calibration,
    fit_lattice_statistics,
    infer_signed_tvt_lattice,
    infer_signed_tvt_posterior_mean,
    interpolate_typewell_gr,
)
from .base import InferenceWell, RunContext, TrainingWell


LATTICE_VERSION = "signed_tvt_probabilistic_lattice_mvp_v1"
DEFAULT_LATTICE_CONFIG = LatticeConfig()


@dataclass(frozen=True)
class LatticeInferenceWell(InferenceWell):
    """Compact inference-only payload; no target-bearing field is present."""

    anchor_md: float
    prediction_md: np.ndarray
    prediction_gr: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray
    calibration: PrefixCalibration


@dataclass(frozen=True)
class LatticeModelState:
    """Pickle-safe fold model containing only training-derived statistics."""

    statistics: LatticeStatistics
    config: LatticeConfig


class ProbabilisticLatticeModel:
    """Squared-loss MBR predictor from an exact signed-state lattice."""

    model_id = "lattice"

    def __init__(self, *, config: LatticeConfig = DEFAULT_LATTICE_CONFIG) -> None:
        self.config = config

    def prepare(
        self, well: WellFiles, *, split: str, cache_root: Path
    ) -> LatticeInferenceWell:
        # Preparation is deterministic and uses only the inference schemas.
        del split, cache_root
        horizontal = load_horizontal_inference(well.horizontal_path)
        typewell = load_typewell_inference(well.typewell_path)
        known_count, prediction_count = validate_prefix(horizontal["TVT_input"])

        md = horizontal["MD"].to_numpy(dtype=np.float64)
        gr = horizontal["GR"].to_numpy(dtype=np.float64)
        tvt_input = horizontal["TVT_input"].to_numpy(dtype=np.float64)
        typewell_tvt = typewell["TVT"].to_numpy(dtype=np.float64)
        typewell_gr = typewell["GR"].to_numpy(dtype=np.float64)
        if not np.isfinite(md).all() or np.any(np.diff(md) <= 0):
            raise ValueError(f"Horizontal MD must be finite and increasing: {well.well_id}")
        if not np.isfinite(typewell_tvt).all() or np.any(np.diff(typewell_tvt) <= 0):
            raise ValueError(f"Typewell TVT must be finite and increasing: {well.well_id}")

        calibration = derive_prefix_calibration(
            md,
            gr,
            tvt_input,
            typewell_tvt,
            typewell_gr,
            config=self.config,
        )
        row_index = np.arange(known_count, len(horizontal), dtype=np.int64)
        prediction_md = md[known_count:].copy()
        prediction_gr = gr[known_count:].copy()
        if len(row_index) != prediction_count:
            raise RuntimeError(f"Prediction-row construction failed: {well.well_id}")
        # Base protocol consumers expect a row-aligned feature matrix.  The
        # lattice itself uses the typed payload below.
        features = np.column_stack((prediction_md, prediction_gr)).astype(
            np.float64, copy=False
        )
        return LatticeInferenceWell(
            well_id=well.well_id,
            row_index=row_index,
            anchor=float(tvt_input[known_count - 1]),
            features=features,
            anchor_md=float(md[known_count - 1]),
            prediction_md=prediction_md,
            prediction_gr=prediction_gr,
            typewell_tvt=typewell_tvt.copy(),
            typewell_gr=typewell_gr.copy(),
            calibration=calibration,
        )

    def fit(
        self, training_wells: Sequence[TrainingWell], *, context: RunContext
    ) -> LatticeModelState:
        # The MVP is deterministic; the context is intentionally not a source
        # of validation-well information or stochastic state.
        del context
        if not training_wells:
            raise ValueError("Lattice fit requires at least one training well")
        transition_blocks: list[np.ndarray] = []
        emission_blocks: list[np.ndarray] = []
        for training in training_wells:
            inference = training.inference
            if not isinstance(inference, LatticeInferenceWell):
                raise TypeError("Lattice fit requires LatticeInferenceWell payloads")
            target = np.asarray(training.target, dtype=np.float64)
            if target.shape != inference.prediction_md.shape:
                raise ValueError(
                    f"Target shape mismatch for lattice well {inference.well_id}"
                )
            if not np.isfinite(target).all():
                raise ValueError(f"Non-finite lattice target: {inference.well_id}")

            complete_tvt = np.concatenate(([inference.anchor], target))
            complete_md = np.concatenate(
                ([inference.anchor_md], inference.prediction_md)
            )
            rates = np.diff(complete_tvt) / np.diff(complete_md)
            transition_blocks.append(rates[np.isfinite(rates)])

            reference_gr = interpolate_typewell_gr(
                inference.typewell_tvt,
                inference.typewell_gr,
                target,
            )
            expected_gr = (
                inference.calibration.gr_intercept
                + inference.calibration.gr_slope * reference_gr
            )
            residual = inference.prediction_gr - expected_gr
            emission_blocks.append(residual[np.isfinite(residual)])

        transition_rates = np.concatenate(transition_blocks)
        if len(transition_rates) == 0:
            raise ValueError("Lattice fit found no finite training transitions")
        finite_emissions = [block for block in emission_blocks if len(block)]
        emission_residuals = (
            np.concatenate(finite_emissions)
            if finite_emissions
            else np.empty(0, dtype=np.float64)
        )
        statistics = fit_lattice_statistics(
            transition_rates, emission_residuals, config=self.config
        )
        return LatticeModelState(statistics=statistics, config=self.config)

    def predict_with_diagnostics(
        self, state: LatticeModelState, well: InferenceWell
    ) -> LatticeResult:
        if not isinstance(well, LatticeInferenceWell):
            raise TypeError("Lattice prediction requires LatticeInferenceWell")
        return infer_signed_tvt_lattice(
            well.prediction_md,
            well.prediction_gr,
            anchor_md=well.anchor_md,
            anchor_tvt=well.anchor,
            typewell_tvt=well.typewell_tvt,
            typewell_gr=well.typewell_gr,
            statistics=state.statistics,
            calibration=well.calibration,
            config=state.config,
        )

    def predict(
        self, state: LatticeModelState, well: InferenceWell
    ) -> np.ndarray:
        if not isinstance(well, LatticeInferenceWell):
            raise TypeError("Lattice prediction requires LatticeInferenceWell")
        prediction = infer_signed_tvt_posterior_mean(
            well.prediction_md,
            well.prediction_gr,
            anchor_md=well.anchor_md,
            anchor_tvt=well.anchor,
            typewell_tvt=well.typewell_tvt,
            typewell_gr=well.typewell_gr,
            statistics=state.statistics,
            calibration=well.calibration,
            config=state.config,
        )
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != (len(well.row_index),):
            raise ValueError(f"Lattice prediction shape mismatch: {well.well_id}")
        if not np.isfinite(prediction).all():
            raise ValueError(f"Lattice produced non-finite prediction: {well.well_id}")
        return prediction


def lattice_manifest(
    config: LatticeConfig = DEFAULT_LATTICE_CONFIG,
) -> dict[str, object]:
    """Machine-readable inference and leakage contract for the MVP."""

    return {
        "version": LATTICE_VERSION,
        "horizontal_source_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
        "horizontal_columns_used": ["MD", "GR", "TVT_input"],
        "typewell_source_columns": ["TVT", "GR"],
        "target_use": "outer-fold training suffix only, in fit",
        "state": "bounded physical TVT grid centered exactly on last-known anchor",
        "time": "horizontal MD",
        "transition": "banded signed Student-t rate factor; negative/zero/positive",
        "emission": "robust own-typewell GR likelihood with visible-prefix affine calibration",
        "decoder": "exact forward-backward row posterior mean (squared-loss MBR)",
        "diagnostics": [
            "Viterbi MAP path",
            "marginal MAP",
            "posterior entropy",
            "posterior variance",
            "state-grid edge mass",
            "log evidence",
        ],
        "config": asdict(config),
    }


__all__ = [
    "DEFAULT_LATTICE_CONFIG",
    "LATTICE_VERSION",
    "LatticeInferenceWell",
    "LatticeModelState",
    "ProbabilisticLatticeModel",
    "lattice_manifest",
]
