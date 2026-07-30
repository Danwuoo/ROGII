"""Observation-likelihood facade for the locked prefix calibration."""

from __future__ import annotations

import pandas as pd

from .contracts import PrefixCalibrationConfig, PrefixLikelihoodResult
from .prefix_calibration import calibrate_prefix_likelihood


def prefix_observation_likelihood(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    well_id: str,
    reference_well_id: str,
    control_id: str,
    config: PrefixCalibrationConfig,
) -> PrefixLikelihoodResult:
    return calibrate_prefix_likelihood(
        horizontal,
        typewell,
        well_id=well_id,
        reference_well_id=reference_well_id,
        control_id=control_id,
        config=config,
    )
