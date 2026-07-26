"""Target-bearing research diagnostics kept separate from deployable models."""

from .slp_e0 import (
    accumulate_pooled_metric,
    accumulate_recurrence_diagnostic,
    fit_continuous_piecewise,
    fit_discontinuous_piecewise,
    fit_polynomial_oracle,
    nearest_prefix_distance,
    run_slp_e0_diagnostic,
    surface_identity_stats,
)

__all__ = [
    "accumulate_pooled_metric",
    "accumulate_recurrence_diagnostic",
    "fit_continuous_piecewise",
    "fit_discontinuous_piecewise",
    "fit_polynomial_oracle",
    "nearest_prefix_distance",
    "run_slp_e0_diagnostic",
    "surface_identity_stats",
]
"""Research diagnostics that are not deployable competition models."""
