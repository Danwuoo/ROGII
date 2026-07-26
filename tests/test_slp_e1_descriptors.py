from __future__ import annotations

import inspect

import numpy as np

from rogii.stratigraphic.descriptors import (
    DescriptorConfig,
    build_descriptor_batch,
    resample_gr_window,
)


def _descriptor(signal: np.ndarray):
    coordinate = np.arange(signal.size, dtype=np.float64)
    return build_descriptor_batch(
        coordinate,
        signal,
        center_rows=(signal.size // 2,),
        half_span=16.0,
    )


def test_raw_shape_is_robust_to_amplitude_shift_and_scale() -> None:
    x = np.arange(65, dtype=np.float64)
    signal = 70.0 + np.sin(x / 4.0) + 0.25 * np.cos(x / 2.0)
    baseline = _descriptor(signal).raw_shape
    shifted = _descriptor(signal + 400.0).raw_shape
    scaled = _descriptor(7.5 * signal - 13.0).raw_shape
    np.testing.assert_allclose(baseline, shifted, atol=1e-12)
    np.testing.assert_allclose(baseline, scaled, atol=1e-12)


def test_reversed_sequence_changes_order_sensitive_features() -> None:
    x = np.arange(65, dtype=np.float64)
    signal = np.sin(x / 3.5) + 0.03 * x
    forward = _descriptor(signal)
    reverse = _descriptor(signal[::-1])
    assert np.linalg.norm(forward.raw_shape - reverse.raw_shape) > 1.0
    assert np.linalg.norm(forward.barcode - reverse.barcode) > 1.0


def test_derivative_rank_and_barcode_shapes_are_finite() -> None:
    x = np.arange(100, dtype=np.float64)
    batch = build_descriptor_batch(
        x,
        np.sin(x / 7.0),
        center_rows=range(20, 80),
        half_span=16.0,
    )
    assert batch.derivative.shape == (60, 63)
    assert batch.rank.shape == (60, 33)
    assert batch.barcode.shape[0] == 60
    assert batch.barcode.shape[1] > 100
    assert np.isfinite(batch.derivative).all()
    assert np.isfinite(batch.rank).all()
    assert np.isfinite(batch.barcode).all()


def test_flat_window_is_safely_marked_low_information() -> None:
    window = resample_gr_window(
        np.arange(100, dtype=np.float64),
        np.full(100, 42.0),
        center_row=50,
        half_span=16.0,
    )
    assert window.eligible
    assert window.low_information
    np.testing.assert_array_equal(window.values, 0.0)


def test_missing_fraction_over_limit_is_ineligible() -> None:
    x = np.arange(100, dtype=np.float64)
    signal = np.sin(x)
    signal[35:55] = np.nan
    window = resample_gr_window(
        x, signal, center_row=50, half_span=16.0
    )
    assert not window.eligible
    assert window.observed_fraction < 0.80


def test_short_gap_interpolates_but_long_gap_is_rejected() -> None:
    x = np.arange(120, dtype=np.float64)
    signal = np.sin(x / 5.0)
    short = signal.copy()
    short[58:61] = np.nan
    long = signal.copy()
    long[52:63] = np.nan
    assert resample_gr_window(
        x, short, center_row=60, half_span=16.0
    ).eligible
    assert not resample_gr_window(
        x, long, center_row=60, half_span=16.0
    ).eligible


def test_uneven_coordinate_grid_resamples_to_33_points() -> None:
    step = np.where(np.arange(100) % 3 == 0, 0.7, 1.3)
    coordinate = np.cumsum(step)
    signal = np.sin(coordinate / 8.0)
    center = 50
    window = resample_gr_window(
        coordinate,
        signal,
        center_row=center,
        half_span=12.0,
    )
    assert window.eligible
    assert window.values.shape == (33,)


def test_descriptor_api_accepts_no_label_or_formation_fields() -> None:
    parameters = set(inspect.signature(build_descriptor_batch).parameters)
    assert parameters == {
        "coordinate",
        "gr",
        "center_rows",
        "half_span",
        "region_start",
        "region_stop",
        "config",
    }
    source = inspect.getsource(build_descriptor_batch)
    assert "target" not in source.lower()
    assert not any(
        field in source
        for field in ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
    )


def test_boundary_extrapolation_is_ineligible() -> None:
    window = resample_gr_window(
        np.arange(100, dtype=np.float64),
        np.sin(np.arange(100) / 4.0),
        center_row=5,
        half_span=16.0,
        config=DescriptorConfig(),
    )
    assert not window.eligible
