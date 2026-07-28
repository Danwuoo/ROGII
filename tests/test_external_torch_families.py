import json

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from rogii.external_models.torch_families import (  # noqa: E402
    FAMILY_FNO1D,
    FAMILY_PAIRED_PATCH,
    FAMILY_S4D,
    FAMILY_TCN,
    TorchFamilyConfig,
    TorchFamilyError,
    fit_torch_family,
    load_torch_family,
)


CHANNELS = (
    "horizontal:MD",
    "horizontal:GR",
    "typewell:TVT",
    "typewell:GR",
)


def _toy_arrays(
    samples: int = 9, length: int = 32
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    coordinate = np.linspace(0.0, 1.0, length, dtype=np.float32)
    x = np.empty((samples, len(CHANNELS), length), dtype=np.float32)
    y = np.empty((samples, 3), dtype=np.float32)
    for sample in range(samples):
        phase = 0.15 * sample
        x[sample, 0] = coordinate
        x[sample, 1] = np.sin(2.0 * np.pi * coordinate + phase)
        x[sample, 2] = 20.0 + 3.0 * coordinate
        x[sample, 3] = np.sin(2.0 * np.pi * coordinate + phase + 0.2)
        y[sample, 0] = 0.1 * sample
        y[sample, 1] = float(x[sample, 1, -1] - x[sample, 3, -1])
        y[sample, 2] = float(np.mean(x[sample, 1] * x[sample, 3]))
    ids = tuple(f"well-{sample}" for sample in range(samples))
    row_counts = np.arange(1, samples + 1, dtype=np.int64)
    return x, y, ids, row_counts


def _config(family: str, **changes: object) -> TorchFamilyConfig:
    values: dict[str, object] = {
        "family": family,
        "seed": 17,
        "device": "cpu",
        "epochs": 2,
        "learning_rate": 5.0e-3,
        "hidden_dim": 8,
        "layers": 1,
        "patch_sizes": (4, 8),
        "attention_heads": 2,
        "fno_modes": 5,
    }
    values.update(changes)
    return TorchFamilyConfig(**values)


@pytest.mark.parametrize(
    "family",
    [FAMILY_PAIRED_PATCH, FAMILY_S4D, FAMILY_FNO1D, FAMILY_TCN],
)
def test_all_tiny_families_fit_predict_and_manifest(family: str) -> None:
    x, y, ids, row_counts = _toy_arrays()
    state = fit_torch_family(
        x,
        y,
        channel_names=CHANNELS,
        ids=ids,
        row_counts=row_counts,
        config=_config(family),
    )

    prediction = state.predict(x, channel_names=CHANNELS, ids=ids)
    assert prediction.shape == y.shape
    assert prediction.dtype == np.float64
    assert np.isfinite(prediction).all()

    manifest = state.manifest
    json.dumps(manifest)
    assert manifest["family"] == family
    assert manifest["architecture_only"] is True
    assert manifest["fit_device"] == "cpu"
    assert manifest["inference_device"] == "cpu"
    assert manifest["sample_count"] == len(x)
    assert manifest["row_count_sum"] == int(row_counts.sum())
    assert manifest["parameters_total"] > 0
    assert manifest["parameters_trainable"] > 0
    assert len(manifest["parameter_sha256"]) == 64


def test_row_count_weighted_fold_local_scaling_and_frozen_control() -> None:
    x, y, ids, _ = _toy_arrays(samples=3)
    row_counts = np.array([1, 2, 7], dtype=np.int64)
    state = fit_torch_family(
        x,
        y,
        channel_names=CHANNELS,
        ids=ids,
        row_counts=row_counts,
        config=_config(FAMILY_TCN, frozen_random_encoder=True),
    )

    weights = row_counts.astype(np.float64)
    expected_input_mean = np.einsum("n,ncl->c", weights, x.astype(np.float64)) / (
        weights.sum() * x.shape[2]
    )
    expected_target_mean = (
        np.einsum("n,nd->d", weights, y.astype(np.float64)) / weights.sum()
    )
    np.testing.assert_allclose(state.input_mean, expected_input_mean, rtol=1e-6)
    np.testing.assert_allclose(state.target_mean, expected_target_mean, rtol=1e-6)
    assert state.manifest["frozen_random_encoder"] is True
    assert state.manifest["encoder_parameters_total"] > 0
    assert state.manifest["encoder_parameters_trainable"] == 0
    assert state.manifest["head_parameters_trainable"] > 0


def test_seed_reproducibility_and_explicit_no_cross_ablation() -> None:
    x, y, ids, row_counts = _toy_arrays(samples=5)
    cross_config = _config(FAMILY_PAIRED_PATCH, cross_series=True)
    first = fit_torch_family(
        x,
        y,
        channel_names=CHANNELS,
        ids=ids,
        row_counts=row_counts,
        config=cross_config,
    )
    second = fit_torch_family(
        x,
        y,
        channel_names=CHANNELS,
        ids=ids,
        row_counts=row_counts,
        config=cross_config,
    )
    np.testing.assert_array_equal(
        first.predict(x, channel_names=CHANNELS, ids=ids),
        second.predict(x, channel_names=CHANNELS, ids=ids),
    )
    assert first.manifest["parameter_sha256"] == second.manifest["parameter_sha256"]

    no_cross = fit_torch_family(
        x,
        y,
        channel_names=CHANNELS,
        ids=ids,
        row_counts=row_counts,
        config=_config(FAMILY_PAIRED_PATCH, cross_series=False),
    )
    assert first.manifest["variant"] == "cross"
    assert no_cross.manifest["variant"] == "no_cross"
    assert first.manifest["parameters_total"] == no_cross.manifest["parameters_total"]
    assert not np.array_equal(
        first.predict(x, channel_names=CHANNELS, ids=ids),
        no_cross.predict(x, channel_names=CHANNELS, ids=ids),
    )


def test_cpu_serialization_round_trip_and_fail_closed_validation(
    tmp_path,
) -> None:
    x, y, ids, row_counts = _toy_arrays(samples=5)
    state = fit_torch_family(
        x,
        y,
        channel_names=CHANNELS,
        ids=ids,
        row_counts=row_counts,
        config=_config(FAMILY_FNO1D),
    )
    expected = state.predict(x, channel_names=CHANNELS, ids=ids)
    path = state.save(tmp_path / "fno-state.pt")
    loaded = load_torch_family(path)
    actual = loaded.predict(x, channel_names=CHANNELS, ids=ids)
    np.testing.assert_array_equal(actual, expected)
    assert next(loaded.model.parameters()).device.type == "cpu"

    with pytest.raises(ValueError, match="finite"):
        invalid = x.copy()
        invalid[0, 0, 0] = np.nan
        loaded.predict(invalid, channel_names=CHANNELS, ids=ids)
    with pytest.raises(ValueError, match="channel_names/order"):
        loaded.predict(
            x,
            channel_names=tuple(reversed(CHANNELS)),
            ids=ids,
        )
    with pytest.raises(ValueError, match="unique"):
        loaded.predict(
            x,
            channel_names=CHANNELS,
            ids=("duplicate",) * len(ids),
        )

    with torch.no_grad():
        next(loaded.model.parameters()).fill_(float("nan"))
    with pytest.raises(TorchFamilyError, match="non-finite"):
        loaded.predict(x, channel_names=CHANNELS, ids=ids)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch state")
    with pytest.raises(TorchFamilyError, match="failed to load"):
        load_torch_family(corrupt)


def test_pair_schema_and_fit_inputs_fail_closed() -> None:
    x, y, ids, row_counts = _toy_arrays(samples=4)
    with pytest.raises(ValueError, match="prefixes"):
        fit_torch_family(
            x,
            y,
            channel_names=("MD", "GR", "TVT", "type GR"),
            ids=ids,
            row_counts=row_counts,
            config=_config(FAMILY_PAIRED_PATCH),
        )
    with pytest.raises(ValueError, match="positive integers"):
        fit_torch_family(
            x,
            y,
            channel_names=CHANNELS,
            ids=ids,
            row_counts=np.array([1, 2, 0, 4]),
            config=_config(FAMILY_TCN),
        )
    with pytest.raises(ValueError, match="finite"):
        bad_targets = y.copy()
        bad_targets[0, 0] = np.inf
        fit_torch_family(
            x,
            bad_targets,
            channel_names=CHANNELS,
            ids=ids,
            row_counts=row_counts,
            config=_config(FAMILY_TCN),
        )
