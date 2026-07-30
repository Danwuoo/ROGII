from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rogii.external_models.contracts import (
    FixedGridSequenceWell,
    SequenceModelInput,
)
from rogii.external_models.controls import (
    CONTROL_GR_ORDER_SHUFFLE,
    CONTROL_PAIR_SHUFFLE,
    CONTROL_PHASE_DESTROY,
    CONTROL_WRONG_TYPEWELL,
)
from rogii.external_models.data import (
    HORIZONTAL_FEATURE_NAMES,
    MODEL_INPUT_CHANNEL_NAMES,
    TYPEWELL_FEATURE_NAMES,
)
from rogii.external_models.ttm import (
    TTM_BACKBONE_PARAMETER_COUNT,
    TTM_PARAMETER_COUNT,
    freeze_backbone,
    load_ttm_forecaster,
    verify_ttm_checkpoint,
)
from rogii.external_models.ttm_embeddings import (
    TTMHeadConfig,
    TTMPairedInteractionHead,
    apply_ttm_control,
    encode_ttm_batch,
    fit_ttm_scratch_model,
    head_parameter_count,
    make_ttm_batch,
    predict_ttm_scratch_model,
)


WEIGHT_DIR = Path(
    ".runs/ttm-r2-transfer-probe-20260729/weights/"
    "ibm-granite--granite-timeseries-ttm-r2--d6a79570"
)


def synthetic_input(well_id: str, offset: float = 0.0) -> SequenceModelInput:
    coordinate = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    horizontal = np.column_stack(
        [
            np.linspace(0.0, 1000.0, 512),
            coordinate,
            coordinate * 2,
            -coordinate,
            np.sin(8 * coordinate) + offset,
            np.ones(512),
            np.linspace(100.0, 110.0, 512),
            np.r_[np.ones(160), np.zeros(352)],
            np.r_[np.zeros(160), np.ones(352)],
        ]
    ).astype(np.float32)
    typewell = np.column_stack(
        [
            np.linspace(90.0, 130.0, 512),
            np.cos(7 * coordinate) + offset,
            np.ones(512),
        ]
    ).astype(np.float32)
    sequence = FixedGridSequenceWell(
        well_id=well_id,
        row_index=np.arange(4, 12, dtype=np.int64),
        known_count=4,
        anchor_tvt=100.0,
        horizontal=horizontal,
        typewell=typewell,
        output_coordinate=np.linspace(0.0, 1.0, 8),
        horizontal_feature_names=HORIZONTAL_FEATURE_NAMES,
        typewell_feature_names=TYPEWELL_FEATURE_NAMES,
        source_horizontal_rows=12,
        source_typewell_rows=512,
        schema_version="test",
    )
    x = np.tile(coordinate, (len(MODEL_INPUT_CHANNEL_NAMES), 1))
    return SequenceModelInput(
        sequence=sequence,
        incumbent_prediction=np.linspace(100.0, 108.0, 8),
        x=x,
        grid_coordinate=coordinate,
        channel_names=MODEL_INPUT_CHANNEL_NAMES,
    )


def test_resampling_masks_and_controls_are_deterministic() -> None:
    inputs = [synthetic_input(f"well-{index}", index) for index in range(4)]
    original = make_ttm_batch(inputs)
    assert original.horizontal.shape == (4, 512, 1)
    assert original.typewell.shape == (4, 512, 1)
    assert original.prefix_mask.shape == (4, 1, 512)
    for control in (
        CONTROL_WRONG_TYPEWELL,
        CONTROL_PAIR_SHUFFLE,
        CONTROL_GR_ORDER_SHUFFLE,
        CONTROL_PHASE_DESTROY,
    ):
        first = apply_ttm_control(inputs, control_id=control, seed=20260729)
        second = apply_ttm_control(inputs, control_id=control, seed=20260729)
        first_batch = make_ttm_batch(first)
        second_batch = make_ttm_batch(second)
        np.testing.assert_array_equal(first_batch.horizontal, second_batch.horizontal)
        np.testing.assert_array_equal(first_batch.typewell, second_batch.typewell)
        assert first_batch.paired_typewell_ids == second_batch.paired_typewell_ids
    wrong = make_ttm_batch(
        apply_ttm_control(
            inputs,
            control_id=CONTROL_WRONG_TYPEWELL,
            seed=20260729,
        )
    )
    assert all(
        source != paired
        for source, paired in zip(wrong.ids, wrong.paired_typewell_ids, strict=True)
    )


def test_matched_head_parameter_count_and_gradient() -> None:
    random_config = TTMHeadConfig(
        variant_id="ttm_random_frozen",
        seed=20260729,
        epochs=2,
        batch_size=2,
    )
    pretrained_config = TTMHeadConfig(
        variant_id="ttm_pretrained_frozen",
        seed=20260729,
        epochs=2,
        batch_size=2,
    )
    assert head_parameter_count(
        random_config, bypass_channels=16
    ) == head_parameter_count(pretrained_config, bypass_channels=16)
    head = TTMPairedInteractionHead(bypass_channels=16)
    output = head(
        torch.randn(2, 8, 192),
        torch.randn(2, 8, 192),
        torch.randn(2, 16, 512),
        torch.ones(2, 1, 512),
    )
    assert output.shape == (2, 512)
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_adapter_is_explicit_and_has_separate_capacity() -> None:
    frozen = TTMHeadConfig(
        variant_id="ttm_pretrained_frozen",
        seed=20260729,
        epochs=2,
        batch_size=2,
    )
    adapter = TTMHeadConfig(
        variant_id="ttm_pretrained_adapter",
        seed=20260729,
        epochs=2,
        batch_size=2,
        adapter_rank=8,
    )
    assert head_parameter_count(adapter, bypass_channels=16) > head_parameter_count(
        frozen,
        bypass_channels=16,
    )
    with pytest.raises(ValueError, match="positive adapter_rank"):
        TTMHeadConfig(
            variant_id="ttm_pretrained_adapter",
            seed=20260729,
            epochs=2,
            batch_size=2,
        )


@pytest.mark.skipif(not WEIGHT_DIR.is_dir(), reason="pinned TTM cache is absent")
def test_random_pretrained_architecture_shape_and_frozen_gradients() -> None:
    identity = verify_ttm_checkpoint(WEIGHT_DIR)
    assert identity["weight_sha256"].startswith("a706726a")
    random_model = load_ttm_forecaster(
        WEIGHT_DIR,
        pretrained=False,
        device="cpu",
    )
    pretrained_model = load_ttm_forecaster(
        WEIGHT_DIR,
        pretrained=True,
        device="cpu",
    )
    assert sum(parameter.numel() for parameter in random_model.parameters()) == (
        TTM_PARAMETER_COUNT
    )
    assert sum(
        parameter.numel() for parameter in pretrained_model.backbone.parameters()
    ) == TTM_BACKBONE_PARAMETER_COUNT
    random_backbone = freeze_backbone(random_model)
    pretrained_backbone = freeze_backbone(pretrained_model)
    batch = make_ttm_batch(
        [synthetic_input("well-a"), synthetic_input("well-b", 1.0)]
    )
    random_h, random_t = encode_ttm_batch(
        random_backbone,
        batch,
        device=torch.device("cpu"),
        batch_size=2,
    )
    pretrained_h, pretrained_t = encode_ttm_batch(
        pretrained_backbone,
        batch,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert random_h.shape == pretrained_h.shape == (2, 8, 192)
    assert random_t.shape == pretrained_t.shape == (2, 8, 192)
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in pretrained_backbone.parameters()
    )


@pytest.mark.skipif(not WEIGHT_DIR.is_dir(), reason="pinned TTM cache is absent")
def test_scratch_trainable_capacity_control_smoke(tmp_path: Path) -> None:
    batch = make_ttm_batch(
        [synthetic_input("scratch-a"), synthetic_input("scratch-b", 1.0)]
    )
    config = TTMHeadConfig(
        variant_id="ttm_scratch_trainable",
        seed=20260729,
        epochs=1,
        batch_size=2,
    )
    forecaster = load_ttm_forecaster(
        WEIGHT_DIR,
        pretrained=False,
        device="cpu",
    )
    checkpoint = fit_ttm_scratch_model(
        forecaster=forecaster,
        train_batch=batch,
        targets=np.zeros((2, 512), dtype=np.float32),
        row_counts=np.asarray([8, 8], dtype=np.int64),
        validation_ids=("validation-only",),
        checkpoint_path=tmp_path / "scratch.pt",
        config=config,
        provenance={"fold": 0},
        device=torch.device("cpu"),
        resume=False,
        embedding_batch_size=2,
    )
    prediction = predict_ttm_scratch_model(
        forecaster=load_ttm_forecaster(
            WEIGHT_DIR,
            pretrained=False,
            device="cpu",
        ),
        batch=batch,
        checkpoint=checkpoint,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert checkpoint["completed"] is True
    assert checkpoint["trainable_parameter_count"] > TTM_BACKBONE_PARAMETER_COUNT
    assert prediction.shape == (2, 512)
    assert np.isfinite(prediction).all()
