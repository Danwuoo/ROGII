from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rogii.external_models.ttm_embeddings import (
    FoldLocalTTMScaler,
    TTMHeadConfig,
    fit_ttm_head,
    make_ttm_batch,
    predict_ttm_head,
)
from test_ttm_embeddings import synthetic_input


def test_fold_local_scaler_rejects_validation_fit_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FoldLocalTTMScaler.fit(
            np.zeros((2, 8, 192), dtype=np.float32),
            np.zeros((2, 8, 192), dtype=np.float32),
            np.zeros((2, 16, 512), dtype=np.float32),
            fit_ids=("a", "b"),
            validation_ids=("b",),
        )


def test_checkpoint_resume_is_exact_and_records_fit_ids(tmp_path: Path) -> None:
    batch = make_ttm_batch(
        [synthetic_input("train-a"), synthetic_input("train-b", 1.0)]
    )
    random = np.random.default_rng(5)
    horizontal = random.normal(size=(2, 8, 192)).astype(np.float32)
    typewell = random.normal(size=(2, 8, 192)).astype(np.float32)
    targets = random.normal(size=(2, 512)).astype(np.float32)
    counts = np.asarray([8, 9], dtype=np.int64)
    config = TTMHeadConfig(
        variant_id="ttm_pretrained_frozen",
        seed=20260729,
        epochs=3,
        batch_size=1,
    )
    checkpoint = tmp_path / "resume.pt"
    first = fit_ttm_head(
        train_horizontal=horizontal,
        train_typewell=typewell,
        train_batch=batch,
        targets=targets,
        row_counts=counts,
        validation_ids=("validation-only",),
        checkpoint_path=checkpoint,
        config=config,
        provenance={"fold": 0},
        device=torch.device("cpu"),
        resume=False,
        max_epochs_this_call=1,
    )
    assert first["epoch"] == 1
    resumed = fit_ttm_head(
        train_horizontal=horizontal,
        train_typewell=typewell,
        train_batch=batch,
        targets=targets,
        row_counts=counts,
        validation_ids=("validation-only",),
        checkpoint_path=checkpoint,
        config=config,
        provenance={"fold": 0},
        device=torch.device("cpu"),
        resume=True,
    )
    prediction = predict_ttm_head(
        horizontal=horizontal,
        typewell=typewell,
        batch=batch,
        checkpoint=resumed,
        device=torch.device("cpu"),
        batch_size=1,
    )
    fresh_path = tmp_path / "fresh.pt"
    fresh = fit_ttm_head(
        train_horizontal=horizontal,
        train_typewell=typewell,
        train_batch=batch,
        targets=targets,
        row_counts=counts,
        validation_ids=("validation-only",),
        checkpoint_path=fresh_path,
        config=config,
        provenance={"fold": 0},
        device=torch.device("cpu"),
        resume=False,
    )
    fresh_prediction = predict_ttm_head(
        horizontal=horizontal,
        typewell=typewell,
        batch=batch,
        checkpoint=fresh,
        device=torch.device("cpu"),
        batch_size=1,
    )
    np.testing.assert_array_equal(prediction, fresh_prediction)
    assert resumed["provenance"]["fit_ids"] == ["train-a", "train-b"]
    assert resumed["provenance"]["validation_ids"] == ["validation-only"]
    assert resumed["provenance"]["outer_validation_labels_used_for_fit"] is False
