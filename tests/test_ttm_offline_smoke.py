from __future__ import annotations

from pathlib import Path
import socket

import pytest

from rogii.external_models.artifacts import capture_data_manifest
from rogii.external_models.ttm import (
    forbid_network,
    freeze_backbone,
    load_ttm_forecaster,
    verify_ttm_checkpoint,
)


WEIGHT_DIR = Path(
    ".runs/ttm-r2-transfer-probe-20260729/weights/"
    "ibm-granite--granite-timeseries-ttm-r2--d6a79570"
)
DATA_ROOT = Path("data/rogii-wellbore-geology-prediction")


def test_forced_offline_guard_blocks_network() -> None:
    with forbid_network(True), pytest.raises(RuntimeError, match="forbidden"):
        socket.create_connection(("example.com", 443))


@pytest.mark.skipif(not WEIGHT_DIR.is_dir(), reason="pinned TTM cache is absent")
def test_cached_offline_reload_cpu_and_data_manifest_unchanged() -> None:
    before = capture_data_manifest(DATA_ROOT)
    identity = verify_ttm_checkpoint(WEIGHT_DIR)
    with forbid_network(True):
        model = load_ttm_forecaster(
            WEIGHT_DIR,
            pretrained=True,
            device="cpu",
        )
    backbone = freeze_backbone(model)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert identity["weight_sha256"] == (
        "a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710"
    )
    assert capture_data_manifest(DATA_ROOT) == before
