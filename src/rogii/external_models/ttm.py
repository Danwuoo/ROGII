"""Pinned IBM Granite Tiny Time Mixer R2 identity and model helpers.

The forecasting head is never used as a TVT predictor.  Callers consume only
``TinyTimeMixerForPrediction.backbone(...).last_hidden_state`` and attach the
project-owned paired residual head.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Any, Iterator

import torch


TTM_REPOSITORY = "https://github.com/ibm-granite/granite-tsfm"
TTM_REPOSITORY_COMMIT = "31c7a6bd640c41e53777c0afe3a13d24e00f7fe7"
TTM_PACKAGE = "granite-tsfm"
TTM_PACKAGE_VERSION = "0.3.6"
TTM_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
TTM_MODEL_REVISION = "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f"
TTM_WEIGHT_FILE = "model.safetensors"
TTM_WEIGHT_SHA256 = (
    "a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710"
)
TTM_WEIGHT_BYTES = 3_240_592
TTM_PARAMETER_COUNT = 805_280
TTM_BACKBONE_PARAMETER_COUNT = 515_584
TTM_CONTEXT_LENGTH = 512
TTM_PREDICTION_LENGTH = 96
TTM_PATCH_LENGTH = 64
TTM_PATCH_COUNT = 8
TTM_D_MODEL = 192
TTM_NUM_LAYERS = 2
TTM_ADAPTIVE_PATCH_LEVELS = 3
TTM_RANDOM_BACKBONE_SEED = 2026072901
TTM_ALLOWED_FILES = (
    "README.md",
    "config.json",
    "generation_config.json",
    TTM_WEIGHT_FILE,
)

EXPECTED_CONFIG = {
    "adaptive_patching_levels": TTM_ADAPTIVE_PATCH_LEVELS,
    "context_length": TTM_CONTEXT_LENGTH,
    "d_model": TTM_D_MODEL,
    "num_layers": TTM_NUM_LAYERS,
    "num_input_channels": 1,
    "num_patches": TTM_PATCH_COUNT,
    "patch_length": TTM_PATCH_LENGTH,
    "patch_stride": TTM_PATCH_LENGTH,
    "prediction_length": TTM_PREDICTION_LENGTH,
    "resolution_prefix_tuning": False,
    "scaling": "std",
}


class TTMIdentityError(RuntimeError):
    """Raised when code, package, config, or weights differ from the lock."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_weight_dir(cache_dir: str | Path) -> Path:
    return (
        Path(cache_dir)
        / "weights"
        / "ibm-granite--granite-timeseries-ttm-r2--d6a79570"
    )


def _installed_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise TTMIdentityError(
            f"{name} is not installed; use the pinned TTM environment"
        ) from exc


def verify_ttm_checkpoint(weight_dir: str | Path) -> dict[str, Any]:
    """Verify the exact local config and safetensors bytes without network I/O."""

    root = Path(weight_dir).resolve()
    config_path = root / "config.json"
    weight_path = root / TTM_WEIGHT_FILE
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(
            f"TTM checkpoint requires config.json and {TTM_WEIGHT_FILE}: {root}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TTMIdentityError(f"invalid TTM config: {config_path}") from exc
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise TTMIdentityError(f"TTM config differs from lock: {mismatches}")
    actual_bytes = weight_path.stat().st_size
    actual_sha256 = sha256_file(weight_path)
    if actual_bytes != TTM_WEIGHT_BYTES or actual_sha256 != TTM_WEIGHT_SHA256:
        raise TTMIdentityError(
            "TTM weights differ from lock: "
            f"bytes={actual_bytes}, sha256={actual_sha256}"
        )
    package_version = _installed_version(TTM_PACKAGE)
    if package_version != TTM_PACKAGE_VERSION:
        raise TTMIdentityError(
            f"{TTM_PACKAGE} version differs: {package_version} != "
            f"{TTM_PACKAGE_VERSION}"
        )
    return {
        "model_id": TTM_MODEL_ID,
        "revision": TTM_MODEL_REVISION,
        "repository": TTM_REPOSITORY,
        "repository_commit": TTM_REPOSITORY_COMMIT,
        "package": TTM_PACKAGE,
        "package_version": package_version,
        "license": "Apache-2.0",
        "weight_path": str(weight_path),
        "weight_bytes": actual_bytes,
        "weight_sha256": actual_sha256,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "expected_config": dict(EXPECTED_CONFIG),
    }


def download_ttm_checkpoint(
    cache_dir: str | Path,
    *,
    weight_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Download only the four allowlisted files at the immutable revision."""

    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = (
        default_weight_dir(cache)
        if weight_dir is None
        else Path(weight_dir).resolve()
    )
    if destination == cache or cache in destination.parents:
        pass
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=TTM_MODEL_ID,
        revision=TTM_MODEL_REVISION,
        local_dir=destination,
        allow_patterns=list(TTM_ALLOWED_FILES),
        cache_dir=cache / "hf-hub",
    )
    unexpected = sorted(
        str(path.relative_to(destination)).replace("\\", "/")
        for path in destination.rglob("*")
        if path.is_file()
        and ".cache" not in path.parts
        and path.name not in TTM_ALLOWED_FILES
    )
    if unexpected:
        raise TTMIdentityError(f"unexpected files in TTM checkpoint: {unexpected}")
    return verify_ttm_checkpoint(destination)


def configure_offline_environment(cache_dir: str | Path) -> dict[str, str]:
    cache = Path(cache_dir).resolve()
    values = {
        "HF_HOME": str(cache / "hf-home"),
        "HF_HUB_CACHE": str(cache / "hf-home" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    os.environ.update(values)
    return values


@contextmanager
def forbid_network(enabled: bool = True) -> Iterator[None]:
    """Fail closed if an offline smoke attempts DNS or a socket connection."""

    if not enabled:
        yield
        return
    original_create = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access is forbidden in offline TTM execution")

    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    socket.socket.connect = blocked
    try:
        yield
    finally:
        socket.create_connection = original_create
        socket.getaddrinfo = original_getaddrinfo
        socket.socket.connect = original_connect


def resolve_device(name: str) -> torch.device:
    requested = str(name).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(
        "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
    )


def _model_classes() -> tuple[type[Any], type[Any]]:
    try:
        from tsfm_public.models.tinytimemixer import (
            TinyTimeMixerConfig,
            TinyTimeMixerForPrediction,
        )
    except ImportError as exc:
        raise TTMIdentityError(
            "granite-tsfm imports failed; activate the pinned optional environment"
        ) from exc
    return TinyTimeMixerConfig, TinyTimeMixerForPrediction


def load_ttm_forecaster(
    weight_dir: str | Path,
    *,
    pretrained: bool,
    device: str | torch.device,
    random_seed: int = TTM_RANDOM_BACKBONE_SEED,
) -> torch.nn.Module:
    """Load pretrained bytes or official same-config random initialization."""

    identity = verify_ttm_checkpoint(weight_dir)
    config_class, model_class = _model_classes()
    root = Path(identity["weight_path"]).parent
    config = config_class.from_pretrained(root, local_files_only=True)
    if pretrained:
        model = model_class.from_pretrained(
            root,
            config=config,
            local_files_only=True,
        )
    else:
        devices = [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(random_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(random_seed))
            model = model_class(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    backbone = sum(parameter.numel() for parameter in model.backbone.parameters())
    if total != TTM_PARAMETER_COUNT or backbone != TTM_BACKBONE_PARAMETER_COUNT:
        raise TTMIdentityError(
            f"TTM parameter counts differ: total={total}, backbone={backbone}"
        )
    return model.to(resolve_device(str(device)) if isinstance(device, str) else device)


def freeze_backbone(model: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(model, "backbone"):
        raise TypeError("TTM forecaster has no backbone")
    backbone = model.backbone
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return backbone


def environment_manifest() -> dict[str, Any]:
    packages = (
        "granite-tsfm",
        "transformers",
        "huggingface-hub",
        "safetensors",
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


__all__ = [
    "EXPECTED_CONFIG",
    "TTM_ADAPTIVE_PATCH_LEVELS",
    "TTM_ALLOWED_FILES",
    "TTM_BACKBONE_PARAMETER_COUNT",
    "TTM_CONTEXT_LENGTH",
    "TTM_D_MODEL",
    "TTMIdentityError",
    "TTM_MODEL_ID",
    "TTM_MODEL_REVISION",
    "TTM_NUM_LAYERS",
    "TTM_PACKAGE_VERSION",
    "TTM_PARAMETER_COUNT",
    "TTM_PATCH_COUNT",
    "TTM_PATCH_LENGTH",
    "TTM_PREDICTION_LENGTH",
    "TTM_RANDOM_BACKBONE_SEED",
    "TTM_REPOSITORY_COMMIT",
    "TTM_WEIGHT_BYTES",
    "TTM_WEIGHT_SHA256",
    "configure_offline_environment",
    "default_weight_dir",
    "download_ttm_checkpoint",
    "environment_manifest",
    "forbid_network",
    "freeze_backbone",
    "load_ttm_forecaster",
    "resolve_device",
    "sha256_file",
    "verify_ttm_checkpoint",
]
