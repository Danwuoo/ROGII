"""Tiny, fold-local PyTorch sequence-family controls.

This module deliberately has no dependency on the higher-level external-model
contracts.  It accepts a fixed-grid NumPy tensor ``x`` with shape ``[N, C, L]``
and a matrix target with shape ``[N, D]``.  All learned statistics, including
input and target scaling, are fitted inside :func:`fit_torch_family`.

The implementations are architecture-only controls, not verbatim reproductions
of the research models whose inductive biases they exercise:

* ``paired_patch_cross``: multi-resolution patch transformers with an explicit
  horizontal/typewell cross-attention path and a matched no-cross ablation.
* ``s4d``: diagonal, long causal convolutions evaluated with an FFT.
* ``fno1d``: low-mode Fourier operator blocks with a coordinate channel.
* ``tcn``: a compact causal dilated-convolution control.

The fitted state always returns to CPU, validates every inference input, raises
on non-finite state or output, and serializes only tensors and primitive
metadata.  There is no silent identity fallback: a caller can safely choose the
incumbent path after catching :class:`TorchFamilyError`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


STATE_SCHEMA_VERSION = "rogii_torch_family_v1"
FAMILY_PAIRED_PATCH = "paired_patch_cross"
FAMILY_S4D = "s4d"
FAMILY_FNO1D = "fno1d"
FAMILY_TCN = "tcn"
TORCH_FAMILIES = (
    FAMILY_PAIRED_PATCH,
    FAMILY_S4D,
    FAMILY_FNO1D,
    FAMILY_TCN,
)


class TorchFamilyError(RuntimeError):
    """Raised when a family cannot fit or predict without unsafe output."""


@dataclass(frozen=True)
class TorchFamilyConfig:
    """Small deterministic training and architecture configuration."""

    family: str
    seed: int = 0
    device: str = "cpu"
    epochs: int = 60
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip: float = 5.0
    hidden_dim: int = 16
    layers: int = 2
    patch_sizes: tuple[int, ...] = (8, 16, 32)
    attention_heads: int = 2
    fno_modes: int = 12
    tcn_kernel_size: int = 3
    cross_series: bool = True
    frozen_random_encoder: bool = False
    horizontal_channels: tuple[str, ...] = ()
    typewell_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        family = str(self.family)
        if family not in TORCH_FAMILIES:
            raise ValueError(f"family must be one of {TORCH_FAMILIES}")
        seed = int(self.seed)
        if isinstance(self.seed, bool) or seed != self.seed or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        device = str(self.device)
        try:
            torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid torch device: {device!r}") from exc
        if not device:
            raise ValueError("device must not be empty")

        for name, value, lower in (
            ("epochs", self.epochs, 1),
            ("hidden_dim", self.hidden_dim, 4),
            ("layers", self.layers, 1),
            ("attention_heads", self.attention_heads, 1),
            ("fno_modes", self.fno_modes, 1),
            ("tcn_kernel_size", self.tcn_kernel_size, 2),
        ):
            integer = int(value)
            if isinstance(value, bool) or integer != value or integer < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
            object.__setattr__(self, name, integer)

        hidden = int(self.hidden_dim)
        heads = int(self.attention_heads)
        if hidden % heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if int(self.layers) > 6:
            raise ValueError("layers must be <= 6 for the small-family harness")

        patches = tuple(int(value) for value in self.patch_sizes)
        if (
            not patches
            or len(set(patches)) != len(patches)
            or any(value < 2 for value in patches)
        ):
            raise ValueError("patch_sizes must be unique integers >= 2")
        object.__setattr__(self, "patch_sizes", patches)

        for name, value in (
            ("learning_rate", self.learning_rate),
            ("gradient_clip", self.gradient_clip),
        ):
            scalar = float(value)
            if not math.isfinite(scalar) or scalar <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, scalar)
        decay = float(self.weight_decay)
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        object.__setattr__(self, "weight_decay", decay)

        if not isinstance(self.cross_series, bool):
            raise TypeError("cross_series must be bool")
        if not isinstance(self.frozen_random_encoder, bool):
            raise TypeError("frozen_random_encoder must be bool")

        horizontal = tuple(map(str, self.horizontal_channels))
        typewell = tuple(map(str, self.typewell_channels))
        if bool(horizontal) != bool(typewell):
            raise ValueError(
                "horizontal_channels and typewell_channels must both be set or empty"
            )
        if (
            len(set(horizontal)) != len(horizontal)
            or len(set(typewell)) != len(typewell)
            or set(horizontal).intersection(typewell)
            or any(not value for value in (*horizontal, *typewell))
        ):
            raise ValueError("paired channel groups must be non-empty and disjoint")
        object.__setattr__(self, "horizontal_channels", horizontal)
        object.__setattr__(self, "typewell_channels", typewell)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "device", device)


def available_torch_families() -> tuple[str, ...]:
    """Return the stable family identifiers accepted by the harness."""

    return TORCH_FAMILIES


def _validated_channels(channel_names: Sequence[str], expected: int) -> tuple[str, ...]:
    names = tuple(map(str, channel_names))
    if (
        len(names) != expected
        or len(set(names)) != len(names)
        or any(not value for value in names)
    ):
        raise ValueError(
            "channel_names must be unique, non-empty, and match x.shape[1]"
        )
    return names


def _validated_ids(ids: Sequence[str], expected: int) -> tuple[str, ...]:
    values = tuple(map(str, ids))
    if (
        len(values) != expected
        or len(set(values)) != len(values)
        or any(not value for value in values)
    ):
        raise ValueError("ids must be unique, non-empty, and match x.shape[0]")
    return values


def _validated_x(x: Any) -> np.ndarray:
    try:
        result = np.array(x, dtype=np.float32, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("x must be convertible to a float32 array") from exc
    if result.ndim != 3 or min(result.shape) < 1 or result.shape[2] < 4:
        raise ValueError("x must have shape [N, C, L] with L >= 4")
    if not np.isfinite(result).all():
        raise ValueError("x must contain only finite values")
    return result


def _validated_targets(targets: Any, expected_rows: int) -> np.ndarray:
    try:
        result = np.array(targets, dtype=np.float32, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("targets must be convertible to a float32 array") from exc
    if result.ndim != 2 or result.shape[0] != expected_rows or result.shape[1] < 1:
        raise ValueError("targets must have shape [N, D] with D >= 1")
    if not np.isfinite(result).all():
        raise ValueError("targets must contain only finite values")
    return result


def _validated_row_counts(row_counts: Any, expected: int) -> np.ndarray:
    if row_counts is None:
        return np.ones(expected, dtype=np.int64)
    try:
        raw = np.asarray(row_counts, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("row_counts must be numeric") from exc
    if (
        raw.ndim != 1
        or len(raw) != expected
        or not np.isfinite(raw).all()
        or np.any(raw < 1.0)
        or not np.array_equal(raw, np.floor(raw))
    ):
        raise ValueError("row_counts must be positive integers aligned with ids")
    return raw.astype(np.int64)


def _weighted_scalers(
    x: np.ndarray, targets: np.ndarray, row_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = row_counts.astype(np.float64)
    denominator_x = float(weights.sum() * x.shape[2])
    x64 = x.astype(np.float64)
    input_mean = np.einsum("n,ncl->c", weights, x64, optimize=True) / denominator_x
    input_variance = (
        np.einsum(
            "n,ncl->c",
            weights,
            np.square(x64 - input_mean[None, :, None]),
            optimize=True,
        )
        / denominator_x
    )
    target64 = targets.astype(np.float64)
    denominator_y = float(weights.sum())
    target_mean = np.einsum("n,nd->d", weights, target64, optimize=True) / denominator_y
    target_variance = (
        np.einsum(
            "n,nd->d",
            weights,
            np.square(target64 - target_mean[None, :]),
            optimize=True,
        )
        / denominator_y
    )

    input_scale = np.sqrt(np.maximum(input_variance, 0.0))
    target_scale = np.sqrt(np.maximum(target_variance, 0.0))
    input_scale[input_scale < 1.0e-6] = 1.0
    target_scale[target_scale < 1.0e-6] = 1.0
    arrays = tuple(
        np.asarray(value, dtype=np.float32)
        for value in (input_mean, input_scale, target_mean, target_scale)
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise TorchFamilyError("fold-local scaling produced non-finite values")
    return arrays  # type: ignore[return-value]


def _pair_role(name: str) -> str | None:
    normalized = name.casefold().replace("-", "_")
    horizontal_prefixes = (
        "horizontal:",
        "horizontal/",
        "horizontal_",
        "h:",
        "h/",
    )
    typewell_prefixes = (
        "typewell:",
        "typewell/",
        "typewell_",
        "tw:",
        "tw/",
    )
    if normalized.startswith(horizontal_prefixes):
        return "horizontal"
    if normalized.startswith(typewell_prefixes):
        return "typewell"
    return None


def _paired_channel_indices(
    channel_names: tuple[str, ...], config: TorchFamilyConfig
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if config.horizontal_channels:
        expected = set(config.horizontal_channels) | set(config.typewell_channels)
        actual = set(channel_names)
        if expected != actual:
            missing = sorted(expected - actual)
            unassigned = sorted(actual - expected)
            raise ValueError(
                "explicit paired groups must cover channel_names exactly; "
                f"missing={missing}, unassigned={unassigned}"
            )
        horizontal = tuple(
            index
            for index, name in enumerate(channel_names)
            if name in config.horizontal_channels
        )
        typewell = tuple(
            index
            for index, name in enumerate(channel_names)
            if name in config.typewell_channels
        )
    else:
        roles = tuple(_pair_role(name) for name in channel_names)
        if any(role is None for role in roles):
            unassigned = [
                name for name, role in zip(channel_names, roles) if role is None
            ]
            raise ValueError(
                "paired_patch_cross requires horizontal:/typewell: channel "
                f"prefixes or explicit groups; unassigned={unassigned}"
            )
        horizontal = tuple(
            index for index, role in enumerate(roles) if role == "horizontal"
        )
        typewell = tuple(
            index for index, role in enumerate(roles) if role == "typewell"
        )
    if not horizontal or not typewell:
        raise ValueError("paired_patch_cross requires both channel groups")
    return horizontal, typewell


class _PatchBranch(nn.Module):
    def __init__(
        self,
        horizontal_channels: int,
        typewell_channels: int,
        sequence_length: int,
        patch_size: int,
        hidden_dim: int,
        attention_heads: int,
        layers: int,
        cross_series: bool,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.padding = (-int(sequence_length)) % self.patch_size
        tokens = (int(sequence_length) + self.padding) // self.patch_size
        self.cross_series = bool(cross_series)
        self.horizontal_projection = nn.Conv1d(
            horizontal_channels,
            hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.typewell_projection = nn.Conv1d(
            typewell_channels,
            hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.position = nn.Parameter(torch.zeros(1, tokens, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.self_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.horizontal_norm = nn.LayerNorm(hidden_dim)
        self.typewell_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def _patch(self, values: Tensor, projection: nn.Conv1d) -> Tensor:
        if self.padding:
            values = F.pad(values, (0, self.padding), mode="replicate")
        return projection(values).transpose(1, 2) + self.position

    def forward(self, horizontal: Tensor, typewell: Tensor) -> Tensor:
        horizontal_tokens = self.self_encoder(
            self._patch(horizontal, self.horizontal_projection)
        )
        typewell_tokens = self.self_encoder(
            self._patch(typewell, self.typewell_projection)
        )
        horizontal_mean = horizontal_tokens.mean(dim=1)
        typewell_mean = typewell_tokens.mean(dim=1)
        if self.cross_series:
            horizontal_cross, _ = self.cross_attention(
                horizontal_tokens,
                typewell_tokens,
                typewell_tokens,
                need_weights=False,
            )
            typewell_cross, _ = self.cross_attention(
                typewell_tokens,
                horizontal_tokens,
                horizontal_tokens,
                need_weights=False,
            )
            horizontal_cross_mean = self.horizontal_norm(
                horizontal_tokens + horizontal_cross
            ).mean(dim=1)
            typewell_cross_mean = self.typewell_norm(
                typewell_tokens + typewell_cross
            ).mean(dim=1)
            product = horizontal_cross_mean * typewell_cross_mean
            absolute_difference = torch.abs(horizontal_cross_mean - typewell_cross_mean)
        else:
            # Keeping four zero blocks gives the no-cross ablation an identical
            # head shape and total parameter count.  Its linear head can combine
            # independent towers additively but cannot create pair interactions.
            zero = torch.zeros_like(horizontal_mean)
            horizontal_cross_mean = zero
            typewell_cross_mean = zero
            product = zero
            absolute_difference = zero
        return torch.cat(
            [
                horizontal_mean,
                typewell_mean,
                horizontal_cross_mean,
                typewell_cross_mean,
                product,
                absolute_difference,
            ],
            dim=1,
        )


class _PairedPatchEncoder(nn.Module):
    def __init__(
        self,
        horizontal_indices: tuple[int, ...],
        typewell_indices: tuple[int, ...],
        sequence_length: int,
        config: TorchFamilyConfig,
    ) -> None:
        super().__init__()
        self.horizontal_indices = horizontal_indices
        self.typewell_indices = typewell_indices
        self.branches = nn.ModuleList(
            [
                _PatchBranch(
                    horizontal_channels=len(horizontal_indices),
                    typewell_channels=len(typewell_indices),
                    sequence_length=sequence_length,
                    patch_size=patch_size,
                    hidden_dim=config.hidden_dim,
                    attention_heads=config.attention_heads,
                    layers=config.layers,
                    cross_series=config.cross_series,
                )
                for patch_size in config.patch_sizes
            ]
        )
        self.output_dim = 6 * int(config.hidden_dim) * len(tuple(config.patch_sizes))

    def forward(self, x: Tensor) -> Tensor:
        horizontal = x[:, self.horizontal_indices, :]
        typewell = x[:, self.typewell_indices, :]
        return torch.cat(
            [branch(horizontal, typewell) for branch in self.branches], dim=1
        )


class _S4DLayer(nn.Module):
    """A small diagonal SSM kernel evaluated as a causal FFT convolution."""

    def __init__(self, hidden_dim: int, sequence_length: int) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.log_decay = nn.Parameter(torch.linspace(-2.0, 1.0, hidden_dim))
        self.frequency = nn.Parameter(torch.linspace(0.0, 4.0, hidden_dim))
        self.amplitude = nn.Parameter(torch.full((hidden_dim,), 0.2))
        self.skip = nn.Parameter(torch.ones(hidden_dim))
        self.mix = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[-1]
        time_axis = torch.arange(length, dtype=x.dtype, device=x.device) / max(
            length - 1, 1
        )
        decay = F.softplus(self.log_decay).to(dtype=x.dtype) + 1.0e-4
        frequency = self.frequency.to(dtype=x.dtype)
        kernel = (
            self.amplitude.to(dtype=x.dtype)[:, None]
            * torch.exp(-decay[:, None] * time_axis[None, :])
            * torch.cos(2.0 * math.pi * frequency[:, None] * time_axis[None, :])
        )
        fft_length = 2 * length
        signal_fft = torch.fft.rfft(x, n=fft_length)
        kernel_fft = torch.fft.rfft(kernel, n=fft_length)
        convolved = torch.fft.irfft(signal_fft * kernel_fft[None, :, :], n=fft_length)[
            ..., :length
        ]
        convolved = convolved / math.sqrt(float(length))
        update = self.mix(F.gelu(convolved + self.skip[None, :, None] * x))
        return self.norm(x + update)


class _S4DEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        config: TorchFamilyConfig,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(
            input_channels, config.hidden_dim, kernel_size=1
        )
        self.blocks = nn.ModuleList(
            [
                _S4DLayer(config.hidden_dim, sequence_length)
                for _ in range(config.layers)
            ]
        )
        self.output_dim = 3 * int(config.hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        state = self.input_projection(x)
        for block in self.blocks:
            state = block(state)
        return torch.cat(
            [
                state.mean(dim=-1),
                state[..., -1],
                state[..., -1] - state[..., 0],
            ],
            dim=1,
        )


class _SpectralConv1d(nn.Module):
    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.modes = int(modes)
        scale = 1.0 / max(1, channels)
        self.weight_real = nn.Parameter(
            scale * torch.randn(channels, channels, self.modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(channels, channels, self.modes)
        )

    def forward(self, x: Tensor) -> Tensor:
        transformed = torch.fft.rfft(x, dim=-1)
        modes = min(self.modes, transformed.shape[-1])
        weights = torch.complex(
            self.weight_real[..., :modes], self.weight_imag[..., :modes]
        )
        selected = torch.einsum("bim,iom->bom", transformed[..., :modes], weights)
        output = torch.zeros(
            x.shape[0],
            x.shape[1],
            transformed.shape[-1],
            dtype=transformed.dtype,
            device=x.device,
        )
        output[..., :modes] = selected
        return torch.fft.irfft(output, n=x.shape[-1], dim=-1)


class _FNOBlock1d(nn.Module):
    def __init__(self, hidden_dim: int, modes: int) -> None:
        super().__init__()
        self.spectral = _SpectralConv1d(hidden_dim, modes)
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        update = F.gelu(self.spectral(x) + self.pointwise(x))
        return self.norm(x + update)


class _FNO1dEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        config: TorchFamilyConfig,
    ) -> None:
        super().__init__()
        self.padding = min(8, max(1, sequence_length // 8))
        self.input_projection = nn.Conv1d(
            input_channels + 1, config.hidden_dim, kernel_size=1
        )
        self.blocks = nn.ModuleList(
            [
                _FNOBlock1d(config.hidden_dim, config.fno_modes)
                for _ in range(config.layers)
            ]
        )
        self.output_dim = 3 * int(config.hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        coordinate = torch.linspace(
            -1.0, 1.0, x.shape[-1], dtype=x.dtype, device=x.device
        )
        coordinate = coordinate[None, None, :].expand(x.shape[0], 1, -1)
        state = self.input_projection(torch.cat([x, coordinate], dim=1))
        state = F.pad(state, (0, self.padding))
        for block in self.blocks:
            state = block(state)
        state = state[..., : -self.padding]
        return torch.cat(
            [
                state.mean(dim=-1),
                state[..., -1],
                state[..., -1] - state[..., 0],
            ],
            dim=1,
        )


class _CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.convolution(F.pad(x, (self.left_padding, 0)))


class _TCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.first = _CausalConv1d(hidden_dim, kernel_size, dilation)
        self.second = _CausalConv1d(hidden_dim, kernel_size, dilation)
        self.norm = nn.GroupNorm(1, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        update = F.gelu(self.first(x))
        update = self.second(update)
        return F.gelu(self.norm(x + update))


class _TCNEncoder(nn.Module):
    def __init__(self, input_channels: int, config: TorchFamilyConfig) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(
            input_channels, config.hidden_dim, kernel_size=1
        )
        self.blocks = nn.ModuleList(
            [
                _TCNBlock(
                    config.hidden_dim,
                    config.tcn_kernel_size,
                    dilation=2**layer,
                )
                for layer in range(config.layers)
            ]
        )
        self.output_dim = 3 * int(config.hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        state = self.input_projection(x)
        for block in self.blocks:
            state = block(state)
        return torch.cat(
            [
                state.mean(dim=-1),
                state[..., -1],
                state[..., -1] - state[..., 0],
            ],
            dim=1,
        )


class _EncoderRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, output_dim: int, target_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(output_dim, target_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x))


def _build_model(
    config: TorchFamilyConfig,
    *,
    input_channels: int,
    sequence_length: int,
    target_dim: int,
    channel_names: tuple[str, ...],
) -> _EncoderRegressor:
    if config.family == FAMILY_PAIRED_PATCH:
        if any(patch > sequence_length for patch in config.patch_sizes):
            raise ValueError(
                "every patch_size must be <= the fixed-grid sequence length"
            )
        horizontal, typewell = _paired_channel_indices(channel_names, config)
        encoder: nn.Module = _PairedPatchEncoder(
            horizontal,
            typewell,
            sequence_length,
            config,
        )
    elif config.family == FAMILY_S4D:
        encoder = _S4DEncoder(input_channels, sequence_length, config)
    elif config.family == FAMILY_FNO1D:
        encoder = _FNO1dEncoder(input_channels, sequence_length, config)
    elif config.family == FAMILY_TCN:
        encoder = _TCNEncoder(input_channels, config)
    else:  # pragma: no cover - guarded by TorchFamilyConfig
        raise ValueError(f"unsupported family: {config.family}")
    output_dim = int(getattr(encoder, "output_dim"))
    model = _EncoderRegressor(encoder, output_dim, target_dim)
    if config.frozen_random_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    return model


def _resolved_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TorchFamilyError(f"requested CUDA device is unavailable: {name}")
    if device.type not in {"cpu", "cuda"}:
        raise TorchFamilyError(
            "only cpu and cuda devices are supported by this deterministic harness"
        )
    return device


@contextmanager
def _deterministic_execution() -> Any:
    """Temporarily require deterministic PyTorch kernels without global leakage."""

    previous_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_getter = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    previous_warn_only = (
        bool(warn_only_getter()) if warn_only_getter is not None else False
    )
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        try:
            torch.use_deterministic_algorithms(
                previous_enabled, warn_only=previous_warn_only
            )
        except TypeError:  # Older PyTorch did not expose warn_only.
            torch.use_deterministic_algorithms(previous_enabled)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic


def _model_is_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.detach()).all().item())
        for parameter in model.parameters()
    )


def _parameter_counts(model: _EncoderRegressor) -> dict[str, int]:
    def count(module: nn.Module, *, trainable_only: bool = False) -> int:
        return int(
            sum(
                parameter.numel()
                for parameter in module.parameters()
                if not trainable_only or parameter.requires_grad
            )
        )

    return {
        "parameters_total": count(model),
        "parameters_trainable": count(model, trainable_only=True),
        "encoder_parameters_total": count(model.encoder),
        "encoder_parameters_trainable": count(model.encoder, trainable_only=True),
        "head_parameters_total": count(model.head),
        "head_parameters_trainable": count(model.head, trainable_only=True),
    }


def _parameter_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _ids_sha256(ids: tuple[str, ...]) -> str:
    encoded = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _readonly_float_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise TorchFamilyError(f"{name} is invalid")
    result.setflags(write=False)
    return result


class TorchFamilyState:
    """Fitted, CPU-resident external-family state."""

    def __init__(
        self,
        *,
        config: TorchFamilyConfig,
        channel_names: tuple[str, ...],
        sequence_length: int,
        target_dim: int,
        input_mean: Any,
        input_scale: Any,
        target_mean: Any,
        target_scale: Any,
        model: _EncoderRegressor,
        manifest: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.channel_names = _validated_channels(channel_names, len(channel_names))
        self.sequence_length = int(sequence_length)
        self.target_dim = int(target_dim)
        if self.sequence_length < 4 or self.target_dim < 1:
            raise TorchFamilyError("serialized dimensions are invalid")
        channels = len(self.channel_names)
        self.input_mean = _readonly_float_vector(
            input_mean, name="input_mean", length=channels
        )
        self.input_scale = _readonly_float_vector(
            input_scale, name="input_scale", length=channels
        )
        self.target_mean = _readonly_float_vector(
            target_mean, name="target_mean", length=self.target_dim
        )
        self.target_scale = _readonly_float_vector(
            target_scale, name="target_scale", length=self.target_dim
        )
        if np.any(self.input_scale <= 0.0) or np.any(self.target_scale <= 0.0):
            raise TorchFamilyError("serialized scales must be positive")
        self.model = model.to("cpu")
        self.model.eval()
        if not _model_is_finite(self.model):
            raise TorchFamilyError("model state contains non-finite parameters")
        self._manifest = dict(manifest)

    @classmethod
    def fit(
        cls,
        x: Any,
        targets: Any,
        *,
        channel_names: Sequence[str],
        ids: Sequence[str],
        row_counts: Any = None,
        config: TorchFamilyConfig,
    ) -> "TorchFamilyState":
        return fit_torch_family(
            x,
            targets,
            channel_names=channel_names,
            ids=ids,
            row_counts=row_counts,
            config=config,
        )

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the training manifest."""

        return json.loads(json.dumps(self._manifest, sort_keys=True))

    def predict(
        self,
        x: Any,
        *,
        channel_names: Sequence[str],
        ids: Sequence[str],
    ) -> np.ndarray:
        values = _validated_x(x)
        names = _validated_channels(channel_names, values.shape[1])
        _validated_ids(ids, values.shape[0])
        if names != self.channel_names:
            raise ValueError(
                "prediction channel_names/order differ from the fitted state"
            )
        if values.shape[2] != self.sequence_length:
            raise ValueError(
                "prediction fixed-grid length differs from the fitted state"
            )
        if not _model_is_finite(self.model):
            raise TorchFamilyError("model state became non-finite")

        scaled = (values - self.input_mean[None, :, None]) / self.input_scale[
            None, :, None
        ]
        tensor = torch.from_numpy(np.asarray(scaled, dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            standardized = self.model(tensor)
        if standardized.shape != (len(values), self.target_dim):
            raise TorchFamilyError("model returned an unexpected prediction shape")
        if not bool(torch.isfinite(standardized).all().item()):
            raise TorchFamilyError("model returned non-finite predictions")
        result = standardized.cpu().numpy().astype(np.float64) * self.target_scale[
            None, :
        ].astype(np.float64) + self.target_mean[None, :].astype(np.float64)
        if not np.isfinite(result).all():
            raise TorchFamilyError("inverse scaling returned non-finite predictions")
        return result

    def save(self, path: str | Path) -> Path:
        """Atomically save a CPU-only state dictionary and primitive metadata."""

        destination = Path(path)
        if destination.exists() and destination.is_dir():
            raise TorchFamilyError("state path points to a directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "config": asdict(self.config),
            "channel_names": list(self.channel_names),
            "sequence_length": self.sequence_length,
            "target_dim": self.target_dim,
            "input_mean": torch.from_numpy(np.array(self.input_mean, copy=True)),
            "input_scale": torch.from_numpy(np.array(self.input_scale, copy=True)),
            "target_mean": torch.from_numpy(np.array(self.target_mean, copy=True)),
            "target_scale": torch.from_numpy(np.array(self.target_scale, copy=True)),
            "model_state": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
            },
            "manifest": self.manifest,
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        except (OSError, RuntimeError, ValueError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise TorchFamilyError(f"failed to save state: {destination}") from exc
        return destination


def fit_torch_family(
    x: Any,
    targets: Any,
    *,
    channel_names: Sequence[str],
    ids: Sequence[str],
    row_counts: Any = None,
    config: TorchFamilyConfig,
) -> TorchFamilyState:
    """Fit one family using only the supplied fold's arrays and statistics."""

    if not isinstance(config, TorchFamilyConfig):
        raise TypeError("config must be a TorchFamilyConfig")
    values = _validated_x(x)
    labels = _validated_targets(targets, values.shape[0])
    names = _validated_channels(channel_names, values.shape[1])
    identifiers = _validated_ids(ids, values.shape[0])
    counts = _validated_row_counts(row_counts, values.shape[0])
    input_mean, input_scale, target_mean, target_scale = _weighted_scalers(
        values, labels, counts
    )
    standardized_x = (values - input_mean[None, :, None]) / input_scale[None, :, None]
    standardized_y = (labels - target_mean[None, :]) / target_scale[None, :]
    if not np.isfinite(standardized_x).all() or not np.isfinite(standardized_y).all():
        raise TorchFamilyError("standardization produced non-finite values")

    device = _resolved_device(config.device)
    started = time.perf_counter()
    with _deterministic_execution(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = _build_model(
            config,
            input_channels=values.shape[1],
            sequence_length=values.shape[2],
            target_dim=labels.shape[1],
            channel_names=names,
        ).to(device)
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise TorchFamilyError("model has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        tensor_x = torch.from_numpy(np.asarray(standardized_x, dtype=np.float32)).to(
            device
        )
        tensor_y = torch.from_numpy(np.asarray(standardized_y, dtype=np.float32)).to(
            device
        )
        normalized_weights = counts.astype(np.float32)
        normalized_weights /= float(normalized_weights.sum())
        tensor_weights = torch.from_numpy(normalized_weights).to(device)

        final_loss = math.nan
        model.train()
        for _ in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tensor_x)
            per_well = torch.square(prediction - tensor_y).mean(dim=1)
            loss = torch.sum(tensor_weights * per_well)
            if not bool(torch.isfinite(loss).item()):
                raise TorchFamilyError("training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            optimizer.step()
            if not _model_is_finite(model):
                raise TorchFamilyError("optimizer produced non-finite parameters")
            final_loss = float(loss.detach().cpu())

        model.eval()
        with torch.no_grad():
            training_prediction = model(tensor_x)
        if not bool(torch.isfinite(training_prediction).all().item()):
            raise TorchFamilyError("fitted model produced non-finite output")
        model = model.to("cpu")

    elapsed = time.perf_counter() - started
    counts_manifest = _parameter_counts(model)
    parameter_digest = _parameter_sha256(model)
    manifest: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "family": config.family,
        "variant": (
            "no_cross"
            if config.family == FAMILY_PAIRED_PATCH and not config.cross_series
            else "cross"
            if config.family == FAMILY_PAIRED_PATCH
            else "default"
        ),
        "architecture_only": True,
        "frozen_random_encoder": config.frozen_random_encoder,
        "seed": config.seed,
        "fit_device": str(device),
        "inference_device": "cpu",
        "deterministic_algorithms": True,
        "dtype": "float32",
        "torch_version": str(torch.__version__),
        "sample_count": int(values.shape[0]),
        "row_count_sum": int(counts.sum()),
        "input_channels": int(values.shape[1]),
        "sequence_length": int(values.shape[2]),
        "target_dim": int(labels.shape[1]),
        "channel_names": list(names),
        "fit_ids_sha256": _ids_sha256(identifiers),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "final_weighted_standardized_mse": final_loss,
        "fit_seconds": float(elapsed),
        "parameter_sha256": parameter_digest,
        **counts_manifest,
    }
    return TorchFamilyState(
        config=config,
        channel_names=names,
        sequence_length=values.shape[2],
        target_dim=labels.shape[1],
        input_mean=input_mean,
        input_scale=input_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        model=model,
        manifest=manifest,
    )


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility for trusted local artifacts.
        return torch.load(path, map_location="cpu")


def load_torch_family(path: str | Path) -> TorchFamilyState:
    """Load and strictly validate a state saved by :meth:`TorchFamilyState.save`."""

    source = Path(path)
    if not source.is_file():
        raise TorchFamilyError(f"state file does not exist: {source}")
    try:
        payload = _safe_torch_load(source)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise TorchFamilyError(f"failed to load state: {source}") from exc
    if not isinstance(payload, dict):
        raise TorchFamilyError("serialized state must be a mapping")
    required = {
        "schema_version",
        "config",
        "channel_names",
        "sequence_length",
        "target_dim",
        "input_mean",
        "input_scale",
        "target_mean",
        "target_scale",
        "model_state",
        "manifest",
    }
    if not required.issubset(payload):
        missing = sorted(required - set(payload))
        raise TorchFamilyError(f"serialized state is incomplete: {missing}")
    if payload["schema_version"] != STATE_SCHEMA_VERSION:
        raise TorchFamilyError("serialized state schema is unsupported")
    try:
        config = TorchFamilyConfig(**dict(payload["config"]))
        names = tuple(map(str, payload["channel_names"]))
        sequence_length = int(payload["sequence_length"])
        target_dim = int(payload["target_dim"])
        names = _validated_channels(names, len(names))
        model = _build_model(
            config,
            input_channels=len(names),
            sequence_length=sequence_length,
            target_dim=target_dim,
            channel_names=names,
        )
        model.load_state_dict(payload["model_state"], strict=True)
        model = model.to("cpu")
        manifest = dict(payload["manifest"])
        state = TorchFamilyState(
            config=config,
            channel_names=names,
            sequence_length=sequence_length,
            target_dim=target_dim,
            input_mean=payload["input_mean"].cpu().numpy(),
            input_scale=payload["input_scale"].cpu().numpy(),
            target_mean=payload["target_mean"].cpu().numpy(),
            target_scale=payload["target_scale"].cpu().numpy(),
            model=model,
            manifest=manifest,
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        TorchFamilyError,
    ) as exc:
        raise TorchFamilyError("serialized state failed validation") from exc

    expected_digest = str(state.manifest.get("parameter_sha256", ""))
    if not expected_digest or _parameter_sha256(state.model) != expected_digest:
        raise TorchFamilyError("serialized model parameter digest does not match")
    if (
        state.manifest.get("schema_version") != STATE_SCHEMA_VERSION
        or state.manifest.get("family") != state.config.family
        or int(state.manifest.get("sequence_length", -1)) != state.sequence_length
        or int(state.manifest.get("target_dim", -1)) != state.target_dim
    ):
        raise TorchFamilyError("serialized manifest is inconsistent")
    return state
