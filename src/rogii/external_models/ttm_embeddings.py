"""Frozen TTM embeddings, paired controls, fold-local scaling, and residual head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import FixedGridSequenceWell, SequenceModelInput
from .controls import (
    CONTROL_GR_ORDER_SHUFFLE,
    CONTROL_NONE,
    CONTROL_PAIR_SHUFFLE,
    CONTROL_PHASE_DESTROY,
    CONTROL_WRONG_TYPEWELL,
    ControlSpec,
    apply_individual_control,
    paired_control_inputs,
)
from .ttm import (
    TTM_CONTEXT_LENGTH,
    TTM_D_MODEL,
    TTM_PATCH_COUNT,
    TTM_PATCH_LENGTH,
)


FROZEN_VARIANTS = (
    "ttm_random_frozen",
    "ttm_pretrained_frozen",
    "ttm_pretrained_wrong_typewell",
    "ttm_pretrained_pair_shuffle",
    "ttm_pretrained_gr_order_shuffle",
    "ttm_pretrained_phase_destroy",
)
OPTIONAL_VARIANTS = (
    "ttm_scratch_trainable",
    "ttm_pretrained_adapter",
)
TTM_VARIANTS = (*FROZEN_VARIANTS, *OPTIONAL_VARIANTS)
VARIANT_CONTROL = {
    "ttm_random_frozen": CONTROL_NONE,
    "ttm_pretrained_frozen": CONTROL_NONE,
    "ttm_pretrained_wrong_typewell": CONTROL_WRONG_TYPEWELL,
    "ttm_pretrained_pair_shuffle": CONTROL_PAIR_SHUFFLE,
    "ttm_pretrained_gr_order_shuffle": CONTROL_GR_ORDER_SHUFFLE,
    "ttm_pretrained_phase_destroy": CONTROL_PHASE_DESTROY,
    "ttm_scratch_trainable": CONTROL_NONE,
    "ttm_pretrained_adapter": CONTROL_NONE,
}
PRETRAINED_VARIANTS = frozenset(
    variant for variant in TTM_VARIANTS if "random" not in variant and "scratch" not in variant
)
ELIGIBLE_FOR_MODEL_SELECTION = {
    "ttm_random_frozen": True,
    "ttm_pretrained_frozen": True,
    "ttm_scratch_trainable": False,
    "ttm_pretrained_adapter": False,
    "ttm_pretrained_wrong_typewell": False,
    "ttm_pretrained_pair_shuffle": False,
    "ttm_pretrained_gr_order_shuffle": False,
    "ttm_pretrained_phase_destroy": False,
}


def _stable_uint64(seed: int, *parts: str) -> int:
    payload = "\0".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _column(values: np.ndarray, names: Sequence[str], name: str) -> np.ndarray:
    try:
        index = tuple(names).index(name)
    except ValueError as exc:
        raise ValueError(f"TTM input lacks channel {name!r}") from exc
    return np.asarray(values[:, index], dtype=np.float32)


def _replace_sequence_gr(
    sequence: FixedGridSequenceWell,
    *,
    horizontal_gr: np.ndarray,
    horizontal_mask: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_mask: np.ndarray,
) -> FixedGridSequenceWell:
    horizontal = np.array(sequence.horizontal, dtype=np.float32, copy=True)
    typewell = np.array(sequence.typewell, dtype=np.float32, copy=True)
    horizontal[:, sequence.horizontal_feature_names.index("GR_filled")] = horizontal_gr
    horizontal[:, sequence.horizontal_feature_names.index("GR_observed")] = horizontal_mask
    typewell[:, sequence.typewell_feature_names.index("GR_filled")] = typewell_gr
    typewell[:, sequence.typewell_feature_names.index("GR_observed")] = typewell_mask
    return FixedGridSequenceWell(
        well_id=sequence.well_id,
        row_index=sequence.row_index,
        known_count=sequence.known_count,
        anchor_tvt=sequence.anchor_tvt,
        horizontal=horizontal,
        typewell=typewell,
        output_coordinate=sequence.output_coordinate,
        horizontal_feature_names=sequence.horizontal_feature_names,
        typewell_feature_names=sequence.typewell_feature_names,
        source_horizontal_rows=sequence.source_horizontal_rows,
        source_typewell_rows=sequence.source_typewell_rows,
        schema_version=sequence.schema_version,
    )


def _phase_destroy(values: np.ndarray, *, seed: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    center = float(np.mean(source))
    spectrum = np.fft.rfft(source - center)
    random = np.random.default_rng(seed)
    if len(spectrum) > 1:
        phase = random.uniform(-np.pi, np.pi, len(spectrum))
        phase[0] = 0.0
        if len(source) % 2 == 0:
            phase[-1] = 0.0
        spectrum = np.abs(spectrum) * np.exp(1j * phase)
        spectrum[0] = 0.0
        if len(source) % 2 == 0:
            spectrum[-1] = complex(float(np.real(spectrum[-1])), 0.0)
    result = np.fft.irfft(spectrum, n=len(source)) + center
    if not np.isfinite(result).all():
        raise ValueError("TTM phase control produced non-finite GR")
    return result.astype(np.float32)


def apply_ttm_control(
    inputs: Sequence[SequenceModelInput],
    *,
    control_id: str,
    seed: int,
) -> dict[str, SequenceModelInput]:
    """Apply controls to both the bypass channels and raw TTM GR streams."""

    values = tuple(inputs)
    if control_id in {CONTROL_NONE, CONTROL_WRONG_TYPEWELL, CONTROL_PAIR_SHUFFLE}:
        return paired_control_inputs(values, control_id=control_id, seed=seed)
    if control_id not in {CONTROL_GR_ORDER_SHUFFLE, CONTROL_PHASE_DESTROY}:
        raise ValueError(f"unsupported TTM control: {control_id}")
    result: dict[str, SequenceModelInput] = {}
    for input_well in values:
        controlled = apply_individual_control(
            input_well,
            spec=ControlSpec(control_id, seed),
        )
        sequence = input_well.sequence
        horizontal_gr = _column(
            sequence.horizontal,
            sequence.horizontal_feature_names,
            "GR_filled",
        )
        horizontal_mask = _column(
            sequence.horizontal,
            sequence.horizontal_feature_names,
            "GR_observed",
        )
        typewell_gr = _column(
            sequence.typewell,
            sequence.typewell_feature_names,
            "GR_filled",
        )
        typewell_mask = _column(
            sequence.typewell,
            sequence.typewell_feature_names,
            "GR_observed",
        )
        if control_id == CONTROL_GR_ORDER_SHUFFLE:
            h_order = np.random.default_rng(
                _stable_uint64(seed, sequence.well_id, "horizontal-order")
            ).permutation(len(horizontal_gr))
            t_order = np.random.default_rng(
                _stable_uint64(seed, sequence.well_id, "typewell-order")
            ).permutation(len(typewell_gr))
            horizontal_gr = horizontal_gr[h_order]
            horizontal_mask = horizontal_mask[h_order]
            typewell_gr = typewell_gr[t_order]
            typewell_mask = typewell_mask[t_order]
        else:
            horizontal_gr = _phase_destroy(
                horizontal_gr,
                seed=_stable_uint64(seed, sequence.well_id, "horizontal-phase"),
            )
            typewell_gr = _phase_destroy(
                typewell_gr,
                seed=_stable_uint64(seed, sequence.well_id, "typewell-phase"),
            )
        replaced = _replace_sequence_gr(
            sequence,
            horizontal_gr=horizontal_gr,
            horizontal_mask=horizontal_mask,
            typewell_gr=typewell_gr,
            typewell_mask=typewell_mask,
        )
        result[input_well.well_id] = SequenceModelInput(
            sequence=replaced,
            incumbent_prediction=controlled.incumbent_prediction,
            x=controlled.x,
            grid_coordinate=controlled.grid_coordinate,
            channel_names=controlled.channel_names,
            control_id=control_id,
            paired_typewell_id=controlled.paired_typewell_id,
            cross_series_enabled=True,
        )
    return result


@dataclass(frozen=True)
class TTMInputBatch:
    ids: tuple[str, ...]
    horizontal: np.ndarray
    horizontal_mask: np.ndarray
    typewell: np.ndarray
    typewell_mask: np.ndarray
    bypass: np.ndarray
    prefix_mask: np.ndarray
    paired_typewell_ids: tuple[str, ...]
    control_id: str

    def __post_init__(self) -> None:
        ids = tuple(map(str, self.ids))
        n = len(ids)
        if n < 1 or len(set(ids)) != n:
            raise ValueError("TTM batch IDs must be non-empty and unique")
        expected_sequence = (n, TTM_CONTEXT_LENGTH, 1)
        horizontal = np.asarray(self.horizontal, dtype=np.float32)
        horizontal_mask = np.asarray(self.horizontal_mask, dtype=np.float32)
        typewell = np.asarray(self.typewell, dtype=np.float32)
        typewell_mask = np.asarray(self.typewell_mask, dtype=np.float32)
        bypass = np.asarray(self.bypass, dtype=np.float32)
        prefix_mask = np.asarray(self.prefix_mask, dtype=np.float32)
        if any(
            array.shape != expected_sequence
            for array in (horizontal, horizontal_mask, typewell, typewell_mask)
        ):
            raise ValueError("TTM GR values/masks must have shape [B,512,1]")
        if bypass.ndim != 3 or bypass.shape[0] != n or bypass.shape[2] != 512:
            raise ValueError("TTM bypass must have shape [B,C,512]")
        if prefix_mask.shape != (n, 1, 512):
            raise ValueError("TTM prefix mask must have shape [B,1,512]")
        if not all(
            np.isfinite(array).all()
            for array in (
                horizontal,
                horizontal_mask,
                typewell,
                typewell_mask,
                bypass,
                prefix_mask,
            )
        ):
            raise ValueError("TTM batch contains non-finite values")
        if not all(
            np.isin(mask, (0.0, 1.0)).all()
            for mask in (horizontal_mask, typewell_mask, prefix_mask)
        ):
            raise ValueError("TTM masks must be binary")
        paired = tuple(map(str, self.paired_typewell_ids))
        if len(paired) != n or any(not value for value in paired):
            raise ValueError("paired typewell IDs must align with batch IDs")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "horizontal", horizontal)
        object.__setattr__(self, "horizontal_mask", horizontal_mask)
        object.__setattr__(self, "typewell", typewell)
        object.__setattr__(self, "typewell_mask", typewell_mask)
        object.__setattr__(self, "bypass", bypass)
        object.__setattr__(self, "prefix_mask", prefix_mask)
        object.__setattr__(self, "paired_typewell_ids", paired)

    def select(self, indices: np.ndarray | Sequence[int]) -> "TTMInputBatch":
        selected = np.asarray(indices, dtype=np.int64)
        return TTMInputBatch(
            ids=tuple(self.ids[index] for index in selected),
            horizontal=self.horizontal[selected],
            horizontal_mask=self.horizontal_mask[selected],
            typewell=self.typewell[selected],
            typewell_mask=self.typewell_mask[selected],
            bypass=self.bypass[selected],
            prefix_mask=self.prefix_mask[selected],
            paired_typewell_ids=tuple(
                self.paired_typewell_ids[index] for index in selected
            ),
            control_id=self.control_id,
        )


def make_ttm_batch(
    inputs: Mapping[str, SequenceModelInput] | Sequence[SequenceModelInput],
    *,
    ids: Sequence[str] | None = None,
) -> TTMInputBatch:
    if isinstance(inputs, Mapping):
        by_id = {str(key): value for key, value in inputs.items()}
        order = tuple(sorted(by_id) if ids is None else map(str, ids))
        values = tuple(by_id[well_id] for well_id in order)
    else:
        values = tuple(inputs)
        by_id = {item.well_id: item for item in values}
        if len(by_id) != len(values):
            raise ValueError("TTM inputs contain duplicate well IDs")
        order = tuple(item.well_id for item in values) if ids is None else tuple(map(str, ids))
        values = tuple(by_id[well_id] for well_id in order)
    if not values:
        raise ValueError("TTM inputs must not be empty")
    if any(not isinstance(item, SequenceModelInput) for item in values):
        raise TypeError("TTM inputs must contain SequenceModelInput values")
    controls = {item.control_id for item in values}
    if len(controls) != 1:
        raise ValueError("TTM input batch mixes controls")

    horizontal: list[np.ndarray] = []
    horizontal_mask: list[np.ndarray] = []
    typewell: list[np.ndarray] = []
    typewell_mask: list[np.ndarray] = []
    bypass: list[np.ndarray] = []
    prefix: list[np.ndarray] = []
    paired: list[str] = []
    for item in values:
        sequence = item.sequence
        if len(sequence.horizontal) != 512 or len(sequence.typewell) != 512:
            raise ValueError("TTM requires deterministic 512-point raw grids")
        horizontal.append(
            _column(
                sequence.horizontal,
                sequence.horizontal_feature_names,
                "GR_filled",
            )[:, None]
        )
        horizontal_mask.append(
            (
                _column(
                    sequence.horizontal,
                    sequence.horizontal_feature_names,
                    "GR_observed",
                )
                >= 0.5
            ).astype(np.float32)[:, None]
        )
        typewell.append(
            _column(
                sequence.typewell,
                sequence.typewell_feature_names,
                "GR_filled",
            )[:, None]
        )
        typewell_mask.append(
            (
                _column(
                    sequence.typewell,
                    sequence.typewell_feature_names,
                    "GR_observed",
                )
                >= 0.5
            ).astype(np.float32)[:, None]
        )
        bypass.append(np.asarray(item.x, dtype=np.float32))
        prefix.append(
            (
                _column(
                    sequence.horizontal,
                    sequence.horizontal_feature_names,
                    "TVT_input_observed",
                )
                >= 0.5
            ).astype(np.float32)[None, :]
        )
        paired.append(str(item.paired_typewell_id))
    return TTMInputBatch(
        ids=order,
        horizontal=np.stack(horizontal),
        horizontal_mask=np.stack(horizontal_mask),
        typewell=np.stack(typewell),
        typewell_mask=np.stack(typewell_mask),
        bypass=np.stack(bypass),
        prefix_mask=np.stack(prefix),
        paired_typewell_ids=tuple(paired),
        control_id=next(iter(controls)),
    )


def encode_ttm_batch(
    backbone: nn.Module,
    batch: TTMInputBatch,
    *,
    device: torch.device,
    batch_size: int,
    require_frozen: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size < 1:
        raise ValueError("embedding batch size must be positive")
    if require_frozen and any(
        parameter.requires_grad for parameter in backbone.parameters()
    ):
        raise ValueError("frozen TTM embedding extraction received trainable backbone")
    backbone.eval()
    horizontal: list[np.ndarray] = []
    typewell: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(batch.ids), batch_size):
            stop = min(start + batch_size, len(batch.ids))
            h = torch.from_numpy(batch.horizontal[start:stop]).to(device)
            hm = torch.from_numpy(batch.horizontal_mask[start:stop]).to(device)
            t = torch.from_numpy(batch.typewell[start:stop]).to(device)
            tm = torch.from_numpy(batch.typewell_mask[start:stop]).to(device)
            h_output = backbone(
                h,
                past_observed_mask=hm,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            t_output = backbone(
                t,
                past_observed_mask=tm,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            for name, output in (("horizontal", h_output), ("typewell", t_output)):
                if tuple(output.shape[1:]) != (1, TTM_PATCH_COUNT, TTM_D_MODEL):
                    raise ValueError(
                        f"unexpected TTM {name} embedding shape: {tuple(output.shape)}"
                    )
            horizontal.append(h_output[:, 0].detach().cpu().numpy())
            typewell.append(t_output[:, 0].detach().cpu().numpy())
    h_array = np.concatenate(horizontal).astype(np.float32)
    t_array = np.concatenate(typewell).astype(np.float32)
    if not np.isfinite(h_array).all() or not np.isfinite(t_array).all():
        raise ValueError("TTM embeddings are non-finite")
    return h_array, t_array


@dataclass(frozen=True)
class FoldLocalTTMScaler:
    horizontal_mean: np.ndarray
    horizontal_scale: np.ndarray
    typewell_mean: np.ndarray
    typewell_scale: np.ndarray
    bypass_mean: np.ndarray
    bypass_scale: np.ndarray
    fit_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]

    @staticmethod
    def fit(
        horizontal: np.ndarray,
        typewell: np.ndarray,
        bypass: np.ndarray,
        *,
        fit_ids: Sequence[str],
        validation_ids: Sequence[str],
    ) -> "FoldLocalTTMScaler":
        fit = tuple(map(str, fit_ids))
        validation = tuple(map(str, validation_ids))
        if not fit or set(fit) & set(validation):
            raise ValueError("fold-local scaler fit/validation IDs overlap or are empty")
        h = np.asarray(horizontal, dtype=np.float64)
        t = np.asarray(typewell, dtype=np.float64)
        b = np.asarray(bypass, dtype=np.float64)
        if (
            h.shape != (len(fit), TTM_PATCH_COUNT, TTM_D_MODEL)
            or t.shape != h.shape
            or b.ndim != 3
            or b.shape[0] != len(fit)
            or b.shape[2] != 512
        ):
            raise ValueError("fold-local scaler inputs have invalid shapes")

        def stats(array: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
            mean = np.mean(array, axis=axes)
            scale = np.std(array, axis=axes)
            scale = np.where(scale > 1.0e-6, scale, 1.0)
            return mean.astype(np.float32), scale.astype(np.float32)

        h_mean, h_scale = stats(h, (0, 1))
        t_mean, t_scale = stats(t, (0, 1))
        b_mean, b_scale = stats(b, (0, 2))
        return FoldLocalTTMScaler(
            horizontal_mean=h_mean,
            horizontal_scale=h_scale,
            typewell_mean=t_mean,
            typewell_scale=t_scale,
            bypass_mean=b_mean,
            bypass_scale=b_scale,
            fit_ids=fit,
            validation_ids=validation,
        )

    def transform(
        self,
        horizontal: np.ndarray,
        typewell: np.ndarray,
        bypass: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = (
            np.asarray(horizontal, dtype=np.float32)
            - self.horizontal_mean[None, None, :]
        ) / self.horizontal_scale[None, None, :]
        t = (
            np.asarray(typewell, dtype=np.float32)
            - self.typewell_mean[None, None, :]
        ) / self.typewell_scale[None, None, :]
        b = (
            np.asarray(bypass, dtype=np.float32)
            - self.bypass_mean[None, :, None]
        ) / self.bypass_scale[None, :, None]
        if not np.isfinite(h).all() or not np.isfinite(t).all() or not np.isfinite(b).all():
            raise ValueError("fold-local TTM scaler produced non-finite values")
        return h.astype(np.float32), t.astype(np.float32), b.astype(np.float32)

    def payload(self) -> dict[str, Any]:
        return {
            "horizontal_mean": self.horizontal_mean.tolist(),
            "horizontal_scale": self.horizontal_scale.tolist(),
            "typewell_mean": self.typewell_mean.tolist(),
            "typewell_scale": self.typewell_scale.tolist(),
            "bypass_mean": self.bypass_mean.tolist(),
            "bypass_scale": self.bypass_scale.tolist(),
            "fit_ids": list(self.fit_ids),
            "validation_ids": list(self.validation_ids),
        }

    @staticmethod
    def from_payload(payload: Mapping[str, Any]) -> "FoldLocalTTMScaler":
        return FoldLocalTTMScaler(
            horizontal_mean=np.asarray(payload["horizontal_mean"], dtype=np.float32),
            horizontal_scale=np.asarray(payload["horizontal_scale"], dtype=np.float32),
            typewell_mean=np.asarray(payload["typewell_mean"], dtype=np.float32),
            typewell_scale=np.asarray(payload["typewell_scale"], dtype=np.float32),
            bypass_mean=np.asarray(payload["bypass_mean"], dtype=np.float32),
            bypass_scale=np.asarray(payload["bypass_scale"], dtype=np.float32),
            fit_ids=tuple(map(str, payload["fit_ids"])),
            validation_ids=tuple(map(str, payload["validation_ids"])),
        )


class LowRankResidualAdapter(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        if rank < 1 or rank >= width:
            raise ValueError("adapter rank must lie in [1, width)")
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.up(F.gelu(self.down(value)))


class TTMPairedInteractionHead(nn.Module):
    """Small paired interaction head shared by every frozen contrast."""

    def __init__(
        self,
        *,
        bypass_channels: int,
        hidden_dim: int = 32,
        adapter_rank: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("TTM head hidden_dim must be >= 8")
        self.hidden_dim = int(hidden_dim)
        self.adapter_rank = int(adapter_rank)
        self.horizontal_adapter = (
            LowRankResidualAdapter(TTM_D_MODEL, adapter_rank)
            if adapter_rank > 0
            else nn.Identity()
        )
        self.typewell_adapter = (
            LowRankResidualAdapter(TTM_D_MODEL, adapter_rank)
            if adapter_rank > 0
            else nn.Identity()
        )
        self.query = nn.Linear(TTM_D_MODEL, hidden_dim)
        self.key = nn.Linear(TTM_D_MODEL, hidden_dim)
        self.value = nn.Linear(TTM_D_MODEL, hidden_dim)
        self.bypass = nn.Linear(int(bypass_channels) + 1, hidden_dim)
        self.interaction = nn.Sequential(
            nn.Linear(5 * hidden_dim + 2, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.patch_output = nn.Linear(2 * hidden_dim, TTM_PATCH_LENGTH)

    def forward(
        self,
        horizontal: torch.Tensor,
        typewell: torch.Tensor,
        bypass: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        if horizontal.shape != typewell.shape or horizontal.shape[1:] != (
            TTM_PATCH_COUNT,
            TTM_D_MODEL,
        ):
            raise ValueError("paired TTM embeddings have invalid shapes")
        h = self.horizontal_adapter(horizontal)
        t = self.typewell_adapter(typewell)
        query = self.query(h)
        key = self.key(t)
        value = self.value(t)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.hidden_dim
        )
        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention, value)
        correlation_max = torch.max(logits, dim=-1, keepdim=True).values
        correlation_mean = torch.mean(logits, dim=-1, keepdim=True)
        interaction = torch.cat(
            [
                query,
                attended,
                query - attended,
                query * attended,
                torch.abs(query - attended),
                correlation_max,
                correlation_mean,
            ],
            dim=-1,
        )
        paired = self.interaction(interaction)
        pooled_bypass = F.adaptive_avg_pool1d(
            torch.cat([bypass, prefix_mask], dim=1),
            TTM_PATCH_COUNT,
        ).transpose(1, 2)
        bypass_features = F.gelu(self.bypass(pooled_bypass))
        patches = self.patch_output(torch.cat([paired, bypass_features], dim=-1))
        return patches.reshape(horizontal.shape[0], TTM_CONTEXT_LENGTH)


@dataclass(frozen=True)
class TTMHeadConfig:
    variant_id: str
    seed: int
    epochs: int
    batch_size: int
    hidden_dim: int = 32
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip: float = 5.0
    adapter_rank: int = 0

    def __post_init__(self) -> None:
        if self.variant_id not in TTM_VARIANTS:
            raise ValueError(f"unsupported TTM variant: {self.variant_id}")
        if self.epochs < 1 or self.batch_size < 1 or self.hidden_dim < 8:
            raise ValueError("invalid TTM head training dimensions")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if self.variant_id == "ttm_pretrained_adapter":
            if self.adapter_rank < 1:
                raise ValueError("adapter variant requires positive adapter_rank")
        elif self.adapter_rank != 0:
            raise ValueError("only adapter variant may set adapter_rank")


def head_parameter_count(config: TTMHeadConfig, *, bypass_channels: int) -> int:
    devices = [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(config.seed))
        model = TTMPairedInteractionHead(
            bypass_channels=bypass_channels,
            hidden_dim=config.hidden_dim,
            adapter_rank=config.adapter_rank,
        )
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def load_head_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"failed to load TTM head checkpoint: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "ttm_head_v1":
        raise ValueError("TTM head checkpoint schema differs")
    return payload


def fit_ttm_head(
    *,
    train_horizontal: np.ndarray,
    train_typewell: np.ndarray,
    train_batch: TTMInputBatch,
    targets: np.ndarray,
    row_counts: np.ndarray,
    validation_ids: Sequence[str],
    checkpoint_path: str | Path,
    config: TTMHeadConfig,
    provenance: Mapping[str, Any],
    device: torch.device,
    resume: bool,
    max_epochs_this_call: int | None = None,
) -> dict[str, Any]:
    """Fit only the paired head/optional adapter and persist every epoch."""

    if config.variant_id == "ttm_scratch_trainable":
        raise ValueError("scratch-trainable uses fit_ttm_scratch_model")
    fit_ids = tuple(train_batch.ids)
    validation = tuple(map(str, validation_ids))
    if set(fit_ids) & set(validation):
        raise ValueError("TTM head training and validation IDs overlap")
    y = np.asarray(targets, dtype=np.float32)
    counts = np.asarray(row_counts, dtype=np.float32)
    if y.shape != (len(fit_ids), TTM_CONTEXT_LENGTH):
        raise ValueError("TTM targets must have shape [B,512]")
    if counts.shape != (len(fit_ids),) or np.any(counts <= 0):
        raise ValueError("TTM row counts must be positive and align with targets")
    scaler = FoldLocalTTMScaler.fit(
        train_horizontal,
        train_typewell,
        train_batch.bypass,
        fit_ids=fit_ids,
        validation_ids=validation,
    )
    h, t, b = scaler.transform(
        train_horizontal,
        train_typewell,
        train_batch.bypass,
    )
    prefix = np.asarray(train_batch.prefix_mask, dtype=np.float32)
    expected_provenance = json.loads(
        json.dumps(dict(provenance), sort_keys=True, allow_nan=False)
    )
    expected_provenance.update(
        {
            "fit_ids": list(fit_ids),
            "validation_ids": list(validation),
            "fit_validation_intersection": [],
            "outer_validation_labels_used_for_fit": False,
            "scaler_fit_ids": list(fit_ids),
        }
    )
    path = Path(checkpoint_path)
    devices = [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.seed))
        model = TTMPairedInteractionHead(
            bypass_channels=b.shape[1],
            hidden_dim=config.hidden_dim,
            adapter_rank=config.adapter_rank,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 0
    if path.exists():
        if not resume:
            raise FileExistsError(f"TTM checkpoint exists without --resume: {path}")
        payload = load_head_checkpoint(path)
        if (
            payload.get("config") != asdict(config)
            or payload.get("provenance") != expected_provenance
        ):
            raise ValueError("TTM resume checkpoint provenance/config differs")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        restored_scaler = FoldLocalTTMScaler.from_payload(payload["scaler"])
        if restored_scaler.fit_ids != scaler.fit_ids or any(
            not np.array_equal(
                getattr(restored_scaler, name),
                getattr(scaler, name),
            )
            for name in (
                "horizontal_mean",
                "horizontal_scale",
                "typewell_mean",
                "typewell_scale",
                "bypass_mean",
                "bypass_scale",
            )
        ):
            raise ValueError("TTM resume fold-local scaler differs")
        start_epoch = int(payload["epoch"])
        if start_epoch < 0 or start_epoch > config.epochs:
            raise ValueError("TTM resume epoch is invalid")

    stop_epoch = config.epochs
    if max_epochs_this_call is not None:
        if max_epochs_this_call < 1:
            raise ValueError("max_epochs_this_call must be positive")
        stop_epoch = min(stop_epoch, start_epoch + int(max_epochs_this_call))
    tensors = {
        "h": torch.from_numpy(h).to(device),
        "t": torch.from_numpy(t).to(device),
        "b": torch.from_numpy(b).to(device),
        "prefix": torch.from_numpy(prefix).to(device),
        "y": torch.from_numpy(y).to(device),
        "weights": torch.from_numpy(counts / float(np.mean(counts))).to(device),
    }
    final_loss = float("nan")
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(len(fit_ids))
        losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            indices = torch.as_tensor(
                order[start : start + config.batch_size],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                tensors["h"][indices],
                tensors["t"][indices],
                tensors["b"][indices],
                tensors["prefix"][indices],
            )
            sample_mse = torch.mean(
                torch.square(prediction - tensors["y"][indices]),
                dim=1,
            )
            loss = torch.mean(sample_mse * tensors["weights"][indices])
            if not torch.isfinite(loss):
                raise ValueError("TTM head loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        _atomic_torch_save(
            path,
            {
                "schema_version": "ttm_head_v1",
                "config": asdict(config),
                "provenance": expected_provenance,
                "epoch": epoch + 1,
                "completed": epoch + 1 == config.epochs,
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "scaler": scaler.payload(),
                "final_loss": final_loss,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            },
        )
    payload = load_head_checkpoint(path)
    if int(payload["epoch"]) != stop_epoch:
        raise ValueError("TTM checkpoint epoch differs after serialization")
    return payload


def predict_ttm_head(
    *,
    horizontal: np.ndarray,
    typewell: np.ndarray,
    batch: TTMInputBatch,
    checkpoint: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    config = TTMHeadConfig(**dict(checkpoint["config"]))
    scaler = FoldLocalTTMScaler.from_payload(checkpoint["scaler"])
    h, t, b = scaler.transform(horizontal, typewell, batch.bypass)
    devices = [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(config.seed))
        model = TTMPairedInteractionHead(
            bypass_channels=b.shape[1],
            hidden_dim=config.hidden_dim,
            adapter_rank=config.adapter_rank,
        ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(batch.ids), batch_size):
            stop = min(start + batch_size, len(batch.ids))
            prediction = model(
                torch.from_numpy(h[start:stop]).to(device),
                torch.from_numpy(t[start:stop]).to(device),
                torch.from_numpy(b[start:stop]).to(device),
                torch.from_numpy(batch.prefix_mask[start:stop]).to(device),
            )
            output.append(prediction.detach().cpu().numpy())
    result = np.concatenate(output).astype(np.float64)
    if result.shape != (len(batch.ids), TTM_CONTEXT_LENGTH) or not np.isfinite(
        result
    ).all():
        raise ValueError("TTM head returned invalid grid predictions")
    return result


class TTMScratchResidualModel(nn.Module):
    """Official random TTM backbone plus the same paired residual head."""

    def __init__(
        self,
        backbone: nn.Module,
        head: TTMPairedInteractionHead,
        scaler: FoldLocalTTMScaler,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.register_buffer(
            "horizontal_mean",
            torch.from_numpy(scaler.horizontal_mean)[None, None, :],
        )
        self.register_buffer(
            "horizontal_scale",
            torch.from_numpy(scaler.horizontal_scale)[None, None, :],
        )
        self.register_buffer(
            "typewell_mean",
            torch.from_numpy(scaler.typewell_mean)[None, None, :],
        )
        self.register_buffer(
            "typewell_scale",
            torch.from_numpy(scaler.typewell_scale)[None, None, :],
        )
        self.register_buffer(
            "bypass_mean",
            torch.from_numpy(scaler.bypass_mean)[None, :, None],
        )
        self.register_buffer(
            "bypass_scale",
            torch.from_numpy(scaler.bypass_scale)[None, :, None],
        )

    def forward(
        self,
        horizontal: torch.Tensor,
        horizontal_mask: torch.Tensor,
        typewell: torch.Tensor,
        typewell_mask: torch.Tensor,
        bypass: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        horizontal_hidden = self.backbone(
            horizontal,
            past_observed_mask=horizontal_mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state[:, 0]
        typewell_hidden = self.backbone(
            typewell,
            past_observed_mask=typewell_mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state[:, 0]
        h = (horizontal_hidden - self.horizontal_mean) / self.horizontal_scale
        t = (typewell_hidden - self.typewell_mean) / self.typewell_scale
        b = (bypass - self.bypass_mean) / self.bypass_scale
        return self.head(h, t, b, prefix_mask)


def fit_ttm_scratch_model(
    *,
    forecaster: nn.Module,
    train_batch: TTMInputBatch,
    targets: np.ndarray,
    row_counts: np.ndarray,
    validation_ids: Sequence[str],
    checkpoint_path: str | Path,
    config: TTMHeadConfig,
    provenance: Mapping[str, Any],
    device: torch.device,
    resume: bool,
    embedding_batch_size: int,
    max_epochs_this_call: int | None = None,
) -> dict[str, Any]:
    """Train the official same-config random backbone as a capacity control."""

    if config.variant_id != "ttm_scratch_trainable":
        raise ValueError("scratch trainer requires ttm_scratch_trainable config")
    fit_ids = tuple(train_batch.ids)
    validation = tuple(map(str, validation_ids))
    if set(fit_ids) & set(validation):
        raise ValueError("scratch training and validation IDs overlap")
    backbone = forecaster.backbone.to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    initial_h, initial_t = encode_ttm_batch(
        backbone,
        train_batch,
        device=device,
        batch_size=embedding_batch_size,
        require_frozen=False,
    )
    scaler = FoldLocalTTMScaler.fit(
        initial_h,
        initial_t,
        train_batch.bypass,
        fit_ids=fit_ids,
        validation_ids=validation,
    )
    devices = [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(config.seed))
        head = TTMPairedInteractionHead(
            bypass_channels=train_batch.bypass.shape[1],
            hidden_dim=config.hidden_dim,
            adapter_rank=0,
        )
    model = TTMScratchResidualModel(backbone, head, scaler).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    y = np.asarray(targets, dtype=np.float32)
    counts = np.asarray(row_counts, dtype=np.float32)
    if y.shape != (len(fit_ids), 512) or counts.shape != (len(fit_ids),):
        raise ValueError("scratch targets/row counts have invalid shapes")
    expected_provenance = json.loads(
        json.dumps(dict(provenance), sort_keys=True, allow_nan=False)
    )
    expected_provenance.update(
        {
            "fit_ids": list(fit_ids),
            "validation_ids": list(validation),
            "fit_validation_intersection": [],
            "outer_validation_labels_used_for_fit": False,
            "scaler_fit_ids": list(fit_ids),
        }
    )
    path = Path(checkpoint_path)
    start_epoch = 0
    if path.exists():
        if not resume:
            raise FileExistsError(f"scratch checkpoint exists: {path}")
        payload = load_head_checkpoint(path)
        if (
            payload.get("mode") != "scratch_trainable"
            or payload.get("config") != asdict(config)
            or payload.get("provenance") != expected_provenance
        ):
            raise ValueError("scratch resume checkpoint differs")
        model.load_state_dict(payload["scratch_model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"])
    stop_epoch = config.epochs
    if max_epochs_this_call is not None:
        stop_epoch = min(
            stop_epoch,
            start_epoch + int(max_epochs_this_call),
        )
    arrays = {
        "horizontal": torch.from_numpy(train_batch.horizontal).to(device),
        "horizontal_mask": torch.from_numpy(train_batch.horizontal_mask).to(device),
        "typewell": torch.from_numpy(train_batch.typewell).to(device),
        "typewell_mask": torch.from_numpy(train_batch.typewell_mask).to(device),
        "bypass": torch.from_numpy(train_batch.bypass).to(device),
        "prefix": torch.from_numpy(train_batch.prefix_mask).to(device),
        "target": torch.from_numpy(y).to(device),
        "weights": torch.from_numpy(counts / float(np.mean(counts))).to(device),
    }
    final_loss = float("nan")
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(len(fit_ids))
        losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            indices = torch.as_tensor(
                order[start : start + config.batch_size],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                arrays["horizontal"][indices],
                arrays["horizontal_mask"][indices],
                arrays["typewell"][indices],
                arrays["typewell_mask"][indices],
                arrays["bypass"][indices],
                arrays["prefix"][indices],
            )
            sample_mse = torch.mean(
                torch.square(prediction - arrays["target"][indices]),
                dim=1,
            )
            loss = torch.mean(sample_mse * arrays["weights"][indices])
            if not torch.isfinite(loss):
                raise ValueError("scratch loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        _atomic_torch_save(
            path,
            {
                "schema_version": "ttm_head_v1",
                "mode": "scratch_trainable",
                "config": asdict(config),
                "provenance": expected_provenance,
                "epoch": epoch + 1,
                "completed": epoch + 1 == config.epochs,
                "scratch_model_state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "scaler": scaler.payload(),
                "final_loss": final_loss,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            },
        )
    return load_head_checkpoint(path)


def predict_ttm_scratch_model(
    *,
    forecaster: nn.Module,
    batch: TTMInputBatch,
    checkpoint: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if checkpoint.get("mode") != "scratch_trainable":
        raise ValueError("checkpoint is not a scratch-trainable TTM model")
    config = TTMHeadConfig(**dict(checkpoint["config"]))
    scaler = FoldLocalTTMScaler.from_payload(checkpoint["scaler"])
    head = TTMPairedInteractionHead(
        bypass_channels=batch.bypass.shape[1],
        hidden_dim=config.hidden_dim,
        adapter_rank=0,
    )
    model = TTMScratchResidualModel(
        forecaster.backbone,
        head,
        scaler,
    ).to(device)
    model.load_state_dict(checkpoint["scratch_model_state"])
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(batch.ids), batch_size):
            stop = min(start + batch_size, len(batch.ids))
            prediction = model(
                torch.from_numpy(batch.horizontal[start:stop]).to(device),
                torch.from_numpy(batch.horizontal_mask[start:stop]).to(device),
                torch.from_numpy(batch.typewell[start:stop]).to(device),
                torch.from_numpy(batch.typewell_mask[start:stop]).to(device),
                torch.from_numpy(batch.bypass[start:stop]).to(device),
                torch.from_numpy(batch.prefix_mask[start:stop]).to(device),
            )
            output.append(prediction.detach().cpu().numpy())
    result = np.concatenate(output).astype(np.float64)
    if result.shape != (len(batch.ids), 512) or not np.isfinite(result).all():
        raise ValueError("scratch TTM prediction is invalid")
    return result


__all__ = [
    "ELIGIBLE_FOR_MODEL_SELECTION",
    "FROZEN_VARIANTS",
    "FoldLocalTTMScaler",
    "OPTIONAL_VARIANTS",
    "PRETRAINED_VARIANTS",
    "TTMHeadConfig",
    "TTMInputBatch",
    "TTMPairedInteractionHead",
    "TTM_VARIANTS",
    "VARIANT_CONTROL",
    "apply_ttm_control",
    "encode_ttm_batch",
    "fit_ttm_head",
    "fit_ttm_scratch_model",
    "head_parameter_count",
    "load_head_checkpoint",
    "make_ttm_batch",
    "predict_ttm_head",
    "predict_ttm_scratch_model",
]
