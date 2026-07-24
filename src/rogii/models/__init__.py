"""Active model registry."""

from __future__ import annotations

from .base import Model
from .flat import FlatModel
from .p3 import P3Model


MODEL_IDS = ("flat", "p3")


def get_model(model_id: str) -> Model:
    if model_id == "flat":
        return FlatModel()
    if model_id == "p3":
        return P3Model()
    raise KeyError(f"Unknown model {model_id!r}; expected one of {MODEL_IDS}")


__all__ = ["MODEL_IDS", "Model", "get_model"]
