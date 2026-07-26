"""Active model registry."""

from __future__ import annotations

from .base import Model
from .flat import FlatModel
from .path_pca import PathPCAModel
from .path_pca_anchor import AnchorPathPCAModel
from .path_pca_nested import NestedPathPCAModel
from .p3 import P3Model
from .probabilistic_lattice import ProbabilisticLatticeModel


MODEL_IDS = (
    "flat",
    "p3",
    "lattice",
    "path_pca",
    "path_pca_anchor",
    "path_pca_nested",
)


def get_model(model_id: str) -> Model:
    if model_id == "flat":
        return FlatModel()
    if model_id == "p3":
        return P3Model()
    if model_id == "lattice":
        return ProbabilisticLatticeModel()
    if model_id == "path_pca":
        return PathPCAModel()
    if model_id == "path_pca_anchor":
        return AnchorPathPCAModel()
    if model_id == "path_pca_nested":
        return NestedPathPCAModel()
    raise KeyError(f"Unknown model {model_id!r}; expected one of {MODEL_IDS}")


__all__ = ["MODEL_IDS", "Model", "get_model"]
