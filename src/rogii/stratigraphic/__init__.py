"""Inference-safe stratigraphic descriptors and loop-edge generation."""

from .descriptors import (
    DescriptorBatch,
    DescriptorConfig,
    ResampledWindow,
    build_descriptor_batch,
    resample_gr_window,
)
from .loop_edges import EdgeGenerationConfig, EdgeGenerationResult, generate_loop_edges

__all__ = [
    "DescriptorBatch",
    "DescriptorConfig",
    "EdgeGenerationConfig",
    "EdgeGenerationResult",
    "ResampledWindow",
    "build_descriptor_batch",
    "generate_loop_edges",
    "resample_gr_window",
]
