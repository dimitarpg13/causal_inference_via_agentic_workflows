"""Synthetic data generators for causal inference demonstrations."""

from causal_inference.data.synthetic import (
    generate_confounded_data,
    generate_frontdoor_data,
    generate_collider_data,
    generate_rct_data,
    generate_iv_data,
)

__all__ = [
    "generate_confounded_data",
    "generate_frontdoor_data",
    "generate_collider_data",
    "generate_rct_data",
    "generate_iv_data",
]
