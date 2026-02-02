"""Neural feature architecture for automatic feature synthesis.

Provides neural feature crosses, learned embeddings for categorical
variables, and neural-based feature transformation discovery.
Optional dependency on PyTorch.
"""

from __future__ import annotations

from forge.neural.features import (
    NeuralEmbeddingGenerator,
    NeuralFeatureCross,
    NeuralFeatureGenerator,
)

__all__ = [
    "NeuralEmbeddingGenerator",
    "NeuralFeatureCross",
    "NeuralFeatureGenerator",
]
