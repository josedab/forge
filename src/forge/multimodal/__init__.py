"""Multi-modal feature fusion for tabular, image, and signal data.

Provides unified feature extraction from multiple data modalities
and cross-modal feature fusion capabilities.
"""

from __future__ import annotations

from forge.multimodal.features import (
    ImageFeatureExtractor,
    ModalityType,
    MultiModalFusion,
    SignalFeatureExtractor,
)

__all__ = [
    "ImageFeatureExtractor",
    "ModalityType",
    "MultiModalFusion",
    "SignalFeatureExtractor",
]
