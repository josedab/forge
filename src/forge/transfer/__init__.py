"""Cross-dataset feature transfer.

Learn reusable feature transformations from one dataset and apply
them to similar datasets using feature fingerprinting and matching.
"""

from __future__ import annotations

from forge.transfer.engine import (
    ColumnFingerprint,
    DatasetFingerprint,
    FeatureTransferEngine,
    TransferResult,
)
from forge.transfer.hub import (
    AdaptationResult,
    DiscoveryMatch,
    PipelineRecord,
    TransferHub,
)

__all__ = [
    "AdaptationResult",
    "ColumnFingerprint",
    "DatasetFingerprint",
    "DiscoveryMatch",
    "FeatureTransferEngine",
    "PipelineRecord",
    "TransferHub",
    "TransferResult",
]
