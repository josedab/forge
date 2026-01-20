"""Text feature generators for Forge."""

from __future__ import annotations

from forge.generators.text.basic import TextBasicFeatures
from forge.generators.text.tfidf import TfidfGenerator

__all__ = [
    "TextBasicFeatures",
    "TfidfGenerator",
]
