"""Domain-specific feature packs.

Pre-built feature generation recipes for specific verticals:
finance, healthcare, e-commerce, geospatial, IoT, and marketing.
"""

from __future__ import annotations

from forge.packs.base import FeaturePack, FeaturePackRegistry
from forge.packs.ecommerce import EcommerceFeaturePack
from forge.packs.finance import FinanceFeaturePack
from forge.packs.geospatial import GeospatialFeaturePack
from forge.packs.healthcare import HealthcareFeaturePack
from forge.packs.iot import IoTFeaturePack, MarketingFeaturePack

__all__ = [
    "EcommerceFeaturePack",
    "FeaturePack",
    "FeaturePackRegistry",
    "FinanceFeaturePack",
    "GeospatialFeaturePack",
    "HealthcareFeaturePack",
    "IoTFeaturePack",
    "MarketingFeaturePack",
]
