"""Interactive Feature Studio for visual feature engineering.

Provides a programmatic API for building, previewing, comparing,
and exporting feature pipelines. Can be used standalone or with
Streamlit for visual interaction.
"""

from __future__ import annotations

from forge.studio.app import (
    AppConfig,
    ColumnAnalysis,
    PipelineConfig,
    analyze_dataframe,
    compare_dataframes,
    create_streamlit_app_code,
    generate_distribution_summary,
    launch_studio,
)
from forge.studio.builder import (
    ComparisonResult,
    FeatureStudio,
    PipelineStep,
    PreviewResult,
)

__all__ = [
    "AppConfig",
    "ColumnAnalysis",
    "ComparisonResult",
    "FeatureStudio",
    "PipelineConfig",
    "PipelineStep",
    "PreviewResult",
    "analyze_dataframe",
    "compare_dataframes",
    "create_streamlit_app_code",
    "generate_distribution_summary",
    "launch_studio",
]
