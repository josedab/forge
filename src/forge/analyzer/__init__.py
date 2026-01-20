"""Data analysis module for Forge."""

from __future__ import annotations

from forge.analyzer.base import DataAnalyzer
from forge.analyzer.quality import QualityAssessor
from forge.analyzer.report import ReportBuilder
from forge.analyzer.statistics import StatisticsProfiler
from forge.analyzer.type_inference import TypeInferrer

__all__ = [
    "DataAnalyzer",
    "TypeInferrer",
    "StatisticsProfiler",
    "QualityAssessor",
    "ReportBuilder",
]
