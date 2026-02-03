"""Explainable feature lineage dashboard and dataset profiling.

Provides lineage DAG construction, interactive visualization,
export to MLflow/Weights & Biases, dataset profiling, and
lineage versioning with snapshot/diff capabilities.
"""

from __future__ import annotations

from forge.dashboard.app import create_dashboard
from forge.dashboard.export import HTMLReportExporter, MLflowExporter, WandbExporter
from forge.dashboard.lineage import FeatureLineageGraph, LineageEdge, LineageNode
from forge.dashboard.profiler import (
    ColumnProfile,
    DatasetProfile,
    DatasetProfiler,
    create_profiling_dashboard,
)
from forge.dashboard.versioning import (
    LineageDiff,
    LineageEvent,
    LineageEventRecorder,
    LineageSnapshot,
    LineageVersionStore,
)

__all__ = [
    "ColumnProfile",
    "DatasetProfile",
    "DatasetProfiler",
    "FeatureLineageGraph",
    "HTMLReportExporter",
    "LineageDiff",
    "LineageEdge",
    "LineageEvent",
    "LineageEventRecorder",
    "LineageNode",
    "LineageSnapshot",
    "LineageVersionStore",
    "MLflowExporter",
    "WandbExporter",
    "create_dashboard",
    "create_profiling_dashboard",
]
