"""Time-series metric storage and Prometheus-compatible export.

Provides persistent metric storage, historical querying, and
Prometheus-format export for feature observability dashboards.

Example:
    >>> from forge.monitoring.metrics_store import MetricsStore
    >>> store = MetricsStore(storage_path="./metrics")
    >>> store.record("user_age", {"mean": 34.5, "null_rate": 0.02})
    >>> history = store.query("user_age", last_n=100)
    >>> prometheus_text = store.export_prometheus()
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric observation at a point in time."""

    feature_name: str
    metric_name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "feature_name": self.feature_name,
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricPoint:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class HealthStatus:
    """Health status for a single feature."""

    feature_name: str
    status: str  # "healthy", "warning", "critical"
    drift_score: float = 0.0
    null_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    issues: list[str] = field(default_factory=list)


@dataclass
class GovernanceReport:
    """Compliance/governance report for features."""

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_features: int = 0
    healthy_count: int = 0
    warning_count: int = 0
    critical_count: int = 0
    feature_statuses: list[HealthStatus] = field(default_factory=list)
    sla_violations: list[str] = field(default_factory=list)
    drift_summary: dict[str, float] = field(default_factory=dict)

    @property
    def compliance_rate(self) -> float:
        """Percentage of features in healthy state."""
        if self.total_features == 0:
            return 1.0
        return self.healthy_count / self.total_features

    def to_dict(self) -> dict[str, Any]:
        """Serialize for export."""
        return {
            "generated_at": self.generated_at,
            "total_features": self.total_features,
            "healthy_count": self.healthy_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "compliance_rate": round(self.compliance_rate, 4),
            "sla_violations": self.sla_violations,
            "drift_summary": self.drift_summary,
        }

    def to_markdown(self) -> str:
        """Export as markdown report."""
        lines = [
            "# Feature Governance Report",
            f"**Generated**: {self.generated_at}",
            "",
            "## Summary",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Features | {self.total_features} |",
            f"| Healthy | {self.healthy_count} |",
            f"| Warning | {self.warning_count} |",
            f"| Critical | {self.critical_count} |",
            f"| Compliance Rate | {self.compliance_rate:.1%} |",
            "",
        ]

        if self.sla_violations:
            lines.append("## SLA Violations")
            for v in self.sla_violations:
                lines.append(f"- {v}")
            lines.append("")

        if self.drift_summary:
            lines.append("## Drift Summary")
            lines.append("| Feature | PSI Score |")
            lines.append("|---------|-----------|")
            for feat, score in sorted(self.drift_summary.items(), key=lambda x: -x[1]):
                lines.append(f"| {feat} | {score:.4f} |")

        return "\n".join(lines)


class MetricsStore:
    """Time-series metric storage for feature observability.

    Stores metric observations over time, supports querying
    historical data, and exports in Prometheus format.

    Parameters:
        storage_path: Path for persistent storage. None for in-memory.
        max_history: Maximum observations per feature-metric pair.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        max_history: int = 10000,
    ) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_history = max_history
        self._data: dict[str, list[MetricPoint]] = defaultdict(list)
        if self.storage_path:
            self._load()

    def record(
        self,
        feature_name: str,
        metrics: dict[str, float],
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record metric observations for a feature.

        Args:
            feature_name: Feature name.
            metrics: Dictionary of metric_name -> value.
            labels: Optional labels for the observation.
        """
        timestamp = time.time()
        for metric_name, value in metrics.items():
            key = f"{feature_name}:{metric_name}"
            point = MetricPoint(
                feature_name=feature_name,
                metric_name=metric_name,
                value=value,
                timestamp=timestamp,
                labels=labels or {},
            )
            self._data[key].append(point)

            # Enforce max history
            if len(self._data[key]) > self.max_history:
                self._data[key] = self._data[key][-self.max_history:]

    def query(
        self,
        feature_name: str,
        metric_name: str | None = None,
        last_n: int | None = None,
        since: float | None = None,
    ) -> list[MetricPoint]:
        """Query stored metrics.

        Args:
            feature_name: Feature to query.
            metric_name: Optional specific metric.
            last_n: Return only the last N observations.
            since: Only return observations after this timestamp.

        Returns:
            List of MetricPoint observations.
        """
        results: list[MetricPoint] = []

        for key, points in self._data.items():
            for point in points:
                if point.feature_name != feature_name:
                    continue
                if metric_name and point.metric_name != metric_name:
                    continue
                if since and point.timestamp < since:
                    continue
                results.append(point)

        results.sort(key=lambda p: p.timestamp)
        if last_n:
            results = results[-last_n:]
        return results

    def get_latest(self, feature_name: str) -> dict[str, float]:
        """Get the latest value for each metric of a feature.

        Args:
            feature_name: Feature name.

        Returns:
            Dictionary of metric_name -> latest value.
        """
        latest: dict[str, float] = {}
        for key, points in self._data.items():
            if not points:
                continue
            last = points[-1]
            if last.feature_name == feature_name:
                latest[last.metric_name] = last.value
        return latest

    def feature_health(
        self,
        feature_name: str,
        drift_threshold: float = 0.2,
        null_threshold: float = 0.1,
    ) -> HealthStatus:
        """Assess health status of a feature.

        Args:
            feature_name: Feature to assess.
            drift_threshold: PSI threshold for drift warning.
            null_threshold: Null rate threshold for warning.

        Returns:
            HealthStatus for the feature.
        """
        latest = self.get_latest(feature_name)
        issues: list[str] = []
        status = "healthy"

        drift_score = latest.get("psi", 0.0)
        null_rate = latest.get("null_rate", 0.0)

        if drift_score > drift_threshold * 2:
            status = "critical"
            issues.append(f"High drift detected (PSI={drift_score:.4f})")
        elif drift_score > drift_threshold:
            status = "warning"
            issues.append(f"Moderate drift detected (PSI={drift_score:.4f})")

        if null_rate > null_threshold * 2:
            if status != "critical":
                status = "critical"
            issues.append(f"High null rate ({null_rate:.1%})")
        elif null_rate > null_threshold:
            if status == "healthy":
                status = "warning"
            issues.append(f"Elevated null rate ({null_rate:.1%})")

        return HealthStatus(
            feature_name=feature_name,
            status=status,
            drift_score=drift_score,
            null_rate=null_rate,
            issues=issues,
        )

    def generate_governance_report(
        self,
        feature_names: list[str],
        drift_threshold: float = 0.2,
        null_threshold: float = 0.1,
    ) -> GovernanceReport:
        """Generate a governance/compliance report.

        Args:
            feature_names: Features to include in the report.
            drift_threshold: PSI threshold for drift warnings.
            null_threshold: Null rate threshold for warnings.

        Returns:
            GovernanceReport with health summary.
        """
        statuses: list[HealthStatus] = []
        drift_summary: dict[str, float] = {}

        for name in feature_names:
            health = self.feature_health(name, drift_threshold, null_threshold)
            statuses.append(health)
            if health.drift_score > 0:
                drift_summary[name] = health.drift_score

        healthy = sum(1 for s in statuses if s.status == "healthy")
        warning = sum(1 for s in statuses if s.status == "warning")
        critical = sum(1 for s in statuses if s.status == "critical")

        violations: list[str] = []
        for s in statuses:
            for issue in s.issues:
                violations.append(f"{s.feature_name}: {issue}")

        return GovernanceReport(
            total_features=len(feature_names),
            healthy_count=healthy,
            warning_count=warning,
            critical_count=critical,
            feature_statuses=statuses,
            sla_violations=violations,
            drift_summary=drift_summary,
        )

    def export_prometheus(self, prefix: str = "forge_feature") -> str:
        """Export metrics in Prometheus text format.

        Args:
            prefix: Metric name prefix.

        Returns:
            Prometheus exposition format text.
        """
        lines: list[str] = []
        seen_metrics: set[str] = set()

        for key, points in self._data.items():
            if not points:
                continue
            last = points[-1]
            metric_id = f"{prefix}_{last.metric_name}"

            if metric_id not in seen_metrics:
                lines.append(f"# HELP {metric_id} Feature metric: {last.metric_name}")
                lines.append(f"# TYPE {metric_id} gauge")
                seen_metrics.add(metric_id)

            label_str = f'feature="{last.feature_name}"'
            if last.labels:
                extra = ",".join(f'{k}="{v}"' for k, v in last.labels.items())
                label_str = f"{label_str},{extra}"

            lines.append(f"{metric_id}{{{label_str}}} {last.value}")

        return "\n".join(lines) + "\n" if lines else ""

    @property
    def feature_names(self) -> list[str]:
        """Get all tracked feature names."""
        names: set[str] = set()
        for points in self._data.values():
            if points:
                names.add(points[-1].feature_name)
        return sorted(names)

    @property
    def size(self) -> int:
        """Total number of stored metric points."""
        return sum(len(v) for v in self._data.values())

    def save(self) -> None:
        """Persist metrics to disk."""
        if self.storage_path is None:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        data_path = self.storage_path / "metrics.json"
        serialized: dict[str, list[dict]] = {}
        for key, points in self._data.items():
            serialized[key] = [p.to_dict() for p in points[-self.max_history:]]
        data_path.write_text(json.dumps(serialized))

    def _load(self) -> None:
        """Load metrics from disk."""
        if self.storage_path is None:
            return
        data_path = self.storage_path / "metrics.json"
        if not data_path.exists():
            return
        try:
            raw = json.loads(data_path.read_text())
            for key, points in raw.items():
                self._data[key] = [MetricPoint.from_dict(p) for p in points]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load metrics: %s", e)
