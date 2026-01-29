"""Feature observability platform for production monitoring.

Provides continuous feature monitoring, anomaly detection on feature values,
alerting via configurable channels, and feature SLA tracking with dashboards.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
from sklearn.base import BaseEstimator

from forge.monitoring.drift import calculate_psi

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of feature metrics tracked."""

    NULL_RATE = "null_rate"
    MEAN = "mean"
    STD = "std"
    MIN = "min"
    MAX = "max"
    CARDINALITY = "cardinality"
    PSI = "psi"
    OUT_OF_RANGE = "out_of_range"


@dataclass
class AlertRule:
    """Configurable alert rule for feature monitoring.

    Attributes:
    ----------
    name : str
        Human-readable rule name.
    metric : MetricType
        The metric to evaluate.
    condition : str
        Condition operator: 'gt', 'lt', 'gte', 'lte', 'eq'.
    threshold : float
        Threshold value for the condition.
    severity : AlertSeverity
        Alert severity when triggered.
    columns : list[str] | None
        Columns to apply this rule to. None means all columns.
    """

    name: str
    metric: MetricType
    condition: Literal["gt", "lt", "gte", "lte", "eq"]
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    columns: list[str] | None = None


@dataclass
class Alert:
    """A triggered alert from the observability system.

    Attributes:
    ----------
    rule_name : str
        Name of the alert rule that triggered.
    feature : str
        Feature name that triggered the alert.
    metric : MetricType
        Metric type that was evaluated.
    value : float
        Actual metric value.
    threshold : float
        Threshold that was exceeded.
    severity : AlertSeverity
        Severity of the alert.
    timestamp : float
        Unix timestamp when alert was created.
    message : str
        Human-readable alert message.
    """

    rule_name: str
    feature: str
    metric: MetricType
    value: float
    threshold: float
    severity: AlertSeverity
    timestamp: float = 0.0
    message: str = ""

    def __post_init__(self) -> None:
        """Set timestamp and message if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not self.message:
            self.message = (
                f"[{self.severity.value.upper()}] {self.rule_name}: "
                f"Feature '{self.feature}' {self.metric.value}={self.value:.4f} "
                f"exceeded threshold {self.threshold:.4f}"
            )


@dataclass
class FeatureMetrics:
    """Computed metrics for a single feature at a point in time.

    Attributes:
    ----------
    feature_name : str
        Name of the feature.
    timestamp : float
        Unix timestamp of computation.
    metrics : dict[str, float]
        Map of metric name to value.
    """

    feature_name: str
    timestamp: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SLADefinition:
    """Feature SLA definition.

    Attributes:
    ----------
    name : str
        SLA name.
    description : str
        What this SLA measures.
    metric : MetricType
        Metric to track.
    target : float
        Target value (e.g., null_rate < 0.05).
    condition : str
        'lt', 'gt', 'lte', 'gte' — the condition for compliance.
    window_size : int
        Number of recent checks to consider for compliance rate.
    """

    name: str
    description: str
    metric: MetricType
    target: float
    condition: Literal["gt", "lt", "gte", "lte"] = "lt"
    window_size: int = 10


@dataclass
class SLAReport:
    """SLA compliance report.

    Attributes:
    ----------
    sla_name : str
        Name of the SLA.
    feature : str
        Feature being evaluated.
    compliant : bool
        Whether currently compliant.
    compliance_rate : float
        Fraction of recent checks that were compliant (0.0-1.0).
    current_value : float
        Current metric value.
    target : float
        SLA target.
    history : list[bool]
        Recent compliance history.
    """

    sla_name: str
    feature: str
    compliant: bool
    compliance_rate: float
    current_value: float
    target: float
    history: list[bool] = field(default_factory=list)


class FeatureObserver(BaseEstimator):
    """Continuous feature observability platform.

    Monitors feature quality metrics, evaluates alert rules, tracks SLA
    compliance, and dispatches notifications to configured channels.

    Parameters
    ----------
    columns : list[str] | None
        Columns to monitor. None monitors all numeric columns.
    alert_rules : list[AlertRule] | None
        Alert rules to evaluate on each check.
    sla_definitions : list[SLADefinition] | None
        SLA definitions to track compliance.
    notification_handlers : list[Callable[[Alert], None]] | None
        Callables that receive Alert objects for dispatching notifications.
    history_size : int
        Max number of metric snapshots to retain per feature.

    Attributes:
    ----------
    baseline_stats_ : dict[str, dict[str, float]]
        Baseline statistics computed during fit.
    metrics_history_ : dict[str, list[FeatureMetrics]]
        Historical metrics per feature.
    alerts_ : list[Alert]
        All triggered alerts.
    sla_reports_ : dict[str, dict[str, SLAReport]]
        Latest SLA reports keyed by sla_name then feature.

    Examples:
    --------
    >>> from forge.monitoring.observability import FeatureObserver, AlertRule
    >>> from forge.monitoring.observability import MetricType, AlertSeverity
    >>>
    >>> rules = [
    ...     AlertRule("high_null", MetricType.NULL_RATE, "gt", 0.1,
    ...              AlertSeverity.CRITICAL),
    ...     AlertRule("drift", MetricType.PSI, "gt", 0.2,
    ...              AlertSeverity.WARNING),
    ... ]
    >>> observer = FeatureObserver(alert_rules=rules)
    >>> observer.fit(X_train)
    >>> observer.check(X_production)
    >>> print(observer.get_summary())
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        alert_rules: list[AlertRule] | None = None,
        sla_definitions: list[SLADefinition] | None = None,
        notification_handlers: list[Callable[[Alert], None]] | None = None,
        history_size: int = 100,
    ) -> None:
        self.columns = columns
        self.alert_rules = alert_rules or []
        self.sla_definitions = sla_definitions or []
        self.notification_handlers = notification_handlers or []
        self.history_size = history_size

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Fit observer on baseline data.

        Computes baseline statistics used for drift detection and
        out-of-range checks.

        Parameters
        ----------
        X : pd.DataFrame
            Baseline (training) data.
        y : Any
            Ignored.

        Returns:
        -------
        Self
            Fitted observer.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(X).__name__}")

        if self.columns is not None:
            self.feature_names_in_ = [c for c in self.columns if c in X.columns]
        else:
            self.feature_names_in_ = list(X.select_dtypes(include=[np.number]).columns)

        self.baseline_stats_: dict[str, dict[str, float]] = {}
        self._baseline_values: dict[str, np.ndarray] = {}

        for col in self.feature_names_in_:
            values = X[col].dropna().values.astype(float)
            self._baseline_values[col] = values
            self.baseline_stats_[col] = {
                "mean": float(np.mean(values)) if len(values) > 0 else 0.0,
                "std": float(np.std(values)) if len(values) > 0 else 0.0,
                "min": float(np.min(values)) if len(values) > 0 else 0.0,
                "max": float(np.max(values)) if len(values) > 0 else 0.0,
                "null_rate": float(X[col].isna().mean()),
                "cardinality": float(X[col].nunique()),
            }

        self.metrics_history_: dict[str, list[FeatureMetrics]] = {
            col: [] for col in self.feature_names_in_
        }
        self.alerts_: list[Alert] = []
        self.sla_reports_: dict[str, dict[str, SLAReport]] = {}
        self._sla_compliance_history: dict[str, dict[str, list[bool]]] = {}

        self._is_fitted = True
        return self

    def check(self, X: pd.DataFrame) -> list[Alert]:
        """Check features against baseline and evaluate alert rules.

        Parameters
        ----------
        X : pd.DataFrame
            Production data to check.

        Returns:
        -------
        list[Alert]
            Alerts triggered during this check.
        """
        import pandas as pd

        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("FeatureObserver must be fitted before check().")

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(X).__name__}")

        now = time.time()
        new_alerts: list[Alert] = []

        for col in self.feature_names_in_:
            if col not in X.columns:
                continue

            metrics = self._compute_metrics(X, col, now)

            self.metrics_history_[col].append(metrics)
            if len(self.metrics_history_[col]) > self.history_size:
                self.metrics_history_[col] = self.metrics_history_[col][
                    -self.history_size :
                ]

            col_alerts = self._evaluate_rules(col, metrics)
            new_alerts.extend(col_alerts)

        self.alerts_.extend(new_alerts)

        for handler in self.notification_handlers:
            for alert in new_alerts:
                try:
                    handler(alert)
                except Exception:
                    logger.exception("Notification handler failed for alert: %s", alert.rule_name)

        self._update_sla_reports()

        return new_alerts

    def _compute_metrics(
        self, X: pd.DataFrame, col: str, timestamp: float
    ) -> FeatureMetrics:
        """Compute all metrics for a single feature."""
        values = X[col].dropna().values.astype(float)
        total = len(X[col])

        metrics: dict[str, float] = {}
        metrics["null_rate"] = float(X[col].isna().sum() / total) if total > 0 else 0.0
        if len(values) > 0:
            metrics["mean"] = float(np.mean(values))
            metrics["std"] = float(np.std(values))
            metrics["min"] = float(np.min(values))
            metrics["max"] = float(np.max(values))
            metrics["cardinality"] = float(len(np.unique(values)))
        else:
            metrics["mean"] = 0.0
            metrics["std"] = 0.0
            metrics["min"] = 0.0
            metrics["max"] = 0.0
            metrics["cardinality"] = 0.0

        # PSI against baseline
        if col in self._baseline_values and len(values) > 0:
            metrics["psi"] = calculate_psi(self._baseline_values[col], values)
        else:
            metrics["psi"] = 0.0

        # Out of range rate
        if col in self.baseline_stats_ and len(values) > 0:
            bl = self.baseline_stats_[col]
            out_of_range = np.sum((values < bl["min"]) | (values > bl["max"]))
            metrics["out_of_range"] = float(out_of_range / len(values))
        else:
            metrics["out_of_range"] = 0.0

        return FeatureMetrics(feature_name=col, timestamp=timestamp, metrics=metrics)

    def _evaluate_rules(self, col: str, metrics: FeatureMetrics) -> list[Alert]:
        """Evaluate alert rules against computed metrics."""
        alerts: list[Alert] = []

        for rule in self.alert_rules:
            if rule.columns is not None and col not in rule.columns:
                continue

            metric_key = rule.metric.value
            if metric_key not in metrics.metrics:
                continue

            value = metrics.metrics[metric_key]
            triggered = self._check_condition(value, rule.condition, rule.threshold)

            if triggered:
                alerts.append(
                    Alert(
                        rule_name=rule.name,
                        feature=col,
                        metric=rule.metric,
                        value=value,
                        threshold=rule.threshold,
                        severity=rule.severity,
                    )
                )

        return alerts

    @staticmethod
    def _check_condition(value: float, condition: str, threshold: float) -> bool:
        """Check if a value meets an alert condition."""
        if np.isnan(value):
            return False
        if condition == "gt":
            return value > threshold
        if condition == "lt":
            return value < threshold
        if condition == "gte":
            return value >= threshold
        if condition == "lte":
            return value <= threshold
        if condition == "eq":
            return abs(value - threshold) < 1e-10
        return False

    def _update_sla_reports(self) -> None:
        """Update SLA compliance reports based on latest metrics."""
        for sla in self.sla_definitions:
            if sla.name not in self._sla_compliance_history:
                self._sla_compliance_history[sla.name] = {}
            if sla.name not in self.sla_reports_:
                self.sla_reports_[sla.name] = {}

            for col in self.feature_names_in_:
                history = self.metrics_history_.get(col, [])
                if not history:
                    continue

                latest = history[-1]
                metric_key = sla.metric.value
                if metric_key not in latest.metrics:
                    continue

                current_value = latest.metrics[metric_key]
                compliant = self._check_condition(
                    current_value, sla.condition, sla.target
                )

                if col not in self._sla_compliance_history[sla.name]:
                    self._sla_compliance_history[sla.name][col] = []

                sla_history = self._sla_compliance_history[sla.name][col]
                sla_history.append(compliant)
                if len(sla_history) > sla.window_size:
                    sla_history[:] = sla_history[-sla.window_size :]

                compliance_rate = sum(sla_history) / len(sla_history)

                self.sla_reports_[sla.name][col] = SLAReport(
                    sla_name=sla.name,
                    feature=col,
                    compliant=compliant,
                    compliance_rate=compliance_rate,
                    current_value=current_value,
                    target=sla.target,
                    history=list(sla_history),
                )

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive observability summary.

        Returns:
        -------
        dict
            Summary including metrics, alerts, and SLA status.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            return {"error": "Observer not fitted."}

        latest_metrics: dict[str, dict[str, float]] = {}
        for col, history in self.metrics_history_.items():
            if history:
                latest_metrics[col] = history[-1].metrics

        alert_counts = {s.value: 0 for s in AlertSeverity}
        for alert in self.alerts_:
            alert_counts[alert.severity.value] += 1

        sla_summary: dict[str, dict[str, bool]] = {}
        for sla_name, features in self.sla_reports_.items():
            sla_summary[sla_name] = {
                feat: report.compliant for feat, report in features.items()
            }

        return {
            "features_monitored": len(self.feature_names_in_),
            "total_checks": max(
                (len(h) for h in self.metrics_history_.values()), default=0
            ),
            "latest_metrics": latest_metrics,
            "total_alerts": len(self.alerts_),
            "alert_counts": alert_counts,
            "sla_compliance": sla_summary,
        }

    def get_alerts(
        self, severity: AlertSeverity | None = None, feature: str | None = None
    ) -> list[Alert]:
        """Get filtered alerts.

        Parameters
        ----------
        severity : AlertSeverity | None
            Filter by severity.
        feature : str | None
            Filter by feature name.

        Returns:
        -------
        list[Alert]
            Filtered alerts.
        """
        alerts = self.alerts_
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        if feature is not None:
            alerts = [a for a in alerts if a.feature == feature]
        return alerts

    def export_metrics(self, format: Literal["json", "dict"] = "dict") -> Any:
        """Export all metrics history.

        Parameters
        ----------
        format : str
            Output format: 'json' or 'dict'.

        Returns:
        -------
        Any
            Metrics data in the requested format.
        """
        data: dict[str, list[dict[str, Any]]] = {}
        for col, history in self.metrics_history_.items():
            data[col] = [
                {"timestamp": m.timestamp, "metrics": m.metrics} for m in history
            ]

        if format == "json":
            return json.dumps(data, default=str)
        return data
