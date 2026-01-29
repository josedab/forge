"""Tests for feature observability platform."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.monitoring.observability import (
    Alert,
    AlertRule,
    AlertSeverity,
    FeatureObserver,
    MetricType,
    SLADefinition,
)


@pytest.fixture
def sample_baseline():
    np.random.seed(42)
    return pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 1000),
        "feature_b": np.random.normal(5, 2, 1000),
        "feature_c": np.random.uniform(0, 10, 1000),
    })


@pytest.fixture
def sample_production_stable(sample_baseline):
    np.random.seed(43)
    return pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 500),
        "feature_b": np.random.normal(5, 2, 500),
        "feature_c": np.random.uniform(0, 10, 500),
    })


@pytest.fixture
def sample_production_drifted():
    np.random.seed(44)
    return pd.DataFrame({
        "feature_a": np.random.normal(3, 1, 500),
        "feature_b": np.random.normal(5, 2, 500),
        "feature_c": np.random.uniform(0, 10, 500),
    })


class TestFeatureObserver:
    """Tests for FeatureObserver."""

    def test_fit_basic(self, sample_baseline):
        observer = FeatureObserver()
        observer.fit(sample_baseline)
        assert observer._is_fitted
        assert len(observer.feature_names_in_) == 3
        assert "feature_a" in observer.baseline_stats_

    def test_fit_with_columns(self, sample_baseline):
        observer = FeatureObserver(columns=["feature_a", "feature_b"])
        observer.fit(sample_baseline)
        assert observer.feature_names_in_ == ["feature_a", "feature_b"]

    def test_fit_rejects_non_dataframe(self):
        observer = FeatureObserver()
        with pytest.raises(ValueError, match="Expected pd.DataFrame"):
            observer.fit([[1, 2], [3, 4]])

    def test_check_without_fit_raises(self, sample_baseline):
        observer = FeatureObserver()
        with pytest.raises(RuntimeError, match="must be fitted"):
            observer.check(sample_baseline)

    def test_check_stable_data_no_alerts(self, sample_baseline, sample_production_stable):
        rules = [
            AlertRule("high_psi", MetricType.PSI, "gt", 0.2, AlertSeverity.CRITICAL),
        ]
        observer = FeatureObserver(alert_rules=rules)
        observer.fit(sample_baseline)
        alerts = observer.check(sample_production_stable)
        psi_alerts = [a for a in alerts if a.metric == MetricType.PSI]
        assert len(psi_alerts) == 0

    def test_check_drifted_data_triggers_alert(self, sample_baseline, sample_production_drifted):
        rules = [
            AlertRule("high_psi", MetricType.PSI, "gt", 0.2, AlertSeverity.CRITICAL),
        ]
        observer = FeatureObserver(alert_rules=rules)
        observer.fit(sample_baseline)
        alerts = observer.check(sample_production_drifted)
        assert len(alerts) > 0
        assert any(a.feature == "feature_a" for a in alerts)

    def test_check_null_rate_alert(self, sample_baseline):
        rules = [
            AlertRule("high_nulls", MetricType.NULL_RATE, "gt", 0.05, AlertSeverity.WARNING),
        ]
        observer = FeatureObserver(alert_rules=rules)
        observer.fit(sample_baseline)

        prod = sample_baseline.copy()
        prod.loc[prod.index[:100], "feature_a"] = np.nan
        alerts = observer.check(prod)
        null_alerts = [a for a in alerts if a.metric == MetricType.NULL_RATE]
        assert len(null_alerts) > 0

    def test_metrics_history_tracking(self, sample_baseline, sample_production_stable):
        observer = FeatureObserver()
        observer.fit(sample_baseline)
        observer.check(sample_production_stable)
        observer.check(sample_production_stable)
        assert len(observer.metrics_history_["feature_a"]) == 2

    def test_metrics_history_respects_size_limit(self, sample_baseline, sample_production_stable):
        observer = FeatureObserver(history_size=3)
        observer.fit(sample_baseline)
        for _ in range(5):
            observer.check(sample_production_stable)
        assert len(observer.metrics_history_["feature_a"]) == 3

    def test_notification_handler_called(self, sample_baseline, sample_production_drifted):
        received_alerts: list[Alert] = []
        rules = [
            AlertRule("drift", MetricType.PSI, "gt", 0.2, AlertSeverity.WARNING),
        ]
        observer = FeatureObserver(
            alert_rules=rules,
            notification_handlers=[received_alerts.append],
        )
        observer.fit(sample_baseline)
        observer.check(sample_production_drifted)
        assert len(received_alerts) > 0

    def test_notification_handler_failure_doesnt_crash(self, sample_baseline, sample_production_drifted):
        def bad_handler(alert: Alert) -> None:
            raise RuntimeError("Handler failed")

        rules = [
            AlertRule("drift", MetricType.PSI, "gt", 0.2, AlertSeverity.WARNING),
        ]
        observer = FeatureObserver(
            alert_rules=rules,
            notification_handlers=[bad_handler],
        )
        observer.fit(sample_baseline)
        alerts = observer.check(sample_production_drifted)
        assert len(alerts) > 0

    def test_get_summary(self, sample_baseline, sample_production_stable):
        observer = FeatureObserver()
        observer.fit(sample_baseline)
        observer.check(sample_production_stable)
        summary = observer.get_summary()
        assert summary["features_monitored"] == 3
        assert "latest_metrics" in summary
        assert "feature_a" in summary["latest_metrics"]

    def test_get_alerts_filtered(self, sample_baseline, sample_production_drifted):
        rules = [
            AlertRule("drift", MetricType.PSI, "gt", 0.2, AlertSeverity.WARNING),
            AlertRule("nulls", MetricType.NULL_RATE, "gt", 0.5, AlertSeverity.CRITICAL),
        ]
        observer = FeatureObserver(alert_rules=rules)
        observer.fit(sample_baseline)
        observer.check(sample_production_drifted)
        warning_alerts = observer.get_alerts(severity=AlertSeverity.WARNING)
        assert all(a.severity == AlertSeverity.WARNING for a in warning_alerts)

    def test_export_metrics_dict(self, sample_baseline, sample_production_stable):
        observer = FeatureObserver()
        observer.fit(sample_baseline)
        observer.check(sample_production_stable)
        data = observer.export_metrics(format="dict")
        assert "feature_a" in data
        assert len(data["feature_a"]) == 1

    def test_export_metrics_json(self, sample_baseline, sample_production_stable):
        observer = FeatureObserver()
        observer.fit(sample_baseline)
        observer.check(sample_production_stable)
        json_str = observer.export_metrics(format="json")
        assert isinstance(json_str, str)
        import json
        parsed = json.loads(json_str)
        assert "feature_a" in parsed


class TestSLATracking:
    """Tests for SLA compliance tracking."""

    def test_sla_compliant(self, sample_baseline, sample_production_stable):
        sla = SLADefinition(
            name="low_nulls",
            description="Null rate should be below 5%",
            metric=MetricType.NULL_RATE,
            target=0.05,
            condition="lt",
        )
        observer = FeatureObserver(sla_definitions=[sla])
        observer.fit(sample_baseline)
        observer.check(sample_production_stable)
        reports = observer.sla_reports_
        assert "low_nulls" in reports
        for feat, report in reports["low_nulls"].items():
            assert report.compliant is True

    def test_sla_non_compliant(self, sample_baseline):
        sla = SLADefinition(
            name="low_nulls",
            description="Null rate should be below 5%",
            metric=MetricType.NULL_RATE,
            target=0.05,
            condition="lt",
        )
        observer = FeatureObserver(sla_definitions=[sla])
        observer.fit(sample_baseline)

        prod = sample_baseline.copy()
        prod.loc[prod.index[:200], "feature_a"] = np.nan
        observer.check(prod)
        report = observer.sla_reports_["low_nulls"]["feature_a"]
        assert report.compliant is False
        assert report.current_value > 0.05

    def test_sla_compliance_rate_window(self, sample_baseline, sample_production_stable):
        sla = SLADefinition(
            name="low_nulls",
            description="Null rate < 5%",
            metric=MetricType.NULL_RATE,
            target=0.05,
            condition="lt",
            window_size=3,
        )
        observer = FeatureObserver(sla_definitions=[sla])
        observer.fit(sample_baseline)

        for _ in range(3):
            observer.check(sample_production_stable)

        report = observer.sla_reports_["low_nulls"]["feature_a"]
        assert report.compliance_rate == 1.0
        assert len(report.history) == 3


class TestAlertRule:
    """Tests for AlertRule evaluation."""

    def test_condition_gt(self):
        assert FeatureObserver._check_condition(0.5, "gt", 0.3) is True
        assert FeatureObserver._check_condition(0.2, "gt", 0.3) is False

    def test_condition_lt(self):
        assert FeatureObserver._check_condition(0.2, "lt", 0.3) is True
        assert FeatureObserver._check_condition(0.5, "lt", 0.3) is False

    def test_condition_gte(self):
        assert FeatureObserver._check_condition(0.3, "gte", 0.3) is True
        assert FeatureObserver._check_condition(0.2, "gte", 0.3) is False

    def test_condition_lte(self):
        assert FeatureObserver._check_condition(0.3, "lte", 0.3) is True
        assert FeatureObserver._check_condition(0.5, "lte", 0.3) is False

    def test_condition_nan_returns_false(self):
        assert FeatureObserver._check_condition(float("nan"), "gt", 0.3) is False

    def test_rule_column_filtering(self, sample_baseline, sample_production_drifted):
        rules = [
            AlertRule(
                "drift_b_only", MetricType.PSI, "gt", 0.2,
                AlertSeverity.WARNING, columns=["feature_b"],
            ),
        ]
        observer = FeatureObserver(alert_rules=rules)
        observer.fit(sample_baseline)
        alerts = observer.check(sample_production_drifted)
        for alert in alerts:
            assert alert.feature == "feature_b"


class TestAlertDataclass:
    """Tests for Alert dataclass."""

    def test_auto_timestamp(self):
        alert = Alert(
            rule_name="test",
            feature="col",
            metric=MetricType.PSI,
            value=0.5,
            threshold=0.2,
            severity=AlertSeverity.WARNING,
        )
        assert alert.timestamp > 0

    def test_auto_message(self):
        alert = Alert(
            rule_name="drift_check",
            feature="col_a",
            metric=MetricType.PSI,
            value=0.5,
            threshold=0.2,
            severity=AlertSeverity.CRITICAL,
        )
        assert "drift_check" in alert.message
        assert "col_a" in alert.message
        assert "CRITICAL" in alert.message
