"""Tests for the metrics store and governance reporting."""

from __future__ import annotations

from forge.monitoring.metrics_store import (
    MetricsStore,
)


class TestMetricsStore:
    def test_record_and_query(self):
        store = MetricsStore()
        store.record("user_age", {"mean": 34.5, "std": 12.0})

        results = store.query("user_age")
        assert len(results) == 2

    def test_query_by_metric(self):
        store = MetricsStore()
        store.record("user_age", {"mean": 34.5, "std": 12.0})

        results = store.query("user_age", metric_name="mean")
        assert len(results) == 1
        assert results[0].value == 34.5

    def test_query_last_n(self):
        store = MetricsStore()
        for i in range(10):
            store.record("f1", {"mean": float(i)})

        results = store.query("f1", metric_name="mean", last_n=3)
        assert len(results) == 3

    def test_get_latest(self):
        store = MetricsStore()
        store.record("f1", {"mean": 1.0, "std": 0.5})
        store.record("f1", {"mean": 2.0, "std": 0.6})

        latest = store.get_latest("f1")
        assert latest["mean"] == 2.0
        assert latest["std"] == 0.6

    def test_feature_health_healthy(self):
        store = MetricsStore()
        store.record("f1", {"psi": 0.05, "null_rate": 0.01})

        health = store.feature_health("f1")
        assert health.status == "healthy"
        assert len(health.issues) == 0

    def test_feature_health_warning(self):
        store = MetricsStore()
        store.record("f1", {"psi": 0.25, "null_rate": 0.01})

        health = store.feature_health("f1")
        assert health.status == "warning"
        assert len(health.issues) >= 1

    def test_feature_health_critical(self):
        store = MetricsStore()
        store.record("f1", {"psi": 0.5, "null_rate": 0.3})

        health = store.feature_health("f1")
        assert health.status == "critical"

    def test_governance_report(self):
        store = MetricsStore()
        store.record("f1", {"psi": 0.05, "null_rate": 0.01})
        store.record("f2", {"psi": 0.3, "null_rate": 0.15})
        store.record("f3", {"psi": 0.6, "null_rate": 0.5})

        report = store.generate_governance_report(["f1", "f2", "f3"])
        assert report.total_features == 3
        assert report.healthy_count >= 1
        assert report.compliance_rate > 0

    def test_governance_report_to_dict(self):
        store = MetricsStore()
        store.record("f1", {"psi": 0.05})
        report = store.generate_governance_report(["f1"])
        d = report.to_dict()
        assert "total_features" in d
        assert "compliance_rate" in d

    def test_governance_report_to_markdown(self):
        store = MetricsStore()
        store.record("f1", {"psi": 0.3})
        report = store.generate_governance_report(["f1"])
        md = report.to_markdown()
        assert "# Feature Governance Report" in md

    def test_export_prometheus(self):
        store = MetricsStore()
        store.record("f1", {"mean": 34.5, "null_rate": 0.02})

        text = store.export_prometheus()
        assert "forge_feature_mean" in text
        assert 'feature="f1"' in text
        assert "34.5" in text

    def test_prometheus_with_labels(self):
        store = MetricsStore()
        store.record("f1", {"mean": 10.0}, labels={"env": "prod"})

        text = store.export_prometheus()
        assert 'env="prod"' in text

    def test_persistence(self, tmp_path):
        store1 = MetricsStore(storage_path=tmp_path / "metrics")
        store1.record("f1", {"mean": 42.0})
        store1.save()

        store2 = MetricsStore(storage_path=tmp_path / "metrics")
        results = store2.query("f1")
        assert len(results) == 1
        assert results[0].value == 42.0

    def test_max_history(self):
        store = MetricsStore(max_history=5)
        for i in range(20):
            store.record("f1", {"val": float(i)})

        results = store.query("f1")
        assert len(results) <= 5

    def test_feature_names(self):
        store = MetricsStore()
        store.record("f1", {"mean": 1.0})
        store.record("f2", {"mean": 2.0})

        names = store.feature_names
        assert "f1" in names
        assert "f2" in names

    def test_size(self):
        store = MetricsStore()
        store.record("f1", {"mean": 1.0, "std": 0.5})
        assert store.size == 2
