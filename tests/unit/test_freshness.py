"""Tests for freshness policies and backfill engine."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from forge.freshness import (
    BackfillEngine,
    FreshnessPolicy,
    FreshnessStatus,
)


def dummy_compute(start: datetime, end: datetime) -> pd.DataFrame:
    """Dummy compute function for testing."""
    return pd.DataFrame({"value": [1, 2, 3]})


def failing_compute(start: datetime, end: datetime) -> pd.DataFrame:
    raise RuntimeError("compute failed")


class TestFreshnessPolicy:
    def test_is_stale_none(self):
        policy = FreshnessPolicy(name="test", refresh_hours=24)
        assert policy.is_stale(None)

    def test_is_stale_recent(self):
        policy = FreshnessPolicy(name="test", refresh_hours=24)
        assert not policy.is_stale(datetime.now())

    def test_is_stale_old(self):
        policy = FreshnessPolicy(name="test", refresh_hours=1)
        old = datetime.now() - timedelta(hours=2)
        assert policy.is_stale(old)

    def test_is_critical(self):
        policy = FreshnessPolicy(name="test", max_staleness_hours=48)
        old = datetime.now() - timedelta(hours=50)
        assert policy.is_critical(old)

    def test_not_critical(self):
        policy = FreshnessPolicy(name="test", max_staleness_hours=48)
        recent = datetime.now() - timedelta(hours=10)
        assert not policy.is_critical(recent)

    def test_to_dict(self):
        policy = FreshnessPolicy(name="test", refresh_hours=12)
        d = policy.to_dict()
        assert d["name"] == "test"
        assert d["refresh_hours"] == 12


class TestBackfillEngine:
    def test_register_and_list(self):
        engine = BackfillEngine()
        policy = FreshnessPolicy(name="p1", refresh_hours=24)
        engine.register("f1", policy, dummy_compute)
        assert "f1" in engine.registered_features

    def test_unregister(self):
        engine = BackfillEngine()
        engine.register("f1", FreshnessPolicy(name="p1"), dummy_compute)
        assert engine.unregister("f1")
        assert "f1" not in engine.registered_features
        assert not engine.unregister("nonexistent")

    def test_check_freshness_new_feature_is_stale(self):
        engine = BackfillEngine()
        engine.register("f1", FreshnessPolicy(name="p1", refresh_hours=24), dummy_compute)
        statuses = engine.check_freshness()
        assert len(statuses) == 1
        assert statuses[0].is_stale
        assert statuses[0].is_critical

    def test_mark_updated(self):
        engine = BackfillEngine()
        engine.register("f1", FreshnessPolicy(name="p1", refresh_hours=24), dummy_compute)
        engine.mark_updated("f1")
        statuses = engine.check_freshness()
        assert not statuses[0].is_stale

    def test_refresh_stale(self):
        engine = BackfillEngine()
        engine.register("f1", FreshnessPolicy(name="p1", refresh_hours=24), dummy_compute)
        refreshed = engine.refresh_stale()
        assert "f1" in refreshed
        status = engine.get_status("f1")
        assert not status.is_stale

    def test_refresh_skips_non_stale(self):
        engine = BackfillEngine()
        engine.register("f1", FreshnessPolicy(name="p1", refresh_hours=24), dummy_compute)
        engine.mark_updated("f1")
        refreshed = engine.refresh_stale()
        assert len(refreshed) == 0

    def test_refresh_handles_failure(self):
        engine = BackfillEngine()
        engine.register("f1", FreshnessPolicy(name="p1", refresh_hours=24), failing_compute)
        refreshed = engine.refresh_stale()
        assert "f1" not in refreshed
        status = engine.get_status("f1")
        assert status.consecutive_failures == 1
        assert "compute failed" in status.last_error

    def test_max_retries(self):
        engine = BackfillEngine()
        engine.register(
            "f1",
            FreshnessPolicy(name="p1", refresh_hours=24, max_retries=2),
            failing_compute,
        )
        engine.refresh_stale()  # fail 1
        engine.refresh_stale()  # fail 2
        engine.refresh_stale()  # should be skipped (max retries)
        status = engine.get_status("f1")
        assert status.consecutive_failures == 2

    def test_backfill_success(self):
        engine = BackfillEngine()
        engine.register("f1", FreshnessPolicy(name="p1"), dummy_compute)
        job = engine.backfill("f1", "2025-01-01", "2025-01-05")
        assert job.status == "completed"
        assert job.rows_processed > 0

    def test_backfill_failure(self):
        engine = BackfillEngine()
        engine.register("f1", FreshnessPolicy(name="p1"), failing_compute)
        job = engine.backfill("f1", "2025-01-01", "2025-01-02")
        assert job.status == "failed"
        assert "compute failed" in job.error

    def test_backfill_missing_feature(self):
        engine = BackfillEngine()
        with pytest.raises(KeyError):
            engine.backfill("nonexistent", "2025-01-01", "2025-01-02")

    def test_sla_violation_callback(self):
        violations: list[str] = []
        engine = BackfillEngine(
            on_sla_violation=lambda name, status: violations.append(name)
        )
        engine.register("f1", FreshnessPolicy(name="p1", max_staleness_hours=0.001), dummy_compute)
        engine.check_freshness()
        assert "f1" in violations

    def test_get_status_missing(self):
        engine = BackfillEngine()
        assert engine.get_status("nonexistent") is None

    def test_backfill_jobs_list(self):
        engine = BackfillEngine()
        engine.register("f1", FreshnessPolicy(name="p1"), dummy_compute)
        engine.backfill("f1", "2025-01-01", "2025-01-02")
        assert len(engine.backfill_jobs) == 1

    def test_freshness_status_to_dict(self):
        s = FreshnessStatus(feature_name="f1", is_stale=True)
        d = s.to_dict()
        assert d["feature_name"] == "f1"

    def test_priority_ordering(self):
        """Higher-priority features refresh first."""
        order: list[str] = []

        def track_compute(name: str):
            def fn(start, end):
                order.append(name)
                return pd.DataFrame({"v": [1]})
            return fn

        engine = BackfillEngine()
        engine.register("low", FreshnessPolicy(name="low", priority=3), track_compute("low"))
        engine.register("high", FreshnessPolicy(name="high", priority=1), track_compute("high"))
        engine.register("mid", FreshnessPolicy(name="mid", priority=2), track_compute("mid"))

        engine.refresh_stale()
        assert order == ["high", "mid", "low"]
