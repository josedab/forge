"""Tests for retraining triggers and importance tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.monitoring.retraining import (
    ImportanceTracker,
    RetrainingDecision,
    RetrainingTrigger,
    WebhookNotifier,
)


@pytest.fixture
def baseline_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 500),
        "y": rng.normal(5, 2, 500),
    })


@pytest.fixture
def drifted_df() -> pd.DataFrame:
    rng = np.random.RandomState(99)
    return pd.DataFrame({
        "x": rng.normal(5, 1, 500),  # shifted mean
        "y": rng.normal(5, 2, 500),
    })


class TestImportanceTracker:
    def test_no_shift(self) -> None:
        tracker = ImportanceTracker(shift_threshold=0.3)
        tracker.set_baseline({"a": 0.5, "b": 0.3})
        shifts = tracker.record({"a": 0.5, "b": 0.3})
        assert len(shifts) == 0

    def test_detects_shift(self) -> None:
        tracker = ImportanceTracker(shift_threshold=0.3)
        tracker.set_baseline({"a": 0.5, "b": 0.3})
        shifts = tracker.record({"a": 0.1, "b": 0.3})
        assert "a" in shifts

    def test_history(self) -> None:
        tracker = ImportanceTracker()
        tracker.set_baseline({"a": 0.5})
        tracker.record({"a": 0.5})
        tracker.record({"a": 0.4})
        assert len(tracker.history) == 2


class TestRetrainingTrigger:
    def test_no_drift(self, baseline_df: pd.DataFrame) -> None:
        trigger = RetrainingTrigger(psi_threshold=0.2)
        trigger.fit(baseline_df)
        decision = trigger.check(baseline_df)
        assert not decision.should_retrain

    def test_drift_detected(self, baseline_df: pd.DataFrame, drifted_df: pd.DataFrame) -> None:
        trigger = RetrainingTrigger(psi_threshold=0.1, min_drifted_fraction=0.3)
        trigger.fit(baseline_df)
        decision = trigger.check(drifted_df)
        assert decision.should_retrain
        assert len(decision.reasons) > 0

    def test_importance_shift_triggers(self, baseline_df: pd.DataFrame) -> None:
        trigger = RetrainingTrigger(importance_shift_threshold=0.2)
        trigger.fit(baseline_df, importance={"x": 0.6, "y": 0.4})
        decision = trigger.check(baseline_df, importance={"x": 0.1, "y": 0.4})
        assert decision.should_retrain

    def test_callback_fired(self, baseline_df: pd.DataFrame, drifted_df: pd.DataFrame) -> None:
        fired: list[RetrainingDecision] = []
        trigger = RetrainingTrigger(
            psi_threshold=0.1,
            min_drifted_fraction=0.3,
            callbacks=[fired.append],
        )
        trigger.fit(baseline_df)
        trigger.check(drifted_df)
        assert len(fired) >= 1

    def test_not_fitted_raises(self) -> None:
        trigger = RetrainingTrigger()
        with pytest.raises(RuntimeError):
            trigger.check(pd.DataFrame({"x": [1]}))

    def test_decision_history(self, baseline_df: pd.DataFrame) -> None:
        trigger = RetrainingTrigger()
        trigger.fit(baseline_df)
        trigger.check(baseline_df)
        trigger.check(baseline_df)
        assert len(trigger.decision_history) == 2


class TestWebhookNotifier:
    def test_notify_fails_gracefully(self) -> None:
        notifier = WebhookNotifier(url="http://localhost:99999/fake")
        ok = notifier.notify({"msg": "test"})
        assert not ok  # connection error
        assert len(notifier.sent_payloads) == 1

    def test_to_dict(self) -> None:
        d = RetrainingDecision(should_retrain=True, reasons=["drift"])
        data = d.to_dict()
        assert data["should_retrain"] is True
