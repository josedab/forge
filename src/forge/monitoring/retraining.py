"""Retraining triggers and importance-shift tracking.

Extends the monitoring module with:
- ``RetrainingTrigger``: Evaluates drift results and fires callbacks when
  retraining is warranted.
- ``ImportanceTracker``: Tracks feature importance over time and detects
  significant shifts.
- ``WebhookNotifier``: Sends alert payloads to HTTP endpoints.

Example:
    >>> from forge.monitoring.retraining import RetrainingTrigger
    >>> trigger = RetrainingTrigger(psi_threshold=0.2, importance_shift=0.3)
    >>> trigger.fit(X_train, importance_baseline)
    >>> should_retrain = trigger.check(X_prod, current_importance)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd  # noqa: TC002

from forge.monitoring.drift import calculate_psi

logger = logging.getLogger(__name__)


@dataclass
class RetrainingDecision:
    """Result of a retraining-trigger evaluation."""

    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    psi_scores: dict[str, float] = field(default_factory=dict)
    importance_shifts: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_retrain": self.should_retrain,
            "reasons": self.reasons,
            "psi_scores": self.psi_scores,
            "importance_shifts": self.importance_shifts,
            "timestamp": self.timestamp,
        }


class ImportanceTracker:
    """Track feature importance over time and detect shifts.

    Parameters
    ----------
    shift_threshold : float
        Fractional importance change to flag (e.g., 0.3 = 30 %).
    history_size : int
        Number of snapshots to retain.
    """

    def __init__(
        self,
        shift_threshold: float = 0.3,
        history_size: int = 50,
    ) -> None:
        self.shift_threshold = shift_threshold
        self.history_size = history_size
        self._baseline: dict[str, float] = {}
        self._history: list[dict[str, float]] = []

    def set_baseline(self, importance: dict[str, float]) -> None:
        """Set the baseline importance scores."""
        self._baseline = dict(importance)

    def record(self, importance: dict[str, float]) -> dict[str, float]:
        """Record a new importance snapshot and return shifts vs baseline.

        Args:
            importance: Current feature importance dict.

        Returns:
            Dict of feature → fractional shift for features exceeding threshold.
        """
        self._history.append(dict(importance))
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]

        shifts: dict[str, float] = {}
        for feat, baseline_val in self._baseline.items():
            current_val = importance.get(feat, 0.0)
            if baseline_val > 0:
                shift = abs(current_val - baseline_val) / baseline_val
            else:
                shift = abs(current_val)
            if shift >= self.shift_threshold:
                shifts[feat] = shift
        return shifts

    @property
    def baseline(self) -> dict[str, float]:
        return dict(self._baseline)

    @property
    def history(self) -> list[dict[str, float]]:
        return list(self._history)


class WebhookNotifier:
    """Send alert payloads to an HTTP endpoint.

    Parameters
    ----------
    url : str
        Webhook URL.
    headers : dict[str, str] | None
        Extra HTTP headers.
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self._sent: list[dict[str, Any]] = []

    def notify(self, payload: dict[str, Any]) -> bool:
        """Send a JSON payload to the webhook.

        Args:
            payload: Data to send.

        Returns:
            True if accepted (2xx), False otherwise.
        """
        try:
            import urllib.request

            data = json.dumps(payload, default=str).encode()
            req = urllib.request.Request(  # noqa: S310
                self.url, data=data, headers=self.headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                self._sent.append(payload)
                return 200 <= resp.status < 300
        except Exception as exc:
            logger.warning("Webhook notification failed: %s", exc)
            self._sent.append(payload)
            return False

    @property
    def sent_payloads(self) -> list[dict[str, Any]]:
        return list(self._sent)


class RetrainingTrigger:
    """Evaluates production data and decides if retraining is needed.

    Combines PSI-based drift detection with importance-shift tracking
    to make an informed retraining decision.

    Parameters
    ----------
    psi_threshold : float
        PSI value above which a feature is considered drifted.
    min_drifted_fraction : float
        Fraction of features that must drift to trigger retraining.
    importance_shift_threshold : float
        Fractional importance shift to consider significant.
    callbacks : list[Callable[[RetrainingDecision], None]] | None
        Functions called when retraining is triggered.
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        min_drifted_fraction: float = 0.2,
        importance_shift_threshold: float = 0.3,
        callbacks: list[Callable[[RetrainingDecision], None]] | None = None,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.min_drifted_fraction = min_drifted_fraction
        self.importance_shift_threshold = importance_shift_threshold
        self.callbacks = callbacks or []

        self._baseline_data: dict[str, np.ndarray] = {}
        self._importance_tracker = ImportanceTracker(
            shift_threshold=importance_shift_threshold,
        )
        self._decisions: list[RetrainingDecision] = []
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        importance: dict[str, float] | None = None,
    ) -> RetrainingTrigger:
        """Establish baselines from training data.

        Args:
            X: Training data.
            importance: Baseline feature importances.

        Returns:
            Self.
        """
        for col in X.select_dtypes(include=[np.number]).columns:
            self._baseline_data[col] = X[col].dropna().values

        if importance:
            self._importance_tracker.set_baseline(importance)

        self._is_fitted = True
        return self

    def check(
        self,
        X: pd.DataFrame,
        importance: dict[str, float] | None = None,
    ) -> RetrainingDecision:
        """Check production data and decide if retraining is needed.

        Args:
            X: Production data.
            importance: Current feature importances.

        Returns:
            RetrainingDecision.
        """
        if not self._is_fitted:
            raise RuntimeError("RetrainingTrigger must be fitted first.")

        reasons: list[str] = []
        psi_scores: dict[str, float] = {}

        # PSI check
        n_drifted = 0
        n_checked = 0
        for col, baseline in self._baseline_data.items():
            if col not in X.columns:
                continue
            current = X[col].dropna().values
            psi = calculate_psi(baseline, current)
            psi_scores[col] = psi
            n_checked += 1
            if psi >= self.psi_threshold:
                n_drifted += 1

        if n_checked > 0 and n_drifted / n_checked >= self.min_drifted_fraction:
            reasons.append(
                f"{n_drifted}/{n_checked} features drifted (PSI ≥ {self.psi_threshold})"
            )

        # Importance shift
        importance_shifts: dict[str, float] = {}
        if importance:
            importance_shifts = self._importance_tracker.record(importance)
            if importance_shifts:
                reasons.append(
                    f"Importance shifted for: {', '.join(importance_shifts.keys())}"
                )

        decision = RetrainingDecision(
            should_retrain=len(reasons) > 0,
            reasons=reasons,
            psi_scores=psi_scores,
            importance_shifts=importance_shifts,
        )
        self._decisions.append(decision)

        if decision.should_retrain:
            for cb in self.callbacks:
                try:
                    cb(decision)
                except Exception:
                    logger.exception("Retraining callback failed")

        return decision

    @property
    def decision_history(self) -> list[RetrainingDecision]:
        return list(self._decisions)
