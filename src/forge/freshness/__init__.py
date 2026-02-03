"""Feature freshness policies and backfill engine.

Provides declarative freshness policies, a backfill scheduler for
re-materializing features over historical windows, and SLA monitoring
integration with the MetricsStore.

Example:
    >>> from forge.freshness import FreshnessPolicy, BackfillEngine
    >>> policy = FreshnessPolicy(name="daily_features", refresh_hours=24)
    >>> engine = BackfillEngine()
    >>> engine.register("user_age", policy, compute_fn=compute_user_age)
    >>> engine.check_freshness()
    >>> engine.backfill("user_age", start="2025-01-01", end="2025-01-31")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FreshnessPolicy:
    """Declarative freshness policy for a feature group."""

    name: str
    refresh_hours: float = 24.0
    max_staleness_hours: float = 48.0
    priority: int = 1  # 1=highest
    retry_on_failure: bool = True
    max_retries: int = 3
    backfill_window_days: int = 30

    def is_stale(self, last_updated: datetime | None) -> bool:
        """Check if feature is stale given last update time."""
        if last_updated is None:
            return True
        elapsed = (datetime.now() - last_updated).total_seconds() / 3600
        return elapsed > self.refresh_hours

    def is_critical(self, last_updated: datetime | None) -> bool:
        """Check if staleness exceeds maximum threshold."""
        if last_updated is None:
            return True
        elapsed = (datetime.now() - last_updated).total_seconds() / 3600
        return elapsed > self.max_staleness_hours

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "refresh_hours": self.refresh_hours,
            "max_staleness_hours": self.max_staleness_hours,
            "priority": self.priority,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
            "backfill_window_days": self.backfill_window_days,
        }


@dataclass
class FreshnessStatus:
    """Current freshness status of a registered feature."""

    feature_name: str
    last_updated: datetime | None = None
    is_stale: bool = True
    is_critical: bool = True
    staleness_hours: float = float("inf")
    next_refresh_at: datetime | None = None
    consecutive_failures: int = 0
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "is_stale": self.is_stale,
            "is_critical": self.is_critical,
            "staleness_hours": round(self.staleness_hours, 2),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
        }


@dataclass
class BackfillJob:
    """A backfill job for a feature over a time range."""

    feature_name: str
    start_date: datetime
    end_date: datetime
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    rows_processed: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "status": self.status,
            "rows_processed": self.rows_processed,
            "error": self.error,
        }


ComputeFunction = Callable[[datetime, datetime], pd.DataFrame]


@dataclass
class _FeatureRegistration:
    """Internal registration of a feature with its policy and compute function."""

    feature_name: str
    policy: FreshnessPolicy
    compute_fn: ComputeFunction
    last_updated: datetime | None = None
    consecutive_failures: int = 0
    last_error: str = ""


class BackfillEngine:
    """Manages feature freshness and backfill operations.

    Registers features with freshness policies and compute functions,
    checks staleness, and orchestrates backfill over time windows.

    Parameters:
        on_sla_violation: Optional callback when SLA is violated.
    """

    def __init__(
        self,
        on_sla_violation: Callable[[str, FreshnessStatus], None] | None = None,
    ) -> None:
        self._registrations: dict[str, _FeatureRegistration] = {}
        self._backfill_jobs: list[BackfillJob] = []
        self.on_sla_violation = on_sla_violation

    def register(
        self,
        feature_name: str,
        policy: FreshnessPolicy,
        compute_fn: ComputeFunction,
    ) -> None:
        """Register a feature with a freshness policy.

        Args:
            feature_name: Feature identifier.
            policy: Freshness policy.
            compute_fn: Function(start_date, end_date) -> DataFrame.
        """
        self._registrations[feature_name] = _FeatureRegistration(
            feature_name=feature_name,
            policy=policy,
            compute_fn=compute_fn,
        )

    def unregister(self, feature_name: str) -> bool:
        """Remove a feature registration."""
        if feature_name in self._registrations:
            del self._registrations[feature_name]
            return True
        return False

    def mark_updated(self, feature_name: str) -> None:
        """Mark a feature as freshly updated."""
        if feature_name in self._registrations:
            self._registrations[feature_name].last_updated = datetime.now()
            self._registrations[feature_name].consecutive_failures = 0
            self._registrations[feature_name].last_error = ""

    def check_freshness(self) -> list[FreshnessStatus]:
        """Check freshness of all registered features.

        Returns:
            List of FreshnessStatus for each feature.
        """
        statuses: list[FreshnessStatus] = []

        for name, reg in self._registrations.items():
            stale = reg.policy.is_stale(reg.last_updated)
            critical = reg.policy.is_critical(reg.last_updated)

            if reg.last_updated:
                staleness = (datetime.now() - reg.last_updated).total_seconds() / 3600
                next_refresh = reg.last_updated + timedelta(hours=reg.policy.refresh_hours)
            else:
                staleness = float("inf")
                next_refresh = None

            status = FreshnessStatus(
                feature_name=name,
                last_updated=reg.last_updated,
                is_stale=stale,
                is_critical=critical,
                staleness_hours=staleness,
                next_refresh_at=next_refresh,
                consecutive_failures=reg.consecutive_failures,
                last_error=reg.last_error,
            )
            statuses.append(status)

            if critical and self.on_sla_violation:
                self.on_sla_violation(name, status)

        return statuses

    def refresh_stale(self) -> list[str]:
        """Refresh all stale features.

        Returns:
            List of feature names that were refreshed.
        """
        refreshed: list[str] = []
        statuses = self.check_freshness()

        # Sort by priority (lower number = higher priority)
        stale = [s for s in statuses if s.is_stale]
        stale.sort(key=lambda s: self._registrations[s.feature_name].policy.priority)

        for status in stale:
            name = status.feature_name
            reg = self._registrations[name]

            if reg.consecutive_failures >= reg.policy.max_retries:
                logger.warning("Skipping %s: max retries exceeded", name)
                continue

            now = datetime.now()
            start = now - timedelta(hours=reg.policy.refresh_hours)

            try:
                reg.compute_fn(start, now)
                reg.last_updated = now
                reg.consecutive_failures = 0
                reg.last_error = ""
                refreshed.append(name)
            except Exception as e:
                reg.consecutive_failures += 1
                reg.last_error = str(e)
                logger.error("Failed to refresh %s: %s", name, e)

        return refreshed

    def backfill(
        self,
        feature_name: str,
        start_date: str | datetime,
        end_date: str | datetime,
        chunk_days: int = 1,
    ) -> BackfillJob:
        """Backfill a feature over a historical time range.

        Args:
            feature_name: Feature to backfill.
            start_date: Start of backfill range.
            end_date: End of backfill range.
            chunk_days: Process in chunks of N days.

        Returns:
            BackfillJob with results.

        Raises:
            KeyError: If feature not registered.
        """
        if feature_name not in self._registrations:
            raise KeyError(f"Feature '{feature_name}' not registered")

        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        reg = self._registrations[feature_name]
        job = BackfillJob(
            feature_name=feature_name,
            start_date=start_date,
            end_date=end_date,
            status="running",
        )
        self._backfill_jobs.append(job)

        total_rows = 0
        current = start_date

        try:
            while current < end_date:
                chunk_end = min(current + timedelta(days=chunk_days), end_date)
                result = reg.compute_fn(current, chunk_end)
                total_rows += len(result)
                current = chunk_end

            job.status = "completed"
            job.rows_processed = total_rows
            job.completed_at = datetime.now()
            reg.last_updated = datetime.now()
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error("Backfill failed for %s: %s", feature_name, e)

        return job

    def get_status(self, feature_name: str) -> FreshnessStatus | None:
        """Get freshness status for a single feature."""
        reg = self._registrations.get(feature_name)
        if reg is None:
            return None

        stale = reg.policy.is_stale(reg.last_updated)
        critical = reg.policy.is_critical(reg.last_updated)
        staleness = (
            (datetime.now() - reg.last_updated).total_seconds() / 3600
            if reg.last_updated
            else float("inf")
        )

        return FreshnessStatus(
            feature_name=feature_name,
            last_updated=reg.last_updated,
            is_stale=stale,
            is_critical=critical,
            staleness_hours=staleness,
            consecutive_failures=reg.consecutive_failures,
            last_error=reg.last_error,
        )

    @property
    def registered_features(self) -> list[str]:
        """All registered feature names."""
        return sorted(self._registrations.keys())

    @property
    def backfill_jobs(self) -> list[BackfillJob]:
        """All backfill jobs."""
        return list(self._backfill_jobs)
