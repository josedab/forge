"""Pareto-optimal feature set selection and early stopping utilities.

Provides tools for selecting feature sets that balance performance
versus feature count (complexity), and for terminating search early
when convergence is detected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from forge.search.search import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier.

    Attributes:
        n_features: Number of features in this set.
        score: Cross-validated fitness score.
        features: The feature results in this set.
    """

    n_features: int
    score: float
    features: list[SearchResult] = field(default_factory=list)


def pareto_optimal_sets(
    results: list[SearchResult],
    max_sets: int = 10,
    min_features: int = 1,
    max_features: int | None = None,
) -> list[ParetoPoint]:
    """Find Pareto-optimal feature sets balancing performance vs count.

    Constructs feature sets of increasing size by greedily adding the
    next highest-scoring feature, then identifies sets on the Pareto
    frontier (no other set is both smaller AND higher-scoring).

    Args:
        results: Sorted search results (best first).
        max_sets: Maximum number of Pareto points to return.
        min_features: Minimum feature set size to consider.
        max_features: Maximum feature set size. None for all.

    Returns:
        List of Pareto-optimal points, sorted by n_features.
    """
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    if max_features is None:
        max_features = len(sorted_results)
    max_features = min(max_features, len(sorted_results))

    # Build cumulative feature sets and track scores
    points: list[ParetoPoint] = []
    cumulative_score = 0.0
    for i in range(max_features):
        cumulative_score = sorted_results[i].score
        n = i + 1
        if n < min_features:
            continue
        points.append(ParetoPoint(
            n_features=n,
            score=cumulative_score,
            features=list(sorted_results[:n]),
        ))

    # Find Pareto frontier: no point is dominated
    frontier: list[ParetoPoint] = []
    for p in points:
        dominated = False
        for q in points:
            if q.n_features < p.n_features and q.score >= p.score:
                dominated = True
                break
        if not dominated:
            frontier.append(p)

    frontier.sort(key=lambda p: p.n_features)
    return frontier[:max_sets]


def select_knee_point(frontier: list[ParetoPoint]) -> ParetoPoint | None:
    """Select the knee point on the Pareto frontier.

    The knee point represents the best trade-off between performance
    and complexity, identified by maximum curvature.

    Args:
        frontier: Pareto-optimal points sorted by n_features.

    Returns:
        The knee point, or None if frontier is empty.
    """
    if not frontier:
        return None
    if len(frontier) <= 2:
        return frontier[0]

    # Normalize axes to [0, 1]
    n_vals = np.array([p.n_features for p in frontier], dtype=float)
    s_vals = np.array([p.score for p in frontier], dtype=float)

    n_range = n_vals.max() - n_vals.min()
    s_range = s_vals.max() - s_vals.min()

    if n_range == 0 or s_range == 0:
        return frontier[0]

    n_norm = (n_vals - n_vals.min()) / n_range
    s_norm = (s_vals - s_vals.min()) / s_range

    # Line from first to last point
    v_line = np.array([n_norm[-1] - n_norm[0], s_norm[-1] - s_norm[0]])
    v_line_len = np.linalg.norm(v_line)
    if v_line_len == 0:
        return frontier[0]

    # Find point with maximum distance to this line
    max_dist = -1.0
    knee_idx = 0
    for i in range(len(frontier)):
        v_point = np.array([n_norm[i] - n_norm[0], s_norm[i] - s_norm[0]])
        cross = abs(v_line[0] * v_point[1] - v_line[1] * v_point[0])
        dist = cross / v_line_len
        if dist > max_dist:
            max_dist = dist
            knee_idx = i

    return frontier[knee_idx]


class EarlyStoppingMonitor:
    """Monitor search progress and detect convergence.

    Triggers early stopping when the best score has not improved
    by at least `min_delta` for `patience` consecutive generations.

    Args:
        patience: Number of generations without improvement before stopping.
        min_delta: Minimum improvement to count as progress.

    Example:
        >>> monitor = EarlyStoppingMonitor(patience=5, min_delta=0.001)
        >>> for gen in range(100):
        ...     score = evaluate_generation(gen)
        ...     if monitor.should_stop(score):
        ...         break
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_score: float = float("-inf")
        self._wait: int = 0
        self._stopped_generation: int | None = None
        self._history: list[float] = []

    def should_stop(self, score: float) -> bool:
        """Check whether search should stop.

        Args:
            score: Current generation's best score.

        Returns:
            True if search should terminate.
        """
        self._history.append(score)

        if score > self._best_score + self.min_delta:
            self._best_score = score
            self._wait = 0
            return False

        self._wait += 1
        if self._wait >= self.patience:
            self._stopped_generation = len(self._history) - 1
            logger.info(
                "Early stopping triggered after %d generations without improvement.",
                self.patience,
            )
            return True
        return False

    @property
    def best_score(self) -> float:
        """Best score observed so far."""
        return self._best_score

    @property
    def stopped_generation(self) -> int | None:
        """Generation at which early stopping was triggered, or None."""
        return self._stopped_generation

    @property
    def history(self) -> list[float]:
        """Score history for all generations."""
        return list(self._history)
