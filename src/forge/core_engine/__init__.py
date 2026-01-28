"""High-performance core engine for hot-path feature operations.

Provides NumPy-vectorized implementations of common feature engineering
operations, with an interface designed for future Rust/PyO3 bindings.

All functions accept and return NumPy arrays for minimal overhead.
The Engine class wraps these functions with a dispatch layer that
selects the fastest available backend.

Example:
    >>> from forge.core_engine import Engine
    >>> engine = Engine()
    >>> result = engine.interaction_features(X, [(0, 1, "multiply")])
    >>> result = engine.quantile_bin(X[:, 0], n_bins=10)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Vectorized primitives ────────────────────────────────────────────


def fast_one_hot(values: np.ndarray, categories: np.ndarray | None = None) -> np.ndarray:
    """Vectorized one-hot encoding.

    Args:
        values: 1-D integer or object array.
        categories: Unique category values. If None, inferred from data.

    Returns:
        2-D binary array of shape (len(values), len(categories)).
    """
    if categories is None:
        categories = np.unique(values)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    out = np.zeros((len(values), len(categories)), dtype=np.float64)
    for i, v in enumerate(values):
        idx = cat_to_idx.get(v, -1)
        if idx >= 0:
            out[i, idx] = 1.0
    return out


def fast_target_encode(
    values: np.ndarray,
    targets: np.ndarray,
    smoothing: float = 10.0,
) -> tuple[np.ndarray, dict[Any, float]]:
    """Vectorized target encoding with smoothing.

    Args:
        values: Category array.
        targets: Numeric target array.
        smoothing: Smoothing factor (higher = more regularization).

    Returns:
        Tuple of (encoded values, category-to-mean mapping).
    """
    global_mean = float(np.mean(targets))
    mapping: dict[Any, float] = {}
    cats = np.unique(values)

    for cat in cats:
        mask = values == cat
        n = int(np.sum(mask))
        cat_mean = float(np.mean(targets[mask]))
        # Bayesian smoothing
        mapping[cat] = (n * cat_mean + smoothing * global_mean) / (n + smoothing)

    result = np.array([mapping.get(v, global_mean) for v in values], dtype=np.float64)
    return result, mapping


def fast_interaction(
    col_a: np.ndarray,
    col_b: np.ndarray,
    operation: str = "multiply",
) -> np.ndarray:
    """Vectorized interaction feature computation.

    Args:
        col_a: First column.
        col_b: Second column.
        operation: One of "multiply", "add", "subtract", "divide".

    Returns:
        Result array.
    """
    if operation == "multiply":
        return col_a * col_b
    elif operation == "add":
        return col_a + col_b
    elif operation == "subtract":
        return col_a - col_b
    elif operation == "divide":
        return np.where(col_b != 0, col_a / col_b, 0.0)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def fast_quantile_bin(values: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized quantile binning.

    Args:
        values: 1-D numeric array.
        n_bins: Number of quantile bins.

    Returns:
        Tuple of (bin assignments, bin edges).
    """
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, quantiles)
    edges[-1] = edges[-1] + 1e-10  # Ensure max value is included
    bins = np.digitize(values, edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins, edges


def fast_rolling_stats(
    values: np.ndarray,
    window: int = 3,
    stats: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Vectorized rolling statistics using cumulative sums.

    Args:
        values: 1-D numeric array.
        window: Window size.
        stats: List of stats to compute. Default: ["mean", "std"].

    Returns:
        Dict mapping stat name to result array.
    """
    stats = stats or ["mean", "std"]
    n = len(values)
    result: dict[str, np.ndarray] = {}

    if "mean" in stats or "std" in stats:
        # Cumulative sum approach for O(n) rolling mean
        cs = np.cumsum(np.insert(values, 0, 0))
        rolling_sum = np.empty(n)
        rolling_sum[:window] = np.nan
        rolling_sum[window:] = cs[window + 1:n + 1] - cs[1:n - window + 1]
        # First window element: we have exactly window values
        if n >= window:
            rolling_sum[window - 1] = cs[window] - cs[0]

        if "mean" in stats:
            rolling_mean = rolling_sum / window
            result["mean"] = rolling_mean

        if "std" in stats:
            cs2 = np.cumsum(np.insert(values ** 2, 0, 0))
            rolling_sum2 = np.empty(n)
            rolling_sum2[:window] = np.nan
            rolling_sum2[window:] = cs2[window + 1:n + 1] - cs2[1:n - window + 1]
            if n >= window:
                rolling_sum2[window - 1] = cs2[window] - cs2[0]

            rolling_mean = rolling_sum / window
            var = rolling_sum2 / window - rolling_mean ** 2
            var = np.maximum(var, 0)  # Numerical stability
            result["std"] = np.sqrt(var)

    if "min" in stats:
        rm = np.full(n, np.nan)
        for i in range(window - 1, n):
            rm[i] = np.min(values[i - window + 1: i + 1])
        result["min"] = rm

    if "max" in stats:
        rm = np.full(n, np.nan)
        for i in range(window - 1, n):
            rm[i] = np.max(values[i - window + 1: i + 1])
        result["max"] = rm

    return result


def fast_polynomial_features(
    X: np.ndarray,
    degree: int = 2,
    interaction_only: bool = False,
) -> np.ndarray:
    """Vectorized polynomial feature generation.

    Args:
        X: 2-D numeric array (n_samples, n_features).
        degree: Maximum polynomial degree.
        interaction_only: If True, only interaction terms.

    Returns:
        Array with polynomial features appended.
    """
    n_samples, n_features = X.shape
    features = [X]

    if degree >= 2:
        if not interaction_only:
            # Squared terms
            features.append(X ** 2)

        # Interaction terms
        for i in range(n_features):
            for j in range(i + 1, n_features):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))

    if degree >= 3 and not interaction_only:
        features.append(X ** 3)

    return np.hstack(features)


def fast_log_transform(values: np.ndarray, offset: float = 1.0) -> np.ndarray:
    """Safe log transform with offset.

    Args:
        values: Input array.
        offset: Added to values before log to avoid log(0).

    Returns:
        log(values + offset) array.
    """
    return np.log(np.maximum(values + offset, 1e-10))


def fast_null_indicator(values: np.ndarray) -> np.ndarray:
    """Create null/NaN indicator column.

    Returns:
        Binary array: 1 where NaN, 0 otherwise.
    """
    return np.where(np.isnan(values), 1.0, 0.0)


# ── Engine dispatcher ────────────────────────────────────────────────


class Engine:
    """High-performance feature engineering engine.

    Dispatches to the fastest available backend:
    1. Rust/PyO3 bindings (future, not yet available)
    2. NumPy vectorized (current default)

    Parameters:
        backend: "auto", "numpy", or "rust" (future).
    """

    def __init__(self, backend: str = "auto") -> None:
        self.backend = backend
        self._rust_available = False

        if backend == "rust":
            try:
                import forge_rust  # noqa: F401
                self._rust_available = True
            except ImportError:
                logger.warning("Rust backend not available, falling back to NumPy")

    @property
    def active_backend(self) -> str:
        if self._rust_available:
            return "rust"
        return "numpy"

    def one_hot(self, values: np.ndarray, categories: np.ndarray | None = None) -> np.ndarray:
        """One-hot encode values."""
        return fast_one_hot(values, categories)

    def target_encode(
        self,
        values: np.ndarray,
        targets: np.ndarray,
        smoothing: float = 10.0,
    ) -> tuple[np.ndarray, dict[Any, float]]:
        """Target encode with smoothing."""
        return fast_target_encode(values, targets, smoothing)

    def interaction_features(
        self,
        X: np.ndarray,
        pairs: list[tuple[int, int, str]],
    ) -> np.ndarray:
        """Compute interaction features for column index pairs.

        Args:
            X: 2-D array.
            pairs: List of (col_a_idx, col_b_idx, operation).

        Returns:
            Array of interaction features (n_samples, len(pairs)).
        """
        results = []
        for a_idx, b_idx, op in pairs:
            results.append(fast_interaction(X[:, a_idx], X[:, b_idx], op))
        return np.column_stack(results)

    def quantile_bin(self, values: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Quantile-based binning."""
        return fast_quantile_bin(values, n_bins)

    def rolling_stats(
        self,
        values: np.ndarray,
        window: int = 3,
        stats: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Rolling window statistics."""
        return fast_rolling_stats(values, window, stats)

    def polynomial_features(
        self,
        X: np.ndarray,
        degree: int = 2,
        interaction_only: bool = False,
    ) -> np.ndarray:
        """Polynomial feature generation."""
        return fast_polynomial_features(X, degree, interaction_only)

    def log_transform(self, values: np.ndarray, offset: float = 1.0) -> np.ndarray:
        """Safe log transform."""
        return fast_log_transform(values, offset)

    def null_indicator(self, values: np.ndarray) -> np.ndarray:
        """Null/NaN indicator."""
        return fast_null_indicator(values)

    def batch_transform(
        self,
        X: np.ndarray,
        operations: list[dict[str, Any]],
    ) -> np.ndarray:
        """Apply a batch of operations to data.

        Args:
            X: Input 2-D array.
            operations: List of operation dicts, each with "op" key
                and relevant parameters.

        Returns:
            Horizontally stacked result of all operations.
        """
        results = [X]
        for op_spec in operations:
            op = op_spec["op"]
            if op == "log":
                col = op_spec.get("col", 0)
                offset = op_spec.get("offset", 1.0)
                results.append(
                    fast_log_transform(X[:, col], offset).reshape(-1, 1)
                )
            elif op == "null_indicator":
                col = op_spec.get("col", 0)
                results.append(
                    fast_null_indicator(X[:, col].astype(float)).reshape(-1, 1)
                )
            elif op == "interaction":
                a, b = op_spec["cols"]
                iop = op_spec.get("interaction_op", "multiply")
                results.append(
                    fast_interaction(X[:, a], X[:, b], iop).reshape(-1, 1)
                )
            elif op == "polynomial":
                degree = op_spec.get("degree", 2)
                results.append(
                    fast_polynomial_features(X, degree, interaction_only=True)
                )
            else:
                raise ValueError(f"Unknown batch operation: {op}")
        return np.hstack(results)
