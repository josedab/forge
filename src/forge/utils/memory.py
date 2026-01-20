"""Memory estimation and monitoring utilities.

This module provides tools for estimating memory usage and preventing
out-of-memory errors during feature engineering operations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class MemoryEstimate:
    """Memory usage estimate for a feature engineering operation.

    Attributes
    ----------
    input_size_mb : float
        Estimated size of input data in MB.
    output_size_mb : float
        Estimated size of output data in MB.
    peak_size_mb : float
        Estimated peak memory usage during processing in MB.
    available_mb : float
        Available system memory in MB.
    is_safe : bool
        Whether the operation is safe to perform given available memory.
    recommendation : str
        Recommended action if memory may be insufficient.
    """

    input_size_mb: float
    output_size_mb: float
    peak_size_mb: float
    available_mb: float
    is_safe: bool
    recommendation: str

    def __str__(self) -> str:
        """Return human-readable summary."""
        status = "✓ Safe" if self.is_safe else "⚠ Warning"
        return (
            f"Memory Estimate ({status}):\n"
            f"  Input:     {self.input_size_mb:,.1f} MB\n"
            f"  Output:    {self.output_size_mb:,.1f} MB\n"
            f"  Peak:      {self.peak_size_mb:,.1f} MB\n"
            f"  Available: {self.available_mb:,.1f} MB\n"
            f"  {self.recommendation}"
        )


def get_available_memory() -> float:
    """Get available system memory in MB.

    Returns
    -------
    float
        Available memory in megabytes.
    """
    try:
        import psutil

        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        # If psutil not available, assume 4GB available
        return 4096.0


def estimate_dataframe_size(df: pd.DataFrame) -> float:
    """Estimate memory usage of a DataFrame in MB.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to estimate.

    Returns
    -------
    float
        Estimated size in megabytes.
    """
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def estimate_feature_engineering_memory(
    X: pd.DataFrame,
    max_features: int | None = None,
    n_generators: int = 5,
    expansion_factor: float = 3.0
) -> MemoryEstimate:
    """Estimate memory requirements for feature engineering.

    This function estimates the peak memory usage during feature engineering
    and provides recommendations if memory may be insufficient.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    max_features : int | None
        Maximum number of output features. If None, no limit.
    n_generators : int
        Estimated number of generators to run.
    expansion_factor : float
        Multiplier for feature expansion (default 3x for interactions).

    Returns
    -------
    MemoryEstimate
        Memory usage estimate with recommendations.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({'a': range(100000), 'b': range(100000)})
    >>> estimate = estimate_feature_engineering_memory(X)
    >>> print(estimate)
    Memory Estimate (✓ Safe):
      Input:     1.5 MB,
      Output:    4.5 MB,
      Peak:      7.5 MB,
      Available: 8,192.0 MB
      Sufficient memory available.
    """
    input_size = estimate_dataframe_size(X)
    n_rows, n_cols = X.shape

    # Estimate output size based on expansion factor
    estimated_output_cols = int(n_cols * expansion_factor * n_generators)
    if max_features is not None:
        estimated_output_cols = min(estimated_output_cols, max_features)

    # Estimate bytes per cell (average of 8 bytes for numeric, 50 for object)
    numeric_cols = X.select_dtypes(include=[np.number]).shape[1]
    object_cols = n_cols - numeric_cols
    avg_bytes_per_cell = (numeric_cols * 8 + object_cols * 50) / max(n_cols, 1)

    output_size = (n_rows * estimated_output_cols * avg_bytes_per_cell) / (1024 * 1024)

    # Peak memory includes input, intermediate, and output
    # Factor of 2.5 accounts for intermediate computations
    peak_size = input_size + output_size * 2.5

    available = get_available_memory()
    safety_margin = 0.8  # Leave 20% headroom
    is_safe = bool(peak_size < available * safety_margin)

    if is_safe:
        recommendation = "Sufficient memory available."
    elif peak_size < available:
        recommendation = (
            f"Memory usage ({peak_size:.0f} MB) is close to available "
            f"({available:.0f} MB). Consider using max_features to limit output."
        )
    else:
        suggested_max = int(max_features * (available * safety_margin / peak_size)) if max_features else 50
        recommendation = (
            f"⚠ Estimated peak ({peak_size:.0f} MB) exceeds available memory "
            f"({available:.0f} MB).\n"
            f"  Suggestions:\n"
            f"  1. Use max_features={suggested_max} or lower\n"
            f"  2. Process data in chunks using process_in_chunks()\n"
            f"  3. Reduce input data size or disable memory-intensive generators"
        )

    return MemoryEstimate(
        input_size_mb = input_size,
        output_size_mb = output_size,
        peak_size_mb = peak_size,
        available_mb = available,
        is_safe = is_safe,
        recommendation=recommendation
    )


def check_memory_and_warn(
    X: pd.DataFrame,
    max_features: int | None = None,
    raise_on_danger: bool = False
) -> MemoryEstimate:
    """Check memory and emit warning if insufficient.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    max_features : int | None
        Maximum number of output features.
    raise_on_danger : bool
        If True, raise MemoryError when memory is insufficient.
        If False (default), only emit a warning.

    Returns
    -------
    MemoryEstimate
        Memory usage estimate.

    Raises
    ------
    MemoryError
        If raise_on_danger is True and memory is insufficient.

    Warns
    -----
    ResourceWarning
        If memory usage is close to or exceeds available memory.
    """
    estimate = estimate_feature_engineering_memory(X, max_features)

    if not estimate.is_safe:
        msg = (
            f"Feature engineering may exceed available memory.\n"
            f"Estimated peak: {estimate.peak_size_mb:.0f} MB, "
            f"Available: {estimate.available_mb:.0f} MB.\n"
            f"{estimate.recommendation}"
        )
        if raise_on_danger:
            raise MemoryError(msg)
        warnings.warn(msg, ResourceWarning, stacklevel=2)

    return estimate


def process_in_chunks(
    transformer,
    X: pd.DataFrame,
    y: pd.Series | None = None,
    chunk_size: int = 100_000,
    verbose: bool = True
) -> pd.DataFrame:
    """Process a large DataFrame in chunks to avoid memory issues.

    This function fits the transformer on the first chunk and then
    transforms all chunks, concatenating the results.

    Parameters
    ----------
    transformer : BaseEstimator
        A fitted or unfitted transformer with fit_transform/transform methods.
    X : pd.DataFrame
        Input DataFrame.
    y : pd.Series | None
        Optional target variable.
    chunk_size : int
        Number of rows per chunk.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    pd.DataFrame
        Concatenated transformed data.

    Examples
    --------
    >>> from forge import AutoFeatureTransformer
    >>> transformer = AutoFeatureTransformer(max_features=50)
    >>> result = process_in_chunks(transformer, large_df, y, chunk_size=100_000)
    """
    import pandas as pd

    n_rows = len(X)
    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    results = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_rows)

        X_chunk = X.iloc[start_idx:end_idx]
        y_chunk = y.iloc[start_idx:end_idx] if y is not None else None

        if verbose:
            print(f"Processing chunk {i + 1}/{n_chunks} (rows {start_idx}-{end_idx})...")

        if i == 0:
            # Fit on first chunk
            result_chunk = transformer.fit_transform(X_chunk, y_chunk)
        else:
            # Transform subsequent chunks
            result_chunk = transformer.transform(X_chunk)

        results.append(result_chunk)

    if verbose:
        print(f"Concatenating {n_chunks} chunks...")

    return pd.concat(results, ignore_index=True)


def get_memory_usage_summary(X: pd.DataFrame) -> dict[str, float]:
    """Get detailed memory usage breakdown by column type.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    dict[str, float]
        Memory usage breakdown in MB.
    """
    total = estimate_dataframe_size(X)

    numeric_cols = X.select_dtypes(include=[np.number])
    categorical_cols = X.select_dtypes(include=["category"])
    object_cols = X.select_dtypes(include=["object"])
    datetime_cols = X.select_dtypes(include=["datetime64"])

    return {
        "total_mb": total,
        "numeric_mb": estimate_dataframe_size(numeric_cols) if len(numeric_cols.columns) > 0 else 0,
        "categorical_mb": estimate_dataframe_size(categorical_cols) if len(categorical_cols.columns) > 0 else 0,
        "object_mb": estimate_dataframe_size(object_cols) if len(object_cols.columns) > 0 else 0,
        "datetime_mb": estimate_dataframe_size(datetime_cols) if len(datetime_cols.columns) > 0 else 0,
        "rows": len(X),
        "columns": len(X.columns),
    }


def suggest_dtype_optimizations(X: pd.DataFrame) -> list[str]:
    """Suggest dtype optimizations to reduce memory usage.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list[str]
        List of optimization suggestions.
    """
    suggestions = []

    for col in X.columns:
        dtype = X[col].dtype

        # Check for object columns that could be category
        if dtype == "object":
            n_unique = X[col].nunique()
            n_rows = len(X)
            if n_unique / n_rows < 0.5:  # Less than 50% unique
                suggestions.append(
                    f"Column '{col}': Convert to category dtype "
                    f"({n_unique} unique values, {n_unique/n_rows*100:.1f}% cardinality)"
                )

        # Check for int64 that could be smaller
        elif dtype == np.int64:
            min_val, max_val = X[col].min(), X[col].max()
            if min_val >= 0 and max_val <= 255:
                suggestions.append(f"Column '{col}': Can use uint8 (range 0-{max_val})")
            elif min_val >= -128 and max_val <= 127:
                suggestions.append(f"Column '{col}': Can use int8 (range {min_val}-{max_val})")
            elif min_val >= 0 and max_val <= 65535:
                suggestions.append(f"Column '{col}': Can use uint16 (range 0-{max_val})")
            elif min_val >= -32768 and max_val <= 32767:
                suggestions.append(f"Column '{col}': Can use int16 (range {min_val}-{max_val})")

        # Check for float64 that could be float32
        elif dtype == np.float64:
            if X[col].abs().max() < 3.4e38:  # float32 range
                suggestions.append(f"Column '{col}': Can use float32 to save memory")

    return suggestions
