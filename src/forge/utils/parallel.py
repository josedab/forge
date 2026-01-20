"""Parallel processing utilities."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


def get_n_jobs(n_jobs: int = -1) -> int:
    """Get actual number of jobs to use.

    Args:
        n_jobs: Number of jobs. -1 means use all CPUs.,

    Returns:
        Actual number of jobs.
    """
    if n_jobs == -1:
        return os.cpu_count() or 1
    elif n_jobs == 0:
        return 1
    else:
        return min(n_jobs, os.cpu_count() or 1)


def parallel_apply(
    func: Callable[..., Any],
    items: Iterable[Any],
    n_jobs: int = -1,
    backend: Literal["threading", "multiprocessing"] = "threading",
    **kwargs: Any,
) -> list[Any]:
    """Apply function to items in parallel.

    Args:
        func: Function to apply.,
        items: Items to process.,
        n_jobs: Number of parallel jobs. -1 for all CPUs.,
        backend: Parallelization backend.
        **kwargs: Additional arguments passed to func.

    Returns:
        List of results.

    Example:
        >>> def process(x):
        ...     return x ** 2
        >>> results = parallel_apply(process, [1, 2, 3, 4], n_jobs=2)
    """
    items_list = list(items)

    if len(items_list) == 0:
        return []

    actual_jobs = get_n_jobs(n_jobs)

    # Don't parallelize for small workloads
    if actual_jobs == 1 or len(items_list) <= 2:
        return [func(item, **kwargs) for item in items_list]

    Executor = ThreadPoolExecutor if backend == "threading" else ProcessPoolExecutor

    with Executor(max_workers=actual_jobs) as executor:
        if kwargs:
            futures = [executor.submit(func, item, **kwargs) for item in items_list]
            return [f.result() for f in futures]
        else:
            return list(executor.map(func, items_list))


def parallel_transform(
    transformers: Sequence[tuple[str, Any]],
    X: pd.DataFrame,
    n_jobs: int = -1,
    method: str = "transform"
) -> dict[str, pd.DataFrame]:
    """Apply multiple transformers in parallel.

    Args:
        transformers: List of (name, transformer) tuples.
        X: Input DataFrame.,
        n_jobs: Number of parallel jobs.,
        method: Method to call on transformers.,

    Returns:
        Dictionary mapping names to transformed DataFrames.

    Example:
        >>> transformers = [
        ...     ("numeric", NumericTransformer()),
        ...     ("categorical", CategoricalTransformer()),
        ... ]
        >>> results = parallel_transform(transformers, X)
    """
    if len(transformers) == 0:
        return {}

    actual_jobs = get_n_jobs(n_jobs)

    def apply_transform(name_transformer: tuple[str, Any]) -> tuple[str, pd.DataFrame]:
        name, transformer = name_transformer
        func = getattr(transformer, method)
        result = func(X)
        return name, result

    # Don't parallelize for small workloads
    if actual_jobs == 1 or len(transformers) <= 2:
        return dict(apply_transform(t) for t in transformers)

    with ThreadPoolExecutor(max_workers=actual_jobs) as executor:
        results = list(executor.map(apply_transform, transformers))

    return dict(results)


def chunked_apply(
    func: Callable[[pd.DataFrame], pd.DataFrame],
    X: pd.DataFrame,
    chunk_size: int = 10000,
    n_jobs: int = 1
) -> pd.DataFrame:
    """Apply function to DataFrame in chunks.

    Useful for memory-efficient processing of large DataFrames.

    Args:
        func: Function to apply to each chunk.,
        X: Input DataFrame.,
        chunk_size: Number of rows per chunk.,
        n_jobs: Number of parallel jobs.,

    Returns:
        Concatenated results.

    Example:
        >>> def expensive_transform(df):
        ...     return df.apply(lambda x: x ** 2)
        >>> result = chunked_apply(expensive_transform, large_df, chunk_size=5000)
    """
    n_rows = len(X)

    if n_rows <= chunk_size:
        return func(X)

    # Split into chunks
    chunks = [
        X.iloc[i : i + chunk_size] for i in range(0, n_rows, chunk_size)
    ]

    actual_jobs = get_n_jobs(n_jobs)

    if actual_jobs == 1:
        results = [func(chunk) for chunk in chunks]
    else:
        with ThreadPoolExecutor(max_workers=actual_jobs) as executor:
            results = list(executor.map(func, chunks))

    return pd.concat(results, axis=0, ignore_index=True)


def parallel_column_apply(
    func: Callable[[pd.Series], pd.Series],
    X: pd.DataFrame,
    columns: list[str] | None = None,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Apply function to columns in parallel.

    Args:
        func: Function to apply to each column.,
        X: Input DataFrame.,
        columns: Columns to process. None for all.,
        n_jobs: Number of parallel jobs.,

    Returns:
        DataFrame with transformed columns.

    Example:
        >>> def normalize(col):
        ...     return (col - col.mean()) / col.std()
        >>> result = parallel_column_apply(normalize, df, n_jobs=-1)
    """
    if columns is None:
        columns = X.columns.tolist()

    actual_jobs = get_n_jobs(n_jobs)

    def process_column(col_name: str) -> tuple[str, pd.Series]:
        return col_name, func(X[col_name])

    if actual_jobs == 1 or len(columns) <= 2:
        results = {col: func(X[col]) for col in columns}
    else:
        with ThreadPoolExecutor(max_workers=actual_jobs) as executor:
            result_pairs = list(executor.map(process_column, columns))
            results = dict(result_pairs)

    return pd.DataFrame(results)


class ParallelProcessor:
    """Context manager for parallel processing with consistent settings.

    Example:
        >>> with ParallelProcessor(n_jobs=4) as proc:
        ...     results = proc.map(expensive_func, items)
    """

    def __init__(
        self,
        n_jobs: int = -1,
        backend: Literal["threading", "multiprocessing"] = "threading"
    ):
        """Initialize processor.

        Args:
            n_jobs: Number of parallel jobs.,
            backend: Parallelization backend.
        """
        self.n_jobs = get_n_jobs(n_jobs)
        self.backend = backend
        self._executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None

    def __enter__(self) -> "ParallelProcessor":
        if self.backend == "threading":
            self._executor = ThreadPoolExecutor(max_workers=self.n_jobs)
        else:
            self._executor = ProcessPoolExecutor(max_workers=self.n_jobs)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def map(
        self,
        func: Callable[..., Any],
        items: Iterable[Any],
    ) -> list[Any]:
        """Map function over items.

        Args:
            func: Function to apply.,
            items: Items to process.,

        Returns:
            List of results.
        """
        if self._executor is None:
            raise RuntimeError("Processor must be used as context manager")

        items_list = list(items)
        if not items_list:
            return []

        return list(self._executor.map(func, items_list))

    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Submit a single task.

        Args:
            func: Function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Future object.
        """
        if self._executor is None:
            raise RuntimeError("Processor must be used as context manager")

        return self._executor.submit(func, *args, **kwargs)


def batch_generator(
    items: Sequence[Any],
    batch_size: int,
) -> Iterable[list[Any]]:
    """Generate batches from items.

    Args:
        items: Items to batch.,
        batch_size: Size of each batch.,

    Yields:
        Batches of items.

    Example:
        >>> for batch in batch_generator(range(10), 3):
        ...     print(batch)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    """
    items_list = list(items)
    for i in range(0, len(items_list), batch_size):
        yield items_list[i : i + batch_size]


def estimate_memory_usage(X: pd.DataFrame) -> dict[str, Any]:
    """Estimate memory usage of DataFrame.

    Args:
        X: Input DataFrame.,

    Returns:
        Dictionary with memory statistics.

    Example:
        >>> stats = estimate_memory_usage(df)
        >>> print(f"Total: {stats['total_mb']:.2f} MB")
    """
    memory_usage = X.memory_usage(deep=True)

    return {
        "total_bytes": memory_usage.sum(),
        "total_mb": memory_usage.sum() / (1024 * 1024),
        "per_column": memory_usage.to_dict(),
        "n_rows": len(X),
        "n_cols": len(X.columns),
        "avg_bytes_per_row": memory_usage.sum() / max(len(X), 1),
    }


def suggest_chunk_size(
    X: pd.DataFrame,
    target_memory_mb: float = 100.0
) -> int:
    """Suggest chunk size based on available memory.

    Args:
        X: Input DataFrame.,
        target_memory_mb: Target memory per chunk in MB.,

    Returns:
        Suggested chunk size (number of rows).

    Example:
        >>> chunk_size = suggest_chunk_size(large_df, target_memory_mb=50)
        >>> for chunk in chunked_apply(func, large_df, chunk_size=chunk_size):
        ...     process(chunk)
    """
    stats = estimate_memory_usage(X)
    bytes_per_row = stats["avg_bytes_per_row"]

    if bytes_per_row == 0:
        return len(X)

    target_bytes = target_memory_mb * 1024 * 1024
    suggested_size = int(target_bytes / bytes_per_row)

    # Ensure reasonable bounds
    return max(100, min(suggested_size, len(X)))
