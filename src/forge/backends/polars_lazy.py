"""Polars lazy evaluation and streaming for large datasets.

Extends the Polars backend with:
- Lazy feature generation pipeline using Polars LazyFrame
- Streaming mode for datasets larger than memory
- Feature generation pipeline that compiles to a Polars plan
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_POLARS_AVAILABLE = False
try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore[assignment]


class PolarsLazyPipeline:
    """Lazy feature generation pipeline using Polars LazyFrames.

    Builds a query plan that only executes when `.collect()` is called,
    enabling Polars to optimize the execution graph.

    Args:
        include_originals: Whether to include original columns in output.

    Example:
        >>> pipeline = PolarsLazyPipeline()
        >>> pipeline.add_interaction("price", "quantity")
        >>> pipeline.add_log("revenue")
        >>> pipeline.add_polynomial("age", degree=2)
        >>> result_df = pipeline.execute(df)
    """

    def __init__(self, include_originals: bool = True) -> None:
        if not _POLARS_AVAILABLE:
            raise ImportError("Polars is required: pip install polars")
        self.include_originals = include_originals
        self._operations: list[dict[str, Any]] = []

    def add_interaction(self, col_a: str, col_b: str, name: str | None = None) -> PolarsLazyPipeline:
        """Add a multiplication interaction feature."""
        self._operations.append({
            "type": "interaction",
            "col_a": col_a,
            "col_b": col_b,
            "name": name or f"{col_a}_x_{col_b}",
        })
        return self

    def add_ratio(self, col_a: str, col_b: str, name: str | None = None) -> PolarsLazyPipeline:
        """Add a ratio feature (a / b, safe division)."""
        self._operations.append({
            "type": "ratio",
            "col_a": col_a,
            "col_b": col_b,
            "name": name or f"{col_a}_div_{col_b}",
        })
        return self

    def add_log(self, column: str, name: str | None = None) -> PolarsLazyPipeline:
        """Add log1p transformation."""
        self._operations.append({
            "type": "log",
            "column": column,
            "name": name or f"{column}_log1p",
        })
        return self

    def add_sqrt(self, column: str, name: str | None = None) -> PolarsLazyPipeline:
        """Add square root transformation."""
        self._operations.append({
            "type": "sqrt",
            "column": column,
            "name": name or f"{column}_sqrt",
        })
        return self

    def add_polynomial(self, column: str, degree: int = 2) -> PolarsLazyPipeline:
        """Add polynomial features up to the given degree."""
        for d in range(2, degree + 1):
            self._operations.append({
                "type": "power",
                "column": column,
                "degree": d,
                "name": f"{column}_pow{d}",
            })
        return self

    def add_bin(self, column: str, n_bins: int = 10, name: str | None = None) -> PolarsLazyPipeline:
        """Add quantile binning."""
        self._operations.append({
            "type": "bin",
            "column": column,
            "n_bins": n_bins,
            "name": name or f"{column}_bin{n_bins}",
        })
        return self

    def add_all_interactions(self, columns: list[str]) -> PolarsLazyPipeline:
        """Add all pairwise interactions between columns."""
        for c1, c2 in combinations(columns, 2):
            self.add_interaction(c1, c2)
        return self

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the pipeline on a pandas DataFrame.

        Converts to Polars, builds lazy plan, collects, and converts back.

        Args:
            df: Input pandas DataFrame.

        Returns:
            pandas DataFrame with generated features.
        """
        lf = pl.from_pandas(df).lazy()
        lf = self._build_plan(lf)
        result = lf.collect()
        result_pd = result.to_pandas()
        result_pd.index = df.index
        return result_pd

    def execute_lazy(self, lf: Any) -> Any:
        """Execute on a Polars LazyFrame (stays lazy until collected).

        Args:
            lf: Polars LazyFrame.

        Returns:
            Polars LazyFrame with operations applied.
        """
        return self._build_plan(lf)

    def _build_plan(self, lf: Any) -> Any:
        """Build the Polars lazy execution plan."""
        new_exprs: list[Any] = []

        for op in self._operations:
            if op["type"] == "interaction":
                new_exprs.append(
                    (pl.col(op["col_a"]) * pl.col(op["col_b"])).alias(op["name"])
                )
            elif op["type"] == "ratio":
                new_exprs.append(
                    pl.when(pl.col(op["col_b"]) != 0)
                    .then(pl.col(op["col_a"]) / pl.col(op["col_b"]))
                    .otherwise(0.0)
                    .alias(op["name"])
                )
            elif op["type"] == "log":
                new_exprs.append(
                    pl.col(op["column"]).clip(0).log1p().alias(op["name"])
                )
            elif op["type"] == "sqrt":
                new_exprs.append(
                    pl.col(op["column"]).clip(0).sqrt().alias(op["name"])
                )
            elif op["type"] == "power":
                new_exprs.append(
                    pl.col(op["column"]).pow(op["degree"]).alias(op["name"])
                )
            elif op["type"] == "bin":
                # Binning requires eager evaluation
                new_exprs.append(
                    pl.col(op["column"])
                    .qcut(op["n_bins"], labels=[str(i) for i in range(op["n_bins"])])
                    .alias(op["name"])
                )

        if self.include_originals:
            return lf.with_columns(new_exprs)
        else:
            return lf.select(new_exprs)

    @property
    def operation_count(self) -> int:
        """Number of operations in the pipeline."""
        return len(self._operations)

    def describe(self) -> list[str]:
        """Human-readable description of operations."""
        return [
            f"{op['type']}: {op.get('name', 'unnamed')}" for op in self._operations
        ]


def process_in_streaming_chunks(
    source_path: str,
    pipeline: PolarsLazyPipeline,
    output_path: str | None = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame | None:
    """Process a large file in streaming chunks using Polars.

    Reads the file in chunks, applies the pipeline to each chunk,
    and optionally writes results to a Parquet file.

    Args:
        source_path: Path to source CSV or Parquet file.
        pipeline: PolarsLazyPipeline to apply.
        output_path: Optional path to write results (Parquet).
        chunk_size: Rows per chunk.

    Returns:
        Combined DataFrame if no output_path, else None.
    """
    if not _POLARS_AVAILABLE:
        raise ImportError("Polars is required for streaming: pip install polars")

    # Use Polars scan for lazy file reading
    if source_path.endswith(".parquet"):
        lf = pl.scan_parquet(source_path)
    else:
        lf = pl.scan_csv(source_path)

    # Apply pipeline lazily
    result_lf = pipeline.execute_lazy(lf)

    if output_path:
        result_lf.sink_parquet(output_path)
        return None

    return result_lf.collect().to_pandas()


class PolarsFeatureGenerator:
    """High-level feature generator that uses Polars for all operations.

    Provides a simple API for common feature generation patterns,
    automatically using Polars for performance.

    Args:
        interactions: Generate pairwise interactions.
        polynomials: Generate polynomial features.
        log_transforms: Apply log transforms to skewed columns.
        max_degree: Maximum polynomial degree.
        skewness_threshold: Skewness threshold for log transform.

    Example:
        >>> gen = PolarsFeatureGenerator(interactions=True, polynomials=True)
        >>> X_new = gen.fit_transform(df)
    """

    def __init__(
        self,
        interactions: bool = True,
        polynomials: bool = False,
        log_transforms: bool = True,
        max_degree: int = 2,
        skewness_threshold: float = 2.0,
    ) -> None:
        if not _POLARS_AVAILABLE:
            raise ImportError("Polars required: pip install polars")
        self.interactions = interactions
        self.polynomials = polynomials
        self.log_transforms = log_transforms
        self.max_degree = max_degree
        self.skewness_threshold = skewness_threshold
        self._pipeline: PolarsLazyPipeline | None = None
        self._numeric_cols: list[str] = []
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, y: Any = None) -> PolarsFeatureGenerator:
        """Analyze the DataFrame and build the feature pipeline.

        Args:
            df: Training data.
            y: Ignored.

        Returns:
            Self for chaining.
        """
        self._numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        self._pipeline = PolarsLazyPipeline(include_originals=True)

        if self.interactions and len(self._numeric_cols) >= 2:
            cols = self._numeric_cols[:10]  # Limit to avoid combinatorial explosion
            self._pipeline.add_all_interactions(cols)

        if self.polynomials:
            for col in self._numeric_cols[:5]:
                self._pipeline.add_polynomial(col, degree=self.max_degree)

        if self.log_transforms:
            for col in self._numeric_cols:
                skew = float(df[col].skew())
                if abs(skew) > self.skewness_threshold:
                    self._pipeline.add_log(col)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features using the fitted pipeline.

        Args:
            df: Data to transform.

        Returns:
            DataFrame with generated features.
        """
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("PolarsFeatureGenerator not fitted")
        return self._pipeline.execute(df)

    def fit_transform(self, df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, y)
        return self.transform(df)

    def get_feature_names_out(self) -> list[str]:
        """Get names of generated features."""
        if self._pipeline is None:
            return []
        return [op.get("name", "") for op in self._pipeline._operations]
