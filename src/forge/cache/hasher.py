"""Content-addressable hashing for DataFrames and transformer parameters."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


class DataFrameHasher:
    """Compute deterministic hashes of DataFrames and parameters.

    Uses a combination of schema, shape, and sampled data to produce
    a fast, collision-resistant hash without reading every cell.

    Args:
        sample_rows: Number of rows to sample for hashing.
        include_schema: Whether to include column names and dtypes.
        include_shape: Whether to include row/column counts.

    Example:
        >>> hasher = DataFrameHasher()
        >>> key = hasher.hash_key(X, params={"degree": 2})
    """

    def __init__(
        self,
        sample_rows: int = 100,
        include_schema: bool = True,
        include_shape: bool = True,
    ) -> None:
        self.sample_rows = sample_rows
        self.include_schema = include_schema
        self.include_shape = include_shape

    def hash_dataframe(self, df: pd.DataFrame) -> str:
        """Compute a hash of a DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Hex digest string.
        """
        h = hashlib.sha256()

        if self.include_shape:
            h.update(f"shape:{df.shape}".encode())

        if self.include_schema:
            schema = "|".join(f"{c}:{d}" for c, d in zip(df.columns, df.dtypes))
            h.update(schema.encode())

        # Sample deterministic rows for content hash
        if len(df) <= self.sample_rows:
            sample = df
        else:
            indices = np.linspace(0, len(df) - 1, self.sample_rows, dtype=int)
            sample = df.iloc[indices]

        # Hash the sampled content
        h.update(pd.util.hash_pandas_object(sample).values.tobytes())  # type: ignore[union-attr]

        return h.hexdigest()

    def hash_params(self, params: dict[str, Any]) -> str:
        """Compute a hash of parameter dictionary.

        Args:
            params: Parameter dictionary.

        Returns:
            Hex digest string.
        """
        serialized = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def hash_key(
        self,
        df: pd.DataFrame,
        params: dict[str, Any] | None = None,
        transformer_name: str = "",
    ) -> str:
        """Compute a combined cache key from data + params + transformer.

        Args:
            df: Input DataFrame.
            params: Transformer parameters.
            transformer_name: Name of the transformer class.

        Returns:
            Combined hex digest cache key.
        """
        h = hashlib.sha256()
        h.update(self.hash_dataframe(df).encode())
        if params:
            h.update(self.hash_params(params).encode())
        if transformer_name:
            h.update(transformer_name.encode())
        return h.hexdigest()
