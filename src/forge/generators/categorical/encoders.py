"""Categorical encoding generators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class OneHotEncoder(BaseFeatureGenerator):
    """One-hot encodes categorical columns.

    Creates binary indicator columns for each category.

    Example:
        >>> encoder = OneHotEncoder(columns=["color"])
        >>> X_encoded = encoder.fit_transform(X)
        # Creates: color_red, color_blue, color_green
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        max_categories: int = 50,
        handle_unknown: str = "ignore",
        drop_first: bool = False,
        min_frequency: float | int | None = None
    ) -> None:
        """Initialize the one-hot encoder.

        Args:
            columns: Columns to encode. If None, all object/category columns.,
            max_categories: Maximum categories per column to encode.,
            handle_unknown: How to handle unknown categories ("ignore", "error").,
            drop_first: Whether to drop the first category (avoid collinearity).,
            min_frequency: Minimum frequency for a category to be encoded.
        """
        super().__init__()
        self.columns = columns
        self.max_categories = max_categories
        self.handle_unknown = handle_unknown
        self.drop_first = drop_first
        self.min_frequency = min_frequency

        self._categories: dict[str, list[Any]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the encoder by learning categories.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._categories = {}
        self._feature_names_out = []

        for col in cols:
            value_counts = X[col].value_counts()

            # Filter by frequency if specified
            if self.min_frequency is not None:
                if isinstance(self.min_frequency, float):
                    min_count = self.min_frequency * len(X)
                else:
                    min_count = self.min_frequency
                value_counts = value_counts[value_counts >= min_count]

            # Limit categories,
            categories = value_counts.head(self.max_categories).index.tolist()

            if self.drop_first and len(categories) > 0:
                categories = categories[1:]

            self._categories[col] = categories

            for cat in categories:
                self._feature_names_out.append(f"{col}_{cat}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by creating one-hot encoded features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with one-hot encoded features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col in self._columns_fitted:
            categories = self._categories[col]
            for cat in categories:
                feature_name = f"{col}_{cat}"
                result_data[feature_name] = (X[col] == cat).astype(int).values

        return pd.DataFrame(result_data, index=X.index)


class TargetEncoder(BaseFeatureGenerator):
    """Target encodes categorical columns.

    Replaces categories with the mean of the target variable
    with smoothing to prevent overfitting.

    Example:
        >>> encoder = TargetEncoder(columns=["city"], smoothing=10)
        >>> X_encoded = encoder.fit_transform(X, y)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        smoothing: float = 10.0,
        handle_unknown: str = "global_mean",
        min_samples: int = 1
    ) -> None:
        """Initialize the target encoder.

        Args:
            columns: Columns to encode. If None, all object/category columns.,
            smoothing: Smoothing parameter for regularization.,
            handle_unknown: How to handle unknown categories.,
            min_samples: Minimum samples for a category to have its own mean.
        """
        super().__init__()
        self.columns = columns
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        self.min_samples = min_samples

        self._encodings: dict[str, dict[Any, float]] = {}
        self._global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the encoder by computing target means.

        Args:
            X: Input DataFrame.,
            y: Target variable (required).,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if y is None:
            raise ValueError("Target variable y is required for TargetEncoder")

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._global_mean = float(y.mean())
        self._encodings = {}
        self._feature_names_out = [f"{c}_target" for c in cols]

        for col in cols:
            # Compute category means and counts,
            df_temp = pd.DataFrame({col: X[col], "target": y})
            agg = df_temp.groupby(col, observed=True)["target"].agg(["mean", "count"])

            # Apply smoothing,
            smoothed = (
                agg["count"] * agg["mean"] + self.smoothing * self._global_mean
            ) / (agg["count"] + self.smoothing)

            # Filter by min samples,
            valid = agg["count"] >= self.min_samples
            encoding = smoothed[valid].to_dict()
            self._encodings[col] = encoding

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by applying target encoding.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with target encoded features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col, feature_name in zip(self._columns_fitted, self._feature_names_out):
            encoding = self._encodings[col]
            values = X[col].map(encoding).fillna(self._global_mean).values
            result_data[feature_name] = values

        return pd.DataFrame(result_data, index=X.index)


class FrequencyEncoder(BaseFeatureGenerator):
    """Frequency encodes categorical columns.

    Replaces categories with their frequency in the training data.

    Example:
        >>> encoder = FrequencyEncoder(columns=["category"])
        >>> X_encoded = encoder.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        normalize: bool = True,
        handle_unknown: str = "zero"
    ) -> None:
        """Initialize the frequency encoder.

        Args:
            columns: Columns to encode. If None, all object/category columns.,
            normalize: If True, use proportions instead of counts.,
            handle_unknown: How to handle unknown categories.
        """
        super().__init__()
        self.columns = columns
        self.normalize = normalize
        self.handle_unknown = handle_unknown

        self._frequencies: dict[str, dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the encoder by computing frequencies.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._frequencies = {}
        self._feature_names_out = [f"{c}_freq" for c in cols]

        for col in cols:
            value_counts = X[col].value_counts(normalize=self.normalize)
            self._frequencies[col] = value_counts.to_dict()

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by applying frequency encoding.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with frequency encoded features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col, feature_name in zip(self._columns_fitted, self._feature_names_out):
            frequencies = self._frequencies[col]
            values = X[col].map(frequencies).fillna(0).values
            result_data[feature_name] = values

        return pd.DataFrame(result_data, index=X.index)


class OrdinalEncoder(BaseFeatureGenerator):
    """Ordinal encodes categorical columns.

    Assigns integer values to categories based on order or frequency.

    Example:
        >>> encoder = OrdinalEncoder(
        ...     columns=["size"]
        ...     order={"size": ["S", "M", "L", "XL"]}
        ... )
        >>> X_encoded = encoder.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        order: dict[str, list[Any]] | None = None,
        order_by: str = "frequency",
        handle_unknown: str = "missing"
    ) -> None:
        """Initialize the ordinal encoder.

        Args:
            columns: Columns to encode. If None, all object/category columns.,
            order: Explicit ordering for columns.,
            order_by: How to determine order ("frequency", "alphabetical").,
            handle_unknown: How to handle unknown categories.
        """
        super().__init__()
        self.columns = columns
        self.order = order or {}
        self.order_by = order_by
        self.handle_unknown = handle_unknown

        self._mappings: dict[str, dict[Any, int]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the encoder by learning ordinal mappings.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._mappings = {}
        self._feature_names_out = [f"{c}_ordinal" for c in cols]

        for col in cols:
            if col in self.order:
                categories = self.order[col]
            elif self.order_by == "frequency":
                categories = X[col].value_counts().index.tolist()
            else:  # alphabetical,
                categories = sorted(X[col].dropna().unique())

            self._mappings[col] = {cat: i for i, cat in enumerate(categories)}

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by applying ordinal encoding.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with ordinal encoded features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col, feature_name in zip(self._columns_fitted, self._feature_names_out):
            mapping = self._mappings[col]

            if self.handle_unknown == "missing":
                unknown_val = -1
            else:
                unknown_val = len(mapping)

            values = X[col].map(mapping).fillna(unknown_val).astype(int).values
            result_data[feature_name] = values

        return pd.DataFrame(result_data, index=X.index)
