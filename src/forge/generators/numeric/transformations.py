"""Numeric transformation generators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class LogTransformer(BaseFeatureGenerator):
    """Applies log transformation to numeric columns.

    Useful for reducing skewness in right-skewed distributions.

    Example:
        >>> transformer = LogTransformer(columns=["price", "quantity"])
        >>> X_log = transformer.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        log_type: str = "log1p",
        handle_negative: str = "abs"
    ) -> None:
        """Initialize the log transformer.

        Args:
            columns: Columns to transform. If None, all positive numeric.,
            log_type: Type of log ("log", "log1p", "log10").,
            handle_negative: How to handle negatives ("abs", "shift", "skip").
        """
        super().__init__()
        self.columns = columns
        self.log_type = log_type
        self.handle_negative = handle_negative

        if log_type not in ("log", "log1p", "log10"):
            raise ValueError(f"log_type must be 'log', 'log1p', or 'log10', got '{log_type}'")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the transformer.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()
            # Filter to columns that can have log applied,
            cols = [c for c in cols if (X[c].dropna() > 0).any()]
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._feature_names_out = [f"{c}_{self.log_type}" for c in cols]

        # Store shifts for negative handling,
        self._shifts: dict[str, float] = {}
        if self.handle_negative == "shift":
            for col in cols:
                min_val = X[col].min()
                if min_val <= 0:
                    self._shifts[col] = abs(min_val) + 1
                else:
                    self._shifts[col] = 0

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with log-transformed features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col, feature_name in zip(self._columns_fitted, self._feature_names_out):
            values = X[col].values.astype(float)

            # Handle negative values
            if self.handle_negative == "abs":
                values = np.abs(values)
            elif self.handle_negative == "shift":
                values = values + self._shifts.get(col, 0)

            # Apply log
            if self.log_type == "log":
                result = np.where(values > 0, np.log(values), 0)
            elif self.log_type == "log1p":
                result = np.log1p(np.maximum(values, 0))
            else:  # log10,
                result = np.where(values > 0, np.log10(values), 0)

            result_data[feature_name] = result

        return pd.DataFrame(result_data, index=X.index)


class PowerTransformer(BaseFeatureGenerator):
    """Applies power transformations to numeric columns.

    Includes sqrt, square, and Box-Cox/Yeo-Johnson transforms.

    Example:
        >>> transformer = PowerTransformer(transforms=["sqrt", "yeo-johnson"])
        >>> X_power = transformer.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        transforms: list[str] | None = None
    ) -> None:
        """Initialize the power transformer.

        Args:
            columns: Columns to transform. If None, all numeric.,
            transforms: Transforms to apply. Default: ["sqrt", "square"].
        """
        super().__init__()
        self.columns = columns
        self.transforms = transforms or ["sqrt", "square"]

        valid = {"sqrt", "square", "cbrt", "yeo-johnson", "box-cox"}
        invalid = set(self.transforms) - valid
        if invalid:
            raise ValueError(f"Invalid transforms: {invalid}. Valid: {valid}")

        self._lambda_params: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the transformer.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)

        # Build feature names and fit parametric transforms,
        self._feature_names_out = []
        for col in cols:
            for transform in self.transforms:
                self._feature_names_out.append(f"{col}_{transform}")

                # Fit Yeo-Johnson or Box-Cox
                if transform in ("yeo-johnson", "box-cox"):
                    values = X[col].dropna().values
                    if len(values) > 0 and np.std(values) > 0:
                        if transform == "yeo-johnson":
                            _, lam = stats.yeojohnson(values)
                        else:
                            # Box-Cox requires positive values
                            if (values > 0).all():
                                _, lam = stats.boxcox(values)
                            else:
                                lam = 1.0
                        self._lambda_params.setdefault(col, {})[transform] = lam
                    else:
                        self._lambda_params.setdefault(col, {})[transform] = 1.0

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply power transformations.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with power-transformed features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col in self._columns_fitted:
            values = X[col].values.astype(float)

            for transform in self.transforms:
                feature_name = f"{col}_{transform}"

                if transform == "sqrt":
                    result = np.sqrt(np.maximum(values, 0))
                elif transform == "square":
                    result = values ** 2
                elif transform == "cbrt":
                    result = np.cbrt(values)
                elif transform == "yeo-johnson":
                    lam = self._lambda_params.get(col, {}).get(transform, 1.0)
                    result = stats.yeojohnson(values, lmbda=lam)
                elif transform == "box-cox":
                    lam = self._lambda_params.get(col, {}).get(transform, 1.0)
                    if (values > 0).all():
                        result = stats.boxcox(values, lmbda=lam)
                    else:
                        result = values  # Fallback

                result_data[feature_name] = result

        return pd.DataFrame(result_data, index=X.index)


class BinningTransformer(BaseFeatureGenerator):
    """Bins numeric columns into categories.

    Supports equal-width, equal-frequency (quantile), and custom binning.

    Example:
        >>> transformer = BinningTransformer(n_bins=5, strategy="quantile")
        >>> X_binned = transformer.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        n_bins: int = 5,
        strategy: str = "quantile",
        encode: str = "ordinal"
    ) -> None:
        """Initialize the binning transformer.

        Args:
            columns: Columns to bin. If None, all numeric.,
            n_bins: Number of bins.,
            strategy: Binning strategy ("uniform", "quantile", "kmeans").,
            encode: Encoding ("ordinal", "onehot").
        """
        super().__init__()
        self.columns = columns
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode

        if strategy not in ("uniform", "quantile"):
            raise ValueError(f"strategy must be 'uniform' or 'quantile', got '{strategy}'")

        self._bin_edges: dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the transformer by computing bin edges.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)

        # Compute bin edges
        for col in cols:
            values = X[col].dropna().values

            if self.strategy == "uniform":
                edges = np.linspace(values.min(), values.max(), self.n_bins + 1)
            else:  # quantile,
                edges = np.percentile(values, np.linspace(0, 100, self.n_bins + 1))
                edges = np.unique(edges)

            self._bin_edges[col] = edges

        # Build feature names
        if self.encode == "ordinal":
            self._feature_names_out = [f"{c}_binned" for c in cols]
        else:
            self._feature_names_out = []
            for col in cols:
                n_actual_bins = len(self._bin_edges[col]) - 1
                for i in range(n_actual_bins):
                    self._feature_names_out.append(f"{col}_bin_{i}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply binning transformation.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with binned features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, Any] = {}

        for col in self._columns_fitted:
            values = X[col].values
            edges = self._bin_edges[col]

            # Digitize values,
            binned = np.digitize(values, edges[1:-1])

            if self.encode == "ordinal":
                result_data[f"{col}_binned"] = binned
            else:
                n_actual_bins = len(edges) - 1
                for i in range(n_actual_bins):
                    result_data[f"{col}_bin_{i}"] = (binned == i).astype(int)

        return pd.DataFrame(result_data, index=X.index)


class NumericTransformer(BaseFeatureGenerator):
    """Combined numeric transformer that applies multiple transformations.

    A convenience class that combines log, power, and binning transforms.

    Example:
        >>> transformer = NumericTransformer(
        ...     log=True
        ...     sqrt=True
        ...     binning=True
        ...     n_bins=5
        ... )
        >>> X_transformed = transformer.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        log: bool = True,
        sqrt: bool = True,
        square: bool = False,
        binning: bool = False,
        n_bins: int = 5
    ) -> None:
        """Initialize the numeric transformer.

        Args:
            columns: Columns to transform. If None, all numeric.,
            log: Whether to apply log1p transform.,
            sqrt: Whether to apply sqrt transform.,
            square: Whether to apply square transform.,
            binning: Whether to apply binning.,
            n_bins: Number of bins if binning is True.
        """
        super().__init__()
        self.columns = columns
        self.log = log
        self.sqrt = sqrt
        self.square = square
        self.binning = binning
        self.n_bins = n_bins

        self._transformers: list[BaseFeatureGenerator] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit all transformers.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._transformers = []
        self._feature_names_out = []

        # Add transformers
        if self.log:
            log_t = LogTransformer(columns=cols)
            log_t.fit(X, y)
            self._transformers.append(log_t)
            self._feature_names_out.extend(log_t.get_feature_names_out())

        if self.sqrt:
            power_t = PowerTransformer(columns=cols, transforms=["sqrt"])
            power_t.fit(X, y)
            self._transformers.append(power_t)
            self._feature_names_out.extend(power_t.get_feature_names_out())

        if self.square:
            power_t = PowerTransformer(columns=cols, transforms=["square"])
            power_t.fit(X, y)
            self._transformers.append(power_t)
            self._feature_names_out.extend(power_t.get_feature_names_out())

        if self.binning:
            bin_t = BinningTransformer(columns=cols, n_bins=self.n_bins)
            bin_t.fit(X, y)
            self._transformers.append(bin_t)
            self._feature_names_out.extend(bin_t.get_feature_names_out())

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with all transformed features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_frames = [t.transform(X) for t in self._transformers]
        return pd.concat(result_frames, axis=1)
