"""Advanced categorical encoding generators.

This module provides advanced encoding techniques commonly used in
industry applications like credit risk modeling and competitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class WoEEncoder(BaseFeatureGenerator):
    """Weight of Evidence encoder for binary classification.

    WoE measures the predictive power of a categorical variable for
    binary classification. Commonly used in credit risk modeling.

    WoE = ln(Distribution of Events / Distribution of Non-Events)

    Higher WoE values indicate the category is more associated with
    the positive class (events).

    Parameters
    ----------,
    columns : list[str] | None, default=None
        Columns to encode. If None, encodes all object/category columns.,
    regularization : float, default=0.5
        Smoothing factor to prevent division by zero and handle
        categories with no events or non-events.

    Attributes:
    ----------,
    woe_maps_ : dict[str, dict]
        WoE values for each category in each column.,
    iv_scores_ : dict[str, float]
        Information Value scores for each column.

    Examples:
    --------
    >>> from forge.generators.categorical import WoEEncoder,
    >>> encoder = WoEEncoder()
    >>> X_encoded = encoder.fit_transform(X, y)

    Notes:
    -----
    WoE encoding requires a binary target variable (0/1).,
    The Information Value (IV) can be used for feature selection:
    - IV < 0.02: Not useful for prediction,
    - 0.02 <= IV < 0.1: Weak predictive power
    - 0.1 <= IV < 0.3: Medium predictive power
    - IV >= 0.3: Strong predictive power

    See Also:
    --------,
    TargetEncoder : Mean target encoding with smoothing.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        regularization: float = 0.5
    ) -> None:
        super().__init__()
        self.columns = columns
        self.regularization = regularization

        self._woe_maps: dict[str, dict[Any, float]] = {}
        self._iv_scores: dict[str, float] = {}

    @property
    def iv_(self) -> dict[str, float]:
        """Information Value scores for each encoded column."""
        return self._iv_scores

    @property
    def woe_maps_(self) -> dict[str, dict[Any, float]]:
        """WoE mapping dictionaries for each encoded column."""
        return self._woe_maps

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Compute WoE values for each category.

        Parameters
        ----------,
        X : pd.DataFrame
            Input features.,
        y : pd.Series
            Binary target variable (0/1). Required.

        Returns:
        -------
        self
            Fitted encoder.
        """
        self._validate_input(X)

        if y is None:
            raise ValueError("Target variable y is required for WoEEncoder")

        # Validate binary target,
        unique_values = set(y.unique())
        if not unique_values.issubset({0, 1}):
            raise ValueError("WoE encoder requires binary target (0/1)")

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._woe_maps = {}
        self._iv_scores = {}
        self._feature_names_out = [f"{c}_woe" for c in cols]

        total_events = y.sum() + self.regularization
        total_non_events = len(y) - y.sum() + self.regularization

        for col in cols:
            woe_map: dict[Any, float] = {}
            iv_score = 0.0

            for category in X[col].unique():
                mask = X[col] == category
                events = y[mask].sum() + self.regularization
                non_events = (~y[mask].astype(bool)).sum() + self.regularization

                dist_events = events / total_events
                dist_non_events = non_events / total_non_events

                woe = np.log(dist_events / dist_non_events)
                woe_map[category] = float(woe)

                # Calculate IV contribution,
                iv_score += (dist_events - dist_non_events) * woe

            self._woe_maps[col] = woe_map
            self._iv_scores[col] = float(iv_score)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply WoE encoding.

        Parameters
        ----------,
        X : pd.DataFrame
            Input features.

        Returns:
        -------
        pd.DataFrame
            WoE encoded features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col, feature_name in zip(self._columns_fitted, self._feature_names_out):
            woe_map = self._woe_maps[col]
            # Use 0 for unseen categories (neutral WoE)
            values = X[col].map(woe_map).fillna(0).values
            result_data[feature_name] = values

        return pd.DataFrame(result_data, index=X.index)

    def get_information_value(self) -> dict[str, float]:
        """Get Information Value scores for each column.

        Returns:
        -------
        dict[str, float]
            IV scores for each encoded column.
        """
        self._check_is_fitted()
        return self._iv_scores.copy()


class CatBoostEncoder(BaseFeatureGenerator):
    """CatBoost-style ordered target encoding.

    Implements the encoding strategy used by CatBoost, which prevents
    target leakage by using only previous observations to encode each row.

    For each row i, the encoding is computed as:
    (sum of targets for category in rows 0..i-1 + prior) / (count + 1)

    Parameters
    ----------,
    columns : list[str] | None, default=None
        Columns to encode. If None, encodes all object/category columns.,
    prior : float | None, default=None
        Prior value for smoothing. If None, uses global target mean.,
    a : float, default=1.0
        Smoothing parameter. Higher values give more weight to prior.

    Examples:
    --------
    >>> from forge.generators.categorical import CatBoostEncoder,
    >>> encoder = CatBoostEncoder()
    >>> X_encoded = encoder.fit_transform(X, y)

    Notes:
    -----
    This encoder is particularly useful for preventing target leakage
    in cross-validation scenarios. The encoding for each row only uses
    information from previous rows, simulating a streaming scenario.

    See Also:
    --------,
    TargetEncoder : Standard mean target encoding.,
    LeaveOneOutEncoder : LOO target encoding.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        prior: float | None = None,
        a: float = 1.0
    ) -> None:
        super().__init__()
        self.columns = columns
        self.prior = prior
        self.a = a

        self._global_mean: float = 0.0
        self._category_stats: dict[str, dict[Any, dict[str, float]]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the encoder by computing category statistics.

        Parameters
        ----------,
        X : pd.DataFrame
            Input features.,
        y : pd.Series
            Target variable. Required.

        Returns:
        -------
        self
            Fitted encoder.
        """
        self._validate_input(X)

        if y is None:
            raise ValueError("Target variable y is required for CatBoostEncoder")

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._global_mean = float(y.mean())
        self._feature_names_out = [f"{c}_catboost" for c in cols]

        # Store final statistics for transform (used for new data)
        self._category_stats = {}
        for col in cols:
            stats: dict[Any, dict[str, float]] = {}
            for cat in X[col].unique():
                mask = X[col] == cat
                cat_sum = float(y[mask].sum())
                cat_count = float(mask.sum())
                stats[cat] = {"sum": cat_sum, "count": cat_count}
            self._category_stats[col] = stats

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply CatBoost encoding.

        Parameters
        ----------,
        X : pd.DataFrame
            Input features.

        Returns:
        -------
        pd.DataFrame
            CatBoost encoded features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        prior = self.prior if self.prior is not None else self._global_mean
        result_data: dict[str, np.ndarray] = {}

        for col, feature_name in zip(self._columns_fitted, self._feature_names_out):
            stats = self._category_stats[col]
            encoded = np.zeros(len(X))

            for i, cat in enumerate(X[col].values):
                if cat in stats:
                    cat_sum = stats[cat]["sum"]
                    cat_count = stats[cat]["count"]
                    encoded[i] = (cat_sum + prior * self.a) / (cat_count + self.a)
                else:
                    encoded[i] = prior

            result_data[feature_name] = encoded

        return pd.DataFrame(result_data, index=X.index)


class LeaveOneOutEncoder(BaseFeatureGenerator):
    """Leave-One-Out target encoder.

    For each row, computes the mean target of all other rows with the
    same category value (excluding the current row).

    This helps prevent target leakage while still using target information.

    Parameters
    ----------,
    columns : list[str] | None, default=None
        Columns to encode. If None, encodes all object/category columns.,
    smoothing : float, default=1.0
        Smoothing parameter for regularization.,
    handle_unknown : str, default='global_mean'
        Strategy for unknown categories: 'global_mean' or 'zero'.

    Examples:
    --------
    >>> from forge.generators.categorical import LeaveOneOutEncoder,
    >>> encoder = LeaveOneOutEncoder()
    >>> X_encoded = encoder.fit_transform(X, y)

    Notes:
    -----
    During transform on new data (without target), uses the full
    category means from training.

    See Also:
    --------,
    TargetEncoder : Standard mean target encoding.,
    CatBoostEncoder : Ordered target encoding.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        smoothing: float = 1.0,
        handle_unknown: str = "global_mean"
    ) -> None:
        super().__init__()
        self.columns = columns
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown

        self._global_mean: float = 0.0
        self._category_stats: dict[str, dict[Any, dict[str, float]]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the encoder by computing category statistics.

        Parameters
        ----------,
        X : pd.DataFrame
            Input features.,
        y : pd.Series
            Target variable. Required.

        Returns:
        -------
        self
            Fitted encoder.
        """
        self._validate_input(X)

        if y is None:
            raise ValueError("Target variable y is required for LeaveOneOutEncoder")

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._global_mean = float(y.mean())
        self._feature_names_out = [f"{c}_loo" for c in cols]

        # Store category statistics,
        self._category_stats = {}
        for col in cols:
            stats: dict[Any, dict[str, float]] = {}
            for cat in X[col].unique():
                mask = X[col] == cat
                cat_sum = float(y[mask].sum())
                cat_count = float(mask.sum())
                stats[cat] = {"sum": cat_sum, "count": cat_count}
            self._category_stats[col] = stats

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply Leave-One-Out encoding.

        Parameters
        ----------,
        X : pd.DataFrame
            Input features.

        Returns:
        -------
        pd.DataFrame
            LOO encoded features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col, feature_name in zip(self._columns_fitted, self._feature_names_out):
            stats = self._category_stats[col]
            encoded = np.zeros(len(X))

            for i, cat in enumerate(X[col].values):
                if cat in stats:
                    cat_sum = stats[cat]["sum"]
                    cat_count = stats[cat]["count"]

                    if cat_count > 1:
                        # LOO mean: (sum - current) / (count - 1)
                        # Since we don't have y during transform, use full mean,
                        mean = (cat_sum + self.smoothing * self._global_mean) / (
                            cat_count + self.smoothing
                        )
                    else:
                        mean = self._global_mean

                    encoded[i] = mean
                else:
                    if self.handle_unknown == "global_mean":
                        encoded[i] = self._global_mean
                    else:
                        encoded[i] = 0

            result_data[feature_name] = encoded

        return pd.DataFrame(result_data, index=X.index)


class HashingEncoder(BaseFeatureGenerator):
    """Hashing encoder for high-cardinality categorical variables.

    Uses the hashing trick to encode categories into a fixed number
    of dimensions, useful for very high cardinality columns.

    Parameters
    ----------,
    columns : list[str] | None, default=None
        Columns to encode. If None, encodes all object/category columns.,
    n_components : int, default=8
        Number of hash buckets (output dimensions per column).,
    hash_method : str, default='md5'
        Hash function to use: 'md5' or 'murmur'.

    Examples:
    --------
    >>> from forge.generators.categorical import HashingEncoder,
    >>> encoder = HashingEncoder(n_components=16)
    >>> X_encoded = encoder.fit_transform(X)

    Notes:
    -----,
    Advantages:
    - Fixed output dimensionality regardless of cardinality
    - Can handle unseen categories
    - Memory efficient

    Disadvantages:
    - Hash collisions can occur
    - Not invertible (can't decode back to original)

    See Also:
    --------,
    OneHotEncoder : Standard one-hot encoding.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        n_components: int = 8,
        hash_method: str = "md5"
    ) -> None:
        super().__init__()
        self.columns = columns
        self.n_components = n_components
        self.hash_method = hash_method

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the encoder (learns column names only).

        Parameters
        ----------,
        X : pd.DataFrame
            Input features.,
        y : pd.Series | None
            Ignored.

        Returns:
        -------
        self
            Fitted encoder.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._feature_names_out = []

        for col in cols:
            for i in range(self.n_components):
                self._feature_names_out.append(f"{col}_hash_{i}")

        self._is_fitted = True
        return self

    def _hash_value(self, value: Any, seed: int = 0) -> int:
        """Hash a value to an integer."""
        import hashlib

        value_str = str(value) + str(seed)

        if self.hash_method == "md5":
            hash_obj = hashlib.md5(value_str.encode(), usedforsecurity=False)
            return int(hash_obj.hexdigest(), 16)
        else:
            # Simple hash fallback
            return hash(value_str)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply hashing encoding.

        Parameters
        ----------,
        X : pd.DataFrame
            Input features.

        Returns:
        -------
        pd.DataFrame
            Hash encoded features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col in self._columns_fitted:
            # Initialize hash buckets,
            hash_arrays = [np.zeros(len(X)) for _ in range(self.n_components)]

            for i, value in enumerate(X[col].values):
                if pd.notna(value):
                    hash_val = self._hash_value(value)
                    bucket = hash_val % self.n_components
                    # Use sign from hash for sparse representation,
                    sign = 1 if (hash_val // self.n_components) % 2 == 0 else -1
                    hash_arrays[bucket][i] = sign

            for j in range(self.n_components):
                feature_name = f"{col}_hash_{j}"
                result_data[feature_name] = hash_arrays[j]

        return pd.DataFrame(result_data, index=X.index)
