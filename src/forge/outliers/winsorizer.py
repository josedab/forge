"""Winsorizer transformer for capping outliers at percentiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from typing_extensions import Self


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap outliers at specified percentiles.

    Winsorization replaces extreme values with values at specified percentiles,
    reducing the impact of outliers while preserving all data points.

    Parameters
    ----------
    columns : list[str] | None, default=None
        Columns to winsorize. If None, applies to all numeric columns.
    lower_percentile : float, default=0.01
        Lower percentile for capping (e.g., 0.01 = 1st percentile).
        Values below this percentile are set to the percentile value.
    upper_percentile : float, default=0.99
        Upper percentile for capping (e.g., 0.99 = 99th percentile).
        Values above this percentile are set to the percentile value.

    Attributes
    ----------
    bounds_ : dict[str, tuple[float, float]]
        Dictionary mapping column names to (lower, upper) bound tuples.
    feature_names_in_ : list[str]
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from forge.outliers import Winsorizer
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'value': [1, 2, 3, 4, 5, 100],
    ...     'other': [10, 20, 30, 40, 50, 60]
    ... })
    >>> winsorizer = Winsorizer(columns=['value'], upper_percentile=0.9)
    >>> result = winsorizer.fit_transform(df)
    >>> result['value'].max() < 100  # Outlier capped
    True

    See Also
    --------
    IQRCapper : Cap outliers using IQR method.,
    ArbitraryCapper : Cap outliers at specific values.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99
    ) -> None:
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Compute percentile bounds from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.Series | None, default=None
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        self
            Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.lower_percentile < 0 or self.lower_percentile > 1:
            raise ValueError("lower_percentile must be between 0 and 1")
        if self.upper_percentile < 0 or self.upper_percentile > 1:
            raise ValueError("upper_percentile must be between 0 and 1")
        if self.lower_percentile >= self.upper_percentile:
            raise ValueError("lower_percentile must be less than upper_percentile")

        cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()

        self.bounds_: dict[str, tuple[float, float]] = {}
        for col in cols:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            lower = float(X[col].quantile(self.lower_percentile))
            upper = float(X[col].quantile(self.upper_percentile))
            self.bounds_[col] = (lower, upper)

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization to data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Transformed data with outliers capped.
        """
        check_is_fitted(self, ["bounds_"])

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        X_out = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].clip(lower=lower, upper=upper)

        return X_out

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Get output feature names.

        Parameters
        ----------
        input_features : list[str] | None, default=None
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        list[str]
            Output feature names (unchanged from input).
        """
        check_is_fitted(self, ["feature_names_in_"])
        return self.feature_names_in_
