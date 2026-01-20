"""Capper transformers for outlier handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from typing_extensions import Self


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap outliers using the Interquartile Range (IQR) method.

    Values are capped at Q1 - factor * IQR and Q3 + factor * IQR,
    where IQR = Q3 - Q1. This is a common method for outlier detection
    based on the distribution of the data.

    Parameters
    ----------
    columns : list[str] | None, default=None
        Columns to cap. If None, applies to all numeric columns.
    factor : float, default=1.5
        IQR multiplier for determining bounds. Common values:
        - 1.5: Standard outlier detection (Tukey's method)
        - 3.0: Extreme outlier detection

    Attributes
    ----------
    bounds_ : dict[str, tuple[float, float]]
        Dictionary mapping column names to (lower, upper) bound tuples.
    iqr_stats_ : dict[str, dict]
        Dictionary with Q1, Q3, and IQR for each column.
    feature_names_in_ : list[str]
        Names of features seen during fit.

    Examples
    --------
    >>> from forge.outliers import IQRCapper
    >>> import pandas as pd
    >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})
    >>> capper = IQRCapper(factor=1.5)
    >>> result = capper.fit_transform(df)
    >>> result['value'].max() < 100  # Extreme outlier capped
    True

    See Also
    --------
    Winsorizer : Cap outliers at percentiles.,
    ArbitraryCapper : Cap outliers at specific values.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        factor: float = 1.5
    ) -> None:
        self.columns = columns
        self.factor = factor

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Compute IQR bounds from training data.

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

        if self.factor <= 0:
            raise ValueError("factor must be positive")

        cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()

        self.bounds_: dict[str, tuple[float, float]] = {}
        self.iqr_stats_: dict[str, dict[str, float]] = {}

        for col in cols:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            q1 = float(X[col].quantile(0.25))
            q3 = float(X[col].quantile(0.75))
            iqr = q3 - q1

            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr

            self.bounds_[col] = (lower, upper)
            self.iqr_stats_[col] = {"q1": q1, "q3": q3, "iqr": iqr}

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply IQR capping to data.

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
        """Get output feature names."""
        check_is_fitted(self, ["feature_names_in_"])
        return self.feature_names_in_


class ArbitraryCapper(BaseEstimator, TransformerMixin):
    """Cap outliers at user-specified arbitrary values.

    Allows setting custom lower and upper bounds for each column,
    useful when domain knowledge dictates specific valid ranges.

    Parameters
    ----------
    capping_dict : dict[str, dict[str, float]]
        Dictionary mapping column names to bounds. Each value should be
        a dict with optional 'lower' and/or 'upper' keys.
        Example: {'age': {'lower': 0, 'upper': 120}}

    Attributes
    ----------
    capping_dict_ : dict[str, dict[str, float]]
        Validated capping dictionary.
    feature_names_in_ : list[str]
        Names of features seen during fit.

    Examples
    --------
    >>> from forge.outliers import ArbitraryCapper
    >>> import pandas as pd
    >>> df = pd.DataFrame({'age': [-5, 25, 150], 'score': [0, 50, 200]})
    >>> capper = ArbitraryCapper({
    ...     'age': {'lower': 0, 'upper': 120},
    ...     'score': {'lower': 0, 'upper': 100}
    ... })
    >>> result = capper.fit_transform(df)
    >>> result['age'].min() >= 0  # Negative age capped
    True
    >>> result['score'].max() <= 100  # Score over 100 capped
    True

    See Also
    --------
    Winsorizer : Cap outliers at percentiles.,
    IQRCapper : Cap outliers using IQR method.
    """

    def __init__(
        self,
        capping_dict: dict[str, dict[str, float]],
    ) -> None:
        self.capping_dict = capping_dict

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Validate capping dictionary against data.

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

        if not isinstance(self.capping_dict, dict):
            raise TypeError("capping_dict must be a dictionary")

        # Validate capping_dict structure
        for col, bounds in self.capping_dict.items():
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            if not isinstance(bounds, dict):
                raise TypeError(f"Bounds for '{col}' must be a dictionary")
            if "lower" not in bounds and "upper" not in bounds:
                raise ValueError(
                    f"Bounds for '{col}' must contain 'lower' and/or 'upper'"
                )
            if "lower" in bounds and "upper" in bounds:
                if bounds["lower"] >= bounds["upper"]:
                    raise ValueError(
                        f"Lower bound must be less than upper bound for '{col}'"
                    )

        self.capping_dict_ = self.capping_dict.copy()
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply arbitrary capping to data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Transformed data with outliers capped.
        """
        check_is_fitted(self, ["capping_dict_"])

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        X_out = X.copy()
        for col, bounds in self.capping_dict_.items():
            if col in X_out.columns:
                lower = bounds.get("lower")
                upper = bounds.get("upper")
                X_out[col] = X_out[col].clip(lower=lower, upper=upper)

        return X_out

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Get output feature names."""
        check_is_fitted(self, ["feature_names_in_"])
        return self.feature_names_in_
