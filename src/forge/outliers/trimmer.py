"""Trimmer transformer for removing outlier rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from typing_extensions import Self


class Trimmer(BaseEstimator, TransformerMixin):
    """Remove rows containing outliers from the dataset.

    Unlike Winsorizer and Capper which modify values, Trimmer removes
    entire rows that contain outliers. Use with caution as this reduces
    dataset size.

    Parameters
    ----------
    columns : list[str] | None, default=None
        Columns to check for outliers. If None, checks all numeric columns.
    method : str, default='iqr'
        Outlier detection method:
        - 'iqr': Use IQR method (Q1 - factor*IQR, Q3 + factor*IQR)
        - 'percentile': Use percentile method
    factor : float, default=1.5
        IQR multiplier when method='iqr'.
    lower_percentile : float, default=0.01
        Lower percentile when method='percentile'.
    upper_percentile : float, default=0.99
        Upper percentile when method='percentile'.

    Attributes:
    ----------
    bounds_ : dict[str, tuple[float, float]]
        Dictionary mapping column names to (lower, upper) bound tuples.
    feature_names_in_ : list[str]
        Names of features seen during fit.

    Examples:
    --------
    >>> from forge.outliers import Trimmer
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'value': [1, 2, 3, 4, 5, 100],
    ...     'label': ['a', 'b', 'c', 'd', 'e', 'f']
    ... })
    >>> trimmer = Trimmer(columns=['value'], method='iqr')
    >>> result = trimmer.fit_transform(df)
    >>> len(result) < len(df)  # Row with outlier removed
    True

    Notes:
    -----
    Trimmer removes rows during transform, which means the output will have
    fewer rows than the input. This can cause issues with corresponding
    target arrays in supervised learning. Consider using Winsorizer or
    IQRCapper instead if preserving all rows is important.

    See Also:
    --------
    Winsorizer : Cap outliers at percentiles (preserves rows).,
    IQRCapper : Cap outliers using IQR method (preserves rows).
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        method: str = "iqr",
        factor: float = 1.5,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99
    ) -> None:
        self.columns = columns
        self.method = method
        self.factor = factor
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Compute outlier bounds from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.Series | None, default=None
            Ignored. Present for sklearn compatibility.

        Returns:
        -------
        self
            Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.method not in ("iqr", "percentile"):
            raise ValueError("method must be 'iqr' or 'percentile'")

        cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()

        self.bounds_: dict[str, tuple[float, float]] = {}

        for col in cols:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            if self.method == "iqr":
                q1 = float(X[col].quantile(0.25))
                q3 = float(X[col].quantile(0.75))
                iqr = q3 - q1
                lower = q1 - self.factor * iqr
                upper = q3 + self.factor * iqr
            else:  # percentile,
                lower = float(X[col].quantile(self.lower_percentile))
                upper = float(X[col].quantile(self.upper_percentile))

            self.bounds_[col] = (lower, upper)

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove rows containing outliers.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns:
        -------
        pd.DataFrame
            Data with outlier rows removed.
        """
        check_is_fitted(self, ["bounds_"])

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Create mask for valid rows (no outliers)
        mask = pd.Series(True, index=X.index)

        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                col_mask = (X[col] >= lower) & (X[col] <= upper)
                mask = mask & col_mask

        return X.loc[mask].reset_index(drop=True)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Get output feature names."""
        check_is_fitted(self, ["feature_names_in_"])
        return self.feature_names_in_
