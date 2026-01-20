"""Outlier handling transformers for feature engineering.

This module provides sklearn-compatible transformers for detecting and
handling outliers in numeric data.

Classes
-------
Winsorizer
    Cap outliers at specified percentiles.
IQRCapper
    Cap outliers using Interquartile Range method.
ArbitraryCapper
    Cap outliers at user-specified values.
Trimmer
    Remove rows containing outliers.

Examples
--------
>>> from forge.outliers import Winsorizer
>>> import pandas as pd
>>> df = pd.DataFrame({'value': [1, 2, 3, 100, 5]})
>>> winsorizer = Winsorizer(upper_percentile=0.95)
>>> df_capped = winsorizer.fit_transform(df)
"""

from forge.outliers.winsorizer import Winsorizer
from forge.outliers.capper import IQRCapper, ArbitraryCapper
from forge.outliers.trimmer import Trimmer

__all__ = [
    "Winsorizer",
    "IQRCapper",
    "ArbitraryCapper",
    "Trimmer",
]
