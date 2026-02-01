"""Multi-modal feature extraction and fusion.

Extracts features from images (via statistical descriptors or pre-trained
model embeddings), audio/signals (spectral features), and fuses them
with tabular data. Falls back to numpy-based extraction when deep
learning libraries are not available.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Data modality types."""

    TABULAR = "tabular"
    IMAGE = "image"
    SIGNAL = "signal"


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from image data columns.

    Handles image data stored as numpy arrays (flattened or 2D/3D)
    in DataFrame columns. Extracts statistical and structural features.

    Parameters
    ----------
    columns : list[str] | None
        Columns containing image data (numpy arrays). None auto-detects.
    features : list[str] | None
        Features to extract. None uses all available:
        ['mean', 'std', 'min', 'max', 'entropy', 'edge_density',
         'spatial_mean', 'histogram'].
    histogram_bins : int
        Number of bins for histogram features.
    output_prefix : str
        Prefix for output feature names.

    Examples:
    --------
    >>> from forge.multimodal import ImageFeatureExtractor
    >>> extractor = ImageFeatureExtractor(columns=["image_data"])
    >>> X_with_features = extractor.fit_transform(X)
    """

    DEFAULT_FEATURES = (
        "mean", "std", "min", "max", "entropy",
        "edge_density", "contrast", "histogram",
    )

    def __init__(
        self,
        columns: list[str] | None = None,
        features: list[str] | None = None,
        histogram_bins: int = 8,
        output_prefix: str = "img",
    ) -> None:
        self.columns = columns
        self.features = features
        self.histogram_bins = histogram_bins
        self.output_prefix = output_prefix

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Fit the extractor (learn column selection).

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : Any
            Ignored.

        Returns:
        -------
        Self
            Fitted extractor.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(X).__name__}")

        if self.columns is not None:
            self._columns = [c for c in self.columns if c in X.columns]
        else:
            # Auto-detect object columns that contain arrays
            self._columns = []
            for col in X.select_dtypes(include=["object"]).columns:
                sample = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else None
                if isinstance(sample, np.ndarray):
                    self._columns.append(col)

        self._features_to_extract = list(self.features or self.DEFAULT_FEATURES)
        self.feature_names_out_: list[str] = self._build_feature_names()
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract image features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data with image columns.

        Returns:
        -------
        pd.DataFrame
            Data with image features appended.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("ImageFeatureExtractor must be fitted.")

        if not self._columns:
            return X.copy()

        result = X.copy()
        for col in self._columns:
            features = self._extract_from_column(X[col])
            for feat_name, values in features.items():
                result[feat_name] = values

        return result

    def _extract_from_column(
        self, series: pd.Series
    ) -> dict[str, np.ndarray]:
        """Extract features from a single image column."""
        n = len(series)
        features: dict[str, list[float]] = {name: [] for name in self.feature_names_out_
                                              if name.startswith(f"{self.output_prefix}_{series.name}_")}

        for idx in range(n):
            val = series.iloc[idx]
            if not isinstance(val, np.ndarray):
                for key in features:
                    features[key].append(np.nan)
                continue

            img = val.astype(float).flatten()

            for feat in self._features_to_extract:
                key = f"{self.output_prefix}_{series.name}_{feat}"
                if feat == "mean":
                    features.setdefault(key, []).append(float(np.mean(img)))
                elif feat == "std":
                    features.setdefault(key, []).append(float(np.std(img)))
                elif feat == "min":
                    features.setdefault(key, []).append(float(np.min(img)))
                elif feat == "max":
                    features.setdefault(key, []).append(float(np.max(img)))
                elif feat == "entropy":
                    hist = np.histogram(img, bins=256)[0].astype(float)
                    hist = hist / (hist.sum() + 1e-10)
                    ent = -float(np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
                    features.setdefault(key, []).append(ent)
                elif feat == "edge_density":
                    # Approximate edge detection via gradient magnitude
                    grad = np.abs(np.diff(img))
                    edge_density = float(np.mean(grad > np.std(img)))
                    features.setdefault(key, []).append(edge_density)
                elif feat == "contrast":
                    contrast = float(np.max(img) - np.min(img))
                    features.setdefault(key, []).append(contrast)
                elif feat == "histogram":
                    hist = np.histogram(img, bins=self.histogram_bins)[0].astype(float)
                    hist = hist / (hist.sum() + 1e-10)
                    for b in range(self.histogram_bins):
                        hkey = f"{self.output_prefix}_{series.name}_hist_{b}"
                        features.setdefault(hkey, []).append(float(hist[b]))

        return {k: np.array(v) for k, v in features.items()}

    def _build_feature_names(self) -> list[str]:
        """Build output feature names."""
        names: list[str] = []
        for col in self._columns:
            for feat in self._features_to_extract:
                if feat == "histogram":
                    for b in range(self.histogram_bins):
                        names.append(f"{self.output_prefix}_{col}_hist_{b}")
                else:
                    names.append(f"{self.output_prefix}_{col}_{feat}")
        return names

    def get_feature_names_out(self) -> list[str]:
        """Get output feature names."""
        return self.feature_names_out_


class SignalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from 1D signal/time-series data.

    Handles signal data stored as numpy arrays in DataFrame columns.
    Extracts statistical, spectral, and temporal features.

    Parameters
    ----------
    columns : list[str] | None
        Columns containing signal arrays.
    features : list[str] | None
        Features to extract. Defaults include:
        ['mean', 'std', 'rms', 'zero_crossings', 'peak_freq',
         'spectral_entropy', 'spectral_centroid'].
    output_prefix : str
        Prefix for output names.

    Examples:
    --------
    >>> from forge.multimodal import SignalFeatureExtractor
    >>> extractor = SignalFeatureExtractor(columns=["audio_signal"])
    >>> X_features = extractor.fit_transform(X)
    """

    DEFAULT_FEATURES = (
        "mean", "std", "rms", "max_abs", "zero_crossings",
        "peak_freq", "spectral_entropy", "spectral_centroid",
    )

    def __init__(
        self,
        columns: list[str] | None = None,
        features: list[str] | None = None,
        output_prefix: str = "sig",
    ) -> None:
        self.columns = columns
        self.features = features
        self.output_prefix = output_prefix

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Fit the extractor.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : Any
            Ignored.

        Returns:
        -------
        Self
            Fitted extractor.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(X).__name__}")

        if self.columns is not None:
            self._columns = [c for c in self.columns if c in X.columns]
        else:
            self._columns = []
            for col in X.select_dtypes(include=["object"]).columns:
                sample = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else None
                if isinstance(sample, np.ndarray) and sample.ndim == 1:
                    self._columns.append(col)

        self._features_to_extract = list(self.features or self.DEFAULT_FEATURES)
        self.feature_names_out_ = [
            f"{self.output_prefix}_{col}_{feat}"
            for col in self._columns
            for feat in self._features_to_extract
        ]

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract signal features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns:
        -------
        pd.DataFrame
            Data with signal features appended.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("SignalFeatureExtractor must be fitted.")

        if not self._columns:
            return X.copy()

        result = X.copy()
        for col in self._columns:
            for feat in self._features_to_extract:
                col_name = f"{self.output_prefix}_{col}_{feat}"
                result[col_name] = X[col].apply(
                    lambda sig: self._extract_feature(sig, feat)
                )

        return result

    @staticmethod
    def _extract_feature(signal: Any, feature: str) -> float:
        """Extract a single feature from a signal."""
        if not isinstance(signal, np.ndarray):
            return np.nan

        sig = signal.astype(float).flatten()
        if len(sig) == 0:
            return np.nan

        if feature == "mean":
            return float(np.mean(sig))
        elif feature == "std":
            return float(np.std(sig))
        elif feature == "rms":
            return float(np.sqrt(np.mean(sig ** 2)))
        elif feature == "max_abs":
            return float(np.max(np.abs(sig)))
        elif feature == "zero_crossings":
            return float(np.sum(np.diff(np.sign(sig)) != 0))
        elif feature == "peak_freq":
            fft = np.abs(np.fft.rfft(sig))
            return float(np.argmax(fft[1:]) + 1)  # Skip DC
        elif feature == "spectral_entropy":
            fft = np.abs(np.fft.rfft(sig)) ** 2
            psd = fft / (fft.sum() + 1e-10)
            return -float(np.sum(psd[psd > 0] * np.log2(psd[psd > 0])))
        elif feature == "spectral_centroid":
            fft = np.abs(np.fft.rfft(sig))
            freqs = np.arange(len(fft))
            return float(np.sum(freqs * fft) / (np.sum(fft) + 1e-10))
        return np.nan

    def get_feature_names_out(self) -> list[str]:
        """Get output feature names."""
        return self.feature_names_out_


class MultiModalFusion(BaseEstimator, TransformerMixin):
    """Fuse features from multiple modalities.

    Orchestrates feature extraction from tabular, image, and signal
    modalities, then optionally applies dimensionality reduction.

    Parameters
    ----------
    image_columns : list[str] | None
        Columns containing image arrays.
    signal_columns : list[str] | None
        Columns containing signal arrays.
    tabular_columns : list[str] | None
        Tabular columns to keep. None uses all numeric.
    reduce_dim : int | None
        Apply PCA to reduce fused features to this dimension. None keeps all.
    random_state : int | None
        Random seed.

    Examples:
    --------
    >>> from forge.multimodal import MultiModalFusion
    >>> fusion = MultiModalFusion(
    ...     image_columns=["thumbnail"],
    ...     signal_columns=["audio"],
    ... )
    >>> X_fused = fusion.fit_transform(X)
    """

    def __init__(
        self,
        image_columns: list[str] | None = None,
        signal_columns: list[str] | None = None,
        tabular_columns: list[str] | None = None,
        reduce_dim: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.image_columns = image_columns
        self.signal_columns = signal_columns
        self.tabular_columns = tabular_columns
        self.reduce_dim = reduce_dim
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Fit all modality extractors.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : Any
            Ignored.

        Returns:
        -------
        Self
            Fitted fusion.
        """
        self._image_extractor = ImageFeatureExtractor(
            columns=self.image_columns
        )
        self._signal_extractor = SignalFeatureExtractor(
            columns=self.signal_columns
        )

        self._image_extractor.fit(X, y)
        self._signal_extractor.fit(X, y)

        if self.tabular_columns:
            self._tabular_cols = [c for c in self.tabular_columns if c in X.columns]
        else:
            self._tabular_cols = list(X.select_dtypes(include=[np.number]).columns)

        # Fit PCA if reduction requested
        self._pca = None
        if self.reduce_dim is not None:
            full_result = self._extract_all(X)
            numeric_cols = full_result.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > self.reduce_dim:
                from sklearn.decomposition import PCA
                self._pca = PCA(
                    n_components=self.reduce_dim,
                    random_state=self.random_state,
                )
                self._pca.fit(full_result[numeric_cols].fillna(0))
                self._pca_columns = list(numeric_cols)

        self.feature_names_out_ = self._compute_feature_names(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract and fuse multi-modal features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns:
        -------
        pd.DataFrame
            Fused feature set.
        """
        import pandas as pd

        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("MultiModalFusion must be fitted.")

        result = self._extract_all(X)

        if self._pca is not None:
            numeric_result = result[self._pca_columns].fillna(0)
            reduced = self._pca.transform(numeric_result)
            pca_df = pd.DataFrame(
                reduced,
                columns=[f"fused_pc_{i}" for i in range(reduced.shape[1])],
                index=result.index,
            )
            # Keep non-numeric columns + PCA
            non_numeric = result.select_dtypes(exclude=[np.number])
            result = pd.concat([non_numeric, pca_df], axis=1)

        return result

    def _extract_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features from all modalities."""
        import pandas as pd

        result = X[self._tabular_cols].copy() if self._tabular_cols else pd.DataFrame(index=X.index)
        img_result = self._image_extractor.transform(X)
        sig_result = self._signal_extractor.transform(X)

        # Add image features
        img_new = [c for c in img_result.columns if c not in X.columns]
        for c in img_new:
            result[c] = img_result[c]

        # Add signal features
        sig_new = [c for c in sig_result.columns if c not in X.columns]
        for c in sig_new:
            result[c] = sig_result[c]

        return result

    def _compute_feature_names(self, X: pd.DataFrame) -> list[str]:
        """Compute output feature names."""
        if self._pca is not None:
            return [f"fused_pc_{i}" for i in range(self.reduce_dim or 0)]

        names = list(self._tabular_cols)
        names.extend(self._image_extractor.get_feature_names_out())
        names.extend(self._signal_extractor.get_feature_names_out())
        return names

    def get_feature_names_out(self) -> list[str]:
        """Get output feature names."""
        return self.feature_names_out_
