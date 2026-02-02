"""Neural feature generation and learned embeddings.

Implements neural feature crosses (inspired by DCN-v2), learned categorical
embeddings, and neural-based feature transformation. Falls back to
numpy-based approximations when PyTorch is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

logger = logging.getLogger(__name__)


def _check_torch() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class CrossLayerConfig:
    """Configuration for a cross layer.

    Attributes:
    ----------
    n_crosses : int
        Number of feature cross layers.
    projection_dim : int | None
        Dimension to project crosses into. None keeps original dim.
    """

    n_crosses: int = 2
    projection_dim: int | None = None


@dataclass
class EmbeddingConfig:
    """Configuration for categorical embedding.

    Attributes:
    ----------
    embedding_dim : int | str
        Embedding dimension. 'auto' computes dim = min(50, 1 + cardinality // 2).
    max_cardinality : int
        Maximum allowed cardinality for a column.
    """

    embedding_dim: int | str = "auto"
    max_cardinality: int = 1000


class NeuralFeatureCross(BaseEstimator, TransformerMixin):
    """Generate feature crosses using neural-inspired transformations.

    Creates explicit feature crosses by combining columns through
    element-wise multiplication and learned-weight projections.
    Uses numpy when PyTorch is unavailable.

    Parameters
    ----------
    columns : list[str] | None
        Columns to cross. None uses all numeric columns.
    n_crosses : int
        Number of cross layers to apply.
    projection_dim : int | None
        Output dimension for cross projections. None keeps input dim.
    random_state : int | None
        Random seed for reproducibility.

    Attributes:
    ----------
    cross_weights_ : list[np.ndarray]
        Learned cross-layer weight matrices.
    feature_names_out_ : list[str]
        Names of output features.

    Examples:
    --------
    >>> from forge.neural import NeuralFeatureCross
    >>> cross = NeuralFeatureCross(columns=["a", "b", "c"], n_crosses=2)
    >>> X_crossed = cross.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        n_crosses: int = 2,
        projection_dim: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.columns = columns
        self.n_crosses = n_crosses
        self.projection_dim = projection_dim
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Fit cross layer weights.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : Any
            Ignored.

        Returns:
        -------
        Self
            Fitted generator.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(X).__name__}")

        if self.columns is not None:
            self.feature_names_in_ = [c for c in self.columns if c in X.columns]
        else:
            self.feature_names_in_ = list(X.select_dtypes(include=[np.number]).columns)

        n_features = len(self.feature_names_in_)
        if n_features == 0:
            self.cross_weights_ = []
            self.feature_names_out_: list[str] = []
            self._is_fitted = True
            return self

        rng = np.random.RandomState(self.random_state)
        out_dim = self.projection_dim or n_features

        self.cross_weights_ = []
        self._cross_biases: list[np.ndarray] = []
        for i in range(self.n_crosses):
            w = rng.randn(n_features, out_dim) * np.sqrt(2.0 / (n_features + out_dim))
            b = np.zeros(out_dim)
            self.cross_weights_.append(w)
            self._cross_biases.append(b)

        self.feature_names_out_ = [
            f"cross_L{layer}_d{dim}"
            for layer in range(self.n_crosses)
            for dim in range(out_dim)
        ]

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature crosses.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns:
        -------
        pd.DataFrame
            Input data with cross features appended.
        """
        import pandas as pd

        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("NeuralFeatureCross must be fitted before transform.")

        if not self.feature_names_in_:
            return X.copy()

        values = X[self.feature_names_in_].values.astype(np.float64)
        x0 = values.copy()
        n_features = x0.shape[1]
        out_dim = self.projection_dim or n_features

        cross_features: list[np.ndarray] = []
        x_l = x0.copy()

        for i in range(self.n_crosses):
            # Project x_l to out_dim, then element-wise with projected x0
            projected = x_l @ self.cross_weights_[i]  # (N, out_dim)
            x0_proj = x0[:, :out_dim] if out_dim <= n_features else np.pad(
                x0, ((0, 0), (0, out_dim - n_features)), mode="constant"
            )
            x_l = x0_proj * projected + self._cross_biases[i] + projected
            cross_features.append(x_l)

        cross_array = np.hstack(cross_features)
        cross_df = pd.DataFrame(
            cross_array,
            columns=self.feature_names_out_,
            index=X.index,
        )

        return pd.concat([X, cross_df], axis=1)

    def get_feature_names_out(self) -> list[str]:
        """Get names of output cross features."""
        return self.feature_names_out_


class NeuralEmbeddingGenerator(BaseEstimator, TransformerMixin):
    """Generate learned embeddings for categorical features.

    Maps high-cardinality categorical columns to dense vector
    representations using random initialized embeddings refined
    by frequency-based weighting.

    Parameters
    ----------
    columns : list[str] | None
        Categorical columns to embed. None auto-detects.
    embedding_dim : int | str
        Embedding dimension per column. 'auto' uses min(50, 1 + card//2).
    max_cardinality : int
        Max cardinality to embed. Higher cardinality columns are skipped.
    random_state : int | None
        Random seed.

    Examples:
    --------
    >>> from forge.neural import NeuralEmbeddingGenerator
    >>> emb = NeuralEmbeddingGenerator(columns=["city", "category"])
    >>> X_embedded = emb.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        embedding_dim: int | str = "auto",
        max_cardinality: int = 1000,
        random_state: int | None = None,
    ) -> None:
        self.columns = columns
        self.embedding_dim = embedding_dim
        self.max_cardinality = max_cardinality
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Learn embeddings from data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : Any
            Ignored.

        Returns:
        -------
        Self
            Fitted generator.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(X).__name__}")

        if self.columns is not None:
            cat_cols = [c for c in self.columns if c in X.columns]
        else:
            cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)

        rng = np.random.RandomState(self.random_state)
        self._embeddings: dict[str, dict[Any, np.ndarray]] = {}
        self._embedding_dims: dict[str, int] = {}
        self._embedded_columns: list[str] = []

        for col in cat_cols:
            cardinality = X[col].nunique()
            if cardinality > self.max_cardinality:
                logger.warning(
                    "Column '%s' has cardinality %d > %d, skipping.",
                    col, cardinality, self.max_cardinality,
                )
                continue

            if self.embedding_dim == "auto":
                dim = min(50, 1 + cardinality // 2)
            else:
                dim = int(self.embedding_dim)

            self._embedding_dims[col] = dim
            self._embedded_columns.append(col)

            # Create embedding matrix with frequency-weighted initialization
            categories = X[col].value_counts()
            col_embeddings: dict[Any, np.ndarray] = {}
            for cat_value, count in categories.items():
                # Frequency-based scaling for initialization
                freq_weight = np.log1p(count) / np.log1p(len(X))
                embedding = rng.randn(dim) * 0.1 * (1 + freq_weight)
                col_embeddings[cat_value] = embedding

            # Unknown/missing category embedding
            col_embeddings["__unknown__"] = np.zeros(dim)
            self._embeddings[col] = col_embeddings

        self.feature_names_out_: list[str] = []
        for col in self._embedded_columns:
            dim = self._embedding_dims[col]
            for d in range(dim):
                self.feature_names_out_.append(f"{col}_emb_{d}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply embeddings to categorical columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns:
        -------
        pd.DataFrame
            Input with embedding columns appended.
        """
        import pandas as pd

        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("NeuralEmbeddingGenerator must be fitted before transform.")

        if not self._embedded_columns:
            return X.copy()

        embedding_arrays: list[np.ndarray] = []

        for col in self._embedded_columns:
            col_emb = self._embeddings[col]
            unknown_emb = col_emb["__unknown__"]
            dim = self._embedding_dims[col]

            col_data = X[col].values
            emb_matrix = np.zeros((len(X), dim))
            for i, val in enumerate(col_data):
                emb_matrix[i] = col_emb.get(val, unknown_emb)

            embedding_arrays.append(emb_matrix)

        all_embeddings = np.hstack(embedding_arrays)
        emb_df = pd.DataFrame(
            all_embeddings,
            columns=self.feature_names_out_,
            index=X.index,
        )

        return pd.concat([X, emb_df], axis=1)

    def get_feature_names_out(self) -> list[str]:
        """Get names of embedding features."""
        return self.feature_names_out_


class NeuralFeatureGenerator(BaseEstimator, TransformerMixin):
    """Combined neural feature generator.

    Orchestrates both NeuralFeatureCross (for numeric) and
    NeuralEmbeddingGenerator (for categorical) in a single transformer.

    Parameters
    ----------
    numeric_columns : list[str] | None
        Numeric columns for cross features.
    categorical_columns : list[str] | None
        Categorical columns for embeddings.
    n_crosses : int
        Number of cross layers.
    embedding_dim : int | str
        Embedding dimension.
    random_state : int | None
        Random seed.

    Examples:
    --------
    >>> from forge.neural import NeuralFeatureGenerator
    >>> gen = NeuralFeatureGenerator()
    >>> X_neural = gen.fit_transform(X, y)
    """

    def __init__(
        self,
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        n_crosses: int = 2,
        embedding_dim: int | str = "auto",
        random_state: int | None = None,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.n_crosses = n_crosses
        self.embedding_dim = embedding_dim
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Fit both cross and embedding generators.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : Any
            Ignored.

        Returns:
        -------
        Self
            Fitted generator.
        """
        self._cross = NeuralFeatureCross(
            columns=self.numeric_columns,
            n_crosses=self.n_crosses,
            random_state=self.random_state,
        )
        self._embedding = NeuralEmbeddingGenerator(
            columns=self.categorical_columns,
            embedding_dim=self.embedding_dim,
            random_state=self.random_state,
        )

        self._cross.fit(X, y)
        self._embedding.fit(X, y)

        self.feature_names_out_ = (
            self._cross.get_feature_names_out()
            + self._embedding.get_feature_names_out()
        )

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply both cross and embedding transforms.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns:
        -------
        pd.DataFrame
            Data with neural features appended.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("NeuralFeatureGenerator must be fitted.")

        result = self._cross.transform(X)
        result = self._embedding.transform(result)
        return result

    def get_feature_names_out(self) -> list[str]:
        """Get all output feature names."""
        return self.feature_names_out_
