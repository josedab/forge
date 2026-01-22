"""LLM-powered feature generator that executes suggestions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.exceptions import FeatureGenerationError, NotFittedError, ValidationError
from forge.llm.metadata import MetadataExtractor, DatasetMetadata
from forge.llm.suggestions import FeatureSuggester, FeatureSuggestion, LLMProvider

if TYPE_CHECKING:
    from typing_extensions import Self


class LLMFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generates features based on LLM suggestions.

    This transformer uses an LLM to suggest features based on dataset
    metadata, then implements and applies those features. It supports
    various transformation types and handles edge cases gracefully.

    Example:
        >>> from forge.llm import LLMFeatureGenerator
        >>> generator = LLMFeatureGenerator(provider="openai")
        >>> X_new = generator.fit_transform(X, y)
        >>> print(generator.get_feature_explanations())
    """

    SUPPORTED_TRANSFORMATIONS = {
        "interaction",
        "ratio",
        "difference",
        "log",
        "sqrt",
        "square",
        "power",
        "bin",
        "datetime_extract",
        "target_encode",
        "frequency_encode",
        "text_length",
        "word_count",
        "aggregation",
    }

    def __init__(
        self,
        provider: str | LLMProvider = "openai",
        api_key: str | None = None,
        model: str | None = None,
        max_features: int = 20,
        min_confidence: float = 0.5,
        include_originals: bool = True,
        additional_context: str | None = None,
        verbose: int = 0,
    ) -> None:
        """Initialize the LLM feature generator.

        Args:
            provider: LLM provider name or instance.
            api_key: API key for the provider.
            model: Model name override.
            max_features: Maximum number of features to generate.
            min_confidence: Minimum confidence score to include a suggestion.
            include_originals: Whether to include original columns in output.
            additional_context: Additional domain context for the LLM.
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.max_features = max_features
        self.min_confidence = min_confidence
        self.include_originals = include_originals
        self.additional_context = additional_context
        self.verbose = verbose

        self._is_fitted = False
        self._suggester: FeatureSuggester | None = None
        self._suggestions: list[FeatureSuggestion] = []
        self._implemented_features: list[dict[str, Any]] = []
        self._feature_names_out: list[str] = []
        self._target_encodings: dict[str, dict[Any, float]] = {}
        self._frequency_encodings: dict[str, dict[Any, float]] = {}
        self._input_columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator by getting LLM suggestions and learning encodings.

        Args:
            X: Input DataFrame.
            y: Optional target variable (required for target encoding).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._input_columns = list(X.columns)

        # Initialize suggester
        suggester_kwargs: dict[str, Any] = {
            "provider": self.provider,
            "max_suggestions": self.max_features,
        }
        if self.api_key:
            suggester_kwargs["api_key"] = self.api_key
        if self.model:
            suggester_kwargs["model"] = self.model

        self._suggester = FeatureSuggester(**suggester_kwargs)

        # Extract metadata and get suggestions
        extractor = MetadataExtractor()
        metadata = extractor.extract(X, y)

        if self.verbose >= 1:
            print(f"Extracted metadata for {metadata.n_columns} columns")

        self._suggestions = self._suggester.suggest(
            metadata,
            additional_context=self.additional_context,
        )

        # Filter by confidence
        self._suggestions = [
            s for s in self._suggestions
            if s.confidence >= self.min_confidence
        ]

        if self.verbose >= 1:
            print(f"Got {len(self._suggestions)} feature suggestions")

        # Learn encodings from training data
        self._learn_encodings(X, y)

        # Determine which features can be implemented
        self._implemented_features = []
        for suggestion in self._suggestions:
            if self._can_implement(suggestion, X):
                self._implemented_features.append({
                    "suggestion": suggestion,
                    "name": suggestion.name,
                })

        # Build output feature names
        self._feature_names_out = []
        if self.include_originals:
            self._feature_names_out.extend(self._input_columns)
        self._feature_names_out.extend(
            f["name"] for f in self._implemented_features
        )

        if self.verbose >= 1:
            print(f"Will generate {len(self._implemented_features)} new features")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by generating LLM-suggested features.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with generated features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X.copy() if self.include_originals else pd.DataFrame(index=X.index)

        for feature_info in self._implemented_features:
            suggestion = feature_info["suggestion"]
            try:
                feature_values = self._generate_feature(X, suggestion)
                result[suggestion.name] = feature_values
            except Exception as e:
                if self.verbose >= 2:
                    print(f"Warning: Failed to generate {suggestion.name}: {e}")
                # Fill with NaN if generation fails
                result[suggestion.name] = np.nan

        return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Input DataFrame.
            y: Optional target variable.

        Returns:
            DataFrame with generated features.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get names of output features.

        Args:
            input_features: Ignored, for sklearn compatibility.

        Returns:
            List of feature names.
        """
        self._check_is_fitted()
        return self._feature_names_out.copy()

    def get_suggestions(self) -> list[FeatureSuggestion]:
        """Get the LLM suggestions used for feature generation.

        Returns:
            List of feature suggestions.
        """
        self._check_is_fitted()
        return self._suggestions.copy()

    def get_implemented_features(self) -> list[dict[str, Any]]:
        """Get information about implemented features.

        Returns:
            List of feature information dictionaries.
        """
        self._check_is_fitted()
        return self._implemented_features.copy()

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")
        if len(X) == 0:
            raise ValidationError("Input DataFrame is empty")

    def _check_is_fitted(self) -> None:
        """Check if the generator has been fitted."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

    def _learn_encodings(self, X: pd.DataFrame, y: pd.Series | None) -> None:
        """Learn target and frequency encodings from training data."""
        for suggestion in self._suggestions:
            if suggestion.transformation == "target_encode" and y is not None:
                for col in suggestion.source_columns:
                    if col in X.columns:
                        self._target_encodings[col] = (
                            X.groupby(col)[y.name if y.name else "target"]
                            .apply(lambda x: y.loc[x.index].mean())
                            .to_dict()
                        )

            elif suggestion.transformation == "frequency_encode":
                for col in suggestion.source_columns:
                    if col in X.columns:
                        freq = X[col].value_counts(normalize=True)
                        self._frequency_encodings[col] = freq.to_dict()

    def _can_implement(self, suggestion: FeatureSuggestion, X: pd.DataFrame) -> bool:
        """Check if a suggestion can be implemented."""
        # Check source columns exist
        for col in suggestion.source_columns:
            if col not in X.columns:
                return False

        # Check transformation is supported
        if suggestion.transformation not in self.SUPPORTED_TRANSFORMATIONS:
            # Try to map to a supported transformation
            transformation_map = {
                "multiply": "interaction",
                "divide": "ratio",
                "subtract": "difference",
                "logarithm": "log",
                "square_root": "sqrt",
                "binning": "bin",
            }
            if suggestion.transformation in transformation_map:
                suggestion.transformation = transformation_map[suggestion.transformation]
            else:
                return False

        return True

    def _generate_feature(
        self, X: pd.DataFrame, suggestion: FeatureSuggestion
    ) -> pd.Series:
        """Generate a single feature based on a suggestion."""
        source_cols = suggestion.source_columns
        transformation = suggestion.transformation

        if transformation == "interaction":
            if len(source_cols) >= 2:
                result = X[source_cols[0]] * X[source_cols[1]]
            else:
                result = X[source_cols[0]] ** 2
            return result.astype(float)

        elif transformation == "ratio":
            if len(source_cols) >= 2:
                denominator = X[source_cols[1]].replace(0, np.nan)
                result = X[source_cols[0]] / denominator
            else:
                raise FeatureGenerationError(f"Ratio requires 2 columns, got {len(source_cols)}")
            return result.astype(float)

        elif transformation == "difference":
            if len(source_cols) >= 2:
                result = X[source_cols[0]] - X[source_cols[1]]
            else:
                raise FeatureGenerationError(f"Difference requires 2 columns")
            return result.astype(float)

        elif transformation == "log":
            col = source_cols[0]
            values = pd.to_numeric(X[col], errors="coerce")
            # Use log1p to handle zeros
            result = np.log1p(values.clip(lower=0))
            return result

        elif transformation == "sqrt":
            col = source_cols[0]
            values = pd.to_numeric(X[col], errors="coerce")
            result = np.sqrt(values.clip(lower=0))
            return result

        elif transformation == "square":
            col = source_cols[0]
            values = pd.to_numeric(X[col], errors="coerce")
            return values ** 2

        elif transformation == "power":
            col = source_cols[0]
            values = pd.to_numeric(X[col], errors="coerce")
            # Default to cube
            return values ** 3

        elif transformation == "bin":
            col = source_cols[0]
            values = pd.to_numeric(X[col], errors="coerce")
            result = pd.qcut(values, q=5, labels=False, duplicates="drop")
            return result.astype(float)

        elif transformation == "datetime_extract":
            col = source_cols[0]
            dt_values = pd.to_datetime(X[col], errors="coerce")

            # Determine which component from feature name
            name_lower = suggestion.name.lower()
            if "year" in name_lower:
                return dt_values.dt.year.astype(float)
            elif "month" in name_lower:
                return dt_values.dt.month.astype(float)
            elif "day" in name_lower and "week" not in name_lower:
                return dt_values.dt.day.astype(float)
            elif "hour" in name_lower:
                return dt_values.dt.hour.astype(float)
            elif "dayofweek" in name_lower or "weekday" in name_lower:
                return dt_values.dt.dayofweek.astype(float)
            elif "week" in name_lower:
                return dt_values.dt.isocalendar().week.astype(float)
            else:
                return dt_values.dt.dayofyear.astype(float)

        elif transformation == "target_encode":
            col = source_cols[0]
            if col in self._target_encodings:
                encoding = self._target_encodings[col]
                global_mean = np.mean(list(encoding.values()))
                return X[col].map(encoding).fillna(global_mean)
            else:
                raise FeatureGenerationError(f"No target encoding for {col}")

        elif transformation == "frequency_encode":
            col = source_cols[0]
            if col in self._frequency_encodings:
                encoding = self._frequency_encodings[col]
                return X[col].map(encoding).fillna(0.0)
            else:
                # Compute frequency on the fly
                freq = X[col].value_counts(normalize=True)
                return X[col].map(freq).fillna(0.0)

        elif transformation == "text_length":
            col = source_cols[0]
            return X[col].astype(str).str.len().astype(float)

        elif transformation == "word_count":
            col = source_cols[0]
            return X[col].astype(str).str.split().str.len().astype(float)

        elif transformation == "aggregation":
            # For aggregations, we need a group column and a value column
            if len(source_cols) >= 2:
                group_col = source_cols[0]
                value_col = source_cols[1]
                agg = X.groupby(group_col)[value_col].transform("mean")
                return agg.astype(float)
            else:
                raise FeatureGenerationError("Aggregation requires group and value columns")

        else:
            raise FeatureGenerationError(
                f"Unknown transformation: {transformation}"
            )
