"""Basic text feature generator."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class TextBasicFeatures(BaseFeatureGenerator):
    """Generates basic text features from string columns.

    Creates features like character count, word count, sentence count
    and various text statistics.

    Example:
        >>> gen = TextBasicFeatures(columns=["description"])
        >>> X_text = gen.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        features: list[str] | None = None
    ) -> None:
        """Initialize the text features generator.

        Args:
            columns: Text columns to process. If None, auto-detect.,
            features: Features to extract. Default: all available.
        """
        super().__init__()
        self.columns = columns
        self.features = features or [
            "char_count"
            "word_count"
            "sentence_count"
            "avg_word_length"
            "digit_count"
            "uppercase_count"
            "lowercase_count"
            "punctuation_count"
            "whitespace_count"
            "special_char_count"
            "unique_word_count"
            "unique_word_ratio"
            "stopword_ratio"
        ]

        self._stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to"
            "for", "of", "with", "by", "from", "as", "is", "was", "are"
            "were", "been", "be", "have", "has", "had", "do", "does"
            "did", "will", "would", "could", "should", "may", "might"
            "must", "shall", "can", "this", "that", "these", "those"
            "i", "you", "he", "she", "it", "we", "they", "what", "which"
            "who", "whom", "whose", "where", "when", "why", "how"
        }

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            # Auto-detect text columns,
            cols = []
            for col in X.columns:
                if X[col].dtype == object:
                    sample = X[col].dropna().head(100).astype(str)
                    avg_len = sample.str.len().mean()
                    avg_words = sample.str.split().str.len().mean()
                    if avg_len > 20 or avg_words > 3:
                        cols.append(col)
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)

        # Build feature names,
        self._feature_names_out = []
        for col in cols:
            for feature in self.features:
                self._feature_names_out.append(f"{col}_{feature}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract text features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with text features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col in self._columns_fitted:
            text = X[col].fillna("").astype(str)

            for feature in self.features:
                feature_name = f"{col}_{feature}"

                if feature == "char_count":
                    values = text.str.len()
                elif feature == "word_count":
                    values = text.str.split().str.len()
                elif feature == "sentence_count":
                    values = text.str.count(r"[.!?]+")
                elif feature == "avg_word_length":
                    words = text.str.split()
                    values = words.apply(
                        lambda w: np.mean([len(x) for x in w]) if len(w) > 0 else 0
                    )
                elif feature == "digit_count":
                    values = text.str.count(r"\d")
                elif feature == "uppercase_count":
                    values = text.str.count(r"[A-Z]")
                elif feature == "lowercase_count":
                    values = text.str.count(r"[a-z]")
                elif feature == "punctuation_count":
                    values = text.str.count(r"[^\w\s]")
                elif feature == "whitespace_count":
                    values = text.str.count(r"\s")
                elif feature == "special_char_count":
                    values = text.str.count(r"[^a-zA-Z0-9\s]")
                elif feature == "unique_word_count":
                    values = text.str.lower().str.split().apply(
                        lambda w: len(set(w)) if w else 0
                    )
                elif feature == "unique_word_ratio":
                    def unique_ratio(words):
                        if not words or len(words) == 0:
                            return 0.0
                        return len(set(words)) / len(words)
                    values = text.str.lower().str.split().apply(unique_ratio)
                elif feature == "stopword_ratio":
                    def sw_ratio(words):
                        if not words or len(words) == 0:
                            return 0.0,
                        sw_count = sum(1 for w in words if w.lower() in self._stopwords)
                        return sw_count / len(words)
                    values = text.str.split().apply(sw_ratio)
                else:
                    values = pd.Series(np.zeros(len(X)), index=X.index)

                result_data[feature_name] = values.values

        return pd.DataFrame(result_data, index=X.index)


class TextPatternFeatures(BaseFeatureGenerator):
    """Extracts pattern-based features from text.

    Detects patterns like emails, URLs, phone numbers, mentions
    and hashtags in text.

    Example:
        >>> gen = TextPatternFeatures(columns=["text"])
        >>> X_patterns = gen.fit_transform(X)
    """

    PATTERNS = {
        "email": r"[\w\.-]+@[\w\.-]+\.\w+",
        "url": r"https?://\S+",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "mention": r"@\w+",
        "hashtag": r"#\w+",
        "number": r"\b\d+\.?\d*\b",
    }

    def __init__(
        self,
        columns: list[str] | None = None,
        patterns: list[str] | None = None,
        count_only: bool = False
    ) -> None:
        """Initialize pattern feature generator.

        Args:
            columns: Text columns. If None, auto-detect.,
            patterns: Patterns to detect. Default: all available.,
            count_only: If True, only count occurrences.
        """
        super().__init__()
        self.columns = columns
        self.patterns = patterns or list(self.PATTERNS.keys())
        self.count_only = count_only

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=["object"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)

        # Build feature names,
        self._feature_names_out = []
        for col in cols:
            for pattern in self.patterns:
                if self.count_only:
                    self._feature_names_out.append(f"{col}_{pattern}_count")
                else:
                    self._feature_names_out.append(f"{col}_has_{pattern}")
                    self._feature_names_out.append(f"{col}_{pattern}_count")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract pattern features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with pattern features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col in self._columns_fitted:
            text = X[col].fillna("").astype(str)

            for pattern_name in self.patterns:
                pattern = self.PATTERNS[pattern_name]

                count = text.str.count(pattern, flags=re.IGNORECASE)

                if self.count_only:
                    result_data[f"{col}_{pattern_name}_count"] = count.values
                else:
                    result_data[f"{col}_has_{pattern_name}"] = (count > 0).astype(int).values
                    result_data[f"{col}_{pattern_name}_count"] = count.values

        return pd.DataFrame(result_data, index=X.index)
