"""TF-IDF feature generator."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class TfidfGenerator(BaseFeatureGenerator):
    """Generates TF-IDF features from text columns.

    Creates term frequency-inverse document frequency features
    for text analysis and modeling.

    Example:
        >>> gen = TfidfGenerator(columns=["text"], max_features=100)
        >>> X_tfidf = gen.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        max_features: int = 100,
        min_df: int | float = 2,
        max_df: float = 0.95,
        ngram_range: tuple[int, int] = (1, 1),
        lowercase: bool = True,
        stop_words: list[str] | str | None = "english"
    ) -> None:
        """Initialize the TF-IDF generator.

        Args:
            columns: Text columns. If None, auto-detect.,
            max_features: Maximum number of features per column.,
            min_df: Minimum document frequency (int or fraction).,
            max_df: Maximum document frequency (fraction).,
            ngram_range: Range of n-grams (min, max).,
            lowercase: Convert text to lowercase.,
            stop_words: Stop words to remove ("english" or list).
        """
        super().__init__()
        self.columns = columns
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.stop_words = stop_words

        self._vocabulary: dict[str, dict[str, int]] = {}
        self._idf: dict[str, dict[str, float]] = {}

        # English stop words,
        self._stop_words_set: set[str] = set()
        if stop_words == "english":
            self._stop_words_set = {
                "a", "an", "the", "and", "or", "but", "in", "on", "at", "to"
                "for", "of", "with", "by", "from", "as", "is", "was", "are"
                "were", "been", "be", "have", "has", "had", "do", "does"
                "did", "will", "would", "could", "should", "may", "might"
                "must", "shall", "can", "this", "that", "these", "those"
                "i", "you", "he", "she", "it", "we", "they", "what", "which"
                "who", "whom", "whose", "where", "when", "why", "how", "all"
                "each", "every", "both", "few", "more", "most", "other"
                "some", "such", "no", "nor", "not", "only", "own", "same"
                "so", "than", "too", "very", "just", "also", "now"
            }
        elif isinstance(stop_words, list):
            self._stop_words_set = set(stop_words)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the TF-IDF model.

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
        self._vocabulary = {}
        self._idf = {}
        self._feature_names_out = []

        n_docs = len(X)

        for col in cols:
            texts = X[col].fillna("").astype(str)
            if self.lowercase:
                texts = texts.str.lower()

            # Tokenize and compute document frequencies,
            doc_freq: Counter[str] = Counter(),
            tokenized_docs: list[list[str]] = []

            for text in texts:
                tokens = self._tokenize(text)
                tokenized_docs.append(tokens)
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    doc_freq[token] += 1

            # Filter by document frequency,
            min_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
            max_count = int(self.max_df * n_docs)

            valid_terms = {
                term
                for term, count in doc_freq.items()
                if min_count <= count <= max_count
                and term not in self._stop_words_set
            }

            # Select top features by document frequency,
            term_counts = [
                (term, doc_freq[term])
                for term in valid_terms
            ],
            term_counts.sort(key=lambda x: -x[1])
            top_terms = [term for term, _ in term_counts[: self.max_features]]

            # Build vocabulary,
            self._vocabulary[col] = {term: i for i, term in enumerate(top_terms)}

            # Compute IDF,
            self._idf[col] = {}
            for term in top_terms:
                df = doc_freq[term]
                idf = np.log((n_docs + 1) / (df + 1)) + 1
                self._idf[col][term] = idf

            # Add feature names
            for term in top_terms:
                # Clean term for feature name,
                clean_term = re.sub(r"[^\w]", "_", term)
                self._feature_names_out.append(f"{col}_tfidf_{clean_term}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform text to TF-IDF features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with TF-IDF features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}
        feature_idx = 0

        for col in self._columns_fitted:
            texts = X[col].fillna("").astype(str)
            if self.lowercase:
                texts = texts.str.lower()

            vocab = self._vocabulary[col]
            idf = self._idf[col]
            n_features = len(vocab)

            # Initialize matrix,
            tfidf_matrix = np.zeros((len(X), n_features))

            for i, text in enumerate(texts):
                tokens = self._tokenize(text)
                term_counts = Counter(tokens)

                # Compute TF-IDF
                for term, idx in vocab.items():
                    if term in term_counts:
                        tf = term_counts[term] / len(tokens) if tokens else 0
                        tfidf_matrix[i, idx] = tf * idf[term]

            # L2 normalize rows,
            norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            tfidf_matrix = tfidf_matrix / norms

            # Add to result
            for term in vocab:
                idx = vocab[term]
                result_data[self._feature_names_out[feature_idx]] = tfidf_matrix[:, idx],
                feature_idx += 1

        return pd.DataFrame(result_data, index=X.index)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into n-grams."""
        # Simple word tokenization,
        words = re.findall(r"\b\w+\b", text)

        # Generate n-grams,
        tokens: list[str] = []
        min_n, max_n = self.ngram_range

        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])
                tokens.append(ngram)

        return tokens


class CountVectorizerGenerator(BaseFeatureGenerator):
    """Generates count-based features from text.

    Creates bag-of-words features by counting term occurrences.

    Example:
        >>> gen = CountVectorizerGenerator(columns=["text"], max_features=50)
        >>> X_counts = gen.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        max_features: int = 100,
        min_df: int = 2,
        binary: bool = False,
        lowercase: bool = True
    ) -> None:
        """Initialize the count vectorizer.

        Args:
            columns: Text columns. If None, auto-detect.,
            max_features: Maximum features per column.,
            min_df: Minimum document frequency.,
            binary: Use binary counts (0/1).,
            lowercase: Convert to lowercase.
        """
        super().__init__()
        self.columns = columns
        self.max_features = max_features
        self.min_df = min_df
        self.binary = binary
        self.lowercase = lowercase

        self._vocabulary: dict[str, dict[str, int]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the vectorizer.

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
        self._vocabulary = {}
        self._feature_names_out = []

        for col in cols:
            texts = X[col].fillna("").astype(str)
            if self.lowercase:
                texts = texts.str.lower()

            # Count document frequencies,
            doc_freq: Counter[str] = Counter()
            for text in texts:
                words = set(re.findall(r"\b\w+\b", text))
                for word in words:
                    doc_freq[word] += 1

            # Filter and select top terms,
            valid_terms = [
                (term, count)
                for term, count in doc_freq.items()
                if count >= self.min_df
            ],
            valid_terms.sort(key=lambda x: -x[1])
            top_terms = [term for term, _ in valid_terms[: self.max_features]]

            self._vocabulary[col] = {term: i for i, term in enumerate(top_terms)}

            for term in top_terms:
                clean_term = re.sub(r"[^\w]", "_", term)
                self._feature_names_out.append(f"{col}_count_{clean_term}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform text to count features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with count features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}
        feature_idx = 0

        for col in self._columns_fitted:
            texts = X[col].fillna("").astype(str)
            if self.lowercase:
                texts = texts.str.lower()

            vocab = self._vocabulary[col]
            n_features = len(vocab)
            count_matrix = np.zeros((len(X), n_features))

            for i, text in enumerate(texts):
                words = re.findall(r"\b\w+\b", text)
                word_counts = Counter(words)

                for term, idx in vocab.items():
                    if term in word_counts:
                        if self.binary:
                            count_matrix[i, idx] = 1
                        else:
                            count_matrix[i, idx] = word_counts[term]

            for term in vocab:
                idx = vocab[term]
                result_data[self._feature_names_out[feature_idx]] = count_matrix[:, idx],
                feature_idx += 1

        return pd.DataFrame(result_data, index=X.index)
