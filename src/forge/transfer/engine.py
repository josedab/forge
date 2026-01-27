"""Feature transfer engine for cross-dataset feature reuse.

Learns feature fingerprints from source datasets and matches/transfers
transformations to target datasets based on column similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

logger = logging.getLogger(__name__)


@dataclass
class ColumnFingerprint:
    """Statistical fingerprint of a single column.

    Attributes:
    ----------
    name : str
        Column name.
    dtype : str
        Data type category: 'numeric', 'categorical', 'datetime', 'text'.
    stats : dict[str, float]
        Statistical summary (mean, std, min, max, null_rate, cardinality).
    distribution : np.ndarray | None
        Histogram counts (10-bin) for distribution matching.
    """

    name: str
    dtype: str
    stats: dict[str, float] = field(default_factory=dict)
    distribution: np.ndarray | None = None

    def similarity(self, other: ColumnFingerprint) -> float:
        """Compute similarity score against another fingerprint.

        Parameters
        ----------
        other : ColumnFingerprint
            Fingerprint to compare against.

        Returns:
        -------
        float
            Similarity score between 0 and 1.
        """
        if self.dtype != other.dtype:
            return 0.0

        score = 0.0
        weights = 0.0

        # Statistical similarity
        shared_keys = set(self.stats.keys()) & set(other.stats.keys())
        for key in shared_keys:
            v1 = self.stats[key]
            v2 = other.stats[key]
            denom = max(abs(v1), abs(v2), 1e-10)
            key_sim = 1.0 - min(abs(v1 - v2) / denom, 1.0)
            score += key_sim
            weights += 1.0

        # Distribution similarity (histogram intersection)
        if self.distribution is not None and other.distribution is not None:
            d1 = self.distribution / (self.distribution.sum() + 1e-10)
            d2 = other.distribution / (other.distribution.sum() + 1e-10)
            hist_sim = float(np.minimum(d1, d2).sum())
            score += hist_sim * 3.0  # Weight distribution more heavily
            weights += 3.0

        return score / weights if weights > 0 else 0.0


@dataclass
class DatasetFingerprint:
    """Fingerprint of an entire dataset.

    Attributes:
    ----------
    name : str
        Dataset identifier.
    n_rows : int
        Number of rows.
    n_columns : int
        Number of columns.
    columns : dict[str, ColumnFingerprint]
        Per-column fingerprints.
    """

    name: str
    n_rows: int
    n_columns: int
    columns: dict[str, ColumnFingerprint] = field(default_factory=dict)


@dataclass
class TransferResult:
    """Result of a feature transfer operation.

    Attributes:
    ----------
    source_column : str
        Source dataset column name.
    target_column : str
        Matched target dataset column name.
    similarity : float
        Match similarity score.
    transform_name : str
        Name of the transfer transform applied.
    success : bool
        Whether the transfer was successful.
    """

    source_column: str
    target_column: str
    similarity: float
    transform_name: str
    success: bool


class FeatureTransferEngine(BaseEstimator, TransformerMixin):
    """Transfer learned feature transformations across datasets.

    Fingerprints a source dataset, learns which transformations are
    beneficial, and transfers them to a target dataset by matching
    columns based on statistical similarity.

    Parameters
    ----------
    similarity_threshold : float
        Minimum similarity score to consider a column match (0-1).
    transformations : list[str] | None
        Transformations to attempt. None uses defaults:
        ['log', 'sqrt', 'square', 'bin', 'standard_scale'].
    max_matches : int
        Maximum number of column matches per source column.

    Attributes:
    ----------
    source_fingerprint_ : DatasetFingerprint
        Fingerprint of the source dataset.
    learned_transforms_ : dict[str, list[str]]
        Transforms learned per source column.
    column_mappings_ : dict[str, str]
        Source-to-target column mappings.

    Examples:
    --------
    >>> from forge.transfer import FeatureTransferEngine
    >>>
    >>> engine = FeatureTransferEngine(similarity_threshold=0.7)
    >>> engine.fit(source_df)
    >>> result = engine.transform(target_df)
    """

    DEFAULT_TRANSFORMS = ("log", "sqrt", "square", "standard_scale", "bin")

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        transformations: list[str] | None = None,
        max_matches: int = 1,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.transformations = transformations
        self.max_matches = max_matches

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Learn feature fingerprints and transformations from source data.

        Parameters
        ----------
        X : pd.DataFrame
            Source dataset.
        y : Any
            Ignored.

        Returns:
        -------
        Self
            Fitted engine.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(X).__name__}")

        self.source_fingerprint_ = self._compute_fingerprint(X, "source")

        transforms = list(self.transformations or self.DEFAULT_TRANSFORMS)
        self.learned_transforms_: dict[str, list[str]] = {}
        self._source_numeric_stats: dict[str, dict[str, float]] = {}

        for col in X.select_dtypes(include=[np.number]).columns:
            values = X[col].dropna().values.astype(float)
            if len(values) == 0:
                continue

            self._source_numeric_stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

            applicable: list[str] = []
            for t in transforms:
                if self._is_transform_applicable(values, t):
                    applicable.append(t)

            if applicable:
                self.learned_transforms_[col] = applicable

        self.column_mappings_: dict[str, str] = {}
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transfer learned transformations to target dataset.

        Matches target columns to source columns by fingerprint
        similarity, then applies applicable transformations.

        Parameters
        ----------
        X : pd.DataFrame
            Target dataset.

        Returns:
        -------
        pd.DataFrame
            Target data with transferred features appended.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("FeatureTransferEngine must be fitted first.")

        target_fp = self._compute_fingerprint(X, "target")
        result = X.copy()
        self.column_mappings_ = {}
        self.transfer_results_: list[TransferResult] = []

        for src_col, src_fp in self.source_fingerprint_.columns.items():
            if src_col not in self.learned_transforms_:
                continue

            best_match = None
            best_sim = 0.0

            for tgt_col, tgt_fp in target_fp.columns.items():
                sim = src_fp.similarity(tgt_fp)
                if sim > best_sim and sim >= self.similarity_threshold:
                    best_sim = sim
                    best_match = tgt_col

            if best_match is None:
                continue

            self.column_mappings_[src_col] = best_match

            for t_name in self.learned_transforms_[src_col]:
                try:
                    new_col = self._apply_transform(
                        result[best_match].values, t_name
                    )
                    col_name = f"{best_match}_{t_name}_transfer"
                    result[col_name] = new_col
                    self.transfer_results_.append(
                        TransferResult(
                            source_column=src_col,
                            target_column=best_match,
                            similarity=best_sim,
                            transform_name=t_name,
                            success=True,
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to transfer %s from %s to %s: %s",
                        t_name, src_col, best_match, e,
                    )
                    self.transfer_results_.append(
                        TransferResult(
                            source_column=src_col,
                            target_column=best_match,
                            similarity=best_sim,
                            transform_name=t_name,
                            success=False,
                        )
                    )

        return result

    def get_feature_names_out(self) -> list[str]:
        """Get names of transferred features."""
        return [
            r.target_column + f"_{r.transform_name}_transfer"
            for r in self.transfer_results_
            if r.success
        ] if hasattr(self, "transfer_results_") else []

    def match_columns(self, target: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
        """Find matching columns between source and target.

        Parameters
        ----------
        target : pd.DataFrame
            Target dataset.

        Returns:
        -------
        dict
            Source column -> list of (target_column, similarity) matches.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("Must be fitted first.")

        target_fp = self._compute_fingerprint(target, "target")
        matches: dict[str, list[tuple[str, float]]] = {}

        for src_col, src_fp in self.source_fingerprint_.columns.items():
            col_matches: list[tuple[str, float]] = []
            for tgt_col, tgt_fp in target_fp.columns.items():
                sim = src_fp.similarity(tgt_fp)
                if sim >= self.similarity_threshold:
                    col_matches.append((tgt_col, sim))

            col_matches.sort(key=lambda x: -x[1])
            if col_matches:
                matches[src_col] = col_matches[: self.max_matches]

        return matches

    @staticmethod
    def _compute_fingerprint(X: pd.DataFrame, name: str) -> DatasetFingerprint:
        """Compute dataset fingerprint."""
        columns: dict[str, ColumnFingerprint] = {}

        for col in X.columns:
            dtype = "numeric"
            if X[col].dtype == "object" or hasattr(X[col], "cat"):
                dtype = "categorical"
            elif np.issubdtype(X[col].dtype, np.datetime64):
                dtype = "datetime"

            stats: dict[str, float] = {
                "null_rate": float(X[col].isna().mean()),
                "cardinality": float(X[col].nunique()),
            }

            distribution = None
            if dtype == "numeric":
                values = X[col].dropna().values.astype(float)
                if len(values) > 0:
                    stats["mean"] = float(np.mean(values))
                    stats["std"] = float(np.std(values))
                    stats["min"] = float(np.min(values))
                    stats["max"] = float(np.max(values))
                    stats["skew"] = float(
                        np.mean(((values - np.mean(values)) / (np.std(values) + 1e-10)) ** 3)
                    )
                    distribution = np.histogram(values, bins=10)[0].astype(float)

            columns[col] = ColumnFingerprint(
                name=col, dtype=dtype, stats=stats, distribution=distribution,
            )

        return DatasetFingerprint(
            name=name, n_rows=len(X), n_columns=len(X.columns), columns=columns,
        )

    @staticmethod
    def _is_transform_applicable(values: np.ndarray, transform: str) -> bool:
        """Check if a transform is applicable to the data."""
        if transform in ("log",):
            return bool(np.all(values > 0))
        if transform in ("sqrt",):
            return bool(np.all(values >= 0))
        return True

    @staticmethod
    def _apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
        """Apply a single transform."""
        values = np.asarray(values, dtype=float)
        nan_mask = np.isnan(values)

        if transform == "log":
            result = np.where(nan_mask, np.nan, np.log1p(np.abs(values)))
        elif transform == "sqrt":
            result = np.where(nan_mask, np.nan, np.sqrt(np.abs(values)))
        elif transform == "square":
            result = np.where(nan_mask, np.nan, values ** 2)
        elif transform == "standard_scale":
            mean = np.nanmean(values)
            std = np.nanstd(values)
            result = np.where(nan_mask, np.nan, (values - mean) / (std + 1e-10))
        elif transform == "bin":
            valid = values[~nan_mask]
            if len(valid) > 0:
                result = np.full_like(values, np.nan)
                result[~nan_mask] = np.digitize(
                    valid, np.percentile(valid, [25, 50, 75])
                ).astype(float)
            else:
                result = np.full_like(values, np.nan)
        else:
            raise ValueError(f"Unknown transform: {transform}")

        return result
