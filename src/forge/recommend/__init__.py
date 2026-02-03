"""Intelligent feature recommendation engine.

Uses dataset fingerprinting, historical experiment tracking, and
heuristics to suggest the best feature engineering strategies for
a given dataset.

Example:
    >>> from forge.recommend import FeatureRecommender
    >>> recommender = FeatureRecommender()
    >>> recommendations = recommender.recommend(X, y)
    >>> for rec in recommendations:
    ...     print(f"{rec.name}: {rec.description} (score={rec.confidence:.2f})")
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Type of ML task."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class TransformCategory(str, Enum):
    """Category of feature transformation."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    INTERACTION = "interaction"
    SELECTION = "selection"
    TEXT = "text"


@dataclass
class Recommendation:
    """A single feature engineering recommendation.

    Attributes:
        name: Short name for the recommendation.
        description: Human-readable description of what to do.
        category: Category of transformation.
        confidence: Confidence score from 0 to 1.
        columns: Columns this recommendation applies to.
        parameters: Suggested parameters for the transformation.
        rationale: Why this recommendation was made.
    """

    name: str
    description: str
    category: TransformCategory
    confidence: float
    columns: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "confidence": self.confidence,
            "columns": self.columns,
            "parameters": self.parameters,
            "rationale": self.rationale,
        }


@dataclass
class DatasetProfile:
    """Statistical profile of a dataset for recommendation matching."""

    n_rows: int
    n_columns: int
    n_numeric: int
    n_categorical: int
    n_temporal: int
    n_text: int
    numeric_skewness: dict[str, float] = field(default_factory=dict)
    cardinalities: dict[str, int] = field(default_factory=dict)
    null_rates: dict[str, float] = field(default_factory=dict)
    correlations: dict[str, float] = field(default_factory=dict)
    task_type: TaskType = TaskType.UNKNOWN
    target_cardinality: int | None = None
    fingerprint_hash: str = ""

    def compute_hash(self) -> str:
        """Compute a content-based hash for matching."""
        content = json.dumps({
            "n_rows_bucket": self.n_rows // 1000,
            "n_numeric": self.n_numeric,
            "n_categorical": self.n_categorical,
            "task_type": self.task_type.value,
        }, sort_keys=True)
        self.fingerprint_hash = hashlib.md5(content.encode()).hexdigest()[:12]  # noqa: S324
        return self.fingerprint_hash

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_numeric": self.n_numeric,
            "n_categorical": self.n_categorical,
            "n_temporal": self.n_temporal,
            "n_text": self.n_text,
            "task_type": self.task_type.value,
            "target_cardinality": self.target_cardinality,
            "fingerprint_hash": self.fingerprint_hash,
        }


@dataclass
class ExperimentRecord:
    """Record of a past feature engineering experiment."""

    profile_hash: str
    transformations: list[str]
    score_before: float
    score_after: float
    metric: str = "accuracy"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def improvement(self) -> float:
        """Score improvement from the transformations."""
        return self.score_after - self.score_before

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "profile_hash": self.profile_hash,
            "transformations": self.transformations,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "metric": self.metric,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        """Create from dictionary."""
        return cls(**data)


class KnowledgeBase:
    """Local knowledge base for storing experiment results."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialize the knowledge base.

        Args:
            path: Path to persist records. None for in-memory only.
        """
        self.path = Path(path) if path else None
        self._records: list[ExperimentRecord] = []
        if self.path and self.path.exists():
            self._load()

    def record(self, experiment: ExperimentRecord) -> None:
        """Record a new experiment result."""
        self._records.append(experiment)
        if self.path:
            self._save()

    def query(self, profile_hash: str, top_k: int = 5) -> list[ExperimentRecord]:
        """Retrieve experiments matching a profile hash, sorted by improvement."""
        matching = [r for r in self._records if r.profile_hash == profile_hash]
        matching.sort(key=lambda r: r.improvement, reverse=True)
        return matching[:top_k]

    def query_all(self, top_k: int = 10) -> list[ExperimentRecord]:
        """Get top experiments across all profiles."""
        sorted_records = sorted(self._records, key=lambda r: r.improvement, reverse=True)
        return sorted_records[:top_k]

    @property
    def size(self) -> int:
        """Number of experiment records."""
        return len(self._records)

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._records]
        self.path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self._records = [ExperimentRecord.from_dict(r) for r in data]
        except (json.JSONDecodeError, KeyError):
            self._records = []


class DatasetProfiler:
    """Profiles a dataset for recommendation matching."""

    def profile(self, X: pd.DataFrame, y: pd.Series | None = None) -> DatasetProfile:
        """Create a statistical profile of a dataset.

        Args:
            X: Feature DataFrame.
            y: Optional target series.

        Returns:
            DatasetProfile with statistical summary.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
        text_cols = self._detect_text_columns(X, categorical_cols)
        categorical_cols = [c for c in categorical_cols if c not in text_cols]

        skewness: dict[str, float] = {}
        for col in numeric_cols:
            try:
                sk = float(X[col].dropna().skew())
                if np.isfinite(sk):
                    skewness[col] = sk
            except (ValueError, TypeError):
                pass

        cardinalities: dict[str, int] = {}
        for col in categorical_cols:
            cardinalities[col] = int(X[col].nunique())

        null_rates: dict[str, float] = {}
        for col in X.columns:
            rate = float(X[col].isnull().mean())
            if rate > 0:
                null_rates[col] = rate

        correlations: dict[str, float] = {}
        if len(numeric_cols) >= 2:
            corr_matrix = X[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1:]:
                    val = corr_matrix.loc[col1, col2]
                    if np.isfinite(val) and abs(val) > 0.5:
                        correlations[f"{col1}__x__{col2}"] = float(val)

        task_type = TaskType.UNKNOWN
        target_cardinality = None
        if y is not None:
            target_cardinality = int(y.nunique())
            if target_cardinality <= 20:
                task_type = TaskType.CLASSIFICATION
            else:
                task_type = TaskType.REGRESSION

        profile = DatasetProfile(
            n_rows=len(X),
            n_columns=len(X.columns),
            n_numeric=len(numeric_cols),
            n_categorical=len(categorical_cols),
            n_temporal=len(datetime_cols),
            n_text=len(text_cols),
            numeric_skewness=skewness,
            cardinalities=cardinalities,
            null_rates=null_rates,
            correlations=correlations,
            task_type=task_type,
            target_cardinality=target_cardinality,
        )
        profile.compute_hash()
        return profile

    def _detect_text_columns(self, X: pd.DataFrame, str_cols: list[str]) -> list[str]:
        """Detect which string columns are likely free text."""
        text_cols = []
        for col in str_cols:
            sample = X[col].dropna().head(50)
            if len(sample) == 0:
                continue
            avg_words = sample.astype(str).str.split().str.len().mean()
            if avg_words > 3:
                text_cols.append(col)
        return text_cols


class FeatureRecommender:
    """Recommends feature engineering strategies for a dataset.

    Combines rule-based heuristics with historical experiment data
    to produce ranked recommendations.

    Parameters:
        knowledge_base_path: Path to persist experiment knowledge base.
        max_recommendations: Maximum number of recommendations to return.

    Example:
        >>> recommender = FeatureRecommender()
        >>> recs = recommender.recommend(X_train, y_train)
        >>> for r in recs[:5]:
        ...     print(f"{r.name} ({r.confidence:.0%}): {r.description}")
    """

    def __init__(
        self,
        knowledge_base_path: str | Path | None = None,
        max_recommendations: int = 20,
    ) -> None:
        self.knowledge_base_path = knowledge_base_path
        self.max_recommendations = max_recommendations
        self._profiler = DatasetProfiler()
        self._kb = KnowledgeBase(knowledge_base_path)
        self._last_profile: DatasetProfile | None = None

    def recommend(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> list[Recommendation]:
        """Generate feature engineering recommendations for a dataset.

        Args:
            X: Feature DataFrame.
            y: Optional target series.

        Returns:
            List of Recommendation objects sorted by confidence (descending).
        """
        profile = self._profiler.profile(X, y)
        self._last_profile = profile

        recommendations: list[Recommendation] = []
        recommendations.extend(self._skewness_recommendations(X, profile))
        recommendations.extend(self._cardinality_recommendations(X, profile))
        recommendations.extend(self._correlation_recommendations(profile))
        recommendations.extend(self._null_recommendations(profile))
        recommendations.extend(self._interaction_recommendations(X, profile))
        recommendations.extend(self._temporal_recommendations(X, profile))
        recommendations.extend(self._selection_recommendations(profile))
        recommendations.extend(self._scaling_recommendations(X, profile))
        recommendations.extend(self._knowledge_base_recommendations(profile))

        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        return recommendations[:self.max_recommendations]

    def record_experiment(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        transformations: list[str],
        score_before: float,
        score_after: float,
        metric: str = "accuracy",
    ) -> None:
        """Record the result of a feature engineering experiment.

        Args:
            X: Feature DataFrame used.
            y: Target series.
            transformations: List of transformations applied.
            score_before: Model score before transformations.
            score_after: Model score after transformations.
            metric: Scoring metric name.
        """
        profile = self._profiler.profile(X, y)
        record = ExperimentRecord(
            profile_hash=profile.fingerprint_hash,
            transformations=transformations,
            score_before=score_before,
            score_after=score_after,
            metric=metric,
        )
        self._kb.record(record)

    @property
    def last_profile(self) -> DatasetProfile | None:
        """The most recently computed dataset profile."""
        return self._last_profile

    @property
    def knowledge_base(self) -> KnowledgeBase:
        """Access the underlying knowledge base."""
        return self._kb

    # --- Rule-based recommendation generators ---

    def _skewness_recommendations(
        self, X: pd.DataFrame, profile: DatasetProfile
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []
        right_skewed_positive = [
            col for col, sk in profile.numeric_skewness.items()
            if sk > 2.0 and (X[col].dropna() > 0).all()
        ]
        if right_skewed_positive:
            recs.append(Recommendation(
                name="log_transform",
                description=f"Apply log transform to {len(right_skewed_positive)} highly right-skewed columns",
                category=TransformCategory.NUMERIC,
                confidence=0.85,
                columns=right_skewed_positive,
                parameters={"base": "natural"},
                rationale="Columns with skewness > 2.0 and positive values benefit from log normalization",
            ))

        moderate_skew = [
            col for col, sk in profile.numeric_skewness.items()
            if 1.0 < abs(sk) <= 2.0
        ]
        if moderate_skew:
            recs.append(Recommendation(
                name="sqrt_transform",
                description=f"Apply sqrt transform to {len(moderate_skew)} moderately skewed columns",
                category=TransformCategory.NUMERIC,
                confidence=0.65,
                columns=moderate_skew,
                parameters={},
                rationale="Columns with skewness between 1.0 and 2.0 may benefit from sqrt normalization",
            ))
        return recs

    def _cardinality_recommendations(
        self, X: pd.DataFrame, profile: DatasetProfile
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []
        low_card = [c for c, v in profile.cardinalities.items() if v <= 10]
        high_card = [c for c, v in profile.cardinalities.items() if v > 10]

        if low_card:
            recs.append(Recommendation(
                name="onehot_encoding",
                description=f"One-hot encode {len(low_card)} low-cardinality categorical columns",
                category=TransformCategory.CATEGORICAL,
                confidence=0.90,
                columns=low_card,
                parameters={"drop": "first"},
                rationale="Columns with ≤10 unique values are ideal for one-hot encoding",
            ))
        if high_card:
            encoding = "target" if profile.task_type != TaskType.UNKNOWN else "frequency"
            recs.append(Recommendation(
                name=f"{encoding}_encoding",
                description=f"Apply {encoding} encoding to {len(high_card)} high-cardinality columns",
                category=TransformCategory.CATEGORICAL,
                confidence=0.80,
                columns=high_card,
                parameters={"smoothing": 1.0} if encoding == "target" else {},
                rationale=f"Columns with >10 unique values benefit from {encoding} encoding",
            ))
        return recs

    def _correlation_recommendations(self, profile: DatasetProfile) -> list[Recommendation]:
        recs: list[Recommendation] = []
        high_corr = {k: v for k, v in profile.correlations.items() if abs(v) > 0.9}
        if high_corr:
            recs.append(Recommendation(
                name="remove_correlated",
                description=f"Remove or consolidate {len(high_corr)} highly correlated feature pairs",
                category=TransformCategory.SELECTION,
                confidence=0.75,
                columns=list(high_corr.keys()),
                parameters={"threshold": 0.9},
                rationale="Feature pairs with correlation >0.9 add redundancy",
            ))
        return recs

    def _null_recommendations(self, profile: DatasetProfile) -> list[Recommendation]:
        recs: list[Recommendation] = []
        high_null = {k: v for k, v in profile.null_rates.items() if v > 0.3}
        some_null = {k: v for k, v in profile.null_rates.items() if 0 < v <= 0.3}

        if high_null:
            recs.append(Recommendation(
                name="missing_indicators",
                description=f"Add missing value indicators for {len(high_null)} columns with >30% nulls",
                category=TransformCategory.NUMERIC,
                confidence=0.80,
                columns=list(high_null.keys()),
                parameters={"strategy": "indicator"},
                rationale="Columns with high null rates often carry signal in the missingness pattern",
            ))
        if some_null:
            recs.append(Recommendation(
                name="imputation",
                description=f"Impute {len(some_null)} columns with missing values",
                category=TransformCategory.NUMERIC,
                confidence=0.70,
                columns=list(some_null.keys()),
                parameters={"strategy": "median"},
                rationale="Median imputation is a robust default for moderate null rates",
            ))
        return recs

    def _interaction_recommendations(
        self, X: pd.DataFrame, profile: DatasetProfile
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if 2 <= len(numeric_cols) <= 20:
            recs.append(Recommendation(
                name="pairwise_interactions",
                description=f"Generate pairwise interactions from {len(numeric_cols)} numeric columns",
                category=TransformCategory.INTERACTION,
                confidence=0.70,
                columns=numeric_cols[:10],
                parameters={"operations": ["multiply", "divide"]},
                rationale="Pairwise interactions capture non-linear relationships",
            ))
        if profile.n_numeric >= 3 and profile.n_rows >= 500:
            recs.append(Recommendation(
                name="polynomial_features",
                description="Generate degree-2 polynomial features for numeric columns",
                category=TransformCategory.INTERACTION,
                confidence=0.55,
                columns=numeric_cols[:5],
                parameters={"degree": 2, "interaction_only": False},
                rationale="Polynomial features capture non-linear patterns with sufficient samples",
            ))
        return recs

    def _temporal_recommendations(
        self, X: pd.DataFrame, profile: DatasetProfile
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []
        datetime_cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
        if datetime_cols:
            recs.append(Recommendation(
                name="datetime_components",
                description=f"Extract temporal components from {len(datetime_cols)} datetime columns",
                category=TransformCategory.TEMPORAL,
                confidence=0.90,
                columns=datetime_cols,
                parameters={"components": ["year", "month", "dayofweek", "hour"]},
                rationale="Datetime columns almost always benefit from component extraction",
            ))
        return recs

    def _selection_recommendations(self, profile: DatasetProfile) -> list[Recommendation]:
        recs: list[Recommendation] = []
        total_features = profile.n_numeric + profile.n_categorical

        if total_features > 50:
            recs.append(Recommendation(
                name="importance_selection",
                description=f"Select top features from {total_features} using tree-based importance",
                category=TransformCategory.SELECTION,
                confidence=0.75,
                parameters={"method": "importance", "max_features": min(50, total_features)},
                rationale="High-dimensional datasets benefit from feature selection",
            ))
        if profile.n_numeric > 10:
            recs.append(Recommendation(
                name="variance_filter",
                description="Remove near-zero variance features",
                category=TransformCategory.SELECTION,
                confidence=0.60,
                parameters={"threshold": 0.01},
                rationale="Low-variance features contribute little discriminative power",
            ))
        return recs

    def _scaling_recommendations(
        self, X: pd.DataFrame, profile: DatasetProfile
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            ranges = {}
            for col in numeric_cols:
                col_range = X[col].max() - X[col].min()
                if np.isfinite(col_range):
                    ranges[col] = float(col_range)

            if ranges:
                max_range = max(ranges.values())
                min_range = min(v for v in ranges.values() if v > 0) if any(v > 0 for v in ranges.values()) else 1
                if max_range / max(min_range, 1e-10) > 100:
                    recs.append(Recommendation(
                        name="standard_scaling",
                        description=f"Standardize {len(numeric_cols)} numeric columns with very different scales",
                        category=TransformCategory.NUMERIC,
                        confidence=0.70,
                        columns=numeric_cols,
                        parameters={"method": "standard"},
                        rationale="Features with >100x range difference benefit from standardization",
                    ))
        return recs

    def _knowledge_base_recommendations(
        self, profile: DatasetProfile
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []
        past = self._kb.query(profile.fingerprint_hash, top_k=3)
        for experiment in past:
            if experiment.improvement > 0:
                recs.append(Recommendation(
                    name="historical_strategy",
                    description=f"Apply previously successful strategy: {', '.join(experiment.transformations)}",
                    category=TransformCategory.INTERACTION,
                    confidence=min(0.90, 0.5 + experiment.improvement),
                    parameters={"transformations": experiment.transformations},
                    rationale=f"This strategy improved {experiment.metric} by {experiment.improvement:.4f} on a similar dataset",
                ))
        return recs


__all__ = [
    "DatasetProfile",
    "DatasetProfiler",
    "ExperimentRecord",
    "FeatureRecommender",
    "KnowledgeBase",
    "Recommendation",
    "TaskType",
    "TransformCategory",
]
