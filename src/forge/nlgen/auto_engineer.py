"""Natural Language Feature Engineer.

Accepts a plain-English problem description and a DataFrame, then
automatically detects the domain, selects relevant feature strategies,
and generates engineered features—all without manual configuration.

Example:
    >>> from forge.nlgen.auto_engineer import NaturalLanguageFeatureEngineer
    >>> eng = NaturalLanguageFeatureEngineer()
    >>> X_new = eng.fit_transform(
    ...     X, y,
    ...     problem="I'm predicting customer churn for a SaaS product",
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.exceptions import NotFittedError, ValidationError

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "ecommerce": [
        "churn", "customer", "purchase", "order", "cart", "product",
        "revenue", "retention", "subscription", "saas", "ltv",
        "lifetime value", "basket", "conversion", "shopping",
    ],
    "finance": [
        "credit", "loan", "default", "fraud", "risk", "portfolio",
        "stock", "trading", "interest", "bank", "investment",
        "insurance", "claim", "underwriting",
    ],
    "healthcare": [
        "patient", "diagnosis", "clinical", "medical", "hospital",
        "treatment", "drug", "disease", "health", "readmission",
        "mortality", "survival", "ehr",
    ],
    "marketing": [
        "campaign", "click", "impression", "ad", "conversion",
        "audience", "segment", "engagement", "email", "ctr",
    ],
    "iot": [
        "sensor", "device", "telemetry", "signal", "vibration",
        "temperature", "pressure", "anomaly", "predictive maintenance",
    ],
}


@dataclass
class DomainDetectionResult:
    """Result of domain detection from a problem description."""

    domain: str
    confidence: float
    keywords_matched: list[str] = field(default_factory=list)


@dataclass
class FeatureStrategy:
    """A strategy for generating features."""

    name: str
    description: str
    generator_type: str
    columns: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoEngineerResult:
    """Result of automatic feature engineering."""

    domain: DomainDetectionResult
    strategies_applied: list[FeatureStrategy]
    n_original_features: int
    n_generated_features: int
    feature_names: list[str] = field(default_factory=list)


def detect_domain(problem_description: str) -> DomainDetectionResult:
    """Detect the domain from a problem description.

    Args:
        problem_description: Plain-English problem description.

    Returns:
        DomainDetectionResult with detected domain and confidence.
    """
    text = problem_description.lower()
    scores: dict[str, tuple[float, list[str]]] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in text]
        score = len(matched) / len(keywords)
        scores[domain] = (score, matched)

    if not scores:
        return DomainDetectionResult(domain="general", confidence=0.0)

    best_domain = max(scores, key=lambda d: scores[d][0])
    best_score, best_matched = scores[best_domain]

    if best_score < 0.05:
        return DomainDetectionResult(
            domain="general", confidence=0.0, keywords_matched=[]
        )

    return DomainDetectionResult(
        domain=best_domain,
        confidence=min(best_score * 5, 1.0),  # Scale up
        keywords_matched=best_matched,
    )


# ---------------------------------------------------------------------------
# Strategy selection per domain + column types
# ---------------------------------------------------------------------------

def _infer_column_types(
    X: pd.DataFrame,
) -> dict[str, list[str]]:
    """Classify columns into numeric, categorical, temporal, text."""
    types: dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "temporal": [],
        "text": [],
    }
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            types["temporal"].append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            types["numeric"].append(col)
        elif pd.api.types.is_object_dtype(X[col]):
            avg_len = X[col].dropna().astype(str).str.len().mean()
            if avg_len > 50:
                types["text"].append(col)
            else:
                types["categorical"].append(col)
        else:
            types["categorical"].append(col)
    return types


def _select_strategies(
    domain: str,
    col_types: dict[str, list[str]],
    has_target: bool,
) -> list[FeatureStrategy]:
    """Select feature generation strategies based on domain and column types."""
    strategies: list[FeatureStrategy] = []
    numeric = col_types["numeric"]
    categorical = col_types["categorical"]
    temporal = col_types["temporal"]

    # --- Universal numeric strategies ---
    if len(numeric) >= 2:
        strategies.append(FeatureStrategy(
            name="numeric_interactions",
            description="Pairwise interactions (multiply, divide) among numeric columns",
            generator_type="interaction",
            columns=numeric[:10],
            params={"operations": ["multiply", "divide"]},
        ))
    if numeric:
        strategies.append(FeatureStrategy(
            name="numeric_transforms",
            description="Log and sqrt transforms for skewed numeric columns",
            generator_type="transformation",
            columns=numeric[:15],
            params={"transformations": ["log", "sqrt"]},
        ))

    # --- Universal categorical strategies ---
    if categorical and has_target:
        strategies.append(FeatureStrategy(
            name="target_encoding",
            description="Target encoding for categorical columns",
            generator_type="target_encode",
            columns=categorical[:10],
        ))
    if categorical:
        strategies.append(FeatureStrategy(
            name="frequency_encoding",
            description="Frequency encoding for categorical columns",
            generator_type="frequency_encode",
            columns=categorical[:10],
        ))

    # --- Temporal strategies ---
    if temporal:
        strategies.append(FeatureStrategy(
            name="datetime_components",
            description="Extract date/time components",
            generator_type="datetime_extract",
            columns=temporal,
        ))

    # --- Domain-specific ---
    if domain == "ecommerce":
        if numeric:
            strategies.append(FeatureStrategy(
                name="rfm_features",
                description="Recency/Frequency/Monetary style features",
                generator_type="binning",
                columns=numeric[:5],
                params={"n_bins": 5},
            ))
    elif domain == "finance":
        if numeric:
            strategies.append(FeatureStrategy(
                name="risk_ratios",
                description="Financial risk ratio features",
                generator_type="ratio",
                columns=numeric[:6],
            ))
    elif domain == "healthcare" and numeric:
            strategies.append(FeatureStrategy(
                name="clinical_bins",
                description="Clinical metric binning",
                generator_type="binning",
                columns=numeric[:8],
                params={"n_bins": 4},
            ))

    return strategies


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NaturalLanguageFeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Automatically engineer features from a plain-English problem statement.

    Detects the domain, infers column types, selects appropriate strategies,
    and generates features in one ``fit_transform`` call.

    Parameters
    ----------
    problem : str
        Plain-English description of the ML problem.
    max_features : int
        Maximum number of new features to generate.
    include_originals : bool
        Whether to include original columns in the output.
    verbose : int
        Verbosity level.

    Examples:
    --------
    >>> eng = NaturalLanguageFeatureEngineer(
    ...     problem="Predicting customer churn for a SaaS product"
    ... )
    >>> X_new = eng.fit_transform(X, y)
    """

    def __init__(
        self,
        problem: str = "",
        max_features: int = 50,
        include_originals: bool = True,
        verbose: int = 0,
    ) -> None:
        self.problem = problem
        self.max_features = max_features
        self.include_originals = include_originals
        self.verbose = verbose

        self._is_fitted = False
        self._domain: DomainDetectionResult | None = None
        self._strategies: list[FeatureStrategy] = []
        self._feature_names_out: list[str] = []
        self._input_columns: list[str] = []
        self._target_encodings: dict[str, dict[Any, float]] = {}
        self._frequency_encodings: dict[str, dict[Any, float]] = {}
        self._numeric_stats: dict[str, dict[str, float]] = {}
        self._result: AutoEngineerResult | None = None

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the auto-engineer by analysing columns and learning encodings.

        Args:
            X: Input DataFrame.
            y: Optional target variable.

        Returns:
            Self.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")
        if len(X) == 0:
            raise ValidationError("Input DataFrame is empty")

        self._input_columns = list(X.columns)
        self._domain = detect_domain(self.problem)
        col_types = _infer_column_types(X)

        if self.verbose >= 1:
            logger.info("Detected domain: %s (%.2f)", self._domain.domain, self._domain.confidence)

        self._strategies = _select_strategies(
            self._domain.domain, col_types, has_target=y is not None,
        )

        # Learn encodings from training data
        self._learn_encodings(X, y, col_types)

        # Compute numeric stats for binning
        for col in col_types["numeric"]:
            vals = X[col].dropna()
            if len(vals) > 0:
                self._numeric_stats[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()) if len(vals) > 1 else 1.0,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate features for new data.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with generated features.
        """
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")

        result = X.copy() if self.include_originals else pd.DataFrame(index=X.index)
        generated: list[str] = []

        for strategy in self._strategies:
            new_cols = self._apply_strategy(X, strategy)
            for col_name, col_values in new_cols.items():
                if len(generated) >= self.max_features:
                    break
                result[col_name] = col_values
                generated.append(col_name)

        self._feature_names_out = list(result.columns)

        self._result = AutoEngineerResult(
            domain=self._domain or DomainDetectionResult("general", 0.0),
            strategies_applied=self._strategies,
            n_original_features=len(self._input_columns),
            n_generated_features=len(generated),
            feature_names=generated,
        )
        return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None, **fit_params: Any) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return list(self._feature_names_out)

    def get_result(self) -> AutoEngineerResult | None:
        """Return the last AutoEngineerResult (populated after transform)."""
        return self._result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _learn_encodings(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        col_types: dict[str, list[str]],
    ) -> None:
        """Learn target and frequency encodings from training data."""
        for col in col_types["categorical"]:
            freq = X[col].value_counts(normalize=True)
            self._frequency_encodings[col] = freq.to_dict()

            if y is not None:
                means = X.groupby(col).apply(
                    lambda g: y.loc[g.index].mean(),
                    include_groups=False,
                )
                self._target_encodings[col] = means.to_dict()

    def _apply_strategy(
        self, X: pd.DataFrame, strategy: FeatureStrategy,
    ) -> dict[str, pd.Series]:
        """Apply a single strategy and return new columns."""
        results: dict[str, pd.Series] = {}
        cols = [c for c in strategy.columns if c in X.columns]

        try:
            if strategy.generator_type == "interaction":
                results.update(self._gen_interactions(X, cols))
            elif strategy.generator_type == "transformation":
                results.update(self._gen_transforms(X, cols, strategy.params))
            elif strategy.generator_type == "target_encode":
                results.update(self._gen_target_encode(X, cols))
            elif strategy.generator_type == "frequency_encode":
                results.update(self._gen_freq_encode(X, cols))
            elif strategy.generator_type == "datetime_extract":
                results.update(self._gen_datetime(X, cols))
            elif strategy.generator_type == "binning":
                n_bins = strategy.params.get("n_bins", 5)
                results.update(self._gen_binning(X, cols, n_bins))
            elif strategy.generator_type == "ratio":
                results.update(self._gen_ratios(X, cols))
        except Exception as exc:
            logger.warning("Strategy %s failed: %s", strategy.name, exc)

        return results

    # --- generators ---------------------------------------------------

    def _gen_interactions(
        self, X: pd.DataFrame, cols: list[str],
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                out[f"{a}_x_{b}"] = (X[a] * X[b]).astype(float)
                denom = X[b].replace(0, np.nan)
                out[f"{a}_div_{b}"] = (X[a] / denom).astype(float)
        return out

    def _gen_transforms(
        self, X: pd.DataFrame, cols: list[str], params: dict[str, Any],
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        transforms = params.get("transformations", ["log"])
        for col in cols:
            vals = pd.to_numeric(X[col], errors="coerce")
            if "log" in transforms:
                out[f"{col}_log1p"] = np.log1p(vals.clip(lower=0))
            if "sqrt" in transforms:
                out[f"{col}_sqrt"] = np.sqrt(vals.clip(lower=0))
        return out

    def _gen_target_encode(
        self, X: pd.DataFrame, cols: list[str],
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for col in cols:
            if col in self._target_encodings:
                enc = self._target_encodings[col]
                global_mean = float(np.mean(list(enc.values())))
                out[f"{col}_target_enc"] = X[col].map(enc).fillna(global_mean).astype(float)
        return out

    def _gen_freq_encode(
        self, X: pd.DataFrame, cols: list[str],
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for col in cols:
            if col in self._frequency_encodings:
                out[f"{col}_freq"] = X[col].map(
                    self._frequency_encodings[col]
                ).fillna(0.0).astype(float)
        return out

    def _gen_datetime(
        self, X: pd.DataFrame, cols: list[str],
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for col in cols:
            dt = pd.to_datetime(X[col], errors="coerce")
            out[f"{col}_year"] = dt.dt.year.astype(float)
            out[f"{col}_month"] = dt.dt.month.astype(float)
            out[f"{col}_dayofweek"] = dt.dt.dayofweek.astype(float)
            out[f"{col}_hour"] = dt.dt.hour.astype(float)
        return out

    def _gen_binning(
        self, X: pd.DataFrame, cols: list[str], n_bins: int,
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for col in cols:
            vals = pd.to_numeric(X[col], errors="coerce")
            try:
                binned = pd.qcut(vals, q=n_bins, labels=False, duplicates="drop")
                out[f"{col}_bin{n_bins}"] = binned.astype(float)
            except (ValueError, TypeError):
                pass
        return out

    def _gen_ratios(
        self, X: pd.DataFrame, cols: list[str],
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                denom = X[b].replace(0, np.nan)
                out[f"{a}_over_{b}"] = (X[a] / denom).astype(float)
        return out
