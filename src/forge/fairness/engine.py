"""Fairness and bias detection engine.

Implements disparate impact analysis, demographic parity checks,
equalized odds assessment, and fair feature generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class FairnessMetric(str, Enum):
    """Fairness metrics for evaluation."""

    DISPARATE_IMPACT = "disparate_impact"
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    CORRELATION = "correlation"


@dataclass
class BiasReport:
    """Report of bias detection for a feature.

    Attributes:
    ----------
    feature : str
        Feature name.
    protected_attribute : str
        Protected attribute checked against.
    metric : FairnessMetric
        Metric used for evaluation.
    score : float
        Bias score (interpretation depends on metric).
    is_biased : bool
        Whether bias exceeds the threshold.
    threshold : float
        Threshold used for determination.
    details : dict[str, Any]
        Additional details about the analysis.
    recommendation : str
        Suggested remediation.
    """

    feature: str
    protected_attribute: str
    metric: FairnessMetric
    score: float
    is_biased: bool
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def __str__(self) -> str:
        """Human-readable bias report."""
        icon = "🚨" if self.is_biased else "✓"
        return (
            f"{icon} {self.feature} vs {self.protected_attribute}: "
            f"{self.metric.value}={self.score:.4f} "
            f"(threshold={self.threshold:.4f}) "
            f"{'BIASED' if self.is_biased else 'OK'}"
        )


class BiasDetector(BaseEstimator):
    """Detect bias in features with respect to protected attributes.

    Scans features for correlations and disparate impact against
    protected attributes (e.g., gender, race, age group).

    Parameters
    ----------
    protected_columns : list[str]
        Columns considered protected attributes.
    metrics : list[FairnessMetric] | None
        Fairness metrics to evaluate. None uses all available.
    threshold : float
        Threshold for flagging bias. Interpretation depends on metric:
        - disparate_impact: ratio < threshold (typically 0.8)
        - demographic_parity: difference > threshold (typically 0.1)
        - correlation: abs(correlation) > threshold (typically 0.3)
    scan_all_features : bool
        If True, scan all non-protected features against all protected.

    Attributes:
    ----------
    bias_reports_ : list[BiasReport]
        All generated bias reports.
    biased_features_ : list[str]
        Features flagged as biased.

    Examples:
    --------
    >>> from forge.fairness import BiasDetector, FairnessMetric
    >>>
    >>> detector = BiasDetector(
    ...     protected_columns=["gender", "age_group"],
    ...     threshold=0.3,
    ... )
    >>> detector.fit(X, y)
    >>> biased = detector.biased_features_
    >>> for report in detector.bias_reports_:
    ...     if report.is_biased:
    ...         print(report)
    """

    def __init__(
        self,
        protected_columns: list[str] | None = None,
        metrics: list[FairnessMetric] | None = None,
        threshold: float = 0.3,
        scan_all_features: bool = True,
    ) -> None:
        self.protected_columns = protected_columns or []
        self.metrics = metrics or [
            FairnessMetric.CORRELATION,
            FairnessMetric.DEMOGRAPHIC_PARITY,
        ]
        self.threshold = threshold
        self.scan_all_features = scan_all_features

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Analyze features for bias.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing features and protected attributes.
        y : Any
            Target variable (used for equalized odds).

        Returns:
        -------
        Self
            Fitted detector with bias reports.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(X).__name__}")

        self.bias_reports_: list[BiasReport] = []
        self.biased_features_: list[str] = []

        protected = [c for c in self.protected_columns if c in X.columns]
        if not protected:
            logger.warning("No protected columns found in data.")
            self._is_fitted = True
            return self

        feature_cols = [c for c in X.columns if c not in protected]

        for feat in feature_cols:
            for prot in protected:
                for metric in self.metrics:
                    report = self._evaluate_bias(X, feat, prot, metric, y)
                    if report is not None:
                        self.bias_reports_.append(report)
                        if report.is_biased and feat not in self.biased_features_:
                            self.biased_features_.append(feat)

        self._is_fitted = True
        return self

    def _evaluate_bias(
        self,
        X: pd.DataFrame,
        feature: str,
        protected: str,
        metric: FairnessMetric,
        y: Any,
    ) -> BiasReport | None:
        """Evaluate a single feature-protected pair."""
        try:
            if metric == FairnessMetric.CORRELATION:
                return self._check_correlation(X, feature, protected)
            elif metric == FairnessMetric.DEMOGRAPHIC_PARITY:
                return self._check_demographic_parity(X, feature, protected)
            elif metric == FairnessMetric.DISPARATE_IMPACT:
                return self._check_disparate_impact(X, feature, protected)
            elif metric == FairnessMetric.EQUALIZED_ODDS:
                return self._check_equalized_odds(X, feature, protected, y)
        except Exception as e:
            logger.debug(
                "Could not evaluate %s for %s vs %s: %s",
                metric.value, feature, protected, e,
            )
        return None

    def _check_correlation(
        self, X: pd.DataFrame, feature: str, protected: str
    ) -> BiasReport:
        """Check correlation between feature and protected attribute."""
        feat_values = self._encode_column(X[feature])
        prot_values = self._encode_column(X[protected])

        valid = ~(np.isnan(feat_values) | np.isnan(prot_values))
        if valid.sum() < 10:
            return BiasReport(
                feature=feature, protected_attribute=protected,
                metric=FairnessMetric.CORRELATION, score=0.0,
                is_biased=False, threshold=self.threshold,
                recommendation="Insufficient data for correlation analysis.",
            )

        corr = float(np.corrcoef(feat_values[valid], prot_values[valid])[0, 1])
        is_biased = abs(corr) > self.threshold

        return BiasReport(
            feature=feature,
            protected_attribute=protected,
            metric=FairnessMetric.CORRELATION,
            score=abs(corr),
            is_biased=is_biased,
            threshold=self.threshold,
            details={"correlation": corr, "direction": "positive" if corr > 0 else "negative"},
            recommendation=(
                f"Consider decorrelating '{feature}' from '{protected}'."
                if is_biased else "No significant correlation detected."
            ),
        )

    def _check_demographic_parity(
        self, X: pd.DataFrame, feature: str, protected: str
    ) -> BiasReport:
        """Check demographic parity for a feature across groups."""
        groups = X.groupby(protected)[feature]
        group_means: dict[str, float] = {}

        for name, group in groups:
            values = self._encode_column(group)
            valid = ~np.isnan(values)
            if valid.sum() > 0:
                group_means[str(name)] = float(np.mean(values[valid]))

        if len(group_means) < 2:
            return BiasReport(
                feature=feature, protected_attribute=protected,
                metric=FairnessMetric.DEMOGRAPHIC_PARITY, score=0.0,
                is_biased=False, threshold=self.threshold,
                recommendation="Insufficient groups for parity analysis.",
            )

        means = list(group_means.values())
        max_diff = max(means) - min(means)
        # Normalize by range
        range_val = max(abs(max(means)), abs(min(means)), 1e-10)
        normalized_diff = max_diff / range_val

        is_biased = normalized_diff > self.threshold

        return BiasReport(
            feature=feature,
            protected_attribute=protected,
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            score=normalized_diff,
            is_biased=is_biased,
            threshold=self.threshold,
            details={"group_means": group_means, "max_difference": max_diff},
            recommendation=(
                f"Feature '{feature}' shows significant disparity across '{protected}' groups."
                if is_biased else "Acceptable demographic parity."
            ),
        )

    def _check_disparate_impact(
        self, X: pd.DataFrame, feature: str, protected: str
    ) -> BiasReport:
        """Check disparate impact ratio."""
        groups = X.groupby(protected)[feature]
        group_positive_rates: dict[str, float] = {}

        for name, group in groups:
            values = self._encode_column(group)
            valid = ~np.isnan(values)
            if valid.sum() > 0:
                median = float(np.median(values[valid]))
                rate = float(np.mean(values[valid] > median))
                group_positive_rates[str(name)] = rate

        if len(group_positive_rates) < 2:
            return BiasReport(
                feature=feature, protected_attribute=protected,
                metric=FairnessMetric.DISPARATE_IMPACT, score=1.0,
                is_biased=False, threshold=0.8,
            )

        rates = list(group_positive_rates.values())
        min_rate = min(rates)
        max_rate = max(rates)
        ratio = min_rate / max_rate if max_rate > 0 else 1.0

        # Disparate impact: ratio < 0.8 is typically considered biased
        di_threshold = 0.8
        is_biased = ratio < di_threshold

        return BiasReport(
            feature=feature,
            protected_attribute=protected,
            metric=FairnessMetric.DISPARATE_IMPACT,
            score=ratio,
            is_biased=is_biased,
            threshold=di_threshold,
            details={"group_rates": group_positive_rates, "ratio": ratio},
            recommendation=(
                f"Disparate impact detected: ratio {ratio:.3f} < {di_threshold}."
                if is_biased else "No disparate impact detected."
            ),
        )

    def _check_equalized_odds(
        self, X: pd.DataFrame, feature: str, protected: str, y: Any
    ) -> BiasReport | None:
        """Check equalized odds (requires target)."""
        if y is None:
            return None

        target = np.asarray(y)
        if len(target) != len(X):
            return None

        groups = X[protected].values
        feat_values = self._encode_column(X[feature])
        valid = ~np.isnan(feat_values)

        unique_groups = np.unique(groups[valid])
        if len(unique_groups) < 2:
            return None

        # Check if feature-target correlation varies by group
        group_corrs: dict[str, float] = {}
        for g in unique_groups:
            mask = (groups == g) & valid
            if mask.sum() > 10:
                corr = float(
                    np.corrcoef(feat_values[mask], target[mask])[0, 1]
                )
                if not np.isnan(corr):
                    group_corrs[str(g)] = corr

        if len(group_corrs) < 2:
            return None

        corr_values = list(group_corrs.values())
        max_diff = max(corr_values) - min(corr_values)
        is_biased = max_diff > self.threshold

        return BiasReport(
            feature=feature,
            protected_attribute=protected,
            metric=FairnessMetric.EQUALIZED_ODDS,
            score=max_diff,
            is_biased=is_biased,
            threshold=self.threshold,
            details={"group_correlations": group_corrs},
            recommendation=(
                "Feature-target relationship varies significantly across groups."
                if is_biased else "Equalized odds check passed."
            ),
        )

    @staticmethod
    def _encode_column(series: pd.Series) -> np.ndarray:
        """Encode a column to numeric values for analysis."""
        if series.dtype in (np.float64, np.int64, np.float32, np.int32):
            return series.values.astype(float)

        # Label encode categorical
        categories = series.dropna().unique()
        cat_map = {cat: i for i, cat in enumerate(categories)}
        return np.array([cat_map.get(v, np.nan) for v in series.values])

    def get_summary(self) -> dict[str, Any]:
        """Get bias detection summary.

        Returns:
        -------
        dict
            Summary of bias analysis results.
        """
        if not hasattr(self, "bias_reports_"):
            return {"error": "Not fitted yet."}

        return {
            "total_checks": len(self.bias_reports_),
            "biased_features": self.biased_features_,
            "bias_count": sum(1 for r in self.bias_reports_ if r.is_biased),
            "clean_count": sum(1 for r in self.bias_reports_ if not r.is_biased),
            "by_metric": {
                m.value: sum(
                    1 for r in self.bias_reports_
                    if r.metric == m and r.is_biased
                )
                for m in FairnessMetric
            },
        }


class FairFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate fair alternative features by decorrelating from protected attributes.

    Removes linear dependence on protected attributes from features
    using residualization (orthogonal projection).

    Parameters
    ----------
    protected_columns : list[str]
        Protected attribute columns.
    method : str
        Decorrelation method: 'residualize' or 'adversarial'.
    threshold : float
        Post-decorrelation correlation threshold for validation.

    Examples:
    --------
    >>> from forge.fairness import FairFeatureGenerator
    >>> gen = FairFeatureGenerator(protected_columns=["gender"])
    >>> X_fair = gen.fit_transform(X)
    """

    def __init__(
        self,
        protected_columns: list[str] | None = None,
        method: str = "residualize",
        threshold: float = 0.05,
    ) -> None:
        self.protected_columns = protected_columns or []
        self.method = method
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Learn decorrelation parameters.

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

        self._protected = [c for c in self.protected_columns if c in X.columns]
        self._feature_cols = [
            c for c in X.columns
            if c not in self._protected
            and X[c].dtype in (np.float64, np.int64, np.float32, np.int32)
        ]

        # Learn residualization coefficients
        self._coefficients: dict[str, np.ndarray] = {}
        self._intercepts: dict[str, float] = {}

        if self._protected and self._feature_cols:
            prot_matrix = self._get_protected_matrix(X)

            for feat in self._feature_cols:
                feat_values = X[feat].values.astype(float)
                valid = ~np.isnan(feat_values)
                if valid.sum() > len(self._protected) + 1:
                    P = prot_matrix[valid]
                    y_feat = feat_values[valid]
                    # Least squares: y = P @ beta + intercept
                    P_with_intercept = np.column_stack([P, np.ones(P.shape[0])])
                    try:
                        beta, _, _, _ = np.linalg.lstsq(P_with_intercept, y_feat, rcond=None)
                        self._coefficients[feat] = beta[:-1]
                        self._intercepts[feat] = float(beta[-1])
                    except np.linalg.LinAlgError:
                        pass

        self.feature_names_out_ = [f"{c}_fair" for c in self._feature_cols]
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply decorrelation to create fair features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns:
        -------
        pd.DataFrame
            Data with fair alternative features appended.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("FairFeatureGenerator must be fitted.")

        result = X.copy()

        if not self._protected or not self._feature_cols:
            return result

        prot_matrix = self._get_protected_matrix(X)

        for feat in self._feature_cols:
            if feat not in self._coefficients:
                continue

            feat_values = X[feat].values.astype(float)
            prediction = prot_matrix @ self._coefficients[feat] + self._intercepts[feat]
            residual = feat_values - prediction
            result[f"{feat}_fair"] = residual

        return result

    def get_feature_names_out(self) -> list[str]:
        """Get names of fair features."""
        return self.feature_names_out_

    def _get_protected_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Build numeric matrix from protected columns."""
        arrays: list[np.ndarray] = []
        for col in self._protected:
            if X[col].dtype in (np.float64, np.int64, np.float32, np.int32):
                arrays.append(X[col].values.astype(float))
            else:
                # Label encode
                categories = X[col].dropna().unique()
                cat_map = {cat: i for i, cat in enumerate(categories)}
                arrays.append(
                    np.array([cat_map.get(v, 0) for v in X[col].values], dtype=float)
                )
        return np.column_stack(arrays) if arrays else np.zeros((len(X), 1))
