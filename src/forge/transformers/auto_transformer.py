"""AutoFeatureTransformer - main entry point for automatic feature engineering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from typing_extensions import Self

from forge.analyzer import DataAnalyzer
from forge.exceptions import NotFittedError
from forge.generators.categorical import FrequencyEncoder, OneHotEncoder, TargetEncoder
from forge.generators.numeric import (
    InteractionGenerator,
    LogTransformer,
    PolynomialGenerator,
)
from forge.generators.temporal import DateTimeComponents
from forge.generators.text import TextBasicFeatures
from forge.missing import AutoImputer
from forge.selectors import CorrelationSelector, ImportanceSelector, VarianceSelector
from forge.types import ColumnType


class AutoFeatureTransformer(BaseEstimator, TransformerMixin):
    """Automatic feature engineering transformer.

    The main entry point for Forge. Automatically analyzes input data,
    generates features based on column types, and selects the most
    important features.

    Example:
        >>> from forge import AutoFeatureTransformer
        >>> transformer = AutoFeatureTransformer(max_features=100)
        >>> X_engineered = transformer.fit_transform(X, y)
        >>> print(transformer.get_feature_importance().head(10))

    Example with sklearn Pipeline:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> pipeline = Pipeline([
        ...     ("features", AutoFeatureTransformer(max_features=50)),
        ...     ("classifier", RandomForestClassifier()),
        ... ])
        >>> pipeline.fit(X_train, y_train)
    """

    def __init__(
        self,
        max_features: int | float | None = None,
        numeric_transformations: list[str] | None = None,
        categorical_encoding: str = "auto",
        temporal_features: list[str] | None = None,
        missing_strategy: str = "auto",
        selection_method: str = "importance",
        generate_interactions: bool = True,
        generate_polynomials: bool = False,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.0,
        n_jobs: int = -1,
        random_state: int | None = None,
        verbose: int = 0
    ) -> None:
        """Initialize the AutoFeatureTransformer.

        Args:
            max_features: Maximum features to keep. Int for count, float for fraction.
            numeric_transformations: List of transforms ("log", "sqrt", "bin").
            categorical_encoding: Encoding strategy ("auto", "onehot", "target", "frequency").
            temporal_features: Temporal components to extract.,
            missing_strategy: Imputation strategy ("auto", "mean", "median", "mode").
            selection_method: Selection method ("importance", "statistical", "shap").
            generate_interactions: Whether to generate feature interactions.,
            generate_polynomials: Whether to generate polynomial features.,
            correlation_threshold: Threshold for removing correlated features.,
            variance_threshold: Threshold for removing low-variance features.,
            n_jobs: Number of parallel jobs (-1 for all cores).,
            random_state: Random state for reproducibility.,
            verbose: Verbosity level (0, 1, or 2).
        """
        self.max_features = max_features
        self.numeric_transformations = numeric_transformations
        self.categorical_encoding = categorical_encoding
        self.temporal_features = temporal_features
        self.missing_strategy = missing_strategy
        self.selection_method = selection_method
        self.generate_interactions = generate_interactions
        self.generate_polynomials = generate_polynomials
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Internal state
        self._is_fitted: bool = False
        self._analyzer: DataAnalyzer | None = None
        self._column_types: dict[str, ColumnType] = {}
        self._generators: list[tuple[str, Any]] = []
        self._imputer: AutoImputer | None = None
        self._selectors: list[tuple[str, Any]] = []
        self._feature_names_out: list[str] = []
        self._feature_importance: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the transformer to the data.

        Analyzes input data, fits generators and selectors, and learns
        which features to generate and select.

        Args:
            X: Input DataFrame.,
            y: Target variable (required for some features).,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.verbose:
            print(f"Fitting AutoFeatureTransformer on {X.shape} data...")

        # Analyze data
        self._analyzer = DataAnalyzer()
        report = self._analyzer.analyze(X, y)
        self._column_types = report.column_types

        if self.verbose:
            print(f"  Detected column types: {dict(report.column_types)}")

        # Fit imputer
        self._imputer = AutoImputer(
            numeric_strategy = "median" if self.missing_strategy == "auto" else self.missing_strategy,
            add_indicator=True
        )
        self._imputer.fit(X, y)

        # Fit generators based on column types
        self._generators = []
        self._fit_generators(X, y)

        # Fit selectors
        self._selectors = []
        self._fit_selectors(X, y)

        # Determine final feature set
        self._compute_final_features(X, y)

        self._is_fitted = True

        if self.verbose:
            print(f"  Generated {len(self._feature_names_out)} features")

        return self

    def _fit_generators(self, X: pd.DataFrame, y: pd.Series | None) -> None:
        """Fit feature generators based on column types."""
        numeric_cols = [
            col for col, t in self._column_types.items()
            if t == ColumnType.NUMERIC
        ]
        categorical_cols = [
            col for col, t in self._column_types.items()
            if t == ColumnType.CATEGORICAL
        ]
        datetime_cols = [
            col for col, t in self._column_types.items()
            if t == ColumnType.DATETIME
        ]
        text_cols = [
            col for col, t in self._column_types.items()
            if t == ColumnType.TEXT
        ]

        # Numeric transformations
        if numeric_cols:
            transforms = self.numeric_transformations or ["log"]
            if "log" in transforms:
                log_gen = LogTransformer(columns=numeric_cols)
                log_gen.fit(X, y)
                self._generators.append(("log", log_gen))

            if "sqrt" in transforms:
                from forge.generators.numeric.transformations import PowerTransformer
                sqrt_gen = PowerTransformer(columns=numeric_cols, transforms=["sqrt"])
                sqrt_gen.fit(X, y)
                self._generators.append(("sqrt", sqrt_gen))

        # Interactions
        if numeric_cols and self.generate_interactions and len(numeric_cols) >= 2:
            int_gen = InteractionGenerator(
                columns = numeric_cols[:10],  # Limit to avoid explosion,
                max_interactions=50
            )
            int_gen.fit(X, y)
            self._generators.append(("interactions", int_gen))

        # Polynomials
        if numeric_cols and self.generate_polynomials:
            poly_gen = PolynomialGenerator(
                columns = numeric_cols[:5],
                degree = 2,
                max_features=20
            )
            poly_gen.fit(X, y)
            self._generators.append(("polynomials", poly_gen))

        # Categorical encoding
        if categorical_cols:
            encoding = self.categorical_encoding
            if encoding == "auto":
                # Use target encoding if we have target, else frequency
                encoding = "target" if y is not None else "frequency"

            if encoding == "onehot":
                enc = OneHotEncoder(columns=categorical_cols, max_categories=20)
            elif encoding == "target" and y is not None:
                enc = TargetEncoder(columns=categorical_cols)
            else:
                enc = FrequencyEncoder(columns=categorical_cols)

            enc.fit(X, y)
            self._generators.append(("categorical", enc))

        # Datetime features
        if datetime_cols:
            dt_gen = DateTimeComponents(
                columns = datetime_cols,
                components=self.temporal_features or [
                    "year", "month", "day", "dayofweek", "hour"
                ],
            )
            dt_gen.fit(X, y)
            self._generators.append(("datetime", dt_gen))

        # Text features
        if text_cols:
            text_gen = TextBasicFeatures(columns=text_cols)
            text_gen.fit(X, y)
            self._generators.append(("text", text_gen))

    def _fit_selectors(self, X: pd.DataFrame, y: pd.Series | None) -> None:
        """Fit feature selectors."""
        # Variance selector
        if self.variance_threshold > 0:
            var_sel = VarianceSelector(threshold=self.variance_threshold)
            self._selectors.append(("variance", var_sel))

        # Correlation selector
        if self.correlation_threshold < 1.0:
            corr_sel = CorrelationSelector(threshold=self.correlation_threshold)
            self._selectors.append(("correlation", corr_sel))

        # Importance-based selection
        if self.max_features is not None and y is not None:
            imp_sel = ImportanceSelector(
                n_features = self.max_features,
                random_state=self.random_state
            )
            self._selectors.append(("importance", imp_sel))

    def _compute_final_features(self, X: pd.DataFrame, y: pd.Series | None) -> None:
        """Compute the final set of features."""
        # Generate all features
        X_transformed = self._transform_all(X)

        # Apply selectors sequentially
        X_selected = X_transformed.copy()
        for name, selector in self._selectors:
            if hasattr(selector, "fit"):
                selector.fit(X_selected, y)
            X_selected = selector.transform(X_selected)

        self._feature_names_out = list(X_selected.columns)

        # Compute feature importance if we have a target
        if y is not None:
            self._compute_feature_importance(X_transformed, y)

    def _transform_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all generators and combine results."""
        # Start with imputed data
        if self._imputer is not None:
            result = self._imputer.transform(X)
        else:
            result = X.copy()

        # Add original numeric columns
        numeric_cols = [
            col for col, t in self._column_types.items()
            if t == ColumnType.NUMERIC and col in X.columns
        ]
        for col in numeric_cols:
            if col not in result.columns:
                result[col] = X[col]

        # Apply generators
        for name, gen in self._generators:
            try:
                gen_features = gen.transform(X)
                for col in gen_features.columns:
                    if col not in result.columns:
                        result[col] = gen_features[col]
            except Exception:
                # Skip failed generators
                pass

        return result

    def _compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Compute feature importance scores."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Determine task type
        unique_ratio = len(np.unique(y)) / len(y)
        is_classification = unique_ratio < 0.05 or len(np.unique(y)) <= 10

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].fillna(0)

        if is_classification:
            model = RandomForestClassifier(
                n_estimators = 50,
                random_state = self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            model = RandomForestRegressor(
                n_estimators = 50,
                random_state = self.random_state,
                n_jobs=self.n_jobs
            )

        model.fit(X_numeric, y)

        self._feature_importance = pd.DataFrame({
            "feature": numeric_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by generating and selecting features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with engineered features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        # Generate all features
        X_transformed = self._transform_all(X)

        # Select final features
        available = [c for c in self._feature_names_out if c in X_transformed.columns]
        return X_transformed[available]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Input DataFrame.,
            y: Target variable.,

        Returns:
            DataFrame with engineered features.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get names of output features.

        Args:
            input_features: Ignored, for sklearn compatibility.

        Returns:
            List of output feature names.
        """
        self._check_is_fitted()
        return self._feature_names_out.copy()

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.

        Returns:
            DataFrame with feature names and importance scores.
        """
        self._check_is_fitted()
        if self._feature_importance is None:
            return pd.DataFrame(columns=["feature", "importance"])
        return self._feature_importance.copy()

    def get_analysis_report(self) -> Any:
        """Get the data analysis report.

        Returns:
            AnalysisReport from the analyzer.
        """
        self._check_is_fitted()
        if self._analyzer is None:
            return None
        return self._analyzer

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X).__name__}")
        if len(X) == 0:
            raise ValueError("Input DataFrame is empty")

    def _check_is_fitted(self) -> None:
        """Check if the transformer is fitted."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
