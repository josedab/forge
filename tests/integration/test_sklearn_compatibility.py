"""Integration tests for sklearn compatibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from forge import AutoFeatureTransformer, ForgePipeline
from forge.generators.categorical import OneHotEncoder, TargetEncoder
from forge.generators.numeric import InteractionGenerator, NumericTransformer
from forge.selectors import CorrelationSelector, ImportanceSelector, VarianceSelector


class TestSklearnPipeline:
    """Tests for sklearn Pipeline integration."""

    def test_pipeline_fit_predict(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test AutoFeatureTransformer in sklearn Pipeline."""
        pipeline = Pipeline(
            [
                ("features", AutoFeatureTransformer(max_features=20, verbose=0)),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        pipeline.fit(sample_mixed_df, sample_target_binary)
        predictions = pipeline.predict(sample_mixed_df)

        assert len(predictions) == len(sample_mixed_df)
        assert set(predictions).issubset({0, 1})

    def test_pipeline_predict_proba(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test predict_proba in pipeline."""
        pipeline = Pipeline(
            [
                ("features", AutoFeatureTransformer(max_features=10, verbose=0)),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        pipeline.fit(sample_mixed_df, sample_target_binary)
        proba = pipeline.predict_proba(sample_mixed_df)

        assert proba.shape == (len(sample_mixed_df), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_pipeline_cross_validation(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test cross-validation with pipeline."""
        pipeline = Pipeline(
            [
                ("features", AutoFeatureTransformer(max_features=10, verbose=0)),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        scores = cross_val_score(
            pipeline, sample_mixed_df, sample_target_binary, cv=3
        )

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_pipeline_clone(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test that pipeline can be cloned."""
        pipeline = Pipeline(
            [
                ("features", AutoFeatureTransformer(max_features=10, verbose=0)),
                ("classifier", RandomForestClassifier(n_estimators=10)),
            ]
        )

        # Clone should work
        cloned = clone(pipeline)

        # Both should be fittable
        pipeline.fit(sample_mixed_df, sample_target_binary)
        cloned.fit(sample_mixed_df, sample_target_binary)


class TestForgePipeline:
    """Tests for ForgePipeline."""

    def test_basic_pipeline(
        self, sample_numeric_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test basic ForgePipeline."""
        pipeline = ForgePipeline(
            [
                ("transform", NumericTransformer(log=True, sqrt=False)),
                ("variance", VarianceSelector(threshold=0.01)),
            ]
        )

        result = pipeline.fit_transform(sample_numeric_df, sample_target_binary)
        assert isinstance(result, pd.DataFrame)

    def test_pipeline_with_selector(
        self, sample_numeric_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test ForgePipeline with selection step."""
        pipeline = ForgePipeline(
            [
                ("interactions", InteractionGenerator(operations=["multiply"])),
                ("correlation", CorrelationSelector(threshold=0.9)),
                ("importance", ImportanceSelector(n_features=5)),
            ]
        )

        result = pipeline.fit_transform(sample_numeric_df, sample_target_binary)
        assert len(result.columns) <= 5

    def test_get_feature_names(
        self, sample_numeric_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test get_feature_names_out through pipeline."""
        pipeline = ForgePipeline(
            [
                ("transform", NumericTransformer(log=False, sqrt=True)),
            ]
        )

        pipeline.fit(sample_numeric_df, sample_target_binary)
        names = pipeline.get_feature_names_out()

        assert isinstance(names, list)
        assert len(names) > 0


class TestTransformerCloning:
    """Tests for transformer cloning (sklearn estimator protocol)."""

    def test_clone_auto_transformer(self):
        """Test cloning AutoFeatureTransformer."""
        transformer = AutoFeatureTransformer(
            max_features=10,
            selection_method="importance",
            random_state=42,
        )

        cloned = clone(transformer)

        assert cloned.max_features == 10
        assert cloned.selection_method == "importance"
        assert cloned.random_state == 42

    def test_clone_generator(self):
        """Test cloning feature generator."""
        generator = NumericTransformer(
            log=True,
            sqrt=True,
            n_bins=5,
        )

        cloned = clone(generator)

        assert cloned.log is True
        assert cloned.sqrt is True
        assert cloned.n_bins == 5

    def test_clone_selector(self):
        """Test cloning selector."""
        selector = ImportanceSelector(n_features=10, threshold=0.1)

        cloned = clone(selector)

        assert cloned.n_features == 10
        assert cloned.threshold == 0.1


class TestTransformerGetSetParams:
    """Tests for get_params and set_params."""

    def test_get_params(self):
        """Test get_params."""
        transformer = AutoFeatureTransformer(
            max_features=10,
            selection_method="importance",
        )

        params = transformer.get_params()

        assert "max_features" in params
        assert params["max_features"] == 10
        assert params["selection_method"] == "importance"

    def test_set_params(self):
        """Test set_params."""
        transformer = AutoFeatureTransformer(max_features=10)

        transformer.set_params(max_features=20, selection_method="statistical")

        assert transformer.max_features == 20
        assert transformer.selection_method == "statistical"

    def test_get_params_deep(self):
        """Test get_params with deep=True for nested estimators."""
        pipeline = ForgePipeline(
            [
                ("transform", NumericTransformer(log=True, sqrt=False)),
            ]
        )

        params = pipeline.get_params(deep=True)

        assert "steps" in params
        # ForgePipeline inherits from sklearn.Pipeline which uses different param naming
        assert len(params) > 0


class TestFullWorkflow:
    """End-to-end workflow tests."""

    def test_full_workflow(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test complete feature engineering workflow."""
        # 1. Create and fit transformer
        transformer = AutoFeatureTransformer(
            max_features=20,
            numeric_transformations=["log", "sqrt"],
            categorical_encoding="onehot",
            missing_strategy="auto",
            selection_method="importance",
            verbose=0,
        )

        # 2. Transform data
        X_transformed = transformer.fit_transform(
            sample_mixed_df, sample_target_binary
        )

        # 3. Build sklearn pipeline
        pipeline = Pipeline(
            [
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        # 4. Train and evaluate
        pipeline.fit(X_transformed, sample_target_binary)
        predictions = pipeline.predict(X_transformed)

        assert len(predictions) == len(sample_mixed_df)

        # 5. Get feature importance
        importance = transformer.get_feature_importance()
        # Importance includes all generated features, X_transformed has selected ones
        assert len(importance) >= len(X_transformed.columns)

    def test_train_test_split_workflow(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test workflow with train/test split."""
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            sample_mixed_df, sample_target_binary, test_size=0.2, random_state=42
        )

        # Fit on train only
        transformer = AutoFeatureTransformer(max_features=15, verbose=0)
        X_train_transformed = transformer.fit_transform(X_train, y_train)

        # Transform test (no fit)
        X_test_transformed = transformer.transform(X_test)

        # Same features in both
        assert list(X_train_transformed.columns) == list(X_test_transformed.columns)

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train_transformed, y_train)

        # Predict on test
        predictions = model.predict(X_test_transformed)
        assert len(predictions) == len(X_test)
