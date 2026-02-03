"""Tests for the feature recommendation engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.recommend import (
    DatasetProfiler,
    ExperimentRecord,
    FeatureRecommender,
    KnowledgeBase,
    Recommendation,
    TaskType,
    TransformCategory,
)


@pytest.fixture
def mixed_df():
    """DataFrame with mixed column types."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "income": np.random.exponential(50000, n),  # right-skewed
        "age": np.random.randint(18, 80, n),
        "score": np.random.uniform(0, 1, n),
        "big_value": np.random.uniform(0, 1_000_000, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "region": np.random.choice([f"region_{i}" for i in range(50)], n),
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
    })


@pytest.fixture
def binary_target():
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 200), name="target")


@pytest.fixture
def continuous_target():
    np.random.seed(42)
    return pd.Series(np.random.normal(100, 20, 200), name="target")


class TestDatasetProfiler:
    def test_profile_basic(self, mixed_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(mixed_df)

        assert profile.n_rows == 200
        assert profile.n_columns == 7
        assert profile.n_numeric >= 3
        assert profile.n_categorical >= 1
        assert profile.n_temporal >= 1

    def test_profile_with_target_classification(self, mixed_df, binary_target):
        profiler = DatasetProfiler()
        profile = profiler.profile(mixed_df, binary_target)

        assert profile.task_type == TaskType.CLASSIFICATION
        assert profile.target_cardinality == 2

    def test_profile_with_target_regression(self, mixed_df, continuous_target):
        profiler = DatasetProfiler()
        profile = profiler.profile(mixed_df, continuous_target)

        assert profile.task_type == TaskType.REGRESSION

    def test_profile_computes_skewness(self, mixed_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(mixed_df)

        assert len(profile.numeric_skewness) > 0
        # Exponential distribution should be right-skewed
        assert "income" in profile.numeric_skewness

    def test_profile_computes_cardinalities(self, mixed_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(mixed_df)

        assert "category" in profile.cardinalities
        assert profile.cardinalities["category"] == 3

    def test_profile_hash_deterministic(self, mixed_df):
        profiler = DatasetProfiler()
        p1 = profiler.profile(mixed_df)
        p2 = profiler.profile(mixed_df)

        assert p1.fingerprint_hash == p2.fingerprint_hash

    def test_profile_to_dict(self, mixed_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(mixed_df)
        d = profile.to_dict()

        assert "n_rows" in d
        assert d["n_rows"] == 200

    def test_detects_null_rates(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, None, 4.0, None],
            "b": [1, 2, 3, 4, 5],
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert "a" in profile.null_rates
        assert profile.null_rates["a"] == pytest.approx(0.4)

    def test_detects_text_columns(self):
        df = pd.DataFrame({
            "short_cat": ["A", "B", "C", "A", "B"],
            "long_text": [
                "This is a long text with many words in it",
                "Another sentence that has several words too",
                "Yet another example of free text content here",
                "More text data for testing the text detection feature",
                "Final example text that should be detected as text",
            ],
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert profile.n_text >= 1


class TestKnowledgeBase:
    def test_record_and_query(self):
        kb = KnowledgeBase()
        record = ExperimentRecord(
            profile_hash="abc123",
            transformations=["log_transform", "interaction"],
            score_before=0.80,
            score_after=0.85,
        )
        kb.record(record)

        results = kb.query("abc123")
        assert len(results) == 1
        assert results[0].improvement == pytest.approx(0.05)

    def test_query_sorted_by_improvement(self):
        kb = KnowledgeBase()
        kb.record(ExperimentRecord("h1", ["a"], 0.8, 0.82))
        kb.record(ExperimentRecord("h1", ["b"], 0.8, 0.90))
        kb.record(ExperimentRecord("h1", ["c"], 0.8, 0.85))

        results = kb.query("h1")
        assert results[0].improvement > results[1].improvement

    def test_persistence(self, tmp_path):
        path = tmp_path / "kb.json"
        kb1 = KnowledgeBase(path)
        kb1.record(ExperimentRecord("h1", ["log"], 0.8, 0.9))
        assert kb1.size == 1

        kb2 = KnowledgeBase(path)
        assert kb2.size == 1
        assert kb2.query("h1")[0].improvement == pytest.approx(0.1)

    def test_query_all(self):
        kb = KnowledgeBase()
        kb.record(ExperimentRecord("h1", ["a"], 0.8, 0.82))
        kb.record(ExperimentRecord("h2", ["b"], 0.7, 0.85))

        results = kb.query_all()
        assert len(results) == 2
        assert results[0].improvement >= results[1].improvement


class TestFeatureRecommender:
    def test_recommend_returns_recommendations(self, mixed_df, binary_target):
        recommender = FeatureRecommender()
        recs = recommender.recommend(mixed_df, binary_target)

        assert len(recs) > 0
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_recommendations_sorted_by_confidence(self, mixed_df):
        recommender = FeatureRecommender()
        recs = recommender.recommend(mixed_df)

        for i in range(len(recs) - 1):
            assert recs[i].confidence >= recs[i + 1].confidence

    def test_recommend_skewed_data(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "highly_skewed": np.random.exponential(1, 200),
            "normal": np.random.normal(0, 1, 200),
        })
        recommender = FeatureRecommender()
        recs = recommender.recommend(df)

        rec_names = [r.name for r in recs]
        assert "log_transform" in rec_names or "sqrt_transform" in rec_names

    def test_recommend_categorical_data(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "low_card": np.random.choice(["A", "B", "C"], 100),
            "high_card": np.random.choice([f"val_{i}" for i in range(50)], 100),
            "num": np.random.normal(0, 1, 100),
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        recommender = FeatureRecommender()
        recs = recommender.recommend(df, y)

        rec_names = [r.name for r in recs]
        assert "onehot_encoding" in rec_names
        assert "target_encoding" in rec_names

    def test_recommend_temporal_data(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "value": np.random.normal(0, 1, 100),
        })
        recommender = FeatureRecommender()
        recs = recommender.recommend(df)

        rec_names = [r.name for r in recs]
        assert "datetime_components" in rec_names

    def test_recommend_with_nulls(self):
        df = pd.DataFrame({
            "a": [1.0, None, 3.0, None, 5.0, None, 7.0, None, 9.0, None],
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        recommender = FeatureRecommender()
        recs = recommender.recommend(df)

        rec_names = [r.name for r in recs]
        assert "missing_indicators" in rec_names or "imputation" in rec_names

    def test_recommend_max_recommendations(self, mixed_df):
        recommender = FeatureRecommender(max_recommendations=3)
        recs = recommender.recommend(mixed_df)

        assert len(recs) <= 3

    def test_record_experiment_feeds_knowledge(self, mixed_df, binary_target):
        recommender = FeatureRecommender()
        recommender.record_experiment(
            X=mixed_df, y=binary_target,
            transformations=["log_transform"],
            score_before=0.80, score_after=0.88,
        )
        assert recommender.knowledge_base.size == 1

        # Now recommend - should include historical strategy
        recs = recommender.recommend(mixed_df, binary_target)
        rec_names = [r.name for r in recs]
        assert "historical_strategy" in rec_names

    def test_last_profile_stored(self, mixed_df):
        recommender = FeatureRecommender()
        assert recommender.last_profile is None

        recommender.recommend(mixed_df)
        assert recommender.last_profile is not None
        assert recommender.last_profile.n_rows == 200

    def test_recommendation_to_dict(self):
        rec = Recommendation(
            name="test",
            description="Test recommendation",
            category=TransformCategory.NUMERIC,
            confidence=0.8,
            columns=["a", "b"],
        )
        d = rec.to_dict()
        assert d["name"] == "test"
        assert d["confidence"] == 0.8

    def test_knowledge_base_persistence(self, mixed_df, binary_target, tmp_path):
        kb_path = tmp_path / "kb.json"

        r1 = FeatureRecommender(knowledge_base_path=kb_path)
        r1.record_experiment(
            X=mixed_df, y=binary_target,
            transformations=["log"], score_before=0.8, score_after=0.9,
        )

        r2 = FeatureRecommender(knowledge_base_path=kb_path)
        assert r2.knowledge_base.size == 1
