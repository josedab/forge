"""Tests for feature suggestion module."""

import numpy as np
import pandas as pd
import pytest

from forge.llm.metadata import MetadataExtractor, DatasetMetadata
from forge.llm.suggestions import (
    FeatureSuggestion,
    FeatureSuggester,
    SuggestionCache,
    LLMProvider,
)
from forge.types import ColumnType


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        "price": np.random.uniform(10, 100, 100),
        "quantity": np.random.randint(1, 10, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "date": pd.date_range("2020-01-01", periods=100),
    })
    y = pd.Series(np.random.choice([0, 1], 100), name="target")

    extractor = MetadataExtractor()
    return extractor.extract(df, y)


class TestFeatureSuggestion:
    """Tests for FeatureSuggestion class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        suggestion = FeatureSuggestion(
            name="price_x_quantity",
            description="Product of price and quantity",
            source_columns=["price", "quantity"],
            transformation="interaction",
            rationale="Captures total value",
            confidence=0.85,
        )

        data = suggestion.to_dict()
        assert data["name"] == "price_x_quantity"
        assert data["confidence"] == 0.85
        assert "price" in data["source_columns"]

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "test_feature",
            "description": "Test",
            "source_columns": ["col1"],
            "transformation": "log",
            "rationale": "Test rationale",
            "confidence": 0.9,
        }

        suggestion = FeatureSuggestion.from_dict(data)
        assert suggestion.name == "test_feature"
        assert suggestion.confidence == 0.9


class TestSuggestionCache:
    """Tests for SuggestionCache class."""

    def test_cache_set_get(self):
        """Test basic cache operations."""
        cache = SuggestionCache(ttl_seconds=60)

        suggestion = FeatureSuggestion(
            name="test",
            description="Test",
            source_columns=["col1"],
            transformation="log",
            rationale="Test",
        )

        cache.set("key1", [suggestion])
        result = cache.get("key1")

        assert result is not None
        assert len(result) == 1
        assert result[0].name == "test"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = SuggestionCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = SuggestionCache()

        suggestion = FeatureSuggestion(
            name="test",
            description="Test",
            source_columns=["col1"],
            transformation="log",
            rationale="Test",
        )

        cache.set("key1", [suggestion])
        cache.clear()

        assert cache.get("key1") is None


class TestFeatureSuggester:
    """Tests for FeatureSuggester class."""

    def test_fallback_suggestions(self, sample_metadata):
        """Test that fallback suggestions are generated."""
        # Create a suggester with a mock provider that returns invalid JSON
        class MockProvider(LLMProvider):
            def generate(self, prompt, system_prompt=None):
                return "Invalid response"

        suggester = FeatureSuggester(provider=MockProvider(), max_suggestions=10)
        suggestions = suggester.suggest(sample_metadata, use_cache=False)

        # Should get fallback suggestions
        assert len(suggestions) > 0
        assert all(isinstance(s, FeatureSuggestion) for s in suggestions)

    def test_suggestion_validation(self, sample_metadata):
        """Test that suggestions are validated against available columns."""
        # Create a suggester with mock provider returning valid JSON
        class MockProvider(LLMProvider):
            def generate(self, prompt, system_prompt=None):
                return """[
                    {
                        "name": "price_log",
                        "description": "Log of price",
                        "source_columns": ["price"],
                        "transformation": "log",
                        "rationale": "Reduces skewness",
                        "confidence": 0.8
                    },
                    {
                        "name": "invalid_feature",
                        "description": "Uses invalid column",
                        "source_columns": ["nonexistent_column"],
                        "transformation": "log",
                        "rationale": "Test",
                        "confidence": 0.8
                    }
                ]"""

        suggester = FeatureSuggester(provider=MockProvider())
        suggestions = suggester.suggest(sample_metadata, use_cache=False)

        # Only valid suggestions should be returned
        valid_names = [s.name for s in suggestions]
        assert "price_log" in valid_names
        # Invalid feature should be filtered out
        assert "invalid_feature" not in valid_names

    def test_max_suggestions_limit(self, sample_metadata):
        """Test that max_suggestions is respected."""
        class MockProvider(LLMProvider):
            def generate(self, prompt, system_prompt=None):
                return "Invalid"

        suggester = FeatureSuggester(provider=MockProvider(), max_suggestions=3)
        suggestions = suggester.suggest(sample_metadata, use_cache=False)

        assert len(suggestions) <= 3

    def test_caching(self, sample_metadata):
        """Test that caching works correctly."""
        call_count = 0

        class MockProvider(LLMProvider):
            def generate(self, prompt, system_prompt=None):
                nonlocal call_count
                call_count += 1
                return "Invalid"

        suggester = FeatureSuggester(provider=MockProvider())

        # First call
        suggester.suggest(sample_metadata, use_cache=True)
        # Second call with same metadata
        suggester.suggest(sample_metadata, use_cache=True)

        # Provider should only be called once due to caching
        assert call_count == 1
