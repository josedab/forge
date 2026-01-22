"""Tests for LLM-powered feature generator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from forge.exceptions import NotFittedError, ValidationError
from forge.llm.generator import LLMFeatureGenerator
from forge.llm.suggestions import FeatureSuggestion


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "price": np.random.uniform(10, 100, 100),
        "quantity": np.random.randint(1, 10, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "date": pd.date_range("2020-01-01", periods=100),
        "description": ["item " + str(i) for i in range(100)],
    })


@pytest.fixture
def sample_target():
    """Create a sample target series."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 100), name="target")


@pytest.fixture
def mock_suggestions():
    """Create mock feature suggestions."""
    return [
        FeatureSuggestion(
            name="price_x_quantity",
            description="Product of price and quantity",
            source_columns=["price", "quantity"],
            transformation="interaction",
            rationale="Captures total value",
            confidence=0.9,
        ),
        FeatureSuggestion(
            name="log_price",
            description="Log of price",
            source_columns=["price"],
            transformation="log",
            rationale="Reduces skewness",
            confidence=0.85,
        ),
        FeatureSuggestion(
            name="category_freq",
            description="Frequency encoding of category",
            source_columns=["category"],
            transformation="frequency_encode",
            rationale="Encodes category distribution",
            confidence=0.8,
        ),
    ]


class TestLLMFeatureGeneratorInit:
    """Tests for LLMFeatureGenerator initialization."""

    def test_default_parameters(self):
        """Test initialization with default parameters."""
        generator = LLMFeatureGenerator()

        assert generator.provider == "openai"
        assert generator.max_features == 20
        assert generator.min_confidence == 0.5
        assert generator.include_originals is True
        assert generator.verbose == 0

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        generator = LLMFeatureGenerator(
            provider="anthropic",
            api_key="test_key",
            model="claude-3",
            max_features=10,
            min_confidence=0.7,
            include_originals=False,
            verbose=2,
        )

        assert generator.provider == "anthropic"
        assert generator.api_key == "test_key"
        assert generator.max_features == 10
        assert generator.min_confidence == 0.7
        assert generator.include_originals is False

    def test_supported_transformations(self):
        """Test that common transformations are supported."""
        generator = LLMFeatureGenerator()

        expected = {
            "interaction", "ratio", "difference", "log", "sqrt",
            "square", "bin", "target_encode", "frequency_encode",
        }
        assert expected.issubset(generator.SUPPORTED_TRANSFORMATIONS)


class TestLLMFeatureGeneratorFit:
    """Tests for LLMFeatureGenerator fit method."""

    def test_fit_stores_suggestions(self, sample_df, sample_target, mock_suggestions):
        """Test that fit stores suggestions."""
        with patch.object(
            LLMFeatureGenerator, "_suggester", create=True
        ) as mock_attr:
            # Create generator and mock the suggester
            generator = LLMFeatureGenerator(min_confidence=0.5)

            # Mock the suggester creation and suggestion
            mock_suggester = MagicMock()
            mock_suggester.suggest.return_value = mock_suggestions

            with patch(
                "forge.llm.generator.FeatureSuggester", return_value=mock_suggester
            ):
                generator.fit(sample_df, sample_target)

            assert generator._is_fitted
            assert len(generator._suggestions) == 3

    def test_fit_filters_by_confidence(self, sample_df, sample_target, mock_suggestions):
        """Test that fit filters suggestions by confidence."""
        # Add a low-confidence suggestion
        mock_suggestions.append(
            FeatureSuggestion(
                name="low_conf",
                description="Low confidence",
                source_columns=["price"],
                transformation="sqrt",
                rationale="Test",
                confidence=0.3,
            )
        )

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(min_confidence=0.5)
            generator.fit(sample_df, sample_target)

            # Low confidence suggestion should be filtered
            names = [s.name for s in generator._suggestions]
            assert "low_conf" not in names
            assert "price_x_quantity" in names

    def test_fit_validates_input(self):
        """Test that fit validates input."""
        generator = LLMFeatureGenerator()

        with pytest.raises(ValidationError):
            generator.fit([1, 2, 3])  # Not a DataFrame

        with pytest.raises(ValidationError):
            generator.fit(pd.DataFrame())  # Empty DataFrame


class TestLLMFeatureGeneratorTransform:
    """Tests for LLMFeatureGenerator transform method."""

    @pytest.fixture
    def fitted_generator(self, sample_df, sample_target, mock_suggestions):
        """Create a fitted generator for testing transform."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(min_confidence=0.5)
            generator.fit(sample_df, sample_target)
            return generator

    def test_transform_without_fit_raises(self, sample_df):
        """Test that transform without fit raises error."""
        generator = LLMFeatureGenerator()

        with pytest.raises(NotFittedError):
            generator.transform(sample_df)

    def test_transform_includes_originals(self, fitted_generator, sample_df):
        """Test that transform includes original columns when configured."""
        result = fitted_generator.transform(sample_df)

        assert "price" in result.columns
        assert "quantity" in result.columns

    def test_transform_generates_features(self, fitted_generator, sample_df):
        """Test that transform generates new features."""
        result = fitted_generator.transform(sample_df)

        # Should have generated features
        assert "price_x_quantity" in result.columns or len(result.columns) > len(sample_df.columns)

    def test_transform_excludes_originals(self, sample_df, sample_target, mock_suggestions):
        """Test that transform excludes original columns when configured."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(include_originals=False)
            generator.fit(sample_df, sample_target)
            result = generator.transform(sample_df)

            # Original columns should not be present (only generated features)
            # Note: Some columns might still appear if they're source columns
            assert len(result.columns) <= len(mock_suggestions)


class TestLLMFeatureGeneratorTransformations:
    """Tests for individual transformation implementations."""

    @pytest.fixture
    def generator(self):
        """Create a generator for testing transformations."""
        return LLMFeatureGenerator()

    def test_interaction_transformation(self, generator, sample_df):
        """Test interaction (multiplication) transformation."""
        suggestion = FeatureSuggestion(
            name="price_x_quantity",
            source_columns=["price", "quantity"],
            transformation="interaction",
            description="Test",
            rationale="Test",
        )

        result = generator._generate_feature(sample_df, suggestion)

        expected = sample_df["price"] * sample_df["quantity"]
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_ratio_transformation(self, generator, sample_df):
        """Test ratio (division) transformation."""
        suggestion = FeatureSuggestion(
            name="price_per_quantity",
            source_columns=["price", "quantity"],
            transformation="ratio",
            description="Test",
            rationale="Test",
        )

        result = generator._generate_feature(sample_df, suggestion)

        # Handle division by zero
        expected = sample_df["price"] / sample_df["quantity"].replace(0, np.nan)
        np.testing.assert_array_almost_equal(
            result.values, expected.values, decimal=5
        )

    def test_difference_transformation(self, generator, sample_df):
        """Test difference (subtraction) transformation."""
        suggestion = FeatureSuggestion(
            name="price_minus_quantity",
            source_columns=["price", "quantity"],
            transformation="difference",
            description="Test",
            rationale="Test",
        )

        result = generator._generate_feature(sample_df, suggestion)

        expected = sample_df["price"] - sample_df["quantity"]
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_log_transformation(self, generator, sample_df):
        """Test log transformation."""
        suggestion = FeatureSuggestion(
            name="log_price",
            source_columns=["price"],
            transformation="log",
            description="Test",
            rationale="Test",
        )

        result = generator._generate_feature(sample_df, suggestion)

        expected = np.log1p(sample_df["price"].clip(lower=0))
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_sqrt_transformation(self, generator, sample_df):
        """Test square root transformation."""
        suggestion = FeatureSuggestion(
            name="sqrt_price",
            source_columns=["price"],
            transformation="sqrt",
            description="Test",
            rationale="Test",
        )

        result = generator._generate_feature(sample_df, suggestion)

        expected = np.sqrt(sample_df["price"].clip(lower=0))
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_square_transformation(self, generator, sample_df):
        """Test square transformation."""
        suggestion = FeatureSuggestion(
            name="price_squared",
            source_columns=["price"],
            transformation="square",
            description="Test",
            rationale="Test",
        )

        result = generator._generate_feature(sample_df, suggestion)

        expected = sample_df["price"] ** 2
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_text_length_transformation(self, generator, sample_df):
        """Test text length transformation."""
        suggestion = FeatureSuggestion(
            name="description_length",
            source_columns=["description"],
            transformation="text_length",
            description="Test",
            rationale="Test",
        )

        result = generator._generate_feature(sample_df, suggestion)

        expected = sample_df["description"].astype(str).str.len()
        np.testing.assert_array_equal(result.values, expected.values)

    def test_word_count_transformation(self, generator, sample_df):
        """Test word count transformation."""
        suggestion = FeatureSuggestion(
            name="description_words",
            source_columns=["description"],
            transformation="word_count",
            description="Test",
            rationale="Test",
        )

        result = generator._generate_feature(sample_df, suggestion)

        expected = sample_df["description"].astype(str).str.split().str.len()
        np.testing.assert_array_equal(result.values, expected.values)

    def test_frequency_encode_transformation(self, generator, sample_df):
        """Test frequency encoding transformation."""
        # Pre-populate frequency encodings
        freq = sample_df["category"].value_counts(normalize=True)
        generator._frequency_encodings = {"category": freq.to_dict()}

        suggestion = FeatureSuggestion(
            name="category_freq",
            source_columns=["category"],
            transformation="frequency_encode",
            description="Test",
            rationale="Test",
        )

        result = generator._generate_feature(sample_df, suggestion)

        expected = sample_df["category"].map(freq)
        np.testing.assert_array_almost_equal(result.values, expected.values)


class TestLLMFeatureGeneratorHelpers:
    """Tests for helper methods."""

    def test_can_implement_with_valid_suggestion(self, sample_df):
        """Test _can_implement with valid suggestion."""
        generator = LLMFeatureGenerator()

        suggestion = FeatureSuggestion(
            name="test",
            source_columns=["price", "quantity"],
            transformation="interaction",
            description="Test",
            rationale="Test",
        )

        assert generator._can_implement(suggestion, sample_df) is True

    def test_can_implement_with_missing_column(self, sample_df):
        """Test _can_implement with missing column."""
        generator = LLMFeatureGenerator()

        suggestion = FeatureSuggestion(
            name="test",
            source_columns=["price", "nonexistent"],
            transformation="interaction",
            description="Test",
            rationale="Test",
        )

        assert generator._can_implement(suggestion, sample_df) is False

    def test_can_implement_maps_transformation_aliases(self, sample_df):
        """Test _can_implement maps transformation aliases."""
        generator = LLMFeatureGenerator()

        suggestion = FeatureSuggestion(
            name="test",
            source_columns=["price", "quantity"],
            transformation="multiply",  # Alias for interaction
            description="Test",
            rationale="Test",
        )

        # Should map "multiply" to "interaction"
        result = generator._can_implement(suggestion, sample_df)
        assert result is True
        assert suggestion.transformation == "interaction"

    def test_can_implement_with_unsupported_transformation(self, sample_df):
        """Test _can_implement with unsupported transformation."""
        generator = LLMFeatureGenerator()

        suggestion = FeatureSuggestion(
            name="test",
            source_columns=["price"],
            transformation="unsupported_transform",
            description="Test",
            rationale="Test",
        )

        assert generator._can_implement(suggestion, sample_df) is False


class TestLLMFeatureGeneratorAccessors:
    """Tests for accessor methods."""

    def test_get_feature_names_out(self, sample_df, sample_target, mock_suggestions):
        """Test get_feature_names_out method."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator()
            generator.fit(sample_df, sample_target)

            names = generator.get_feature_names_out()

            assert isinstance(names, list)
            assert len(names) > 0

    def test_get_feature_names_out_without_fit_raises(self):
        """Test get_feature_names_out without fit raises error."""
        generator = LLMFeatureGenerator()

        with pytest.raises(NotFittedError):
            generator.get_feature_names_out()

    def test_get_suggestions(self, sample_df, sample_target, mock_suggestions):
        """Test get_suggestions method."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator()
            generator.fit(sample_df, sample_target)

            suggestions = generator.get_suggestions()

            assert isinstance(suggestions, list)
            assert all(isinstance(s, FeatureSuggestion) for s in suggestions)

    def test_get_implemented_features(self, sample_df, sample_target, mock_suggestions):
        """Test get_implemented_features method."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator()
            generator.fit(sample_df, sample_target)

            features = generator.get_implemented_features()

            assert isinstance(features, list)


class TestLLMFeatureGeneratorSklearnCompat:
    """Tests for sklearn compatibility."""

    def test_fit_transform(self, sample_df, sample_target, mock_suggestions):
        """Test fit_transform method."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator()
            result = generator.fit_transform(sample_df, sample_target)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_df)

    def test_get_params(self):
        """Test get_params method (sklearn compatibility)."""
        generator = LLMFeatureGenerator(
            max_features=10,
            min_confidence=0.6,
        )

        params = generator.get_params()

        assert params["max_features"] == 10
        assert params["min_confidence"] == 0.6

    def test_set_params(self):
        """Test set_params method (sklearn compatibility)."""
        generator = LLMFeatureGenerator()
        generator.set_params(max_features=15, verbose=1)

        assert generator.max_features == 15
        assert generator.verbose == 1
