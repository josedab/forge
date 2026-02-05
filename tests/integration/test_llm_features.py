"""Integration tests for LLM-powered feature generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from forge import FeatureSuggester, LLMFeatureGenerator, MetadataExtractor
from forge.llm.suggestions import FeatureSuggestion


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        "price": np.random.uniform(10, 100, n_samples),
        "quantity": np.random.randint(1, 10, n_samples),
        "category": np.random.choice(["A", "B", "C"], n_samples),
        "date": pd.date_range("2020-01-01", periods=n_samples),
        "description": [f"Item {i} - good product" for i in range(n_samples)],
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
        FeatureSuggestion(
            name="date_year",
            description="Year from date",
            source_columns=["date"],
            transformation="datetime_extract",
            rationale="Extract temporal feature",
            confidence=0.75,
        ),
        FeatureSuggestion(
            name="description_length",
            description="Length of description text",
            source_columns=["description"],
            transformation="text_length",
            rationale="Text feature",
            confidence=0.7,
        ),
    ]


class TestLLMFeatureGeneratorIntegration:
    """Integration tests for LLMFeatureGenerator."""

    def test_fit_transform_with_mock(self, sample_df, sample_target, mock_suggestions):
        """Test fit_transform with mocked LLM suggestions."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(
                provider="openai",
                min_confidence=0.5,
                verbose=0,
            )

            result = generator.fit_transform(sample_df, sample_target)

            # Should include original columns plus generated features
            assert len(result) == len(sample_df)
            assert "price_x_quantity" in result.columns
            assert "log_price" in result.columns
            assert "category_freq" in result.columns

    def test_transform_after_fit(self, sample_df, sample_target, mock_suggestions):
        """Test that transform works after fit."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(min_confidence=0.5)
            generator.fit(sample_df, sample_target)

            # Transform same data
            result1 = generator.transform(sample_df)

            # Transform new data
            new_df = sample_df.head(50).copy()
            result2 = generator.transform(new_df)

            # Same columns
            assert list(result1.columns) == list(result2.columns)
            assert len(result2) == 50

    def test_get_feature_names_out(self, sample_df, sample_target, mock_suggestions):
        """Test get_feature_names_out method."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(min_confidence=0.5, include_originals=True)
            generator.fit(sample_df, sample_target)

            names = generator.get_feature_names_out()

            assert isinstance(names, list)
            # Should include originals
            assert "price" in names
            assert "quantity" in names
            # Should include generated
            assert "price_x_quantity" in names

    def test_exclude_originals(self, sample_df, sample_target, mock_suggestions):
        """Test with include_originals=False."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(
                min_confidence=0.5,
                include_originals=False,
            )
            result = generator.fit_transform(sample_df, sample_target)

            # Should not include all original columns
            assert len(result.columns) <= len(mock_suggestions)

    def test_confidence_filtering(self, sample_df, sample_target, mock_suggestions):
        """Test that low-confidence suggestions are filtered."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            # High confidence threshold
            generator = LLMFeatureGenerator(min_confidence=0.85)
            generator.fit(sample_df, sample_target)

            # Only price_x_quantity (0.9) and log_price (0.85) should pass
            suggestions = generator.get_suggestions()
            assert len(suggestions) == 2

    def test_get_suggestions(self, sample_df, sample_target, mock_suggestions):
        """Test get_suggestions method."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(min_confidence=0.5)
            generator.fit(sample_df, sample_target)

            suggestions = generator.get_suggestions()

            assert isinstance(suggestions, list)
            assert all(isinstance(s, FeatureSuggestion) for s in suggestions)

    def test_get_implemented_features(self, sample_df, sample_target, mock_suggestions):
        """Test get_implemented_features method."""
        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(min_confidence=0.5)
            generator.fit(sample_df, sample_target)

            features = generator.get_implemented_features()

            assert isinstance(features, list)
            assert len(features) > 0


class TestTransformationTypes:
    """Tests for different transformation types."""

    def test_interaction_transformation(self, sample_df, sample_target):
        """Test interaction (multiplication) transformation."""
        suggestion = FeatureSuggestion(
            name="price_x_quantity",
            source_columns=["price", "quantity"],
            transformation="interaction",
            description="Test",
            rationale="Test",
            confidence=0.9,
        )

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = [suggestion]

            generator = LLMFeatureGenerator()
            result = generator.fit_transform(sample_df, sample_target)

            expected = sample_df["price"] * sample_df["quantity"]
            np.testing.assert_array_almost_equal(
                result["price_x_quantity"].values, expected.values
            )

    def test_ratio_transformation(self, sample_df, sample_target):
        """Test ratio (division) transformation."""
        suggestion = FeatureSuggestion(
            name="price_per_quantity",
            source_columns=["price", "quantity"],
            transformation="ratio",
            description="Test",
            rationale="Test",
            confidence=0.9,
        )

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = [suggestion]

            generator = LLMFeatureGenerator()
            result = generator.fit_transform(sample_df, sample_target)

            assert "price_per_quantity" in result.columns

    def test_log_transformation(self, sample_df, sample_target):
        """Test log transformation."""
        suggestion = FeatureSuggestion(
            name="log_price",
            source_columns=["price"],
            transformation="log",
            description="Test",
            rationale="Test",
            confidence=0.9,
        )

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = [suggestion]

            generator = LLMFeatureGenerator()
            result = generator.fit_transform(sample_df, sample_target)

            expected = np.log1p(sample_df["price"].clip(lower=0))
            np.testing.assert_array_almost_equal(
                result["log_price"].values, expected.values
            )

    def test_sqrt_transformation(self, sample_df, sample_target):
        """Test square root transformation."""
        suggestion = FeatureSuggestion(
            name="sqrt_price",
            source_columns=["price"],
            transformation="sqrt",
            description="Test",
            rationale="Test",
            confidence=0.9,
        )

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = [suggestion]

            generator = LLMFeatureGenerator()
            result = generator.fit_transform(sample_df, sample_target)

            expected = np.sqrt(sample_df["price"].clip(lower=0))
            np.testing.assert_array_almost_equal(
                result["sqrt_price"].values, expected.values
            )

    def test_frequency_encode_transformation(self, sample_df, sample_target):
        """Test frequency encoding transformation."""
        suggestion = FeatureSuggestion(
            name="category_freq",
            source_columns=["category"],
            transformation="frequency_encode",
            description="Test",
            rationale="Test",
            confidence=0.9,
        )

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = [suggestion]

            generator = LLMFeatureGenerator()
            result = generator.fit_transform(sample_df, sample_target)

            assert "category_freq" in result.columns
            # Frequency values should be between 0 and 1
            assert (result["category_freq"] >= 0).all()
            assert (result["category_freq"] <= 1).all()

    def test_text_length_transformation(self, sample_df, sample_target):
        """Test text length transformation."""
        suggestion = FeatureSuggestion(
            name="description_len",
            source_columns=["description"],
            transformation="text_length",
            description="Test",
            rationale="Test",
            confidence=0.9,
        )

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = [suggestion]

            generator = LLMFeatureGenerator()
            result = generator.fit_transform(sample_df, sample_target)

            expected = sample_df["description"].str.len()
            np.testing.assert_array_equal(
                result["description_len"].values, expected.values
            )

    def test_word_count_transformation(self, sample_df, sample_target):
        """Test word count transformation."""
        suggestion = FeatureSuggestion(
            name="description_words",
            source_columns=["description"],
            transformation="word_count",
            description="Test",
            rationale="Test",
            confidence=0.9,
        )

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = [suggestion]

            generator = LLMFeatureGenerator()
            result = generator.fit_transform(sample_df, sample_target)

            expected = sample_df["description"].str.split().str.len()
            np.testing.assert_array_equal(
                result["description_words"].values, expected.values
            )

    def test_datetime_extract_year(self, sample_df, sample_target):
        """Test datetime year extraction."""
        suggestion = FeatureSuggestion(
            name="date_year",
            source_columns=["date"],
            transformation="datetime_extract",
            description="Test",
            rationale="Test",
            confidence=0.9,
        )

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = [suggestion]

            generator = LLMFeatureGenerator()
            result = generator.fit_transform(sample_df, sample_target)

            assert "date_year" in result.columns
            # All dates are in 2020
            assert (result["date_year"] == 2020).all()


class TestMetadataExtractor:
    """Tests for MetadataExtractor class."""

    def test_extract_basic_metadata(self, sample_df, sample_target):
        """Test basic metadata extraction."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df, sample_target)

        assert metadata.n_rows == len(sample_df)
        assert metadata.n_columns == len(sample_df.columns)
        assert "price" in metadata.columns

    def test_metadata_column_types(self, sample_df, sample_target):
        """Test column type inference in metadata."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df, sample_target)

        # columns is a dict mapping name -> ColumnMetadata
        assert metadata.columns["price"].dtype in ["float64", "float"]
        assert metadata.columns["category"].dtype == "object"


class TestFeatureSuggester:
    """Tests for FeatureSuggester class."""

    def test_suggester_initialization_requires_api_key(self):
        """Test that suggester requires API key."""
        from forge.exceptions import ConfigurationError

        # Without API key, should raise error
        with pytest.raises(ConfigurationError):
            FeatureSuggester(provider="openai", max_suggestions=10)

    def test_suggester_with_mock_provider(self, sample_df, sample_target):
        """Test suggester with mocked provider."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df, sample_target)

        with patch("forge.llm.suggestions.FeatureSuggester._create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.suggest.return_value = [
                FeatureSuggestion(
                    name="price_x_quantity",
                    description="Product of price and quantity",
                    source_columns=["price", "quantity"],
                    transformation="interaction",
                    rationale="Captures total value",
                    confidence=0.9,
                )
            ]
            mock_create.return_value = mock_provider

            suggester = FeatureSuggester(provider="openai", api_key="test_key")
            # The suggest method uses the provider internally
            assert suggester is not None


class TestSklearnCompatibility:
    """Tests for sklearn compatibility."""

    def test_clone(self, sample_df, sample_target, mock_suggestions):
        """Test that generator can be cloned."""
        from sklearn.base import clone

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(
                max_features=10,
                min_confidence=0.6,
            )

            cloned = clone(generator)

            assert cloned.max_features == 10
            assert cloned.min_confidence == 0.6

    def test_get_set_params(self, mock_suggestions):
        """Test get_params and set_params methods."""
        generator = LLMFeatureGenerator(
            max_features=10,
            min_confidence=0.6,
        )

        params = generator.get_params()
        assert params["max_features"] == 10
        assert params["min_confidence"] == 0.6

        generator.set_params(max_features=20)
        assert generator.max_features == 20


class TestIntegrationWorkflow:
    """End-to-end integration tests."""

    def test_full_workflow_with_sklearn_pipeline(self, sample_df, sample_target, mock_suggestions):
        """Test full workflow with sklearn pipeline."""
        from sklearn.ensemble import RandomForestClassifier

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            # Use LLMFeatureGenerator
            generator = LLMFeatureGenerator(min_confidence=0.5, verbose=0)
            X_features = generator.fit_transform(sample_df, sample_target)

            # Select only numeric columns for the classifier
            numeric_cols = X_features.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X_features[numeric_cols].fillna(0)

            # Train classifier
            classifier = RandomForestClassifier(n_estimators=10, random_state=42)
            classifier.fit(X_numeric, sample_target)
            predictions = classifier.predict(X_numeric)

            assert len(predictions) == len(sample_df)

    def test_train_test_workflow(self, sample_df, sample_target, mock_suggestions):
        """Test train/test split workflow."""
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            sample_df, sample_target, test_size=0.2, random_state=42
        )

        with patch("forge.llm.generator.FeatureSuggester") as MockSuggester:
            MockSuggester.return_value.suggest.return_value = mock_suggestions

            generator = LLMFeatureGenerator(min_confidence=0.5)

            # Fit on train
            X_train_transformed = generator.fit_transform(X_train, y_train)

            # Transform test
            X_test_transformed = generator.transform(X_test)

            # Same columns
            assert list(X_train_transformed.columns) == list(X_test_transformed.columns)
