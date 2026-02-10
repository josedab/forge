"""Tests for enhanced documentation generator."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from forge.documentation.enhanced import (
    EnhancedDocGenerator,
    RelationshipInfo,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 100
    a = rng.normal(0, 1, n)
    return pd.DataFrame({
        "price": a,
        "quantity": a * 2 + rng.normal(0, 0.1, n),  # highly correlated
        "category": rng.choice(["A", "B", "C"], n),
        "flag": rng.choice([True, False], n),
    })


class TestEnhancedDocGenerator:
    def test_generate_basic(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator()
        doc = gen.generate(sample_df, name="test_ds")
        assert doc.base.name == "test_ds"
        assert doc.base.n_samples == 100
        assert doc.base.n_features == 4

    def test_features_documented(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator()
        doc = gen.generate(sample_df)
        names = [f.name for f in doc.base.features]
        assert "price" in names
        assert "category" in names

    def test_relationships_found(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator(correlation_threshold=0.5)
        doc = gen.generate(sample_df)
        assert len(doc.relationships) >= 1
        pair = {doc.relationships[0].feature_a, doc.relationships[0].feature_b}
        assert pair == {"price", "quantity"}

    def test_high_threshold_no_relationships(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator(correlation_threshold=0.9999)
        doc = gen.generate(sample_df)
        # may or may not find any
        for r in doc.relationships:
            assert abs(r.correlation) >= 0.9999

    def test_summary_text(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator()
        doc = gen.generate(sample_df, name="mydata")
        assert "mydata" in doc.summary_text
        assert "100" in doc.summary_text

    def test_to_markdown(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator()
        doc = gen.generate(sample_df, name="test_md")
        md = gen.to_markdown(doc)
        assert "# test_md" in md
        assert "| Name |" in md
        assert "price" in md

    def test_to_html(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator()
        doc = gen.generate(sample_df, name="test_html")
        html = gen.to_html(doc)
        assert "<html>" in html
        assert "test_html" in html
        assert "<table>" in html

    def test_to_json(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator()
        doc = gen.generate(sample_df, name="test_json")
        j = gen.to_json(doc)
        data = json.loads(j)
        assert data["base"]["name"] == "test_json"

    def test_numeric_stats(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator()
        doc = gen.generate(sample_df)
        price_feat = next(f for f in doc.base.features if f.name == "price")
        assert "mean" in price_feat.statistics
        assert "std" in price_feat.statistics

    def test_category_detection(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator()
        doc = gen.generate(sample_df)
        cat_feat = next(f for f in doc.base.features if f.name == "category")
        assert cat_feat.category == "categorical"

    def test_example_values_limit(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator(max_example_values=3)
        doc = gen.generate(sample_df)
        for feat in doc.base.features:
            assert len(feat.example_values) <= 3

    def test_to_dict(self, sample_df: pd.DataFrame) -> None:
        gen = EnhancedDocGenerator()
        doc = gen.generate(sample_df)
        d = doc.to_dict()
        assert "base" in d
        assert "relationships" in d
        assert "summary_text" in d


class TestRelationshipInfo:
    def test_defaults(self) -> None:
        r = RelationshipInfo(feature_a="a", feature_b="b", correlation=0.9)
        assert r.relationship_type == "linear"
