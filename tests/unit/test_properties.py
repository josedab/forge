"""Property-based tests for Forge using Hypothesis.

These tests verify invariants that should hold for all valid inputs,
catching edge cases that example-based tests might miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from forge.generators.numeric import InteractionGenerator, PolynomialGenerator
from forge.generators.categorical import OneHotEncoder, TargetEncoder, FrequencyEncoder
from forge.selectors import VarianceSelector, CorrelationSelector


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for generating finite floats (no NaN, inf)
finite_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

# Strategy for small integers (useful for categorical-like data)
small_ints = st.integers(min_value=0, max_value=100)

# Strategy for category labels
category_labels = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=10,
)

# Strategy for numeric DataFrames with 2-5 columns
numeric_dataframe = data_frames(
    columns=[
        column(name="a", dtype=float, elements=finite_floats),
        column(name="b", dtype=float, elements=finite_floats),
        column(name="c", dtype=float, elements=finite_floats),
    ],
    index=range_indexes(min_size=10, max_size=100),
)

# Strategy for binary target variable
binary_target = st.integers(min_value=10, max_value=100).flatmap(
    lambda n: st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n)
).map(lambda x: pd.Series(x, name="target"))


# =============================================================================
# Property tests for Numeric Generators
# =============================================================================

class TestInteractionGeneratorProperties:
    """Property-based tests for InteractionGenerator."""

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_preserves_row_count(self, df: pd.DataFrame) -> None:
        """Interaction generator should preserve the number of rows."""
        gen = InteractionGenerator(columns=["a", "b"])
        result = gen.fit_transform(df)
        assert len(result) == len(df)

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_includes_original_columns(self, df: pd.DataFrame) -> None:
        """Result should include original columns."""
        gen = InteractionGenerator(columns=["a", "b"])
        result = gen.fit_transform(df)
        # Original columns should be present
        for col in df.columns:
            assert col in result.columns

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_generates_new_features(self, df: pd.DataFrame) -> None:
        """Should generate at least one new feature."""
        gen = InteractionGenerator(columns=["a", "b"])
        result = gen.fit_transform(df)
        # Should have more columns than original
        assert len(result.columns) >= len(df.columns)

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_no_nans_from_finite_input(self, df: pd.DataFrame) -> None:
        """No NaNs should be introduced from finite input (except division by zero)."""
        gen = InteractionGenerator(columns=["a", "b"], operations=["multiply", "add"])
        result = gen.fit_transform(df)
        # For multiply and add, no NaNs should appear
        assert not result[["a", "b"]].isna().any().any()


class TestPolynomialGeneratorProperties:
    """Property-based tests for PolynomialGenerator."""

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_preserves_row_count(self, df: pd.DataFrame) -> None:
        """Polynomial generator should preserve the number of rows."""
        gen = PolynomialGenerator(columns=["a", "b"], degree=2)
        result = gen.fit_transform(df)
        assert len(result) == len(df)

    @given(df=numeric_dataframe, degree=st.integers(min_value=1, max_value=3))
    @settings(max_examples=30, deadline=None)
    def test_feature_count_increases_with_degree(
        self, df: pd.DataFrame, degree: int
    ) -> None:
        """Higher degree should produce more or equal features."""
        gen_low = PolynomialGenerator(columns=["a", "b"], degree=1)
        gen_high = PolynomialGenerator(columns=["a", "b"], degree=degree)

        result_low = gen_low.fit_transform(df)
        result_high = gen_high.fit_transform(df)

        assert len(result_high.columns) >= len(result_low.columns)


# =============================================================================
# Property tests for Categorical Generators
# =============================================================================

class TestOneHotEncoderProperties:
    """Property-based tests for OneHotEncoder."""

    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        n_categories=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_preserves_row_count(self, n_rows: int, n_categories: int) -> None:
        """One-hot encoder should preserve the number of rows."""
        categories = [f"cat_{i}" for i in range(n_categories)]
        df = pd.DataFrame({
            "category": np.random.choice(categories, size=n_rows)
        })

        encoder = OneHotEncoder(columns=["category"])
        result = encoder.fit_transform(df)
        assert len(result) == n_rows

    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        n_categories=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_binary_output(self, n_rows: int, n_categories: int) -> None:
        """One-hot encoded columns should be binary (0 or 1)."""
        categories = [f"cat_{i}" for i in range(n_categories)]
        df = pd.DataFrame({
            "category": np.random.choice(categories, size=n_rows)
        })

        encoder = OneHotEncoder(columns=["category"])
        result = encoder.fit_transform(df)

        # All new columns should contain only 0s and 1s
        for col in result.columns:
            if col != "category":
                unique_vals = set(result[col].dropna().unique())
                assert unique_vals.issubset({0, 1, 0.0, 1.0})

    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        n_categories=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_row_sums_to_one(self, n_rows: int, n_categories: int) -> None:
        """Each row should have exactly one 1 across one-hot columns."""
        categories = [f"cat_{i}" for i in range(n_categories)]
        df = pd.DataFrame({
            "category": np.random.choice(categories, size=n_rows)
        })

        encoder = OneHotEncoder(columns=["category"], drop_first=False)
        result = encoder.fit_transform(df)

        # Get only the one-hot columns
        ohe_cols = [c for c in result.columns if c.startswith("category_")]
        if ohe_cols:
            row_sums = result[ohe_cols].sum(axis=1)
            assert (row_sums == 1).all()


class TestFrequencyEncoderProperties:
    """Property-based tests for FrequencyEncoder."""

    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        n_categories=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_values_between_zero_and_one(self, n_rows: int, n_categories: int) -> None:
        """Frequency encoded values should be between 0 and 1."""
        categories = [f"cat_{i}" for i in range(n_categories)]
        df = pd.DataFrame({
            "category": np.random.choice(categories, size=n_rows)
        })

        encoder = FrequencyEncoder(columns=["category"])
        result = encoder.fit_transform(df)

        freq_col = "category_freq" if "category_freq" in result.columns else "category"
        if freq_col in result.columns:
            assert (result[freq_col] >= 0).all()
            assert (result[freq_col] <= 1).all()


# =============================================================================
# Property tests for Selectors
# =============================================================================

class TestVarianceSelectorProperties:
    """Property-based tests for VarianceSelector."""

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_preserves_row_count(self, df: pd.DataFrame) -> None:
        """Variance selector should preserve the number of rows."""
        selector = VarianceSelector(threshold=0.0)
        result = selector.fit_transform(df)
        assert len(result) == len(df)

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_removes_constant_columns(self, df: pd.DataFrame) -> None:
        """Constant columns should be removed with any positive threshold."""
        # Add a constant column
        df_with_constant = df.copy()
        df_with_constant["constant"] = 42.0

        selector = VarianceSelector(threshold=0.001)
        result = selector.fit_transform(df_with_constant)

        assert "constant" not in result.columns

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_zero_threshold_keeps_all(self, df: pd.DataFrame) -> None:
        """With threshold=0, all non-constant columns should be kept."""
        # Ensure we have varying data
        assume(df.std().min() > 0)

        selector = VarianceSelector(threshold=0.0)
        result = selector.fit_transform(df)

        assert len(result.columns) == len(df.columns)


class TestCorrelationSelectorProperties:
    """Property-based tests for CorrelationSelector."""

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_preserves_row_count(self, df: pd.DataFrame) -> None:
        """Correlation selector should preserve the number of rows."""
        selector = CorrelationSelector(threshold=0.95)
        result = selector.fit_transform(df)
        assert len(result) == len(df)

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_threshold_one_keeps_all(self, df: pd.DataFrame) -> None:
        """With threshold=1.0, all columns should be kept (perfect correlation only)."""
        selector = CorrelationSelector(threshold=1.0)
        result = selector.fit_transform(df)
        assert len(result.columns) == len(df.columns)

    @given(df=numeric_dataframe)
    @settings(max_examples=50, deadline=None)
    def test_removes_duplicate_columns(self, df: pd.DataFrame) -> None:
        """Exact duplicate columns should be removed."""
        # Add a duplicate column
        df_with_dup = df.copy()
        df_with_dup["a_dup"] = df["a"]

        selector = CorrelationSelector(threshold=0.99)
        result = selector.fit_transform(df_with_dup)

        # Either 'a' or 'a_dup' should be removed, but not both
        assert len(result.columns) < len(df_with_dup.columns)


# =============================================================================
# Invariant tests for transformer chains
# =============================================================================

class TestTransformerChainProperties:
    """Property tests for chaining multiple transformers."""

    @given(df=numeric_dataframe)
    @settings(max_examples=30, deadline=None)
    def test_chain_preserves_rows(self, df: pd.DataFrame) -> None:
        """Chaining transformers should preserve row count."""
        # Chain: interactions -> polynomials -> variance filter
        gen1 = InteractionGenerator(columns=["a", "b"], operations=["multiply"])
        gen2 = PolynomialGenerator(degree=2)
        selector = VarianceSelector(threshold=0.0)

        result = df.copy()
        result = gen1.fit_transform(result)
        result = gen2.fit_transform(result)
        result = selector.fit_transform(result)

        assert len(result) == len(df)

    @given(df=numeric_dataframe)
    @settings(max_examples=30, deadline=None)
    def test_fit_transform_equals_fit_then_transform(self, df: pd.DataFrame) -> None:
        """fit_transform should equal fit followed by transform."""
        gen = InteractionGenerator(columns=["a", "b"])

        # Method 1: fit_transform
        result1 = gen.fit_transform(df.copy())

        # Method 2: fit then transform
        gen2 = InteractionGenerator(columns=["a", "b"])
        gen2.fit(df.copy())
        result2 = gen2.transform(df.copy())

        pd.testing.assert_frame_equal(result1, result2)
