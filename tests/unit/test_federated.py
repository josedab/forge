"""Tests for federated feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.privacy.federated import (
    FederatedFeatureEngineer,
    secure_aggregate_mean,
)


@pytest.fixture
def party_dfs() -> list[pd.DataFrame]:
    rng = np.random.RandomState(42)
    return [
        pd.DataFrame({"x": rng.normal(10, 2, 200), "y": rng.normal(5, 1, 200)}),
        pd.DataFrame({"x": rng.normal(10, 2, 300), "y": rng.normal(5, 1, 300)}),
        pd.DataFrame({"x": rng.normal(10, 2, 100), "y": rng.normal(5, 1, 100)}),
    ]


class TestSecureAggregateMean:
    def test_basic(self) -> None:
        vals = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        mean, std = secure_aggregate_mean(
            vals, clip_low=0, clip_high=10, epsilon=10.0,
        )
        assert abs(mean - 3.5) < 2.0  # DP noise
        assert std >= 0

    def test_single_party(self) -> None:
        vals = [np.array([10.0, 10.0, 10.0])]
        mean, _ = secure_aggregate_mean(
            vals, clip_low=0, clip_high=20, epsilon=100.0,
        )
        assert abs(mean - 10.0) < 1.0

    def test_empty(self) -> None:
        mean, std = secure_aggregate_mean([], epsilon=1.0)
        assert mean == 0.0
        assert std == 0.0


class TestFederatedFeatureEngineer:
    def test_fit_federated(self, party_dfs: list[pd.DataFrame]) -> None:
        fed = FederatedFeatureEngineer(epsilon=5.0, clip_low=-20, clip_high=30)
        fed.fit_federated(party_dfs)
        assert fed._is_fitted
        assert "x" in fed._means
        assert "y" in fed._means

    def test_transform(self, party_dfs: list[pd.DataFrame]) -> None:
        fed = FederatedFeatureEngineer(epsilon=5.0, clip_low=-20, clip_high=30)
        fed.fit_federated(party_dfs)
        result = fed.transform(party_dfs[0])
        assert result.shape == party_dfs[0].shape
        # Transformed values should be roughly centered around 0
        assert abs(result["x"].mean()) < 5.0

    def test_single_party_fit(self, party_dfs: list[pd.DataFrame]) -> None:
        fed = FederatedFeatureEngineer(epsilon=5.0, clip_low=-20, clip_high=30)
        fed.fit(party_dfs[0])
        assert fed._is_fitted

    def test_not_fitted_raises(self) -> None:
        fed = FederatedFeatureEngineer()
        with pytest.raises(RuntimeError):
            fed.transform(pd.DataFrame({"x": [1]}))

    def test_epsilon_tracking(self, party_dfs: list[pd.DataFrame]) -> None:
        fed = FederatedFeatureEngineer(epsilon=2.0, clip_low=-20, clip_high=30)
        fed.fit_federated(party_dfs)
        assert fed.total_epsilon_spent > 0
        assert fed.total_epsilon_spent <= 2.0 + 1e-6

    def test_aggregation_results(self, party_dfs: list[pd.DataFrame]) -> None:
        fed = FederatedFeatureEngineer(epsilon=5.0, clip_low=-20, clip_high=30)
        fed.fit_federated(party_dfs)
        results = fed.get_aggregation_results()
        assert len(results) == 2
        assert results[0].n_parties == 3

    def test_get_feature_names_out(self, party_dfs: list[pd.DataFrame]) -> None:
        fed = FederatedFeatureEngineer(epsilon=5.0, clip_low=-20, clip_high=30)
        fed.fit_federated(party_dfs)
        names = fed.get_feature_names_out()
        assert "x" in names
        assert "y" in names

    def test_empty_party_list_raises(self) -> None:
        fed = FederatedFeatureEngineer()
        with pytest.raises(ValueError):
            fed.fit_federated([])
