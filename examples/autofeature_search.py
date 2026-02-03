#!/usr/bin/env python
"""Automatic feature search example for Forge.

This example demonstrates how to use AutoFeatureSearch,
EvolutionarySearch, and RandomSearch to automatically discover
useful feature transformations using cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forge.search import AutoFeatureSearch, EvolutionarySearch, RandomSearch


def create_sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a classification dataset with known feature relationships."""
    np.random.seed(42)
    n = 500

    X = pd.DataFrame({
        "income": np.random.lognormal(10, 1, n),
        "age": np.random.randint(20, 70, n).astype(float),
        "debt": np.random.exponential(5000, n),
        "score": np.random.normal(600, 100, n),
    })

    # Target depends on ratio and log transforms
    prob = 1 / (1 + np.exp(-(
        0.3 * np.log1p(X["income"]) / 3
        - 0.2 * X["debt"] / X["income"]
        + 0.1 * X["score"] / 200
        - 1.5
    )))
    y = pd.Series((np.random.random(n) < prob).astype(int), name="approved")
    return X, y


def example_random_search():
    """Example: Discover features with RandomSearch."""
    print("=" * 60)
    print("Example: RandomSearch Feature Discovery")
    print("=" * 60)

    X, y = create_sample_data()

    search = RandomSearch(
        n_candidates=30,
        scoring="accuracy",
        cv=3,
        random_state=42,
    )
    search.fit(X, y)

    print(f"\nEvaluated {len(search.results_)} valid candidates")
    print(f"Top {len(search.best_features_)} features kept\n")

    print("Top 5 discovered features:")
    for i, result in enumerate(search.best_features_[:5], 1):
        print(f"  {i}. {result.name}  (score={result.score:.4f})")


def example_evolutionary_search():
    """Example: Evolve features with EvolutionarySearch."""
    print("\n" + "=" * 60)
    print("Example: EvolutionarySearch (Genetic Programming)")
    print("=" * 60)

    X, y = create_sample_data()

    search = EvolutionarySearch(
        population_size=20,
        n_generations=5,
        mutation_rate=0.3,
        crossover_rate=0.5,
        scoring="accuracy",
        cv=3,
        random_state=42,
    )
    search.fit(X, y)

    print(f"\nEvolution history:")
    for gen_info in search.history_:
        print(
            f"  Generation {int(gen_info['generation'])}: "
            f"best={gen_info['best_score']:.4f}, "
            f"mean={gen_info['mean_score']:.4f}"
        )

    print(f"\nBest features found:")
    for i, result in enumerate(search.best_features_[:5], 1):
        print(
            f"  {i}. {result.name}  "
            f"(score={result.score:.4f}, gen={result.generation})"
        )


def example_auto_feature_search():
    """Example: sklearn-compatible AutoFeatureSearch transformer."""
    print("\n" + "=" * 60)
    print("Example: AutoFeatureSearch as sklearn Transformer")
    print("=" * 60)

    X, y = create_sample_data()
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train = y.iloc[:train_size]

    search = AutoFeatureSearch(
        method="random",
        n_features=10,
        n_candidates=30,
        scoring="accuracy",
        cv=3,
        random_state=42,
    )

    # Fit and transform (sklearn-compatible)
    X_train_new = search.fit_transform(X_train, y_train)
    X_test_new = search.transform(X_test)

    print(f"\nOriginal features:    {X_train.shape[1]}")
    print(f"Discovered features:  {X_train_new.shape[1]}")
    print(f"Train shape: {X_train_new.shape}")
    print(f"Test shape:  {X_test_new.shape}")

    print(f"\nDiscovered feature names:")
    for name in search.get_feature_names_out():
        print(f"  - {name}")

    results = search.get_search_results()
    print(f"\nSearch results: {len(results)} features selected")


if __name__ == "__main__":
    example_random_search()
    example_evolutionary_search()
    example_auto_feature_search()

    print("\n" + "=" * 60)
    print("All search examples completed!")
    print("=" * 60)
