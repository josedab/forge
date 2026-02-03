#!/usr/bin/env python
"""Feature catalog example for Forge.

This example demonstrates CatalogStore and FeatureCatalogStore
for documenting, searching, tagging, and managing feature metadata.
"""

from __future__ import annotations

from forge.catalog import CatalogEntry, CatalogStore, FeatureCatalogStore


def example_catalog_crud():
    """Example: Create, read, update, delete catalog entries."""
    print("=" * 60)
    print("Example: Feature Catalog CRUD Operations")
    print("=" * 60)

    store = CatalogStore()

    # Add features to the catalog
    features = [
        CatalogEntry(name="age", description="Customer age in years", dtype="int64",
                      source_columns=["birth_date"], transformation="current_date - birth_date",
                      owner="ml-team", tags={"domain": "customer", "type": "demographic"}),
        CatalogEntry(name="log_income", description="Log-transformed annual income",
                      dtype="float64", source_columns=["income"], transformation="log1p(income)",
                      owner="ml-team", tags={"domain": "financial", "type": "numeric"}),
        CatalogEntry(name="purchase_frequency", description="Average purchases per month",
                      dtype="float64", source_columns=["order_count", "account_age_months"],
                      transformation="order_count / account_age_months", owner="data-eng",
                      tags={"domain": "ecommerce", "type": "behavioral"}),
        CatalogEntry(name="credit_risk_score", description="Composite credit risk indicator",
                      dtype="float64", source_columns=["credit_score", "debt_ratio"],
                      transformation="weighted_combination", owner="risk-team",
                      tags={"domain": "financial", "type": "risk"}),
    ]

    for entry in features:
        store.add(entry)
    print(f"\nAdded {store.size} features to catalog")

    # Read a feature
    entry = store.get("log_income")
    print(f"\nRetrieved: {entry.name}")
    print(f"  Description: {entry.description}")
    print(f"  Owner: {entry.owner}")
    print(f"  Source columns: {entry.source_columns}")

    # List all
    all_entries = store.list_all()
    print(f"\nAll features ({len(all_entries)}):")
    for e in all_entries:
        print(f"  - {e.name} ({e.owner})")


def example_search_and_tags():
    """Example: Search features and manage tags."""
    print("\n" + "=" * 60)
    print("Example: Search and Tag Features")
    print("=" * 60)

    store = CatalogStore()
    entries = [
        CatalogEntry(name="price_sma_20", description="20-day simple moving average of price",
                      owner="quant-team", tags={"domain": "finance", "type": "technical"}),
        CatalogEntry(name="price_volatility", description="Price volatility over 30 days",
                      owner="quant-team", tags={"domain": "finance", "type": "risk"}),
        CatalogEntry(name="customer_ltv", description="Customer lifetime value estimate",
                      owner="ml-team", tags={"domain": "ecommerce", "type": "monetary"}),
        CatalogEntry(name="bmi", description="Body mass index from height and weight",
                      owner="health-team", tags={"domain": "healthcare", "type": "clinical"}),
        CatalogEntry(name="avg_session_duration", description="Average browsing session length",
                      owner="ml-team", tags={"domain": "ecommerce", "type": "behavioral"}),
    ]
    for e in entries:
        store.add(e)

    # Full-text search
    results = store.search("price")
    print(f"\nSearch 'price': {len(results)} results")
    for r in results:
        print(f"  - {r.name}: {r.description}")

    # Search by owner
    results = store.search(owner="quant-team")
    print(f"\nFeatures owned by 'quant-team': {len(results)}")
    for r in results:
        print(f"  - {r.name}")

    # Search by tags
    results = store.search(tags={"domain": "ecommerce"})
    print(f"\nFeatures with domain=ecommerce: {len(results)}")
    for r in results:
        print(f"  - {r.name}")

    # Add tags to a feature
    store.tag("bmi", {"priority": "high", "validated": "true"})
    updated = store.get("bmi")
    print(f"\nUpdated tags for 'bmi': {updated.tags}")

    # Deprecate a feature
    store.deprecate("avg_session_duration", reason="Replaced by median_session_duration")
    deprecated = store.get("avg_session_duration")
    print(f"\nDeprecated '{deprecated.name}': reason='{deprecated.deprecation_reason}'")

    # Search excluding deprecated
    active = store.search(include_deprecated=False)
    print(f"\nActive features: {len(active)}")

    # Catalog statistics
    stats = store.get_statistics()
    print(f"\nCatalog stats: {stats}")


def example_catalog_from_transformer():
    """Example: Populate catalog from transformer feature names."""
    print("\n" + "=" * 60)
    print("Example: FeatureCatalogStore from Transformer")
    print("=" * 60)

    store = FeatureCatalogStore()

    # Simulate a fitted transformer with feature names
    class MockTransformer:
        def get_feature_names_out(self):
            return ["log_price", "sqrt_quantity", "price_x_quantity",
                    "category_encoded", "price_zscore"]

    transformer = MockTransformer()
    count = store.add_from_transformer(
        transformer, owner="pipeline-v2", tags={"pipeline": "v2"}
    )
    print(f"\nAdded {count} features from transformer")

    results = store.search("price")
    print(f"\nSearch 'price' in catalog: {len(results)} results")
    for r in results:
        print(f"  - {r.name} (owner={r.owner}, tags={r.tags})")


if __name__ == "__main__":
    example_catalog_crud()
    example_search_and_tags()
    example_catalog_from_transformer()

    print("\n" + "=" * 60)
    print("All catalog examples completed!")
    print("=" * 60)
