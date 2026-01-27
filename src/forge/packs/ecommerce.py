"""E-commerce domain feature pack.

Generates RFM metrics, session features, cart behavior signals,
and customer lifetime value indicators.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from forge.packs.base import FeaturePack


class EcommerceFeaturePack(FeaturePack):
    """Generate e-commerce features from transaction/event data.

    Auto-detects common e-commerce columns (customer_id, amount,
    timestamp, product) and generates RFM, session, and behavioral features.

    Args:
        features: Specific features to generate. None for all.
        customer_col: Column for customer identifier.
        amount_col: Column for transaction amount.
        timestamp_col: Column for event timestamp.
        product_col: Column for product identifier.

    Example:
        >>> pack = EcommerceFeaturePack(customer_col="user_id")
        >>> features = pack.fit_transform(transactions)
    """

    domain = "ecommerce"

    def __init__(
        self,
        features: list[str] | None = None,
        customer_col: str | None = None,
        amount_col: str | None = None,
        timestamp_col: str | None = None,
        product_col: str | None = None,
        prefix: str | None = None,
    ) -> None:
        super().__init__(features=features, prefix=prefix)
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.timestamp_col = timestamp_col
        self.product_col = product_col

    def available_features(self) -> list[str]:
        return [
            "recency", "frequency", "monetary", "avg_order_value",
            "order_count", "total_spend", "spend_std", "days_since_first",
            "days_since_last", "purchase_frequency", "basket_size",
            "unique_products", "avg_time_between", "is_repeat_customer",
            "spend_trend", "amount_zscore",
        ]

    def _resolve_columns(self, X: pd.DataFrame) -> dict[str, str | None]:
        cols_lower = {c.lower(): c for c in X.columns}
        mapping: dict[str, str | None] = {}

        # Customer ID
        if self.customer_col and self.customer_col in X.columns:
            mapping["customer"] = self.customer_col
        else:
            for c in ["customer_id", "user_id", "cust_id", "customer"]:
                if c in cols_lower:
                    mapping["customer"] = cols_lower[c]
                    break
            else:
                mapping["customer"] = None

        # Amount
        if self.amount_col and self.amount_col in X.columns:
            mapping["amount"] = self.amount_col
        else:
            for c in ["amount", "total", "price", "revenue", "order_total"]:
                if c in cols_lower:
                    mapping["amount"] = cols_lower[c]
                    break
            else:
                mapping["amount"] = None

        # Timestamp
        if self.timestamp_col and self.timestamp_col in X.columns:
            mapping["timestamp"] = self.timestamp_col
        else:
            for c in ["timestamp", "date", "order_date", "created_at", "event_time"]:
                if c in cols_lower:
                    mapping["timestamp"] = cols_lower[c]
                    break
            else:
                mapping["timestamp"] = None

        # Product
        if self.product_col and self.product_col in X.columns:
            mapping["product"] = self.product_col
        else:
            for c in ["product_id", "product", "item_id", "sku"]:
                if c in cols_lower:
                    mapping["product"] = cols_lower[c]
                    break
            else:
                mapping["product"] = None

        return mapping

    def _generate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(X)
        result: dict[str, pd.Series] = {}
        active = set(self.features) if self.features else set(self.available_features())
        p = self.prefix

        cust = cols.get("customer")
        amt = cols.get("amount")
        ts = cols.get("timestamp")
        prod = cols.get("product")

        # Amount-based features (no grouping needed)
        if amt:
            amount = X[amt].astype(float)
            if "amount_zscore" in active:
                std = amount.std()
                if std > 0:
                    result[f"{p}amount_zscore"] = (amount - amount.mean()) / std
                else:
                    result[f"{p}amount_zscore"] = pd.Series(0.0, index=X.index)

        # Customer-grouped features
        if cust and amt:
            amount = X[amt].astype(float)
            if "total_spend" in active:
                result[f"{p}total_spend"] = X.groupby(cust)[amt].transform("sum")
            if "avg_order_value" in active:
                result[f"{p}avg_order_value"] = X.groupby(cust)[amt].transform("mean")
            if "order_count" in active or "frequency" in active:
                result[f"{p}order_count"] = X.groupby(cust)[amt].transform("count")
            if "monetary" in active:
                result[f"{p}monetary"] = X.groupby(cust)[amt].transform("sum")
            if "spend_std" in active:
                result[f"{p}spend_std"] = X.groupby(cust)[amt].transform("std").fillna(0)
            if "is_repeat_customer" in active:
                counts: pd.Series[Any] = X.groupby(cust)[amt].transform("count")
                result[f"{p}is_repeat_customer"] = (counts > 1).astype(int)

        # Timestamp-based features
        if cust and ts:
            try:
                timestamps = pd.to_datetime(X[ts])
                ref_date = timestamps.max()

                if "recency" in active or "days_since_last" in active:
                    last_date = timestamps.groupby(X[cust]).transform("max")
                    result[f"{p}days_since_last"] = (ref_date - last_date).dt.days

                if "days_since_first" in active:
                    first_date = timestamps.groupby(X[cust]).transform("min")
                    result[f"{p}days_since_first"] = (ref_date - first_date).dt.days

                if "frequency" in active and f"{p}order_count" in result:
                    tenure = result.get(f"{p}days_since_first")
                    count = result.get(f"{p}order_count")
                    if tenure is not None and count is not None:
                        safe_tenure = tenure.replace(0, 1)
                        result[f"{p}purchase_frequency"] = count / safe_tenure

            except Exception:
                pass

        # Product diversity
        if cust and prod:
            if "unique_products" in active:
                result[f"{p}unique_products"] = X.groupby(cust)[prod].transform("nunique")
            if "basket_size" in active:
                result[f"{p}basket_size"] = X.groupby(cust)[prod].transform("count")

        return pd.DataFrame(result, index=X.index)
