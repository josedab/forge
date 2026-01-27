"""Finance domain feature pack.

Generates technical indicators, risk ratios, and financial metrics
from price/volume/return data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forge.packs.base import FeaturePack


class FinanceFeaturePack(FeaturePack):
    """Generate financial features from price and volume data.

    Detects columns containing price, volume, and return data
    and generates standard technical indicators.

    Args:
        features: Specific features to generate. None for all.
        price_col: Column name for price data. Auto-detected if None.
        volume_col: Column name for volume data. Auto-detected if None.
        windows: Rolling window sizes for indicators.

    Example:
        >>> pack = FinanceFeaturePack(windows=[5, 10, 20])
        >>> features = pack.fit_transform(stock_data)
    """

    domain = "finance"

    def __init__(
        self,
        features: list[str] | None = None,
        price_col: str | None = None,
        volume_col: str | None = None,
        windows: list[int] | None = None,
        prefix: str | None = None,
    ) -> None:
        super().__init__(features=features, prefix=prefix)
        self.price_col = price_col
        self.volume_col = volume_col
        self.windows = windows or [5, 10, 20]
        self._resolved_price: str = ""
        self._resolved_volume: str | None = None

    def available_features(self) -> list[str]:
        return [
            "returns", "log_returns", "volatility", "sma", "ema",
            "rsi", "bollinger_upper", "bollinger_lower", "macd",
            "macd_signal", "vwap", "obv", "momentum", "roc",
            "atr", "drawdown", "max_drawdown", "sharpe_ratio",
            "price_to_sma_ratio", "volume_sma",
        ]

    def _resolve_columns(self, X: pd.DataFrame) -> None:
        if self.price_col and self.price_col in X.columns:
            self._resolved_price = self.price_col
        else:
            price_candidates = ["close", "price", "adj_close", "Close", "Price"]
            for c in price_candidates:
                if c in X.columns:
                    self._resolved_price = c
                    break
            if not self._resolved_price:
                numeric = X.select_dtypes(include=[np.number]).columns
                if len(numeric) > 0:
                    self._resolved_price = numeric[0]

        if self.volume_col and self.volume_col in X.columns:
            self._resolved_volume = self.volume_col
        else:
            vol_candidates = ["volume", "Volume", "vol"]
            for c in vol_candidates:
                if c in X.columns:
                    self._resolved_volume = c
                    break

    def _generate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self._resolve_columns(X)
        if not self._resolved_price:
            return pd.DataFrame(index=X.index)

        price = X[self._resolved_price].astype(float)
        result: dict[str, pd.Series] = {}
        active = set(self.features) if self.features else set(self.available_features())
        p = self.prefix

        # Returns
        if "returns" in active:
            result[f"{p}returns"] = price.pct_change()
        if "log_returns" in active:
            result[f"{p}log_returns"] = np.log(price / price.shift(1))

        for w in self.windows:
            # Simple Moving Average
            if "sma" in active:
                result[f"{p}sma_{w}"] = price.rolling(w, min_periods=1).mean()
            # Exponential Moving Average
            if "ema" in active:
                result[f"{p}ema_{w}"] = price.ewm(span=w, min_periods=1).mean()
            # Volatility
            if "volatility" in active:
                result[f"{p}volatility_{w}"] = price.pct_change().rolling(
                    w, min_periods=1
                ).std()
            # Momentum
            if "momentum" in active:
                result[f"{p}momentum_{w}"] = price - price.shift(w)
            # Rate of change
            if "roc" in active:
                shifted = price.shift(w)
                result[f"{p}roc_{w}"] = (price - shifted) / shifted.replace(0, np.nan)

        # RSI
        if "rsi" in active:
            result[f"{p}rsi_14"] = self._compute_rsi(price, period=14)

        # Bollinger Bands
        if "bollinger_upper" in active or "bollinger_lower" in active:
            sma20 = price.rolling(20, min_periods=1).mean()
            std20 = price.rolling(20, min_periods=1).std()
            if "bollinger_upper" in active:
                result[f"{p}bollinger_upper"] = sma20 + 2 * std20
            if "bollinger_lower" in active:
                result[f"{p}bollinger_lower"] = sma20 - 2 * std20

        # MACD
        if "macd" in active or "macd_signal" in active:
            ema12 = price.ewm(span=12, min_periods=1).mean()
            ema26 = price.ewm(span=26, min_periods=1).mean()
            macd = ema12 - ema26
            if "macd" in active:
                result[f"{p}macd"] = macd
            if "macd_signal" in active:
                result[f"{p}macd_signal"] = macd.ewm(span=9, min_periods=1).mean()

        # Price-to-SMA ratio
        if "price_to_sma_ratio" in active:
            sma = price.rolling(20, min_periods=1).mean()
            result[f"{p}price_to_sma_ratio"] = price / sma.replace(0, np.nan)

        # Drawdown
        if "drawdown" in active or "max_drawdown" in active:
            cummax = price.cummax()
            dd: pd.Series[Any] = (price - cummax) / cummax.replace(0, np.nan)
            if "drawdown" in active:
                result[f"{p}drawdown"] = dd
            if "max_drawdown" in active:
                result[f"{p}max_drawdown"] = dd.cummin()

        # Volume features
        if self._resolved_volume:
            volume = X[self._resolved_volume].astype(float)
            if "vwap" in active:
                cum_vol = volume.cumsum()
                cum_pv: pd.Series[Any] = (price * volume).cumsum()
                result[f"{p}vwap"] = cum_pv / cum_vol.replace(0, np.nan)
            if "obv" in active:
                direction = np.sign(price.diff())
                result[f"{p}obv"] = (volume * direction).cumsum()
            if "volume_sma" in active:
                for w in self.windows:
                    result[f"{p}volume_sma_{w}"] = volume.rolling(
                        w, min_periods=1
                    ).mean()

        return pd.DataFrame(result, index=X.index)

    @staticmethod
    def _compute_rsi(price: pd.Series, period: int = 14) -> pd.Series:
        delta = price.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))  # type: ignore[no-any-return]
