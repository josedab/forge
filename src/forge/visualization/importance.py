"""Feature importance visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from forge.exceptions import MissingDependencyError

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_feature_importance(
    importance_df: pd.DataFrame | None = None,
    transformer: Any | None = None,
    top_n: int = 20,
    figsize: tuple[int, int] = (10, 8),
    color: str = "steelblue",
    title: str = "Feature Importance",
    ax: "Axes | None" = None
) -> "Figure":
    """Plot feature importance as a horizontal bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns.,
        transformer: Fitted transformer with get_feature_importance method.,
        top_n: Number of top features to show.,
        figsize: Figure size.,
        color: Bar color.,
        title: Plot title.,
        ax: Optional axes to plot on.,

    Returns:
        Matplotlib figure.

    Example:
        >>> from forge.visualization import plot_feature_importance
        >>> fig = plot_feature_importance(transformer=fitted_transformer, top_n=15)
        >>> fig.savefig("importance.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MissingDependencyError("matplotlib", "visualization")

    # Get importance DataFrame
    if importance_df is None:
        if transformer is None:
            raise ValueError("Either importance_df or transformer must be provided")
        if not hasattr(transformer, "get_feature_importance"):
            raise ValueError("Transformer must have get_feature_importance method")
        importance_df = transformer.get_feature_importance()

    if importance_df.empty:
        raise ValueError("No importance data available")

    # Prepare data
    df = importance_df.head(top_n).copy()
    df = df.sort_values("importance", ascending=True)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Create horizontal bar chart
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["importance"], color=color, alpha=0.8)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Importance")
    ax.set_title(title)

    # Add value labels
    for i, v in enumerate(df["importance"]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    return fig


def plot_importance_comparison(
    importance_dfs: dict[str, pd.DataFrame],
    top_n: int = 15,
    figsize: tuple[int, int] = (12, 8),
) -> "Figure":
    """Plot feature importance comparison across multiple methods.

    Args:
        importance_dfs: Dictionary mapping method names to importance DataFrames.,
        top_n: Number of top features to show.,
        figsize: Figure size.,

    Returns:
        Matplotlib figure.

    Example:
        >>> importances = {
        ...     "Random Forest": rf_importance,
        ...     "SHAP": shap_importance,
        ... }
        >>> fig = plot_importance_comparison(importances)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MissingDependencyError("matplotlib", "visualization")

    n_methods = len(importance_dfs)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True)

    if n_methods == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    for i, (method, df) in enumerate(importance_dfs.items()):
        df_top = df.head(top_n).sort_values("importance", ascending=True)
        y_pos = np.arange(len(df_top))

        axes[i].barh(y_pos, df_top["importance"], color=colors[i], alpha=0.8)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(df_top["feature"])
        axes[i].set_title(method)
        axes[i].set_xlabel("Importance")

    plt.tight_layout()
    return fig


def plot_cumulative_importance(
    importance_df: pd.DataFrame,
    threshold: float = 0.95,
    figsize: tuple[int, int] = (10, 6),
) -> "Figure":
    """Plot cumulative feature importance.

    Shows how many features are needed to reach a given importance threshold.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns.,
        threshold: Importance threshold to highlight.,
        figsize: Figure size.,

    Returns:
        Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MissingDependencyError("matplotlib", "visualization")

    # Normalize and compute cumulative
    df = importance_df.copy()
    df["importance_normalized"] = df["importance"] / df["importance"].sum()
    df = df.sort_values("importance", ascending=False)
    df["cumulative"] = df["importance_normalized"].cumsum()

    # Find threshold crossing
    n_features_threshold = (df["cumulative"] <= threshold).sum() + 1

    fig, ax = plt.subplots(figsize=figsize)

    x = range(1, len(df) + 1)
    ax.plot(x, df["cumulative"], "b-", linewidth=2, label="Cumulative Importance")
    ax.axhline(y=threshold, color="r", linestyle="--", label=f"{threshold:.0%} threshold")
    ax.axvline(x=n_features_threshold, color="g", linestyle=":", alpha=0.7)

    ax.fill_between(x, 0, df["cumulative"], alpha=0.3)

    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Cumulative Importance")
    ax.set_title("Cumulative Feature Importance")
    ax.legend()

    ax.annotate(
        f"{n_features_threshold} features",
        xy = (n_features_threshold, threshold),
        xytext = (n_features_threshold + 5, threshold - 0.1),
        arrowprops=dict(arrowstyle="->", color="gray")
    )

    plt.tight_layout()
    return fig
