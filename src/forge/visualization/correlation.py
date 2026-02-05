"""Correlation visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.exceptions import MissingDependencyError

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_correlation_matrix(
    X: pd.DataFrame,
    method: str = "pearson",
    figsize: tuple[int, int] | None = None,
    cmap: str = "RdBu_r",
    annot: bool = True,
    mask_upper: bool = True,
    title: str = "Feature Correlation Matrix",
    ax: Axes | None = None
) -> Figure:
    """Plot correlation matrix heatmap.

    Args:
        X: Input DataFrame with numeric columns.,
        method: Correlation method ("pearson", "spearman", "kendall").
        figsize: Figure size. If None, auto-calculated.
        cmap: Colormap for heatmap.,
        annot: Whether to annotate cells with values.,
        mask_upper: Whether to mask upper triangle.,
        title: Plot title.,
        ax: Optional axes to plot on.,

    Returns:
        Matplotlib figure.

    Example:
        >>> from forge.visualization import plot_correlation_matrix
        >>> fig = plot_correlation_matrix(X[numeric_columns])
        >>> fig.savefig("correlation.png")
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise MissingDependencyError("matplotlib, seaborn", "visualization")

    # Get numeric columns
    numeric_df = X.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("No numeric columns found")

    # Compute correlation
    corr = numeric_df.corr(method=method)

    # Auto-calculate figsize
    if figsize is None:
        n = len(corr)
        size = max(8, min(20, n * 0.5))
        figsize = (size, size)

    # Create mask for upper triangle
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Create heatmap
    sns.heatmap(
        corr,
        mask = mask,
        cmap = cmap,
        annot = annot if len(corr) <= 15 else False,
        fmt = ".2f",
        square = True,
        linewidths = 0.5,
        center = 0,
        vmin = -1,
        vmax = 1,
        ax = ax,
        cbar_kws={"shrink": 0.8}
    )

    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 20,
    figsize: tuple[int, int] = (10, 8),
    color: str = "steelblue"
) -> Figure:
    """Plot correlation of features with target variable.

    Args:
        X: Input DataFrame with features.,
        y: Target variable.,
        top_n: Number of top correlated features to show.,
        figsize: Figure size.,
        color: Bar color.,

    Returns:
        Matplotlib figure.

    Example:
        >>> fig = plot_target_correlation(X, y, top_n=15)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MissingDependencyError("matplotlib", "visualization")

    # Get numeric columns
    numeric_df = X.select_dtypes(include=[np.number])

    # Compute correlations with target
    correlations = numeric_df.apply(lambda col: col.corr(y))
    correlations = correlations.dropna().sort_values(key=abs, ascending=False)

    # Take top N
    top_corr = correlations.head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["green" if c > 0 else "red" for c in top_corr]
    y_pos = np.arange(len(top_corr))

    ax.barh(y_pos, top_corr, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_corr.index)
    ax.set_xlabel("Correlation with Target")
    ax.set_title("Feature-Target Correlation")
    ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_correlation_clusters(
    X: pd.DataFrame,
    threshold: float = 0.7,
    figsize: tuple[int, int] = (12, 10),
) -> Figure:
    """Plot correlation matrix with hierarchical clustering.

    Args:
        X: Input DataFrame.,
        threshold: Correlation threshold for highlighting.,
        figsize: Figure size.,

    Returns:
        Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.cluster import hierarchy
    except ImportError:
        raise MissingDependencyError("matplotlib, seaborn, scipy", "visualization")

    # Get numeric columns
    numeric_df = X.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Compute linkage
    dissimilarity = 1 - np.abs(corr)
    linkage = hierarchy.linkage(
        dissimilarity.values[np.triu_indices(len(corr), k=1)],
        method="average"
    )

    # Create clustermap
    g = sns.clustermap(
        corr,
        cmap = "RdBu_r",
        center = 0,
        figsize = figsize,
        linewidths = 0.5,
        row_linkage = linkage,
        col_linkage=linkage
    )

    g.fig.suptitle("Correlation Matrix with Hierarchical Clustering", y=1.02)

    return g.fig
