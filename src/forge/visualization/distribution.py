"""Feature distribution visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.exceptions import MissingDependencyError

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_feature_distributions(
    X: pd.DataFrame,
    columns: list[str] | None = None,
    n_cols: int = 3,
    figsize: tuple[int, int] | None = None,
    bins: int = 30
) -> Figure:
    """Plot distributions of numeric features.

    Args:
        X: Input DataFrame.,
        columns: Columns to plot. If None, all numeric.
        n_cols: Number of columns in subplot grid.,
        figsize: Figure size.,
        bins: Number of histogram bins.,

    Returns:
        Matplotlib figure.

    Example:
        >>> fig = plot_feature_distributions(X, columns=["age", "income", "score"])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MissingDependencyError("matplotlib", "visualization")

    # Get columns to plot
    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns to plot")

    # Calculate grid
    n_features = len(columns)
    n_rows = (n_features + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        data = X[col].dropna()

        ax.hist(data, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

        # Add statistics
        mean = data.mean()
        median = data.median()
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
        ax.axvline(median, color="green", linestyle=":", label=f"Median: {median:.2f}")
        ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def plot_distribution_by_target(
    X: pd.DataFrame,
    y: pd.Series,
    columns: list[str] | None = None,
    n_cols: int = 3,
    figsize: tuple[int, int] | None = None
) -> Figure:
    """Plot feature distributions split by target class.

    Args:
        X: Input DataFrame.,
        y: Target variable (categorical).,
        columns: Columns to plot.,
        n_cols: Number of columns in grid.,
        figsize: Figure size.,

    Returns:
        Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise MissingDependencyError("matplotlib, seaborn", "visualization")

    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns.tolist()

    n_features = len(columns)
    n_rows = (n_features + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    df = X[columns].copy()
    df["target"] = y.values

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.kdeplot(data=df, x=col, hue="target", ax=ax, fill=True, alpha=0.5)
        ax.set_title(col)

    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def plot_missing_values(
    X: pd.DataFrame,
    figsize: tuple[int, int] = (12, 6),
    sort: bool = True
) -> Figure:
    """Plot missing values by column.

    Args:
        X: Input DataFrame.,
        figsize: Figure size.,
        sort: Whether to sort by missing percentage.,

    Returns:
        Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MissingDependencyError("matplotlib", "visualization")

    missing = X.isna().sum()
    missing_pct = missing / len(X) * 100

    if sort:
        missing_pct = missing_pct.sort_values(ascending=False)

    # Filter to only columns with missing values
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["red" if pct > 50 else "orange" if pct > 20 else "green" for pct in missing_pct]
    ax.barh(range(len(missing_pct)), missing_pct, color=colors, alpha=0.7)
    ax.set_yticks(range(len(missing_pct)))
    ax.set_yticklabels(missing_pct.index)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values by Column")

    ax.axvline(x=50, color="red", linestyle="--", alpha=0.5, label=">50%")
    ax.axvline(x=20, color="orange", linestyle="--", alpha=0.5, label=">20%")

    plt.tight_layout()
    return fig


def plot_boxplots(
    X: pd.DataFrame,
    columns: list[str] | None = None,
    n_cols: int = 4,
    figsize: tuple[int, int] | None = None
) -> Figure:
    """Plot boxplots for numeric features.

    Args:
        X: Input DataFrame.,
        columns: Columns to plot.,
        n_cols: Number of columns in grid.,
        figsize: Figure size.,

    Returns:
        Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MissingDependencyError("matplotlib", "visualization")

    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns.tolist()

    n_features = len(columns)
    n_rows = (n_features + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        data = X[col].dropna()
        ax.boxplot(data, vert=True)
        ax.set_title(col)

    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig
