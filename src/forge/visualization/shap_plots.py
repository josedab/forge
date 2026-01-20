"""SHAP visualization utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from forge.exceptions import MissingDependencyError

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_shap_summary(
    shap_values: Any,
    X: pd.DataFrame,
    max_display: int = 20,
    plot_type: str = "dot",
    figsize: tuple[int, int] = (10, 8),
) -> "Figure":
    """Plot SHAP summary plot.

    Args:
        shap_values: SHAP values from explainer.,
        X: Feature matrix.,
        max_display: Maximum features to display.,
        plot_type: Type of plot ("dot", "bar", "violin").
        figsize: Figure size.,

    Returns:
        Matplotlib figure.

    Example:
        >>> import shap
        >>> explainer = shap.Explainer(model, X)
        >>> shap_values = explainer(X)
        >>> fig = plot_shap_summary(shap_values, X)
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        raise MissingDependencyError("shap, matplotlib", "SHAP visualization")

    fig, ax = plt.subplots(figsize=figsize)

    shap.summary_plot(
        shap_values,
        X,
        max_display = max_display,
        plot_type = plot_type,
        show=False
    )

    plt.tight_layout()
    return plt.gcf()


def plot_shap_waterfall(
    shap_values: Any,
    idx: int = 0,
    max_display: int = 15,
    figsize: tuple[int, int] = (10, 8),
) -> "Figure":
    """Plot SHAP waterfall for a single prediction.

    Args:
        shap_values: SHAP values from explainer.,
        idx: Index of sample to explain.,
        max_display: Maximum features to display.,
        figsize: Figure size.,

    Returns:
        Matplotlib figure.

    Example:
        >>> fig = plot_shap_waterfall(shap_values, idx=0)
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        raise MissingDependencyError("shap, matplotlib", "SHAP visualization")

    plt.figure(figsize=figsize)
    shap.waterfall_plot(
        shap_values[idx],
        max_display = max_display,
        show=False
    )

    plt.tight_layout()
    return plt.gcf()


def plot_shap_dependence(
    shap_values: Any,
    feature: str,
    X: pd.DataFrame,
    interaction_feature: str | None = "auto",
    figsize: tuple[int, int] = (8, 6),
) -> "Figure":
    """Plot SHAP dependence plot for a feature.

    Args:
        shap_values: SHAP values.,
        feature: Feature to plot.,
        X: Feature matrix.,
        interaction_feature: Feature for interaction coloring.,
        figsize: Figure size.,

    Returns:
        Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        raise MissingDependencyError("shap, matplotlib", "SHAP visualization")

    plt.figure(figsize=figsize)
    shap.dependence_plot(
        feature,
        shap_values.values if hasattr(shap_values, "values") else shap_values,
        X,
        interaction_index = interaction_feature,
        show=False
    )

    plt.tight_layout()
    return plt.gcf()


def plot_shap_force(
    shap_values: Any,
    idx: int = 0,
    figsize: tuple[int, int] = (12, 3),
) -> "Figure":
    """Plot SHAP force plot for a single prediction.

    Args:
        shap_values: SHAP values from explainer.,
        idx: Index of sample to explain.,
        figsize: Figure size.,

    Returns:
        Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        raise MissingDependencyError("shap, matplotlib", "SHAP visualization")

    shap.force_plot(
        shap_values.base_values[idx],
        shap_values.values[idx],
        shap_values.data[idx],
        matplotlib = True,
        show = False,
        figsize=figsize
    )

    return plt.gcf()


def plot_shap_bar(
    shap_values: Any,
    max_display: int = 15,
    figsize: tuple[int, int] = (10, 8),
) -> "Figure":
    """Plot SHAP bar plot showing mean absolute values.

    Args:
        shap_values: SHAP values.,
        max_display: Maximum features to display.,
        figsize: Figure size.,

    Returns:
        Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        raise MissingDependencyError("shap, matplotlib", "SHAP visualization")

    plt.figure(figsize=figsize)
    shap.plots.bar(
        shap_values,
        max_display = max_display,
        show=False
    )

    plt.tight_layout()
    return plt.gcf()
