"""Streamlit dashboard application for interactive lineage exploration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from forge.exceptions import MissingDependencyError

if TYPE_CHECKING:
    from forge.dashboard.lineage import FeatureLineageGraph

logger = logging.getLogger(__name__)


def create_dashboard(
    graph: FeatureLineageGraph,
    title: str = "Forge Feature Lineage",
    port: int = 8501,
    host: str = "localhost",
) -> Any:
    """Create and return a Streamlit dashboard app for the lineage graph.

    This returns a callable that can be run with Streamlit.
    If Streamlit is not installed, raises MissingDependencyError.

    Args:
        graph: Feature lineage graph to visualize.
        title: Dashboard title.
        port: Port to serve on.
        host: Host to bind to.

    Returns:
        A callable dashboard function.

    Example:
        >>> from forge.dashboard import create_dashboard, FeatureLineageGraph
        >>> graph = FeatureLineageGraph.from_transformer(transformer)
        >>> app = create_dashboard(graph)
        >>> # Run with: streamlit run <script>
    """
    try:
        import streamlit
    except ImportError:
        raise MissingDependencyError("streamlit", "interactive lineage dashboard")

    def run_dashboard() -> None:
        """Render the Streamlit dashboard."""
        streamlit.set_page_config(page_title=title, layout="wide")
        streamlit.title(title)

        # Sidebar: filters
        streamlit.sidebar.header("Filters")
        node_types = list({n.node_type for n in graph.nodes})
        selected_types: list[str] = streamlit.sidebar.multiselect(
            "Node types", node_types, default=node_types
        )

        # Summary
        streamlit.subheader("Summary")
        streamlit.text(graph.summary())

        # Feature table
        streamlit.subheader("Feature Details")
        df = graph.to_dataframe()
        filtered_df = df[df["type"].isin(selected_types)]
        streamlit.dataframe(filtered_df, use_container_width=True)

        # Importance chart
        ranking = graph.get_importance_ranking()
        if ranking:
            streamlit.subheader("Feature Importance")
            import pandas as pd

            imp_df = pd.DataFrame(ranking, columns=["Feature", "Importance"])
            imp_df = imp_df.head(20)
            streamlit.bar_chart(imp_df.set_index("Feature"))

        # Lineage explorer
        streamlit.subheader("Lineage Explorer")
        feature_names = [n.name for n in graph.nodes]
        selected: str | None = streamlit.selectbox("Select a feature", feature_names)
        if selected:
            sources = graph.get_sources(selected)
            derived = graph.get_derived(selected)
            col1, col2 = streamlit.columns(2)
            with col1:
                streamlit.write("**Source columns:**")
                for s in sources:
                    streamlit.write(f"  - {s}")
            with col2:
                streamlit.write("**Derived features:**")
                for d in derived:
                    streamlit.write(f"  - {d}")

    return run_dashboard
