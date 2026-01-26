"""Streamlit-based visual feature engineering studio.

Provides a web-based interactive interface for building feature
pipelines, visualizing data, and exporting code.

Launch with: `streamlit run -m forge.studio.app` or
             `python -m forge.studio.app`

Example:
    >>> from forge.studio.app import create_app_config, launch_studio
    >>> launch_studio(data=my_dataframe, port=8501)
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Configuration for the visual studio application.

    Attributes:
        title: Application title.
        max_preview_rows: Maximum rows to show in preview.
        max_columns_display: Maximum columns to show in summaries.
        theme: UI theme ('light' or 'dark').
        port: Server port number.
    """

    title: str = "Forge Feature Studio"
    max_preview_rows: int = 100
    max_columns_display: int = 50
    theme: str = "light"
    port: int = 8501


@dataclass
class ColumnAnalysis:
    """Analysis of a single column for the UI."""

    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for display."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "null_pct": f"{self.null_pct:.1%}",
            "unique": self.unique_count,
            **{k: round(v, 4) for k, v in self.stats.items()},
        }


@dataclass
class PipelineConfig:
    """Serializable pipeline configuration from the UI.

    Captures user selections for code export.
    """

    steps: list[dict[str, Any]] = field(default_factory=list)
    target_column: str | None = None
    max_features: int | None = None
    selection_method: str = "importance"

    def to_code(self) -> str:
        """Export as executable Python code."""
        lines = [
            "import pandas as pd",
            "from forge import AutoFeatureTransformer",
            "from forge.transformers import ForgePipeline",
            "",
            "# Load data",
            '# X = pd.read_csv("your_data.csv")',
            '# y = X.pop("target")',
            "",
        ]

        if not self.steps:
            lines.extend([
                "# Auto feature engineering",
                "transformer = AutoFeatureTransformer(",
                f"    max_features={self.max_features},",
                f'    selection_method="{self.selection_method}",',
                ")",
                "X_engineered = transformer.fit_transform(X, y)",
            ])
        else:
            lines.append("# Custom pipeline")
            lines.append("pipeline = ForgePipeline([")
            for step in self.steps:
                name = step.get("name", "step")
                transformer_code = step.get("code", "None")
                lines.append(f'    ("{name}", {transformer_code}),')
            lines.append("])")
            lines.append("X_engineered = pipeline.fit_transform(X, y)")

        lines.extend([
            "",
            "print(f'Generated {X_engineered.shape[1]} features')",
        ])

        return "\n".join(lines)


def analyze_dataframe(df: pd.DataFrame) -> list[ColumnAnalysis]:
    """Analyze all columns in a DataFrame for the studio UI.

    Args:
        df: Input DataFrame.

    Returns:
        List of ColumnAnalysis objects, one per column.
    """
    analyses = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = null_count / len(series) if len(series) > 0 else 0.0

        # Determine dtype category
        if pd.api.types.is_numeric_dtype(series):
            dtype_cat = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            dtype_cat = "datetime"
        elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            dtype_cat = "categorical"
        else:
            dtype_cat = str(series.dtype)

        stats: dict[str, float] = {}
        if dtype_cat == "numeric":
            desc = series.describe()
            stats = {
                "mean": float(desc.get("mean", 0)),
                "std": float(desc.get("std", 0)),
                "min": float(desc.get("min", 0)),
                "max": float(desc.get("max", 0)),
            }

        sample = series.dropna().head(5).tolist()

        analyses.append(ColumnAnalysis(
            name=col,
            dtype=dtype_cat,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=int(series.nunique()),
            sample_values=sample,
            stats=stats,
        ))

    return analyses


def generate_distribution_summary(
    df: pd.DataFrame, column: str, bins: int = 20
) -> dict[str, Any]:
    """Generate distribution summary for a column.

    Args:
        df: Input DataFrame.
        column: Column name.
        bins: Number of histogram bins.

    Returns:
        Dictionary with distribution data for charting.
    """
    series = df[column].dropna()

    if pd.api.types.is_numeric_dtype(series):
        counts, edges = np.histogram(series, bins=bins)
        return {
            "type": "histogram",
            "counts": counts.tolist(),
            "edges": edges.tolist(),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "skew": float(series.skew()),
        }
    else:
        value_counts = series.value_counts().head(bins)
        return {
            "type": "bar",
            "labels": value_counts.index.tolist(),
            "counts": value_counts.values.tolist(),
            "unique_count": int(series.nunique()),
        }


def compare_dataframes(
    before: pd.DataFrame, after: pd.DataFrame
) -> dict[str, Any]:
    """Compare two DataFrames for A/B pipeline comparison.

    Args:
        before: Original DataFrame.
        after: Transformed DataFrame.

    Returns:
        Comparison summary dictionary.
    """
    new_cols = set(after.columns) - set(before.columns)
    removed_cols = set(before.columns) - set(after.columns)
    common_cols = set(before.columns) & set(after.columns)

    stat_changes: dict[str, dict[str, float]] = {}
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(before[col]) and pd.api.types.is_numeric_dtype(after[col]):
            before_mean = float(before[col].mean())
            after_mean = float(after[col].mean())
            if before_mean != 0:
                pct_change = (after_mean - before_mean) / abs(before_mean) * 100
            else:
                pct_change = 0.0
            stat_changes[col] = {
                "before_mean": before_mean,
                "after_mean": after_mean,
                "pct_change": pct_change,
            }

    return {
        "columns_before": len(before.columns),
        "columns_after": len(after.columns),
        "new_columns": sorted(new_cols),
        "removed_columns": sorted(removed_cols),
        "rows_before": len(before),
        "rows_after": len(after),
        "stat_changes": stat_changes,
    }


def create_streamlit_app_code(config: AppConfig | None = None) -> str:
    """Generate the Streamlit application code.

    This generates the full Streamlit app as a string that can be
    written to a file and executed with `streamlit run`.

    Args:
        config: Application configuration.

    Returns:
        Complete Streamlit application code as a string.
    """
    config = config or AppConfig()

    return textwrap.dedent(f'''\
    """Forge Feature Studio - Interactive Feature Engineering UI."""

    import streamlit as st
    import pandas as pd
    import numpy as np

    st.set_page_config(
        page_title="{config.title}",
        page_icon="🔨",
        layout="wide",
    )

    st.title("🔨 {config.title}")

    # Sidebar - Data Loading
    st.sidebar.header("📁 Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {{len(df)}} rows, {{len(df.columns)}} columns")

        # Column Analysis Tab
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Data Overview", "⚙️ Pipeline Builder", "📈 Comparison", "💻 Export Code"]
        )

        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df.head({config.max_preview_rows}))

            st.subheader("Column Analysis")
            from forge.studio.app import analyze_dataframe
            analyses = analyze_dataframe(df)
            analysis_df = pd.DataFrame([a.to_dict() for a in analyses])
            st.dataframe(analysis_df, use_container_width=True)

        with tab2:
            st.subheader("Feature Pipeline Builder")
            target_col = st.selectbox("Target Column", [None] + list(df.columns))
            max_features = st.slider("Max Features", 10, 500, 100)
            selection_method = st.selectbox(
                "Selection Method", ["importance", "correlation", "statistical"]
            )

            if st.button("🚀 Run Feature Engineering"):
                from forge import AutoFeatureTransformer
                y = df[target_col] if target_col else None
                X = df.drop(columns=[target_col]) if target_col else df
                transformer = AutoFeatureTransformer(
                    max_features=max_features,
                    selection_method=selection_method,
                )
                X_transformed = transformer.fit_transform(X, y)
                st.session_state["X_before"] = X
                st.session_state["X_after"] = X_transformed
                st.success(f"Generated {{X_transformed.shape[1]}} features")
                st.dataframe(X_transformed.head(20))

        with tab3:
            st.subheader("Before/After Comparison")
            if "X_before" in st.session_state and "X_after" in st.session_state:
                from forge.studio.app import compare_dataframes
                comparison = compare_dataframes(
                    st.session_state["X_before"],
                    st.session_state["X_after"],
                )
                col1, col2, col3 = st.columns(3)
                col1.metric("Columns Before", comparison["columns_before"])
                col2.metric("Columns After", comparison["columns_after"])
                col3.metric("New Features", len(comparison["new_columns"]))

                if comparison["new_columns"]:
                    st.write("**New columns:**", ", ".join(comparison["new_columns"][:20]))
            else:
                st.info("Run the pipeline builder first to see comparisons.")

        with tab4:
            st.subheader("Export Python Code")
            from forge.studio.app import PipelineConfig
            pipeline_config = PipelineConfig(
                target_column=target_col if "target_col" in dir() else None,
                max_features=max_features if "max_features" in dir() else 100,
                selection_method=selection_method if "selection_method" in dir() else "importance",
            )
            code = pipeline_config.to_code()
            st.code(code, language="python")
            st.download_button("📥 Download Script", code, "forge_pipeline.py", "text/python")
    else:
        st.info("👈 Upload a CSV file to get started")

        st.markdown("""
        ### Quick Start
        1. Upload your dataset (CSV format)
        2. Explore column types and statistics
        3. Configure your feature pipeline
        4. Compare before/after results
        5. Export the pipeline as Python code
        """)
    ''')


def launch_studio(
    data: pd.DataFrame | None = None,
    config: AppConfig | None = None,
) -> str:
    """Generate and return the Streamlit app code.

    To actually launch the app, write the returned code to a file
    and run: `streamlit run <file.py>`

    Args:
        data: Optional pre-loaded DataFrame.
        config: Application configuration.

    Returns:
        The Streamlit application code as a string.
    """
    config = config or AppConfig()
    return create_streamlit_app_code(config)
