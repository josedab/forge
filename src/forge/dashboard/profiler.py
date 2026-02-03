"""Dataset profiling for the Forge dashboard.

Provides column-level statistics, type inference summaries, quality scores,
and generator recommendations for interactive exploration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """Statistical profile for a single column."""

    name: str
    dtype: str
    inferred_type: str
    count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    mean: float | None = None
    std: float | None = None
    min_val: Any = None
    max_val: Any = None
    median: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    quality_score: float = 1.0
    quality_issues: list[str] = field(default_factory=list)
    recommended_generators: list[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    """Complete dataset profiling result."""

    n_rows: int
    n_columns: int
    memory_mb: float
    column_profiles: list[ColumnProfile]
    type_summary: dict[str, int]
    overall_quality: float
    duplicate_rows: int
    duplicate_pct: float
    recommendations: list[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert column profiles to a DataFrame for display."""
        records = []
        for cp in self.column_profiles:
            records.append({
                "Column": cp.name,
                "Type": cp.inferred_type,
                "Dtype": cp.dtype,
                "Non-Null": cp.count - cp.null_count,
                "Null %": round(cp.null_pct, 1),
                "Unique": cp.unique_count,
                "Unique %": round(cp.unique_pct, 1),
                "Mean": round(cp.mean, 4) if cp.mean is not None else None,
                "Std": round(cp.std, 4) if cp.std is not None else None,
                "Min": cp.min_val,
                "Max": cp.max_val,
                "Quality": round(cp.quality_score, 2),
                "Issues": len(cp.quality_issues),
                "Generators": ", ".join(cp.recommended_generators),
            })
        return pd.DataFrame(records)


class DatasetProfiler:
    """Profiles a dataset for the interactive dashboard.

    Computes column-level statistics, infers types, assesses quality,
    and recommends appropriate feature generators.

    Args:
        max_categories: Threshold for categorical detection.
        high_null_threshold: Fraction above which null rate is flagged.
        high_correlation_threshold: Correlation above which pairs are flagged.

    Example:
        >>> profiler = DatasetProfiler()
        >>> profile = profiler.profile(df)
        >>> print(profile.overall_quality)
    """

    def __init__(
        self,
        max_categories: int = 50,
        high_null_threshold: float = 0.3,
        high_correlation_threshold: float = 0.95,
    ) -> None:
        self.max_categories = max_categories
        self.high_null_threshold = high_null_threshold
        self.high_correlation_threshold = high_correlation_threshold

    def profile(self, df: pd.DataFrame) -> DatasetProfile:
        """Profile the entire dataset.

        Args:
            df: Input DataFrame to profile.

        Returns:
            DatasetProfile with column-level and dataset-level statistics.
        """
        column_profiles = [self._profile_column(df, col) for col in df.columns]
        type_summary = self._compute_type_summary(column_profiles)

        duplicates = int(df.duplicated().sum())
        n_rows = len(df)
        quality_scores = [cp.quality_score for cp in column_profiles]
        overall_quality = float(np.mean(quality_scores)) if quality_scores else 1.0

        recommendations = self._generate_dataset_recommendations(
            df, column_profiles, type_summary
        )

        return DatasetProfile(
            n_rows=n_rows,
            n_columns=len(df.columns),
            memory_mb=round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            column_profiles=column_profiles,
            type_summary=type_summary,
            overall_quality=round(overall_quality, 3),
            duplicate_rows=duplicates,
            duplicate_pct=round(100.0 * duplicates / max(n_rows, 1), 1),
            recommendations=recommendations,
        )

    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        """Profile a single column."""
        series = df[col]
        n = len(series)
        null_count = int(series.isna().sum())
        null_pct = 100.0 * null_count / max(n, 1)
        unique_count = int(series.nunique())
        unique_pct = 100.0 * unique_count / max(n, 1)

        inferred = self._infer_type(series, unique_count)

        profile = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            inferred_type=inferred,
            count=n,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
        )

        non_null = series.dropna()
        if inferred == "numeric" and len(non_null) > 0:
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric) > 0:
                profile.mean = float(numeric.mean())
                profile.std = float(numeric.std())
                profile.min_val = float(numeric.min())
                profile.max_val = float(numeric.max())
                profile.median = float(numeric.median())
                if len(numeric) > 2:
                    profile.skewness = float(numeric.skew())
                    profile.kurtosis = float(numeric.kurtosis())
        elif inferred == "categorical" and len(non_null) > 0:
            vc = non_null.value_counts().head(10)
            profile.top_values = list(zip(vc.index.tolist(), vc.values.tolist()))

        quality_score, issues = self._assess_quality(profile, inferred)
        profile.quality_score = quality_score
        profile.quality_issues = issues
        profile.recommended_generators = self._recommend_generators(profile, inferred)

        return profile

    def _infer_type(self, series: pd.Series, unique_count: int) -> str:  # type: ignore[type-arg]
        """Infer semantic type of a column."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return "temporal"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            if unique_count <= 2:
                return "boolean"
            return "numeric"
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                avg_len = non_null.astype(str).str.len().mean()
                if avg_len > 50:
                    return "text"
            if unique_count <= self.max_categories:
                return "categorical"
            return "text"
        return "other"

    def _assess_quality(
        self, profile: ColumnProfile, inferred: str
    ) -> tuple[float, list[str]]:
        """Compute quality score and list issues."""
        score = 1.0
        issues: list[str] = []

        if profile.null_pct > self.high_null_threshold * 100:
            score -= 0.3
            issues.append(f"High null rate ({profile.null_pct:.1f}%)")
        elif profile.null_pct > 5:
            score -= 0.1
            issues.append(f"Moderate null rate ({profile.null_pct:.1f}%)")

        if profile.unique_count == 1:
            score -= 0.4
            issues.append("Constant column (zero variance)")
        elif inferred == "numeric" and profile.unique_count <= 3:
            score -= 0.1
            issues.append("Very low cardinality for numeric column")

        if profile.unique_pct > 99 and inferred == "categorical":
            score -= 0.2
            issues.append("Near-unique categorical (likely an ID column)")

        if inferred == "numeric" and profile.skewness is not None:
            if abs(profile.skewness) > 5:
                score -= 0.1
                issues.append(f"Highly skewed (skew={profile.skewness:.2f})")

        return max(0.0, round(score, 3)), issues

    def _recommend_generators(
        self, profile: ColumnProfile, inferred: str
    ) -> list[str]:
        """Recommend feature generators based on column profile."""
        recs: list[str] = []

        if inferred == "numeric":
            recs.append("PolynomialGenerator")
            recs.append("InteractionGenerator")
            if profile.skewness is not None and abs(profile.skewness) > 2:
                recs.append("LogTransform")
            if profile.unique_count > 20:
                recs.append("BinningTransform")

        elif inferred == "categorical":
            recs.append("TargetEncoder")
            if profile.unique_count <= 10:
                recs.append("OneHotEncoder")
            else:
                recs.append("FrequencyEncoder")

        elif inferred == "temporal":
            recs.append("TemporalComponents")
            recs.append("LagGenerator")
            recs.append("RollingWindowGenerator")

        elif inferred == "text":
            recs.append("TFIDFGenerator")
            recs.append("TextStatsGenerator")

        return recs

    def _compute_type_summary(
        self, profiles: list[ColumnProfile]
    ) -> dict[str, int]:
        """Count columns per inferred type."""
        summary: dict[str, int] = {}
        for p in profiles:
            summary[p.inferred_type] = summary.get(p.inferred_type, 0) + 1
        return summary

    def _generate_dataset_recommendations(
        self,
        df: pd.DataFrame,
        profiles: list[ColumnProfile],
        type_summary: dict[str, int],
    ) -> list[str]:
        """Generate dataset-level recommendations."""
        recs: list[str] = []

        high_null_cols = [p.name for p in profiles if p.null_pct > 30]
        if high_null_cols:
            recs.append(
                f"Consider imputation for {len(high_null_cols)} high-null columns: "
                f"{', '.join(high_null_cols[:5])}"
            )

        constant_cols = [p.name for p in profiles if p.unique_count == 1]
        if constant_cols:
            recs.append(
                f"Drop {len(constant_cols)} constant column(s): "
                f"{', '.join(constant_cols[:5])}"
            )

        numeric_count = type_summary.get("numeric", 0)
        if numeric_count >= 3:
            recs.append(
                f"With {numeric_count} numeric columns, consider InteractionGenerator "
                "for pairwise feature interactions"
            )

        cat_count = type_summary.get("categorical", 0)
        if cat_count >= 2:
            recs.append(
                f"With {cat_count} categorical columns, consider target encoding "
                "or frequency encoding"
            )

        temporal_count = type_summary.get("temporal", 0)
        if temporal_count >= 1:
            recs.append(
                "Temporal columns detected — extract date/time components and "
                "consider lag/rolling features"
            )

        return recs


def create_profiling_dashboard(
    df: pd.DataFrame,
    title: str = "Forge Dataset Profiler",
    profiler: DatasetProfiler | None = None,
) -> Any:
    """Create a Streamlit profiling dashboard for a dataset.

    Args:
        df: DataFrame to profile.
        title: Dashboard title.
        profiler: Optional pre-configured profiler instance.

    Returns:
        A callable Streamlit app function.

    Raises:
        MissingDependencyError: If streamlit is not installed.
    """
    from forge.exceptions import MissingDependencyError

    try:
        import streamlit
    except ImportError:
        raise MissingDependencyError("streamlit", "profiling dashboard")

    if profiler is None:
        profiler = DatasetProfiler()

    profile = profiler.profile(df)

    def run_profiling_dashboard() -> None:
        """Render the profiling dashboard."""
        streamlit.set_page_config(page_title=title, layout="wide")
        streamlit.title(title)

        # Dataset overview
        streamlit.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = streamlit.columns(4)
        col1.metric("Rows", f"{profile.n_rows:,}")
        col2.metric("Columns", profile.n_columns)
        col3.metric("Memory", f"{profile.memory_mb:.1f} MB")
        col4.metric("Quality", f"{profile.overall_quality:.0%}")

        col5, col6 = streamlit.columns(2)
        col5.metric("Duplicate Rows", f"{profile.duplicate_rows:,}")
        col6.metric("Duplicate %", f"{profile.duplicate_pct:.1f}%")

        # Type distribution
        streamlit.subheader("🏷️ Column Types")
        type_df = pd.DataFrame(
            list(profile.type_summary.items()), columns=["Type", "Count"]
        )
        streamlit.bar_chart(type_df.set_index("Type"))

        # Column details table
        streamlit.subheader("📋 Column Profiles")
        streamlit.dataframe(profile.to_dataframe(), use_container_width=True)

        # Quality heatmap
        streamlit.subheader("🔍 Quality Assessment")
        quality_data = {
            cp.name: cp.quality_score for cp in profile.column_profiles
        }
        quality_df = pd.DataFrame(
            list(quality_data.items()), columns=["Column", "Quality Score"]
        )
        streamlit.bar_chart(quality_df.set_index("Column"))

        issues_list = []
        for cp in profile.column_profiles:
            for issue in cp.quality_issues:
                issues_list.append({"Column": cp.name, "Issue": issue})
        if issues_list:
            streamlit.subheader("⚠️ Quality Issues")
            streamlit.table(pd.DataFrame(issues_list))

        # Recommendations
        if profile.recommendations:
            streamlit.subheader("💡 Recommendations")
            for rec in profile.recommendations:
                streamlit.info(rec)

        # Column deep-dive
        streamlit.subheader("🔬 Column Explorer")
        selected_col = streamlit.selectbox(
            "Select a column", [cp.name for cp in profile.column_profiles]
        )
        if selected_col:
            cp = next(p for p in profile.column_profiles if p.name == selected_col)
            c1, c2, c3 = streamlit.columns(3)
            c1.write(f"**Type**: {cp.inferred_type}")
            c2.write(f"**Null %**: {cp.null_pct:.1f}%")
            c3.write(f"**Unique**: {cp.unique_count}")

            if cp.mean is not None:
                c4, c5, c6 = streamlit.columns(3)
                c4.write(f"**Mean**: {cp.mean:.4f}")
                c5.write(f"**Std**: {cp.std:.4f}" if cp.std else "**Std**: N/A")
                c6.write(
                    f"**Skewness**: {cp.skewness:.2f}" if cp.skewness else "**Skew**: N/A"
                )

            if cp.top_values:
                streamlit.write("**Top Values:**")
                tv_df = pd.DataFrame(cp.top_values, columns=["Value", "Count"])
                streamlit.dataframe(tv_df, use_container_width=True)

            if cp.recommended_generators:
                streamlit.write("**Recommended Generators:**")
                for gen in cp.recommended_generators:
                    streamlit.write(f"  ✅ {gen}")

        # Export pipeline code
        streamlit.subheader("📝 Generated Pipeline Code")
        code = _generate_pipeline_code(profile)
        streamlit.code(code, language="python")

    return run_profiling_dashboard


def _generate_pipeline_code(profile: DatasetProfile) -> str:
    """Generate Python code for a Forge pipeline based on the profile."""
    lines = [
        "from forge import AutoFeatureTransformer",
        "from forge.transformers import ForgePipeline",
        "",
        "# Auto-generated pipeline based on dataset profile",
    ]

    numeric_cols = [
        cp.name for cp in profile.column_profiles if cp.inferred_type == "numeric"
    ]
    cat_cols = [
        cp.name for cp in profile.column_profiles if cp.inferred_type == "categorical"
    ]
    temporal_cols = [
        cp.name for cp in profile.column_profiles if cp.inferred_type == "temporal"
    ]

    lines.append("steps = []")
    if numeric_cols:
        lines.append("")
        lines.append(f"# Numeric columns: {numeric_cols[:5]}")
        lines.append("from forge.generators.numeric import InteractionGenerator, PolynomialGenerator")
        lines.append(
            f"steps.append(('interactions', InteractionGenerator(columns={numeric_cols[:5]})))"
        )

    if cat_cols:
        lines.append("")
        lines.append(f"# Categorical columns: {cat_cols[:5]}")
        lines.append("from forge.generators.categorical import TargetEncoder")
        lines.append(
            f"steps.append(('encoding', TargetEncoder(columns={cat_cols[:5]})))"
        )

    if temporal_cols:
        lines.append("")
        lines.append(f"# Temporal columns: {temporal_cols[:5]}")
        lines.append("from forge.generators.temporal import TemporalComponentGenerator")
        lines.append(
            f"steps.append(('temporal', TemporalComponentGenerator(columns={temporal_cols[:5]})))"
        )

    lines.append("")
    lines.append("pipeline = ForgePipeline(steps)")
    lines.append("X_features = pipeline.fit_transform(X, y)")

    return "\n".join(lines)
