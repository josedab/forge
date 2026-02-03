"""Interactive notebook experience for feature engineering.

Provides column profiling, pipeline building, and code export
utilities designed for use in Jupyter notebooks. Works as pure
Python (no widget dependencies required) with optional ipywidgets
integration when available.

Example:
    >>> from forge.notebook import ColumnProfiler, PipelineBuilder
    >>> profiler = ColumnProfiler(df)
    >>> profiler.profile("user_age")
    >>> builder = PipelineBuilder(df, y)
    >>> builder.add_step("numeric", columns=["age", "income"])
    >>> code = builder.export_code()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from IPython.display import HTML, display
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""

    name: str
    dtype: str
    n_rows: int
    n_null: int
    null_pct: float
    n_unique: int
    inferred_type: str  # "numeric", "categorical", "temporal", "text", "boolean"
    stats: dict[str, Any] = field(default_factory=dict)
    sample_values: list[Any] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "n_rows": self.n_rows,
            "n_null": self.n_null,
            "null_pct": round(self.null_pct, 4),
            "n_unique": self.n_unique,
            "inferred_type": self.inferred_type,
            "stats": self.stats,
            "recommendations": self.recommendations,
        }

    def to_html(self) -> str:
        """Render as an HTML card for Jupyter display."""
        stats_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in self.stats.items()
        )
        rec_items = "".join(f"<li>{r}</li>" for r in self.recommendations)
        return f"""
        <div style="border:1px solid #ddd; border-radius:8px; padding:12px; margin:8px 0; max-width:500px;">
            <h3 style="margin:0 0 8px 0;">📊 {self.name}</h3>
            <table style="width:100%; font-size:13px;">
                <tr><td><b>Type</b></td><td>{self.inferred_type} ({self.dtype})</td></tr>
                <tr><td><b>Rows</b></td><td>{self.n_rows:,}</td></tr>
                <tr><td><b>Nulls</b></td><td>{self.n_null:,} ({self.null_pct:.1%})</td></tr>
                <tr><td><b>Unique</b></td><td>{self.n_unique:,}</td></tr>
                {stats_rows}
            </table>
            {"<h4>💡 Recommendations</h4><ul>" + rec_items + "</ul>" if self.recommendations else ""}
        </div>"""


class ColumnProfiler:
    """Profile DataFrame columns with statistics and FE recommendations.

    Parameters:
        df: DataFrame to profile.
        max_categories: Threshold for categorical vs text inference.
    """

    def __init__(self, df: pd.DataFrame, max_categories: int = 50) -> None:
        self.df = df
        self.max_categories = max_categories
        self._profiles: dict[str, ColumnProfile] = {}

    def profile(self, column: str) -> ColumnProfile:
        """Profile a single column.

        Args:
            column: Column name.

        Returns:
            ColumnProfile with stats and recommendations.
        """
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        col = self.df[column]
        n_rows = len(col)
        n_null = int(col.isna().sum())
        null_pct = n_null / n_rows if n_rows > 0 else 0.0
        n_unique = int(col.nunique())

        inferred_type = self._infer_type(col, n_unique)
        stats = self._compute_stats(col, inferred_type)
        recs = self._recommend(col, inferred_type, null_pct, n_unique, n_rows)
        sample_values = col.dropna().head(5).tolist()

        profile = ColumnProfile(
            name=column,
            dtype=str(col.dtype),
            n_rows=n_rows,
            n_null=n_null,
            null_pct=null_pct,
            n_unique=n_unique,
            inferred_type=inferred_type,
            stats=stats,
            sample_values=sample_values,
            recommendations=recs,
        )
        self._profiles[column] = profile
        return profile

    def profile_all(self) -> list[ColumnProfile]:
        """Profile all columns."""
        return [self.profile(col) for col in self.df.columns]

    def summary_table(self) -> pd.DataFrame:
        """Generate a summary DataFrame of all profiles."""
        if not self._profiles:
            self.profile_all()
        rows = []
        for p in self._profiles.values():
            rows.append({
                "column": p.name,
                "type": p.inferred_type,
                "dtype": p.dtype,
                "nulls": p.n_null,
                "null_pct": f"{p.null_pct:.1%}",
                "unique": p.n_unique,
                "recommendations": len(p.recommendations),
            })
        return pd.DataFrame(rows)

    def display(self, column: str) -> None:
        """Display profile as HTML in Jupyter (falls back to print)."""
        profile = self.profile(column)
        if _HAS_IPYTHON:
            display(HTML(profile.to_html()))
        else:
            print(f"=== {profile.name} ({profile.inferred_type}) ===")
            for k, v in profile.stats.items():
                print(f"  {k}: {v}")

    def _infer_type(self, col: pd.Series, n_unique: int) -> str:
        """Infer semantic column type."""
        if pd.api.types.is_bool_dtype(col):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(col):
            return "temporal"
        if pd.api.types.is_numeric_dtype(col):
            if n_unique <= 2:
                return "boolean"
            return "numeric"
        if pd.api.types.is_object_dtype(col) or pd.api.types.is_categorical_dtype(col):
            if n_unique <= self.max_categories:
                return "categorical"
            avg_len = col.dropna().astype(str).str.len().mean()
            if avg_len > 50:
                return "text"
            return "categorical"
        return "numeric"

    def _compute_stats(self, col: pd.Series, inferred_type: str) -> dict[str, Any]:
        """Compute type-appropriate statistics."""
        stats: dict[str, Any] = {}
        if inferred_type == "numeric":
            desc = col.describe()
            stats["mean"] = round(float(desc.get("mean", 0)), 4)
            stats["std"] = round(float(desc.get("std", 0)), 4)
            stats["min"] = round(float(desc.get("min", 0)), 4)
            stats["max"] = round(float(desc.get("max", 0)), 4)
            stats["skew"] = round(float(col.skew()), 4) if len(col.dropna()) > 2 else 0
        elif inferred_type == "categorical":
            top = col.value_counts().head(5)
            stats["top_values"] = {str(k): int(v) for k, v in top.items()}
            stats["cardinality"] = int(col.nunique())
        elif inferred_type == "temporal":
            non_null = col.dropna()
            if len(non_null) > 0:
                stats["min_date"] = str(non_null.min())
                stats["max_date"] = str(non_null.max())
        elif inferred_type == "text":
            lengths = col.dropna().astype(str).str.len()
            stats["avg_length"] = round(float(lengths.mean()), 1)
            stats["max_length"] = int(lengths.max())
        return stats

    def _recommend(
        self, col: pd.Series, inferred_type: str,
        null_pct: float, n_unique: int, n_rows: int,
    ) -> list[str]:
        """Generate feature engineering recommendations."""
        recs: list[str] = []

        if null_pct > 0.3:
            recs.append("High null rate → add MissingIndicator + imputation")
        elif null_pct > 0.01:
            recs.append("Has nulls → consider imputation strategy")

        if inferred_type == "numeric":
            skew = float(col.skew()) if len(col.dropna()) > 2 else 0
            if abs(skew) > 1.0:
                recs.append(f"Skewed (skew={skew:.2f}) → try LogTransformer or PowerTransformer")
            recs.append("Consider InteractionGenerator with correlated features")
            if n_unique < 20:
                recs.append("Low unique count → may benefit from binning")

        elif inferred_type == "categorical":
            if n_unique <= 10:
                recs.append("Low cardinality → OneHotEncoder")
            elif n_unique <= 50:
                recs.append("Medium cardinality → TargetEncoder or FrequencyEncoder")
            else:
                recs.append("High cardinality → use TargetEncoder or hashing")

        elif inferred_type == "temporal":
            recs.append("Extract DateTimeComponents (hour, day_of_week, month)")
            recs.append("Consider LagGenerator and RollingWindowGenerator")

        elif inferred_type == "text":
            recs.append("Apply TfidfGenerator or TextBasicFeatures")

        return recs


@dataclass
class PipelineStep:
    """A step in the pipeline builder."""

    name: str
    generator_type: str
    columns: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


class PipelineBuilder:
    """Interactive pipeline builder with code export.

    Parameters:
        df: Input DataFrame.
        y: Target series (optional).
    """

    def __init__(self, df: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.df = df
        self.y = y
        self._steps: list[PipelineStep] = []
        self._profiler = ColumnProfiler(df)

    def add_step(
        self,
        generator_type: str,
        columns: list[str] | None = None,
        name: str | None = None,
        **params: Any,
    ) -> PipelineBuilder:
        """Add a transformation step.

        Args:
            generator_type: Type of generator/transform (e.g., 'numeric', 'categorical').
            columns: Columns to apply to. None for auto-detect.
            name: Optional step name.
            **params: Additional parameters for the generator.

        Returns:
            Self for chaining.
        """
        if columns is None:
            columns = self._auto_detect_columns(generator_type)

        step_name = name or f"{generator_type}_{len(self._steps)}"
        self._steps.append(PipelineStep(
            name=step_name,
            generator_type=generator_type,
            columns=columns,
            params=params,
        ))
        return self

    def remove_step(self, index: int) -> PipelineBuilder:
        """Remove a step by index."""
        if 0 <= index < len(self._steps):
            self._steps.pop(index)
        return self

    def preview(self, max_rows: int = 5) -> pd.DataFrame:
        """Preview what the pipeline would produce (runs non-destructively)."""
        result = self.df.head(max_rows).copy()
        for step in self._steps:
            result = self._apply_step_preview(result, step)
        return result

    def _apply_step_preview(self, df: pd.DataFrame, step: PipelineStep) -> pd.DataFrame:
        """Apply a single step for preview."""
        result = df.copy()
        cols = [c for c in step.columns if c in result.columns]

        if step.generator_type == "numeric":
            for col in cols:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[f"{col}_squared"] = result[col] ** 2
                    result[f"{col}_log1p"] = np.log1p(result[col].clip(lower=0))
        elif step.generator_type == "categorical":
            for col in cols:
                result[f"{col}_freq"] = result[col].map(result[col].value_counts(normalize=True))
        elif step.generator_type == "scaler":
            for col in cols:
                if pd.api.types.is_numeric_dtype(result[col]):
                    mean, std = result[col].mean(), result[col].std()
                    if std > 0:
                        result[col] = (result[col] - mean) / std

        return result

    def export_code(self) -> str:
        """Export the pipeline as executable Forge Python code."""
        lines: list[str] = [
            "# Auto-generated by Forge PipelineBuilder",
            "from forge import AutoFeatureTransformer, ForgePipeline",
            "from sklearn.pipeline import Pipeline",
            "",
        ]

        gen_imports: set[str] = set()
        step_code: list[str] = []

        for i, step in enumerate(self._steps):
            var = f"step_{i}"
            cols_str = repr(step.columns)

            if step.generator_type == "numeric":
                gen_imports.add("from forge.generators.numeric.interactions import InteractionGenerator")
                gen_imports.add("from forge.generators.numeric.transformations import LogTransformer")
                step_code.append(f'{var} = InteractionGenerator(columns={cols_str})')
            elif step.generator_type == "categorical":
                gen_imports.add("from forge.generators.categorical.encoders import TargetEncoder")
                step_code.append(f'{var} = TargetEncoder(columns={cols_str})')
            elif step.generator_type == "scaler":
                gen_imports.add("from sklearn.preprocessing import StandardScaler")
                step_code.append(f'{var} = StandardScaler()')
            elif step.generator_type == "temporal":
                gen_imports.add("from forge.generators.temporal.components import DateTimeComponents")
                step_code.append(f'{var} = DateTimeComponents(columns={cols_str})')
            elif step.generator_type == "selector":
                gen_imports.add("from forge.selectors.variance import VarianceSelector")
                step_code.append(f'{var} = VarianceSelector()')
            else:
                step_code.append(f'# {var} = custom step for {step.generator_type}')

        if gen_imports:
            lines.extend(sorted(gen_imports))
            lines.append("")

        lines.append("# Build pipeline")
        lines.append("pipeline = Pipeline([")
        for i, step in enumerate(self._steps):
            lines.append(f'    ("{step.name}", step_{i}),')
        lines.append("])")
        lines.append("")
        lines.append("# Fit and transform")
        lines.append("X_transformed = pipeline.fit_transform(X, y)")

        full_code = "\n".join(lines)

        # Insert step definitions before pipeline build
        step_block = "\n".join(step_code)
        full_code = full_code.replace("# Build pipeline", step_block + "\n\n# Build pipeline")

        return full_code

    @property
    def steps(self) -> list[PipelineStep]:
        """Current pipeline steps."""
        return list(self._steps)

    def _auto_detect_columns(self, generator_type: str) -> list[str]:
        """Auto-detect columns for a generator type."""
        if generator_type == "numeric":
            return self.df.select_dtypes(include=[np.number]).columns.tolist()
        elif generator_type == "categorical":
            return self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        elif generator_type == "temporal":
            return self.df.select_dtypes(include=["datetime64"]).columns.tolist()
        elif generator_type in ("scaler", "selector"):
            return self.df.select_dtypes(include=[np.number]).columns.tolist()
        return self.df.columns.tolist()


class DatasetExplorer:
    """Quick dataset overview for notebook use.

    Parameters:
        df: DataFrame to explore.
        y: Optional target series.
    """

    def __init__(self, df: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.df = df
        self.y = y
        self._profiler = ColumnProfiler(df)

    def overview(self) -> dict[str, Any]:
        """Get a quick dataset overview."""
        n_numeric = len(self.df.select_dtypes(include=[np.number]).columns)
        n_categorical = len(self.df.select_dtypes(include=["object", "category"]).columns)
        n_datetime = len(self.df.select_dtypes(include=["datetime64"]).columns)

        return {
            "rows": len(self.df),
            "columns": self.df.shape[1],
            "numeric_columns": n_numeric,
            "categorical_columns": n_categorical,
            "datetime_columns": n_datetime,
            "total_nulls": int(self.df.isna().sum().sum()),
            "null_pct": round(float(self.df.isna().mean().mean()), 4),
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1e6, 2),
        }

    def suggest_pipeline(self) -> list[PipelineStep]:
        """Suggest a pipeline based on data analysis."""
        suggestions: list[PipelineStep] = []

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        dt_cols = self.df.select_dtypes(include=["datetime64"]).columns.tolist()

        if numeric_cols:
            suggestions.append(PipelineStep(
                name="numeric_transforms",
                generator_type="numeric",
                columns=numeric_cols,
            ))

        if cat_cols:
            suggestions.append(PipelineStep(
                name="categorical_encoding",
                generator_type="categorical",
                columns=cat_cols,
            ))

        if dt_cols:
            suggestions.append(PipelineStep(
                name="temporal_features",
                generator_type="temporal",
                columns=dt_cols,
            ))

        if numeric_cols:
            suggestions.append(PipelineStep(
                name="feature_selection",
                generator_type="selector",
                columns=numeric_cols,
            ))

        return suggestions
