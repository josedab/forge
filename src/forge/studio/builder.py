"""Feature Studio: interactive pipeline builder and preview engine.

Provides a programmatic API for building feature pipelines with
real-time preview, A/B comparison, and Python code export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single step in the feature pipeline.

    Attributes:
    ----------
    name : str
        Step name/identifier.
    transformer : Any
        sklearn-compatible transformer instance.
    enabled : bool
        Whether this step is active.
    description : str
        Human-readable description.
    """

    name: str
    transformer: Any
    enabled: bool = True
    description: str = ""


@dataclass
class PreviewResult:
    """Result of previewing a pipeline on sample data.

    Attributes:
    ----------
    n_input_features : int
        Number of input features.
    n_output_features : int
        Number of output features.
    new_features : list[str]
        Names of newly generated features.
    sample_output : dict[str, list[Any]]
        Sample of transformed data (first N rows).
    feature_stats : dict[str, dict[str, float]]
        Basic stats for each output feature.
    """

    n_input_features: int
    n_output_features: int
    new_features: list[str]
    sample_output: dict[str, list[Any]] = field(default_factory=dict)
    feature_stats: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing two pipeline configurations.

    Attributes:
    ----------
    pipeline_a_name : str
        Name of pipeline A.
    pipeline_b_name : str
        Name of pipeline B.
    a_features : int
        Feature count from pipeline A.
    b_features : int
        Feature count from pipeline B.
    shared_features : list[str]
        Features present in both.
    a_only_features : list[str]
        Features only in A.
    b_only_features : list[str]
        Features only in B.
    correlation_diff : dict[str, float]
        Correlation differences for shared features with target.
    """

    pipeline_a_name: str
    pipeline_b_name: str
    a_features: int
    b_features: int
    shared_features: list[str] = field(default_factory=list)
    a_only_features: list[str] = field(default_factory=list)
    b_only_features: list[str] = field(default_factory=list)
    correlation_diff: dict[str, float] = field(default_factory=dict)


class FeatureStudio(BaseEstimator):
    """Interactive feature engineering environment.

    Build pipelines step-by-step with instant preview, compare
    configurations, and export to Python code.

    Parameters
    ----------
    sample_size : int
        Number of rows for preview computations.
    random_state : int | None
        Random seed for sampling.

    Examples:
    --------
    >>> from forge.studio import FeatureStudio
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> studio = FeatureStudio()
    >>> studio.load_data(X, y)
    >>> studio.add_step("scale", StandardScaler())
    >>> preview = studio.preview()
    >>> print(f"Generated {preview.n_output_features} features")
    >>> code = studio.export_code()
    """

    def __init__(
        self,
        sample_size: int = 1000,
        random_state: int | None = None,
    ) -> None:
        self.sample_size = sample_size
        self.random_state = random_state
        self._steps: list[PipelineStep] = []
        self._data: pd.DataFrame | None = None
        self._target: Any = None

    def load_data(
        self, X: pd.DataFrame, y: Any = None
    ) -> FeatureStudio:
        """Load data into the studio.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : Any
            Target variable (optional).

        Returns:
        -------
        FeatureStudio
            Self for chaining.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(X).__name__}")

        self._data = X.copy()
        self._target = np.asarray(y) if y is not None else None
        return self

    def add_step(
        self,
        name: str,
        transformer: Any,
        description: str = "",
    ) -> FeatureStudio:
        """Add a transformation step to the pipeline.

        Parameters
        ----------
        name : str
            Unique step name.
        transformer : Any
            sklearn-compatible transformer.
        description : str
            Human-readable description.

        Returns:
        -------
        FeatureStudio
            Self for chaining.
        """
        if any(s.name == name for s in self._steps):
            raise ValueError(f"Step '{name}' already exists. Use update_step().")

        self._steps.append(
            PipelineStep(
                name=name,
                transformer=transformer,
                description=description,
            )
        )
        return self

    def remove_step(self, name: str) -> FeatureStudio:
        """Remove a pipeline step by name.

        Parameters
        ----------
        name : str
            Step name to remove.

        Returns:
        -------
        FeatureStudio
            Self for chaining.
        """
        self._steps = [s for s in self._steps if s.name != name]
        return self

    def toggle_step(self, name: str) -> FeatureStudio:
        """Toggle a step's enabled status.

        Parameters
        ----------
        name : str
            Step name to toggle.

        Returns:
        -------
        FeatureStudio
            Self for chaining.
        """
        for step in self._steps:
            if step.name == name:
                step.enabled = not step.enabled
                return self
        raise ValueError(f"Step '{name}' not found.")

    def reorder_steps(self, order: list[str]) -> FeatureStudio:
        """Reorder pipeline steps.

        Parameters
        ----------
        order : list[str]
            Step names in desired order.

        Returns:
        -------
        FeatureStudio
            Self for chaining.
        """
        step_map = {s.name: s for s in self._steps}
        new_steps: list[PipelineStep] = []
        for name in order:
            if name in step_map:
                new_steps.append(step_map[name])
        # Append any steps not in the order list
        for step in self._steps:
            if step.name not in order:
                new_steps.append(step)
        self._steps = new_steps
        return self

    def get_steps(self) -> list[PipelineStep]:
        """Get current pipeline steps."""
        return list(self._steps)

    def preview(self, max_rows: int = 5) -> PreviewResult:
        """Preview the pipeline on sample data.

        Parameters
        ----------
        max_rows : int
            Number of sample rows to include.

        Returns:
        -------
        PreviewResult
            Preview of transformed data.
        """
        if self._data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        sample = self._get_sample()
        original_cols = set(sample.columns)

        result = sample.copy()
        for step in self._steps:
            if not step.enabled:
                continue
            try:
                result = step.transformer.fit_transform(result, self._target)
            except Exception as e:
                logger.warning("Step '%s' failed during preview: %s", step.name, e)
                continue

        import pandas as pd
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result)

        new_features = [c for c in result.columns if c not in original_cols]

        # Compute stats for numeric features
        feature_stats: dict[str, dict[str, float]] = {}
        for col in result.columns:
            if result[col].dtype in (np.float64, np.int64, np.float32, np.int32):
                values = result[col].dropna().values.astype(float)
                if len(values) > 0:
                    feature_stats[col] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }

        # Sample output
        sample_output: dict[str, list[Any]] = {}
        for col in result.columns[:20]:  # Limit to 20 columns
            sample_output[col] = result[col].head(max_rows).tolist()

        return PreviewResult(
            n_input_features=len(original_cols),
            n_output_features=len(result.columns),
            new_features=new_features,
            sample_output=sample_output,
            feature_stats=feature_stats,
        )

    def compare(
        self,
        other_steps: list[PipelineStep],
        name_a: str = "Pipeline A",
        name_b: str = "Pipeline B",
    ) -> ComparisonResult:
        """Compare current pipeline against an alternative configuration.

        Parameters
        ----------
        other_steps : list[PipelineStep]
            Alternative pipeline steps.
        name_a : str
            Name for current pipeline.
        name_b : str
            Name for alternative pipeline.

        Returns:
        -------
        ComparisonResult
            Comparison of both pipeline outputs.
        """
        if self._data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        import pandas as pd

        sample = self._get_sample()

        # Run pipeline A (current)
        result_a = sample.copy()
        for step in self._steps:
            if step.enabled:
                try:
                    result_a = step.transformer.fit_transform(result_a, self._target)
                except Exception:
                    pass
        if not isinstance(result_a, pd.DataFrame):
            result_a = pd.DataFrame(result_a)

        # Run pipeline B (other)
        result_b = sample.copy()
        for step in other_steps:
            if step.enabled:
                try:
                    result_b = step.transformer.fit_transform(result_b, self._target)
                except Exception:
                    pass
        if not isinstance(result_b, pd.DataFrame):
            result_b = pd.DataFrame(result_b)

        cols_a = set(result_a.columns)
        cols_b = set(result_b.columns)
        shared = sorted(cols_a & cols_b)
        a_only = sorted(cols_a - cols_b)
        b_only = sorted(cols_b - cols_a)

        # Correlation diff with target for shared features
        correlation_diff: dict[str, float] = {}
        if self._target is not None:
            target_sample = self._target[: len(sample)]
            for col in shared:
                try:
                    corr_a = float(np.corrcoef(
                        result_a[col].values.astype(float), target_sample
                    )[0, 1])
                    corr_b = float(np.corrcoef(
                        result_b[col].values.astype(float), target_sample
                    )[0, 1])
                    if not (np.isnan(corr_a) or np.isnan(corr_b)):
                        correlation_diff[col] = abs(corr_a) - abs(corr_b)
                except (ValueError, TypeError):
                    pass

        return ComparisonResult(
            pipeline_a_name=name_a,
            pipeline_b_name=name_b,
            a_features=len(cols_a),
            b_features=len(cols_b),
            shared_features=shared,
            a_only_features=a_only,
            b_only_features=b_only,
            correlation_diff=correlation_diff,
        )

    def export_code(self) -> str:
        """Export current pipeline as Python code.

        Returns:
        -------
        str
            Python source code recreating the pipeline.
        """
        lines: list[str] = [
            '"""Auto-generated feature pipeline from Forge Studio."""',
            "",
            "from sklearn.pipeline import Pipeline",
        ]

        # Collect imports
        for step in self._steps:
            cls = type(step.transformer)
            module = cls.__module__
            name = cls.__name__
            lines.append(f"from {module} import {name}")

        lines.extend(["", "", "def create_pipeline() -> Pipeline:"])
        lines.append('    """Create the feature engineering pipeline."""')
        lines.append("    return Pipeline([")

        for step in self._steps:
            cls = type(step.transformer)
            name = cls.__name__

            params = step.transformer.get_params() if hasattr(step.transformer, "get_params") else {}
            params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())

            enabled_str = "" if step.enabled else "  # DISABLED"
            lines.append(
                f'        ("{step.name}", {name}({params_str})),{enabled_str}'
            )

        lines.append("    ])")
        lines.append("")

        return "\n".join(lines)

    def _get_sample(self) -> pd.DataFrame:
        """Get a sample of the loaded data."""
        if self._data is None:
            raise RuntimeError("No data loaded.")

        if len(self._data) <= self.sample_size:
            return self._data.copy()

        return self._data.sample(
            n=self.sample_size, random_state=self.random_state
        ).reset_index(drop=True)

    def data_summary(self) -> dict[str, Any]:
        """Get summary of loaded data.

        Returns:
        -------
        dict
            Data summary including shape, types, and null counts.
        """
        if self._data is None:
            return {"error": "No data loaded."}

        return {
            "n_rows": len(self._data),
            "n_columns": len(self._data.columns),
            "columns": list(self._data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self._data.dtypes.items()},
            "null_counts": self._data.isnull().sum().to_dict(),
            "has_target": self._target is not None,
        }
