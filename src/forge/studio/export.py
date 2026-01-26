"""Feature Studio code export — generate Python code from pipeline configurations.

Converts FeatureStudio pipeline definitions into executable Python code,
enabling no-code-to-code workflows.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CodeExporter:
    """Export feature pipelines as executable Python code.

    Converts a FeatureStudio pipeline into Python source code
    that can be saved, version-controlled, and executed independently.

    Example:
        >>> exporter = CodeExporter()
        >>> code = exporter.export_pipeline(studio)
        >>> print(code)
        >>> # Save to file
        >>> exporter.export_to_file(studio, "my_pipeline.py")
    """

    def __init__(self, style: str = "script") -> None:
        """Initialize code exporter.

        Args:
            style: Export style — "script" for standalone script,
                "function" for a reusable function, "class" for
                a sklearn transformer class.
        """
        self.style = style

    def export_pipeline(
        self,
        steps: list[dict[str, Any]],
        pipeline_name: str = "feature_pipeline",
        include_imports: bool = True,
    ) -> str:
        """Export pipeline steps as Python code.

        Args:
            steps: List of step definitions with "name", "transformer_class",
                "params" keys.
            pipeline_name: Variable name for the pipeline.
            include_imports: Whether to include import statements.

        Returns:
            Python source code as a string.
        """
        if self.style == "function":
            return self._export_as_function(steps, pipeline_name, include_imports)
        if self.style == "class":
            return self._export_as_class(steps, pipeline_name, include_imports)
        return self._export_as_script(steps, pipeline_name, include_imports)

    def _export_as_script(
        self,
        steps: list[dict[str, Any]],
        pipeline_name: str,
        include_imports: bool,
    ) -> str:
        lines: list[str] = []

        if include_imports:
            lines.extend(self._generate_imports(steps))
            lines.append("")

        lines.append(f"# Feature pipeline: {pipeline_name}")
        lines.append(f'{pipeline_name} = ForgePipeline([')

        for step in steps:
            name = step.get("name", "step")
            cls_name = step.get("transformer_class", "BaseFeatureGenerator")
            params = step.get("params", {})
            params_str = ", ".join(f"{k}={_repr_value(v)}" for k, v in params.items())
            lines.append(f'    ("{name}", {cls_name}({params_str})),')

        lines.append("])")
        lines.append("")
        lines.append("# Usage:")
        lines.append(f"# X_transformed = {pipeline_name}.fit_transform(X, y)")

        return "\n".join(lines)

    def _export_as_function(
        self,
        steps: list[dict[str, Any]],
        pipeline_name: str,
        include_imports: bool,
    ) -> str:
        lines: list[str] = []

        if include_imports:
            lines.extend(self._generate_imports(steps))
            lines.append("")

        lines.append(f"def create_{pipeline_name}() -> ForgePipeline:")
        lines.append(f'    """Create the {pipeline_name} feature pipeline."""')
        lines.append("    return ForgePipeline([")

        for step in steps:
            name = step.get("name", "step")
            cls_name = step.get("transformer_class", "BaseFeatureGenerator")
            params = step.get("params", {})
            params_str = ", ".join(f"{k}={_repr_value(v)}" for k, v in params.items())
            lines.append(f'        ("{name}", {cls_name}({params_str})),')

        lines.append("    ])")

        return "\n".join(lines)

    def _export_as_class(
        self,
        steps: list[dict[str, Any]],
        pipeline_name: str,
        include_imports: bool,
    ) -> str:
        class_name = "".join(
            word.capitalize() for word in pipeline_name.split("_")
        ) + "Transformer"

        lines: list[str] = []
        if include_imports:
            lines.extend(self._generate_imports(steps))
            lines.append("from sklearn.base import BaseEstimator, TransformerMixin")
            lines.append("")

        lines.append(f"class {class_name}(BaseEstimator, TransformerMixin):")
        lines.append(f'    """Auto-generated feature transformer for {pipeline_name}."""')
        lines.append("")
        lines.append("    def __init__(self) -> None:")
        lines.append("        self._pipeline = ForgePipeline([")
        for step in steps:
            name = step.get("name", "step")
            cls_name = step.get("transformer_class", "BaseFeatureGenerator")
            params = step.get("params", {})
            params_str = ", ".join(f"{k}={_repr_value(v)}" for k, v in params.items())
            lines.append(f'            ("{name}", {cls_name}({params_str})),')
        lines.append("        ])")
        lines.append("")
        lines.append("    def fit(self, X, y=None):")
        lines.append("        self._pipeline.fit(X, y)")
        lines.append("        return self")
        lines.append("")
        lines.append("    def transform(self, X):")
        lines.append("        return self._pipeline.transform(X)")
        lines.append("")
        lines.append("    def get_feature_names_out(self, input_features=None):")
        lines.append("        return self._pipeline.get_feature_names_out(input_features)")

        return "\n".join(lines)

    def _generate_imports(self, steps: list[dict[str, Any]]) -> list[str]:
        """Generate import statements based on transformers used."""
        imports: set[str] = set()
        imports.add("from forge.transformers import ForgePipeline")

        module_map = {
            "InteractionGenerator": "from forge.generators.numeric import InteractionGenerator",
            "PolynomialGenerator": "from forge.generators.numeric import PolynomialGenerator",
            "TargetEncoder": "from forge.generators.categorical import TargetEncoder",
            "AutoFeatureTransformer": "from forge import AutoFeatureTransformer",
        }

        for step in steps:
            cls = step.get("transformer_class", "")
            if cls in module_map:
                imports.add(module_map[cls])

        return sorted(imports)

    def export_to_file(
        self,
        steps: list[dict[str, Any]],
        filepath: str,
        pipeline_name: str = "feature_pipeline",
    ) -> None:
        """Export pipeline to a Python file.

        Args:
            steps: Pipeline step definitions.
            filepath: Output file path.
            pipeline_name: Pipeline variable name.
        """
        code = self.export_pipeline(steps, pipeline_name)
        with open(filepath, "w") as f:
            f.write(code)
        logger.info("Exported pipeline to %s", filepath)


def _repr_value(value: Any) -> str:
    """Represent a value as Python source code."""
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return repr(value)
    return str(value)


def export_studio_to_dsl(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Export studio pipeline as a DSL specification.

    Args:
        steps: Pipeline step definitions.

    Returns:
        DSL-compatible dictionary specification.
    """
    dsl_steps = []
    for step in steps:
        dsl_steps.append({
            "name": step.get("name", "step"),
            "type": step.get("transformer_class", "unknown"),
            "params": step.get("params", {}),
        })

    return {
        "version": "1.0",
        "pipeline": {
            "steps": dsl_steps,
        },
    }
