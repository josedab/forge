"""DSL compiler: converts PipelineSpec into executable ForgePipeline.

Compiles parsed DSL specifications into sklearn-compatible transformer
pipelines and generates equivalent Python code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from forge.dsl.parser import FeatureDSLError, FeatureSpec, PipelineSpec

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Maps transform names to (module_path, class_name, default_params)
_TRANSFORM_REGISTRY: dict[str, tuple[str, str, dict[str, Any]]] = {
    # Numeric
    "log": ("forge.generators.numeric.transformations", "TransformationGenerator", {"transformations": ["log"]}),
    "sqrt": ("forge.generators.numeric.transformations", "TransformationGenerator", {"transformations": ["sqrt"]}),
    "square": ("forge.generators.numeric.transformations", "TransformationGenerator", {"transformations": ["power"]}),
    "polynomial": ("forge.generators.numeric.polynomials", "PolynomialGenerator", {}),
    "interaction": ("forge.generators.numeric.interactions", "InteractionGenerator", {}),
    "bin": ("forge.generators.numeric.transformations", "TransformationGenerator", {"transformations": ["bin"]}),
    # Categorical
    "onehot": ("forge.generators.categorical.encoders", "OneHotEncoder", {}),
    "target": ("forge.generators.categorical.encoders", "TargetEncoder", {}),
    "frequency": ("forge.generators.categorical.encoders", "FrequencyEncoder", {}),
    "ordinal": ("forge.generators.categorical.encoders", "OrdinalEncoder", {}),
    # Text
    "tfidf": ("forge.generators.text.tfidf", "TfidfGenerator", {}),
    "length": ("forge.generators.text.basic", "TextStatsGenerator", {}),
    "word_count": ("forge.generators.text.basic", "TextStatsGenerator", {}),
    "char_count": ("forge.generators.text.basic", "TextStatsGenerator", {}),
}

# Selection method mapping
_SELECTION_REGISTRY: dict[str, tuple[str, str]] = {
    "importance": ("forge.selectors.importance", "ImportanceSelector"),
    "statistical": ("forge.selectors.statistical", "StatisticalSelector"),
    "correlation": ("forge.selectors.correlation", "CorrelationSelector"),
    "variance": ("forge.selectors.variance", "VarianceSelector"),
    "shap": ("forge.selectors.shap_selector", "ShapSelector"),
}


class DSLCompiler:
    """Compile PipelineSpec into executable pipelines.

    Takes a parsed PipelineSpec and produces either a ForgePipeline
    (for execution) or Python source code (for reproducibility).

    Examples:
    --------
    >>> from forge.dsl import DSLParser, DSLCompiler
    >>>
    >>> definition = {
    ...     "name": "my_pipeline",
    ...     "version": "1.0",
    ...     "features": [
    ...         {"name": "price_log", "source": "price", "transform": "log"},
    ...     ],
    ... }
    >>> parser = DSLParser()
    >>> compiler = DSLCompiler()
    >>> spec = parser.parse(definition)
    >>> code = compiler.to_code(spec)
    >>> print(code)
    """

    def compile(self, spec: PipelineSpec) -> Pipeline:
        """Compile a PipelineSpec into an sklearn Pipeline.

        Parameters
        ----------
        spec : PipelineSpec
            Parsed pipeline specification.

        Returns:
        -------
        sklearn.pipeline.Pipeline
            Executable sklearn pipeline.

        Raises:
        ------
        FeatureDSLError
            If compilation fails due to missing modules or invalid specs.
        """
        from sklearn.pipeline import Pipeline

        steps: list[tuple[str, Any]] = []

        for feat in spec.features:
            try:
                transformer = self._build_transformer(feat)
                step_name = f"{feat.name}_{feat.transform}"
                steps.append((step_name, transformer))
            except Exception as e:
                raise FeatureDSLError(
                    f"Failed to compile feature '{feat.name}': {e}"
                ) from e

        if spec.selection:
            try:
                selector = self._build_selector(spec.selection)
                steps.append(("selection", selector))
            except Exception as e:
                logger.warning("Failed to compile selection step: %s", e)

        if not steps:
            raise FeatureDSLError("No valid steps could be compiled.")

        return Pipeline(steps)

    def _build_transformer(self, feat: FeatureSpec) -> Any:
        """Build a transformer instance for a feature spec."""
        if feat.transform not in _TRANSFORM_REGISTRY:
            raise FeatureDSLError(
                f"No compiler mapping for transform '{feat.transform}'."
            )

        module_path, class_name, defaults = _TRANSFORM_REGISTRY[feat.transform]

        import importlib
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            raise FeatureDSLError(
                f"Cannot import {module_path}: {e}"
            ) from e

        cls = getattr(mod, class_name, None)
        if cls is None:
            raise FeatureDSLError(
                f"Class '{class_name}' not found in {module_path}."
            )

        params = {**defaults, **feat.params}

        # Set source columns
        source = feat.source if isinstance(feat.source, list) else [feat.source]
        if "columns" not in params:
            params["columns"] = source

        return cls(**params)

    def _build_selector(self, selection: Any) -> Any:
        """Build a selector instance."""
        if selection.method not in _SELECTION_REGISTRY:
            raise FeatureDSLError(
                f"Unknown selection method: '{selection.method}'."
            )

        module_path, class_name = _SELECTION_REGISTRY[selection.method]

        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)

        return cls(**selection.params)

    def to_code(self, spec: PipelineSpec) -> str:
        """Generate Python source code from a PipelineSpec.

        Parameters
        ----------
        spec : PipelineSpec
            Parsed pipeline specification.

        Returns:
        -------
        str
            Python source code that recreates the pipeline.
        """
        lines: list[str] = []
        lines.append(f'"""Auto-generated pipeline: {spec.name}')
        if spec.description:
            lines.append(f"\n{spec.description}")
        lines.append('"""')
        lines.append("")

        # Collect imports
        imports: set[str] = {"from sklearn.pipeline import Pipeline"}
        for feat in spec.features:
            if feat.transform in _TRANSFORM_REGISTRY:
                module_path, class_name, _ = _TRANSFORM_REGISTRY[feat.transform]
                imports.add(f"from {module_path} import {class_name}")

        if spec.selection and spec.selection.method in _SELECTION_REGISTRY:
            module_path, class_name = _SELECTION_REGISTRY[spec.selection.method]
            imports.add(f"from {module_path} import {class_name}")

        for imp in sorted(imports):
            lines.append(imp)

        lines.append("")
        lines.append("")
        lines.append("def create_pipeline() -> Pipeline:")
        lines.append(f'    """Create {spec.name} pipeline."""')
        lines.append("    steps = []")
        lines.append("")

        for feat in spec.features:
            if feat.transform in _TRANSFORM_REGISTRY:
                _, class_name, defaults = _TRANSFORM_REGISTRY[feat.transform]
                params = {**defaults, **feat.params}
                source = feat.source if isinstance(feat.source, list) else [feat.source]
                params["columns"] = source
                params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
                lines.append(f"    # Feature: {feat.name}")
                lines.append(
                    f'    steps.append(("{feat.name}_{feat.transform}", '
                    f"{class_name}({params_str})))"
                )
                lines.append("")

        if spec.selection and spec.selection.method in _SELECTION_REGISTRY:
            _, class_name = _SELECTION_REGISTRY[spec.selection.method]
            params_str = ", ".join(
                f"{k}={v!r}" for k, v in spec.selection.params.items()
            )
            lines.append(f'    steps.append(("selection", {class_name}({params_str})))')
            lines.append("")

        lines.append("    return Pipeline(steps)")
        lines.append("")

        return "\n".join(lines)

    def validate_spec(self, spec: PipelineSpec) -> dict[str, Any]:
        """Validate a spec and return detailed validation report.

        Parameters
        ----------
        spec : PipelineSpec
            Pipeline spec to validate.

        Returns:
        -------
        dict
            Validation report with errors, warnings, and summary.
        """
        errors = spec.validate()

        warnings: list[str] = []
        for feat in spec.features:
            if feat.transform not in _TRANSFORM_REGISTRY:
                warnings.append(
                    f"Feature '{feat.name}': transform '{feat.transform}' "
                    f"has no compiler mapping. Pipeline compilation will fail."
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "feature_count": len(spec.features),
            "has_selection": spec.selection is not None,
        }
