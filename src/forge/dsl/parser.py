"""DSL parser for declarative feature pipeline definitions.

Parses YAML/dict feature definitions into structured specs that can be
compiled into ForgePipeline objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from forge.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Valid transform operations for each data type
VALID_NUMERIC_TRANSFORMS = frozenset({
    "log", "sqrt", "square", "reciprocal", "bin", "standard_scale",
    "min_max_scale", "polynomial", "interaction",
})

VALID_CATEGORICAL_TRANSFORMS = frozenset({
    "onehot", "target", "frequency", "ordinal", "binary", "woe",
})

VALID_TEMPORAL_TRANSFORMS = frozenset({
    "components", "lag", "rolling_mean", "rolling_std", "diff",
    "day_of_week", "month", "year", "hour", "is_weekend",
})

VALID_TEXT_TRANSFORMS = frozenset({
    "length", "word_count", "tfidf", "char_count",
})

ALL_VALID_TRANSFORMS = (
    VALID_NUMERIC_TRANSFORMS
    | VALID_CATEGORICAL_TRANSFORMS
    | VALID_TEMPORAL_TRANSFORMS
    | VALID_TEXT_TRANSFORMS
)


class FeatureDSLError(ConfigurationError):
    """Raised when a DSL definition is invalid."""

    pass


@dataclass
class FeatureSpec:
    """Specification for a single feature transformation.

    Attributes:
    ----------
    name : str
        Output feature name.
    source : str | list[str]
        Source column(s).
    transform : str
        Transform to apply.
    params : dict[str, Any]
        Additional parameters for the transform.
    dtype : str
        Expected data type: 'numeric', 'categorical', 'temporal', 'text', 'auto'.
    """

    name: str
    source: str | list[str]
    transform: str
    params: dict[str, Any] = field(default_factory=dict)
    dtype: str = "auto"

    def validate(self) -> list[str]:
        """Validate the feature spec.

        Returns:
        -------
        list[str]
            List of validation error messages. Empty if valid.
        """
        errors: list[str] = []

        if not self.name:
            errors.append("Feature name is required.")
        if not self.source:
            errors.append(f"Feature '{self.name}': source column(s) required.")
        if not self.transform:
            errors.append(f"Feature '{self.name}': transform is required.")
        elif self.transform not in ALL_VALID_TRANSFORMS:
            errors.append(
                f"Feature '{self.name}': unknown transform '{self.transform}'. "
                f"Valid transforms: {sorted(ALL_VALID_TRANSFORMS)}"
            )
        if self.dtype not in ("numeric", "categorical", "temporal", "text", "auto"):
            errors.append(
                f"Feature '{self.name}': invalid dtype '{self.dtype}'. "
                "Must be 'numeric', 'categorical', 'temporal', 'text', or 'auto'."
            )

        return errors


@dataclass
class SelectionSpec:
    """Specification for feature selection step.

    Attributes:
    ----------
    method : str
        Selection method.
    params : dict[str, Any]
        Method parameters (e.g., k, threshold).
    """

    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSpec:
    """Complete pipeline specification parsed from DSL.

    Attributes:
    ----------
    name : str
        Pipeline name.
    version : str
        Schema version.
    description : str
        Pipeline description.
    features : list[FeatureSpec]
        Feature definitions.
    selection : SelectionSpec | None
        Optional selection step.
    metadata : dict[str, Any]
        Additional metadata.
    """

    name: str
    version: str
    description: str
    features: list[FeatureSpec]
    selection: SelectionSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate the entire pipeline spec.

        Returns:
        -------
        list[str]
            All validation errors. Empty if valid.
        """
        errors: list[str] = []

        if not self.name:
            errors.append("Pipeline name is required.")
        if not self.features:
            errors.append("Pipeline must define at least one feature.")

        # Check for duplicate feature names
        names = [f.name for f in self.features]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            errors.append(f"Duplicate feature names: {sorted(set(duplicates))}")

        for feat in self.features:
            errors.extend(feat.validate())

        if self.selection and self.selection.method not in (
            "importance", "statistical", "correlation", "variance", "shap",
        ):
            errors.append(
                f"Unknown selection method: '{self.selection.method}'."
            )

        return errors


class DSLParser:
    """Parse YAML/dict feature definitions into PipelineSpec.

    Supports loading from Python dicts or YAML strings/files.

    Parameters
    ----------
    strict : bool
        If True, raise on any validation error. If False, log warnings.

    Examples:
    --------
    >>> from forge.dsl import DSLParser
    >>>
    >>> definition = {
    ...     "name": "my_pipeline",
    ...     "version": "1.0",
    ...     "features": [
    ...         {"name": "price_log", "source": "price", "transform": "log"},
    ...         {"name": "cat_encoded", "source": "category", "transform": "target"},
    ...     ],
    ... }
    >>> parser = DSLParser()
    >>> spec = parser.parse(definition)
    >>> print(spec.name)
    my_pipeline
    """

    SUPPORTED_VERSIONS = ("1.0",)

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def parse(self, definition: dict[str, Any]) -> PipelineSpec:
        """Parse a dict definition into a PipelineSpec.

        Parameters
        ----------
        definition : dict
            Pipeline definition dict.

        Returns:
        -------
        PipelineSpec
            Parsed pipeline specification.

        Raises:
        ------
        FeatureDSLError
            If definition is invalid and strict mode is enabled.
        """
        if not isinstance(definition, dict):
            raise FeatureDSLError(
                f"Expected dict, got {type(definition).__name__}"
            )

        version = str(definition.get("version", "1.0"))
        if version not in self.SUPPORTED_VERSIONS:
            raise FeatureDSLError(
                f"Unsupported DSL version '{version}'. "
                f"Supported: {self.SUPPORTED_VERSIONS}"
            )

        features = self._parse_features(definition.get("features", []))
        selection = self._parse_selection(definition.get("selection"))

        spec = PipelineSpec(
            name=definition.get("name", "unnamed_pipeline"),
            version=version,
            description=definition.get("description", ""),
            features=features,
            selection=selection,
            metadata=definition.get("metadata", {}),
        )

        errors = spec.validate()
        if errors:
            if self.strict:
                raise FeatureDSLError(
                    "Pipeline validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                )
            for error in errors:
                logger.warning("DSL validation: %s", error)

        return spec

    def parse_yaml(self, yaml_str: str) -> PipelineSpec:
        """Parse a YAML string into a PipelineSpec.

        Parameters
        ----------
        yaml_str : str
            YAML-formatted pipeline definition.

        Returns:
        -------
        PipelineSpec
            Parsed pipeline specification.

        Raises:
        ------
        FeatureDSLError
            If YAML is invalid or definition fails validation.
        """
        try:
            import yaml
        except ImportError:
            raise FeatureDSLError(
                "PyYAML is required for YAML parsing. Install with: pip install pyyaml"
            )

        try:
            definition = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise FeatureDSLError(f"Invalid YAML: {e}") from e

        if not isinstance(definition, dict):
            raise FeatureDSLError("YAML must define a mapping at the top level.")

        return self.parse(definition)

    def _parse_features(self, features_raw: list[Any]) -> list[FeatureSpec]:
        """Parse feature definitions."""
        if not isinstance(features_raw, list):
            raise FeatureDSLError("'features' must be a list.")

        features: list[FeatureSpec] = []
        for i, feat_dict in enumerate(features_raw):
            if not isinstance(feat_dict, dict):
                raise FeatureDSLError(
                    f"Feature at index {i} must be a dict, got {type(feat_dict).__name__}."
                )

            features.append(
                FeatureSpec(
                    name=feat_dict.get("name", ""),
                    source=feat_dict.get("source", ""),
                    transform=feat_dict.get("transform", ""),
                    params=feat_dict.get("params", {}),
                    dtype=feat_dict.get("dtype", "auto"),
                )
            )

        return features

    def _parse_selection(self, selection_raw: Any) -> SelectionSpec | None:
        """Parse selection spec."""
        if selection_raw is None:
            return None

        if not isinstance(selection_raw, dict):
            raise FeatureDSLError("'selection' must be a dict.")

        return SelectionSpec(
            method=selection_raw.get("method", "importance"),
            params=selection_raw.get("params", {}),
        )
