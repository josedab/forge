"""Declarative feature DSL for defining feature pipelines.

Provides a YAML/dict-based declarative language for defining feature
engineering pipelines that can be compiled to ForgePipeline objects.
"""

from __future__ import annotations

from forge.dsl.compiler import DSLCompiler
from forge.dsl.parser import DSLParser, FeatureDSLError, FeatureSpec, PipelineSpec

__all__ = [
    "DSLCompiler",
    "DSLParser",
    "FeatureDSLError",
    "FeatureSpec",
    "PipelineSpec",
]
