"""Feature explanation generator using LLMs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from forge.exceptions import ConfigurationError
from forge.llm.suggestions import AnthropicProvider, FeatureSuggestion, LLMProvider, OpenAIProvider


@dataclass
class FeatureExplanation:
    """Detailed explanation of a feature."""

    feature_name: str
    description: str
    source_columns: list[str]
    transformation: str
    rationale: str
    domain_interpretation: str
    usage_guidance: str
    potential_issues: list[str] = field(default_factory=list)
    related_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "description": self.description,
            "source_columns": self.source_columns,
            "transformation": self.transformation,
            "rationale": self.rationale,
            "domain_interpretation": self.domain_interpretation,
            "usage_guidance": self.usage_guidance,
            "potential_issues": self.potential_issues,
            "related_features": self.related_features,
        }

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"## {self.feature_name}",
            "",
            f"**Description:** {self.description}",
            "",
            f"**Source Columns:** {', '.join(self.source_columns)}",
            "",
            f"**Transformation:** {self.transformation}",
            "",
            "### Why This Feature Matters",
            self.rationale,
            "",
            "### Domain Interpretation",
            self.domain_interpretation,
            "",
            "### Usage Guidance",
            self.usage_guidance,
        ]

        if self.potential_issues:
            lines.extend([
                "",
                "### Potential Issues",
            ])
            for issue in self.potential_issues:
                lines.append(f"- {issue}")

        if self.related_features:
            lines.extend([
                "",
                f"**Related Features:** {', '.join(self.related_features)}",
            ])

        return "\n".join(lines)


@dataclass
class ExplanationReport:
    """Report containing explanations for multiple features."""

    title: str
    explanations: list[FeatureExplanation]
    summary: str
    recommendations: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Convert full report to markdown."""
        lines = [
            f"# {self.title}",
            "",
            "## Summary",
            self.summary,
            "",
        ]

        if self.recommendations:
            lines.extend([
                "## Recommendations",
            ])
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        lines.append("## Feature Details")
        lines.append("")

        for explanation in self.explanations:
            lines.append(explanation.to_markdown())
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Convert to HTML format."""
        import html as html_module

        def escape(text: str) -> str:
            return html_module.escape(text)

        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1 { color: #2c3e50; }",
            "h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
            "h3 { color: #7f8c8d; }",
            ".feature-card { background: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 8px; }",
            ".meta { color: #666; margin: 10px 0; }",
            ".issues { color: #e74c3c; }",
            ".recommendations { background: #e8f6f3; padding: 15px; border-radius: 5px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{escape(self.title)}</h1>",
            "<h2>Summary</h2>",
            f"<p>{escape(self.summary)}</p>",
        ]

        if self.recommendations:
            lines.append("<div class='recommendations'>")
            lines.append("<h2>Recommendations</h2>")
            lines.append("<ul>")
            for rec in self.recommendations:
                lines.append(f"<li>{escape(rec)}</li>")
            lines.append("</ul>")
            lines.append("</div>")

        lines.append("<h2>Feature Details</h2>")

        for exp in self.explanations:
            lines.extend([
                "<div class='feature-card'>",
                f"<h2>{escape(exp.feature_name)}</h2>",
                f"<p><strong>Description:</strong> {escape(exp.description)}</p>",
                f"<p class='meta'><strong>Source:</strong> {escape(', '.join(exp.source_columns))} | ",
                f"<strong>Transformation:</strong> {escape(exp.transformation)}</p>",
                "<h3>Why This Feature Matters</h3>",
                f"<p>{escape(exp.rationale)}</p>",
                "<h3>Domain Interpretation</h3>",
                f"<p>{escape(exp.domain_interpretation)}</p>",
                "<h3>Usage Guidance</h3>",
                f"<p>{escape(exp.usage_guidance)}</p>",
            ])

            if exp.potential_issues:
                lines.append("<h3 class='issues'>Potential Issues</h3>")
                lines.append("<ul>")
                for issue in exp.potential_issues:
                    lines.append(f"<li>{escape(issue)}</li>")
                lines.append("</ul>")

            lines.append("</div>")

        lines.extend([
            "</body>",
            "</html>",
        ])

        return "\n".join(lines)


class FeatureExplainer:
    """Generates natural language explanations for features.

    Uses an LLM to create human-readable explanations of features,
    their purpose, and how they should be interpreted.

    Example:
        >>> from forge.llm import FeatureExplainer, FeatureSuggestion
        >>> explainer = FeatureExplainer(provider="openai")
        >>> suggestions = [...]  # Feature suggestions
        >>> report = explainer.explain_features(suggestions)
        >>> print(report.to_markdown())
    """

    SYSTEM_PROMPT = """You are a data science expert explaining feature engineering to stakeholders.
Your explanations should be clear, non-technical where possible, and focus on business value.

For each feature, provide:
1. A clear description of what the feature represents
2. Why this feature might be valuable for prediction
3. How to interpret the feature values in a business context
4. Any potential issues or caveats
5. Guidance on how to use the feature

Output valid JSON with this structure:
{
    "explanations": [
        {
            "feature_name": "...",
            "description": "...",
            "rationale": "...",
            "domain_interpretation": "...",
            "usage_guidance": "...",
            "potential_issues": ["...", "..."],
            "related_features": ["...", "..."]
        }
    ],
    "summary": "Overall summary of the feature set",
    "recommendations": ["...", "..."]
}"""

    def __init__(
        self,
        provider: str | LLMProvider = "openai",
        api_key: str | None = None,
        model: str | None = None,
        verbose: int = 0,
    ) -> None:
        """Initialize the feature explainer.

        Args:
            provider: LLM provider name or instance.
            api_key: API key for the provider.
            model: Model name override.
            verbose: Verbosity level.
        """
        self.verbose = verbose

        if isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            self.provider = self._create_provider(provider, api_key, model)

    def _create_provider(
        self, provider_name: str, api_key: str | None, model: str | None
    ) -> LLMProvider:
        """Create an LLM provider instance."""
        if provider_name == "openai":
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if model:
                kwargs["model"] = model
            return OpenAIProvider(**kwargs)
        elif provider_name == "anthropic":
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if model:
                kwargs["model"] = model
            return AnthropicProvider(**kwargs)
        else:
            raise ConfigurationError(f"Unknown provider: {provider_name}")

    def explain_features(
        self,
        suggestions: list[FeatureSuggestion],
        domain_context: str | None = None,
        target_audience: str = "data scientists",
    ) -> ExplanationReport:
        """Generate explanations for a list of feature suggestions.

        Args:
            suggestions: List of feature suggestions to explain.
            domain_context: Additional context about the domain.
            target_audience: Who will read the explanations.

        Returns:
            ExplanationReport with detailed explanations.
        """
        if not suggestions:
            return ExplanationReport(
                title="Feature Explanation Report",
                explanations=[],
                summary="No features to explain.",
                recommendations=[],
            )

        prompt = self._build_prompt(suggestions, domain_context, target_audience)
        response = self.provider.generate(prompt, system_prompt=self.SYSTEM_PROMPT)

        return self._parse_explanations(response, suggestions)

    def explain_single_feature(
        self,
        suggestion: FeatureSuggestion,
        feature_values: pd.Series | None = None,
        domain_context: str | None = None,
    ) -> FeatureExplanation:
        """Generate detailed explanation for a single feature.

        Args:
            suggestion: Feature suggestion to explain.
            feature_values: Optional actual feature values for statistics.
            domain_context: Additional domain context.

        Returns:
            FeatureExplanation with detailed information.
        """
        prompt_parts = [
            "Explain this feature in detail:\n",
            f"Name: {suggestion.name}",
            f"Description: {suggestion.description}",
            f"Source columns: {', '.join(suggestion.source_columns)}",
            f"Transformation: {suggestion.transformation}",
            f"Rationale: {suggestion.rationale}",
        ]

        if feature_values is not None:
            stats = feature_values.describe()
            prompt_parts.append(f"\nFeature statistics:\n{stats.to_string()}")

        if domain_context:
            prompt_parts.append(f"\nDomain context: {domain_context}")

        prompt = "\n".join(prompt_parts)
        response = self.provider.generate(prompt, system_prompt=self.SYSTEM_PROMPT)

        # Parse single explanation
        report = self._parse_explanations(response, [suggestion])
        if report.explanations:
            return report.explanations[0]

        # Fallback explanation
        return FeatureExplanation(
            feature_name=suggestion.name,
            description=suggestion.description,
            source_columns=suggestion.source_columns,
            transformation=suggestion.transformation,
            rationale=suggestion.rationale,
            domain_interpretation="Interpretation not available.",
            usage_guidance="Use standard feature engineering practices.",
        )

    def _build_prompt(
        self,
        suggestions: list[FeatureSuggestion],
        domain_context: str | None,
        target_audience: str,
    ) -> str:
        """Build prompt for explaining features."""
        lines = [
            f"Explain these features for {target_audience}:\n",
        ]

        if domain_context:
            lines.append(f"Domain context: {domain_context}\n")

        lines.append("Features to explain:")
        for i, s in enumerate(suggestions, 1):
            lines.extend([
                f"\n{i}. {s.name}",
                f"   Description: {s.description}",
                f"   Sources: {', '.join(s.source_columns)}",
                f"   Transformation: {s.transformation}",
                f"   Original rationale: {s.rationale}",
            ])

        return "\n".join(lines)

    def _parse_explanations(
        self, response: str, suggestions: list[FeatureSuggestion]
    ) -> ExplanationReport:
        """Parse LLM response into ExplanationReport."""
        explanations = []

        # Try to extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())

                # Parse explanations
                for exp_data in data.get("explanations", []):
                    # Find matching suggestion
                    feature_name = exp_data.get("feature_name", "")
                    matching_suggestion = None
                    for s in suggestions:
                        if s.name == feature_name or feature_name in s.name:
                            matching_suggestion = s
                            break

                    explanation = FeatureExplanation(
                        feature_name=feature_name,
                        description=exp_data.get("description", ""),
                        source_columns=(
                            matching_suggestion.source_columns
                            if matching_suggestion
                            else []
                        ),
                        transformation=(
                            matching_suggestion.transformation
                            if matching_suggestion
                            else "unknown"
                        ),
                        rationale=exp_data.get("rationale", ""),
                        domain_interpretation=exp_data.get("domain_interpretation", ""),
                        usage_guidance=exp_data.get("usage_guidance", ""),
                        potential_issues=exp_data.get("potential_issues", []),
                        related_features=exp_data.get("related_features", []),
                    )
                    explanations.append(explanation)

                return ExplanationReport(
                    title="Feature Explanation Report",
                    explanations=explanations,
                    summary=data.get("summary", ""),
                    recommendations=data.get("recommendations", []),
                )

            except json.JSONDecodeError:
                pass

        # Fallback: create basic explanations from suggestions
        for s in suggestions:
            explanations.append(FeatureExplanation(
                feature_name=s.name,
                description=s.description,
                source_columns=s.source_columns,
                transformation=s.transformation,
                rationale=s.rationale,
                domain_interpretation="Analysis in progress.",
                usage_guidance="Standard feature usage applies.",
            ))

        return ExplanationReport(
            title="Feature Explanation Report",
            explanations=explanations,
            summary=f"Generated explanations for {len(suggestions)} features.",
            recommendations=[],
        )


def generate_feature_documentation(
    suggestions: list[FeatureSuggestion],
    provider: str = "openai",
    api_key: str | None = None,
    output_format: str = "markdown",
    domain_context: str | None = None,
) -> str:
    """Generate documentation for features.

    Convenience function to generate feature documentation.

    Args:
        suggestions: Feature suggestions to document.
        provider: LLM provider name.
        api_key: API key for the provider.
        output_format: Output format ("markdown" or "html").
        domain_context: Additional domain context.

    Returns:
        Formatted documentation string.
    """
    explainer = FeatureExplainer(provider=provider, api_key=api_key)
    report = explainer.explain_features(suggestions, domain_context=domain_context)

    if output_format == "html":
        return report.to_html()
    return report.to_markdown()
