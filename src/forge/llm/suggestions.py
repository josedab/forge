"""Feature suggestion pipeline using LLMs."""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from forge.exceptions import ConfigurationError, MissingDependencyError
from forge.llm.metadata import DatasetMetadata
from forge.types import ColumnType


@dataclass
class FeatureSuggestion:
    """A suggested feature from the LLM."""

    name: str
    description: str
    source_columns: list[str]
    transformation: str
    rationale: str
    confidence: float = 0.8
    category: str = "general"
    implementation_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "source_columns": self.source_columns,
            "transformation": self.transformation,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "category": self.category,
            "implementation_hint": self.implementation_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSuggestion:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            source_columns=data.get("source_columns", []),
            transformation=data.get("transformation", "unknown"),
            rationale=data.get("rationale", ""),
            confidence=data.get("confidence", 0.8),
            category=data.get("category", "general"),
            implementation_hint=data.get("implementation_hint"),
        )


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            The generated text response.
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model to use.
            temperature: Sampling temperature.
        """
        try:
            import openai
        except ImportError:
            raise MissingDependencyError("openai", "LLM feature discovery")

        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigurationError("OpenAI API key not provided")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate response using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.3,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Model to use.
            temperature: Sampling temperature.
        """
        try:
            import anthropic
        except ImportError:
            raise MissingDependencyError("anthropic", "LLM feature discovery")

        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ConfigurationError("Anthropic API key not provided")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate response using Anthropic API."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class LocalModelProvider(LLMProvider):
    """Provider for local models via Ollama."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        """Initialize local model provider.

        Args:
            model: Model name in Ollama.
            base_url: Ollama server URL.
        """
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate response using local Ollama instance."""
        import urllib.error
        import urllib.request

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        data = json.dumps({
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode())
                return result.get("response", "")
        except urllib.error.URLError as e:
            raise ConfigurationError(f"Failed to connect to Ollama: {e}")


class SuggestionCache:
    """Simple in-memory cache for LLM suggestions."""

    def __init__(self, ttl_seconds: int = 900) -> None:
        """Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default 15 minutes).
        """
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, list[FeatureSuggestion]]] = {}

    def get(self, key: str) -> list[FeatureSuggestion] | None:
        """Get cached suggestions if not expired."""
        if key in self._cache:
            timestamp, suggestions = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return suggestions
            del self._cache[key]
        return None

    def set(self, key: str, suggestions: list[FeatureSuggestion]) -> None:
        """Cache suggestions."""
        self._cache[key] = (time.time(), suggestions)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


class FeatureSuggester:
    """Suggests features using an LLM based on dataset metadata.

    This class takes dataset metadata and uses an LLM to suggest
    potentially valuable features. It supports multiple LLM providers
    and includes caching to reduce API costs.

    Example:
        >>> from forge.llm import MetadataExtractor, FeatureSuggester
        >>> extractor = MetadataExtractor()
        >>> metadata = extractor.extract(X, y)
        >>> suggester = FeatureSuggester(provider="openai")
        >>> suggestions = suggester.suggest(metadata)
    """

    SYSTEM_PROMPT = """You are an expert data scientist specializing in feature engineering.
Your task is to analyze dataset metadata and suggest valuable features that could improve
machine learning model performance.

Guidelines:
1. Suggest features that capture meaningful relationships in the data
2. Consider domain-specific transformations based on column names and patterns
3. Look for interaction effects between related columns
4. Suggest temporal features when datetime columns exist
5. Consider aggregations and statistical transformations
6. Each suggestion should be actionable and implementable
7. Provide confidence scores based on how likely the feature is to be valuable

Output your suggestions in JSON format as a list of objects with these fields:
- name: Feature name (snake_case)
- description: Brief description of what the feature captures
- source_columns: List of column names used
- transformation: Type of transformation (interaction, ratio, aggregation, temporal, etc.)
- rationale: Why this feature might be valuable
- confidence: Float 0-1 indicating your confidence
- category: Category (numeric, temporal, text, categorical)
- implementation_hint: Optional code hint

Return ONLY valid JSON, no other text."""

    def __init__(
        self,
        provider: str | LLMProvider = "openai",
        api_key: str | None = None,
        model: str | None = None,
        cache: SuggestionCache | None = None,
        max_suggestions: int = 20,
    ) -> None:
        """Initialize the feature suggester.

        Args:
            provider: LLM provider name ("openai", "anthropic", "local") or instance.
            api_key: API key for the provider.
            model: Model name override.
            cache: Optional cache instance.
            max_suggestions: Maximum suggestions to return.
        """
        self.max_suggestions = max_suggestions
        self.cache = cache or SuggestionCache()

        if isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            self.provider = self._create_provider(provider, api_key, model)

    def _create_provider(
        self, provider_name: str, api_key: str | None, model: str | None
    ) -> LLMProvider:
        """Create an LLM provider instance."""
        if provider_name == "openai":
            kwargs: dict[str, Any] = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return OpenAIProvider(**kwargs)
        elif provider_name == "anthropic":
            kwargs = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return AnthropicProvider(**kwargs)
        elif provider_name == "local":
            kwargs = {}
            if model:
                kwargs["model"] = model
            return LocalModelProvider(**kwargs)
        else:
            raise ConfigurationError(
                f"Unknown provider: {provider_name}. "
                f"Supported: openai, anthropic, local"
            )

    def suggest(
        self,
        metadata: DatasetMetadata,
        use_cache: bool = True,
        additional_context: str | None = None,
    ) -> list[FeatureSuggestion]:
        """Generate feature suggestions based on metadata.

        Args:
            metadata: Dataset metadata from MetadataExtractor.
            use_cache: Whether to use caching.
            additional_context: Extra context about the domain/problem.

        Returns:
            List of feature suggestions.
        """
        cache_key = self._compute_cache_key(metadata, additional_context)

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        prompt = self._build_prompt(metadata, additional_context)
        response = self.provider.generate(prompt, system_prompt=self.SYSTEM_PROMPT)
        suggestions = self._parse_suggestions(response, metadata)

        if use_cache:
            self.cache.set(cache_key, suggestions)

        return suggestions

    def _compute_cache_key(
        self, metadata: DatasetMetadata, additional_context: str | None
    ) -> str:
        """Compute cache key from metadata and context."""
        content = metadata.metadata_hash
        if additional_context:
            content += hashlib.md5(additional_context.encode()).hexdigest()[:8]
        return content

    def _build_prompt(
        self, metadata: DatasetMetadata, additional_context: str | None
    ) -> str:
        """Build the prompt for the LLM."""
        prompt_parts = [
            "Analyze this dataset and suggest valuable features:\n",
            metadata.to_prompt_string(),
            f"\nSuggest up to {self.max_suggestions} features.",
        ]

        if additional_context:
            prompt_parts.insert(1, f"\nContext: {additional_context}\n")

        return "\n".join(prompt_parts)

    def _parse_suggestions(
        self, response: str, metadata: DatasetMetadata
    ) -> list[FeatureSuggestion]:
        """Parse LLM response into feature suggestions."""
        suggestions = []

        # Try to extract JSON from response
        json_match = re.search(r"\[[\s\S]*\]", response)
        if not json_match:
            return self._fallback_suggestions(metadata)

        try:
            raw_suggestions = json.loads(json_match.group())
        except json.JSONDecodeError:
            return self._fallback_suggestions(metadata)

        available_columns = set(metadata.columns.keys())

        for raw in raw_suggestions:
            if not isinstance(raw, dict):
                continue

            # Validate source columns exist
            source_cols = raw.get("source_columns", [])
            valid_sources = [c for c in source_cols if c in available_columns]

            if not valid_sources and source_cols:
                continue  # Skip if no valid source columns

            suggestion = FeatureSuggestion(
                name=raw.get("name", "unnamed_feature"),
                description=raw.get("description", ""),
                source_columns=valid_sources,
                transformation=raw.get("transformation", "unknown"),
                rationale=raw.get("rationale", ""),
                confidence=min(max(float(raw.get("confidence", 0.5)), 0.0), 1.0),
                category=raw.get("category", "general"),
                implementation_hint=raw.get("implementation_hint"),
            )
            suggestions.append(suggestion)

            if len(suggestions) >= self.max_suggestions:
                break

        return suggestions if suggestions else self._fallback_suggestions(metadata)

    def _fallback_suggestions(
        self, metadata: DatasetMetadata
    ) -> list[FeatureSuggestion]:
        """Generate rule-based suggestions when LLM parsing fails."""
        suggestions = []
        numeric_cols = [
            col for col, info in metadata.columns.items()
            if info.inferred_type == ColumnType.NUMERIC
        ]
        datetime_cols = [
            col for col, info in metadata.columns.items()
            if info.inferred_type == ColumnType.DATETIME
        ]
        categorical_cols = [
            col for col, info in metadata.columns.items()
            if info.inferred_type == ColumnType.CATEGORICAL
        ]

        # Suggest interactions for numeric columns
        for i, col1 in enumerate(numeric_cols[:5]):
            for col2 in numeric_cols[i + 1:6]:
                suggestions.append(FeatureSuggestion(
                    name=f"{col1}_x_{col2}",
                    description=f"Interaction between {col1} and {col2}",
                    source_columns=[col1, col2],
                    transformation="interaction",
                    rationale="Numeric interactions often capture non-linear relationships",
                    confidence=0.6,
                    category="numeric",
                ))

                suggestions.append(FeatureSuggestion(
                    name=f"{col1}_div_{col2}",
                    description=f"Ratio of {col1} to {col2}",
                    source_columns=[col1, col2],
                    transformation="ratio",
                    rationale="Ratios normalize scale and capture relative relationships",
                    confidence=0.6,
                    category="numeric",
                ))

        # Suggest log transforms for skewed columns
        for col, info in metadata.columns.items():
            if info.inferred_type == ColumnType.NUMERIC:
                skewness = info.statistics.get("skewness")
                if skewness and abs(skewness) > 1:
                    suggestions.append(FeatureSuggestion(
                        name=f"{col}_log",
                        description=f"Log transform of {col}",
                        source_columns=[col],
                        transformation="log",
                        rationale=f"Column is skewed (skewness={skewness:.2f})",
                        confidence=0.7,
                        category="numeric",
                    ))

        # Suggest datetime extractions
        for col in datetime_cols[:3]:
            for component in ["year", "month", "dayofweek", "hour"]:
                suggestions.append(FeatureSuggestion(
                    name=f"{col}_{component}",
                    description=f"Extract {component} from {col}",
                    source_columns=[col],
                    transformation="datetime_extract",
                    rationale=f"Temporal {component} may have predictive patterns",
                    confidence=0.7,
                    category="temporal",
                ))

        # Suggest target encoding for categorical columns
        for col in categorical_cols[:3]:
            suggestions.append(FeatureSuggestion(
                name=f"{col}_target_encoded",
                description=f"Target encoding for {col}",
                source_columns=[col],
                transformation="target_encode",
                rationale="Target encoding captures target relationship for categories",
                confidence=0.7,
                category="categorical",
            ))

        return suggestions[:self.max_suggestions]


def create_suggester(
    provider: str = "openai",
    api_key: str | None = None,
    **kwargs: Any,
) -> FeatureSuggester:
    """Factory function to create a feature suggester.

    Args:
        provider: LLM provider name.
        api_key: API key for the provider.
        **kwargs: Additional arguments for FeatureSuggester.

    Returns:
        Configured FeatureSuggester instance.
    """
    return FeatureSuggester(provider=provider, api_key=api_key, **kwargs)
