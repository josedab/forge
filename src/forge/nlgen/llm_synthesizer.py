"""LLM-backed natural language feature synthesis.

Provides LLM-enhanced feature synthesis when an API key is available,
falling back to rule-based parsing otherwise.
"""

from __future__ import annotations

import json
import logging
import re

import pandas as pd

from forge.nlgen import ColumnSchema, NLFeatureSynthesizer, SynthesisResult

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a feature engineering expert. Given a dataset schema and a natural language description, generate Python code that creates the described feature.

Rules:
1. Use pandas operations on a DataFrame called 'df'
2. Store new features in a DataFrame called 'result' (start with `result = df.copy()`)
3. Use numpy (imported as np) for math operations
4. Handle edge cases (division by zero, null values)
5. Return ONLY valid Python code, no explanations
6. Feature names should be descriptive snake_case

Response format (JSON):
{
  "code": "result = df.copy()\\nresult['feature_name'] = ...",
  "feature_names": ["feature_name"],
  "warnings": []
}"""


class LLMFeatureSynthesizer:
    """LLM-enhanced natural language to feature code synthesizer.

    Uses an LLM API (OpenAI or Anthropic) for complex descriptions and
    falls back to rule-based parsing for simple patterns.

    Args:
        schema: Column schema.
        provider: LLM provider ('openai' or 'anthropic').
        api_key: API key for the provider.
        model: Model name.
        fallback_to_rules: Fall back to rules if LLM is unavailable.
        validate: Validate generated code.
        temperature: LLM temperature (0-1).

    Example:
        >>> synth = LLMFeatureSynthesizer(
        ...     schema={"income": "numeric", "debt": "numeric"},
        ...     provider="openai",
        ...     api_key="sk-..."
        ... )
        >>> result = synth.synthesize("debt-to-income ratio, binned into quintiles")
        >>> df_new = result.execute(df)
    """

    def __init__(
        self,
        schema: dict[str, str] | list[ColumnSchema] | None = None,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        fallback_to_rules: bool = True,
        validate: bool = True,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._default_model(provider)
        self.fallback_to_rules = fallback_to_rules
        self.validate = validate
        self.temperature = temperature

        self._schema: dict[str, str] = {}
        if isinstance(schema, dict):
            self._schema = schema
        elif isinstance(schema, list):
            self._schema = {cs.name: cs.dtype for cs in schema}

        self._rule_synth = NLFeatureSynthesizer(schema=schema, validate=validate)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        provider: str = "openai",
        api_key: str | None = None,
    ) -> LLMFeatureSynthesizer:
        """Create synthesizer from a DataFrame."""
        schema: dict[str, str] = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                schema[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                schema[col] = "temporal"
            else:
                schema[col] = "categorical"
        return cls(schema=schema, provider=provider, api_key=api_key)

    def synthesize(self, description: str) -> SynthesisResult:
        """Synthesize feature code from natural language.

        Tries LLM first, falls back to rules if unavailable or if LLM fails.

        Args:
            description: Natural language feature description.

        Returns:
            SynthesisResult with generated code.
        """
        # Try rule-based first for simple patterns
        rule_result = self._rule_synth.synthesize(description)
        if rule_result.success and rule_result.feature_names:
            return rule_result

        # Try LLM if API key is available
        if self.api_key:
            try:
                llm_result = self._synthesize_with_llm(description)
                if llm_result.success:
                    return llm_result
            except Exception as e:
                logger.warning("LLM synthesis failed: %s", e)

        # Fallback to rule result (even if failed)
        if self.fallback_to_rules:
            return rule_result

        return SynthesisResult(
            description=description,
            code="",
            feature_names=[],
            success=False,
            error="LLM unavailable and no rule pattern matched",
        )

    def synthesize_batch(self, descriptions: list[str]) -> list[SynthesisResult]:
        """Synthesize multiple features."""
        return [self.synthesize(desc) for desc in descriptions]

    def _synthesize_with_llm(self, description: str) -> SynthesisResult:
        """Use LLM to generate feature code."""
        schema_str = json.dumps(self._schema, indent=2)
        user_prompt = (
            f"Dataset schema:\n{schema_str}\n\n"
            f"Feature request: {description}"
        )

        response_text = self._call_llm(user_prompt)
        return self._parse_llm_response(description, response_text)

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM provider API."""
        if self.provider == "openai":
            return self._call_openai(user_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI API."""
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, user_prompt: str) -> str:
        """Call Anthropic API."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _parse_llm_response(
        self, description: str, response: str
    ) -> SynthesisResult:
        """Parse the LLM response into a SynthesisResult."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                code = data.get("code", "")
                feature_names = data.get("feature_names", [])
                warnings = data.get("warnings", [])
            else:
                # Treat entire response as code
                code = response.strip()
                feature_names = self._extract_feature_names(code)
                warnings = ["LLM response was not in expected JSON format"]

            if self.validate and code:
                validation_warns = self._validate_code(code)
                warnings.extend(validation_warns)

            return SynthesisResult(
                description=description,
                code=code,
                feature_names=feature_names,
                success=bool(code),
                warnings=warnings,
            )
        except (json.JSONDecodeError, KeyError) as e:
            return SynthesisResult(
                description=description,
                code=response.strip(),
                feature_names=[],
                success=False,
                error=f"Failed to parse LLM response: {e}",
            )

    def _extract_feature_names(self, code: str) -> list[str]:
        """Extract feature names from code like result['name'] = ..."""
        return re.findall(r"result\[(['\"])(.+?)\1\]\s*=", code)

    def _validate_code(self, code: str) -> list[str]:
        """Basic validation of generated code."""
        warnings: list[str] = []
        dangerous = ["import os", "import sys", "subprocess", "__import__", "eval(", "exec("]
        for pattern in dangerous:
            if pattern in code:
                warnings.append(f"Potentially unsafe pattern: {pattern}")
        return warnings

    @staticmethod
    def _default_model(provider: str) -> str:
        if provider == "openai":
            return "gpt-4o-mini"
        elif provider == "anthropic":
            return "claude-sonnet-4-20250514"
        return "gpt-4o-mini"
