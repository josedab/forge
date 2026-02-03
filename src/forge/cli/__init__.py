"""Forge CLI — command-line interface for feature engineering.

Provides project scaffolding, pipeline execution, feature serving,
testing, and marketplace publishing commands.

Usage:
    forge init myproject
    forge run pipeline.py
    forge test
    forge serve --port 8000
    forge publish my-pack
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_TEMPLATE = {
    "forge.yaml": """# Forge Feature Engineering Project
name: {name}
version: 0.1.0

pipeline:
  generators:
    - type: numeric
      columns: auto
    - type: categorical
      columns: auto
  selectors:
    - type: variance
      threshold: 0.01

data:
  train: data/train.csv
  test: data/test.csv
  target: target
""",
    "pipeline.py": '''"""Auto-generated Forge pipeline."""

import pandas as pd
from forge import AutoFeatureTransformer

# Load data
X_train = pd.read_csv("data/train.csv")
y_train = X_train.pop("target")

# Feature engineering
transformer = AutoFeatureTransformer(max_features=50)
X_engineered = transformer.fit_transform(X_train, y_train)

print(f"Generated {{len(transformer.get_feature_names_out())}} features")
print(f"Shape: {{X_engineered.shape}}")
''',
    "tests/test_features.py": '''"""Feature validation tests."""

import pandas as pd
import pytest
from pathlib import Path


def test_data_exists():
    """Check that training data exists."""
    assert Path("data/train.csv").exists(), "Training data not found"


def test_pipeline_runs():
    """Check that pipeline completes without error."""
    from forge import AutoFeatureTransformer
    # Minimal smoke test
    X = pd.DataFrame({{"a": [1, 2, 3], "b": [4, 5, 6]}})
    y = pd.Series([0, 1, 0])
    t = AutoFeatureTransformer(max_features=10)
    result = t.fit_transform(X, y)
    assert result.shape[0] == 3
''',
}


@dataclass
class CLIResult:
    """Result of a CLI command."""

    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


class ForgeCLI:
    """Forge CLI command handler.

    Processes CLI commands programmatically, useful for both the
    actual CLI entry point and testing.
    """

    def __init__(self, working_dir: str | Path | None = None) -> None:
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

    def init(self, name: str, template: str = "default") -> CLIResult:
        """Initialize a new Forge project.

        Args:
            name: Project name / directory name.
            template: Project template to use.

        Returns:
            CLIResult with creation status.
        """
        project_dir = self.working_dir / name

        if project_dir.exists():
            return CLIResult(
                success=False,
                message=f"Directory '{name}' already exists",
            )

        try:
            project_dir.mkdir(parents=True)
            (project_dir / "data").mkdir()
            (project_dir / "tests").mkdir()

            for filename, content in PROJECT_TEMPLATE.items():
                file_path = project_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content.format(name=name))

            return CLIResult(
                success=True,
                message=f"Created Forge project '{name}'",
                data={"path": str(project_dir), "files": list(PROJECT_TEMPLATE.keys())},
            )
        except OSError as e:
            return CLIResult(success=False, message=f"Failed to create project: {e}")

    def run(self, script: str = "pipeline.py") -> CLIResult:
        """Run a pipeline script.

        Args:
            script: Path to pipeline script.

        Returns:
            CLIResult with execution status.
        """
        script_path = self.working_dir / script
        if not script_path.exists():
            return CLIResult(
                success=False,
                message=f"Script '{script}' not found",
            )

        try:
            code = script_path.read_text()
            exec(compile(code, str(script_path), "exec"), {"__name__": "__main__"})  # noqa: S102
            return CLIResult(
                success=True,
                message=f"Pipeline '{script}' completed successfully",
            )
        except Exception as e:
            return CLIResult(success=False, message=f"Pipeline failed: {e}")

    def test(self, test_dir: str = "tests") -> CLIResult:
        """Run feature tests.

        Args:
            test_dir: Directory containing tests.

        Returns:
            CLIResult with test results.
        """
        test_path = self.working_dir / test_dir
        if not test_path.exists():
            return CLIResult(
                success=False,
                message=f"Test directory '{test_dir}' not found",
            )

        try:
            import pytest
            exit_code = pytest.main([str(test_path), "-q", "--tb=short", "--no-header"])
            return CLIResult(
                success=exit_code == 0,
                message="Tests passed" if exit_code == 0 else f"Tests failed (exit code {exit_code})",
                data={"exit_code": exit_code},
            )
        except ImportError:
            return CLIResult(
                success=False,
                message="pytest not installed. Run: pip install pytest",
            )

    def validate(self, config_file: str = "forge.yaml") -> CLIResult:
        """Validate a Forge project configuration.

        Args:
            config_file: Path to forge.yaml.

        Returns:
            CLIResult with validation results.
        """
        config_path = self.working_dir / config_file
        if not config_path.exists():
            return CLIResult(
                success=False,
                message=f"Config file '{config_file}' not found",
            )

        try:
            content = config_path.read_text()
            issues: list[str] = []

            if "name:" not in content:
                issues.append("Missing 'name' field")
            if "pipeline:" not in content:
                issues.append("Missing 'pipeline' section")

            if issues:
                return CLIResult(
                    success=False,
                    message=f"Validation failed: {'; '.join(issues)}",
                    data={"issues": issues},
                )

            return CLIResult(
                success=True,
                message="Configuration is valid",
            )
        except Exception as e:
            return CLIResult(success=False, message=f"Validation error: {e}")

    def info(self) -> CLIResult:
        """Show project information."""
        config_path = self.working_dir / "forge.yaml"
        if not config_path.exists():
            return CLIResult(
                success=False,
                message="Not a Forge project (no forge.yaml found)",
            )

        try:
            import forge
            content = config_path.read_text()
            name = "unknown"
            for line in content.splitlines():
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                    break

            py_files = list(self.working_dir.glob("*.py"))
            test_files = list((self.working_dir / "tests").glob("*.py")) if (self.working_dir / "tests").exists() else []

            return CLIResult(
                success=True,
                message=f"Forge project: {name}",
                data={
                    "name": name,
                    "forge_version": forge.__version__,
                    "pipeline_files": len(py_files),
                    "test_files": len(test_files),
                },
            )
        except Exception as e:
            return CLIResult(success=False, message=f"Error: {e}")

    def serve_config(self, port: int = 8000, host: str = "0.0.0.0") -> CLIResult:
        """Generate serving configuration.

        Args:
            port: Port for the feature server.
            host: Host to bind to.

        Returns:
            CLIResult with serving config.
        """
        config = {
            "host": host,
            "port": port,
            "workers": 1,
            "pipeline": "pipeline.py",
        }

        config_path = self.working_dir / "serve_config.json"
        config_path.write_text(json.dumps(config, indent=2))

        return CLIResult(
            success=True,
            message=f"Serving config written to {config_path}",
            data=config,
        )
