"""Tests for Feature Studio code export."""

from __future__ import annotations

import pytest

from forge.studio.export import CodeExporter, export_studio_to_dsl


@pytest.fixture
def sample_steps() -> list[dict]:
    return [
        {
            "name": "interactions",
            "transformer_class": "InteractionGenerator",
            "params": {"columns": ["price", "quantity"], "operations": ["multiply"]},
        },
        {
            "name": "encoding",
            "transformer_class": "TargetEncoder",
            "params": {"columns": ["category"]},
        },
    ]


class TestCodeExporter:
    def test_script_export(self, sample_steps: list[dict]) -> None:
        exporter = CodeExporter(style="script")
        code = exporter.export_pipeline(sample_steps, "my_pipeline")
        assert "ForgePipeline" in code
        assert "InteractionGenerator" in code
        assert "TargetEncoder" in code
        assert "my_pipeline" in code

    def test_function_export(self, sample_steps: list[dict]) -> None:
        exporter = CodeExporter(style="function")
        code = exporter.export_pipeline(sample_steps, "my_pipeline")
        assert "def create_my_pipeline" in code
        assert "ForgePipeline" in code

    def test_class_export(self, sample_steps: list[dict]) -> None:
        exporter = CodeExporter(style="class")
        code = exporter.export_pipeline(sample_steps, "my_pipeline")
        assert "class MyPipelineTransformer" in code
        assert "def fit" in code
        assert "def transform" in code

    def test_no_imports(self, sample_steps: list[dict]) -> None:
        exporter = CodeExporter(style="script")
        code = exporter.export_pipeline(sample_steps, include_imports=False)
        assert "import" not in code

    def test_export_to_file(self, sample_steps: list[dict], tmp_path: object) -> None:
        import pathlib

        filepath = pathlib.Path(str(tmp_path)) / "test_pipeline.py"
        exporter = CodeExporter(style="script")
        exporter.export_to_file(sample_steps, str(filepath))
        assert filepath.exists()
        content = filepath.read_text()
        assert "ForgePipeline" in content

    def test_empty_steps(self) -> None:
        exporter = CodeExporter()
        code = exporter.export_pipeline([], "empty_pipeline")
        assert "ForgePipeline" in code


class TestExportStudioToDSL:
    def test_basic_export(self, sample_steps: list[dict]) -> None:
        dsl = export_studio_to_dsl(sample_steps)
        assert "version" in dsl
        assert "pipeline" in dsl
        assert len(dsl["pipeline"]["steps"]) == 2

    def test_step_structure(self, sample_steps: list[dict]) -> None:
        dsl = export_studio_to_dsl(sample_steps)
        step = dsl["pipeline"]["steps"][0]
        assert step["name"] == "interactions"
        assert step["type"] == "InteractionGenerator"
        assert "columns" in step["params"]

    def test_empty_steps(self) -> None:
        dsl = export_studio_to_dsl([])
        assert dsl["pipeline"]["steps"] == []
