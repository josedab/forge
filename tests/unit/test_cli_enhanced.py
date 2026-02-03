"""Tests for the Forge CLI module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forge.cli import CLIResult, ForgeCLI
from forge.cli.main import build_parser, main


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV file."""
    df = pd.DataFrame({
        "num_a": np.random.randn(50),
        "num_b": np.random.uniform(0, 10, 50),
        "cat": np.random.choice(["x", "y", "z"], 50),
        "target": np.random.choice([0, 1], 50),
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


class TestForgeCLIInit:
    """Tests for ForgeCLI.init()."""

    def test_init_creates_project(self, tmp_dir: Path) -> None:
        cli = ForgeCLI(working_dir=tmp_dir)
        result = cli.init("myproject")
        assert result.success
        assert (tmp_dir / "myproject" / "forge.yaml").exists()
        assert (tmp_dir / "myproject" / "pipeline.py").exists()

    def test_init_existing_dir_fails(self, tmp_dir: Path) -> None:
        (tmp_dir / "existing").mkdir()
        cli = ForgeCLI(working_dir=tmp_dir)
        result = cli.init("existing")
        assert not result.success

    def test_init_creates_data_dir(self, tmp_dir: Path) -> None:
        cli = ForgeCLI(working_dir=tmp_dir)
        cli.init("proj")
        assert (tmp_dir / "proj" / "data").is_dir()


class TestForgeCLIValidate:
    """Tests for ForgeCLI.validate()."""

    def test_validate_valid_config(self, tmp_dir: Path) -> None:
        cli = ForgeCLI(working_dir=tmp_dir)
        cli.init("proj")
        cli_proj = ForgeCLI(working_dir=tmp_dir / "proj")
        result = cli_proj.validate()
        assert result.success

    def test_validate_missing_config(self, tmp_dir: Path) -> None:
        cli = ForgeCLI(working_dir=tmp_dir)
        result = cli.validate()
        assert not result.success


class TestForgeCLIInfo:
    """Tests for ForgeCLI.info()."""

    def test_info_in_project(self, tmp_dir: Path) -> None:
        cli = ForgeCLI(working_dir=tmp_dir)
        cli.init("proj")
        cli_proj = ForgeCLI(working_dir=tmp_dir / "proj")
        result = cli_proj.info()
        assert result.success
        assert result.data["name"] == "proj"

    def test_info_outside_project(self, tmp_dir: Path) -> None:
        cli = ForgeCLI(working_dir=tmp_dir)
        result = cli.info()
        assert not result.success


class TestCLIMainEntryPoint:
    """Tests for the main CLI entry point."""

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main(["--version"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "forge" in captured.out

    def test_no_command_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main([])
        assert ret == 0

    def test_analyze_command(
        self, sample_csv: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = main(["analyze", str(sample_csv)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "DATASET OVERVIEW" in captured.out
        assert "COLUMN TYPES" in captured.out

    def test_analyze_with_output(self, sample_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "report.json"
        ret = main(["analyze", str(sample_csv), "-o", str(output)])
        assert ret == 0
        assert output.exists()
        report = json.loads(output.read_text())
        assert "n_rows" in report
        assert report["n_rows"] == 50


class TestBuildParser:
    """Tests for parser construction."""

    def test_parser_has_subcommands(self) -> None:
        parser = build_parser()
        # Verify it doesn't error
        args = parser.parse_args(["analyze", "test.csv"])
        assert args.command == "analyze"
        assert args.data == "test.csv"

    def test_generate_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["generate", "test.csv", "--target", "y", "--max-features", "100"])
        assert args.command == "generate"
        assert args.target == "y"
        assert args.max_features == 100

    def test_select_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["select", "test.csv", "--method", "variance", "-k", "10"])
        assert args.command == "select"
        assert args.method == "variance"
        assert args.k == 10

    def test_benchmark_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["benchmark", "test.csv", "--runs", "5"])
        assert args.command == "benchmark"
        assert args.runs == 5


class TestCLIResult:
    """Tests for CLIResult dataclass."""

    def test_success_result(self) -> None:
        r = CLIResult(success=True, message="ok")
        assert r.success
        assert r.data == {}

    def test_failure_result(self) -> None:
        r = CLIResult(success=False, message="err", data={"code": 1})
        assert not r.success
        assert r.data["code"] == 1
