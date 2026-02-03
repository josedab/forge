"""Tests for Forge CLI."""

from __future__ import annotations

from forge.cli import CLIResult, ForgeCLI


class TestForgeCLI:
    def test_init_project(self, tmp_path):
        cli = ForgeCLI(working_dir=tmp_path)
        result = cli.init("myproject")
        assert result.success
        assert (tmp_path / "myproject" / "forge.yaml").exists()
        assert (tmp_path / "myproject" / "pipeline.py").exists()
        assert (tmp_path / "myproject" / "tests" / "test_features.py").exists()
        assert (tmp_path / "myproject" / "data").exists()

    def test_init_duplicate(self, tmp_path):
        cli = ForgeCLI(working_dir=tmp_path)
        cli.init("myproject")
        result = cli.init("myproject")
        assert not result.success
        assert "already exists" in result.message

    def test_validate_valid(self, tmp_path):
        cli = ForgeCLI(working_dir=tmp_path)
        cli.init("proj")
        proj_cli = ForgeCLI(working_dir=tmp_path / "proj")
        result = proj_cli.validate()
        assert result.success

    def test_validate_missing_config(self, tmp_path):
        cli = ForgeCLI(working_dir=tmp_path)
        result = cli.validate()
        assert not result.success
        assert "not found" in result.message

    def test_validate_bad_config(self, tmp_path):
        (tmp_path / "forge.yaml").write_text("random: stuff\n")
        cli = ForgeCLI(working_dir=tmp_path)
        result = cli.validate()
        assert not result.success
        assert "name" in result.message or "pipeline" in result.message

    def test_info_no_project(self, tmp_path):
        cli = ForgeCLI(working_dir=tmp_path)
        result = cli.info()
        assert not result.success

    def test_info_with_project(self, tmp_path):
        cli = ForgeCLI(working_dir=tmp_path)
        cli.init("proj")
        proj_cli = ForgeCLI(working_dir=tmp_path / "proj")
        result = proj_cli.info()
        assert result.success
        assert result.data["name"] == "proj"

    def test_serve_config(self, tmp_path):
        cli = ForgeCLI(working_dir=tmp_path)
        result = cli.serve_config(port=9000)
        assert result.success
        assert result.data["port"] == 9000
        assert (tmp_path / "serve_config.json").exists()

    def test_run_missing_script(self, tmp_path):
        cli = ForgeCLI(working_dir=tmp_path)
        result = cli.run("nonexistent.py")
        assert not result.success

    def test_run_script(self, tmp_path):
        (tmp_path / "hello.py").write_text("x = 1 + 1\n")
        cli = ForgeCLI(working_dir=tmp_path)
        result = cli.run("hello.py")
        assert result.success

    def test_run_failing_script(self, tmp_path):
        (tmp_path / "bad.py").write_text("raise RuntimeError('boom')\n")
        cli = ForgeCLI(working_dir=tmp_path)
        result = cli.run("bad.py")
        assert not result.success
        assert "boom" in result.message

    def test_test_missing_dir(self, tmp_path):
        cli = ForgeCLI(working_dir=tmp_path)
        result = cli.test()
        assert not result.success

    def test_cli_result(self):
        r = CLIResult(success=True, message="ok")
        assert r.success
        assert r.message == "ok"
        assert r.data == {}
