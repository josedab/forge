"""Tests for the pack manager and marketplace extensions."""

from __future__ import annotations

import pytest

from forge.marketplace.pack_manager import (
    PackDefinition,
    PackManager,
    ValidationResult,
)


class TestPackManager:
    def test_create_pack(self):
        pm = PackManager()
        pack = pm.create_pack(
            "finance-features",
            author="team-a",
            domain="finance",
            tags=["finance", "ratios"],
            generators=["InteractionGenerator"],
            description="Finance feature engineering pack",
        )
        assert pack.name == "finance-features"
        assert pack.author == "team-a"

    def test_validate_valid_pack(self):
        pm = PackManager()
        pm.create_pack(
            "test-pack",
            author="me",
            description="A test pack",
            generators=["NumericGenerator"],
            tags=["test"],
            domain="testing",
        )
        result = pm.validate_pack("test-pack")
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.quality_score > 0.8

    def test_validate_empty_pack(self):
        pm = PackManager()
        pm.create_pack("empty-pack")
        result = pm.validate_pack("empty-pack")
        assert not result.is_valid
        assert any("generator or selector" in e for e in result.errors)

    def test_validate_missing_pack(self):
        pm = PackManager()
        result = pm.validate_pack("nonexistent")
        assert not result.is_valid
        assert result.quality_score == 0.0

    def test_validate_missing_dependency(self):
        pm = PackManager()
        pm.create_pack(
            "dep-pack",
            generators=["Gen1"],
            dependencies=["nonexistent-dep"],
            description="Has missing dep",
        )
        result = pm.validate_pack("dep-pack")
        assert not result.is_valid
        assert any("not found" in e for e in result.errors)

    def test_validate_circular_deps(self):
        pm = PackManager()
        pm.create_pack("a", generators=["G1"], dependencies=["b"], description="pack a")
        pm.create_pack("b", generators=["G2"], dependencies=["a"], description="pack b")

        result = pm.validate_pack("a")
        assert not result.is_valid
        assert any("Circular" in e for e in result.errors)

    def test_publish_pack(self):
        pm = PackManager()
        pm.create_pack(
            "pub-pack",
            author="me",
            description="Publishable pack",
            generators=["Gen1"],
            tags=["test"],
            domain="testing",
        )
        meta = pm.publish_pack("pub-pack", changelog="Initial release")
        assert meta.name == "pub-pack"
        assert len(meta.versions) == 1

    def test_publish_invalid_pack(self):
        pm = PackManager()
        pm.create_pack("bad-pack")
        with pytest.raises(ValueError, match="failed validation"):
            pm.publish_pack("bad-pack")

    def test_publish_missing_pack(self):
        pm = PackManager()
        with pytest.raises(ValueError, match="not found"):
            pm.publish_pack("nonexistent")

    def test_install_pack(self):
        pm = PackManager()
        pm.create_pack(
            "inst-pack",
            author="me",
            description="Installable",
            generators=["Gen1"],
            tags=["test"],
            domain="testing",
        )
        pm.publish_pack("inst-pack")
        result = pm.install_pack("inst-pack")
        assert result.success
        assert result.pack_name == "inst-pack"

    def test_install_missing_pack(self):
        pm = PackManager()
        result = pm.install_pack("nonexistent")
        assert not result.success

    def test_install_with_dependencies(self):
        pm = PackManager()
        pm.create_pack("dep1", description="dep", generators=["G"], tags=["d"], domain="d")
        pm.publish_pack("dep1")

        pm.create_pack(
            "main",
            description="main",
            generators=["G2"],
            dependencies=["dep1"],
            tags=["m"],
            domain="m",
        )
        pm.publish_pack("main")

        result = pm.install_pack("main")
        assert result.success
        assert "dep1" in result.installed_deps

    def test_list_packs(self):
        pm = PackManager()
        pm.create_pack("p1", description="P1", generators=["G"], tags=["t"], domain="d")
        pm.create_pack("p2", description="P2", generators=["G"], tags=["t"], domain="d")
        pm.publish_pack("p1")
        pm.publish_pack("p2")

        names = pm.list_packs()
        assert "p1" in names
        assert "p2" in names

    def test_list_installed_only(self):
        pm = PackManager()
        pm.create_pack("p1", description="P1", generators=["G"], tags=["t"], domain="d")
        pm.publish_pack("p1")
        pm.install_pack("p1")

        installed = pm.list_packs(installed_only=True)
        assert "p1" in installed

    def test_get_pack_info(self):
        pm = PackManager()
        pm.create_pack("p1", description="desc", author="me", generators=["G"], tags=["t"], domain="d")
        pm.publish_pack("p1")

        info = pm.get_pack_info("p1")
        assert info is not None
        assert info["name"] == "p1"
        assert info["author"] == "me"

    def test_get_pack_info_missing(self):
        pm = PackManager()
        assert pm.get_pack_info("nonexistent") is None

    def test_pack_definition_serialization(self):
        pack = PackDefinition(
            name="test", author="me", generators=["G1"],
            description="desc", domain="finance",
        )
        d = pack.to_dict()
        restored = PackDefinition.from_dict(d)
        assert restored.name == "test"
        assert restored.generators == ["G1"]

    def test_validation_result_serialization(self):
        result = ValidationResult(
            is_valid=True, pack_name="test", quality_score=0.95,
        )
        d = result.to_dict()
        assert d["is_valid"]
        assert d["quality_score"] == 0.95

    def test_save_and_load(self, tmp_path):
        pm1 = PackManager(storage_path=tmp_path / "packs")
        pm1.create_pack("p1", generators=["G"], description="test")
        pm1.save()

        pm2 = PackManager(storage_path=tmp_path / "packs")
        pm2._load()
        assert "p1" in pm2._pack_definitions
