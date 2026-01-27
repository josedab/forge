"""Tests for the marketplace module — hub, pack manager, and compatibility."""

from __future__ import annotations

import json

import pytest

from forge.marketplace.compatibility import (
    check_compatibility,
    check_version_constraint,
    export_manifest,
)
from forge.marketplace.hub import FeatureHub, PackMetadata
from forge.marketplace.pack_manager import (
    PackDefinition,
    PackManager,
)


class TestFeatureHub:
    """Tests for FeatureHub."""

    def test_publish_and_search(self) -> None:
        hub = FeatureHub()
        meta = PackMetadata(name="test-pack", description="A test pack", domain="finance")
        hub.publish(meta, version="1.0.0")
        results = hub.search("test")
        assert len(results) == 1
        assert results[0].name == "test-pack"

    def test_search_by_domain(self) -> None:
        hub = FeatureHub()
        hub.publish(PackMetadata(name="fin1", domain="finance"), version="1.0.0")
        hub.publish(PackMetadata(name="health1", domain="healthcare"), version="1.0.0")
        assert len(hub.search(domain="finance")) == 1
        assert len(hub.search(domain="healthcare")) == 1

    def test_search_by_tags(self) -> None:
        hub = FeatureHub()
        hub.publish(PackMetadata(name="p1", tags=["ml", "nlp"]), version="1.0.0")
        hub.publish(PackMetadata(name="p2", tags=["cv"]), version="1.0.0")
        assert len(hub.search(tags=["nlp"])) == 1

    def test_install_and_download_count(self) -> None:
        hub = FeatureHub()
        hub.publish(PackMetadata(name="p1"), version="1.0.0")
        hub.install("p1")
        assert hub.get_pack("p1").downloads == 1  # type: ignore[union-attr]

    def test_install_nonexistent_raises(self) -> None:
        hub = FeatureHub()
        with pytest.raises(ValueError, match="not found"):
            hub.install("nonexistent")

    def test_uninstall(self) -> None:
        hub = FeatureHub()
        hub.publish(PackMetadata(name="p1"), version="1.0.0")
        hub.install("p1")
        assert hub.uninstall("p1")
        assert not hub.uninstall("p1")

    def test_rate_pack(self) -> None:
        hub = FeatureHub()
        hub.publish(PackMetadata(name="p1"), version="1.0.0")
        hub.rate("p1", 4.0)
        assert hub.get_pack("p1").rating == 4.0  # type: ignore[union-attr]

    def test_rate_invalid(self) -> None:
        hub = FeatureHub()
        hub.publish(PackMetadata(name="p1"), version="1.0.0")
        with pytest.raises(ValueError, match="between 0 and 5"):
            hub.rate("p1", 6.0)

    def test_duplicate_version_raises(self) -> None:
        hub = FeatureHub()
        hub.publish(PackMetadata(name="p1"), version="1.0.0")
        with pytest.raises(ValueError, match="already exists"):
            hub.publish(PackMetadata(name="p1"), version="1.0.0")

    def test_multiple_versions(self) -> None:
        hub = FeatureHub()
        hub.publish(PackMetadata(name="p1"), version="1.0.0")
        hub.publish(PackMetadata(name="p1"), version="2.0.0")
        assert hub.get_pack("p1").latest_version == "2.0.0"  # type: ignore[union-attr]

    def test_get_stats(self) -> None:
        hub = FeatureHub()
        hub.publish(PackMetadata(name="a", domain="finance"), version="1.0.0")
        hub.publish(PackMetadata(name="b", domain="health"), version="1.0.0")
        stats = hub.get_stats()
        assert stats["total_packs"] == 2
        assert len(stats["domains"]) == 2

    def test_list_installed(self) -> None:
        hub = FeatureHub()
        hub.publish(PackMetadata(name="p1"), version="1.0.0")
        hub.install("p1")
        installed = hub.list_installed()
        assert "p1" in installed


class TestPackManager:
    """Tests for PackManager."""

    def test_create_and_validate(self) -> None:
        pm = PackManager()
        pm.create_pack("my-pack", generators=["InteractionGenerator"], description="test")
        result = pm.validate_pack("my-pack")
        assert result.is_valid

    def test_validate_missing_generators(self) -> None:
        pm = PackManager()
        pm.create_pack("empty-pack", description="test")
        result = pm.validate_pack("empty-pack")
        assert not result.is_valid
        assert any("generator or selector" in e for e in result.errors)

    def test_validate_nonexistent(self) -> None:
        pm = PackManager()
        result = pm.validate_pack("nope")
        assert not result.is_valid

    def test_publish_pack(self) -> None:
        pm = PackManager()
        pm.create_pack("pub-pack", generators=["X"], description="desc", author="me")
        meta = pm.publish_pack("pub-pack")
        assert meta.name == "pub-pack"
        assert pm.hub.get_pack("pub-pack") is not None

    def test_publish_invalid_raises(self) -> None:
        pm = PackManager()
        pm.create_pack("bad-pack")
        with pytest.raises(ValueError, match="failed validation"):
            pm.publish_pack("bad-pack")

    def test_install_pack(self) -> None:
        pm = PackManager()
        pm.create_pack("inst-pack", generators=["X"], description="d", author="a")
        pm.publish_pack("inst-pack")
        result = pm.install_pack("inst-pack")
        assert result.success

    def test_install_nonexistent(self) -> None:
        pm = PackManager()
        result = pm.install_pack("nonexistent")
        assert not result.success

    def test_circular_dependency_detection(self) -> None:
        pm = PackManager()
        pm.create_pack("a", generators=["X"], dependencies=["b"], description="d")
        pm.create_pack("b", generators=["X"], dependencies=["a"], description="d")
        result = pm.validate_pack("a")
        assert not result.is_valid
        assert any("Circular" in e for e in result.errors)

    def test_list_packs(self) -> None:
        pm = PackManager()
        pm.create_pack("p1", generators=["X"], description="d")
        pm.publish_pack("p1")
        names = pm.list_packs()
        assert "p1" in names

    def test_get_pack_info(self) -> None:
        pm = PackManager()
        pm.create_pack("info-pack", generators=["X"], description="test desc", author="tester")
        pm.publish_pack("info-pack")
        info = pm.get_pack_info("info-pack")
        assert info is not None
        assert info["author"] == "tester"


class TestPackDefinition:
    """Tests for PackDefinition serialization."""

    def test_round_trip(self) -> None:
        pack = PackDefinition(
            name="test", description="desc", generators=["A"], tags=["ml"]
        )
        d = pack.to_dict()
        pack2 = PackDefinition.from_dict(d)
        assert pack2.name == pack.name
        assert pack2.generators == pack.generators


class TestVersionConstraint:
    """Tests for version constraint checking."""

    def test_gte(self) -> None:
        assert check_version_constraint("0.1.0", ">=0.1.0")
        assert check_version_constraint("0.2.0", ">=0.1.0")
        assert not check_version_constraint("0.0.9", ">=0.1.0")

    def test_eq(self) -> None:
        assert check_version_constraint("1.0.0", "==1.0.0")
        assert not check_version_constraint("1.0.1", "==1.0.0")

    def test_lt(self) -> None:
        assert check_version_constraint("0.9.0", "<1.0.0")
        assert not check_version_constraint("1.0.0", "<1.0.0")

    def test_ne(self) -> None:
        assert check_version_constraint("0.2.0", "!=0.1.0")
        assert not check_version_constraint("0.1.0", "!=0.1.0")

    def test_unparseable_returns_true(self) -> None:
        assert check_version_constraint("1.0.0", "any")


class TestCheckCompatibility:
    """Tests for environment compatibility checking."""

    def test_compatible_environment(self) -> None:
        report = check_compatibility("test-pack")
        assert report.forge_version_ok
        assert report.python_version_ok
        assert isinstance(report.is_compatible, bool)

    def test_missing_package(self) -> None:
        report = check_compatibility(
            "test-pack", dependencies=["nonexistent_pkg_xyz"]
        )
        assert not report.is_compatible
        assert "nonexistent_pkg_xyz" in report.missing_packages

    def test_available_package(self) -> None:
        report = check_compatibility("test-pack", dependencies=["numpy"])
        assert "numpy" not in report.missing_packages


class TestExportManifest:
    """Tests for manifest export."""

    def test_json_export(self) -> None:
        result = export_manifest("my-pack", format="json")
        parsed = json.loads(result)
        assert parsed["name"] == "my-pack"

    def test_yaml_export(self) -> None:
        result = export_manifest(
            "my-pack", tags=["ml"], generators=["PolyGen"], format="yaml"
        )
        assert "name: my-pack" in result
        assert "- ml" in result
        assert "- PolyGen" in result

    def test_all_fields_present(self) -> None:
        result = export_manifest(
            "test", description="desc", author="me", version="2.0.0",
            domain="finance", tags=["fin"], generators=["A"], selectors=["B"],
            dependencies=["numpy"], forge_version=">=0.2.0", format="json",
        )
        parsed = json.loads(result)
        assert parsed["version"] == "2.0.0"
        assert parsed["domain"] == "finance"
        assert parsed["forge_version"] == ">=0.2.0"
