"""Tests for Feature Marketplace and Hub."""

from __future__ import annotations

import pytest

from forge.marketplace import (
    FeatureHub,
    HubConfig,
    PackMetadata,
    PackVersion,
)


@pytest.fixture
def hub():
    return FeatureHub(config=HubConfig(storage_path="/tmp/forge_test_hub"))


@pytest.fixture
def sample_pack():
    return PackMetadata(
        name="finance-basics",
        description="Basic financial feature engineering pack",
        author="forge-team",
        tags=["finance", "ratios", "returns"],
        domain="finance",
        dependencies=["numpy", "pandas"],
    )


@pytest.fixture
def healthcare_pack():
    return PackMetadata(
        name="healthcare-vitals",
        description="Vital signs feature extraction",
        author="medml",
        tags=["healthcare", "vitals", "clinical"],
        domain="healthcare",
    )


class TestFeatureHub:
    """Tests for FeatureHub."""

    def test_publish_basic(self, hub, sample_pack):
        result = hub.publish(
            sample_pack,
            pack_content={"code": "print('hello')"},
            version="1.0.0",
        )
        assert result.name == "finance-basics"
        assert len(result.versions) == 1
        assert result.latest_version == "1.0.0"

    def test_publish_multiple_versions(self, hub, sample_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.publish(sample_pack, version="1.1.0", changelog="Added new features")
        pack = hub.get_pack("finance-basics")
        assert len(pack.versions) == 2
        assert pack.latest_version == "1.1.0"

    def test_publish_duplicate_version_raises(self, hub, sample_pack):
        hub.publish(sample_pack, version="1.0.0")
        with pytest.raises(ValueError, match="already exists"):
            hub.publish(sample_pack, version="1.0.0")

    def test_search_by_query(self, hub, sample_pack, healthcare_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.publish(healthcare_pack, version="1.0.0")
        results = hub.search("finance")
        assert len(results) == 1
        assert results[0].name == "finance-basics"

    def test_search_by_domain(self, hub, sample_pack, healthcare_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.publish(healthcare_pack, version="1.0.0")
        results = hub.search(domain="healthcare")
        assert len(results) == 1
        assert results[0].name == "healthcare-vitals"

    def test_search_by_tags(self, hub, sample_pack, healthcare_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.publish(healthcare_pack, version="1.0.0")
        results = hub.search(tags=["ratios"])
        assert len(results) == 1
        assert results[0].name == "finance-basics"

    def test_search_empty_query_returns_all(self, hub, sample_pack, healthcare_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.publish(healthcare_pack, version="1.0.0")
        results = hub.search()
        assert len(results) == 2

    def test_install(self, hub, sample_pack):
        content = {"code": "def generate(): pass"}
        hub.publish(sample_pack, pack_content=content, version="1.0.0")
        installed = hub.install("finance-basics")
        assert installed == content

    def test_install_specific_version(self, hub, sample_pack):
        hub.publish(sample_pack, pack_content={"v": "1"}, version="1.0.0")
        hub.publish(sample_pack, pack_content={"v": "2"}, version="2.0.0")
        installed = hub.install("finance-basics", version="1.0.0")
        assert installed == {"v": "1"}

    def test_install_nonexistent_raises(self, hub):
        with pytest.raises(ValueError, match="not found"):
            hub.install("nonexistent-pack")

    def test_install_increments_downloads(self, hub, sample_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.install("finance-basics")
        hub.install("finance-basics")
        pack = hub.get_pack("finance-basics")
        assert pack.downloads == 2

    def test_uninstall(self, hub, sample_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.install("finance-basics")
        assert hub.uninstall("finance-basics") is True
        assert "finance-basics" not in hub.list_installed()

    def test_uninstall_not_installed(self, hub):
        assert hub.uninstall("not-installed") is False

    def test_list_installed(self, hub, sample_pack, healthcare_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.publish(healthcare_pack, version="1.0.0")
        hub.install("finance-basics")
        installed = hub.list_installed()
        assert "finance-basics" in installed
        assert installed["finance-basics"] == "1.0.0"

    def test_list_all(self, hub, sample_pack, healthcare_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.publish(healthcare_pack, version="1.0.0")
        all_packs = hub.list_all()
        assert len(all_packs) == 2

    def test_rate_pack(self, hub, sample_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.rate("finance-basics", 4.5)
        pack = hub.get_pack("finance-basics")
        assert pack.rating == 4.5

    def test_rate_invalid_raises(self, hub, sample_pack):
        hub.publish(sample_pack, version="1.0.0")
        with pytest.raises(ValueError, match="between 0 and 5"):
            hub.rate("finance-basics", 6.0)

    def test_rate_nonexistent_raises(self, hub):
        with pytest.raises(ValueError, match="not found"):
            hub.rate("nonexistent", 4.0)

    def test_get_stats(self, hub, sample_pack, healthcare_pack):
        hub.publish(sample_pack, version="1.0.0")
        hub.publish(healthcare_pack, version="1.0.0")
        hub.install("finance-basics")
        stats = hub.get_stats()
        assert stats["total_packs"] == 2
        assert stats["installed"] == 1
        assert "finance" in stats["domains"]


class TestPackMetadata:
    """Tests for PackMetadata."""

    def test_to_dict(self, sample_pack):
        d = sample_pack.to_dict()
        assert d["name"] == "finance-basics"
        assert d["domain"] == "finance"

    def test_from_dict(self):
        data = {
            "name": "test-pack",
            "description": "A test",
            "author": "tester",
            "domain": "general",
            "tags": ["test"],
            "versions": [{"version": "1.0.0", "published_at": 0, "checksum": "", "changelog": ""}],
        }
        pack = PackMetadata.from_dict(data)
        assert pack.name == "test-pack"
        assert len(pack.versions) == 1

    def test_latest_version_empty(self):
        pack = PackMetadata(name="empty")
        assert pack.latest_version == "0.0.0"


class TestPackVersion:
    """Tests for PackVersion."""

    def test_auto_timestamp(self):
        v = PackVersion(version="1.0.0")
        assert v.published_at > 0

    def test_with_changelog(self):
        v = PackVersion(version="2.0.0", changelog="Breaking changes")
        assert v.changelog == "Breaking changes"
