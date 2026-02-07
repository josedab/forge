"""Tests for studio pipeline persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.studio.persistence import PipelineStore, PipelineVersion


@pytest.fixture
def store(tmp_path: Path) -> PipelineStore:
    return PipelineStore(tmp_path / "pipelines")


@pytest.fixture
def sample_steps() -> list[dict]:
    return [
        {"name": "scale", "type": "StandardScaler", "params": {}},
        {"name": "interact", "type": "InteractionGenerator", "params": {"columns": ["a", "b"]}},
    ]


class TestPipelineVersion:
    def test_content_hash_deterministic(self, sample_steps: list[dict]) -> None:
        pv1 = PipelineVersion(name="test", version="0.1.0", steps=sample_steps)
        pv2 = PipelineVersion(name="test", version="0.2.0", steps=sample_steps)
        assert pv1.content_hash == pv2.content_hash

    def test_round_trip(self, sample_steps: list[dict]) -> None:
        pv = PipelineVersion(name="test", version="0.1.0", steps=sample_steps)
        d = pv.to_dict()
        restored = PipelineVersion.from_dict(d)
        assert restored.name == pv.name
        assert restored.steps == pv.steps


class TestPipelineStore:
    def test_save_and_load(self, store: PipelineStore, sample_steps: list[dict]) -> None:
        store.save(sample_steps, name="my_pipe")
        loaded = store.load("my_pipe")
        assert loaded is not None
        assert loaded.name == "my_pipe"
        assert loaded.steps == sample_steps

    def test_auto_versioning(self, store: PipelineStore, sample_steps: list[dict]) -> None:
        store.save(sample_steps, name="p")
        store.save(sample_steps, name="p", description="v2")
        versions = store.list_versions("p")
        assert len(versions) == 2
        assert versions[0].version != versions[1].version

    def test_load_specific_version(self, store: PipelineStore, sample_steps: list[dict]) -> None:
        pv1 = store.save(sample_steps, name="p")
        store.save(sample_steps, name="p")
        loaded = store.load("p", version=pv1.version)
        assert loaded is not None
        assert loaded.version == pv1.version

    def test_load_nonexistent(self, store: PipelineStore) -> None:
        assert store.load("nope") is None

    def test_list_pipelines(self, store: PipelineStore, sample_steps: list[dict]) -> None:
        store.save(sample_steps, name="alpha")
        store.save(sample_steps, name="beta")
        assert store.list_pipelines() == ["alpha", "beta"]

    def test_delete(self, store: PipelineStore, sample_steps: list[dict]) -> None:
        store.save(sample_steps, name="del_me")
        assert store.delete("del_me")
        assert store.load("del_me") is None

    def test_persistence_reload(self, tmp_path: Path, sample_steps: list[dict]) -> None:
        dir_ = tmp_path / "pipes"
        store1 = PipelineStore(dir_)
        store1.save(sample_steps, name="persist_test")
        # Create new store pointing at same directory
        store2 = PipelineStore(dir_)
        loaded = store2.load("persist_test")
        assert loaded is not None
        assert loaded.steps == sample_steps

    def test_in_memory_only(self, sample_steps: list[dict]) -> None:
        store = PipelineStore(None)
        store.save(sample_steps, name="mem")
        assert store.load("mem") is not None
        assert store.size == 1
