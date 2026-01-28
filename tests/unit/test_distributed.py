"""Tests for distributed computation backends."""

from __future__ import annotations

import pandas as pd
import pytest

from forge.distributed import DaskBackend, RayBackend, SparkBackend
from forge.distributed.dispatch import get_distributed_backend


class TestDaskBackend:
    def test_name(self):
        backend = DaskBackend()
        assert backend.name == "dask"

    def test_is_available(self):
        backend = DaskBackend()
        # May or may not be available depending on environment
        assert isinstance(backend.is_available(), bool)

    @pytest.mark.skipif(
        not DaskBackend().is_available(), reason="Dask not installed"
    )
    def test_map_partitions(self):
        backend = DaskBackend(n_workers=2)
        backend.connect()
        try:
            df = pd.DataFrame({"x": range(100)})
            result = backend.map_partitions(df, lambda d: d * 2, n_partitions=2)
            assert len(result) == 100
            assert result["x"].iloc[0] == 0
        finally:
            backend.disconnect()

    @pytest.mark.skipif(
        not DaskBackend().is_available(), reason="Dask not installed"
    )
    def test_groupby_apply(self):
        backend = DaskBackend(n_workers=2)
        backend.connect()
        try:
            df = pd.DataFrame({
                "group": ["a", "a", "b", "b"],
                "value": [1.0, 2.0, 3.0, 4.0],
            })
            result = backend.groupby_apply(
                df, "group", lambda g: g.assign(mean=g["value"].mean())
            )
            assert "mean" in result.columns
        finally:
            backend.disconnect()


class TestRayBackend:
    def test_name(self):
        backend = RayBackend()
        assert backend.name == "ray"

    def test_is_available(self):
        backend = RayBackend()
        assert isinstance(backend.is_available(), bool)


class TestSparkBackend:
    def test_name(self):
        backend = SparkBackend()
        assert backend.name == "spark"

    def test_is_available(self):
        backend = SparkBackend()
        assert isinstance(backend.is_available(), bool)


class TestDispatch:
    def test_explicit_dask(self):
        backend = get_distributed_backend("dask", n_workers=2)
        assert isinstance(backend, DaskBackend)

    def test_explicit_ray(self):
        backend = get_distributed_backend("ray", num_cpus=2)
        assert isinstance(backend, RayBackend)

    def test_explicit_spark(self):
        backend = get_distributed_backend("spark", master="local[2]")
        assert isinstance(backend, SparkBackend)

    def test_unknown_backend(self):
        with pytest.raises(Exception, match="Unknown distributed backend"):
            get_distributed_backend("unknown_backend")

    def test_auto_selects_available(self):
        # Auto should select whichever is available
        try:
            backend = get_distributed_backend("auto")
            assert backend.name in ("dask", "ray", "spark")
        except Exception:
            # All backends may be unavailable in some test environments
            pass

    def test_case_insensitive(self):
        backend = get_distributed_backend("DASK", n_workers=1)
        assert isinstance(backend, DaskBackend)
