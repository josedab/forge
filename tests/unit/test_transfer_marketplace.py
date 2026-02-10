"""Tests for transfer marketplace."""

from __future__ import annotations

from forge.transfer.marketplace import (
    MarketplaceSearch,
    PackBundle,
    PackRating,
    SearchResult,
)


class TestPackRating:
    def test_clamp_score(self) -> None:
        r = PackRating(pack_name="test", score=10)
        assert r.score == 5

    def test_clamp_low(self) -> None:
        r = PackRating(pack_name="test", score=-1)
        assert r.score == 1


class TestMarketplaceSearch:
    def test_empty_search(self) -> None:
        ms = MarketplaceSearch()
        results = ms.search("anything")
        assert results == []

    def test_add_and_get_rating(self) -> None:
        ms = MarketplaceSearch()
        ms.add_rating(PackRating(pack_name="p1", score=4))
        ms.add_rating(PackRating(pack_name="p1", score=5))
        assert ms.get_average_rating("p1") == 4.5

    def test_unrated_pack(self) -> None:
        ms = MarketplaceSearch()
        assert ms.get_average_rating("nonexistent") == 0.0

    def test_get_ratings(self) -> None:
        ms = MarketplaceSearch()
        ms.add_rating(PackRating(pack_name="p1", score=3, comment="ok"))
        ratings = ms.get_ratings("p1")
        assert len(ratings) == 1
        assert ratings[0].comment == "ok"

    def test_bundles(self) -> None:
        ms = MarketplaceSearch()
        ms.add_bundle(PackBundle(name="finance_basic", domain="finance"))
        ms.add_bundle(PackBundle(name="health_basic", domain="health"))
        assert len(ms.get_bundles()) == 2
        assert len(ms.get_bundles(domain="finance")) == 1

    def test_get_bundle_by_name(self) -> None:
        ms = MarketplaceSearch()
        ms.add_bundle(PackBundle(name="my_bundle", description="test"))
        b = ms.get_bundle("my_bundle")
        assert b is not None
        assert b.description == "test"

    def test_get_bundle_missing(self) -> None:
        ms = MarketplaceSearch()
        assert ms.get_bundle("missing") is None


class TestSearchResult:
    def test_defaults(self) -> None:
        sr = SearchResult(name="test", source="hub", relevance=0.8)
        assert sr.metadata == {}


class TestPackBundle:
    def test_defaults(self) -> None:
        b = PackBundle(name="b1")
        assert b.pack_names == []
        assert b.domain == ""
