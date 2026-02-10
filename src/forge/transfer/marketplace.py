"""Extended marketplace features for the transfer hub.

Adds advanced search, ratings, pack bundles, and install APIs
on top of the existing TransferHub and FeatureHub.

Example:
    >>> from forge.transfer.marketplace import MarketplaceSearch, PackRating
    >>> search = MarketplaceSearch(hub)
    >>> results = search.search("churn prediction", domain="saas")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from forge.marketplace.hub import FeatureHub, PackMetadata  # noqa: TC001
from forge.transfer.hub import PipelineRecord, TransferHub  # noqa: TC001

logger = logging.getLogger(__name__)


@dataclass
class PackRating:
    """User rating for a feature pack.

    Attributes:
        pack_name: Name of the rated pack.
        user_id: Identifier of the rater.
        score: Rating score 1-5.
        comment: Optional comment.
    """

    pack_name: str
    user_id: str = "anonymous"
    score: int = 5
    comment: str = ""

    def __post_init__(self) -> None:
        self.score = max(1, min(5, self.score))


@dataclass
class SearchResult:
    """Search result from marketplace.

    Attributes:
        name: Pack or pipeline name.
        source: Where the result comes from ('hub' or 'transfer').
        relevance: Relevance score 0-1.
        metadata: Additional metadata dict.
    """

    name: str
    source: str
    relevance: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PackBundle:
    """A bundle of related feature packs.

    Attributes:
        name: Bundle name.
        description: What the bundle provides.
        pack_names: List of pack names in the bundle.
        domain: Target domain.
    """

    name: str
    description: str = ""
    pack_names: list[str] = field(default_factory=list)
    domain: str = ""


class MarketplaceSearch:
    """Unified search across FeatureHub and TransferHub.

    Parameters
    ----------
    feature_hub : FeatureHub or None
        Marketplace hub for feature packs.
    transfer_hub : TransferHub or None
        Transfer hub for pipeline records.
    """

    def __init__(
        self,
        feature_hub: FeatureHub | None = None,
        transfer_hub: TransferHub | None = None,
    ) -> None:
        self.feature_hub = feature_hub
        self.transfer_hub = transfer_hub
        self._ratings: dict[str, list[PackRating]] = {}
        self._bundles: list[PackBundle] = []

    def search(
        self,
        query: str,
        *,
        domain: str = "",
        tags: list[str] | None = None,
        min_rating: float = 0.0,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Search across both hubs for packs/pipelines.

        Args:
            query: Search query string.
            domain: Filter by domain.
            tags: Filter by tags.
            min_rating: Minimum average rating.
            limit: Maximum results.

        Returns:
            List of SearchResult ordered by relevance.
        """
        results: list[SearchResult] = []
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))

        # Search feature hub
        if self.feature_hub is not None:
            for meta in self._list_hub_packs():
                rel = self._compute_relevance(meta, query_words, domain, tags)
                if rel > 0:
                    avg = self.get_average_rating(meta.name)
                    if avg >= min_rating:
                        results.append(SearchResult(
                            name=meta.name,
                            source="hub",
                            relevance=rel,
                            metadata={
                                "description": meta.description,
                                "domain": meta.domain,
                                "tags": meta.tags,
                                "avg_rating": avg,
                            },
                        ))

        # Search transfer hub
        if self.transfer_hub is not None:
            for record in self._list_transfer_pipelines():
                rel = self._compute_pipeline_relevance(
                    record, query_words,
                )
                if rel > 0:
                    results.append(SearchResult(
                        name=record.name,
                        source="transfer",
                        relevance=rel,
                        metadata={
                            "pipeline_id": record.pipeline_id,
                            "transforms": record.transform_names,
                        },
                    ))

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:limit]

    def add_rating(self, rating: PackRating) -> None:
        """Add a rating for a pack.

        Args:
            rating: PackRating to add.
        """
        self._ratings.setdefault(rating.pack_name, []).append(rating)

    def get_average_rating(self, pack_name: str) -> float:
        """Get the average rating for a pack.

        Args:
            pack_name: Name of the pack.

        Returns:
            Average rating or 0.0 if unrated.
        """
        ratings = self._ratings.get(pack_name, [])
        if not ratings:
            return 0.0
        return sum(r.score for r in ratings) / len(ratings)

    def get_ratings(self, pack_name: str) -> list[PackRating]:
        """Get all ratings for a pack.

        Args:
            pack_name: Name of the pack.

        Returns:
            List of ratings.
        """
        return list(self._ratings.get(pack_name, []))

    def add_bundle(self, bundle: PackBundle) -> None:
        """Register a pack bundle.

        Args:
            bundle: PackBundle to register.
        """
        self._bundles.append(bundle)

    def get_bundles(self, domain: str = "") -> list[PackBundle]:
        """List bundles, optionally filtered by domain.

        Args:
            domain: Filter by domain.

        Returns:
            Matching bundles.
        """
        if not domain:
            return list(self._bundles)
        return [b for b in self._bundles if b.domain == domain]

    def get_bundle(self, name: str) -> PackBundle | None:
        """Get a bundle by name.

        Args:
            name: Bundle name.

        Returns:
            PackBundle or None if not found.
        """
        for b in self._bundles:
            if b.name == name:
                return b
        return None

    def _list_hub_packs(self) -> list[PackMetadata]:
        if self.feature_hub is None:
            return []
        try:
            return list(self.feature_hub.registry.values())
        except Exception:
            return []

    def _list_transfer_pipelines(self) -> list[PipelineRecord]:
        if self.transfer_hub is None:
            return []
        try:
            return list(self.transfer_hub._pipelines.values())
        except Exception:
            return []

    @staticmethod
    def _compute_relevance(
        meta: PackMetadata,
        query_words: set[str],
        domain: str,
        tags: list[str] | None,
    ) -> float:
        if not query_words:
            return 0.5

        text_words = set(
            re.findall(
                r"\w+",
                f"{meta.name} {meta.description} {' '.join(meta.tags)}".lower(),
            )
        )
        overlap = len(query_words & text_words)
        relevance = overlap / max(len(query_words), 1)

        if domain and meta.domain.lower() != domain.lower():
            relevance *= 0.5

        if tags:
            tag_set = {t.lower() for t in meta.tags}
            tag_match = sum(1 for t in tags if t.lower() in tag_set)
            relevance *= (0.5 + 0.5 * tag_match / len(tags))

        return min(relevance, 1.0)

    @staticmethod
    def _compute_pipeline_relevance(
        record: PipelineRecord,
        query_words: set[str],
    ) -> float:
        if not query_words:
            return 0.3
        text = f"{record.name} {' '.join(record.transform_names)}".lower()
        text_words = set(re.findall(r"\w+", text))
        overlap = len(query_words & text_words)
        return min(overlap / max(len(query_words), 1), 1.0)
