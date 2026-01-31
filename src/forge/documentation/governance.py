"""Governance metadata and compliance extensions for feature documentation.

Adds compliance tags, SLA tracking, access control, usage tracking,
and deprecation management to the documentation module.

Example:
    >>> from forge.documentation.governance import GovernanceManager
    >>> gov = GovernanceManager()
    >>> gov.add_feature("user_age", owner="data-team", compliance_tags=["PII"])
    >>> gov.set_sla("user_age", freshness_hours=24, quality_threshold=0.95)
    >>> gov.deprecate("old_feature", reason="Replaced by v2", sunset="2025-06-01")
    >>> report = gov.generate_compliance_report()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

ComplianceTag = Literal[
    "PII", "SENSITIVE", "DERIVED", "EXTERNAL", "INTERNAL",
    "GDPR", "HIPAA", "SOX", "PUBLIC",
]

VALID_COMPLIANCE_TAGS = {
    "PII", "SENSITIVE", "DERIVED", "EXTERNAL", "INTERNAL",
    "GDPR", "HIPAA", "SOX", "PUBLIC",
}


@dataclass
class FeatureGovernance:
    """Governance metadata for a single feature."""

    name: str
    owner: str = ""
    team: str = ""
    compliance_tags: list[str] = field(default_factory=list)
    status: str = "active"  # active, deprecated, experimental, archived
    deprecation_reason: str = ""
    sunset_date: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    lineage: list[str] = field(default_factory=list)
    usage_count: int = 0
    models_using: list[str] = field(default_factory=list)
    sla: SLAConfig | None = None
    access_roles: list[str] = field(default_factory=lambda: ["viewer"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "owner": self.owner,
            "team": self.team,
            "compliance_tags": self.compliance_tags,
            "status": self.status,
            "deprecation_reason": self.deprecation_reason,
            "sunset_date": self.sunset_date,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description": self.description,
            "lineage": self.lineage,
            "usage_count": self.usage_count,
            "models_using": self.models_using,
            "sla": self.sla.to_dict() if self.sla else None,
            "access_roles": self.access_roles,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureGovernance:
        """Deserialize from dictionary."""
        sla_data = data.get("sla")
        sla = SLAConfig.from_dict(sla_data) if sla_data else None
        return cls(
            name=data["name"],
            owner=data.get("owner", ""),
            team=data.get("team", ""),
            compliance_tags=data.get("compliance_tags", []),
            status=data.get("status", "active"),
            deprecation_reason=data.get("deprecation_reason", ""),
            sunset_date=data.get("sunset_date", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            description=data.get("description", ""),
            lineage=data.get("lineage", []),
            usage_count=data.get("usage_count", 0),
            models_using=data.get("models_using", []),
            sla=sla,
            access_roles=data.get("access_roles", ["viewer"]),
        )


@dataclass
class SLAConfig:
    """SLA configuration for a feature."""

    freshness_hours: float = 24.0
    quality_threshold: float = 0.95
    max_null_rate: float = 0.05
    max_drift_psi: float = 0.2
    min_completeness: float = 0.90

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "freshness_hours": self.freshness_hours,
            "quality_threshold": self.quality_threshold,
            "max_null_rate": self.max_null_rate,
            "max_drift_psi": self.max_drift_psi,
            "min_completeness": self.min_completeness,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SLAConfig:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ComplianceReport:
    """A compliance report summarizing governance status."""

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_features: int = 0
    active_count: int = 0
    deprecated_count: int = 0
    pii_features: list[str] = field(default_factory=list)
    unowned_features: list[str] = field(default_factory=list)
    sla_violations: list[str] = field(default_factory=list)
    features_without_sla: list[str] = field(default_factory=list)
    past_sunset: list[str] = field(default_factory=list)
    compliance_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "generated_at": self.generated_at,
            "total_features": self.total_features,
            "active_count": self.active_count,
            "deprecated_count": self.deprecated_count,
            "pii_features": self.pii_features,
            "unowned_features": self.unowned_features,
            "sla_violations": self.sla_violations,
            "features_without_sla": self.features_without_sla,
            "past_sunset": self.past_sunset,
            "compliance_score": round(self.compliance_score, 4),
        }

    def to_markdown(self) -> str:
        """Export as markdown report."""
        lines = [
            "# Compliance Report",
            f"**Generated**: {self.generated_at}",
            "",
            "## Summary",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Features | {self.total_features} |",
            f"| Active | {self.active_count} |",
            f"| Deprecated | {self.deprecated_count} |",
            f"| Compliance Score | {self.compliance_score:.1%} |",
            "",
        ]

        if self.pii_features:
            lines.append("## PII Features")
            for f in self.pii_features:
                lines.append(f"- {f}")
            lines.append("")

        if self.unowned_features:
            lines.append("## Unowned Features (Action Required)")
            for f in self.unowned_features:
                lines.append(f"- {f}")
            lines.append("")

        if self.sla_violations:
            lines.append("## SLA Violations")
            for v in self.sla_violations:
                lines.append(f"- {v}")
            lines.append("")

        if self.past_sunset:
            lines.append("## Past Sunset Date")
            for f in self.past_sunset:
                lines.append(f"- {f}")

        return "\n".join(lines)


class GovernanceManager:
    """Manages governance metadata for features.

    Tracks ownership, compliance tags, SLAs, deprecation,
    usage, and access control for a collection of features.

    Parameters:
        storage_path: Path for persistent storage. None for in-memory.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._features: dict[str, FeatureGovernance] = {}
        if self.storage_path:
            self._load()

    def add_feature(
        self,
        name: str,
        owner: str = "",
        team: str = "",
        compliance_tags: list[str] | None = None,
        description: str = "",
        lineage: list[str] | None = None,
        access_roles: list[str] | None = None,
    ) -> FeatureGovernance:
        """Register a feature with governance metadata.

        Args:
            name: Feature name.
            owner: Feature owner.
            team: Owning team.
            compliance_tags: Compliance/sensitivity tags.
            description: Feature description.
            lineage: Source features/columns.
            access_roles: Roles with access.

        Returns:
            Created FeatureGovernance record.
        """
        gov = FeatureGovernance(
            name=name,
            owner=owner,
            team=team,
            compliance_tags=compliance_tags or [],
            description=description,
            lineage=lineage or [],
            access_roles=access_roles or ["viewer"],
        )
        self._features[name] = gov
        return gov

    def get_feature(self, name: str) -> FeatureGovernance | None:
        """Get governance metadata for a feature."""
        return self._features.get(name)

    def set_sla(
        self,
        feature_name: str,
        freshness_hours: float = 24.0,
        quality_threshold: float = 0.95,
        max_null_rate: float = 0.05,
        max_drift_psi: float = 0.2,
    ) -> SLAConfig:
        """Set SLA for a feature.

        Args:
            feature_name: Feature to configure.
            freshness_hours: Max hours between updates.
            quality_threshold: Min quality score.
            max_null_rate: Max acceptable null rate.
            max_drift_psi: Max PSI for drift.

        Returns:
            The SLA configuration.

        Raises:
            KeyError: If feature not found.
        """
        if feature_name not in self._features:
            raise KeyError(f"Feature '{feature_name}' not found")

        sla = SLAConfig(
            freshness_hours=freshness_hours,
            quality_threshold=quality_threshold,
            max_null_rate=max_null_rate,
            max_drift_psi=max_drift_psi,
        )
        self._features[feature_name].sla = sla
        self._features[feature_name].updated_at = datetime.now().isoformat()
        return sla

    def deprecate(
        self,
        feature_name: str,
        reason: str = "",
        sunset_date: str = "",
    ) -> None:
        """Mark a feature as deprecated.

        Args:
            feature_name: Feature to deprecate.
            reason: Why it's deprecated.
            sunset_date: When it will be removed (ISO format).

        Raises:
            KeyError: If feature not found.
        """
        if feature_name not in self._features:
            raise KeyError(f"Feature '{feature_name}' not found")

        self._features[feature_name].status = "deprecated"
        self._features[feature_name].deprecation_reason = reason
        self._features[feature_name].sunset_date = sunset_date
        self._features[feature_name].updated_at = datetime.now().isoformat()

    def record_usage(
        self,
        feature_name: str,
        model_name: str | None = None,
    ) -> None:
        """Record feature usage.

        Args:
            feature_name: Feature being used.
            model_name: Optional model using the feature.

        Raises:
            KeyError: If feature not found.
        """
        if feature_name not in self._features:
            raise KeyError(f"Feature '{feature_name}' not found")

        self._features[feature_name].usage_count += 1
        if model_name and model_name not in self._features[feature_name].models_using:
            self._features[feature_name].models_using.append(model_name)

    def check_access(self, feature_name: str, role: str) -> bool:
        """Check if a role has access to a feature.

        Args:
            feature_name: Feature to check.
            role: Role to verify.

        Returns:
            True if the role has access.
        """
        gov = self._features.get(feature_name)
        if gov is None:
            return False
        return role in gov.access_roles or role == "admin"

    def search(
        self,
        owner: str | None = None,
        team: str | None = None,
        status: str | None = None,
        compliance_tag: str | None = None,
    ) -> list[FeatureGovernance]:
        """Search features by governance attributes.

        Args:
            owner: Filter by owner.
            team: Filter by team.
            status: Filter by status.
            compliance_tag: Filter by compliance tag.

        Returns:
            List of matching FeatureGovernance records.
        """
        results: list[FeatureGovernance] = []
        for gov in self._features.values():
            if owner and gov.owner != owner:
                continue
            if team and gov.team != team:
                continue
            if status and gov.status != status:
                continue
            if compliance_tag and compliance_tag not in gov.compliance_tags:
                continue
            results.append(gov)
        return results

    def generate_compliance_report(self) -> ComplianceReport:
        """Generate a compliance report.

        Returns:
            ComplianceReport summarizing governance status.
        """
        now = datetime.now()
        pii_features: list[str] = []
        unowned: list[str] = []
        no_sla: list[str] = []
        past_sunset: list[str] = []
        active = 0
        deprecated = 0

        for name, gov in self._features.items():
            if gov.status == "active":
                active += 1
            elif gov.status == "deprecated":
                deprecated += 1

            if "PII" in gov.compliance_tags or "SENSITIVE" in gov.compliance_tags:
                pii_features.append(name)

            if not gov.owner:
                unowned.append(name)

            if gov.sla is None and gov.status == "active":
                no_sla.append(name)

            if gov.sunset_date:
                try:
                    sunset = datetime.fromisoformat(gov.sunset_date)
                    if sunset < now:
                        past_sunset.append(name)
                except ValueError:
                    pass

        total = len(self._features)
        # Compliance score: penalize for unowned, no-SLA, past-sunset
        deductions = 0.0
        if total > 0:
            deductions += len(unowned) / total * 0.3
            deductions += len(no_sla) / total * 0.2
            deductions += len(past_sunset) / total * 0.5
        score = max(0.0, 1.0 - deductions)

        return ComplianceReport(
            total_features=total,
            active_count=active,
            deprecated_count=deprecated,
            pii_features=pii_features,
            unowned_features=unowned,
            features_without_sla=no_sla,
            past_sunset=past_sunset,
            compliance_score=score,
        )

    @property
    def feature_names(self) -> list[str]:
        """All tracked feature names."""
        return sorted(self._features.keys())

    @property
    def size(self) -> int:
        """Number of tracked features."""
        return len(self._features)

    def save(self) -> None:
        """Persist governance data to disk."""
        if self.storage_path is None:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        data_path = self.storage_path / "governance.json"
        data = {k: v.to_dict() for k, v in self._features.items()}
        data_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load governance data from disk."""
        if self.storage_path is None:
            return
        data_path = self.storage_path / "governance.json"
        if not data_path.exists():
            return
        try:
            raw = json.loads(data_path.read_text())
            for name, data in raw.items():
                self._features[name] = FeatureGovernance.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load governance data: %s", e)
