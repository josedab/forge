"""Tests for governance manager and compliance reporting."""

from __future__ import annotations

import pytest

from forge.documentation.governance import (
    FeatureGovernance,
    GovernanceManager,
    SLAConfig,
)


class TestGovernanceManager:
    def test_add_feature(self):
        gov = GovernanceManager()
        result = gov.add_feature("user_age", owner="data-team", compliance_tags=["PII"])
        assert result.name == "user_age"
        assert result.owner == "data-team"
        assert "PII" in result.compliance_tags

    def test_get_feature(self):
        gov = GovernanceManager()
        gov.add_feature("f1", owner="team-a")
        f = gov.get_feature("f1")
        assert f is not None
        assert f.owner == "team-a"

    def test_get_missing_feature(self):
        gov = GovernanceManager()
        assert gov.get_feature("missing") is None

    def test_set_sla(self):
        gov = GovernanceManager()
        gov.add_feature("f1")
        sla = gov.set_sla("f1", freshness_hours=12, quality_threshold=0.99)
        assert sla.freshness_hours == 12
        assert gov.get_feature("f1").sla is not None

    def test_set_sla_missing_feature(self):
        gov = GovernanceManager()
        with pytest.raises(KeyError):
            gov.set_sla("missing")

    def test_deprecate(self):
        gov = GovernanceManager()
        gov.add_feature("old_feature")
        gov.deprecate("old_feature", reason="Replaced by v2", sunset_date="2025-06-01")

        f = gov.get_feature("old_feature")
        assert f.status == "deprecated"
        assert f.deprecation_reason == "Replaced by v2"

    def test_deprecate_missing(self):
        gov = GovernanceManager()
        with pytest.raises(KeyError):
            gov.deprecate("missing")

    def test_record_usage(self):
        gov = GovernanceManager()
        gov.add_feature("f1")
        gov.record_usage("f1", model_name="model_v1")
        gov.record_usage("f1", model_name="model_v2")
        gov.record_usage("f1", model_name="model_v1")  # duplicate model

        f = gov.get_feature("f1")
        assert f.usage_count == 3
        assert len(f.models_using) == 2

    def test_record_usage_missing(self):
        gov = GovernanceManager()
        with pytest.raises(KeyError):
            gov.record_usage("missing")

    def test_check_access(self):
        gov = GovernanceManager()
        gov.add_feature("f1", access_roles=["viewer", "editor"])
        assert gov.check_access("f1", "viewer")
        assert gov.check_access("f1", "editor")
        assert not gov.check_access("f1", "deployer")
        assert gov.check_access("f1", "admin")  # admin always has access

    def test_check_access_missing(self):
        gov = GovernanceManager()
        assert not gov.check_access("missing", "viewer")

    def test_search_by_owner(self):
        gov = GovernanceManager()
        gov.add_feature("f1", owner="team-a")
        gov.add_feature("f2", owner="team-b")
        gov.add_feature("f3", owner="team-a")

        results = gov.search(owner="team-a")
        assert len(results) == 2

    def test_search_by_compliance_tag(self):
        gov = GovernanceManager()
        gov.add_feature("f1", compliance_tags=["PII"])
        gov.add_feature("f2", compliance_tags=["PUBLIC"])

        results = gov.search(compliance_tag="PII")
        assert len(results) == 1
        assert results[0].name == "f1"

    def test_search_by_status(self):
        gov = GovernanceManager()
        gov.add_feature("f1")
        gov.add_feature("f2")
        gov.deprecate("f2", reason="old")

        results = gov.search(status="deprecated")
        assert len(results) == 1

    def test_compliance_report(self):
        gov = GovernanceManager()
        gov.add_feature("f1", owner="team-a", compliance_tags=["PII"])
        gov.set_sla("f1", freshness_hours=24)
        gov.add_feature("f2")  # no owner, no SLA
        gov.add_feature("f3", owner="team-b")

        report = gov.generate_compliance_report()
        assert report.total_features == 3
        assert report.active_count == 3
        assert "f1" in report.pii_features
        assert "f2" in report.unowned_features
        assert report.compliance_score < 1.0

    def test_compliance_report_to_markdown(self):
        gov = GovernanceManager()
        gov.add_feature("f1", owner="team-a")
        report = gov.generate_compliance_report()
        md = report.to_markdown()
        assert "# Compliance Report" in md

    def test_compliance_report_to_dict(self):
        gov = GovernanceManager()
        gov.add_feature("f1", owner="team-a")
        report = gov.generate_compliance_report()
        d = report.to_dict()
        assert "total_features" in d
        assert "compliance_score" in d

    def test_past_sunset_features(self):
        gov = GovernanceManager()
        gov.add_feature("old")
        gov.deprecate("old", reason="old", sunset_date="2020-01-01")

        report = gov.generate_compliance_report()
        assert "old" in report.past_sunset

    def test_persistence(self, tmp_path):
        gov1 = GovernanceManager(storage_path=tmp_path / "gov")
        gov1.add_feature("f1", owner="team-a", compliance_tags=["PII"])
        gov1.set_sla("f1", freshness_hours=12)
        gov1.save()

        gov2 = GovernanceManager(storage_path=tmp_path / "gov")
        f = gov2.get_feature("f1")
        assert f is not None
        assert f.owner == "team-a"
        assert f.sla is not None
        assert f.sla.freshness_hours == 12

    def test_feature_names_and_size(self):
        gov = GovernanceManager()
        gov.add_feature("b_feature")
        gov.add_feature("a_feature")
        assert gov.size == 2
        assert gov.feature_names == ["a_feature", "b_feature"]

    def test_feature_governance_serialization(self):
        fg = FeatureGovernance(
            name="test", owner="me", compliance_tags=["PII"],
            sla=SLAConfig(freshness_hours=12),
        )
        d = fg.to_dict()
        restored = FeatureGovernance.from_dict(d)
        assert restored.name == "test"
        assert restored.sla.freshness_hours == 12
