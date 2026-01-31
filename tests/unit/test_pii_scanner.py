"""Tests for PII detection and compliance tagging."""

from __future__ import annotations

import pandas as pd
import pytest

from forge.documentation.pii_scanner import (
    PIIDetection,
    PIIScanner,
    PIIScanResult,
    auto_tag_compliance,
)


@pytest.fixture
def pii_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "user_email": ["alice@example.com", "bob@test.org", "carol@mail.co"],
        "first_name": ["Alice", "Bob", "Carol"],
        "phone_number": ["+1-555-0101", "+1-555-0102", "+1-555-0103"],
        "age": [25, 30, 35],
        "score": [0.85, 0.92, 0.78],
    })


@pytest.fixture
def clean_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0],
        "feature_b": [4.0, 5.0, 6.0],
        "category": ["x", "y", "z"],
    })


class TestPIIScanner:
    def test_detects_email_column(self, pii_dataframe: pd.DataFrame) -> None:
        scanner = PIIScanner()
        result = scanner.scan(pii_dataframe)
        email_detections = [d for d in result.detections if d.pii_type == "email"]
        assert len(email_detections) >= 1
        assert email_detections[0].column_name == "user_email"

    def test_detects_name_column(self, pii_dataframe: pd.DataFrame) -> None:
        scanner = PIIScanner()
        result = scanner.scan(pii_dataframe)
        name_detections = [d for d in result.detections if d.pii_type == "name"]
        assert len(name_detections) >= 1

    def test_detects_phone_column(self, pii_dataframe: pd.DataFrame) -> None:
        scanner = PIIScanner()
        result = scanner.scan(pii_dataframe)
        phone_detections = [d for d in result.detections if d.pii_type == "phone"]
        assert len(phone_detections) >= 1

    def test_no_false_positives_on_clean_data(self, clean_dataframe: pd.DataFrame) -> None:
        scanner = PIIScanner()
        result = scanner.scan(clean_dataframe)
        assert not result.has_pii

    def test_columns_scanned_count(self, pii_dataframe: pd.DataFrame) -> None:
        scanner = PIIScanner()
        result = scanner.scan(pii_dataframe)
        assert result.columns_scanned == 5

    def test_compliance_tags_needed(self, pii_dataframe: pd.DataFrame) -> None:
        scanner = PIIScanner()
        result = scanner.scan(pii_dataframe)
        assert "PII" in result.compliance_tags_needed
        assert "GDPR" in result.compliance_tags_needed

    def test_summary_output(self, pii_dataframe: pd.DataFrame) -> None:
        scanner = PIIScanner()
        result = scanner.scan(pii_dataframe)
        summary = result.summary()
        assert "PII Scan" in summary
        assert "Flagged" in summary

    def test_no_data_scan(self, pii_dataframe: pd.DataFrame) -> None:
        scanner = PIIScanner(scan_data=False)
        result = scanner.scan(pii_dataframe)
        # Should still detect by column name
        assert result.has_pii

    def test_ssn_detection(self) -> None:
        df = pd.DataFrame({"ssn": ["123-45-6789", "987-65-4321", "111-22-3333"]})
        scanner = PIIScanner()
        result = scanner.scan(df)
        ssn_detections = [d for d in result.detections if d.pii_type == "ssn"]
        assert len(ssn_detections) >= 1
        assert ssn_detections[0].severity == "critical"

    def test_custom_patterns(self) -> None:
        custom = {
            "employee_id": {
                "column_keywords": ["emp_id", "employee_id"],
                "data_pattern": r"EMP\d{5}",
                "compliance_tags": ["INTERNAL"],
                "severity": "low",
            }
        }
        df = pd.DataFrame({"emp_id": ["EMP00001", "EMP00002"]})
        scanner = PIIScanner(custom_patterns=custom)
        result = scanner.scan(df)
        assert result.has_pii


class TestAutoTagCompliance:
    def test_returns_tags(self, pii_dataframe: pd.DataFrame) -> None:
        tags = auto_tag_compliance(pii_dataframe)
        assert isinstance(tags, dict)
        assert "user_email" in tags
        assert "PII" in tags["user_email"]

    def test_clean_data_no_tags(self, clean_dataframe: pd.DataFrame) -> None:
        tags = auto_tag_compliance(clean_dataframe)
        assert len(tags) == 0


class TestPIIDetection:
    def test_to_dict(self) -> None:
        d = PIIDetection(
            column_name="email",
            pii_type="email",
            detection_method="column_name",
            confidence=0.75,
            severity="high",
            compliance_tags=["PII"],
        )
        result = d.to_dict()
        assert result["column_name"] == "email"
        assert result["confidence"] == 0.75


class TestPIIScanResult:
    def test_empty_result(self) -> None:
        r = PIIScanResult()
        assert not r.has_pii
        assert r.compliance_tags_needed == []
        assert r.critical_findings == []

    def test_with_detections(self) -> None:
        r = PIIScanResult(
            detections=[
                PIIDetection("a", "email", "name", 0.9, "high", ["PII"]),
                PIIDetection("b", "ssn", "data", 0.95, "critical", ["PII", "SENSITIVE"]),
            ],
            columns_scanned=10,
            columns_flagged=2,
        )
        assert r.has_pii
        assert len(r.critical_findings) == 1
        assert "SENSITIVE" in r.compliance_tags_needed
