"""PII detection and automatic compliance tagging for feature governance.

Scans feature names and data patterns to automatically detect
personally identifiable information (PII) and assign compliance tags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# PII pattern categories with regex patterns and column name keywords
PII_PATTERNS: dict[str, dict[str, Any]] = {
    "email": {
        "column_keywords": ["email", "e_mail", "mail_address"],
        "data_pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "compliance_tags": ["PII", "GDPR"],
        "severity": "high",
    },
    "phone": {
        "column_keywords": ["phone", "telephone", "mobile", "cell", "fax"],
        "data_pattern": r"[\+]?[\d\s\-\(\)]{7,15}",
        "compliance_tags": ["PII", "GDPR"],
        "severity": "high",
    },
    "ssn": {
        "column_keywords": ["ssn", "social_security", "sin", "national_id", "tax_id"],
        "data_pattern": r"\d{3}[-\s]?\d{2}[-\s]?\d{4}",
        "compliance_tags": ["PII", "SENSITIVE", "GDPR"],
        "severity": "critical",
    },
    "credit_card": {
        "column_keywords": ["credit_card", "card_number", "cc_num", "pan"],
        "data_pattern": r"\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}",
        "compliance_tags": ["PII", "SENSITIVE", "SOX"],
        "severity": "critical",
    },
    "name": {
        "column_keywords": [
            "first_name", "last_name", "full_name", "surname",
            "given_name", "family_name", "user_name", "username",
        ],
        "data_pattern": None,
        "compliance_tags": ["PII", "GDPR"],
        "severity": "medium",
    },
    "address": {
        "column_keywords": [
            "address", "street", "city", "zip_code", "zipcode",
            "postal_code", "state", "country",
        ],
        "data_pattern": None,
        "compliance_tags": ["PII", "GDPR"],
        "severity": "medium",
    },
    "date_of_birth": {
        "column_keywords": ["dob", "birth_date", "date_of_birth", "birthday"],
        "data_pattern": None,
        "compliance_tags": ["PII", "GDPR"],
        "severity": "medium",
    },
    "ip_address": {
        "column_keywords": ["ip_address", "ip_addr", "client_ip"],
        "data_pattern": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        "compliance_tags": ["PII", "GDPR"],
        "severity": "medium",
    },
    "medical": {
        "column_keywords": [
            "diagnosis", "icd_code", "medical_record", "patient_id",
            "prescription", "condition", "treatment",
        ],
        "data_pattern": None,
        "compliance_tags": ["PII", "SENSITIVE", "HIPAA"],
        "severity": "critical",
    },
    "financial": {
        "column_keywords": [
            "account_number", "routing_number", "iban", "swift",
            "bank_account", "salary", "income",
        ],
        "data_pattern": None,
        "compliance_tags": ["PII", "SENSITIVE", "SOX"],
        "severity": "high",
    },
}


@dataclass
class PIIDetection:
    """Result of PII detection for a single column.

    Attributes:
        column_name: Name of the column scanned.
        pii_type: Type of PII detected (e.g., "email", "ssn").
        detection_method: How it was detected ("column_name", "data_pattern", "both").
        confidence: Confidence score (0.0 to 1.0).
        severity: Severity level ("low", "medium", "high", "critical").
        compliance_tags: Recommended compliance tags.
        sample_matches: Sample of matched values (redacted).
    """

    column_name: str
    pii_type: str
    detection_method: str
    confidence: float
    severity: str
    compliance_tags: list[str] = field(default_factory=list)
    sample_matches: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "column_name": self.column_name,
            "pii_type": self.pii_type,
            "detection_method": self.detection_method,
            "confidence": round(self.confidence, 3),
            "severity": self.severity,
            "compliance_tags": self.compliance_tags,
            "sample_matches": self.sample_matches,
        }


@dataclass
class PIIScanResult:
    """Result of scanning an entire dataset for PII.

    Attributes:
        detections: List of PII detections found.
        columns_scanned: Total columns scanned.
        columns_flagged: Number of columns with PII.
        compliance_tags_needed: Union of all compliance tags.
    """

    detections: list[PIIDetection] = field(default_factory=list)
    columns_scanned: int = 0
    columns_flagged: int = 0

    @property
    def compliance_tags_needed(self) -> list[str]:
        """All unique compliance tags from detections."""
        tags: set[str] = set()
        for d in self.detections:
            tags.update(d.compliance_tags)
        return sorted(tags)

    @property
    def has_pii(self) -> bool:
        """Whether any PII was detected."""
        return len(self.detections) > 0

    @property
    def critical_findings(self) -> list[PIIDetection]:
        """Detections with critical severity."""
        return [d for d in self.detections if d.severity == "critical"]

    def summary(self) -> str:
        """Human-readable scan summary."""
        lines = [
            f"PII Scan Results: {self.columns_scanned} columns scanned",
            f"  Flagged: {self.columns_flagged} columns",
            f"  Compliance tags needed: {', '.join(self.compliance_tags_needed) or 'none'}",
        ]
        if self.critical_findings:
            lines.append(f"  ⚠ CRITICAL: {len(self.critical_findings)} critical findings")
        for d in self.detections:
            lines.append(
                f"  [{d.severity.upper()}] {d.column_name}: "
                f"{d.pii_type} (confidence: {d.confidence:.0%})"
            )
        return "\n".join(lines)


class PIIScanner:
    """Automated PII detection scanner for DataFrames.

    Scans column names and data patterns to detect personally
    identifiable information and recommend compliance tags.

    Args:
        scan_data: Whether to scan actual data values (not just names).
        sample_size: Number of rows to sample for data scanning.
        custom_patterns: Additional PII patterns to check.

    Example:
        >>> scanner = PIIScanner()
        >>> result = scanner.scan(df)
        >>> print(result.summary())
        >>> if result.has_pii:
        ...     for d in result.detections:
        ...         print(f"  {d.column_name}: {d.pii_type}")
    """

    def __init__(
        self,
        scan_data: bool = True,
        sample_size: int = 100,
        custom_patterns: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.scan_data = scan_data
        self.sample_size = sample_size
        self._patterns = dict(PII_PATTERNS)
        if custom_patterns:
            self._patterns.update(custom_patterns)

    def scan(self, X: pd.DataFrame) -> PIIScanResult:
        """Scan a DataFrame for PII.

        Args:
            X: DataFrame to scan.

        Returns:
            PIIScanResult with all detections.
        """
        result = PIIScanResult(columns_scanned=len(X.columns))
        flagged_columns: set[str] = set()

        for col in X.columns:
            col_lower = str(col).lower().replace(" ", "_")

            for pii_type, pattern_info in self._patterns.items():
                detection = self._check_column(
                    X, str(col), col_lower, pii_type, pattern_info
                )
                if detection is not None:
                    result.detections.append(detection)
                    flagged_columns.add(str(col))

        result.columns_flagged = len(flagged_columns)

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        result.detections.sort(
            key=lambda d: (severity_order.get(d.severity, 99), d.column_name)
        )

        if result.has_pii:
            logger.warning(
                "PII scan found %d detections in %d columns",
                len(result.detections),
                result.columns_flagged,
            )

        return result

    def _check_column(
        self,
        X: pd.DataFrame,
        col: str,
        col_lower: str,
        pii_type: str,
        pattern_info: dict[str, Any],
    ) -> PIIDetection | None:
        """Check a single column for a specific PII type."""
        name_match = self._check_column_name(col_lower, pattern_info["column_keywords"])
        data_match = False
        data_match_count = 0

        if self.scan_data and pattern_info.get("data_pattern"):
            data_match, data_match_count = self._check_data_pattern(
                X[col], pattern_info["data_pattern"]
            )

        if not name_match and not data_match:
            return None

        # Compute confidence
        if name_match and data_match:
            method = "both"
            confidence = 0.95
        elif name_match:
            method = "column_name"
            confidence = 0.75
        else:
            method = "data_pattern"
            confidence = 0.85

        return PIIDetection(
            column_name=col,
            pii_type=pii_type,
            detection_method=method,
            confidence=confidence,
            severity=pattern_info["severity"],
            compliance_tags=list(pattern_info["compliance_tags"]),
            sample_matches=data_match_count,
        )

    def _check_column_name(self, col_lower: str, keywords: list[str]) -> bool:
        """Check if column name matches PII keywords."""
        for keyword in keywords:
            if keyword in col_lower:
                return True
        return False

    def _check_data_pattern(
        self, series: pd.Series, pattern: str
    ) -> tuple[bool, int]:
        """Check if data values match a PII regex pattern."""
        try:
            sample = series.dropna().head(self.sample_size).astype(str)
            if sample.empty:
                return False, 0
            matches = sample.str.match(pattern, na=False)
            match_count = int(matches.sum())
            match_ratio = match_count / len(sample)
            return match_ratio > 0.3, match_count
        except Exception:
            return False, 0


def auto_tag_compliance(
    X: pd.DataFrame,
    scanner: PIIScanner | None = None,
) -> dict[str, list[str]]:
    """Automatically tag columns with compliance labels.

    Args:
        X: DataFrame to scan.
        scanner: PIIScanner instance. None for default.

    Returns:
        Dict mapping column names to compliance tags.
    """
    scanner = scanner or PIIScanner()
    result = scanner.scan(X)

    tags: dict[str, list[str]] = {}
    for detection in result.detections:
        col = detection.column_name
        if col not in tags:
            tags[col] = []
        for tag in detection.compliance_tags:
            if tag not in tags[col]:
                tags[col].append(tag)

    return tags
