"""Shared type definitions for the guardrails library."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Severity level for detected issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(str, Enum):
    """Action to take when a policy rule matches."""
    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"
    REDACT = "redact"


class RedactionStrategy(str, Enum):
    """Strategy for redacting detected PII."""
    MASK = "mask"           # Replace with ****
    HASH = "hash"           # Replace with hash
    REPLACE = "replace"     # Replace with entity type label
    REMOVE = "remove"       # Remove entirely


@dataclass
class DetectionResult:
    """Result from a single detection check."""
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    detector: str
    severity: Severity = Severity.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Aggregated result from scanning text."""
    text: str
    detections: list[DetectionResult] = field(default_factory=list)
    action: Action = Action.ALLOW
    policy_violations: list[PolicyViolation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.action != Action.DENY

    @property
    def has_warnings(self) -> bool:
        return self.action == Action.WARN or len(self.policy_violations) > 0


@dataclass
class PolicyViolation:
    """A specific policy rule violation."""
    rule_name: str
    policy_name: str
    severity: Severity
    action: Action
    message: str
    detections: list[DetectionResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RedactionResult:
    """Result from redacting text."""
    original_text: str
    redacted_text: str
    redactions: list[RedactionDetail] = field(default_factory=list)


@dataclass
class RedactionDetail:
    """Detail about a single redaction applied."""
    entity_type: str
    original: str
    replacement: str
    start: int
    end: int
    strategy: RedactionStrategy
