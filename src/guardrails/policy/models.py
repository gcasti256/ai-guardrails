"""Pydantic v2 data models for policy definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DetectorType(str, Enum):
    """Supported detector types that policy rules can reference."""

    PII = "pii"
    INJECTION = "injection"
    TOXICITY = "toxicity"
    TOPIC = "topic"
    LANGUAGE = "language"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


class RuleAction(str, Enum):
    """Action to take when a policy rule matches.

    Mirrors :class:`guardrails.types.Action` but lives here so policy
    YAML files can be validated without importing the full types module.
    """

    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"
    REDACT = "redact"


class RuleSeverity(str, Enum):
    """Severity level attached to a policy rule."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyRule(BaseModel):
    """A single rule within a policy.

    Each rule maps to one detector type and declares what action should
    be taken when the detector fires.

    Attributes:
        name: Human-readable rule identifier (unique within a policy).
        description: Optional long-form explanation of the rule.
        detector_type: Which detector family this rule invokes.
        config: Detector-specific configuration passed through verbatim.
        action: Action to take when the rule matches.
        severity: How severe a match is considered.
        enabled: Whether the rule is active.  Disabled rules are skipped
            during evaluation.
        chain: Optional list of rule names to evaluate if this rule fires.
            Enables "if A then also run B" workflows.
    """

    name: str = Field(..., min_length=1, description="Unique rule name within the policy.")
    description: str = Field(default="", description="Human-readable description of the rule.")
    detector_type: DetectorType = Field(..., description="Detector family this rule invokes.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Detector-specific configuration.",
    )
    action: RuleAction = Field(default=RuleAction.WARN, description="Action when the rule matches.")
    severity: RuleSeverity = Field(default=RuleSeverity.MEDIUM, description="Severity of a match.")
    enabled: bool = Field(default=True, description="Whether the rule is active.")
    chain: list[str] = Field(
        default_factory=list,
        description="Rule names to also evaluate when this rule fires.",
    )

    model_config = {"extra": "forbid"}


class Policy(BaseModel):
    """A named collection of rules evaluated together.

    Attributes:
        name: Policy identifier.
        version: Semantic version string (informational).
        description: What this policy is designed to enforce.
        rules: Ordered list of rules belonging to this policy.
        default_action: Fallback action when no rules match.
        metadata: Arbitrary key/value pairs for tagging / audit.
    """

    name: str = Field(..., min_length=1, description="Policy identifier.")
    version: str = Field(default="1.0.0", description="Semantic version string.")
    description: str = Field(default="", description="What this policy enforces.")
    rules: list[PolicyRule] = Field(default_factory=list, description="Ordered list of rules.")
    default_action: RuleAction = Field(
        default=RuleAction.ALLOW,
        description="Fallback action when no rules match.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary key/value pairs.")

    model_config = {"extra": "forbid"}

    def enabled_rules(self) -> list[PolicyRule]:
        """Return only the rules that are currently enabled."""
        return [r for r in self.rules if r.enabled]

    def get_rule(self, name: str) -> PolicyRule | None:
        """Look up a rule by name, or ``None`` if not found."""
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None


class PolicyConfig(BaseModel):
    """Top-level configuration that can hold multiple policies and global settings.

    Attributes:
        policies: List of policies loaded into the engine.
        global_settings: Cross-policy settings (e.g. default sensitivity,
            logging verbosity).
    """

    policies: list[Policy] = Field(default_factory=list, description="Loaded policies.")
    global_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Cross-policy settings.",
    )

    model_config = {"extra": "forbid"}
