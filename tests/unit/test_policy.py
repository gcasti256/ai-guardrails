"""Unit tests for policy modules: loader, models, engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.policy.loader import PolicyLoader, PolicyLoadError
from guardrails.policy.models import (
    DetectorType,
    Policy,
    PolicyRule,
    RuleAction,
    RuleSeverity,
)
from guardrails.policy.engine import PolicyEngine
from guardrails.types import Action, DetectionResult, ScanResult, Severity


POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "policies"


# ---------------------------------------------------------------------------
# PolicyLoader
# ---------------------------------------------------------------------------


class TestPolicyLoader:
    """Tests for PolicyLoader."""

    def test_load_yaml_file(self, policy_loader: PolicyLoader) -> None:
        """Loading a real policy YAML file should return a valid Policy."""
        policy = policy_loader.load_file(POLICIES_DIR / "enterprise-default.yml")
        assert isinstance(policy, Policy)
        assert policy.name == "enterprise-default"
        assert len(policy.rules) > 0

    def test_load_directory(self, policy_loader: PolicyLoader) -> None:
        """Loading a directory should return one Policy per YAML file."""
        policies = policy_loader.load_directory(POLICIES_DIR)
        assert isinstance(policies, list)
        # The policies/ directory contains at least 4 files.
        assert len(policies) >= 4
        names = {p.name for p in policies}
        assert "enterprise-default" in names
        assert "strict" in names

    def test_validate_valid_policy(self, policy_loader: PolicyLoader) -> None:
        """A well-formed policy should produce no validation errors."""
        policy = policy_loader.load_file(POLICIES_DIR / "enterprise-default.yml")
        errors = policy_loader.validate(policy)
        assert errors == []

    def test_validate_catches_duplicate_names(self, policy_loader: PolicyLoader) -> None:
        """A policy with duplicate rule names should produce a validation error."""
        policy = Policy(
            name="test-dupes",
            rules=[
                PolicyRule(name="dup-rule", detector_type=DetectorType.PII),
                PolicyRule(name="dup-rule", detector_type=DetectorType.TOXICITY),
            ],
        )
        errors = policy_loader.validate(policy)
        assert any("Duplicate rule name" in e for e in errors)

    def test_load_string_works(self, policy_loader: PolicyLoader) -> None:
        """load_string should parse a raw YAML string into a Policy."""
        raw = """
name: inline-test
version: "1.0"
description: A test policy loaded from a string
rules:
  - name: test-rule
    detector_type: pii
    action: warn
    severity: medium
    enabled: true
default_action: allow
"""
        policy = policy_loader.load_string(raw)
        assert policy.name == "inline-test"
        assert len(policy.rules) == 1
        assert policy.rules[0].name == "test-rule"

    def test_load_file_not_found(self, policy_loader: PolicyLoader) -> None:
        """Loading a nonexistent file should raise PolicyLoadError."""
        with pytest.raises(PolicyLoadError, match="not found"):
            policy_loader.load_file("/nonexistent/path/fake.yaml")

    def test_load_string_invalid_yaml(self, policy_loader: PolicyLoader) -> None:
        """Malformed YAML should raise PolicyLoadError."""
        with pytest.raises(PolicyLoadError, match="YAML parse error"):
            policy_loader.load_string(":\n  - :\n    :\n  :::bad")

    def test_validate_catches_self_referencing_chain(self, policy_loader: PolicyLoader) -> None:
        """A rule that chains to itself should be flagged as circular."""
        policy = Policy(
            name="self-chain",
            rules=[
                PolicyRule(
                    name="loop-rule",
                    detector_type=DetectorType.TOXICITY,
                    chain=["loop-rule"],
                ),
            ],
        )
        errors = policy_loader.validate(policy)
        assert any("chains to itself" in e for e in errors)

    def test_merge_policies(self, policy_loader: PolicyLoader) -> None:
        """Merging policies should combine rules, with later rules winning on name conflicts."""
        p1 = Policy(
            name="first",
            rules=[
                PolicyRule(name="shared", detector_type=DetectorType.PII, action=RuleAction.WARN),
                PolicyRule(name="only-in-first", detector_type=DetectorType.TOXICITY),
            ],
            default_action=RuleAction.ALLOW,
        )
        p2 = Policy(
            name="second",
            rules=[
                PolicyRule(name="shared", detector_type=DetectorType.PII, action=RuleAction.DENY),
                PolicyRule(name="only-in-second", detector_type=DetectorType.INJECTION),
            ],
        )
        merged = PolicyLoader.merge([p1, p2], name="merged")
        assert merged.name == "merged"
        # "shared" should come from p2 (last wins).
        shared_rule = merged.get_rule("shared")
        assert shared_rule is not None
        assert shared_rule.action == RuleAction.DENY
        # Both unique rules should be present.
        assert merged.get_rule("only-in-first") is not None
        assert merged.get_rule("only-in-second") is not None


# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------


class TestPolicyModel:
    """Tests for the Policy and PolicyRule Pydantic models."""

    def test_enabled_rules_filters_correctly(self) -> None:
        """enabled_rules() should exclude disabled rules."""
        policy = Policy(
            name="filter-test",
            rules=[
                PolicyRule(name="active", detector_type=DetectorType.PII, enabled=True),
                PolicyRule(name="disabled", detector_type=DetectorType.TOXICITY, enabled=False),
                PolicyRule(name="also-active", detector_type=DetectorType.INJECTION, enabled=True),
            ],
        )
        enabled = policy.enabled_rules()
        assert len(enabled) == 2
        enabled_names = {r.name for r in enabled}
        assert "active" in enabled_names
        assert "also-active" in enabled_names
        assert "disabled" not in enabled_names

    def test_get_rule_finds_by_name(self) -> None:
        """get_rule() should return the matching rule or None."""
        policy = Policy(
            name="lookup-test",
            rules=[
                PolicyRule(name="alpha", detector_type=DetectorType.PII),
                PolicyRule(name="beta", detector_type=DetectorType.TOXICITY),
            ],
        )
        assert policy.get_rule("alpha") is not None
        assert policy.get_rule("alpha").name == "alpha"
        assert policy.get_rule("beta") is not None
        assert policy.get_rule("nonexistent") is None

    def test_policy_default_values(self) -> None:
        """A minimal Policy should have sensible defaults."""
        policy = Policy(name="minimal")
        assert policy.version == "1.0.0"
        assert policy.description == ""
        assert policy.rules == []
        assert policy.default_action == RuleAction.ALLOW
        assert policy.metadata == {}

    def test_rule_default_values(self) -> None:
        """A minimal PolicyRule should have sensible defaults."""
        rule = PolicyRule(name="minimal-rule", detector_type=DetectorType.CUSTOM)
        assert rule.action == RuleAction.WARN
        assert rule.severity == RuleSeverity.MEDIUM
        assert rule.enabled is True
        assert rule.chain == []
        assert rule.config == {}


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    @pytest.mark.asyncio
    async def test_evaluate_with_no_rules(self) -> None:
        """Evaluating a policy with no enabled rules should return the default action."""
        engine = PolicyEngine()
        policy = Policy(name="empty", default_action=RuleAction.ALLOW)
        result = await engine.evaluate("hello world", policy)
        assert isinstance(result, ScanResult)
        assert result.action == Action.ALLOW
        assert result.detections == []
        assert result.policy_violations == []

    @pytest.mark.asyncio
    async def test_evaluate_with_matching_rule(self) -> None:
        """A rule whose detector returns detections should produce a violation."""
        engine = PolicyEngine()

        async def fake_detector(text: str, config: dict) -> list[DetectionResult]:
            return [
                DetectionResult(
                    entity_type="TEST",
                    text="match",
                    start=0,
                    end=5,
                    confidence=0.95,
                    detector="fake",
                    severity=Severity.HIGH,
                ),
            ]

        engine.register_detector(DetectorType.CUSTOM, fake_detector)

        policy = Policy(
            name="test-policy",
            rules=[
                PolicyRule(
                    name="custom-check",
                    detector_type=DetectorType.CUSTOM,
                    action=RuleAction.DENY,
                    severity=RuleSeverity.HIGH,
                ),
            ],
        )
        result = await engine.evaluate("some text", policy)
        assert result.action == Action.DENY
        assert len(result.policy_violations) == 1
        assert result.policy_violations[0].rule_name == "custom-check"
        assert len(result.detections) == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_no_match(self) -> None:
        """A rule whose detector returns nothing should not produce a violation."""
        engine = PolicyEngine()

        async def empty_detector(text: str, config: dict) -> list[DetectionResult]:
            return []

        engine.register_detector(DetectorType.CUSTOM, empty_detector)

        policy = Policy(
            name="test-policy",
            rules=[
                PolicyRule(
                    name="silent-rule",
                    detector_type=DetectorType.CUSTOM,
                    action=RuleAction.DENY,
                ),
            ],
        )
        result = await engine.evaluate("clean text", policy)
        assert result.action == Action.ALLOW
        assert result.policy_violations == []

    @pytest.mark.asyncio
    async def test_evaluate_all_merges_results(self) -> None:
        """evaluate_all should merge violations and pick the most severe action."""
        engine = PolicyEngine()

        async def warn_detector(text: str, config: dict) -> list[DetectionResult]:
            return [
                DetectionResult(
                    entity_type="A",
                    text="a",
                    start=0,
                    end=1,
                    confidence=0.9,
                    detector="warn_det",
                ),
            ]

        async def deny_detector(text: str, config: dict) -> list[DetectionResult]:
            return [
                DetectionResult(
                    entity_type="B",
                    text="b",
                    start=0,
                    end=1,
                    confidence=0.9,
                    detector="deny_det",
                ),
            ]

        engine.register_detector(DetectorType.PII, warn_detector)
        engine.register_detector(DetectorType.INJECTION, deny_detector)

        policy_warn = Policy(
            name="warn-policy",
            rules=[
                PolicyRule(name="warn-rule", detector_type=DetectorType.PII, action=RuleAction.WARN),
            ],
        )
        policy_deny = Policy(
            name="deny-policy",
            rules=[
                PolicyRule(name="deny-rule", detector_type=DetectorType.INJECTION, action=RuleAction.DENY),
            ],
        )

        result = await engine.evaluate_all("text", [policy_warn, policy_deny])
        # The most severe action across both policies should win.
        assert result.action == Action.DENY
        assert len(result.policy_violations) == 2
        assert len(result.detections) == 2

    @pytest.mark.asyncio
    async def test_evaluate_all_with_empty_policies(self) -> None:
        """evaluate_all with an empty list should return a default ALLOW result."""
        engine = PolicyEngine()
        result = await engine.evaluate_all("hello", [])
        assert result.action == Action.ALLOW
        assert result.detections == []

    @pytest.mark.asyncio
    async def test_has_detector(self) -> None:
        """has_detector should correctly report registration state."""
        engine = PolicyEngine()
        assert engine.has_detector(DetectorType.PII) is False

        async def noop(text: str, config: dict) -> list[DetectionResult]:
            return []

        engine.register_detector(DetectorType.PII, noop)
        assert engine.has_detector(DetectorType.PII) is True

    @pytest.mark.asyncio
    async def test_missing_detector_skips_rule(self) -> None:
        """If no detector is registered for a rule's type, the rule should be skipped gracefully."""
        engine = PolicyEngine()
        policy = Policy(
            name="no-detector",
            rules=[
                PolicyRule(name="orphan", detector_type=DetectorType.TOXICITY, action=RuleAction.DENY),
            ],
        )
        # Should not raise; the rule is simply skipped.
        result = await engine.evaluate("some text", policy)
        assert result.action == Action.ALLOW
        assert result.policy_violations == []

    @pytest.mark.asyncio
    async def test_worst_action_escalation(self) -> None:
        """When multiple rules match, the most severe action should win."""
        engine = PolicyEngine()

        async def always_detect(text: str, config: dict) -> list[DetectionResult]:
            return [
                DetectionResult(
                    entity_type="X",
                    text="x",
                    start=0,
                    end=1,
                    confidence=0.99,
                    detector="always",
                ),
            ]

        engine.register_detector(DetectorType.PII, always_detect)
        engine.register_detector(DetectorType.TOXICITY, always_detect)

        policy = Policy(
            name="escalation",
            rules=[
                PolicyRule(name="warn-rule", detector_type=DetectorType.PII, action=RuleAction.WARN),
                PolicyRule(name="deny-rule", detector_type=DetectorType.TOXICITY, action=RuleAction.DENY),
            ],
        )
        result = await engine.evaluate("text", policy)
        assert result.action == Action.DENY
