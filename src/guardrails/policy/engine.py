"""Policy evaluation engine.

Evaluates text against a set of :class:`Policy` rules, dispatching each
rule to the appropriate detector module and aggregating the results into
a single :class:`ScanResult`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from guardrails.policy.models import (
    DetectorType,
    Policy,
    PolicyRule,
    RuleAction,
    RuleSeverity,
)
from guardrails.types import (
    Action,
    DetectionResult,
    PolicyViolation,
    ScanResult,
    Severity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity / action ordering (higher index = more severe)
# ---------------------------------------------------------------------------

_ACTION_SEVERITY_ORDER: dict[RuleAction, int] = {
    RuleAction.ALLOW: 0,
    RuleAction.WARN: 1,
    RuleAction.REDACT: 2,
    RuleAction.DENY: 3,
}

_SEVERITY_ORDER: dict[RuleSeverity, int] = {
    RuleSeverity.LOW: 0,
    RuleSeverity.MEDIUM: 1,
    RuleSeverity.HIGH: 2,
    RuleSeverity.CRITICAL: 3,
}

# Mapping from policy model enums to shared types enums.
_ACTION_MAP: dict[RuleAction, Action] = {
    RuleAction.ALLOW: Action.ALLOW,
    RuleAction.WARN: Action.WARN,
    RuleAction.DENY: Action.DENY,
    RuleAction.REDACT: Action.REDACT,
}

_SEVERITY_MAP: dict[RuleSeverity, Severity] = {
    RuleSeverity.LOW: Severity.LOW,
    RuleSeverity.MEDIUM: Severity.MEDIUM,
    RuleSeverity.HIGH: Severity.HIGH,
    RuleSeverity.CRITICAL: Severity.CRITICAL,
}

# Type alias for detector callables registered with the engine.
DetectorFunc = Callable[
    [str, dict[str, Any]],
    Coroutine[Any, Any, list[DetectionResult]],
]


class PolicyEngine:
    """Evaluate text against one or more policies.

    The engine maintains a registry of *detector functions* keyed by
    :class:`DetectorType`.  When a rule references a detector type the
    engine looks up the corresponding function and invokes it.

    Detector functions must have the signature::

        async def detect(text: str, config: dict[str, Any]) -> list[DetectionResult]

    Register detectors at construction time or later via
    :meth:`register_detector`.

    Example::

        engine = PolicyEngine()
        engine.register_detector(DetectorType.PII, pii_detect)
        result = await engine.evaluate(text, policy)
    """

    def __init__(self) -> None:
        self._detectors: dict[DetectorType, DetectorFunc] = {}

    # ------------------------------------------------------------------
    # Detector registration
    # ------------------------------------------------------------------

    def register_detector(
        self,
        detector_type: DetectorType,
        func: DetectorFunc,
    ) -> None:
        """Register an async detector function for *detector_type*.

        Args:
            detector_type: The detector family this function handles.
            func: Async callable ``(text, config) -> list[DetectionResult]``.
        """
        self._detectors[detector_type] = func

    def has_detector(self, detector_type: DetectorType) -> bool:
        """Return ``True`` if a detector is registered for *detector_type*."""
        return detector_type in self._detectors

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def evaluate(self, text: str, policy: Policy) -> ScanResult:
        """Evaluate *text* against a single *policy*.

        Independent rules are evaluated concurrently via
        :func:`asyncio.gather`.  Rules referenced through chains are
        evaluated after the triggering rule completes.

        Args:
            text: The input text to scan.
            policy: Policy whose enabled rules will be checked.

        Returns:
            Aggregated :class:`ScanResult` containing all detections and
            policy violations.
        """
        all_detections: list[DetectionResult] = []
        violations: list[PolicyViolation] = []
        worst_action = policy.default_action

        enabled_rules = policy.enabled_rules()
        if not enabled_rules:
            return ScanResult(
                text=text,
                action=_ACTION_MAP[policy.default_action],
            )

        # --- Concurrent evaluation of top-level rules ---
        tasks = [
            self._evaluate_rule(text, rule, policy.name)
            for rule in enabled_rules
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect chained rule names that need follow-up evaluation.
        pending_chains: list[tuple[PolicyRule, str]] = []  # (parent_rule, chain_name)

        for rule, result in zip(enabled_rules, results):
            if isinstance(result, Exception):
                logger.error(
                    "Rule '%s' in policy '%s' raised: %s",
                    rule.name,
                    policy.name,
                    result,
                )
                continue

            rule_detections, rule_violation = result
            if rule_detections:
                all_detections.extend(rule_detections)
            if rule_violation is not None:
                violations.append(rule_violation)
                worst_action = self._most_severe_action(worst_action, rule.action)

                # Queue chained rules.
                for chain_name in rule.chain:
                    pending_chains.append((rule, chain_name))

        # --- Chained rule evaluation (sequential to respect ordering) ---
        evaluated_chains: set[str] = set()
        for _parent_rule, chain_name in pending_chains:
            if chain_name in evaluated_chains:
                continue
            evaluated_chains.add(chain_name)

            chained_rule = policy.get_rule(chain_name)
            if chained_rule is None or not chained_rule.enabled:
                logger.warning(
                    "Chained rule '%s' not found or disabled; skipping.",
                    chain_name,
                )
                continue

            try:
                chain_detections, chain_violation = await self._evaluate_rule(
                    text, chained_rule, policy.name
                )
            except Exception:
                logger.exception("Chained rule '%s' failed.", chain_name)
                continue

            if chain_detections:
                all_detections.extend(chain_detections)
            if chain_violation is not None:
                violations.append(chain_violation)
                worst_action = self._most_severe_action(worst_action, chained_rule.action)

        return ScanResult(
            text=text,
            detections=all_detections,
            action=_ACTION_MAP[worst_action],
            policy_violations=violations,
        )

    async def evaluate_all(
        self,
        text: str,
        policies: list[Policy],
    ) -> ScanResult:
        """Evaluate *text* against multiple policies and merge results.

        Each policy is evaluated independently (concurrently).  The
        overall action is the most severe across all policies.

        Args:
            text: The input text to scan.
            policies: Policies to evaluate.

        Returns:
            Merged :class:`ScanResult`.
        """
        if not policies:
            return ScanResult(text=text)

        tasks = [self.evaluate(text, policy) for policy in policies]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged_detections: list[DetectionResult] = []
        merged_violations: list[PolicyViolation] = []
        worst_action = Action.ALLOW

        for result in results:
            if isinstance(result, Exception):
                logger.error("Policy evaluation failed: %s", result)
                continue
            merged_detections.extend(result.detections)
            merged_violations.extend(result.policy_violations)
            if _ACTION_SEVERITY_ORDER.get(
                RuleAction(result.action.value), 0
            ) > _ACTION_SEVERITY_ORDER.get(RuleAction(worst_action.value), 0):
                worst_action = result.action

        return ScanResult(
            text=text,
            detections=merged_detections,
            action=worst_action,
            policy_violations=merged_violations,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _evaluate_rule(
        self,
        text: str,
        rule: PolicyRule,
        policy_name: str,
    ) -> tuple[list[DetectionResult], PolicyViolation | None]:
        """Evaluate a single rule and return detections + optional violation.

        Returns:
            A tuple of (detections, violation).  *violation* is ``None``
            when the detector found nothing noteworthy.
        """
        detector = self._detectors.get(rule.detector_type)
        if detector is None:
            logger.warning(
                "No detector registered for type '%s' (rule '%s'); skipping.",
                rule.detector_type.value,
                rule.name,
            )
            return [], None

        detections = await detector(text, rule.config)

        if not detections:
            return [], None

        # Build a violation record.
        violation = PolicyViolation(
            rule_name=rule.name,
            policy_name=policy_name,
            severity=_SEVERITY_MAP[rule.severity],
            action=_ACTION_MAP[rule.action],
            message=(
                f"Rule '{rule.name}' triggered: "
                f"{len(detections)} detection(s) with action={rule.action.value}."
            ),
            detections=detections,
        )

        return detections, violation

    @staticmethod
    def _most_severe_action(current: RuleAction, candidate: RuleAction) -> RuleAction:
        """Return whichever action is more severe."""
        if _ACTION_SEVERITY_ORDER[candidate] > _ACTION_SEVERITY_ORDER[current]:
            return candidate
        return current
