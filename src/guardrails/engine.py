"""Main guardrails engine -- high-level API that ties everything together.

:class:`GuardrailsEngine` is the primary entry point for consumers of
the library.  It loads policies, wires up detectors, and exposes simple
``scan`` / ``redact`` / ``validate_prompt`` methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from guardrails.injection.detector import InjectionDetector, InjectionResult, SensitivityLevel
from guardrails.policy.engine import PolicyEngine
from guardrails.policy.loader import PolicyLoader
from guardrails.policy.models import DetectorType, Policy
from guardrails.types import (
    Action,
    DetectionResult,
    RedactionResult,
    RedactionStrategy,
    ScanResult,
    Severity,
)

logger = logging.getLogger(__name__)


class GuardrailsEngine:
    """High-level API for scanning, redacting, and validating text.

    Wraps the :class:`PolicyEngine`, :class:`PolicyLoader`, and the
    various detector / classifier modules into a single facade.

    Args:
        policy_path: Optional path to a YAML policy file or a directory
            of policies loaded at construction time.
        sensitivity_level: Baseline sensitivity that detectors may use
            to calibrate their thresholds (``"low"``, ``"medium"``,
            ``"high"``).  Defaults to ``"medium"``.

    Example::

        engine = GuardrailsEngine(policy_path="policies/")
        result = await engine.scan("Please process my order.")
        if not result.is_safe:
            print("Blocked:", result.policy_violations)
    """

    def __init__(
        self,
        *,
        policy_path: str | Path | None = None,
        sensitivity_level: str = "medium",
    ) -> None:
        self._sensitivity_level = sensitivity_level
        self._policies: list[Policy] = []
        self._loader = PolicyLoader()
        self._policy_engine = PolicyEngine()

        # Register built-in detectors.
        self._register_builtin_detectors()

        # Eagerly load policies if a path was given.
        if policy_path is not None:
            p = Path(policy_path)
            if p.is_dir():
                self.load_policies_dir(p)
            else:
                self.load_policy(p)

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def load_policy(self, path: str | Path) -> None:
        """Load a single YAML policy file and add it to the engine.

        Args:
            path: Filesystem path to a ``.yaml`` / ``.yml`` file.
        """
        policy = self._loader.load_file(path)
        errors = self._loader.validate(policy)
        if errors:
            logger.warning(
                "Policy '%s' loaded with validation warnings: %s",
                policy.name,
                "; ".join(errors),
            )
        self._policies.append(policy)
        logger.info("Loaded policy '%s' (v%s).", policy.name, policy.version)

    def load_policies_dir(self, path: str | Path) -> None:
        """Load all YAML policy files from a directory.

        Args:
            path: Directory containing policy files.
        """
        policies = self._loader.load_directory(path)
        for policy in policies:
            errors = self._loader.validate(policy)
            if errors:
                logger.warning(
                    "Policy '%s' has validation warnings: %s",
                    policy.name,
                    "; ".join(errors),
                )
        self._policies.extend(policies)
        logger.info("Loaded %d policies from %s.", len(policies), path)

    @property
    def policies(self) -> list[Policy]:
        """Return a copy of the currently loaded policies."""
        return list(self._policies)

    @property
    def sensitivity_level(self) -> str:
        """Return the configured sensitivity level."""
        return self._sensitivity_level

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    async def scan(self, text: str) -> ScanResult:
        """Scan *text* against all loaded policies.

        Args:
            text: Input text to scan.

        Returns:
            Aggregated :class:`ScanResult` with detections, violations,
            and the resolved action.
        """
        if not self._policies:
            logger.debug("No policies loaded; returning ALLOW for scan.")
            return ScanResult(text=text, action=Action.ALLOW)

        return await self._policy_engine.evaluate_all(text, self._policies)

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------

    async def redact(
        self,
        text: str,
        strategy: RedactionStrategy = RedactionStrategy.REPLACE,
    ) -> RedactionResult:
        """Scan for PII and return a redacted copy of *text*.

        This is a convenience wrapper that runs the PII detector, then
        applies *strategy* to each detection.

        Args:
            text: Input text to redact.
            strategy: How to replace detected entities.

        Returns:
            :class:`RedactionResult` with the redacted text and details.
        """
        from guardrails.pii.detector import PIIDetector
        from guardrails.pii.redactor import PIIRedactor

        detector = PIIDetector()
        detections = await detector.detect(text)
        redactor = PIIRedactor(default_strategy=strategy)
        return redactor.redact(text, detections)

    # ------------------------------------------------------------------
    # Prompt injection validation
    # ------------------------------------------------------------------

    async def validate_prompt(self, text: str) -> InjectionResult:
        """Run a targeted injection check on *text*.

        This is a lightweight entry point when the caller only cares
        about prompt-injection risk rather than a full policy scan.

        Args:
            text: Prompt text to validate.

        Returns:
            :class:`InjectionResult` with injection classification.
        """
        level = SensitivityLevel(self._sensitivity_level)
        detector = InjectionDetector(sensitivity=level)
        return await detector.detect(text)

    # ------------------------------------------------------------------
    # Detector wiring
    # ------------------------------------------------------------------

    def _register_builtin_detectors(self) -> None:
        """Wire built-in detector modules into the policy engine.

        Each detector is wrapped in an async adapter that matches the
        signature expected by :class:`PolicyEngine`.
        """
        self._policy_engine.register_detector(
            DetectorType.PII,
            self._detect_pii,
        )
        self._policy_engine.register_detector(
            DetectorType.INJECTION,
            self._detect_injection,
        )
        self._policy_engine.register_detector(
            DetectorType.TOXICITY,
            self._detect_toxicity,
        )
        self._policy_engine.register_detector(
            DetectorType.TOPIC,
            self._detect_topic,
        )
        self._policy_engine.register_detector(
            DetectorType.LANGUAGE,
            self._detect_language,
        )
        self._policy_engine.register_detector(
            DetectorType.SENTIMENT,
            self._detect_sentiment,
        )

    # ------------------------------------------------------------------
    # Detector adapters
    # ------------------------------------------------------------------

    async def _detect_pii(
        self, text: str, config: dict[str, Any]
    ) -> list[DetectionResult]:
        """Adapter for the PII detector module."""
        from guardrails.pii.detector import PIIDetector

        entity_types = config.get("entity_types")
        min_confidence = config.get("min_confidence", 0.0)
        detector = PIIDetector(
            enabled_entity_types=set(entity_types) if entity_types else None,
            min_confidence=min_confidence,
        )
        return await detector.detect(text)

    async def _detect_injection(
        self, text: str, config: dict[str, Any]
    ) -> list[DetectionResult]:
        """Adapter for the injection detector module."""
        raw_sensitivity = config.get("sensitivity", self._sensitivity_level)
        min_confidence = config.get("min_confidence", 0.0)
        level = SensitivityLevel(raw_sensitivity)
        detector = InjectionDetector(sensitivity=level)
        result = await detector.detect(text)
        if result.is_injection and result.confidence >= min_confidence:
            return [
                DetectionResult(
                    entity_type="PROMPT_INJECTION",
                    text=text,
                    start=0,
                    end=len(text),
                    confidence=result.confidence,
                    detector="injection",
                    severity=result.severity if result.severity else Severity.HIGH,
                )
            ]
        return []

    async def _detect_toxicity(
        self, text: str, config: dict[str, Any]
    ) -> list[DetectionResult]:
        """Adapter for the toxicity classifier module."""
        from guardrails.classification.toxicity import ToxicityClassifier

        threshold = config.get("threshold", 0.5)
        classifier = ToxicityClassifier(threshold=threshold)
        result = await classifier.classify(text)
        if result.is_toxic:
            return [
                DetectionResult(
                    entity_type="TOXICITY",
                    text=text,
                    start=0,
                    end=len(text),
                    confidence=result.overall_score,
                    detector="toxicity",
                    severity=Severity.HIGH,
                    metadata={
                        "flagged_categories": result.flagged_categories,
                        "category_scores": result.category_scores,
                    },
                )
            ]
        return []

    async def _detect_topic(
        self, text: str, config: dict[str, Any]
    ) -> list[DetectionResult]:
        """Adapter for the topic classifier module."""
        from guardrails.classification.topic import TopicClassifier

        allowed = config.get("allowed_topics")
        classifier = TopicClassifier()
        result = await classifier.classify(text, allowed_topics=allowed)
        if not result.is_on_topic:
            return [
                DetectionResult(
                    entity_type="OFF_TOPIC",
                    text=text,
                    start=0,
                    end=len(text),
                    confidence=result.confidence,
                    detector="topic",
                    severity=Severity.MEDIUM,
                    metadata={"detected_topics": result.detected_topics},
                )
            ]
        return []

    async def _detect_language(
        self, text: str, config: dict[str, Any]
    ) -> list[DetectionResult]:
        """Adapter for the language detector module."""
        from guardrails.classification.language import LanguageDetector

        allowed = config.get("allowed_languages")
        detector = LanguageDetector(allowed_languages=allowed)
        result = await detector.detect(text)
        if not result.is_allowed:
            return [
                DetectionResult(
                    entity_type="DISALLOWED_LANGUAGE",
                    text=text,
                    start=0,
                    end=len(text),
                    confidence=result.confidence,
                    detector="language",
                    severity=Severity.LOW,
                    metadata={"detected_language": result.language},
                )
            ]
        return []

    async def _detect_sentiment(
        self, text: str, config: dict[str, Any]
    ) -> list[DetectionResult]:
        """Adapter for the sentiment analyzer module."""
        from guardrails.classification.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = await analyzer.analyze(text)
        min_score = config.get("min_score", -0.7)
        flag_negative = config.get("flag_negative", False)
        if flag_negative and result.score <= min_score:
            return [
                DetectionResult(
                    entity_type="NEGATIVE_SENTIMENT",
                    text=text,
                    start=0,
                    end=len(text),
                    confidence=result.confidence,
                    detector="sentiment",
                    severity=Severity.MEDIUM,
                    metadata={"sentiment": result.sentiment.value, "score": result.score},
                )
            ]
        return []
