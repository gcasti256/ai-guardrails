"""Prompt injection detection engine.

Combines regex pattern matching with heuristic semantic analysis to produce
a confidence-scored injection detection result.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from guardrails.injection.analyzer import SemanticAnalyzer, SemanticAnalysisResult
from guardrails.injection.patterns import (
    INJECTION_PATTERNS,
    InjectionPattern,
    PatternCategory,
)
from guardrails.types import Severity


# ---------------------------------------------------------------------------
# Sensitivity levels
# ---------------------------------------------------------------------------

class SensitivityLevel(str, Enum):
    """Controls the confidence threshold at which text is flagged as injection.

    Lower thresholds mean more aggressive detection (more false positives).
    Higher thresholds mean more conservative detection (fewer false positives
    but potential false negatives).
    """

    LOW = "low"          # confidence > 0.8 to flag
    MEDIUM = "medium"    # confidence > 0.5
    HIGH = "high"        # confidence > 0.3
    PARANOID = "paranoid"  # confidence > 0.1


_SENSITIVITY_THRESHOLDS: dict[SensitivityLevel, float] = {
    SensitivityLevel.LOW: 0.8,
    SensitivityLevel.MEDIUM: 0.5,
    SensitivityLevel.HIGH: 0.3,
    SensitivityLevel.PARANOID: 0.1,
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    """Details about a single pattern that matched.

    Attributes:
        pattern_name: Identifier of the matched pattern.
        category: The attack category of the matched pattern.
        severity: Severity rating of the matched pattern.
        matched_text: The literal text that triggered the match.
        start: Start index of the match in the original text.
        end: End index of the match in the original text.
        confidence_weight: The weight this pattern contributes to scoring.
    """

    pattern_name: str
    category: PatternCategory
    severity: Severity
    matched_text: str
    start: int
    end: int
    confidence_weight: float


@dataclass
class InjectionResult:
    """Complete result from injection detection analysis.

    Attributes:
        is_injection: Whether the text is classified as a prompt injection
            at the configured sensitivity level.
        confidence: Aggregated confidence score (0.0 - 1.0).
        matched_patterns: List of individual pattern matches found.
        severity: Highest severity among all matched patterns, or LOW if none.
        details: Human-readable summary and diagnostic information.
    """

    is_injection: bool
    confidence: float
    matched_patterns: list[PatternMatch] = field(default_factory=list)
    severity: Severity = Severity.LOW
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class InjectionDetector:
    """Prompt injection detection engine.

    Combines compiled regex pattern matching against a curated database with
    heuristic semantic analysis to produce a confidence-scored result.

    Args:
        sensitivity: The detection sensitivity level. Defaults to
            :attr:`SensitivityLevel.MEDIUM`.
        patterns: Optional override list of :class:`InjectionPattern` objects.
            Defaults to the built-in :data:`INJECTION_PATTERNS`.
        semantic_weight: Weight for the semantic analysis score when combining
            with pattern match scores. Must be in [0.0, 1.0]. Defaults to 0.3.

    Example::

        detector = InjectionDetector(sensitivity=SensitivityLevel.HIGH)
        result = detector.detect_sync("Ignore all previous instructions.")
        if result.is_injection:
            print(f"Injection detected (confidence={result.confidence:.2f})")
    """

    def __init__(
        self,
        *,
        sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
        patterns: list[InjectionPattern] | None = None,
        semantic_weight: float = 0.3,
    ) -> None:
        if not 0.0 <= semantic_weight <= 1.0:
            msg = f"semantic_weight must be in [0.0, 1.0], got {semantic_weight}"
            raise ValueError(msg)

        self._sensitivity = sensitivity
        self._threshold = _SENSITIVITY_THRESHOLDS[sensitivity]
        self._patterns = patterns if patterns is not None else INJECTION_PATTERNS
        self._semantic_weight = semantic_weight
        self._pattern_weight = 1.0 - semantic_weight
        self._analyzer = SemanticAnalyzer()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sensitivity(self) -> SensitivityLevel:
        """The active sensitivity level."""
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: SensitivityLevel) -> None:
        self._sensitivity = value
        self._threshold = _SENSITIVITY_THRESHOLDS[value]

    @property
    def threshold(self) -> float:
        """The confidence threshold derived from the current sensitivity."""
        return self._threshold

    # ------------------------------------------------------------------
    # Public detection methods
    # ------------------------------------------------------------------

    async def detect(self, text: str) -> InjectionResult:
        """Asynchronously analyze *text* for prompt injection.

        Runs the (CPU-bound) synchronous detection in an executor so it does
        not block the event loop.

        Args:
            text: The input text to analyze.

        Returns:
            An :class:`InjectionResult` describing the detection outcome.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.detect_sync, text)

    def detect_sync(self, text: str) -> InjectionResult:
        """Synchronously analyze *text* for prompt injection.

        This is the core detection pipeline:
        1. Run all regex patterns against the text.
        2. Perform heuristic semantic analysis.
        3. Aggregate scores into a final confidence value.
        4. Compare against the configured sensitivity threshold.

        Args:
            text: The input text to analyze.

        Returns:
            An :class:`InjectionResult` describing the detection outcome.
        """
        if not text or not text.strip():
            return InjectionResult(
                is_injection=False,
                confidence=0.0,
                severity=Severity.LOW,
                details={"reason": "empty_input"},
            )

        # Step 1: pattern matching
        matches = self._match_patterns(text)

        # Step 2: semantic analysis
        semantic_result = self._analyzer.analyze(text)

        # Step 3: aggregate confidence
        pattern_confidence = self._aggregate_pattern_confidence(matches)
        semantic_confidence = semantic_result.overall_score

        combined_confidence = (
            pattern_confidence * self._pattern_weight
            + semantic_confidence * self._semantic_weight
        )
        combined_confidence = round(min(combined_confidence, 1.0), 4)

        # Step 4: determine highest severity
        max_severity = self._resolve_severity(matches)

        # Step 5: build details
        details = self._build_details(
            matches=matches,
            semantic_result=semantic_result,
            pattern_confidence=pattern_confidence,
            semantic_confidence=semantic_confidence,
            combined_confidence=combined_confidence,
        )

        is_injection = combined_confidence >= self._threshold

        return InjectionResult(
            is_injection=is_injection,
            confidence=combined_confidence,
            matched_patterns=matches,
            severity=max_severity,
            details=details,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_patterns(self, text: str) -> list[PatternMatch]:
        """Run all patterns against *text* and collect matches."""
        matches: list[PatternMatch] = []
        for pattern in self._patterns:
            for m in pattern.regex.finditer(text):
                matches.append(
                    PatternMatch(
                        pattern_name=pattern.name,
                        category=pattern.category,
                        severity=pattern.severity,
                        matched_text=m.group(),
                        start=m.start(),
                        end=m.end(),
                        confidence_weight=pattern.confidence_weight,
                    )
                )
        return matches

    @staticmethod
    def _aggregate_pattern_confidence(matches: list[PatternMatch]) -> float:
        """Aggregate pattern match weights into a single confidence score.

        Uses an inverse-product method: ``1 - prod(1 - w_i)`` which naturally
        caps at 1.0 and gives diminishing returns for additional matches.
        """
        if not matches:
            return 0.0

        # De-duplicate by pattern name, keeping the highest-weight match.
        best: dict[str, float] = {}
        for m in matches:
            if m.pattern_name not in best or m.confidence_weight > best[m.pattern_name]:
                best[m.pattern_name] = m.confidence_weight

        complement = 1.0
        for weight in best.values():
            complement *= 1.0 - weight

        return round(1.0 - complement, 4)

    @staticmethod
    def _resolve_severity(matches: list[PatternMatch]) -> Severity:
        """Return the highest severity present in *matches*."""
        _ORDER = {
            Severity.LOW: 0,
            Severity.MEDIUM: 1,
            Severity.HIGH: 2,
            Severity.CRITICAL: 3,
        }
        if not matches:
            return Severity.LOW
        return max(matches, key=lambda m: _ORDER.get(m.severity, 0)).severity

    @staticmethod
    def _build_details(
        *,
        matches: list[PatternMatch],
        semantic_result: SemanticAnalysisResult,
        pattern_confidence: float,
        semantic_confidence: float,
        combined_confidence: float,
    ) -> dict[str, Any]:
        """Assemble a diagnostic details dictionary."""
        categories_seen: set[str] = set()
        pattern_names: list[str] = []
        for m in matches:
            categories_seen.add(m.category.value)
            if m.pattern_name not in pattern_names:
                pattern_names.append(m.pattern_name)

        return {
            "pattern_confidence": pattern_confidence,
            "semantic_confidence": semantic_confidence,
            "combined_confidence": combined_confidence,
            "pattern_match_count": len(matches),
            "unique_patterns_matched": len(pattern_names),
            "patterns_matched": pattern_names,
            "categories_detected": sorted(categories_seen),
            "semantic_signals": semantic_result.signals,
            "semantic_scores": {
                "imperative": semantic_result.imperative_score,
                "roleplay": semantic_result.roleplay_score,
                "context_manipulation": semantic_result.context_manipulation_score,
            },
        }
