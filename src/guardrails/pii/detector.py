"""Regex-based PII detector.

The PIIDetector scans text using compiled regex patterns and optional
validators (e.g. the Luhn algorithm for credit cards).  It supports both
synchronous and asynchronous detection, configurable entity types, and
automatic deduplication of overlapping matches.
"""

from __future__ import annotations

import asyncio
import re
from typing import Iterable, Sequence

from guardrails.pii.patterns import ALL_PATTERNS, PIIPattern
from guardrails.types import DetectionResult


class PIIDetector:
    """Regex-based PII detector.

    Args:
        patterns: Patterns to use.  Defaults to all built-in patterns.
        enabled_entity_types: If provided, only patterns whose entity_type
            is in this set will be used.  ``None`` means all patterns.
        disabled_entity_types: Entity types to explicitly exclude even if
            they appear in *enabled_entity_types*.
        min_confidence: Discard matches below this confidence threshold.

    Example::

        detector = PIIDetector(enabled_entity_types={"SSN", "CREDIT_CARD"})
        results = detector.detect_sync("My SSN is 123-45-6789")
    """

    def __init__(
        self,
        *,
        patterns: Sequence[PIIPattern] | None = None,
        enabled_entity_types: set[str] | None = None,
        disabled_entity_types: set[str] | None = None,
        min_confidence: float = 0.0,
    ) -> None:
        self._all_patterns = list(patterns or ALL_PATTERNS)
        self._enabled_entity_types = enabled_entity_types
        self._disabled_entity_types = disabled_entity_types or set()
        self._min_confidence = min_confidence
        self._active_patterns: list[PIIPattern] = self._resolve_patterns()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _resolve_patterns(self) -> list[PIIPattern]:
        """Build the effective pattern list based on enable/disable config."""
        patterns: list[PIIPattern] = []
        for p in self._all_patterns:
            if self._enabled_entity_types is not None and p.entity_type not in self._enabled_entity_types:
                continue
            if p.entity_type in self._disabled_entity_types:
                continue
            if p.confidence < self._min_confidence:
                continue
            patterns.append(p)
        return patterns

    def enable_entity_type(self, entity_type: str) -> None:
        """Enable detection of a specific entity type at runtime.

        Args:
            entity_type: The entity type to enable (e.g. ``"SSN"``).
        """
        if self._enabled_entity_types is not None:
            self._enabled_entity_types.add(entity_type)
        self._disabled_entity_types.discard(entity_type)
        self._active_patterns = self._resolve_patterns()

    def disable_entity_type(self, entity_type: str) -> None:
        """Disable detection of a specific entity type at runtime.

        Args:
            entity_type: The entity type to disable.
        """
        self._disabled_entity_types.add(entity_type)
        self._active_patterns = self._resolve_patterns()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    async def detect(self, text: str) -> list[DetectionResult]:
        """Detect PII in *text* asynchronously.

        Runs the synchronous detection in the default executor so it does
        not block the event loop.

        Args:
            text: The text to scan.

        Returns:
            Deduplicated list of :class:`DetectionResult` sorted by start
            offset.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.detect_sync, text)

    def detect_sync(self, text: str) -> list[DetectionResult]:
        """Detect PII in *text* synchronously.

        Args:
            text: The text to scan.

        Returns:
            Deduplicated list of :class:`DetectionResult` sorted by start
            offset.
        """
        raw: list[DetectionResult] = []
        for pattern in self._active_patterns:
            raw.extend(self._match_pattern(text, pattern))
        deduped = self._deduplicate(raw)
        deduped.sort(key=lambda d: d.start)
        return deduped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_pattern(
        self,
        text: str,
        pattern: PIIPattern,
    ) -> Iterable[DetectionResult]:
        """Yield DetectionResult for every match of *pattern* in *text*."""
        for match in pattern.pattern.finditer(text):
            matched_text = match.group(0)
            # Run optional validator (e.g. Luhn check for credit cards)
            if pattern.validator is not None and not pattern.validator(matched_text):
                continue
            yield DetectionResult(
                entity_type=pattern.entity_type,
                text=matched_text,
                start=match.start(),
                end=match.end(),
                confidence=pattern.confidence,
                detector="regex",
                severity=pattern.severity,
                metadata={"pattern_name": pattern.name},
            )

    @staticmethod
    def _deduplicate(results: list[DetectionResult]) -> list[DetectionResult]:
        """Remove overlapping detections, keeping the highest-confidence one.

        Two detections overlap if their ``[start, end)`` spans intersect.
        When spans overlap, only the detection with the higher confidence
        (or the earlier one in case of a tie) is retained.
        """
        if not results:
            return []

        # Sort by start, then by confidence descending so that the first
        # item in an overlapping cluster is the best candidate.
        sorted_results = sorted(results, key=lambda d: (d.start, -d.confidence))
        kept: list[DetectionResult] = [sorted_results[0]]

        for current in sorted_results[1:]:
            prev = kept[-1]
            if current.start < prev.end:
                # Overlap — keep the one with higher confidence
                if current.confidence > prev.confidence:
                    kept[-1] = current
                # Otherwise keep prev (already in kept)
            else:
                kept.append(current)

        return kept
