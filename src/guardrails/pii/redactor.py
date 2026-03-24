"""PII redaction engine.

Takes detection results and applies a chosen redaction strategy to replace
sensitive spans in the original text.
"""

from __future__ import annotations

import hashlib

from guardrails.types import (
    DetectionResult,
    RedactionDetail,
    RedactionResult,
    RedactionStrategy,
)


class PIIRedactor:
    """Redact detected PII from text.

    Args:
        default_strategy: The default redaction strategy to use when none
            is specified per call.
        entity_strategies: Optional per-entity-type strategy overrides.
            For example ``{"SSN": RedactionStrategy.MASK,
            "EMAIL_ADDRESS": RedactionStrategy.REPLACE}``.
        mask_char: Character used for the MASK strategy.
        hash_length: Number of hex characters to keep for the HASH
            strategy.

    Example::

        redactor = PIIRedactor(default_strategy=RedactionStrategy.REPLACE)
        result = redactor.redact(text, detections)
        print(result.redacted_text)
    """

    def __init__(
        self,
        *,
        default_strategy: RedactionStrategy = RedactionStrategy.REPLACE,
        entity_strategies: dict[str, RedactionStrategy] | None = None,
        mask_char: str = "*",
        hash_length: int = 8,
    ) -> None:
        self._default_strategy = default_strategy
        self._entity_strategies = entity_strategies or {}
        self._mask_char = mask_char
        self._hash_length = hash_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(
        self,
        text: str,
        detections: list[DetectionResult],
        strategy: RedactionStrategy | None = None,
    ) -> RedactionResult:
        """Redact detected PII spans from *text*.

        Detections are processed from the **end** of the string to the
        **start** so that earlier offsets remain valid as replacements
        change the string length.

        If two detections overlap, the one that starts later is applied
        first.  If they share the same start, the longer span wins and
        the shorter one is skipped.

        Args:
            text: The original text.
            detections: List of :class:`DetectionResult` indicating PII
                spans.
            strategy: Override the redaction strategy for this call.
                When ``None``, the per-entity or default strategy is used.

        Returns:
            A :class:`RedactionResult` with the redacted text and details
            about each redaction applied.
        """
        if not detections:
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                redactions=[],
            )

        # Resolve overlaps: sort by start desc, then by span length desc.
        sorted_detections = sorted(
            detections,
            key=lambda d: (-d.start, -(d.end - d.start)),
        )

        redacted = text
        applied: list[RedactionDetail] = []
        last_start: int | None = None

        for detection in sorted_detections:
            # Skip if this detection's span was already covered by a
            # previously applied (later-starting or longer) detection.
            if last_start is not None and detection.end > last_start:
                # Overlap with an already-applied redaction — skip.
                continue

            effective_strategy = self._resolve_strategy(
                detection.entity_type,
                strategy,
            )
            replacement = self._build_replacement(
                detection.text,
                detection.entity_type,
                effective_strategy,
            )

            # Apply the replacement in the string.
            redacted = (
                redacted[: detection.start]
                + replacement
                + redacted[detection.end :]
            )

            applied.append(
                RedactionDetail(
                    entity_type=detection.entity_type,
                    original=detection.text,
                    replacement=replacement,
                    start=detection.start,
                    end=detection.end,
                    strategy=effective_strategy,
                )
            )
            last_start = detection.start

        # Reverse so the list is in document order (start ascending).
        applied.reverse()

        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            redactions=applied,
        )

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------

    def _resolve_strategy(
        self,
        entity_type: str,
        override: RedactionStrategy | None,
    ) -> RedactionStrategy:
        """Determine the effective strategy for a given entity type."""
        if override is not None:
            return override
        return self._entity_strategies.get(entity_type, self._default_strategy)

    def _build_replacement(
        self,
        original: str,
        entity_type: str,
        strategy: RedactionStrategy,
    ) -> str:
        """Build the replacement string for a matched span.

        Args:
            original: The original matched text.
            entity_type: The entity type label.
            strategy: The redaction strategy to apply.

        Returns:
            The replacement string.
        """
        if strategy == RedactionStrategy.MASK:
            return self._mask_char * len(original)

        if strategy == RedactionStrategy.HASH:
            digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
            truncated = digest[: self._hash_length]
            return f"[{truncated}]"

        if strategy == RedactionStrategy.REPLACE:
            return f"[{entity_type}]"

        if strategy == RedactionStrategy.REMOVE:
            return ""

        # Fallback — should not happen with the enum, but be safe.
        return f"[{entity_type}]"
