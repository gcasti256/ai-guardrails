"""Unit tests for the PII redactor module."""

from __future__ import annotations

import hashlib

import pytest

from guardrails.pii.redactor import PIIRedactor
from guardrails.types import DetectionResult, RedactionStrategy, Severity


# ── Helpers ───────────────────────────────────────────────────────────


def _make_detection(
    entity_type: str,
    text: str,
    start: int,
    end: int,
    confidence: float = 0.90,
) -> DetectionResult:
    """Create a DetectionResult for testing."""
    return DetectionResult(
        entity_type=entity_type,
        text=text,
        start=start,
        end=end,
        confidence=confidence,
        detector="regex",
        severity=Severity.HIGH,
    )


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def replace_redactor() -> PIIRedactor:
    """Redactor using the REPLACE strategy."""
    return PIIRedactor(default_strategy=RedactionStrategy.REPLACE)


@pytest.fixture
def mask_redactor() -> PIIRedactor:
    """Redactor using the MASK strategy."""
    return PIIRedactor(default_strategy=RedactionStrategy.MASK)


@pytest.fixture
def hash_redactor() -> PIIRedactor:
    """Redactor using the HASH strategy."""
    return PIIRedactor(default_strategy=RedactionStrategy.HASH)


@pytest.fixture
def remove_redactor() -> PIIRedactor:
    """Redactor using the REMOVE strategy."""
    return PIIRedactor(default_strategy=RedactionStrategy.REMOVE)


# ── REPLACE strategy ─────────────────────────────────────────────────


class TestReplaceStrategy:
    """Tests for RedactionStrategy.REPLACE."""

    def test_replaces_with_entity_type_label(
        self, replace_redactor: PIIRedactor
    ) -> None:
        text = "My SSN is 123-45-6789"
        detections = [_make_detection("SSN", "123-45-6789", 10, 21)]
        result = replace_redactor.redact(text, detections)
        assert result.redacted_text == "My SSN is [SSN]"

    def test_replace_preserves_surrounding_text(
        self, replace_redactor: PIIRedactor
    ) -> None:
        text = "before 123-45-6789 after"
        detections = [_make_detection("SSN", "123-45-6789", 7, 18)]
        result = replace_redactor.redact(text, detections)
        assert result.redacted_text == "before [SSN] after"

    def test_replace_redaction_detail(self, replace_redactor: PIIRedactor) -> None:
        text = "Email: user@example.com"
        detections = [_make_detection("EMAIL_ADDRESS", "user@example.com", 7, 23)]
        result = replace_redactor.redact(text, detections)
        assert len(result.redactions) == 1
        detail = result.redactions[0]
        assert detail.entity_type == "EMAIL_ADDRESS"
        assert detail.original == "user@example.com"
        assert detail.replacement == "[EMAIL_ADDRESS]"
        assert detail.strategy == RedactionStrategy.REPLACE


# ── MASK strategy ─────────────────────────────────────────────────────


class TestMaskStrategy:
    """Tests for RedactionStrategy.MASK."""

    def test_masks_with_asterisks_matching_length(
        self, mask_redactor: PIIRedactor
    ) -> None:
        text = "SSN: 123-45-6789"
        detections = [_make_detection("SSN", "123-45-6789", 5, 16)]
        result = mask_redactor.redact(text, detections)
        # "123-45-6789" is 11 chars -> 11 asterisks
        assert result.redacted_text == "SSN: ***********"

    def test_mask_uses_custom_char(self) -> None:
        redactor = PIIRedactor(
            default_strategy=RedactionStrategy.MASK, mask_char="#"
        )
        text = "SSN: 123-45-6789"
        detections = [_make_detection("SSN", "123-45-6789", 5, 16)]
        result = redactor.redact(text, detections)
        assert result.redacted_text == "SSN: ###########"

    def test_mask_length_matches_original(self, mask_redactor: PIIRedactor) -> None:
        original = "user@example.com"
        text = f"Email: {original}"
        detections = [_make_detection("EMAIL_ADDRESS", original, 7, 7 + len(original))]
        result = mask_redactor.redact(text, detections)
        masked_portion = result.redacted_text[7:]
        assert len(masked_portion) == len(original)
        assert masked_portion == "*" * len(original)


# ── HASH strategy ─────────────────────────────────────────────────────


class TestHashStrategy:
    """Tests for RedactionStrategy.HASH."""

    def test_hashes_to_sha256_prefix(self, hash_redactor: PIIRedactor) -> None:
        text = "SSN: 123-45-6789"
        detections = [_make_detection("SSN", "123-45-6789", 5, 16)]
        result = hash_redactor.redact(text, detections)
        expected_digest = hashlib.sha256(b"123-45-6789").hexdigest()[:8]
        assert result.redacted_text == f"SSN: [{expected_digest}]"

    def test_hash_custom_length(self) -> None:
        redactor = PIIRedactor(
            default_strategy=RedactionStrategy.HASH, hash_length=12
        )
        text = "SSN: 123-45-6789"
        detections = [_make_detection("SSN", "123-45-6789", 5, 16)]
        result = redactor.redact(text, detections)
        expected_digest = hashlib.sha256(b"123-45-6789").hexdigest()[:12]
        assert result.redacted_text == f"SSN: [{expected_digest}]"

    def test_hash_is_deterministic(self, hash_redactor: PIIRedactor) -> None:
        text = "SSN: 123-45-6789"
        detections = [_make_detection("SSN", "123-45-6789", 5, 16)]
        result1 = hash_redactor.redact(text, detections)
        result2 = hash_redactor.redact(text, detections)
        assert result1.redacted_text == result2.redacted_text


# ── REMOVE strategy ───────────────────────────────────────────────────


class TestRemoveStrategy:
    """Tests for RedactionStrategy.REMOVE."""

    def test_removes_text_entirely(self, remove_redactor: PIIRedactor) -> None:
        text = "SSN: 123-45-6789 end"
        detections = [_make_detection("SSN", "123-45-6789", 5, 16)]
        result = remove_redactor.redact(text, detections)
        assert result.redacted_text == "SSN:  end"

    def test_remove_produces_empty_replacement(
        self, remove_redactor: PIIRedactor
    ) -> None:
        text = "SSN: 123-45-6789"
        detections = [_make_detection("SSN", "123-45-6789", 5, 16)]
        result = remove_redactor.redact(text, detections)
        assert len(result.redactions) == 1
        assert result.redactions[0].replacement == ""


# ── Multiple PII in same text ────────────────────────────────────────


class TestMultiplePII:
    """Tests for redacting multiple PII entities in a single string."""

    def test_multiple_redactions_applied(
        self, replace_redactor: PIIRedactor
    ) -> None:
        text = "SSN 123-45-6789 email user@example.com"
        detections = [
            _make_detection("SSN", "123-45-6789", 4, 15),
            _make_detection("EMAIL_ADDRESS", "user@example.com", 22, 38),
        ]
        result = replace_redactor.redact(text, detections)
        assert "[SSN]" in result.redacted_text
        assert "[EMAIL_ADDRESS]" in result.redacted_text
        # Original PII must not appear
        assert "123-45-6789" not in result.redacted_text
        assert "user@example.com" not in result.redacted_text

    def test_redaction_details_in_document_order(
        self, replace_redactor: PIIRedactor
    ) -> None:
        text = "SSN 123-45-6789 email user@example.com"
        detections = [
            _make_detection("SSN", "123-45-6789", 4, 15),
            _make_detection("EMAIL_ADDRESS", "user@example.com", 22, 38),
        ]
        result = replace_redactor.redact(text, detections)
        assert len(result.redactions) == 2
        # Redaction details should be in document order (start ascending)
        assert result.redactions[0].start < result.redactions[1].start

    def test_three_entities_redacted(self, replace_redactor: PIIRedactor) -> None:
        text = "a]123-45-6789 b]user@test.com c]10.0.0.1"
        detections = [
            _make_detection("SSN", "123-45-6789", 2, 13),
            _make_detection("EMAIL_ADDRESS", "user@test.com", 16, 29),
            _make_detection("IP_ADDRESS", "10.0.0.1", 32, 40),
        ]
        result = replace_redactor.redact(text, detections)
        assert "[SSN]" in result.redacted_text
        assert "[EMAIL_ADDRESS]" in result.redacted_text
        assert "[IP_ADDRESS]" in result.redacted_text


# ── Empty detections ──────────────────────────────────────────────────


class TestEmptyDetections:
    """Tests for handling empty detection lists."""

    def test_no_detections_returns_original_text(
        self, replace_redactor: PIIRedactor
    ) -> None:
        text = "Nothing sensitive here."
        result = replace_redactor.redact(text, [])
        assert result.redacted_text == text
        assert result.original_text == text
        assert result.redactions == []

    def test_empty_text_with_no_detections(
        self, replace_redactor: PIIRedactor
    ) -> None:
        result = replace_redactor.redact("", [])
        assert result.redacted_text == ""
        assert result.redactions == []


# ── Per-entity strategy overrides ─────────────────────────────────────


class TestPerEntityStrategyOverrides:
    """Tests for per-entity-type strategy overrides via entity_strategies."""

    def test_entity_strategy_overrides_default(self) -> None:
        redactor = PIIRedactor(
            default_strategy=RedactionStrategy.REPLACE,
            entity_strategies={"SSN": RedactionStrategy.MASK},
        )
        text = "SSN 123-45-6789 email user@example.com"
        detections = [
            _make_detection("SSN", "123-45-6789", 4, 15),
            _make_detection("EMAIL_ADDRESS", "user@example.com", 22, 38),
        ]
        result = redactor.redact(text, detections)
        # SSN should be masked (11 asterisks), email should be replaced
        assert "***********" in result.redacted_text
        assert "[EMAIL_ADDRESS]" in result.redacted_text

    def test_call_strategy_overrides_both(self) -> None:
        redactor = PIIRedactor(
            default_strategy=RedactionStrategy.REPLACE,
            entity_strategies={"SSN": RedactionStrategy.MASK},
        )
        text = "SSN 123-45-6789 email user@example.com"
        detections = [
            _make_detection("SSN", "123-45-6789", 4, 15),
            _make_detection("EMAIL_ADDRESS", "user@example.com", 22, 38),
        ]
        # Passing strategy= at call level overrides everything
        result = redactor.redact(text, detections, strategy=RedactionStrategy.REMOVE)
        assert "123-45-6789" not in result.redacted_text
        assert "user@example.com" not in result.redacted_text
        # Both should use REMOVE
        for rd in result.redactions:
            assert rd.strategy == RedactionStrategy.REMOVE

    def test_multiple_entity_overrides(self) -> None:
        redactor = PIIRedactor(
            default_strategy=RedactionStrategy.REPLACE,
            entity_strategies={
                "SSN": RedactionStrategy.MASK,
                "EMAIL_ADDRESS": RedactionStrategy.HASH,
            },
        )
        text = "SSN 123-45-6789 email user@example.com"
        detections = [
            _make_detection("SSN", "123-45-6789", 4, 15),
            _make_detection("EMAIL_ADDRESS", "user@example.com", 22, 38),
        ]
        result = redactor.redact(text, detections)
        # SSN should be masked
        assert "***********" in result.redacted_text
        # Email should be hashed
        email_hash = hashlib.sha256(b"user@example.com").hexdigest()[:8]
        assert f"[{email_hash}]" in result.redacted_text


# ── RedactionResult structure ─────────────────────────────────────────


class TestRedactionResultStructure:
    """Tests that verify the structure of RedactionResult."""

    def test_original_text_preserved(self, replace_redactor: PIIRedactor) -> None:
        text = "SSN: 123-45-6789"
        detections = [_make_detection("SSN", "123-45-6789", 5, 16)]
        result = replace_redactor.redact(text, detections)
        assert result.original_text == text

    def test_redaction_detail_fields(self, replace_redactor: PIIRedactor) -> None:
        text = "SSN: 123-45-6789"
        detections = [_make_detection("SSN", "123-45-6789", 5, 16)]
        result = replace_redactor.redact(text, detections)
        detail = result.redactions[0]
        assert detail.entity_type == "SSN"
        assert detail.original == "123-45-6789"
        assert detail.replacement == "[SSN]"
        assert detail.start == 5
        assert detail.end == 16
        assert detail.strategy == RedactionStrategy.REPLACE
