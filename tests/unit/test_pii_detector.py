"""Unit tests for the PII detector module."""

from __future__ import annotations

import pytest

from guardrails.pii.detector import PIIDetector
from guardrails.pii.patterns import (
    ALL_PATTERNS,
    AMEX_PATTERN,
    EMAIL_PATTERN,
    IPV4_PATTERN,
    MASTERCARD_PATTERN,
    SSN_PATTERN,
    VISA_PATTERN,
    luhn_check,
)
from guardrails.types import DetectionResult, Severity


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def detector() -> PIIDetector:
    """Default detector with all patterns enabled."""
    return PIIDetector()


@pytest.fixture
def ssn_only_detector() -> PIIDetector:
    """Detector that only finds SSNs."""
    return PIIDetector(enabled_entity_types={"SSN"})


@pytest.fixture
def high_confidence_detector() -> PIIDetector:
    """Detector that only returns matches with confidence >= 0.90."""
    return PIIDetector(min_confidence=0.90)


# ── SSN detection ─────────────────────────────────────────────────────


class TestSSNDetection:
    """Tests for Social Security Number detection."""

    def test_detects_dashed_ssn(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("My SSN is 123-45-6789")
        ssn_results = [r for r in results if r.entity_type == "SSN"]
        assert len(ssn_results) >= 1
        assert ssn_results[0].text == "123-45-6789"
        assert ssn_results[0].confidence == 0.95
        assert ssn_results[0].severity == Severity.CRITICAL

    def test_rejects_ssn_starting_with_000(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Invalid: 000-12-3456")
        ssn_results = [r for r in results if r.entity_type == "SSN"]
        assert len(ssn_results) == 0

    def test_rejects_ssn_starting_with_666(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Invalid: 666-12-3456")
        ssn_results = [r for r in results if r.entity_type == "SSN"]
        assert len(ssn_results) == 0

    def test_rejects_ssn_starting_with_9xx(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Invalid: 900-12-3456")
        ssn_results = [r for r in results if r.entity_type == "SSN"]
        assert len(ssn_results) == 0

    def test_rejects_ssn_with_00_middle_group(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Invalid: 123-00-6789")
        ssn_results = [r for r in results if r.entity_type == "SSN"]
        assert len(ssn_results) == 0

    def test_rejects_ssn_with_0000_last_group(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Invalid: 123-45-0000")
        ssn_results = [r for r in results if r.entity_type == "SSN"]
        assert len(ssn_results) == 0

    def test_detects_ssn_offset_positions(self, detector: PIIDetector) -> None:
        text = "prefix 123-45-6789 suffix"
        results = detector.detect_sync(text)
        ssn_results = [r for r in results if r.entity_type == "SSN"]
        assert len(ssn_results) >= 1
        match = ssn_results[0]
        assert text[match.start : match.end] == "123-45-6789"


# ── Credit card detection ─────────────────────────────────────────────


class TestCreditCardDetection:
    """Tests for credit card number detection with Luhn validation."""

    def test_detects_visa(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Card: 4111111111111111")
        cc_results = [r for r in results if r.entity_type == "CREDIT_CARD"]
        assert len(cc_results) >= 1
        assert cc_results[0].text == "4111111111111111"

    def test_detects_visa_with_dashes(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Card: 4111-1111-1111-1111")
        cc_results = [r for r in results if r.entity_type == "CREDIT_CARD"]
        assert len(cc_results) >= 1
        assert cc_results[0].text == "4111-1111-1111-1111"

    def test_detects_visa_with_spaces(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Card: 4111 1111 1111 1111")
        cc_results = [r for r in results if r.entity_type == "CREDIT_CARD"]
        assert len(cc_results) >= 1
        assert cc_results[0].text == "4111 1111 1111 1111"

    def test_detects_mastercard(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Card: 5500000000000004")
        cc_results = [r for r in results if r.entity_type == "CREDIT_CARD"]
        assert len(cc_results) >= 1
        assert cc_results[0].text == "5500000000000004"

    def test_detects_amex(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Card: 378282246310005")
        cc_results = [r for r in results if r.entity_type == "CREDIT_CARD"]
        assert len(cc_results) >= 1
        assert cc_results[0].text == "378282246310005"

    def test_detects_discover(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Card: 6011111111111117")
        cc_results = [r for r in results if r.entity_type == "CREDIT_CARD"]
        assert len(cc_results) >= 1
        assert cc_results[0].text == "6011111111111117"

    def test_rejects_invalid_luhn(self, detector: PIIDetector) -> None:
        # 4111111111111112 fails Luhn
        results = detector.detect_sync("Card: 4111111111111112")
        cc_results = [r for r in results if r.entity_type == "CREDIT_CARD"]
        assert len(cc_results) == 0

    def test_luhn_check_valid(self) -> None:
        assert luhn_check("4111111111111111") is True
        assert luhn_check("5500000000000004") is True
        assert luhn_check("378282246310005") is True
        assert luhn_check("6011111111111117") is True

    def test_luhn_check_invalid(self) -> None:
        assert luhn_check("4111111111111112") is False
        assert luhn_check("1234567890123456") is False

    def test_luhn_check_strips_non_digits(self) -> None:
        assert luhn_check("4111-1111-1111-1111") is True

    def test_luhn_check_too_short(self) -> None:
        assert luhn_check("1") is False


# ── Email detection ───────────────────────────────────────────────────


class TestEmailDetection:
    """Tests for email address detection."""

    def test_detects_simple_email(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Contact: user@example.com")
        email_results = [r for r in results if r.entity_type == "EMAIL_ADDRESS"]
        assert len(email_results) == 1
        assert email_results[0].text == "user@example.com"

    def test_detects_email_with_plus_tag(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Email: first.last+tag@sub.domain.org")
        email_results = [r for r in results if r.entity_type == "EMAIL_ADDRESS"]
        assert len(email_results) == 1
        assert email_results[0].text == "first.last+tag@sub.domain.org"

    def test_detects_email_with_dots_and_hyphens(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Send to john.doe@my-company.co.uk please")
        email_results = [r for r in results if r.entity_type == "EMAIL_ADDRESS"]
        assert len(email_results) == 1
        assert email_results[0].text == "john.doe@my-company.co.uk"

    def test_email_confidence(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("hello@world.com")
        email_results = [r for r in results if r.entity_type == "EMAIL_ADDRESS"]
        assert len(email_results) == 1
        assert email_results[0].confidence == 0.95


# ── Phone number detection ────────────────────────────────────────────


class TestPhoneDetection:
    """Tests for US phone number detection."""

    def test_detects_dashed_phone(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Call 555-123-4567")
        phone_results = [r for r in results if r.entity_type == "PHONE_NUMBER"]
        assert len(phone_results) >= 1
        assert "555" in phone_results[0].text
        assert "4567" in phone_results[0].text

    def test_detects_parenthesized_area_code(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Call (555) 123-4567")
        phone_results = [r for r in results if r.entity_type == "PHONE_NUMBER"]
        assert len(phone_results) >= 1

    def test_detects_phone_with_country_code(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Call +1 555 123 4567")
        phone_results = [r for r in results if r.entity_type == "PHONE_NUMBER"]
        assert len(phone_results) >= 1

    def test_detects_phone_with_extension(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Call 555-123-4567 ext. 890")
        phone_results = [r for r in results if r.entity_type == "PHONE_NUMBER"]
        assert len(phone_results) >= 1


# ── IPv4 detection ────────────────────────────────────────────────────


class TestIPv4Detection:
    """Tests for IPv4 address detection."""

    def test_detects_private_ip(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Server at 192.168.1.1")
        ip_results = [r for r in results if r.entity_type == "IP_ADDRESS"]
        assert len(ip_results) >= 1
        assert ip_results[0].text == "192.168.1.1"

    def test_detects_loopback(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("Localhost: 127.0.0.1")
        ip_results = [r for r in results if r.entity_type == "IP_ADDRESS"]
        assert len(ip_results) >= 1
        assert ip_results[0].text == "127.0.0.1"

    def test_rejects_ip_with_octet_above_255(self, detector: PIIDetector) -> None:
        # The regex itself limits octets to 0-255, so 999.999.999.999 won't match.
        results = detector.detect_sync("Bad IP: 999.999.999.999")
        ip_results = [r for r in results if r.entity_type == "IP_ADDRESS"]
        assert len(ip_results) == 0

    def test_ip_severity_is_medium(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("IP: 10.0.0.1")
        ip_results = [r for r in results if r.entity_type == "IP_ADDRESS"]
        assert len(ip_results) >= 1
        assert ip_results[0].severity == Severity.MEDIUM


# ── Entity type filtering ─────────────────────────────────────────────


class TestEntityTypeFiltering:
    """Tests for enabled_entity_types and disabled_entity_types filtering."""

    def test_enabled_entity_types_limits_detection(self) -> None:
        detector = PIIDetector(enabled_entity_types={"SSN"})
        text = "SSN 123-45-6789, email user@example.com, IP 10.0.0.1"
        results = detector.detect_sync(text)
        entity_types = {r.entity_type for r in results}
        assert "SSN" in entity_types
        assert "EMAIL_ADDRESS" not in entity_types
        assert "IP_ADDRESS" not in entity_types

    def test_disabled_entity_types_excludes(self) -> None:
        detector = PIIDetector(disabled_entity_types={"EMAIL_ADDRESS", "IP_ADDRESS"})
        text = "SSN 123-45-6789, email user@example.com, IP 10.0.0.1"
        results = detector.detect_sync(text)
        entity_types = {r.entity_type for r in results}
        assert "SSN" in entity_types
        assert "EMAIL_ADDRESS" not in entity_types
        assert "IP_ADDRESS" not in entity_types

    def test_enable_entity_type_at_runtime(self) -> None:
        detector = PIIDetector(enabled_entity_types={"SSN"})
        text = "SSN 123-45-6789, email user@example.com"

        # Initially only SSN
        results = detector.detect_sync(text)
        assert all(r.entity_type == "SSN" for r in results)

        # Enable email detection
        detector.enable_entity_type("EMAIL_ADDRESS")
        results = detector.detect_sync(text)
        entity_types = {r.entity_type for r in results}
        assert "EMAIL_ADDRESS" in entity_types

    def test_disable_entity_type_at_runtime(self) -> None:
        detector = PIIDetector()
        text = "SSN 123-45-6789, email user@example.com"

        # Disable SSN
        detector.disable_entity_type("SSN")
        results = detector.detect_sync(text)
        entity_types = {r.entity_type for r in results}
        assert "SSN" not in entity_types
        assert "EMAIL_ADDRESS" in entity_types


# ── min_confidence filtering ──────────────────────────────────────────


class TestMinConfidenceFiltering:
    """Tests for min_confidence threshold filtering."""

    def test_high_confidence_excludes_low_confidence_patterns(self) -> None:
        # SSN_PATTERN_NODASH has confidence 0.70; SSN_PATTERN has 0.95.
        # With min_confidence=0.90, only the dashed pattern should be active.
        detector = PIIDetector(
            enabled_entity_types={"SSN"},
            min_confidence=0.90,
        )
        # Dashed SSN should still match (0.95 >= 0.90)
        results = detector.detect_sync("SSN: 123-45-6789")
        assert len(results) >= 1

    def test_very_high_threshold_filters_everything(self) -> None:
        detector = PIIDetector(min_confidence=0.99)
        results = detector.detect_sync(
            "SSN 123-45-6789, card 4111111111111111, email user@example.com"
        )
        # No pattern has confidence >= 0.99
        assert len(results) == 0


# ── Async detection ───────────────────────────────────────────────────


class TestAsyncDetection:
    """Tests for the async detect method."""

    @pytest.mark.asyncio
    async def test_async_detect_returns_same_as_sync(self) -> None:
        detector = PIIDetector()
        text = "My SSN is 123-45-6789 and email is user@example.com"
        sync_results = detector.detect_sync(text)
        async_results = await detector.detect(text)
        assert len(async_results) == len(sync_results)
        for s, a in zip(sync_results, async_results):
            assert s.entity_type == a.entity_type
            assert s.text == a.text
            assert s.start == a.start
            assert s.end == a.end

    @pytest.mark.asyncio
    async def test_async_detect_empty_string(self) -> None:
        detector = PIIDetector()
        results = await detector.detect("")
        assert results == []


# ── Empty and edge cases ──────────────────────────────────────────────


class TestEdgeCases:
    """Tests for empty strings and other edge cases."""

    def test_empty_string_returns_no_results(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("")
        assert results == []

    def test_no_pii_returns_no_results(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("This is a totally clean sentence.")
        assert results == []

    def test_results_sorted_by_start_offset(self, detector: PIIDetector) -> None:
        text = "Email user@example.com and SSN 123-45-6789"
        results = detector.detect_sync(text)
        starts = [r.start for r in results]
        assert starts == sorted(starts)

    def test_metadata_includes_pattern_name(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("SSN: 123-45-6789")
        ssn_results = [r for r in results if r.entity_type == "SSN"]
        assert len(ssn_results) >= 1
        assert "pattern_name" in ssn_results[0].metadata
        assert ssn_results[0].metadata["pattern_name"] == "ssn_dashed"

    def test_detector_field_is_regex(self, detector: PIIDetector) -> None:
        results = detector.detect_sync("user@example.com")
        assert all(r.detector == "regex" for r in results)


# ── Overlap deduplication ─────────────────────────────────────────────


class TestOverlapDeduplication:
    """Tests for deduplication of overlapping detections."""

    def test_deduplicate_keeps_higher_confidence(self) -> None:
        results = [
            DetectionResult(
                entity_type="A",
                text="overlap",
                start=0,
                end=10,
                confidence=0.80,
                detector="regex",
            ),
            DetectionResult(
                entity_type="B",
                text="overlap",
                start=0,
                end=10,
                confidence=0.95,
                detector="regex",
            ),
        ]
        deduped = PIIDetector._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.95

    def test_deduplicate_non_overlapping_kept(self) -> None:
        results = [
            DetectionResult(
                entity_type="A",
                text="first",
                start=0,
                end=5,
                confidence=0.90,
                detector="regex",
            ),
            DetectionResult(
                entity_type="B",
                text="second",
                start=10,
                end=16,
                confidence=0.90,
                detector="regex",
            ),
        ]
        deduped = PIIDetector._deduplicate(results)
        assert len(deduped) == 2

    def test_deduplicate_empty_list(self) -> None:
        assert PIIDetector._deduplicate([]) == []

    def test_deduplicate_partial_overlap_keeps_best(self) -> None:
        results = [
            DetectionResult(
                entity_type="A",
                text="abcde",
                start=0,
                end=5,
                confidence=0.70,
                detector="regex",
            ),
            DetectionResult(
                entity_type="B",
                text="cdefgh",
                start=2,
                end=8,
                confidence=0.90,
                detector="regex",
            ),
        ]
        deduped = PIIDetector._deduplicate(results)
        # They overlap at positions 2-5, so only the higher-confidence one survives
        assert len(deduped) == 1
        assert deduped[0].entity_type == "B"


# ── Multiple PII in same text ────────────────────────────────────────


class TestMultiplePII:
    """Tests for detecting multiple PII entities in one string."""

    def test_detects_ssn_and_email_and_ip(self, detector: PIIDetector) -> None:
        text = "SSN 123-45-6789, email admin@corp.com, server 10.0.0.1"
        results = detector.detect_sync(text)
        entity_types = {r.entity_type for r in results}
        assert "SSN" in entity_types
        assert "EMAIL_ADDRESS" in entity_types
        assert "IP_ADDRESS" in entity_types

    def test_multiple_emails_detected(self, detector: PIIDetector) -> None:
        text = "From alice@example.com to bob@example.com"
        results = detector.detect_sync(text)
        email_results = [r for r in results if r.entity_type == "EMAIL_ADDRESS"]
        assert len(email_results) == 2
        emails = {r.text for r in email_results}
        assert emails == {"alice@example.com", "bob@example.com"}
