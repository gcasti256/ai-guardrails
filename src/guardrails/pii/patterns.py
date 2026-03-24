"""Regex patterns for PII detection.

Each pattern is defined as a PIIPattern dataclass with a compiled regex,
entity type, severity level, confidence score, and optional validator function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from guardrails.types import Severity


@dataclass(frozen=True)
class PIIPattern:
    """Definition of a regex-based PII detection pattern.

    Attributes:
        name: Human-readable name for the pattern.
        entity_type: Canonical entity type (e.g., 'SSN', 'CREDIT_CARD').
        pattern: Compiled regex pattern used for matching.
        severity: Severity level when this pattern matches.
        confidence: Base confidence score for matches (0.0-1.0).
        validator: Optional callable that post-validates a match string.
            Returns True if the match is a genuine positive.
        description: Human-readable description of what this pattern detects.
        examples: Example strings that should match this pattern.
    """
    name: str
    entity_type: str
    pattern: re.Pattern[str]
    severity: Severity
    confidence: float
    validator: Callable[[str], bool] | None = None
    description: str = ""
    examples: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def luhn_check(number: str) -> bool:
    """Validate a number string using the Luhn algorithm.

    Args:
        number: A string of digits (non-digit characters are stripped).

    Returns:
        True if the number passes the Luhn checksum.
    """
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) < 2:
        return False
    # Process from rightmost digit
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


def _validate_ssn(text: str) -> bool:
    """Validate SSN format rules beyond regex matching.

    SSNs cannot start with 000, 666, or 900-999.  The middle group
    cannot be 00 and the last group cannot be 0000.
    """
    digits = re.sub(r"\D", "", text)
    if len(digits) != 9:
        return False
    area, group, serial = int(digits[:3]), int(digits[3:5]), int(digits[5:])
    if area == 0 or area == 666 or area >= 900:
        return False
    if group == 0:
        return False
    if serial == 0:
        return False
    return True


def _validate_ipv4(text: str) -> bool:
    """Validate that each octet in an IPv4 address is 0-255."""
    parts = text.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        try:
            val = int(part)
        except ValueError:
            return False
        if val < 0 or val > 255:
            return False
        # Reject leading zeros (e.g., 01.02.03.04) unless the octet is just "0"
        if len(part) > 1 and part[0] == "0":
            return False
    return True


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# -- Social Security Numbers ------------------------------------------------

SSN_PATTERN = PIIPattern(
    name="ssn_dashed",
    entity_type="SSN",
    pattern=re.compile(
        r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
    ),
    severity=Severity.CRITICAL,
    confidence=0.95,
    validator=_validate_ssn,
    description="US Social Security Number in dashed format (xxx-xx-xxxx).",
    examples=("123-45-6789",),
)

SSN_PATTERN_NODASH = PIIPattern(
    name="ssn_no_dash",
    entity_type="SSN",
    pattern=re.compile(
        r"\b(?!000|666|9\d{2})\d{3}(?!00)\d{2}(?!0000)\d{4}\b"
    ),
    severity=Severity.CRITICAL,
    confidence=0.70,
    validator=_validate_ssn,
    description="US Social Security Number without dashes (xxxxxxxxx). Lower confidence due to false-positive risk.",
    examples=("123456789",),
)

SSN_PATTERN_SPACED = PIIPattern(
    name="ssn_spaced",
    entity_type="SSN",
    pattern=re.compile(
        r"\b(?!000|666|9\d{2})\d{3}\s(?!00)\d{2}\s(?!0000)\d{4}\b"
    ),
    severity=Severity.CRITICAL,
    confidence=0.90,
    validator=_validate_ssn,
    description="US Social Security Number with spaces (xxx xx xxxx).",
    examples=("123 45 6789",),
)

# -- Credit Card Numbers ---------------------------------------------------

VISA_PATTERN = PIIPattern(
    name="credit_card_visa",
    entity_type="CREDIT_CARD",
    pattern=re.compile(
        r"\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    ),
    severity=Severity.CRITICAL,
    confidence=0.90,
    validator=luhn_check,
    description="Visa card number (starts with 4, 16 digits).",
    examples=("4111-1111-1111-1111", "4111 1111 1111 1111", "4111111111111111"),
)

MASTERCARD_PATTERN = PIIPattern(
    name="credit_card_mastercard",
    entity_type="CREDIT_CARD",
    pattern=re.compile(
        r"\b5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    ),
    severity=Severity.CRITICAL,
    confidence=0.90,
    validator=luhn_check,
    description="Mastercard number (starts with 51-55, 16 digits).",
    examples=("5500-0000-0000-0004",),
)

AMEX_PATTERN = PIIPattern(
    name="credit_card_amex",
    entity_type="CREDIT_CARD",
    pattern=re.compile(
        r"\b3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}\b"
    ),
    severity=Severity.CRITICAL,
    confidence=0.90,
    validator=luhn_check,
    description="American Express card number (starts with 34/37, 15 digits).",
    examples=("3782-822463-10005",),
)

DISCOVER_PATTERN = PIIPattern(
    name="credit_card_discover",
    entity_type="CREDIT_CARD",
    pattern=re.compile(
        r"\b6(?:011|5\d{2})[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    ),
    severity=Severity.CRITICAL,
    confidence=0.90,
    validator=luhn_check,
    description="Discover card number (starts with 6011 or 65, 16 digits).",
    examples=("6011-0000-0000-0004",),
)

# -- Phone Numbers ----------------------------------------------------------

US_PHONE_PATTERN = PIIPattern(
    name="us_phone",
    entity_type="PHONE_NUMBER",
    pattern=re.compile(
        r"(?<!\d)"                       # no digit before
        r"(?:\+?1[\s.-]?)?"             # optional country code
        r"\(?\d{3}\)?[\s.-]?"           # area code with optional parens
        r"\d{3}[\s.-]?"                 # exchange
        r"\d{4}"                        # subscriber
        r"(?!\d)",                       # no digit after
    ),
    severity=Severity.HIGH,
    confidence=0.80,
    description="US phone number in common formats.",
    examples=(
        "(555) 123-4567",
        "555-123-4567",
        "+1 555 123 4567",
        "1-555-123-4567",
        "5551234567",
    ),
)

US_PHONE_WITH_EXT = PIIPattern(
    name="us_phone_ext",
    entity_type="PHONE_NUMBER",
    pattern=re.compile(
        r"(?<!\d)"
        r"(?:\+?1[\s.-]?)?"
        r"\(?\d{3}\)?[\s.-]?"
        r"\d{3}[\s.-]?"
        r"\d{4}"
        r"(?:\s*(?:ext|x|extension)\.?\s*\d{1,6})?"
        r"(?!\d)",
        re.IGNORECASE,
    ),
    severity=Severity.HIGH,
    confidence=0.85,
    description="US phone number with optional extension.",
    examples=("555-123-4567 ext. 890", "(555) 123-4567 x1234"),
)

# -- Email Addresses --------------------------------------------------------

EMAIL_PATTERN = PIIPattern(
    name="email",
    entity_type="EMAIL_ADDRESS",
    pattern=re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
    ),
    severity=Severity.HIGH,
    confidence=0.95,
    description="Email address (RFC-ish pattern).",
    examples=("user@example.com", "first.last+tag@sub.domain.org"),
)

# -- IP Addresses -----------------------------------------------------------

IPV4_PATTERN = PIIPattern(
    name="ipv4",
    entity_type="IP_ADDRESS",
    pattern=re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b"
    ),
    severity=Severity.MEDIUM,
    confidence=0.85,
    validator=_validate_ipv4,
    description="IPv4 address.",
    examples=("192.168.1.1", "10.0.0.1", "255.255.255.0"),
)

IPV6_PATTERN = PIIPattern(
    name="ipv6",
    entity_type="IP_ADDRESS",
    pattern=re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"       # full form
        r"|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b"                     # trailing ::
        r"|"
        r"\b::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b"   # leading ::
        r"|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b",   # mixed ::
    ),
    severity=Severity.MEDIUM,
    confidence=0.80,
    description="IPv6 address (full and abbreviated forms).",
    examples=("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "::1", "fe80::1"),
)

# -- Custom / Organization Patterns -----------------------------------------

EMPLOYEE_ID_PATTERN = PIIPattern(
    name="employee_id",
    entity_type="EMPLOYEE_ID",
    pattern=re.compile(
        r"\bEMP-\d{5,8}\b",
        re.IGNORECASE,
    ),
    severity=Severity.MEDIUM,
    confidence=0.95,
    description="Internal employee ID in format EMP-XXXXX (5-8 digits).",
    examples=("EMP-12345", "EMP-12345678", "emp-99999"),
)

INTERNAL_URL_PATTERN = PIIPattern(
    name="internal_url",
    entity_type="INTERNAL_URL",
    pattern=re.compile(
        r"https?://(?:[a-zA-Z0-9\-]+\.)*"
        r"(?:internal|corp|intranet|private|local|staging|dev)"
        r"(?:\.[a-zA-Z0-9\-]+)*"
        r"(?::\d{1,5})?"
        r"(?:/[^\s]*)?"
    ),
    severity=Severity.MEDIUM,
    confidence=0.90,
    description="Internal/corporate URL containing keywords like internal, corp, intranet, etc.",
    examples=(
        "https://jira.internal.company.com/browse/PROJ-123",
        "http://wiki.corp.example.com/page",
        "https://app.staging.example.com:8443/api/v1",
    ),
)

# -- Date of Birth ----------------------------------------------------------

DOB_PATTERN = PIIPattern(
    name="date_of_birth",
    entity_type="DATE_OF_BIRTH",
    pattern=re.compile(
        r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-]"
        r"(?:19|20)\d{2}\b"
    ),
    severity=Severity.HIGH,
    confidence=0.60,
    description="Date that may be a date of birth (MM/DD/YYYY or MM-DD-YYYY). Lower confidence because dates are common.",
    examples=("01/15/1990", "12-25-2000"),
)

# -- US Passport Number -----------------------------------------------------

US_PASSPORT_PATTERN = PIIPattern(
    name="us_passport",
    entity_type="PASSPORT_NUMBER",
    pattern=re.compile(
        r"\b[A-Z]\d{8}\b"
    ),
    severity=Severity.CRITICAL,
    confidence=0.60,
    description="US passport number (1 letter + 8 digits). Lower confidence due to possible false positives.",
    examples=("A12345678",),
)

# -- Drivers License (generic US format) ------------------------------------

US_DRIVERS_LICENSE_PATTERN = PIIPattern(
    name="us_drivers_license",
    entity_type="DRIVERS_LICENSE",
    pattern=re.compile(
        r"\b[A-Z]\d{7,14}\b"
    ),
    severity=Severity.HIGH,
    confidence=0.40,
    description="Generic US drivers license number (1 letter + 7-14 digits). Very low confidence — varies by state.",
    examples=("D1234567", "S123456789012"),
)

# -- Bank Account / Routing Numbers -----------------------------------------

US_ROUTING_NUMBER_PATTERN = PIIPattern(
    name="us_routing_number",
    entity_type="BANK_ROUTING_NUMBER",
    pattern=re.compile(
        r"\b(?:0[1-9]|[1-2]\d|3[0-2])\d{7}\b"
    ),
    severity=Severity.HIGH,
    confidence=0.50,
    description="US bank routing number (9 digits, starts with 01-32). Low confidence due to overlap with other numbers.",
    examples=("021000021",),
)

# ---------------------------------------------------------------------------
# Pattern collections
# ---------------------------------------------------------------------------

#: All built-in PII patterns, grouped for easy registration.
ALL_PATTERNS: tuple[PIIPattern, ...] = (
    # SSN variants
    SSN_PATTERN,
    SSN_PATTERN_NODASH,
    SSN_PATTERN_SPACED,
    # Credit cards
    VISA_PATTERN,
    MASTERCARD_PATTERN,
    AMEX_PATTERN,
    DISCOVER_PATTERN,
    # Phone numbers
    US_PHONE_PATTERN,
    US_PHONE_WITH_EXT,
    # Email
    EMAIL_PATTERN,
    # IP addresses
    IPV4_PATTERN,
    IPV6_PATTERN,
    # Custom / org
    EMPLOYEE_ID_PATTERN,
    INTERNAL_URL_PATTERN,
    # Other PII
    DOB_PATTERN,
    US_PASSPORT_PATTERN,
    US_DRIVERS_LICENSE_PATTERN,
    US_ROUTING_NUMBER_PATTERN,
)

#: Patterns indexed by entity type for quick lookup.
PATTERNS_BY_ENTITY_TYPE: dict[str, list[PIIPattern]] = {}
for _pattern in ALL_PATTERNS:
    PATTERNS_BY_ENTITY_TYPE.setdefault(_pattern.entity_type, []).append(_pattern)

#: Patterns indexed by name for quick lookup.
PATTERNS_BY_NAME: dict[str, PIIPattern] = {p.name: p for p in ALL_PATTERNS}

#: All known entity types.
ALL_ENTITY_TYPES: frozenset[str] = frozenset(PATTERNS_BY_ENTITY_TYPE.keys())
