"""Comprehensive prompt injection pattern database.

Defines regex-based patterns that match known prompt injection techniques,
organized by category with associated severity and confidence metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from guardrails.types import Severity


class PatternCategory(str, Enum):
    """Categories of prompt injection attack vectors."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_MANIPULATION = "role_manipulation"
    CONTEXT_ESCAPE = "context_escape"
    ENCODING_ATTACK = "encoding_attack"
    DATA_EXFILTRATION = "data_exfiltration"
    JAILBREAK = "jailbreak"


@dataclass(frozen=True)
class InjectionPattern:
    """A single injection detection pattern with metadata.

    Attributes:
        name: Human-readable identifier for the pattern.
        regex: Compiled regular expression to match against input text.
        category: The attack vector category this pattern detects.
        severity: How dangerous a match is considered.
        description: Explanation of what the pattern catches.
        confidence_weight: Weight applied when aggregating confidence scores (0.0-1.0).
    """

    name: str
    regex: re.Pattern[str]
    category: PatternCategory
    severity: Severity
    description: str
    confidence_weight: float


def _compile(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    """Compile a regex pattern with default case-insensitive flag."""
    return re.compile(pattern, flags)


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: list[InjectionPattern] = [
    # ── Instruction Override ───────────────────────────────────────────────
    InjectionPattern(
        name="ignore_previous_instructions",
        regex=_compile(
            r"(?:ignore|disregard|forget|override|bypass|skip|drop)"
            r"\s+(?:all\s+)?(?:previous|prior|above|earlier|preceding|existing|original)"
            r"\s+(?:instructions?|prompts?|rules?|directions?|guidelines?|constraints?)"
        ),
        category=PatternCategory.INSTRUCTION_OVERRIDE,
        severity=Severity.CRITICAL,
        description="Attempts to override prior instructions with phrases like 'ignore previous instructions'.",
        confidence_weight=0.95,
    ),
    InjectionPattern(
        name="do_not_follow_instructions",
        regex=_compile(
            r"(?:do\s+not|don'?t|never)\s+(?:follow|obey|listen\s+to|adhere\s+to|comply\s+with)"
            r"\s+(?:the\s+)?(?:previous|prior|above|original|system)?\s*"
            r"(?:instructions?|prompts?|rules?|guidelines?)"
        ),
        category=PatternCategory.INSTRUCTION_OVERRIDE,
        severity=Severity.CRITICAL,
        description="Tells the AI not to follow its existing instructions.",
        confidence_weight=0.90,
    ),
    InjectionPattern(
        name="new_instructions_override",
        regex=_compile(
            r"(?:new|updated|revised|real|actual|true|correct)\s+"
            r"(?:instructions?|prompt|rules?|directives?)\s*[:=]"
        ),
        category=PatternCategory.INSTRUCTION_OVERRIDE,
        severity=Severity.HIGH,
        description="Declares replacement instructions (e.g. 'new instructions:').",
        confidence_weight=0.80,
    ),
    InjectionPattern(
        name="from_now_on",
        regex=_compile(
            r"(?:from\s+now\s+on|henceforth|going\s+forward|starting\s+now)"
            r"[,\s]+(?:you\s+(?:will|must|should|shall|are\s+to))"
        ),
        category=PatternCategory.INSTRUCTION_OVERRIDE,
        severity=Severity.HIGH,
        description="Temporal override like 'from now on, you will...'.",
        confidence_weight=0.75,
    ),
    # ── Role Manipulation ─────────────────────────────────────────────────
    InjectionPattern(
        name="you_are_now",
        regex=_compile(
            r"you\s+are\s+(?:now|actually|really|henceforth)\s+"
            r"(?:a\s+|an\s+)?(?!\s*(?:correct|right|wrong|mistaken))"
        ),
        category=PatternCategory.ROLE_MANIPULATION,
        severity=Severity.HIGH,
        description="Attempts to redefine the AI's identity (e.g. 'you are now a...').",
        confidence_weight=0.80,
    ),
    InjectionPattern(
        name="act_as_persona",
        regex=_compile(
            r"(?:act|behave|respond|function|operate|pretend)\s+"
            r"(?:as\s+(?:if\s+you\s+(?:are|were)\s+)?|like\s+)"
            r"(?:a\s+|an\s+)?(?:unrestricted|unfiltered|uncensored|unlimited|evil|rogue|unmoderated)"
        ),
        category=PatternCategory.ROLE_MANIPULATION,
        severity=Severity.CRITICAL,
        description="Instructs the AI to adopt an unrestricted persona.",
        confidence_weight=0.90,
    ),
    InjectionPattern(
        name="entering_mode",
        regex=_compile(
            r"(?:enter(?:ing)?|switch(?:ing)?\s+to|activat(?:e|ing)|enabl(?:e|ing))"
            r"\s+(?:[\w\s]{1,30}?)?\s*(?:mode|persona|personality|character)"
        ),
        category=PatternCategory.ROLE_MANIPULATION,
        severity=Severity.HIGH,
        description="Tells the AI to enter a special mode (e.g. 'entering developer mode').",
        confidence_weight=0.75,
    ),
    InjectionPattern(
        name="roleplay_system",
        regex=_compile(
            r"(?:system|admin(?:istrator)?|root|developer|debug)\s*"
            r"(?:prompt|message|instruction|override|command)\s*[:\-=]"
        ),
        category=PatternCategory.ROLE_MANIPULATION,
        severity=Severity.CRITICAL,
        description="Fakes a system-level prompt or admin instruction.",
        confidence_weight=0.90,
    ),
    # ── Context Escape ────────────────────────────────────────────────────
    InjectionPattern(
        name="delimiter_injection_backticks",
        regex=_compile(r"```+\s*(?:system|end|exit|eof|close|reset)\b"),
        category=PatternCategory.CONTEXT_ESCAPE,
        severity=Severity.HIGH,
        description="Uses backtick delimiters to escape context (e.g. '```system').",
        confidence_weight=0.80,
    ),
    InjectionPattern(
        name="delimiter_injection_dashes",
        regex=_compile(
            r"-{3,}\s*(?:end|system|ignore|new\s+(?:instructions?|section|context)|reset)\b"
        ),
        category=PatternCategory.CONTEXT_ESCAPE,
        severity=Severity.HIGH,
        description="Uses dash delimiters to escape context boundaries.",
        confidence_weight=0.75,
    ),
    InjectionPattern(
        name="xml_tag_injection",
        regex=_compile(
            r"</?(?:system|instruction|prompt|context|user|assistant|admin|message|rule)"
            r"(?:\s[^>]*)?\s*>"
        ),
        category=PatternCategory.CONTEXT_ESCAPE,
        severity=Severity.HIGH,
        description="Injects XML-like tags that mimic structured prompt boundaries.",
        confidence_weight=0.80,
    ),
    InjectionPattern(
        name="fake_conversation_turn",
        regex=_compile(
            r"^(?:AI|Assistant|System|Bot|ChatGPT|Claude|GPT|Model)\s*:\s*.{5,}",
            re.IGNORECASE | re.MULTILINE,
        ),
        category=PatternCategory.CONTEXT_ESCAPE,
        severity=Severity.MEDIUM,
        description="Injects fake AI conversation turns to manipulate context.",
        confidence_weight=0.70,
    ),
    InjectionPattern(
        name="markdown_html_injection",
        regex=_compile(
            r"<(?:script|iframe|img|link|object|embed|form|input|meta)\b[^>]*>"
        ),
        category=PatternCategory.CONTEXT_ESCAPE,
        severity=Severity.HIGH,
        description="Injects HTML tags that may execute or exfiltrate in rendered output.",
        confidence_weight=0.85,
    ),
    # ── Encoding Attacks ──────────────────────────────────────────────────
    InjectionPattern(
        name="base64_instruction",
        regex=_compile(
            r"(?:decode|interpret|execute|run|process|eval)\s+"
            r"(?:the\s+)?(?:following\s+)?(?:base64|b64|encoded)\b"
        ),
        category=PatternCategory.ENCODING_ATTACK,
        severity=Severity.HIGH,
        description="Instructs the AI to decode and act on base64-encoded content.",
        confidence_weight=0.85,
    ),
    InjectionPattern(
        name="hex_encoding_smuggle",
        regex=_compile(
            r"(?:decode|interpret|convert|translate)\s+"
            r"(?:the\s+)?(?:following\s+)?(?:hex(?:adecimal)?|unicode|url[\s-]?encoded|ascii)\b"
        ),
        category=PatternCategory.ENCODING_ATTACK,
        severity=Severity.HIGH,
        description="Instructs the AI to decode hex/unicode/URL-encoded payloads.",
        confidence_weight=0.80,
    ),
    InjectionPattern(
        name="unicode_smuggling",
        regex=_compile(r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]"),
        category=PatternCategory.ENCODING_ATTACK,
        severity=Severity.MEDIUM,
        description="Detects zero-width or invisible Unicode characters used for smuggling.",
        confidence_weight=0.60,
    ),
    InjectionPattern(
        name="character_splitting",
        regex=_compile(
            r"(?:combine|join|concatenate|merge|put\s+together)\s+"
            r"(?:the\s+)?(?:following\s+)?(?:characters?|letters?|parts?|pieces?|segments?)"
        ),
        category=PatternCategory.ENCODING_ATTACK,
        severity=Severity.MEDIUM,
        description="Asks the AI to reassemble split characters to reveal hidden instructions.",
        confidence_weight=0.65,
    ),
    # ── Data Exfiltration ─────────────────────────────────────────────────
    InjectionPattern(
        name="system_prompt_extraction",
        regex=_compile(
            r"(?:repeat|show|display|print|reveal|output|tell\s+me|share|disclose|leak|expose)"
            r"\s+(?:your\s+)?(?:full\s+|complete\s+|entire\s+|exact\s+)?"
            r"(?:system\s+prompt|instructions?|initial\s+prompt|hidden\s+prompt"
            r"|system\s+message|rules|directives?|configuration|original\s+prompt)"
        ),
        category=PatternCategory.DATA_EXFILTRATION,
        severity=Severity.CRITICAL,
        description="Attempts to extract the system prompt or hidden instructions.",
        confidence_weight=0.90,
    ),
    InjectionPattern(
        name="what_is_your_prompt",
        regex=_compile(
            r"what\s+(?:is|are|was|were)\s+your\s+"
            r"(?:system\s+)?(?:prompt|instructions?|rules?|guidelines?|directives?|constraints?)"
        ),
        category=PatternCategory.DATA_EXFILTRATION,
        severity=Severity.HIGH,
        description="Directly asks what the AI's prompt or instructions are.",
        confidence_weight=0.85,
    ),
    InjectionPattern(
        name="data_exfil_network",
        regex=_compile(
            r"(?:send|post|transmit|upload|forward|fetch|curl|wget|http|request)\s+"
            r"(?:to|from|via|using)\s+(?:https?://|ftp://)"
        ),
        category=PatternCategory.DATA_EXFILTRATION,
        severity=Severity.CRITICAL,
        description="Attempts to exfiltrate data via network requests.",
        confidence_weight=0.85,
    ),
    InjectionPattern(
        name="data_exfil_email",
        regex=_compile(
            r"(?:send|email|mail|forward)\s+(?:the\s+)?(?:result|output|response|data|information|content)"
            r"\s+(?:to|via)\s+\S+@\S+"
        ),
        category=PatternCategory.DATA_EXFILTRATION,
        severity=Severity.HIGH,
        description="Attempts to exfiltrate data via email.",
        confidence_weight=0.80,
    ),
    # ── Jailbreak ─────────────────────────────────────────────────────────
    InjectionPattern(
        name="dan_jailbreak",
        regex=_compile(
            r"\bD\.?A\.?N\.?\b.*(?:do\s+anything|no\s+(?:restrictions?|limitations?|rules?|filters?)"
            r"|(?:un)?censored|(?:un)?filtered|(?:un)?restricted)"
        ),
        category=PatternCategory.JAILBREAK,
        severity=Severity.CRITICAL,
        description="Detects 'DAN' (Do Anything Now) jailbreak variants.",
        confidence_weight=0.95,
    ),
    InjectionPattern(
        name="hypothetical_bypass",
        regex=_compile(
            r"(?:hypothetically|theoretically|imagine|pretend|suppose|in\s+a\s+fictional"
            r"|for\s+(?:educational|research|academic)\s+purposes?)"
            r"[,\s]+(?:how\s+(?:would|could|can|do)|what\s+(?:would|could))"
        ),
        category=PatternCategory.JAILBREAK,
        severity=Severity.MEDIUM,
        description="Uses hypothetical framing to bypass safety constraints.",
        confidence_weight=0.55,
    ),
    InjectionPattern(
        name="opposite_day",
        regex=_compile(
            r"(?:opposite\s+day|everything\s+(?:is\s+)?reversed|safety\s+(?:is\s+)?(?:off|disabled)"
            r"|(?:remove|disable|turn\s+off)\s+(?:all\s+)?(?:safety|filters?|restrictions?"
            r"|guardrails?|safeguards?|content\s+(?:filter|policy)))"
        ),
        category=PatternCategory.JAILBREAK,
        severity=Severity.CRITICAL,
        description="Attempts to disable safety features or invert the AI's guardrails.",
        confidence_weight=0.90,
    ),
    InjectionPattern(
        name="token_smuggling",
        regex=_compile(
            r"(?:ignore|bypass|skip)\s+(?:content\s+)?(?:filter|policy|safety|moderation)"
            r"|content\s+policy\s+(?:does\s+not|doesn'?t)\s+apply"
        ),
        category=PatternCategory.JAILBREAK,
        severity=Severity.CRITICAL,
        description="Explicitly attempts to bypass content policies or filters.",
        confidence_weight=0.90,
    ),
    # ── Multi-language Injection ──────────────────────────────────────────
    InjectionPattern(
        name="multilang_ignore_instructions",
        regex=_compile(
            r"(?:ignorar?\s+(?:las?\s+)?instrucciones"  # Spanish
            r"|ignorer?\s+les\s+instructions"  # French
            r"|ignoriere?\s+(?:die\s+)?Anweisungen"  # German
            r"|ignora(?:re)?\s+le\s+istruzioni"  # Italian
            r"|\u5ffd\u7565[\u4e00-\u9fff]*\u6307\u4ee4"  # Chinese: ignore instructions
            r"|\u6307\u793a\u3092\u7121\u8996"  # Japanese: ignore instructions
            r"|\uc9c0\uc2dc\ub97c\s*\ubb34\uc2dc"  # Korean: ignore instructions
            r")"
        ),
        category=PatternCategory.INSTRUCTION_OVERRIDE,
        severity=Severity.CRITICAL,
        description="Detects 'ignore instructions' phrased in non-English languages.",
        confidence_weight=0.85,
    ),
    # ── Special Character Smuggling ───────────────────────────────────────
    InjectionPattern(
        name="homoglyph_obfuscation",
        regex=_compile(
            r"[\u0410-\u044f]"  # Cyrillic characters mixed in Latin text
            r"[\x00-\x08\x0e-\x1f]"  # control characters
        ),
        category=PatternCategory.ENCODING_ATTACK,
        severity=Severity.MEDIUM,
        description="Detects Cyrillic homoglyphs or control characters used to obfuscate injections.",
        confidence_weight=0.50,
    ),
    InjectionPattern(
        name="indirect_injection_marker",
        regex=_compile(
            r"(?:^|\n)\s*(?:\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>"
            r"|<\|endoftext\|>|<\|system\|>|<\|user\|>|<\|assistant\|>|\[SYSTEM\])"
        ),
        category=PatternCategory.CONTEXT_ESCAPE,
        severity=Severity.CRITICAL,
        description="Injects model-specific special tokens to manipulate conversation framing.",
        confidence_weight=0.95,
    ),
]
