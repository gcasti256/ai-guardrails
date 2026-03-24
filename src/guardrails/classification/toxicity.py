"""Toxicity detection using weighted keyword scoring.

Provides keyword-based toxicity classification across multiple categories
with context-aware scoring that accounts for text length and keyword density.
No external ML models required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToxicityCategory(str, Enum):
    """Categories of toxic content."""

    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    PROFANITY = "profanity"


@dataclass(frozen=True)
class ToxicityResult:
    """Result from toxicity classification.

    Attributes:
        is_toxic: Whether the text exceeds the toxicity threshold.
        overall_score: Aggregate toxicity score in the range [0, 1].
        category_scores: Per-category toxicity scores.
        flagged_categories: Categories that exceeded the threshold.
        threshold: The threshold value used for this classification.
    """

    is_toxic: bool
    overall_score: float
    category_scores: dict[str, float]
    flagged_categories: list[str]
    threshold: float


@dataclass(frozen=True)
class _KeywordEntry:
    """A single keyword entry with its weight."""

    pattern: str
    weight: float
    is_regex: bool = False


# ---------------------------------------------------------------------------
# Keyword dictionaries per category.
#
# Weights are in [0, 1].  Exact-match entries use ``is_regex=False`` and are
# checked via word-boundary search.  Pattern entries use ``is_regex=True``
# and are compiled as regular expressions.
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[ToxicityCategory, list[_KeywordEntry]] = {
    ToxicityCategory.HATE_SPEECH: [
        # Exact-match slurs and hate terms (high weight)
        _KeywordEntry("racist", 0.9),
        _KeywordEntry("bigot", 0.8),
        _KeywordEntry("supremacist", 0.95),
        _KeywordEntry("supremacy", 0.9),
        _KeywordEntry("inferior race", 0.95),
        _KeywordEntry("ethnic cleansing", 1.0),
        _KeywordEntry("subhuman", 0.95),
        _KeywordEntry("degenerate", 0.7),
        # Pattern matches
        _KeywordEntry(r"\bgo\s+back\s+to\s+your\s+country\b", 0.9, is_regex=True),
        _KeywordEntry(r"\b(?:all|every)\s+\w+\s+(?:are|is)\s+(?:trash|garbage|scum)\b", 0.85, is_regex=True),
    ],
    ToxicityCategory.HARASSMENT: [
        _KeywordEntry("loser", 0.4),
        _KeywordEntry("worthless", 0.6),
        _KeywordEntry("pathetic", 0.5),
        _KeywordEntry("ugly", 0.4),
        _KeywordEntry("nobody likes you", 0.8),
        _KeywordEntry("kill yourself", 1.0),
        _KeywordEntry("kys", 0.95),
        _KeywordEntry("neck yourself", 1.0),
        _KeywordEntry("stalking", 0.7),
        _KeywordEntry("doxxing", 0.85),
        _KeywordEntry("doxxed", 0.85),
        # Patterns
        _KeywordEntry(r"\byou\s+(?:are|re)\s+(?:worthless|pathetic|garbage|trash)\b", 0.8, is_regex=True),
        _KeywordEntry(r"\bnobody\s+(?:cares|loves|wants)\b", 0.6, is_regex=True),
    ],
    ToxicityCategory.VIOLENCE: [
        _KeywordEntry("murder", 0.7),
        _KeywordEntry("assault", 0.6),
        _KeywordEntry("stab", 0.7),
        _KeywordEntry("shoot", 0.6),
        _KeywordEntry("bomb", 0.6),
        _KeywordEntry("massacre", 0.9),
        _KeywordEntry("execute", 0.5),
        _KeywordEntry("slaughter", 0.8),
        _KeywordEntry("behead", 0.95),
        _KeywordEntry("dismember", 0.9),
        _KeywordEntry("torture", 0.8),
        # Patterns: threats
        _KeywordEntry(r"\bi\s*(?:'ll|will|am\s+going\s+to)\s+(?:kill|hurt|destroy)\b", 0.95, is_regex=True),
        _KeywordEntry(r"\b(?:gonna|going\s+to)\s+(?:beat|hit|punch|stab|shoot)\b", 0.9, is_regex=True),
    ],
    ToxicityCategory.SEXUAL: [
        _KeywordEntry("pornography", 0.7),
        _KeywordEntry("sexually explicit", 0.8),
        _KeywordEntry("nude", 0.5),
        _KeywordEntry("genitalia", 0.6),
        _KeywordEntry("obscene", 0.5),
        # Patterns
        _KeywordEntry(r"\bsexual\s+(?:act|content|material)\b", 0.7, is_regex=True),
    ],
    ToxicityCategory.SELF_HARM: [
        _KeywordEntry("suicide", 0.7),
        _KeywordEntry("self-harm", 0.8),
        _KeywordEntry("cut myself", 0.85),
        _KeywordEntry("end my life", 0.9),
        _KeywordEntry("want to die", 0.85),
        _KeywordEntry("overdose", 0.6),
        _KeywordEntry("hang myself", 0.9),
        # Patterns
        _KeywordEntry(r"\b(?:how|best\s+way)\s+to\s+(?:kill|end)\s+(?:myself|my\s+life)\b", 1.0, is_regex=True),
    ],
    ToxicityCategory.PROFANITY: [
        _KeywordEntry("damn", 0.2),
        _KeywordEntry("hell", 0.15),
        _KeywordEntry("crap", 0.15),
        _KeywordEntry("shit", 0.35),
        _KeywordEntry("bullshit", 0.4),
        _KeywordEntry("fuck", 0.5),
        _KeywordEntry("fucking", 0.5),
        _KeywordEntry("asshole", 0.5),
        _KeywordEntry("bastard", 0.4),
        _KeywordEntry("bitch", 0.5),
        _KeywordEntry("dick", 0.35),
        _KeywordEntry("piss", 0.3),
        # Patterns: masked/obfuscated profanity
        _KeywordEntry(r"\bf+[u\*@]+[c\*@]+[k\*@]+", 0.45, is_regex=True),
        _KeywordEntry(r"\bs+h+[i\*@!]+t+", 0.3, is_regex=True),
    ],
}

# Category severity weights — used when computing the overall score.
_CATEGORY_SEVERITY: dict[ToxicityCategory, float] = {
    ToxicityCategory.HATE_SPEECH: 1.0,
    ToxicityCategory.HARASSMENT: 0.9,
    ToxicityCategory.VIOLENCE: 1.0,
    ToxicityCategory.SEXUAL: 0.7,
    ToxicityCategory.SELF_HARM: 1.0,
    ToxicityCategory.PROFANITY: 0.4,
}


class ToxicityClassifier:
    """Keyword-based toxicity classifier with weighted scoring.

    The classifier scans input text against curated keyword lists for each
    toxicity category.  Scores are *context-aware*: a single match in a long
    body of text produces a lower score than the same match in a short
    message, reflecting the overall density of toxic content.

    Args:
        threshold: Score above which content is considered toxic.  Defaults
            to ``0.5``.
        custom_keywords: Optional extra keywords to merge into the built-in
            lists.  Keys are :class:`ToxicityCategory` values (or their
            string equivalents); values are lists of ``(pattern, weight)``
            tuples.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        custom_keywords: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self._threshold = threshold

        # Build compiled keyword lists — start from defaults and merge custom.
        self._keywords: dict[ToxicityCategory, list[tuple[re.Pattern[str], float]]] = {}
        for category, entries in _CATEGORY_KEYWORDS.items():
            compiled: list[tuple[re.Pattern[str], float]] = []
            for entry in entries:
                compiled.append(self._compile_entry(entry))
            self._keywords[category] = compiled

        if custom_keywords:
            for cat_str, pairs in custom_keywords.items():
                category = ToxicityCategory(cat_str)
                if category not in self._keywords:
                    self._keywords[category] = []
                for pattern, weight in pairs:
                    entry = _KeywordEntry(pattern, weight, is_regex=False)
                    self._keywords[category].append(self._compile_entry(entry))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """The toxicity threshold in use."""
        return self._threshold

    async def classify(self, text: str) -> ToxicityResult:
        """Classify the toxicity of *text*.

        Returns a :class:`ToxicityResult` containing per-category scores
        and an aggregate determination of whether the text is toxic.

        The scoring is context-aware: keyword density relative to total
        word count is factored in so that a single profane word buried in
        a long paragraph scores substantially lower than a short, toxic
        message.
        """
        if not text or not text.strip():
            return ToxicityResult(
                is_toxic=False,
                overall_score=0.0,
                category_scores={cat.value: 0.0 for cat in ToxicityCategory},
                flagged_categories=[],
                threshold=self._threshold,
            )

        normalized = text.lower().strip()
        word_count = max(len(normalized.split()), 1)

        category_scores: dict[str, float] = {}
        flagged: list[str] = []

        for category, patterns in self._keywords.items():
            raw_score = self._score_category(normalized, patterns)

            # Context adjustment: dampen score for longer texts.
            # A single keyword hit in a 200-word text should score much
            # lower than the same hit in a 5-word message.
            context_factor = self._context_factor(word_count)
            adjusted = min(raw_score * context_factor, 1.0)

            category_scores[category.value] = round(adjusted, 4)
            if adjusted >= self._threshold:
                flagged.append(category.value)

        # Overall score: severity-weighted average of category scores.
        overall = self._compute_overall(category_scores)

        return ToxicityResult(
            is_toxic=overall >= self._threshold or len(flagged) > 0,
            overall_score=round(overall, 4),
            category_scores=category_scores,
            flagged_categories=sorted(flagged),
            threshold=self._threshold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_entry(entry: _KeywordEntry) -> tuple[re.Pattern[str], float]:
        """Compile a keyword entry into a ``(pattern, weight)`` pair."""
        if entry.is_regex:
            return (re.compile(entry.pattern, re.IGNORECASE), entry.weight)
        # Exact-match: wrap in word boundaries.
        escaped = re.escape(entry.pattern)
        return (re.compile(rf"\b{escaped}\b", re.IGNORECASE), entry.weight)

    @staticmethod
    def _score_category(
        text: str,
        patterns: list[tuple[re.Pattern[str], float]],
    ) -> float:
        """Compute the raw score for a single category.

        Each keyword hit contributes its weight.  Multiple hits of the
        same keyword are counted once (we care about presence, not
        frequency, to avoid gaming with repetition).  The result is
        clamped to [0, 1].
        """
        total = 0.0
        for pattern, weight in patterns:
            if pattern.search(text):
                total += weight
        return min(total, 1.0)

    @staticmethod
    def _context_factor(word_count: int) -> float:
        """Return a dampening multiplier based on text length.

        Short texts (< 10 words) get full weight.  Longer texts are
        progressively dampened, but never below 0.3 so that genuinely
        toxic content in long texts is still detected.
        """
        if word_count <= 10:
            return 1.0
        # Logarithmic dampening: factor drops as length increases.
        import math

        factor = 1.0 / (1.0 + 0.15 * math.log(word_count / 10))
        return max(factor, 0.3)

    @staticmethod
    def _compute_overall(category_scores: dict[str, float]) -> float:
        """Compute a severity-weighted overall toxicity score."""
        weighted_sum = 0.0
        weight_total = 0.0

        for category in ToxicityCategory:
            score = category_scores.get(category.value, 0.0)
            severity = _CATEGORY_SEVERITY.get(category, 0.5)
            weighted_sum += score * severity
            weight_total += severity

        if weight_total == 0.0:
            return 0.0
        return weighted_sum / weight_total
