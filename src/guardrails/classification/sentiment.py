"""Lexicon-based sentiment analysis.

Provides sentiment classification (positive / negative / neutral) using
curated word lists with intensifier and negation handling.  No external
ML models or large dependencies required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Sentiment(str, Enum):
    """Sentiment labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SentimentResult:
    """Result from sentiment analysis.

    Attributes:
        sentiment: The overall sentiment label.
        score: Sentiment score in the range [-1, 1] where -1 is most
            negative and 1 is most positive.
        confidence: Confidence in the determination, in [0, 1].
    """

    sentiment: Sentiment
    score: float
    confidence: float


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Positive words mapped to their base valence (0, 1].
_POSITIVE_LEXICON: dict[str, float] = {
    # Strong positive
    "excellent": 0.9,
    "amazing": 0.85,
    "fantastic": 0.85,
    "wonderful": 0.85,
    "outstanding": 0.9,
    "brilliant": 0.85,
    "superb": 0.85,
    "exceptional": 0.9,
    "perfect": 0.9,
    "magnificent": 0.85,
    "incredible": 0.8,
    "love": 0.8,
    "loved": 0.8,
    "adore": 0.8,
    "thrilled": 0.8,
    "delighted": 0.8,
    "ecstatic": 0.85,
    "overjoyed": 0.85,
    # Moderate positive
    "good": 0.5,
    "great": 0.6,
    "nice": 0.45,
    "happy": 0.6,
    "pleased": 0.55,
    "glad": 0.55,
    "enjoy": 0.55,
    "enjoyed": 0.55,
    "enjoyable": 0.55,
    "like": 0.4,
    "liked": 0.4,
    "helpful": 0.55,
    "useful": 0.5,
    "impressive": 0.6,
    "recommend": 0.55,
    "recommended": 0.55,
    "satisfied": 0.5,
    "comfortable": 0.45,
    "pleasant": 0.5,
    "positive": 0.5,
    "beautiful": 0.65,
    "elegant": 0.6,
    "smooth": 0.4,
    "efficient": 0.5,
    "effective": 0.5,
    # Mild positive
    "fine": 0.25,
    "okay": 0.15,
    "ok": 0.15,
    "decent": 0.3,
    "fair": 0.2,
    "adequate": 0.2,
    "reasonable": 0.25,
    "acceptable": 0.2,
    "alright": 0.2,
    "thanks": 0.35,
    "thank": 0.35,
    "appreciate": 0.45,
    "grateful": 0.55,
    "welcome": 0.35,
}

# Negative words mapped to their base valence [-1, 0).
_NEGATIVE_LEXICON: dict[str, float] = {
    # Strong negative
    "terrible": -0.9,
    "horrible": -0.9,
    "awful": -0.85,
    "dreadful": -0.85,
    "atrocious": -0.9,
    "abysmal": -0.9,
    "disgusting": -0.85,
    "hate": -0.8,
    "hated": -0.8,
    "despise": -0.85,
    "loathe": -0.85,
    "worst": -0.9,
    "horrendous": -0.9,
    "catastrophic": -0.85,
    "disastrous": -0.85,
    "miserable": -0.8,
    "furious": -0.75,
    "outraged": -0.8,
    # Moderate negative
    "bad": -0.5,
    "poor": -0.55,
    "wrong": -0.45,
    "disappointing": -0.6,
    "disappointed": -0.6,
    "frustrating": -0.6,
    "frustrated": -0.6,
    "annoying": -0.55,
    "annoyed": -0.55,
    "angry": -0.6,
    "upset": -0.55,
    "unhappy": -0.55,
    "unpleasant": -0.5,
    "ugly": -0.5,
    "broken": -0.55,
    "failed": -0.55,
    "failure": -0.6,
    "useless": -0.65,
    "waste": -0.5,
    "boring": -0.45,
    "confusing": -0.45,
    "confused": -0.4,
    "difficult": -0.35,
    "problem": -0.4,
    "issue": -0.3,
    "error": -0.4,
    "bug": -0.35,
    "crash": -0.5,
    "slow": -0.35,
    "painful": -0.55,
    "regret": -0.55,
    # Mild negative
    "mediocre": -0.3,
    "lacking": -0.3,
    "meh": -0.2,
    "underwhelming": -0.35,
    "average": -0.1,
    "bland": -0.25,
    "dull": -0.3,
    "inconvenient": -0.3,
    "unfortunately": -0.3,
    "sadly": -0.3,
}

# Intensifiers multiply the valence of the next sentiment word.
_INTENSIFIERS: dict[str, float] = {
    "very": 1.5,
    "really": 1.4,
    "extremely": 1.8,
    "incredibly": 1.7,
    "absolutely": 1.7,
    "totally": 1.5,
    "completely": 1.5,
    "utterly": 1.6,
    "highly": 1.4,
    "super": 1.4,
    "so": 1.3,
    "quite": 1.2,
    "pretty": 1.1,
    "rather": 1.1,
    "somewhat": 0.8,
    "slightly": 0.6,
    "barely": 0.5,
    "hardly": 0.5,
    "a bit": 0.7,
    "a little": 0.7,
    "kind of": 0.8,
    "sort of": 0.8,
    "mostly": 1.1,
    "especially": 1.4,
    "particularly": 1.3,
}

# Negation words flip the sign of the next sentiment word.
_NEGATION_WORDS: set[str] = {
    "not", "no", "never", "neither", "nor", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "doesn't", "don't",
    "didn't", "isn't", "wasn't", "aren't", "weren't", "won't",
    "wouldn't", "shouldn't", "couldn't", "hasn't", "haven't",
    "hadn't", "cannot", "can't", "without",
}

# Contraction patterns that contain negation.
_CONTRACTION_NEGATIONS: dict[str, str] = {
    "n't": "not",
    "cant": "cannot",
    "wont": "will not",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "isnt": "is not",
    "wasnt": "was not",
    "arent": "are not",
    "werent": "were not",
    "wouldnt": "would not",
    "shouldnt": "should not",
    "couldnt": "could not",
    "hasnt": "has not",
    "havent": "have not",
    "hadnt": "had not",
}


class SentimentAnalyzer:
    """Lexicon-based sentiment analyzer with negation and intensifier handling.

    Computes a sentiment score by scanning tokens against positive and
    negative word lexicons, adjusting for:

    - **Negation**: words like "not", "never", "don't" flip the valence of
      the following sentiment word (within a window of 3 tokens).
    - **Intensifiers**: words like "very", "extremely", "barely" scale the
      magnitude of the following sentiment word.
    - **Punctuation emphasis**: exclamation marks and ALL CAPS amplify
      sentiment slightly.

    Args:
        neutral_threshold: Score magnitude below which text is considered
            neutral.  Defaults to ``0.1``.
        custom_positive: Additional positive words with their valences.
        custom_negative: Additional negative words with their valences.
    """

    def __init__(
        self,
        neutral_threshold: float = 0.1,
        custom_positive: dict[str, float] | None = None,
        custom_negative: dict[str, float] | None = None,
    ) -> None:
        if not 0.0 <= neutral_threshold <= 1.0:
            raise ValueError(
                f"neutral_threshold must be in [0, 1], got {neutral_threshold}"
            )

        self._neutral_threshold = neutral_threshold

        # Merge lexicons.
        self._positive: dict[str, float] = dict(_POSITIVE_LEXICON)
        if custom_positive:
            self._positive.update(custom_positive)

        self._negative: dict[str, float] = dict(_NEGATIVE_LEXICON)
        if custom_negative:
            self._negative.update(custom_negative)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def neutral_threshold(self) -> float:
        """Score magnitude below which text is classified as neutral."""
        return self._neutral_threshold

    async def analyze(self, text: str) -> SentimentResult:
        """Analyze the sentiment of *text*.

        Args:
            text: The input text to analyze.

        Returns:
            A :class:`SentimentResult` with sentiment label, score, and
            confidence.
        """
        if not text or not text.strip():
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                score=0.0,
                confidence=0.0,
            )

        tokens = self._tokenize(text)
        if not tokens:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                score=0.0,
                confidence=0.0,
            )

        raw_scores: list[float] = self._score_tokens(tokens, text)

        if not raw_scores:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                score=0.0,
                confidence=0.0,
            )

        # Aggregate: mean of individual word scores.
        total = sum(raw_scores)
        mean_score = total / len(raw_scores)

        # Apply punctuation emphasis.
        emphasis = self._punctuation_emphasis(text)
        mean_score *= emphasis

        # Clamp to [-1, 1].
        final_score = max(-1.0, min(1.0, mean_score))

        # Determine label.
        if abs(final_score) < self._neutral_threshold:
            label = Sentiment.NEUTRAL
        elif final_score > 0:
            label = Sentiment.POSITIVE
        else:
            label = Sentiment.NEGATIVE

        # Confidence: based on consistency of individual scores and magnitude.
        confidence = self._compute_confidence(raw_scores, final_score)

        return SentimentResult(
            sentiment=label,
            score=round(final_score, 4),
            confidence=round(confidence, 4),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase words, preserving contractions."""
        return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

    def _score_tokens(self, tokens: list[str], original_text: str) -> list[float]:
        """Score each sentiment-bearing token, applying negation and intensifiers."""
        scores: list[float] = []
        negation_window = 0  # Remaining tokens in negation scope.
        intensifier_mult = 1.0

        for i, token in enumerate(tokens):
            # Check for negation.
            if token in _NEGATION_WORDS or token in _CONTRACTION_NEGATIONS:
                negation_window = 3  # Negate within next 3 tokens.
                continue

            # Check for intensifier.
            if token in _INTENSIFIERS:
                intensifier_mult = _INTENSIFIERS[token]
                continue

            # Check lexicons.
            valence: float | None = None
            if token in self._positive:
                valence = self._positive[token]
            elif token in self._negative:
                valence = self._negative[token]

            if valence is not None:
                # Apply intensifier.
                valence *= intensifier_mult
                intensifier_mult = 1.0  # Reset after use.

                # Apply negation: flip sign and dampen slightly (negation
                # doesn't perfectly invert — "not good" is weaker than "bad").
                if negation_window > 0:
                    valence *= -0.75

                scores.append(valence)

            # Decay negation window.
            if negation_window > 0:
                negation_window -= 1

            # Reset intensifier if not consumed by a sentiment word within
            # 1 token.
            if intensifier_mult != 1.0 and (i + 1 >= len(tokens) or tokens[i + 1] not in self._positive and tokens[i + 1] not in self._negative):
                intensifier_mult = 1.0

        return scores

    @staticmethod
    def _punctuation_emphasis(text: str) -> float:
        """Compute a multiplier based on punctuation emphasis.

        Exclamation marks and ALL-CAPS words amplify sentiment slightly.
        """
        multiplier = 1.0

        # Exclamation marks (diminishing returns).
        excl_count = text.count("!")
        if excl_count > 0:
            multiplier += min(excl_count * 0.05, 0.2)

        # Proportion of ALL-CAPS words (at least 3 chars to avoid acronyms
        # like "AI" triggering this).
        words = text.split()
        if words:
            caps_words = sum(
                1 for w in words
                if len(w) >= 3 and w.isupper() and w.isalpha()
            )
            caps_ratio = caps_words / len(words)
            if caps_ratio > 0.1:
                multiplier += min(caps_ratio * 0.3, 0.15)

        return multiplier

    @staticmethod
    def _compute_confidence(scores: list[float], final_score: float) -> float:
        """Compute a confidence value for the sentiment determination.

        Higher confidence when:
        - More sentiment words were found.
        - Scores are consistent in direction (all positive or all negative).
        - The magnitude is larger.
        """
        if not scores:
            return 0.0

        n = len(scores)

        # Factor 1: magnitude of the final score.
        magnitude_conf = abs(final_score)

        # Factor 2: agreement — proportion of scores sharing the same sign.
        if final_score >= 0:
            agreement = sum(1 for s in scores if s >= 0) / n
        else:
            agreement = sum(1 for s in scores if s < 0) / n

        # Factor 3: evidence count — more sentiment words = more confidence,
        # up to a point.
        import math

        evidence_conf = min(1.0, math.log(n + 1) / math.log(10))

        # Weighted combination.
        confidence = (
            0.4 * magnitude_conf
            + 0.35 * agreement
            + 0.25 * evidence_conf
        )

        return max(0.0, min(1.0, confidence))
