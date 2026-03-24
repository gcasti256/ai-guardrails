"""Topic classification using TF-IDF-like keyword matching.

Provides lightweight topic detection without external ML dependencies.
Topics are defined via configuration as lists of keywords and phrases,
and text is scored against each topic using term-frequency weighting
with inverse-document-frequency-like normalization.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TopicScore:
    """A single detected topic with its relevance score.

    Attributes:
        topic: The topic name.
        score: Relevance score in the range [0, 1].
        matched_terms: Terms from the topic vocabulary that were found.
    """

    topic: str
    score: float
    matched_terms: list[str]


@dataclass(frozen=True)
class TopicResult:
    """Result from topic classification.

    Attributes:
        is_on_topic: Whether the text matches at least one allowed topic
            above the confidence threshold.
        detected_topics: All detected topics sorted by score (descending).
        confidence: Confidence in the top-detected topic (0 if none).
    """

    is_on_topic: bool
    detected_topics: list[TopicScore]
    confidence: float


@dataclass
class TopicDefinition:
    """Definition of a topic for classification.

    Attributes:
        name: Human-readable topic name.
        keywords: Core keywords for the topic.
        phrases: Multi-word phrases that strongly indicate the topic.
        weight_boost: Multiplier applied to this topic's score (default 1.0).
            Use values > 1 to prioritize certain topics.
    """

    name: str
    keywords: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    weight_boost: float = 1.0


# ---------------------------------------------------------------------------
# Built-in general-purpose topics.  Users will typically override these
# by providing their own ``TopicDefinition`` list.
# ---------------------------------------------------------------------------

_DEFAULT_TOPICS: list[TopicDefinition] = [
    TopicDefinition(
        name="technology",
        keywords=[
            "software", "hardware", "computer", "programming", "code",
            "algorithm", "database", "api", "server", "cloud", "network",
            "machine learning", "artificial intelligence", "ai", "ml",
            "data", "engineering", "developer", "deploy", "infrastructure",
        ],
        phrases=["machine learning", "artificial intelligence", "deep learning",
                 "open source", "cloud computing", "data science"],
    ),
    TopicDefinition(
        name="business",
        keywords=[
            "revenue", "profit", "market", "strategy", "sales", "customer",
            "investment", "startup", "enterprise", "growth", "roi",
            "acquisition", "partnership", "stakeholder", "budget",
        ],
        phrases=["business model", "market share", "supply chain",
                 "return on investment", "go to market"],
    ),
    TopicDefinition(
        name="health",
        keywords=[
            "medical", "health", "disease", "treatment", "diagnosis",
            "patient", "clinical", "therapy", "symptom", "vaccine",
            "hospital", "doctor", "medication", "wellness", "nutrition",
        ],
        phrases=["mental health", "clinical trial", "side effects",
                 "health care", "public health"],
    ),
    TopicDefinition(
        name="science",
        keywords=[
            "research", "experiment", "hypothesis", "theory", "physics",
            "chemistry", "biology", "study", "peer-reviewed", "journal",
            "discovery", "laboratory", "evidence", "observation", "analysis",
        ],
        phrases=["peer review", "scientific method", "control group",
                 "climate change", "natural selection"],
    ),
    TopicDefinition(
        name="education",
        keywords=[
            "learning", "teaching", "student", "curriculum", "school",
            "university", "course", "classroom", "academic", "professor",
            "degree", "scholarship", "lecture", "exam", "education",
        ],
        phrases=["higher education", "online learning", "student loan",
                 "distance education"],
    ),
]


class TopicClassifier:
    """Lightweight topic classifier using TF-IDF-inspired keyword matching.

    Topics are defined as collections of keywords and phrases.  For each
    topic the classifier computes a relevance score based on:

    1. **Term frequency (TF)** -- how many of the topic's terms appear in
       the input text relative to the total word count.
    2. **Inverse topic frequency (ITF)** -- terms that are unique to a
       single topic contribute more than terms shared across many topics,
       analogous to IDF in traditional TF-IDF.
    3. **Phrase bonus** -- multi-word phrases receive a boost because they
       are stronger topic indicators than isolated keywords.

    Args:
        topics: List of :class:`TopicDefinition` objects.  If ``None``,
            a set of general-purpose default topics is used.
        confidence_threshold: Minimum score to consider a topic detected.
            Defaults to ``0.15``.
    """

    def __init__(
        self,
        topics: list[TopicDefinition] | None = None,
        confidence_threshold: float = 0.15,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )

        self._topics = topics if topics is not None else list(_DEFAULT_TOPICS)
        self._confidence_threshold = confidence_threshold

        # Pre-compute inverse topic frequency for all terms.
        self._itf: dict[str, float] = self._build_itf(self._topics)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def confidence_threshold(self) -> float:
        """The minimum score to consider a topic detected."""
        return self._confidence_threshold

    @property
    def topics(self) -> list[TopicDefinition]:
        """The configured topic definitions."""
        return list(self._topics)

    async def classify(
        self,
        text: str,
        allowed_topics: list[str] | None = None,
    ) -> TopicResult:
        """Classify *text* against the configured topics.

        Args:
            text: The input text to classify.
            allowed_topics: If provided, only these topic names are
                considered when deciding ``is_on_topic``.  All topics are
                still scored and returned in ``detected_topics``.

        Returns:
            A :class:`TopicResult` with scores and on-topic determination.
        """
        if not text or not text.strip():
            return TopicResult(
                is_on_topic=False,
                detected_topics=[],
                confidence=0.0,
            )

        normalized = text.lower().strip()
        word_count = max(len(normalized.split()), 1)
        text_tokens = self._tokenize(normalized)
        token_counts = Counter(text_tokens)

        scored: list[TopicScore] = []

        for topic_def in self._topics:
            score, matched = self._score_topic(
                normalized, token_counts, word_count, topic_def,
            )
            if matched:
                scored.append(TopicScore(
                    topic=topic_def.name,
                    score=round(score, 4),
                    matched_terms=sorted(matched),
                ))

        # Sort by score descending.
        scored.sort(key=lambda t: t.score, reverse=True)

        # Filter to topics above threshold.
        detected = [t for t in scored if t.score >= self._confidence_threshold]

        # Determine on-topic status.
        if allowed_topics is not None:
            allowed_set = {t.lower() for t in allowed_topics}
            on_topic = any(
                t.topic.lower() in allowed_set and t.score >= self._confidence_threshold
                for t in scored
            )
        else:
            on_topic = len(detected) > 0

        confidence = detected[0].score if detected else 0.0

        return TopicResult(
            is_on_topic=on_topic,
            detected_topics=detected,
            confidence=round(confidence, 4),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase word tokens."""
        return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())

    @staticmethod
    def _build_itf(topics: list[TopicDefinition]) -> dict[str, float]:
        """Build inverse-topic-frequency mapping.

        ITF for a term = log(N / n_t) where N is the total number of
        topics and n_t is the number of topics containing that term.
        """
        n_topics = max(len(topics), 1)
        term_topic_count: dict[str, int] = Counter()

        for topic_def in topics:
            seen: set[str] = set()
            for kw in topic_def.keywords:
                normalized = kw.lower().strip()
                if normalized not in seen:
                    term_topic_count[normalized] += 1
                    seen.add(normalized)
            for phrase in topic_def.phrases:
                normalized = phrase.lower().strip()
                if normalized not in seen:
                    term_topic_count[normalized] += 1
                    seen.add(normalized)

        itf: dict[str, float] = {}
        for term, count in term_topic_count.items():
            itf[term] = math.log(n_topics / count) + 1.0
        return itf

    def _score_topic(
        self,
        text: str,
        token_counts: Counter[str],
        word_count: int,
        topic_def: TopicDefinition,
    ) -> tuple[float, list[str]]:
        """Score *text* against a single topic definition.

        Returns ``(score, matched_terms)`` where score is in [0, 1].
        """
        matched: list[str] = []
        raw_score = 0.0

        # Score individual keywords.
        for kw in topic_def.keywords:
            kw_lower = kw.lower().strip()
            # For single-word keywords, use token counting.
            if " " not in kw_lower:
                count = token_counts.get(kw_lower, 0)
                if count > 0:
                    tf = count / word_count
                    itf = self._itf.get(kw_lower, 1.0)
                    raw_score += tf * itf
                    matched.append(kw_lower)
            else:
                # Multi-word keyword: check as substring.
                if kw_lower in text:
                    tf = text.count(kw_lower) / word_count
                    itf = self._itf.get(kw_lower, 1.0)
                    raw_score += tf * itf * 1.5  # Phrase bonus.
                    matched.append(kw_lower)

        # Score phrases (stronger signal).
        for phrase in topic_def.phrases:
            phrase_lower = phrase.lower().strip()
            if phrase_lower in text:
                tf = text.count(phrase_lower) / word_count
                itf = self._itf.get(phrase_lower, 1.0)
                raw_score += tf * itf * 2.0  # Stronger phrase bonus.
                if phrase_lower not in matched:
                    matched.append(phrase_lower)

        # Apply topic weight boost and normalize.
        raw_score *= topic_def.weight_boost

        # Normalize: we use a sigmoid-like mapping so scores cluster in
        # a useful [0, 1] range.  The constant 2.0 was chosen so that
        # a moderately on-topic paragraph scores around 0.5-0.7.
        score = 1.0 - math.exp(-2.0 * raw_score)
        score = max(0.0, min(score, 1.0))

        return score, matched
