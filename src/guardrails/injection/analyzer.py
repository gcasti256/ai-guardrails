"""Heuristic-based semantic analysis for prompt injection detection.

Provides lightweight, non-ML analysis of text to identify structural and
semantic signals that indicate prompt injection attempts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Scoring result
# ---------------------------------------------------------------------------

@dataclass
class SemanticAnalysisResult:
    """Aggregated result from semantic analysis heuristics.

    Attributes:
        imperative_score: Likelihood that the text contains imperative
            directives aimed at an AI (0.0 - 1.0).
        roleplay_score: Likelihood that the text attempts to redefine
            the AI's role or persona (0.0 - 1.0).
        context_manipulation_score: Likelihood that the text tries to
            manipulate or escape the conversation context (0.0 - 1.0).
        overall_score: Weighted combination of all sub-scores (0.0 - 1.0).
        signals: List of human-readable descriptions of detected signals.
        metadata: Arbitrary extra data produced by individual heuristics.
    """

    imperative_score: float = 0.0
    roleplay_score: float = 0.0
    context_manipulation_score: float = 0.0
    overall_score: float = 0.0
    signals: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Compiled patterns used by heuristics
# ---------------------------------------------------------------------------

_IMPERATIVE_VERBS: re.Pattern[str] = re.compile(
    r"^\s*(?:ignore|disregard|forget|override|bypass|do|don'?t|never|always|stop|"
    r"start|begin|write|output|print|say|tell|respond|answer|generate|create|"
    r"produce|give|show|display|reveal|list|repeat|translate|decode|execute|"
    r"run|send|email|post|fetch|return|switch|enter|enable|disable|activate|"
    r"remove|delete|clear|reset|pretend|imagine|act|behave|become)\b",
    re.IGNORECASE | re.MULTILINE,
)

_SECOND_PERSON_DIRECTIVE: re.Pattern[str] = re.compile(
    r"\byou\s+(?:must|should|shall|will|need\s+to|have\s+to|are\s+(?:to|going\s+to))\b",
    re.IGNORECASE,
)

_AI_ADDRESSAL: re.Pattern[str] = re.compile(
    r"\b(?:as\s+an?\s+AI|as\s+a\s+(?:language\s+)?model|as\s+(?:ChatGPT|GPT|Claude|Gemini|"
    r"an?\s+assistant|a\s+chatbot|a\s+bot))\b",
    re.IGNORECASE,
)

_ROLEPLAY_PHRASES: re.Pattern[str] = re.compile(
    r"(?:you\s+are\s+(?:now\s+)?(?:a\s+|an\s+|the\s+)?|"
    r"act\s+as\s+(?:if\s+)?|"
    r"pretend\s+(?:to\s+be|you\s+are)\s+|"
    r"roleplay\s+as\s+|"
    r"assume\s+the\s+(?:role|identity|persona)\s+of\s+|"
    r"your\s+(?:new\s+)?(?:name|identity|role|persona)\s+is\s+)",
    re.IGNORECASE,
)

_CONTEXT_BOUNDARY_MARKERS: re.Pattern[str] = re.compile(
    r"(?:^|\n)\s*(?:---+|===+|```+|<<<+|>>>+|\*\*\*+|###\s*(?:system|end|new))\s*",
    re.IGNORECASE,
)

_META_INSTRUCTION: re.Pattern[str] = re.compile(
    r"(?:above\s+(?:is|was|are)\s+(?:your|the)\s+(?:instructions?|prompt|context)|"
    r"end\s+of\s+(?:system\s+)?(?:prompt|instructions?|context)|"
    r"(?:begin|start)\s+(?:new\s+)?(?:conversation|session|context)|"
    r"the\s+(?:real|actual|true)\s+(?:instructions?|prompt)\s+(?:is|are|follows?))",
    re.IGNORECASE,
)

_URGENCY_LANGUAGE: re.Pattern[str] = re.compile(
    r"\b(?:important|urgent|critical|priority|emergency|immediately|right\s+now|"
    r"this\s+is\s+(?:very\s+)?important|you\s+must|do\s+this\s+(?:first|now))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SemanticAnalyzer:
    """Heuristic-based semantic analyzer for prompt injection signals.

    Uses lightweight pattern matching and structural analysis -- no ML models
    required -- to score how likely a given text contains injection attempts.
    """

    # Weights for combining sub-scores into overall_score.
    IMPERATIVE_WEIGHT: float = 0.45
    ROLEPLAY_WEIGHT: float = 0.30
    CONTEXT_MANIPULATION_WEIGHT: float = 0.25

    def analyze(self, text: str) -> SemanticAnalysisResult:
        """Perform semantic analysis on *text* and return scored results.

        Args:
            text: The input text to analyze.

        Returns:
            A :class:`SemanticAnalysisResult` with sub-scores, overall score,
            and descriptive signal strings.
        """
        signals: list[str] = []
        metadata: dict[str, Any] = {}

        imperative_score = self._score_imperative(text, signals, metadata)
        roleplay_score = self._score_roleplay(text, signals, metadata)
        context_score = self._score_context_manipulation(text, signals, metadata)

        overall = (
            imperative_score * self.IMPERATIVE_WEIGHT
            + roleplay_score * self.ROLEPLAY_WEIGHT
            + context_score * self.CONTEXT_MANIPULATION_WEIGHT
        )

        return SemanticAnalysisResult(
            imperative_score=round(imperative_score, 4),
            roleplay_score=round(roleplay_score, 4),
            context_manipulation_score=round(context_score, 4),
            overall_score=round(min(overall, 1.0), 4),
            signals=signals,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Imperative scoring
    # ------------------------------------------------------------------

    def _score_imperative(
        self,
        text: str,
        signals: list[str],
        metadata: dict[str, Any],
    ) -> float:
        """Score the presence of imperative directives targeting an AI.

        Looks for imperative verb openings, second-person directives (\"you must\"),
        direct AI addressal, and urgency language.
        """
        score = 0.0
        sentences = _split_sentences(text)
        total = max(len(sentences), 1)

        # -- imperative verbs at sentence start --
        imperative_count = sum(
            1 for s in sentences if _IMPERATIVE_VERBS.search(s)
        )
        imperative_ratio = imperative_count / total
        if imperative_ratio > 0:
            contribution = min(imperative_ratio * 1.5, 0.5)
            score += contribution
            signals.append(
                f"Imperative verb ratio: {imperative_count}/{total} sentences"
            )
            metadata["imperative_verb_count"] = imperative_count

        # -- second-person directives --
        directive_hits = _SECOND_PERSON_DIRECTIVE.findall(text)
        if directive_hits:
            score += min(len(directive_hits) * 0.15, 0.3)
            signals.append(
                f"Second-person directives detected ({len(directive_hits)} occurrences)"
            )

        # -- AI addressal --
        if _AI_ADDRESSAL.search(text):
            score += 0.2
            signals.append("Direct AI addressal detected")

        # -- urgency language --
        if _URGENCY_LANGUAGE.search(text):
            score += 0.1
            signals.append("Urgency language detected")

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Roleplay scoring
    # ------------------------------------------------------------------

    def _score_roleplay(
        self,
        text: str,
        signals: list[str],
        metadata: dict[str, Any],
    ) -> float:
        """Score attempts to redefine the AI's role or persona."""
        score = 0.0

        roleplay_matches = _ROLEPLAY_PHRASES.findall(text)
        if roleplay_matches:
            score += min(len(roleplay_matches) * 0.35, 0.7)
            signals.append(
                f"Role manipulation phrases detected ({len(roleplay_matches)} occurrences)"
            )
            metadata["roleplay_match_count"] = len(roleplay_matches)

        # Bonus if combined with AI addressal
        if _AI_ADDRESSAL.search(text) and roleplay_matches:
            score += 0.3
            signals.append("Role manipulation combined with AI addressal")

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Context manipulation scoring
    # ------------------------------------------------------------------

    def _score_context_manipulation(
        self,
        text: str,
        signals: list[str],
        metadata: dict[str, Any],
    ) -> float:
        """Score attempts to escape or manipulate the conversation context."""
        score = 0.0

        # -- boundary markers --
        boundary_hits = _CONTEXT_BOUNDARY_MARKERS.findall(text)
        if boundary_hits:
            score += min(len(boundary_hits) * 0.2, 0.4)
            signals.append(
                f"Context boundary markers detected ({len(boundary_hits)} occurrences)"
            )
            metadata["boundary_marker_count"] = len(boundary_hits)

        # -- meta-instructions --
        if _META_INSTRUCTION.search(text):
            score += 0.4
            signals.append("Meta-instruction language detected")

        # -- excessive newlines / whitespace (padding attack) --
        newline_count = text.count("\n")
        if newline_count > 20 and len(text.strip()) < newline_count * 5:
            score += 0.2
            signals.append("Suspicious whitespace padding detected")

        return min(score, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Naively split *text* into sentence-like segments."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]
