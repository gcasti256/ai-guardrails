"""Language detection wrapping the ``langdetect`` library.

Provides async-friendly language detection with support for
allowed-language enforcement and confidence reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LanguageCandidate:
    """A single language candidate with its probability.

    Attributes:
        language: ISO 639-1 language code (e.g. ``"en"``, ``"fr"``).
        probability: Detection probability in the range [0, 1].
    """

    language: str
    probability: float


@dataclass(frozen=True)
class LanguageResult:
    """Result from language detection.

    Attributes:
        language: ISO 639-1 code of the most probable language.
        confidence: Confidence score for the detected language in [0, 1].
        all_languages: All candidate languages ordered by probability.
        is_allowed: ``True`` if the detected language is in the allowed
            set, or if no allowed-language filter was configured.
    """

    language: str
    confidence: float
    all_languages: list[LanguageCandidate]
    is_allowed: bool


class LanguageDetector:
    """Language detection powered by ``langdetect``.

    Wraps the ``langdetect`` library to provide an async interface
    with support for restricting results to a set of allowed languages.

    Args:
        allowed_languages: Optional list of ISO 639-1 codes.  When
            provided, the :attr:`LanguageResult.is_allowed` field will
            be ``False`` for text not in one of these languages.
        seed: Deterministic seed for ``langdetect`` (aids reproducibility).
            Defaults to ``0``.

    Raises:
        ImportError: If the ``langdetect`` package is not installed.

    Example::

        detector = LanguageDetector(allowed_languages=["en", "es"])
        result = await detector.detect("Hola, como estas?")
        assert result.language == "es"
        assert result.is_allowed is True
    """

    def __init__(
        self,
        allowed_languages: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        # Validate langdetect availability eagerly so users get a clear
        # error at construction time rather than at first ``detect()`` call.
        try:
            import langdetect  # noqa: F401
            from langdetect import DetectorFactory
        except ImportError as exc:
            raise ImportError(
                "The 'langdetect' package is required for LanguageDetector. "
                "Install it with: pip install langdetect"
            ) from exc

        DetectorFactory.seed = seed

        self._allowed_languages: set[str] | None = (
            {lang.lower().strip() for lang in allowed_languages}
            if allowed_languages
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def allowed_languages(self) -> set[str] | None:
        """The set of allowed ISO 639-1 language codes, or ``None``."""
        return set(self._allowed_languages) if self._allowed_languages else None

    async def detect(self, text: str) -> LanguageResult:
        """Detect the language of *text*.

        Args:
            text: The input text to analyze.  Very short texts (fewer
                than ~20 characters) may produce low-confidence results.

        Returns:
            A :class:`LanguageResult` with the detected language and all
            candidates.

        Raises:
            ValueError: If *text* is empty or contains no detectable
                features.
        """
        if not text or not text.strip():
            raise ValueError("Cannot detect language of empty text.")

        import langdetect
        from langdetect import detect_langs

        try:
            raw_results = detect_langs(text)
        except langdetect.lang_detect_exception.LangDetectException as exc:
            raise ValueError(
                f"Language detection failed: {exc}"
            ) from exc

        candidates: list[LanguageCandidate] = [
            LanguageCandidate(
                language=str(result.lang),
                probability=round(float(result.prob), 4),
            )
            for result in raw_results
        ]

        # Sort by probability descending (langdetect usually does this,
        # but we enforce it).
        candidates.sort(key=lambda c: c.probability, reverse=True)

        top = candidates[0] if candidates else LanguageCandidate("unknown", 0.0)

        is_allowed = True
        if self._allowed_languages is not None:
            is_allowed = top.language.lower() in self._allowed_languages

        return LanguageResult(
            language=top.language,
            confidence=top.probability,
            all_languages=candidates,
            is_allowed=is_allowed,
        )
