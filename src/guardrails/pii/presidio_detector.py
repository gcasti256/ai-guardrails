"""Microsoft Presidio integration for PII detection.

Wraps the Presidio Analyzer to detect PII and maps Presidio entity
types to guardrails entity types.  Falls back gracefully when Presidio
is not installed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from guardrails.types import DetectionResult, Severity

logger = logging.getLogger(__name__)

#: Mapping from Presidio entity types to guardrails entity types.
PRESIDIO_ENTITY_MAP: dict[str, str] = {
    "PERSON": "PERSON_NAME",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "CREDIT_CARD": "CREDIT_CARD",
    "US_SSN": "SSN",
    "US_PASSPORT": "PASSPORT_NUMBER",
    "US_DRIVER_LICENSE": "DRIVERS_LICENSE",
    "IP_ADDRESS": "IP_ADDRESS",
    "IBAN_CODE": "BANK_ACCOUNT",
    "US_BANK_NUMBER": "BANK_ACCOUNT",
    "LOCATION": "LOCATION",
    "DATE_TIME": "DATE",
    "NRP": "ORGANIZATION",
    "MEDICAL_LICENSE": "MEDICAL_ID",
    "URL": "URL",
}

#: Default severity per guardrails entity type (for Presidio-sourced detections).
PRESIDIO_SEVERITY_MAP: dict[str, Severity] = {
    "PERSON_NAME": Severity.HIGH,
    "EMAIL_ADDRESS": Severity.HIGH,
    "PHONE_NUMBER": Severity.HIGH,
    "CREDIT_CARD": Severity.CRITICAL,
    "SSN": Severity.CRITICAL,
    "PASSPORT_NUMBER": Severity.CRITICAL,
    "DRIVERS_LICENSE": Severity.HIGH,
    "IP_ADDRESS": Severity.MEDIUM,
    "BANK_ACCOUNT": Severity.CRITICAL,
    "LOCATION": Severity.LOW,
    "DATE": Severity.LOW,
    "ORGANIZATION": Severity.MEDIUM,
    "MEDICAL_ID": Severity.CRITICAL,
    "URL": Severity.LOW,
}


class PresidioDetector:
    """PII detector backed by Microsoft Presidio.

    Args:
        language: Language code for the analyzer (default ``"en"``).
        entities: Optional list of Presidio entity type strings to detect.
            ``None`` means all supported entities.
        score_threshold: Minimum Presidio score to accept a result.
        extra_recognizers: Additional Presidio ``EntityRecognizer`` instances
            to register with the analyzer engine.

    If Presidio is not installed the detector logs a warning and returns
    empty results from :meth:`detect`.

    Example::

        detector = PresidioDetector(entities=["PERSON", "EMAIL_ADDRESS"])
        results = await detector.detect("Contact alice@example.com")
    """

    def __init__(
        self,
        *,
        language: str = "en",
        entities: list[str] | None = None,
        score_threshold: float = 0.4,
        extra_recognizers: list[Any] | None = None,
    ) -> None:
        self._language = language
        self._entities = entities
        self._score_threshold = score_threshold
        self._extra_recognizers = extra_recognizers or []
        self._analyzer: Any | None = None
        self._available: bool = False
        self._load_analyzer()

    # ------------------------------------------------------------------
    # Analyzer loading
    # ------------------------------------------------------------------

    def _load_analyzer(self) -> None:
        """Attempt to load the Presidio analyzer engine."""
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-untyped]

            self._analyzer = AnalyzerEngine()

            # Register any custom recognizers
            for recognizer in self._extra_recognizers:
                self._analyzer.registry.add_recognizer(recognizer)

            self._available = True
            logger.info("PresidioDetector loaded successfully")
        except ImportError:
            logger.warning(
                "presidio-analyzer is not installed — PresidioDetector will "
                "return no results. Install with: pip install presidio-analyzer"
            )
        except Exception:
            logger.warning(
                "Failed to initialize Presidio AnalyzerEngine",
                exc_info=True,
            )

    @property
    def is_available(self) -> bool:
        """Return True if Presidio was loaded successfully."""
        return self._available

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    async def detect(self, text: str) -> list[DetectionResult]:
        """Detect PII in *text* asynchronously using Presidio.

        Runs the analyzer in the default executor to avoid blocking
        the event loop.

        Args:
            text: The text to scan.

        Returns:
            List of :class:`DetectionResult`.
        """
        if not self._available:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.detect_sync, text)

    def detect_sync(self, text: str) -> list[DetectionResult]:
        """Detect PII in *text* synchronously using Presidio.

        Args:
            text: The text to scan.

        Returns:
            List of :class:`DetectionResult`.
        """
        if not self._available or self._analyzer is None:
            return []

        presidio_results = self._analyzer.analyze(
            text=text,
            language=self._language,
            entities=self._entities,
            score_threshold=self._score_threshold,
        )

        results: list[DetectionResult] = []
        for pr in presidio_results:
            guardrails_type = PRESIDIO_ENTITY_MAP.get(pr.entity_type, pr.entity_type)
            severity = PRESIDIO_SEVERITY_MAP.get(guardrails_type, Severity.MEDIUM)

            results.append(
                DetectionResult(
                    entity_type=guardrails_type,
                    text=text[pr.start:pr.end],
                    start=pr.start,
                    end=pr.end,
                    confidence=pr.score,
                    detector="presidio",
                    severity=severity,
                    metadata={
                        "presidio_entity_type": pr.entity_type,
                        "analysis_explanation": (
                            str(pr.analysis_explanation)
                            if pr.analysis_explanation
                            else None
                        ),
                    },
                )
            )

        return results
