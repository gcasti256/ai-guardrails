"""spaCy NER-based PII detector.

Uses a spaCy language model to detect named entities (PERSON, ORG, GPE,
LOC) and maps them to guardrails entity types.  Falls back gracefully
when spaCy or the requested model is not installed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from guardrails.types import DetectionResult, Severity

logger = logging.getLogger(__name__)

#: Mapping from spaCy entity labels to guardrails entity types.
SPACY_ENTITY_MAP: dict[str, str] = {
    "PERSON": "PERSON_NAME",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",       # Geo-Political Entity → LOCATION
    "LOC": "LOCATION",
    "NORP": "ORGANIZATION",  # Nationalities / groups
    "FAC": "LOCATION",       # Facilities
    "DATE": "DATE",
    "MONEY": "FINANCIAL",
}

#: Default severity per guardrails entity type.
DEFAULT_SEVERITY: dict[str, Severity] = {
    "PERSON_NAME": Severity.HIGH,
    "ORGANIZATION": Severity.MEDIUM,
    "LOCATION": Severity.LOW,
    "DATE": Severity.LOW,
    "FINANCIAL": Severity.HIGH,
}

#: Default confidence per spaCy label.  NER confidence is generally lower
#: than regex because models can hallucinate entity boundaries.
DEFAULT_CONFIDENCE: dict[str, float] = {
    "PERSON": 0.85,
    "ORG": 0.75,
    "GPE": 0.75,
    "LOC": 0.70,
    "NORP": 0.65,
    "FAC": 0.65,
    "DATE": 0.60,
    "MONEY": 0.80,
}


class NERDetector:
    """Named Entity Recognition detector backed by spaCy.

    Args:
        model_name: Name of the spaCy model to load.  Defaults to
            ``"en_core_web_sm"``.
        entity_types: Set of guardrails entity types to detect.
            ``None`` means all mapped types.
        confidence_overrides: Per-spaCy-label confidence overrides.

    If spaCy or the specified model is not installed the detector will
    log a warning and return empty results from :meth:`detect`.
    """

    def __init__(
        self,
        *,
        model_name: str = "en_core_web_sm",
        entity_types: set[str] | None = None,
        confidence_overrides: dict[str, float] | None = None,
    ) -> None:
        self._model_name = model_name
        self._entity_types = entity_types
        self._confidence_map = {**DEFAULT_CONFIDENCE, **(confidence_overrides or {})}
        self._nlp: Any | None = None
        self._available: bool = False
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Attempt to load the spaCy model.  Sets *_available* on success."""
        try:
            import spacy  # type: ignore[import-untyped]

            self._nlp = spacy.load(self._model_name)
            self._available = True
            logger.info("NERDetector loaded spaCy model '%s'", self._model_name)
        except ImportError:
            logger.warning(
                "spaCy is not installed — NERDetector will return no results. "
                "Install with: pip install spacy"
            )
        except OSError:
            logger.warning(
                "spaCy model '%s' is not installed — NERDetector will return "
                "no results. Install with: python -m spacy download %s",
                self._model_name,
                self._model_name,
            )

    @property
    def is_available(self) -> bool:
        """Return True if the spaCy model was loaded successfully."""
        return self._available

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    async def detect(self, text: str) -> list[DetectionResult]:
        """Detect named entities in *text* asynchronously.

        Runs the spaCy pipeline in the default executor to avoid blocking
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
        """Detect named entities in *text* synchronously.

        Args:
            text: The text to scan.

        Returns:
            List of :class:`DetectionResult`.
        """
        if not self._available or self._nlp is None:
            return []

        doc = self._nlp(text)
        results: list[DetectionResult] = []

        for ent in doc.ents:
            guardrails_type = SPACY_ENTITY_MAP.get(ent.label_)
            if guardrails_type is None:
                continue
            if self._entity_types is not None and guardrails_type not in self._entity_types:
                continue

            confidence = self._confidence_map.get(ent.label_, 0.60)
            severity = DEFAULT_SEVERITY.get(guardrails_type, Severity.MEDIUM)

            results.append(
                DetectionResult(
                    entity_type=guardrails_type,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=confidence,
                    detector="spacy_ner",
                    severity=severity,
                    metadata={
                        "spacy_label": ent.label_,
                        "model": self._model_name,
                    },
                )
            )

        return results
