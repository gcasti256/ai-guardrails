"""Detector registry for managing PII detection backends.

The registry provides a central place to register, enable, disable,
and retrieve PII detectors.  Built-in detectors (regex, spaCy NER,
Presidio) are auto-registered on construction.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from guardrails.types import DetectionResult

logger = logging.getLogger(__name__)


@runtime_checkable
class Detector(Protocol):
    """Protocol that all PII detectors must satisfy."""

    async def detect(self, text: str) -> list[DetectionResult]:
        """Detect PII entities in the given text."""
        ...

    def detect_sync(self, text: str) -> list[DetectionResult]:
        """Detect PII entities in the given text synchronously."""
        ...


class _DetectorEntry:
    """Internal wrapper around a registered detector."""

    __slots__ = ("name", "detector", "enabled", "priority")

    def __init__(
        self,
        name: str,
        detector: Any,
        *,
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        self.name = name
        self.detector = detector
        self.enabled = enabled
        self.priority = priority


class DetectorRegistry:
    """Central registry for PII detection backends.

    On construction the registry auto-registers the built-in regex
    detector.  The spaCy NER and Presidio detectors are registered
    lazily (disabled by default) because they have heavy optional
    dependencies.

    Args:
        auto_register_builtins: Whether to register built-in detectors
            automatically.  Set to ``False`` for testing.

    Example::

        registry = DetectorRegistry()
        registry.enable("spacy_ner")
        detectors = registry.get_active_detectors()
    """

    def __init__(self, *, auto_register_builtins: bool = True) -> None:
        self._detectors: dict[str, _DetectorEntry] = {}
        if auto_register_builtins:
            self._register_builtins()

    # ------------------------------------------------------------------
    # Built-in registration
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        """Register built-in detectors."""
        # Regex detector — always available and enabled by default.
        from guardrails.pii.detector import PIIDetector

        self.register(
            "regex",
            PIIDetector(),
            enabled=True,
            priority=10,
        )

        # spaCy NER — optional, disabled by default.
        try:
            from guardrails.pii.ner_detector import NERDetector

            ner = NERDetector()
            self.register(
                "spacy_ner",
                ner,
                enabled=False,
                priority=20,
            )
            if ner.is_available:
                logger.info("spaCy NER detector registered (disabled by default)")
            else:
                logger.debug("spaCy NER detector registered but model not available")
        except Exception:
            logger.debug("spaCy NER detector not registered", exc_info=True)

        # Presidio — optional, disabled by default.
        try:
            from guardrails.pii.presidio_detector import PresidioDetector

            presidio = PresidioDetector()
            self.register(
                "presidio",
                presidio,
                enabled=False,
                priority=30,
            )
            if presidio.is_available:
                logger.info("Presidio detector registered (disabled by default)")
            else:
                logger.debug("Presidio detector registered but not available")
        except Exception:
            logger.debug("Presidio detector not registered", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        detector: Any,
        *,
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        """Register a detector under *name*.

        If a detector with the same name already exists it is replaced.

        Args:
            name: Unique name for this detector.
            detector: An object with ``detect`` and/or ``detect_sync``
                methods matching the :class:`Detector` protocol.
            enabled: Whether the detector starts enabled.
            priority: Execution order hint — lower values run first.
        """
        if name in self._detectors:
            logger.info("Replacing existing detector '%s'", name)

        self._detectors[name] = _DetectorEntry(
            name=name,
            detector=detector,
            enabled=enabled,
            priority=priority,
        )
        logger.debug("Registered detector '%s' (enabled=%s, priority=%d)", name, enabled, priority)

    def unregister(self, name: str) -> None:
        """Remove a detector by name.

        Args:
            name: The detector name to remove.

        Raises:
            KeyError: If no detector with that name is registered.
        """
        if name not in self._detectors:
            raise KeyError(f"No detector registered with name '{name}'")
        del self._detectors[name]
        logger.debug("Unregistered detector '%s'", name)

    def enable(self, name: str) -> None:
        """Enable a registered detector.

        Args:
            name: The detector name to enable.

        Raises:
            KeyError: If no detector with that name is registered.
        """
        entry = self._get_entry(name)
        entry.enabled = True
        logger.debug("Enabled detector '%s'", name)

    def disable(self, name: str) -> None:
        """Disable a registered detector.

        Args:
            name: The detector name to disable.

        Raises:
            KeyError: If no detector with that name is registered.
        """
        entry = self._get_entry(name)
        entry.enabled = False
        logger.debug("Disabled detector '%s'", name)

    def is_enabled(self, name: str) -> bool:
        """Check whether a detector is currently enabled.

        Args:
            name: The detector name to check.

        Returns:
            True if the detector exists and is enabled.
        """
        entry = self._detectors.get(name)
        return entry is not None and entry.enabled

    def get_detector(self, name: str) -> Any:
        """Return the raw detector instance by name.

        Args:
            name: The detector name.

        Raises:
            KeyError: If no detector with that name is registered.
        """
        return self._get_entry(name).detector

    def get_active_detectors(self) -> list[Any]:
        """Return all enabled detectors sorted by priority (ascending).

        Returns:
            List of detector instances that are currently enabled.
        """
        active = [e for e in self._detectors.values() if e.enabled]
        active.sort(key=lambda e: e.priority)
        return [e.detector for e in active]

    def list_detectors(self) -> list[dict[str, Any]]:
        """Return metadata about all registered detectors.

        Returns:
            List of dicts with keys ``name``, ``enabled``, ``priority``,
            and ``type`` (the class name of the detector).
        """
        result: list[dict[str, Any]] = []
        for entry in self._detectors.values():
            result.append({
                "name": entry.name,
                "enabled": entry.enabled,
                "priority": entry.priority,
                "type": type(entry.detector).__name__,
            })
        result.sort(key=lambda d: d["priority"])
        return result

    # ------------------------------------------------------------------
    # Convenience: run all active detectors
    # ------------------------------------------------------------------

    async def detect_all(self, text: str) -> list[DetectionResult]:
        """Run all active detectors on *text* and merge results.

        Results are sorted by start offset.  Deduplication across
        detectors is left to the caller (or a higher-level engine).

        Args:
            text: The text to scan.

        Returns:
            Combined list of :class:`DetectionResult` from all active
            detectors.
        """
        import asyncio

        active = self.get_active_detectors()
        if not active:
            return []

        tasks = []
        for detector in active:
            if hasattr(detector, "detect"):
                tasks.append(detector.detect(text))
            elif hasattr(detector, "detect_sync"):
                loop = asyncio.get_running_loop()
                tasks.append(
                    loop.run_in_executor(None, detector.detect_sync, text)
                )

        all_results: list[DetectionResult] = []
        for coro_result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(coro_result, BaseException):
                logger.error("Detector failed: %s", coro_result, exc_info=coro_result)
                continue
            all_results.extend(coro_result)

        all_results.sort(key=lambda d: d.start)
        return all_results

    def detect_all_sync(self, text: str) -> list[DetectionResult]:
        """Run all active detectors on *text* synchronously and merge results.

        Args:
            text: The text to scan.

        Returns:
            Combined list of :class:`DetectionResult` from all active
            detectors.
        """
        active = self.get_active_detectors()
        all_results: list[DetectionResult] = []

        for detector in active:
            try:
                if hasattr(detector, "detect_sync"):
                    all_results.extend(detector.detect_sync(text))
                else:
                    logger.warning(
                        "Detector %s has no detect_sync method, skipping in sync mode",
                        type(detector).__name__,
                    )
            except Exception:
                logger.error(
                    "Detector %s failed",
                    type(detector).__name__,
                    exc_info=True,
                )

        all_results.sort(key=lambda d: d.start)
        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_entry(self, name: str) -> _DetectorEntry:
        """Look up a detector entry by name or raise KeyError."""
        try:
            return self._detectors[name]
        except KeyError:
            raise KeyError(f"No detector registered with name '{name}'") from None
