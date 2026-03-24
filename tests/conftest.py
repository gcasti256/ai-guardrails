"""Shared test fixtures for AI Guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

from guardrails import GuardrailsEngine
from guardrails.pii.detector import PIIDetector
from guardrails.pii.redactor import PIIRedactor
from guardrails.injection.detector import InjectionDetector
from guardrails.classification.toxicity import ToxicityClassifier
from guardrails.classification.topic import TopicClassifier
from guardrails.classification.language import LanguageDetector
from guardrails.classification.sentiment import SentimentAnalyzer
from guardrails.policy.loader import PolicyLoader


POLICIES_DIR = Path(__file__).parent.parent / "policies"


@pytest.fixture
def pii_detector() -> PIIDetector:
    return PIIDetector()


@pytest.fixture
def redactor() -> PIIRedactor:
    return PIIRedactor()


@pytest.fixture
def injection_detector() -> InjectionDetector:
    return InjectionDetector()


@pytest.fixture
def toxicity_classifier() -> ToxicityClassifier:
    return ToxicityClassifier()


@pytest.fixture
def topic_classifier() -> TopicClassifier:
    return TopicClassifier()


@pytest.fixture
def language_detector() -> LanguageDetector:
    return LanguageDetector()


@pytest.fixture
def sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


@pytest.fixture
def policy_loader() -> PolicyLoader:
    return PolicyLoader()


@pytest.fixture
def engine() -> GuardrailsEngine:
    e = GuardrailsEngine()
    e.load_policies_dir(POLICIES_DIR)
    return e


@pytest.fixture
def policies_dir() -> Path:
    return POLICIES_DIR
