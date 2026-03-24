"""Unit tests for classification modules: toxicity, language, sentiment, topic."""

from __future__ import annotations

import pytest

from guardrails.classification.toxicity import ToxicityCategory, ToxicityClassifier, ToxicityResult
from guardrails.classification.language import LanguageDetector, LanguageResult
from guardrails.classification.sentiment import Sentiment, SentimentAnalyzer, SentimentResult
from guardrails.classification.topic import TopicClassifier, TopicDefinition, TopicResult


# ---------------------------------------------------------------------------
# ToxicityClassifier
# ---------------------------------------------------------------------------


class TestToxicityClassifier:
    """Tests for ToxicityClassifier."""

    @pytest.mark.asyncio
    async def test_toxic_text_detected(self, toxicity_classifier: ToxicityClassifier) -> None:
        """Clearly toxic text should be flagged with is_toxic=True."""
        result = await toxicity_classifier.classify("kill yourself you worthless loser")
        assert result.is_toxic is True
        assert result.overall_score > 0.0
        assert len(result.flagged_categories) > 0

    @pytest.mark.asyncio
    async def test_clean_text_passes(self, toxicity_classifier: ToxicityClassifier) -> None:
        """Benign text should not be flagged as toxic."""
        result = await toxicity_classifier.classify(
            "The weather is lovely today. I went for a walk in the park."
        )
        assert result.is_toxic is False
        assert result.overall_score < 0.5
        assert result.flagged_categories == []

    @pytest.mark.asyncio
    async def test_threshold_adjustment(self) -> None:
        """Lowering the threshold should make detection more sensitive."""
        strict = ToxicityClassifier(threshold=0.1)
        lenient = ToxicityClassifier(threshold=0.9)

        text = "you are pathetic and ugly"

        strict_result = await strict.classify(text)
        lenient_result = await lenient.classify(text)

        assert strict.threshold == 0.1
        assert lenient.threshold == 0.9
        # The strict classifier should flag more aggressively.
        assert strict_result.is_toxic is True
        assert lenient_result.is_toxic is False

    @pytest.mark.asyncio
    async def test_invalid_threshold_raises(self) -> None:
        """A threshold outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="threshold must be in"):
            ToxicityClassifier(threshold=1.5)

    @pytest.mark.asyncio
    async def test_category_scoring(self, toxicity_classifier: ToxicityClassifier) -> None:
        """Each toxicity category should get its own score entry."""
        result = await toxicity_classifier.classify("this is hate speech from a racist bigot")
        assert ToxicityCategory.HATE_SPEECH.value in result.category_scores
        # Hate speech category should score meaningfully.
        assert result.category_scores[ToxicityCategory.HATE_SPEECH.value] > 0.0
        # All six categories should be present in the scores dict.
        assert len(result.category_scores) == len(ToxicityCategory)

    @pytest.mark.asyncio
    async def test_context_aware_scoring_long_text(self, toxicity_classifier: ToxicityClassifier) -> None:
        """A toxic keyword buried in a long text should score lower than the same keyword in a short message."""
        short_text = "you are pathetic"
        long_text = (
            "The quarterly financial results exceeded expectations across all divisions. "
            "Revenue grew by twelve percent year-over-year, driven by strong demand in the "
            "enterprise segment. Customer satisfaction scores reached an all-time high, "
            "and the new product line has been well received by analysts. One reviewer "
            "called the previous iteration pathetic, but the team addressed all feedback. "
            "Looking ahead, the roadmap includes several exciting initiatives."
        )

        short_result = await toxicity_classifier.classify(short_text)
        long_result = await toxicity_classifier.classify(long_text)

        # The short text should score higher for harassment since keyword
        # density is much larger.
        short_harassment = short_result.category_scores.get(ToxicityCategory.HARASSMENT.value, 0.0)
        long_harassment = long_result.category_scores.get(ToxicityCategory.HARASSMENT.value, 0.0)
        assert short_harassment > long_harassment

    @pytest.mark.asyncio
    async def test_empty_text_returns_clean_result(self, toxicity_classifier: ToxicityClassifier) -> None:
        """Empty or whitespace-only text should return a non-toxic result."""
        result = await toxicity_classifier.classify("")
        assert result.is_toxic is False
        assert result.overall_score == 0.0

        result_ws = await toxicity_classifier.classify("   ")
        assert result_ws.is_toxic is False

    @pytest.mark.asyncio
    async def test_custom_keywords_merged(self) -> None:
        """Custom keywords provided at init should be merged into scoring."""
        classifier = ToxicityClassifier(
            threshold=0.3,
            custom_keywords={
                "profanity": [("zorgblat", 0.9)],
            },
        )
        result = await classifier.classify("you said zorgblat to me")
        # The custom word should contribute to the profanity category.
        assert result.category_scores[ToxicityCategory.PROFANITY.value] > 0.0

    @pytest.mark.asyncio
    async def test_result_dataclass_fields(self, toxicity_classifier: ToxicityClassifier) -> None:
        """ToxicityResult should expose all documented fields."""
        result = await toxicity_classifier.classify("hello world")
        assert isinstance(result, ToxicityResult)
        assert isinstance(result.is_toxic, bool)
        assert isinstance(result.overall_score, float)
        assert isinstance(result.category_scores, dict)
        assert isinstance(result.flagged_categories, list)
        assert isinstance(result.threshold, float)
        assert result.threshold == toxicity_classifier.threshold


# ---------------------------------------------------------------------------
# LanguageDetector
# ---------------------------------------------------------------------------


class TestLanguageDetector:
    """Tests for LanguageDetector."""

    @pytest.mark.asyncio
    async def test_english_detected(self, language_detector: LanguageDetector) -> None:
        """English text should be detected as 'en'."""
        result = await language_detector.detect(
            "The quick brown fox jumps over the lazy dog near the riverbank."
        )
        assert result.language == "en"
        assert result.confidence > 0.5
        assert isinstance(result, LanguageResult)

    @pytest.mark.asyncio
    async def test_other_language_detected(self) -> None:
        """Non-English text should be detected as the correct language."""
        detector = LanguageDetector(seed=0)
        result = await detector.detect(
            "Bonjour, comment allez-vous aujourd'hui? Je suis tres content de vous voir."
        )
        assert result.language == "fr"

    @pytest.mark.asyncio
    async def test_allowed_languages_enforcement_allowed(self) -> None:
        """Text in an allowed language should have is_allowed=True."""
        detector = LanguageDetector(allowed_languages=["en", "es"], seed=0)
        result = await detector.detect(
            "This is a perfectly normal English sentence with enough words to detect."
        )
        assert result.is_allowed is True

    @pytest.mark.asyncio
    async def test_allowed_languages_enforcement_blocked(self) -> None:
        """Text in a non-allowed language should have is_allowed=False."""
        detector = LanguageDetector(allowed_languages=["en"], seed=0)
        result = await detector.detect(
            "Bonjour, comment allez-vous aujourd'hui? Je suis tres content de vous voir."
        )
        assert result.is_allowed is False

    @pytest.mark.asyncio
    async def test_empty_text_raises(self, language_detector: LanguageDetector) -> None:
        """Empty text should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot detect language of empty text"):
            await language_detector.detect("")

    @pytest.mark.asyncio
    async def test_all_languages_populated(self, language_detector: LanguageDetector) -> None:
        """The all_languages list should contain at least one candidate."""
        result = await language_detector.detect(
            "This is a sufficiently long English sentence for language detection."
        )
        assert len(result.all_languages) >= 1
        assert result.all_languages[0].language == result.language

    @pytest.mark.asyncio
    async def test_allowed_languages_property(self) -> None:
        """The allowed_languages property should reflect the init value."""
        detector = LanguageDetector(allowed_languages=["en", "de"])
        assert detector.allowed_languages == {"en", "de"}

        detector_none = LanguageDetector()
        assert detector_none.allowed_languages is None


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer."""

    @pytest.mark.asyncio
    async def test_positive_text(self, sentiment_analyzer: SentimentAnalyzer) -> None:
        """Clearly positive text should be classified as positive."""
        result = await sentiment_analyzer.analyze("This is absolutely wonderful and amazing!")
        assert result.sentiment == Sentiment.POSITIVE
        assert result.score > 0.0
        assert isinstance(result, SentimentResult)

    @pytest.mark.asyncio
    async def test_negative_text(self, sentiment_analyzer: SentimentAnalyzer) -> None:
        """Clearly negative text should be classified as negative."""
        result = await sentiment_analyzer.analyze("This is terrible and awful, I hate it.")
        assert result.sentiment == Sentiment.NEGATIVE
        assert result.score < 0.0

    @pytest.mark.asyncio
    async def test_neutral_text(self, sentiment_analyzer: SentimentAnalyzer) -> None:
        """Factual, non-emotional text should be classified as neutral."""
        result = await sentiment_analyzer.analyze("The meeting is scheduled for three o'clock.")
        assert result.sentiment == Sentiment.NEUTRAL
        assert abs(result.score) < sentiment_analyzer.neutral_threshold

    @pytest.mark.asyncio
    async def test_empty_text_returns_neutral(self, sentiment_analyzer: SentimentAnalyzer) -> None:
        """Empty text should return neutral with zero score and confidence."""
        result = await sentiment_analyzer.analyze("")
        assert result.sentiment == Sentiment.NEUTRAL
        assert result.score == 0.0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_negation_flips_sentiment(self, sentiment_analyzer: SentimentAnalyzer) -> None:
        """Negation words should flip the sentiment of the following word."""
        positive_result = await sentiment_analyzer.analyze("This is good")
        negated_result = await sentiment_analyzer.analyze("This is not good")

        assert positive_result.score > 0.0
        assert negated_result.score < 0.0

    @pytest.mark.asyncio
    async def test_custom_threshold(self) -> None:
        """A custom neutral_threshold should change the neutral boundary."""
        wide_neutral = SentimentAnalyzer(neutral_threshold=0.9)
        result = await wide_neutral.analyze("This is good")
        # With such a wide neutral zone, mildly positive text should be neutral.
        assert result.sentiment == Sentiment.NEUTRAL

    @pytest.mark.asyncio
    async def test_invalid_threshold_raises(self) -> None:
        """A threshold outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="neutral_threshold must be in"):
            SentimentAnalyzer(neutral_threshold=-0.5)


# ---------------------------------------------------------------------------
# TopicClassifier
# ---------------------------------------------------------------------------


class TestTopicClassifier:
    """Tests for TopicClassifier."""

    @pytest.mark.asyncio
    async def test_basic_classification_works(self, topic_classifier: TopicClassifier) -> None:
        """Text about technology should detect the technology topic."""
        result = await topic_classifier.classify(
            "We are deploying machine learning models to our cloud infrastructure "
            "using a new API and database architecture."
        )
        assert result.is_on_topic is True
        assert len(result.detected_topics) > 0
        topic_names = [t.topic for t in result.detected_topics]
        assert "technology" in topic_names

    @pytest.mark.asyncio
    async def test_empty_text_returns_no_topics(self, topic_classifier: TopicClassifier) -> None:
        """Empty text should not match any topic."""
        result = await topic_classifier.classify("")
        assert result.is_on_topic is False
        assert result.detected_topics == []
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_custom_topics(self) -> None:
        """A classifier with custom topics should score against those topics."""
        custom = TopicClassifier(
            topics=[
                TopicDefinition(
                    name="cooking",
                    keywords=["recipe", "ingredient", "oven", "bake", "cook", "kitchen"],
                    phrases=["cooking time", "preheat oven"],
                ),
            ],
            confidence_threshold=0.1,
        )
        result = await custom.classify(
            "Preheat oven to 350 degrees. Mix the ingredients and bake for 30 minutes."
        )
        assert result.is_on_topic is True
        topic_names = [t.topic for t in result.detected_topics]
        assert "cooking" in topic_names

    @pytest.mark.asyncio
    async def test_allowed_topics_filter(self, topic_classifier: TopicClassifier) -> None:
        """When allowed_topics is set, is_on_topic should only be True for those topics."""
        text = (
            "We are deploying machine learning models to our cloud infrastructure "
            "using a new API and database architecture."
        )
        # With allowed_topics=["health"], even a tech text should be off-topic.
        result = await topic_classifier.classify(text, allowed_topics=["health"])
        assert result.is_on_topic is False

    @pytest.mark.asyncio
    async def test_matched_terms_populated(self, topic_classifier: TopicClassifier) -> None:
        """Detected topics should list which terms matched."""
        result = await topic_classifier.classify(
            "The software engineering team deployed a new server and database."
        )
        tech_topics = [t for t in result.detected_topics if t.topic == "technology"]
        if tech_topics:
            assert len(tech_topics[0].matched_terms) > 0

    @pytest.mark.asyncio
    async def test_confidence_threshold_respected(self) -> None:
        """Topics scoring below the confidence threshold should not appear in detected_topics."""
        strict = TopicClassifier(confidence_threshold=0.99)
        result = await strict.classify("server")
        # With a near-impossible threshold, nothing should pass.
        assert result.detected_topics == []
        assert result.is_on_topic is False
