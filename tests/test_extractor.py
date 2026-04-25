"""Unit tests for ClaimExtractor — Stage 1."""

import pytest
from unittest.mock import patch, MagicMock
from app.extractor import ClaimExtractor
from app.schemas import Claim


@pytest.fixture
def extractor():
    return ClaimExtractor()


def test_split_sentences(extractor):
    text = "Einstein was born in 1879. He won the Nobel Prize in 1921."
    sentences = extractor.split_sentences(text)
    assert len(sentences) == 2
    assert "Einstein" in sentences[0]
    assert "Nobel" in sentences[1]


def test_split_single_sentence(extractor):
    text = "The Eiffel Tower is in Paris."
    sentences = extractor.split_sentences(text)
    assert len(sentences) == 1


def test_extract_claims_returns_claim_objects(extractor):
    mock_response = [
        {"text": "Einstein was born in 1879.", "claim_type": "factual"},
        {"text": "Einstein was brilliant.", "claim_type": "opinion"},
    ]
    with patch.object(extractor, "_call_groq", return_value=mock_response):
        claims = extractor.extract_claims("Einstein was born in 1879 and was brilliant.")
    assert all(isinstance(c, Claim) for c in claims)
    assert len(claims) == 2


def test_get_factual_claims_filters_opinions(extractor):
    mock_response = [
        {"text": "Einstein was born in 1879.", "claim_type": "factual"},
        {"text": "Einstein was the greatest scientist.", "claim_type": "opinion"},
    ]
    with patch.object(extractor, "_call_groq", return_value=mock_response):
        claims = extractor.get_factual_claims("Einstein was born in 1879.")
    assert len(claims) == 1
    assert claims[0].claim_type == "factual"


def test_extract_claims_handles_groq_failure(extractor):
    """If Groq fails after retries, fallback to treating sentence as one claim."""
    with patch.object(extractor, "_call_groq", side_effect=Exception("API down")):
        claims = extractor.extract_claims("The sky is blue.")
    assert len(claims) == 1
    assert claims[0].text == "The sky is blue."


def test_claim_sentence_index(extractor):
    """Claims must carry the correct sentence index."""
    mock_response = [{"text": "Paris is in France.", "claim_type": "factual"}]
    with patch.object(extractor, "_call_groq", return_value=mock_response):
        claims = extractor.extract_claims("Paris is in France. Rome is in Italy.")
    assert claims[0].sentence_index == 0