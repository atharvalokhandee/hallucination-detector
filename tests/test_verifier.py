"""Unit tests for Verifier — Stage 2 & 3."""

import pytest
from unittest.mock import patch, MagicMock
from app.verifier import Verifier
from app.schemas import Claim, VerificationResult


@pytest.fixture
def verifier():
    return Verifier()


@pytest.fixture
def factual_claim():
    return Claim(text="Einstein was born in Germany.", sentence_index=0, claim_type="factual")


MOCK_TAVILY_RESULTS = [
    {"content": "Albert Einstein was born in Ulm, Germany in 1879.", "url": "https://example.com/einstein"},
    {"content": "Einstein grew up in Germany before moving to Switzerland.", "url": "https://example.com/bio"},
]


def test_build_evidence_returns_scored_snippets(verifier, factual_claim):
    evidence = verifier._build_evidence(factual_claim.text, MOCK_TAVILY_RESULTS)
    assert len(evidence) == 2
    assert all(0.0 <= e.similarity_score <= 1.0 for e in evidence)
    # Best evidence should be first
    assert evidence[0].similarity_score >= evidence[1].similarity_score


def test_verdict_high_score(verifier):
    verdict, confidence = verifier._verdict_from_score(0.9)
    assert verdict == "supported"
    assert confidence == 0.9


def test_verdict_low_score(verifier):
    verdict, confidence = verifier._verdict_from_score(0.2)
    assert verdict == "contradicted"
    assert confidence > 0.5  # 1.0 - 0.2


def test_verdict_ambiguous_score(verifier):
    verdict, _ = verifier._verdict_from_score(0.55)
    assert verdict == "ambiguous"


def test_verify_claim_supported(verifier, factual_claim):
    with patch.object(verifier, "_search", return_value=MOCK_TAVILY_RESULTS):
        result = verifier.verify_claim(factual_claim)
    assert isinstance(result, VerificationResult)
    assert result.verdict in ("supported", "unverifiable")
    assert len(result.sources) > 0


def test_verify_claim_search_failure(verifier, factual_claim):
    """If Tavily is down, return unverifiable gracefully."""
    with patch.object(verifier, "_search", side_effect=Exception("Network error")):
        result = verifier.verify_claim(factual_claim)
    assert result.verdict == "unverifiable"
    assert result.confidence == 0.0


def test_verify_claim_empty_results(verifier, factual_claim):
    with patch.object(verifier, "_search", return_value=[]):
        result = verifier.verify_claim(factual_claim)
    assert result.verdict == "unverifiable"


def test_verify_claims_batch(verifier):
    claims = [
        Claim(text="Paris is in France.", sentence_index=0, claim_type="factual"),
        Claim(text="Rome is in Italy.", sentence_index=1, claim_type="factual"),
    ]
    mock_results = [{"content": "Paris is the capital of France.", "url": "https://example.com"}]
    with patch.object(verifier, "_search", return_value=mock_results):
        results = verifier.verify_claims(claims)
    assert len(results) == 2
    assert all(isinstance(r, VerificationResult) for r in results)