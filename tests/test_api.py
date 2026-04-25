"""Tests for FastAPI /check endpoint."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from app.schemas import (
    AnnotatedResponse, SentenceAnnotation, VerificationResult
)


def _make_mock_result():
    return AnnotatedResponse(
        original_text="Einstein was born in Germany. He invented the telephone.",
        annotations=[
            SentenceAnnotation(
                sentence="Einstein was born in Germany.",
                sentence_index=0,
                verdict="supported",
                confidence=0.95,
                sources=["https://example.com"],
                claims=[
                    VerificationResult(
                        claim="Einstein was born in Germany.",
                        verdict="supported",
                        confidence=0.95,
                        sources=["https://example.com"],
                    )
                ],
            ),
            SentenceAnnotation(
                sentence="He invented the telephone.",
                sentence_index=1,
                verdict="contradicted",
                confidence=0.91,
                sources=["https://example.com/bell"],
                claims=[
                    VerificationResult(
                        claim="Albert Einstein invented the telephone.",
                        verdict="contradicted",
                        confidence=0.91,
                        sources=["https://example.com/bell"],
                    )
                ],
            ),
        ],
        hallucination_count=1,
        verified_count=1,
        unverifiable_count=0,
        overall_trust_score=0.5,
    )


@pytest.fixture
def client():
    with patch("api.main.pipeline") as mock_pipeline:
        mock_pipeline.run.return_value = _make_mock_result()
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_check_returns_200(client):
    response = client.post("/check", json={"text": "Einstein was born in Germany. He invented the telephone."})
    assert response.status_code == 200


def test_check_response_structure(client):
    response = client.post("/check", json={"text": "Einstein was born in Germany. He invented the telephone."})
    data = response.json()
    assert "annotations" in data
    assert "overall_trust_score" in data
    assert "hallucination_count" in data
    assert len(data["annotations"]) == 2


def test_check_verdicts(client):
    response = client.post("/check", json={"text": "Einstein was born in Germany. He invented the telephone."})
    data = response.json()
    verdicts = [a["verdict"] for a in data["annotations"]]
    assert "supported" in verdicts
    assert "contradicted" in verdicts


def test_check_trust_score(client):
    response = client.post("/check", json={"text": "Einstein was born in Germany. He invented the telephone."})
    data = response.json()
    assert 0.0 <= data["overall_trust_score"] <= 1.0


def test_check_rewrite_mode(client):
    response = client.post("/check", json={
        "text": "Einstein was born in Germany. He invented the telephone.",
        "rewrite_hallucinations": True
    })
    assert response.status_code == 200


def test_check_short_text_rejected(client):
    response = client.post("/check", json={"text": "Hi"})
    assert response.status_code == 422


def test_check_missing_text(client):
    response = client.post("/check", json={})
    assert response.status_code == 422