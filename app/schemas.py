from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class Claim(BaseModel):
    text: str
    sentence_index: int
    claim_type: Literal["factual", "opinion", "non-verifiable"]


class EvidenceSnippet(BaseModel):
    content: str
    url: str
    similarity_score: float = 0.0


class VerificationResult(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "unverifiable"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    verification_method: Literal["similarity", "llm_judge", "cache"] = "similarity"


class SentenceAnnotation(BaseModel):
    sentence: str
    sentence_index: int
    verdict: Literal["supported", "contradicted", "unverifiable", "non-factual"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    claims: list[VerificationResult] = Field(default_factory=list)
    rewritten: str | None = None

    @field_validator("verdict", mode="before")
    @classmethod
    def normalise_verdict(cls, v: str) -> str:
        return v.lower().strip()


class AnnotatedResponse(BaseModel):
    original_text: str
    annotations: list[SentenceAnnotation]
    hallucination_count: int
    verified_count: int
    unverifiable_count: int
    overall_trust_score: float = Field(..., ge=0.0, le=1.0)


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=10)
    rewrite_hallucinations: bool = False


class CheckResponse(AnnotatedResponse):
    pass