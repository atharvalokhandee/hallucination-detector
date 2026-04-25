"""
Shared Pydantic schemas for the Hallucination Detection Pipeline.
All pipeline stages import from here — single source of truth.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal


# ── Stage 1 output ────────────────────────────────────────────────────────────

class Claim(BaseModel):
    """A single atomic factual claim extracted from one sentence."""

    text: str = Field(..., description="The factual claim as a complete sentence.")
    sentence_index: int = Field(..., ge=0, description="Which sentence this came from (0-based).")
    claim_type: Literal["factual", "opinion", "non-verifiable"] = Field(
        ..., description="Only 'factual' claims are sent for verification."
    )


# ── Stage 2–4 output ──────────────────────────────────────────────────────────

class EvidenceSnippet(BaseModel):
    """A single piece of web evidence retrieved for a claim."""

    content: str = Field(..., description="The raw text snippet from the web source.")
    url: str = Field(..., description="Source URL.")
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between claim embedding and this snippet's embedding.",
    )


class VerificationResult(BaseModel):
    """The full verification outcome for a single claim."""

    claim: str
    verdict: Literal["supported", "contradicted", "unverifiable"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list, description="URLs supporting the verdict.")
    evidence: list[EvidenceSnippet] = Field(
        default_factory=list,
        description="Raw evidence snippets (useful for UI drill-down).",
    )
    verification_method: Literal["similarity", "llm_judge", "cache"] = Field(
        default="similarity",
        description="Which stage produced this verdict.",
    )


# ── Stage 5 output ────────────────────────────────────────────────────────────

class SentenceAnnotation(BaseModel):
    """Aggregated verdict for one sentence (may contain multiple claims)."""

    sentence: str
    sentence_index: int
    verdict: Literal["supported", "contradicted", "unverifiable", "non-factual"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    claims: list[VerificationResult] = Field(default_factory=list)
    rewritten: str | None = Field(
        default=None,
        description="Hallucination-free rewrite of the sentence, if rewrite mode is enabled.",
    )

    @field_validator("verdict", mode="before")
    @classmethod
    def normalise_verdict(cls, v: str) -> str:
        return v.lower().strip()


class AnnotatedResponse(BaseModel):
    """Final output of the full pipeline — ready to serve via API or render in UI."""

    original_text: str
    annotations: list[SentenceAnnotation]
    hallucination_count: int = Field(..., ge=0)
    verified_count: int = Field(..., ge=0)
    unverifiable_count: int = Field(..., ge=0)
    overall_trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Ratio of supported sentences to total verifiable sentences. "
            "1.0 = fully verified, 0.0 = all hallucinated."
        ),
    )


# ── API request / response wrappers ──────────────────────────────────────────

class CheckRequest(BaseModel):
    """Request body for POST /check."""

    text: str = Field(..., min_length=10, description="The LLM-generated text to verify.")
    rewrite_hallucinations: bool = Field(
        default=False,
        description="If True, contradicted sentences are rewritten by Groq.",
    )


class CheckResponse(AnnotatedResponse):
    """Response body for POST /check — same as AnnotatedResponse for now."""
    pass
