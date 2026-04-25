"""
Stage 2 & 3 — Verifier
Searches the web for evidence (Tavily) then scores each claim
against that evidence using sentence-transformers + cosine similarity.
"""

import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from app.schemas import Claim, EvidenceSnippet, VerificationResult
from app.config import settings

logger = logging.getLogger(__name__)

# Load embedding model once at module level
_embedder = SentenceTransformer("all-MiniLM-L6-v2")


class Verifier:
    def __init__(self):
        self.tavily = TavilyClient(api_key=settings.tavily_api_key)
        self.embedder = _embedder
        self.threshold_high = settings.similarity_threshold_high
        self.threshold_low = settings.similarity_threshold_low

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _search(self, query: str) -> list[dict]:
        """Call Tavily and return raw results."""
        response = self.tavily.search(
            query=query,
            max_results=settings.tavily_max_results,
            search_depth="basic",
        )
        return response.get("results", [])

    def _build_evidence(self, claim_text: str, raw_results: list[dict]) -> list[EvidenceSnippet]:
        """Embed claim + each snippet, compute cosine similarity, return scored list."""
        if not raw_results:
            return []

        snippets = [r.get("content", "") for r in raw_results]
        urls = [r.get("url", "") for r in raw_results]

        # Embed everything in one batch — fast
        all_texts = [claim_text] + snippets
        embeddings = self.embedder.encode(all_texts, convert_to_numpy=True)

        claim_vec = embeddings[0:1]       # shape (1, dim)
        snippet_vecs = embeddings[1:]     # shape (n, dim)

        scores = cosine_similarity(claim_vec, snippet_vecs)[0]  # shape (n,)

        evidence = []
        for snippet, url, score in zip(snippets, urls, scores):
            evidence.append(EvidenceSnippet(
                content=snippet,
                url=url,
                similarity_score=float(score),
            ))

        # Sort best evidence first
        return sorted(evidence, key=lambda e: e.similarity_score, reverse=True)

    def _verdict_from_score(self, top_score: float) -> tuple[str, float]:
        """
        Convert best similarity score to a verdict + confidence.
        Returns (verdict, confidence) — or ("ambiguous", score) for LLM judge.
        """
        if top_score >= self.threshold_high:
            return "supported", top_score
        elif top_score <= self.threshold_low:
            return "contradicted", 1.0 - top_score
        else:
            return "ambiguous", top_score  # needs LLM judge in Stage 4

    def verify_claim(self, claim: Claim) -> VerificationResult:
        """
        Full Stage 2+3 for one claim:
        1. Web search
        2. Semantic scoring
        3. Threshold verdict (or flag as ambiguous for LLM judge)
        """
        try:
            raw_results = self._search(claim.text)
        except Exception as e:
            logger.warning(f"Tavily search failed for claim '{claim.text[:50]}': {e}")
            return VerificationResult(
                claim=claim.text,
                verdict="unverifiable",
                confidence=0.0,
                sources=[],
                evidence=[],
                verification_method="similarity",
            )

        evidence = self._build_evidence(claim.text, raw_results)

        if not evidence:
            return VerificationResult(
                claim=claim.text,
                verdict="unverifiable",
                confidence=0.0,
                sources=[],
                evidence=[],
                verification_method="similarity",
            )

        top_score = evidence[0].similarity_score
        verdict, confidence = self._verdict_from_score(top_score)
        sources = [e.url for e in evidence if e.url]

        return VerificationResult(
            claim=claim.text,
            verdict=verdict if verdict != "ambiguous" else "unverifiable",
            confidence=confidence,
            sources=sources,
            evidence=evidence,
            verification_method="similarity",
        )

    def verify_claims(self, claims: list[Claim]) -> list[VerificationResult]:
        """Verify a list of claims sequentially."""
        results = []
        for claim in claims:
            logger.info(f"Verifying: {claim.text[:60]}...")
            result = self.verify_claim(claim)
            results.append(result)
        return results