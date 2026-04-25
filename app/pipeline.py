"""
Stage 5 — Pipeline Orchestrator
Connects all stages:
  Extractor → Verifier → LLM Judge → Assembler
Returns a complete AnnotatedResponse.
"""

import logging
from app.schemas import (
    Claim, VerificationResult, SentenceAnnotation,
    AnnotatedResponse, CheckRequest
)
from app.extractor import ClaimExtractor
from app.verifier import Verifier
from app.judge import LLMJudge
from app.cache import ClaimCache
from app.config import settings

logger = logging.getLogger(__name__)


class HallucinationDetectionPipeline:
    def __init__(self):
        self.extractor = ClaimExtractor()
        self.verifier = Verifier()
        self.judge = LLMJudge()
        self.cache = ClaimCache()

    def _verify_single(self, claim: Claim) -> VerificationResult:
        cached = self.cache.get(claim.text)
        if cached:
            return cached

        result = self.verifier.verify_claim(claim)

        if not result.evidence:
            self.cache.set(claim.text, result)
            return result

        top_score = result.evidence[0].similarity_score
        if top_score < settings.similarity_threshold_low:
            self.cache.set(claim.text, result)
            return result

        logger.info(f"Sending to LLM judge: {claim.text[:60]}")
        result = self.judge.judge(claim.text, result.evidence)
        self.cache.set(claim.text, result)
        return result

    def _aggregate_sentence_verdict(
        self, results: list[VerificationResult]
    ) -> tuple[str, float, list[str]]:
        if not results:
            return "non-factual", 1.0, []

        verdicts = [r.verdict for r in results]
        confidences = [r.confidence for r in results]
        sources = list(dict.fromkeys(url for r in results for url in r.sources))

        if "contradicted" in verdicts:
            verdict = "contradicted"
        elif "unverifiable" in verdicts:
            verdict = "unverifiable"
        else:
            verdict = "supported"

        avg_confidence = sum(confidences) / len(confidences)
        return verdict, round(avg_confidence, 4), sources

    def _build_annotations(
        self,
        sentences: list[str],
        all_claims: list[Claim],
        all_results: list[VerificationResult],
        rewrite: bool,
    ) -> list[SentenceAnnotation]:
        results_by_sentence: dict[int, list[VerificationResult]] = {
            i: [] for i in range(len(sentences))
        }

        for claim, result in zip(all_claims, all_results):
            results_by_sentence[claim.sentence_index].append(result)

        annotations = []
        for idx, sentence in enumerate(sentences):
            sentence_results = results_by_sentence[idx]

            if not sentence_results:
                annotations.append(SentenceAnnotation(
                    sentence=sentence,
                    sentence_index=idx,
                    verdict="non-factual",
                    confidence=1.0,
                    sources=[],
                    claims=[],
                ))
                continue

            verdict, confidence, sources = self._aggregate_sentence_verdict(sentence_results)

            rewritten = None
            if rewrite and verdict == "contradicted":
                logger.info(f"Rewriting: {sentence[:60]}")
                rewritten = self.judge.rewrite_sentence(sentence)

            annotations.append(SentenceAnnotation(
                sentence=sentence,
                sentence_index=idx,
                verdict=verdict,
                confidence=confidence,
                sources=sources,
                claims=sentence_results,
                rewritten=rewritten,
            ))

        return annotations

    def run(self, request: CheckRequest) -> AnnotatedResponse:
        logger.info(f"Pipeline started — {len(request.text)} chars")

        sentences = self.extractor.split_sentences(request.text)
        all_claims = self.extractor.extract_claims(request.text)
        factual_claims = [c for c in all_claims if c.claim_type == "factual"]

        logger.info(f"Found {len(factual_claims)} factual claims in {len(sentences)} sentences")

        all_results = [self._verify_single(claim) for claim in factual_claims]

        annotations = self._build_annotations(
            sentences, factual_claims, all_results, request.rewrite_hallucinations
        )

        hallucination_count = sum(1 for a in annotations if a.verdict == "contradicted")
        verified_count = sum(1 for a in annotations if a.verdict == "supported")
        unverifiable_count = sum(1 for a in annotations if a.verdict == "unverifiable")
        verifiable_total = hallucination_count + verified_count + unverifiable_count
        trust_score = round(verified_count / verifiable_total, 4) if verifiable_total > 0 else 1.0

        return AnnotatedResponse(
            original_text=request.text,
            annotations=annotations,
            hallucination_count=hallucination_count,
            verified_count=verified_count,
            unverifiable_count=unverifiable_count,
            overall_trust_score=trust_score,
        )