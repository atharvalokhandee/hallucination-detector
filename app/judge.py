"""
Stage 4 — LLM Judge
Only called for ambiguous claims (similarity score 0.40–0.75).
Uses Groq to make a final supported/contradicted/unverifiable decision.
"""

import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import Groq

from app.schemas import EvidenceSnippet, VerificationResult
from app.config import settings

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are a strict fact-checking judge. Your job is to decide if a claim is supported by the provided evidence.

CLAIM: {claim}

EVIDENCE:
{evidence}

Respond ONLY with a JSON object in this exact format — no explanation, no markdown:
{{"verdict": "supported" | "contradicted" | "unverifiable", "confidence": 0.0-1.0, "reason": "one sentence"}}

Rules:
- "supported": evidence clearly confirms the claim
- "contradicted": evidence clearly contradicts the claim  
- "unverifiable": evidence is irrelevant or insufficient to decide
- confidence must be a float between 0.0 and 1.0
"""


class LLMJudge:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _call_groq(self, claim: str, evidence_text: str) -> dict:
        """Call Groq with claim + evidence, return parsed JSON verdict."""
        prompt = JUDGE_PROMPT.format(claim=claim, evidence=evidence_text)
        response = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    def judge(self, claim: str, evidence: list[EvidenceSnippet]) -> VerificationResult:
        """
        Run LLM judge on a claim + its evidence snippets.
        Returns a VerificationResult with verdict from the LLM.
        """
        if not evidence:
            return VerificationResult(
                claim=claim,
                verdict="unverifiable",
                confidence=0.0,
                sources=[],
                evidence=[],
                verification_method="llm_judge",
            )

        # Build evidence text — top 3 snippets only to stay within token limits
        evidence_text = "\n\n".join(
            f"[{i+1}] {e.content}" for i, e in enumerate(evidence[:3])
        )
        sources = [e.url for e in evidence if e.url]

        try:
            result = self._call_groq(claim, evidence_text)
            verdict = result.get("verdict", "unverifiable").lower().strip()
            confidence = float(result.get("confidence", 0.5))

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            # Validate verdict value
            if verdict not in ("supported", "contradicted", "unverifiable"):
                verdict = "unverifiable"

            return VerificationResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                sources=sources,
                evidence=evidence,
                verification_method="llm_judge",
            )

        except Exception as e:
            logger.warning(f"LLM judge failed for claim '{claim[:50]}': {e}")
            return VerificationResult(
                claim=claim,
                verdict="unverifiable",
                confidence=0.0,
                sources=sources,
                evidence=evidence,
                verification_method="llm_judge",
            )

    def rewrite_sentence(self, sentence: str) -> str:
        """
        Rewrite a hallucinated sentence removing the false claim.
        Used in rewrite mode (Stage 5).
        """
        prompt = f"""The following sentence contains a factual error. 
Rewrite it to remove the false claim while keeping any true parts.
If the entire sentence is false, respond with an empty string "".
Respond ONLY with the rewritten sentence — no explanation.

Sentence: "{sentence}"
"""
        try:
            response = self.client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip().strip('"')
        except Exception as e:
            logger.warning(f"Rewrite failed: {e}")
            return sentence