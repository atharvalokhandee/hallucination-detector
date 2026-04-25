"""
Stage 1 — Claim Extractor
Splits LLM response into sentences (spaCy) then extracts
atomic factual claims from each sentence (Groq).
"""

import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

import spacy
from groq import Groq

from app.schemas import Claim
from app.config import settings

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

EXTRACTION_PROMPT = """You are a precise fact-checking assistant.

Given a sentence and its context, extract all atomic factual claims.

IMPORTANT RULES:
- Classify based on the STRUCTURE of the claim, NOT whether you think it is true or false
- Any sentence that asserts a fact about the world — even if it sounds wrong — is "factual"
- Only classify as "opinion" if it uses opinion words like "best", "worst", "beautiful", "should"
- Only classify as "non-verifiable" if it is a greeting, question, or purely subjective feeling
- "The Great Wall is visible from space" → factual (it makes a verifiable claim)
- "Shakespeare wrote The Dark Knight Rises" → factual (it makes a verifiable claim)
- "Einstein was a great man" → opinion

Each claim must be:
- A single standalone verifiable assertion
- Written with NO pronouns — replace he/she/it/they with the actual subject from context
- Classified strictly by structure, not by whether it seems true

Respond ONLY with a JSON array. No explanation, no markdown.

Context: "{context}"
Sentence to analyse: "{sentence}"
"""


class ClaimExtractor:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)

    def split_sentences(self, text: str) -> list[str]:
        """Use spaCy to split text into sentences."""
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _call_groq(self, sentence: str, context: str) -> list[dict]:
        """Call Groq to extract claims from one sentence. Retries up to 3x."""
        response = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "user", "content": EXTRACTION_PROMPT.format(
                    sentence=sentence, context=context
                )}
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw)

    def extract_claims(self, text: str) -> list[Claim]:
        """
        Full Stage 1 pipeline:
        1. Split text into sentences
        2. Extract atomic claims with full text as context for pronoun resolution
        3. Return flat list of Claim objects
        """
        sentences = self.split_sentences(text)
        all_claims: list[Claim] = []

        for idx, sentence in enumerate(sentences):
            logger.info(f"Extracting claims from sentence {idx}: {sentence[:60]}...")
            try:
                raw_claims = self._call_groq(sentence, context=text)
                for item in raw_claims:
                    claim = Claim(
                        text=item["text"],
                        sentence_index=idx,
                        claim_type=item.get("claim_type", "factual"),
                    )
                    all_claims.append(claim)
            except Exception as e:
                logger.warning(f"Failed to extract claims from sentence {idx}: {e}")
                all_claims.append(
                    Claim(text=sentence, sentence_index=idx, claim_type="factual")
                )

        return all_claims

    def get_factual_claims(self, text: str) -> list[Claim]:
        """Convenience method — returns only verifiable factual claims."""
        return [c for c in self.extract_claims(text) if c.claim_type == "factual"]