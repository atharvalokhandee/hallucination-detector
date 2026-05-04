# 🔍 Real-Time Hallucination Detection Layer

A middleware system that verifies LLM-generated text against live web sources and returns colour-coded trust annotations — sentence by sentence, in real time.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Tests](https://img.shields.io/badge/Tests-22%20passing-brightgreen)
![F1 Score](https://img.shields.io/badge/F1%20Score-100%25-brightgreen)

---

## The Problem

LLMs hallucinate — they state wrong facts with complete confidence:

> *"Einstein was born in Germany. **He invented the telephone.** He won the Nobel Prize in 1921."*

The second sentence is completely false. Companies building AI products for healthcare, legal, or finance can't ship this to users. This project is a safety layer that catches it before they do.

---

## How It Works

Every request flows through 5 stages:

```
Input text
    │
    ▼
Stage 1 — Claim Extractor     spaCy splits sentences, Groq extracts atomic claims
    │                         "He invented the phone" → "Einstein invented the phone"
    ▼
Stage 2 — Web Search          Tavily fetches top 5 real-world evidence snippets
    │
    ▼
Stage 3 — Similarity Scoring  sentence-transformers checks if evidence is relevant
    │                         (filters irrelevant results before hitting the LLM)
    ▼
Stage 4 — LLM Judge           Groq reads claim + evidence → supported / contradicted / unverifiable
    │
    ▼
Stage 5 — Assembler           Maps verdicts back to sentences, computes trust score

Redis cache wraps Stages 2–4 (1-hour TTL per claim)
```

---

## Tech Stack

| What | Tool |
|------|------|
| LLM (extraction + judge) | Groq — `llama-3.3-70b-versatile` |
| Sentence splitting | spaCy `en_core_web_sm` |
| Semantic embeddings | `sentence-transformers` all-MiniLM-L6-v2 |
| Live web evidence | Tavily Search API |
| Caching | Redis |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Validation | Pydantic v2 |
| Tests | pytest + pytest-mock (22 tests) |

---

## Setup

```bash
# 1. Clone and create venv
git clone https://github.com/atharvalokhandee/hallucination-detector.git
cd hallucination-detector
python3 -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

# 3. Add API keys
cp .env.example .env
# Fill in GROQ_API_KEY and TAVILY_API_KEY

# 4. Start Redis
brew install redis && brew services start redis

# 5. Run API
uvicorn api.main:app --reload --timeout-keep-alive 120

# 6. Run UI (new terminal)
streamlit run ui/streamlit_app.py
```

---

## API Example

```bash
curl -s -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text": "Einstein was born in Germany. He invented the telephone.", "rewrite_hallucinations": true}' \
  --max-time 120
```

```json
{
  "annotations": [
    {
      "sentence": "Einstein was born in Germany.",
      "verdict": "supported",
      "confidence": 1.0,
      "sources": ["https://www.history.com/..."]
    },
    {
      "sentence": "He invented the telephone.",
      "verdict": "contradicted",
      "confidence": 1.0,
      "rewritten": "Alexander Graham Bell is credited with inventing the telephone.",
      "sources": ["https://www.britannica.com/..."]
    }
  ],
  "hallucination_count": 1,
  "verified_count": 1,
  "overall_trust_score": 0.5
}
```

---

## Evaluation

```bash
python3 scripts/run_eval.py
```

Tested on 15 hand-labelled claims (known hallucinations + true facts):

| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |

---

## A Few Design Decisions Worth Noting

**Similarity does not equal truth.** Early testing showed cosine similarity was marking hallucinations as "supported" — because evidence about telephone invention scores high similarity whether it confirms or contradicts the claim. The fix was to use similarity only for filtering irrelevant evidence, and let the LLM judge make all factual decisions.

**Pronoun resolution matters.** Sending "He invented the telephone" to a search API returns garbage results. The extractor resolves all pronouns using the full paragraph as context before any verification happens.

**Caching is not optional at scale.** Redis stores every verified claim for 1 hour. Repeat claims return in under 50ms instead of 3-5 seconds, and API costs grow with unique claims — not total requests.

---

## Project Structure

```
hallucination-detector/
├── app/
│   ├── extractor.py      # Stage 1
│   ├── verifier.py       # Stage 2 + 3
│   ├── judge.py          # Stage 4
│   ├── pipeline.py       # Orchestrator
│   ├── cache.py          # Redis layer
│   ├── schemas.py        # Pydantic models
│   └── config.py         # Settings from .env
├── api/main.py           # FastAPI /check endpoint
├── ui/streamlit_app.py   # Streamlit UI
├── tests/                # 22 unit tests
├── eval/test_cases.csv   # Labelled evaluation set
└── scripts/run_eval.py   # Precision/Recall/F1 script
```

---

## License

MIT

---

*Built by [Atharva Lokhandee](https://github.com/atharvalokhandee) as a GenAI portfolio project.*
