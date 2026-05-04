# 🔍 Real-Time Hallucination Detection Layer

> A production-grade AI safety middleware that intercepts LLM-generated text, verifies every factual claim against live web sources, and returns colour-coded trust annotations in real time.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-22%20passing-brightgreen)
![F1 Score](https://img.shields.io/badge/F1%20Score-100%25-brightgreen)

---

## 🧠 The Problem

Large language models confidently state false facts. For example:

> *"Einstein was born in Germany in 1879. **He invented the telephone.** He won the Nobel Prize in Physics in 1921."*

The second sentence is completely wrong — yet an LLM states it with full confidence. Companies building AI products for healthcare, legal, or finance cannot afford this. They need a verification layer that catches hallucinations before users see them.

---

## ✅ What This Project Does

This system acts as a **middleware layer** between your LLM and your users:

1. Takes any LLM-generated text as input
2. Splits it into individual sentences using spaCy
3. Extracts atomic factual claims from each sentence using Groq (Llama 3.3 70B)
4. Searches the web for real-world evidence using Tavily Search API
5. Scores evidence relevance using sentence-transformers (cosine similarity)
6. Runs an LLM judge on ambiguous claims for a final verdict
7. Returns every sentence colour-coded — GREEN (verified), RED (hallucinated), AMBER (uncertain)
8. Optionally rewrites hallucinated sentences automatically

---

## 🏗️ Architecture

```
POST /check  {"text": "LLM response here..."}
                        │
                        ▼
        ┌───────────────────────────┐
        │  Stage 1: Claim Extractor │
        │  spaCy sentence split     │
        │  Groq pronoun resolution  │
        │  → List[Claim]            │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  Stage 2: Web Search      │
        │  Tavily Search API        │
        │  → top 5 evidence snippets│
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  Stage 3: Similarity Score│
        │  sentence-transformers    │
        │  cosine similarity        │
        │  score < 0.40 → skip judge│
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  Stage 4: LLM Judge       │
        │  Groq Llama 3.3 70B       │
        │  → supported /            │
        │     contradicted /        │
        │     unverifiable          │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  Stage 5: Assembler       │
        │  Map verdicts → sentences │
        │  Compute trust score      │
        │  → AnnotatedResponse      │
        └─────────────┬─────────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
          FastAPI          Streamlit
          /check           Colour UI
          endpoint         Trust meter
                           Rewrite mode

        Redis cache wraps Stages 2–4
        (1-hour TTL per unique claim)
```

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| LLM | Groq API — `llama-3.3-70b-versatile` | Claim extraction + LLM judge |
| Sentence splitting | spaCy `en_core_web_sm` | Accurate sentence boundary detection |
| Semantic embeddings | `sentence-transformers` all-MiniLM-L6-v2 | Convert text to vectors |
| Evidence retrieval | Tavily Search API | Live web search with clean text output |
| Vector similarity | scikit-learn cosine_similarity | Score claim vs evidence relevance |
| API layer | FastAPI + Uvicorn | Async REST endpoint |
| Caching | Redis | Claim → verdict cache with TTL |
| UI | Streamlit | Colour-coded interactive frontend |
| Schema validation | Pydantic v2 | Request/response validation |
| Config | pydantic-settings + python-dotenv | Environment variable management |
| Retries | Tenacity | Exponential backoff on API calls |
| Testing | pytest + pytest-mock | 22 unit tests, zero real API calls |

---

## 📁 Project Structure

```
hallucination-detector/
├── app/
│   ├── schemas.py        # All Pydantic data models
│   ├── config.py         # Settings loaded from .env
│   ├── extractor.py      # Stage 1: spaCy + Groq claim extraction
│   ├── verifier.py       # Stage 2+3: Tavily search + similarity scoring
│   ├── judge.py          # Stage 4: Groq LLM judge + rewrite
│   ├── pipeline.py       # Stage 5: Orchestrates all stages
│   └── cache.py          # Redis caching layer
├── api/
│   └── main.py           # FastAPI /check endpoint
├── ui/
│   └── streamlit_app.py  # Colour-coded Streamlit UI
├── tests/
│   ├── test_extractor.py # 6 unit tests
│   ├── test_verifier.py  # 8 unit tests
│   └── test_api.py       # 8 unit tests
├── eval/
│   └── test_cases.csv    # 15 labelled test cases
├── scripts/
│   └── run_eval.py       # Precision / Recall / F1 evaluation
├── .env.example          # Copy to .env and add your keys
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/atharvalokhandee/hallucination-detector.git
cd hallucination-detector
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### 4. Configure API keys
```bash
cp .env.example .env
```

Open `.env` and fill in:
```
GROQ_API_KEY=your_groq_key_here       # https://console.groq.com/keys
TAVILY_API_KEY=your_tavily_key_here   # https://app.tavily.com/
```

### 5. Start Redis
```bash
# Option A: Homebrew (macOS)
brew install redis && brew services start redis

# Option B: Docker
docker run -d -p 6379:6379 redis:alpine

# Verify
redis-cli ping   # → PONG
```

### 6. Run the API
```bash
uvicorn api.main:app --reload --timeout-keep-alive 120
# → http://localhost:8000/docs
```

### 7. Run the Streamlit UI
```bash
streamlit run ui/streamlit_app.py
# → http://localhost:8501
```

---

## 📡 API Reference

### `GET /health`
```json
{"status": "ok", "pipeline_loaded": true}
```

### `POST /check`

**Request:**
```json
{
  "text": "Einstein was born in Germany in 1879. He invented the telephone. He won the Nobel Prize in Physics in 1921.",
  "rewrite_hallucinations": true
}
```

**Response:**
```json
{
  "original_text": "Einstein was born in Germany in 1879. He invented the telephone. He won the Nobel Prize in Physics in 1921.",
  "annotations": [
    {
      "sentence": "Einstein was born in Germany in 1879.",
      "verdict": "supported",
      "confidence": 1.0,
      "sources": ["https://www.history.com/...", "https://www.britannica.com/..."],
      "claims": [...],
      "rewritten": null
    },
    {
      "sentence": "He invented the telephone.",
      "verdict": "contradicted",
      "confidence": 1.0,
      "sources": ["https://www.britannica.com/biography/Alexander-Graham-Bell"],
      "claims": [...],
      "rewritten": "Alexander Graham Bell is credited with inventing the telephone."
    },
    {
      "sentence": "He won the Nobel Prize in Physics in 1921.",
      "verdict": "supported",
      "confidence": 1.0,
      "sources": ["https://www.nobelprize.org/prizes/physics/1921/einstein/facts/"],
      "claims": [...],
      "rewritten": null
    }
  ],
  "hallucination_count": 1,
  "verified_count": 2,
  "unverifiable_count": 0,
  "overall_trust_score": 0.6667
}
```

**curl example:**
```bash
curl -s -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text": "Einstein invented the telephone.", "rewrite_hallucinations": true}' \
  --max-time 120 | python3 -m json.tool
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
# 22 passed
```

All tests use mocks — no real API calls required. Tests cover:
- Sentence splitting edge cases
- Claim extraction and fallback handling
- Similarity scoring thresholds
- Graceful failure on API errors
- FastAPI endpoint validation

---

## 📊 Evaluation

```bash
python3 scripts/run_eval.py
```

Results on 15 hand-labelled test cases:

```
📊 EVALUATION RESULTS
==================================================
Total test cases : 15
Scored           : 15
Skipped          : 0
Errors           : 0

True Positives   : 6   (correctly caught hallucinations)
False Positives  : 0   (true facts flagged wrongly)
True Negatives   : 9   (correctly verified true facts)
False Negatives  : 0   (missed hallucinations)

Accuracy         : 100.0%
Precision        : 100.0%
Recall           : 100.0%
F1 Score         : 100.0%
```

---

## 🔑 Key Design Decisions

**Why LLM judge instead of similarity alone?**
Cosine similarity measures topical relevance, not factual accuracy. Evidence about "telephone invention" scores high similarity whether it confirms or contradicts a claim. The LLM judge actually reads the evidence and reasons about it.

**Why Redis caching?**
In production, many users submit the same facts. Caching claim→verdict pairs with a 1-hour TTL means repeat verifications return in under 50ms instead of 3-5 seconds, and API costs scale with unique claims rather than total requests.

**Why `run_in_executor` in FastAPI?**
The pipeline makes synchronous blocking calls (Groq, Tavily). Running them directly in an async endpoint would freeze the event loop. `run_in_executor` offloads work to a thread pool, keeping the server responsive.

**Why pronoun resolution in claim extraction?**
"He invented the telephone" submitted to Tavily returns useless results. Resolving "He" → "Albert Einstein" using the full paragraph as context produces a specific, searchable claim that Tavily can find relevant evidence for.

---

## 🌐 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Groq API key |
| `TAVILY_API_KEY` | required | Tavily Search API key |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `SIMILARITY_THRESHOLD_HIGH` | `0.75` | Above this = relevant evidence |
| `SIMILARITY_THRESHOLD_LOW` | `0.40` | Below this = irrelevant, skip judge |
| `CACHE_TTL_SECONDS` | `3600` | Cache expiry (1 hour) |
| `TAVILY_MAX_RESULTS` | `5` | Evidence snippets per claim |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 👤 Author

**Atharva Lokhandee**


---

*Built as a GenAI portfolio project demonstrating production-grade LLM safety tooling.*
