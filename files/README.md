# 🔍 Real-Time Hallucination Detection Layer

> A production-grade middleware that intercepts LLM responses, verifies every factual claim against live web sources, and returns colour-coded trust annotations — in real time.

---

## 🧠 What It Does

Large language models confidently state false facts. This system adds a **verification layer** between your LLM and your users:

1. **Claim Extraction** — splits any LLM response into atomic factual claims
2. **Live Web Search** — queries Tavily API for real-world evidence
3. **Semantic Scoring** — embeds claim + evidence and computes cosine similarity
4. **LLM Judge** — uses Groq (Llama 3.3 70B) as a second-pass verifier for ambiguous claims
5. **Annotated Output** — returns every sentence labelled GREEN / AMBER / RED with source URLs

---

## 🏗️ Architecture

```
User Input (LLM Response Text)
            │
            ▼
┌───────────────────────┐
│   Stage 1: Extractor  │  spaCy sentence split → Groq claim extraction
└──────────┬────────────┘
           │  List[Claim]
           ▼
┌───────────────────────┐
│   Stage 2: Verifier   │  Tavily web search → top 3–5 snippets per claim
└──────────┬────────────┘
           │  Claim + Evidence
           ▼
┌───────────────────────┐
│   Stage 3: Similarity │  sentence-transformers + cosine similarity
└──────────┬────────────┘
           │  score > 0.75 → supported
           │  score < 0.40 → contradicted
           │  else → ambiguous → Stage 4
           ▼
┌───────────────────────┐
│   Stage 4: LLM Judge  │  Groq Llama judges ambiguous claims (JSON verdict)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Stage 5: Assembler   │  Maps verdicts → sentences → AnnotatedResponse
└──────────┬────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
FastAPI        Streamlit
/check         Colour-coded UI
endpoint       + Trust Score
               + Rewrite Mode

           Redis cache wraps Stage 2–4 (1-hour TTL per claim)
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| LLM (claim extraction + judge) | Groq API — `llama-3.3-70b-versatile` |
| Sentence segmentation | spaCy `en_core_web_sm` |
| Semantic embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Live evidence retrieval | Tavily Search API |
| Vector similarity | FAISS + cosine similarity |
| API layer | FastAPI + Uvicorn |
| Caching | Redis (claim → verdict, 1hr TTL) |
| UI | Streamlit |
| Schema validation | Pydantic v2 |
| Testing | pytest + pytest-asyncio |
| Config | python-dotenv |

---

## 🚀 Setup

### 1. Clone and enter the project
```bash
git clone https://github.com/YOUR_USERNAME/hallucination-detector.git
cd hallucination-detector
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configure API keys
```bash
cp .env.example .env
# Open .env and fill in:
#   GROQ_API_KEY      → https://console.groq.com/keys
#   TAVILY_API_KEY    → https://app.tavily.com/
```

### 5. Start Redis (required for caching)
```bash
# Option A: Docker (recommended)
docker run -d -p 6379:6379 redis:alpine

# Option B: Local install (Ubuntu/macOS)
# sudo apt install redis-server && redis-server
```

### 6. Run the API
```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

### 7. Run the Streamlit UI
```bash
streamlit run ui/streamlit_app.py
# → http://localhost:8501
```

---

## 📡 API Usage

### `POST /check`

**Request:**
```json
{
  "text": "Albert Einstein was born in Germany in 1879. He invented the telephone. He won the Nobel Prize in Physics in 1921."
}
```

**Response:**
```json
{
  "original_text": "Albert Einstein was born in Germany in 1879...",
  "annotations": [
    {
      "sentence": "Albert Einstein was born in Germany in 1879.",
      "sentence_index": 0,
      "verdict": "supported",
      "confidence": 0.94,
      "sources": ["https://en.wikipedia.org/wiki/Albert_Einstein"],
      "claims": [...]
    },
    {
      "sentence": "He invented the telephone.",
      "sentence_index": 1,
      "verdict": "contradicted",
      "confidence": 0.91,
      "sources": ["https://en.wikipedia.org/wiki/Alexander_Graham_Bell"],
      "claims": [...]
    },
    {
      "sentence": "He won the Nobel Prize in Physics in 1921.",
      "sentence_index": 2,
      "verdict": "supported",
      "confidence": 0.97,
      "sources": ["https://www.nobelprize.org/prizes/physics/1921/einstein/facts/"],
      "claims": [...]
    }
  ],
  "hallucination_count": 1,
  "verified_count": 2,
  "unverifiable_count": 0,
  "overall_trust_score": 0.67
}
```

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```

---

## 📊 Evaluation
```bash
python scripts/run_eval.py
# Outputs precision, recall, F1 to console + eval/results.csv
```

---

## 📁 Project Structure

```
hallucination-detector/
├── app/
│   ├── extractor.py      # ClaimExtractor: spaCy + Groq
│   ├── verifier.py       # Tavily search + semantic similarity
│   ├── judge.py          # Groq LLM judge for ambiguous claims
│   ├── pipeline.py       # Orchestrates all 5 stages
│   └── cache.py          # Redis caching layer
├── api/
│   └── main.py           # FastAPI /check endpoint
├── ui/
│   └── streamlit_app.py  # Colour-coded Streamlit UI
├── tests/                # pytest unit tests
├── eval/                 # test_cases.csv + evaluation results
├── scripts/
│   └── run_eval.py       # Evaluation script (precision/recall/F1)
├── .env.example          # Copy to .env and add your keys
├── requirements.txt
└── README.md
```

---

## 🎯 Key Design Decisions

- **Hybrid verification**: Semantic similarity is fast and cheap; LLM judge only runs on ambiguous cases (0.40–0.75 range), keeping costs low
- **Redis caching**: Identical claims verified once per hour — critical for production use where the same facts repeat across users
- **Atomic claims**: Sentences are decomposed into individual claims before verification — one false fact in a sentence doesn't taint the whole sentence
- **Tenacity retries**: All external API calls retry with exponential backoff to handle transient failures

---

## 🤝 Contributing

PRs welcome. Please add a test for any new pipeline stage.

---

## 📄 License

MIT
